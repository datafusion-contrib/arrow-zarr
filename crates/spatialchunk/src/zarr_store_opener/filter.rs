// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use arrow_array::{BooleanArray, RecordBatch};
use arrow_schema::{ArrowError, SchemaRef};
use datafusion::common::cast::as_boolean_array;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion, TreeNodeVisitor};
use datafusion::common::Result as DfResult;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::utils::reassign_expr_columns;
use datafusion::physical_expr::{conjunction, split_conjunction, PhysicalExpr};

use super::util::{ColumnNames, ProjectByNames};

struct ZarrFilterExpression {
    physical_expr: Arc<dyn PhysicalExpr>,
    filter_schema: SchemaRef,
}

impl ZarrFilterExpression {
    fn new(physical_expr: Arc<dyn PhysicalExpr>, table_schema: SchemaRef) -> DfResult<Self> {
        // the physical expressions are created from the full table schema, but
        // the record batches will come in with the filter schema, so the column
        // indices in the expression need to be reassigned.
        let required_columns = pushdown_columns(&physical_expr, table_schema.clone())?;
        let filter_schema = table_schema.project(&required_columns)?;
        let physical_expr = reassign_expr_columns(physical_expr, &filter_schema)?;

        Ok(Self {
            physical_expr,
            filter_schema: Arc::new(filter_schema),
        })
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<BooleanArray, ArrowError> {
        // the record batch can have its columns in a different order than the
        // filter schema, so it's projected to make sure the column indices match
        // between the expression and the record batch.
        let batch = batch.project_by_names(self.filter_schema.fields().iter_column_names())?;

        match self
            .physical_expr
            .evaluate(&batch)
            .and_then(|v| v.into_array(batch.num_rows()))
        {
            Ok(array) => {
                let bool_arr = as_boolean_array(&array)?.clone();
                Ok(bool_arr)
            }
            Err(e) => Err(ArrowError::ComputeError(format!(
                "Error evaluating filter predicate: {e:?}"
            ))),
        }
    }
}

// A struct that implements TreeNodeRewriter to traverse a PhysicalExpr tree structure
// to determine which columns are required to evaluate it.
struct PushdownChecker {
    // Indices into the table schema of the columns required to evaluate the expression
    required_columns: BTreeSet<usize>,
    table_schema: SchemaRef,
}

// Note that the zarr case is simpler than the other cases that support projected
// columns or partition columns. There is not much we need to check here, columns
// are just columns, we just need to check which columns are in which predicate.
impl PushdownChecker {
    fn new(table_schema: SchemaRef) -> Self {
        Self {
            required_columns: BTreeSet::default(),
            table_schema,
        }
    }
}

impl TreeNodeVisitor<'_> for PushdownChecker {
    type Node = Arc<dyn PhysicalExpr>;

    fn f_down(&mut self, node: &Self::Node) -> DfResult<TreeNodeRecursion> {
        if let Some(column) = node.downcast_ref::<Column>() {
            let idx = self.table_schema.index_of(column.name())?;
            self.required_columns.insert(idx);
        }

        Ok(TreeNodeRecursion::Continue)
    }
}

fn pushdown_columns(expr: &Arc<dyn PhysicalExpr>, table_schema: SchemaRef) -> DfResult<Vec<usize>> {
    let mut checker = PushdownChecker::new(table_schema);
    expr.visit(&mut checker)?;
    Ok(checker.required_columns.into_iter().collect())
}

// A filter used to determine whether or not a chunk needs to be read. The
// filter expression is split into one predicate per set of columns, each
// combining every condition on exactly those columns, so that each predicate
// can be evaluated on its own, on a record batch with only its columns.
pub(crate) struct ZarrChunkFilter {
    predicates: HashMap<BTreeSet<String>, ZarrFilterExpression>,
}

impl ZarrChunkFilter {
    pub(crate) fn new(
        expr: &Arc<dyn PhysicalExpr>,
        table_schema: SchemaRef,
    ) -> Result<Self, ArrowError> {
        let mut grouped: HashMap<BTreeSet<String>, Vec<Arc<dyn PhysicalExpr>>> = HashMap::new();
        for pred_expr in split_conjunction(expr) {
            let key = pushdown_columns(pred_expr, table_schema.clone())?
                .iter()
                .map(|i| table_schema.field(*i).name().clone())
                .collect();
            grouped.entry(key).or_default().push(pred_expr.clone());
        }

        let predicates = grouped
            .into_iter()
            .map(|(key, exprs)| {
                let filter_expr =
                    ZarrFilterExpression::new(conjunction(exprs), table_schema.clone())?;
                Ok((key, filter_expr))
            })
            .collect::<DfResult<_>>()?;

        Ok(Self { predicates })
    }

    /// The sets of columns each predicate needs.
    pub(crate) fn keys(&self) -> impl Iterator<Item = &BTreeSet<String>> {
        self.predicates.keys()
    }

    /// Evaluates the predicate whose columns are exactly the columns of the
    /// record batch, returning true if at least one row passes.
    pub(crate) fn evaluate(&self, rec_batch: &RecordBatch) -> Result<bool, ArrowError> {
        let key: BTreeSet<String> = rec_batch.schema().fields().iter_column_names().collect();
        let predicate = self.predicates.get(&key).ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!("No filter predicate for columns {key:?}"))
        })?;

        // true_count ignores nulls (and falses), matching SQL's WHERE
        // semantics, so a chunk with no passing row is correctly skipped.
        Ok(predicate.evaluate(rec_batch)?.true_count() > 0)
    }
}

#[cfg(test)]
mod filter_tests {
    use std::sync::Arc;

    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::expressions::col;
    use datafusion::physical_plan::expressions::binary;

    use super::*;

    // evaluates every predicate in the chunk filter, each on the batch
    // projected to its own columns.
    fn evaluate_all(chunk_filter: &ZarrChunkFilter, batch: &RecordBatch) -> bool {
        chunk_filter.keys().all(|key| {
            chunk_filter
                .evaluate(&batch.project_by_names(key).unwrap())
                .unwrap()
        })
    }

    #[test]
    fn test_single_filter() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
            Field::new("c", DataType::Int32, false),
        ]);
        let a = Int32Array::from(vec![1, 2, 3, 4, 5, 6]);
        let b = Int32Array::from(vec![3, 3, 3, 3, 3, 3]);
        let c = Int32Array::from(vec![4, 4, 4, 4, 4, 4]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(a), Arc::new(b), Arc::new(c)],
        )
        .unwrap();

        // expression: "a > b"
        let expr = binary(
            col("a", &schema).unwrap(),
            Operator::Gt,
            col("b", &schema).unwrap(),
            &schema,
        )
        .unwrap();

        let filter = ZarrFilterExpression::new(expr, Arc::new(schema.clone())).unwrap();
        let mask = filter.evaluate(&batch).unwrap();

        assert_eq!(mask, vec![false, false, false, true, true, true].into());

        // this test in particular is important because it applies the filter
        // to a record batch where the data is ordered differently than in the
        // physical expression for filter.
        // expression: "c < a"
        let expr = binary(
            col("c", &schema).unwrap(),
            Operator::Lt,
            col("a", &schema).unwrap(),
            &schema,
        )
        .unwrap();

        let filter = ZarrFilterExpression::new(expr, Arc::new(schema)).unwrap();
        let mask = filter.evaluate(&batch).unwrap();

        assert_eq!(mask, vec![false, false, false, false, true, true].into());
    }

    #[test]
    fn test_filter_with_nulls() {
        // nullable columns, with rows where a is null, b is null, both are
        // null, and three fully-populated rows.
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
        ]);
        let a = Int32Array::from(vec![None, Some(5), None, Some(10), Some(20), Some(1)]);
        let b = Int32Array::from(vec![Some(3), None, None, Some(5), Some(1), Some(100)]);
        let batch =
            RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a), Arc::new(b)]).unwrap();

        // two comparisons on the same columns, so that they get combined back
        // into a single predicate, where the null handling lives.
        // expression: "a > b AND a >= b"
        let gt = binary(
            col("a", &schema).unwrap(),
            Operator::Gt,
            col("b", &schema).unwrap(),
            &schema,
        )
        .unwrap();
        let ge = binary(
            col("a", &schema).unwrap(),
            Operator::GtEq,
            col("b", &schema).unwrap(),
            &schema,
        )
        .unwrap();
        let expr = binary(gt, Operator::And, ge, &schema).unwrap();

        let chunk_filter = ZarrChunkFilter::new(&expr, Arc::new(schema.clone())).unwrap();

        // rows 3 and 4 satisfy both comparisons, so the chunk passes. the rows
        // with nulls must be treated as "does not pass" rather than panicking.
        let filter_passed = chunk_filter.evaluate(&batch).unwrap();
        assert!(filter_passed);
    }

    #[test]
    fn test_chunk_filter() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
            Field::new("c", DataType::Int32, false),
            Field::new("d", DataType::Int32, false),
        ]);
        let a = Int32Array::from(vec![1, 2, 3, 4, 5, 6]);
        let b = Int32Array::from(vec![3, 3, 3, 3, 3, 3]);
        let c = Int32Array::from(vec![1, 1, 2, 2, 4, 4]);
        let d = Int32Array::from(vec![2, 3, 1, 1, 1, 1]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(a), Arc::new(b), Arc::new(c), Arc::new(d)],
        )
        .unwrap();

        // expression: "b < c AND a < d". each predicate is evaluated on its own,
        // so the chunk passes because some row satisfies "b < c" and some other
        // row satisfies "a < d", even though no single row satisfies both.
        let expr = binary(
            binary(
                col("b", &schema).unwrap(),
                Operator::Lt,
                col("c", &schema).unwrap(),
                &schema,
            )
            .unwrap(),
            Operator::And,
            binary(
                col("a", &schema).unwrap(),
                Operator::Lt,
                col("d", &schema).unwrap(),
                &schema,
            )
            .unwrap(),
            &schema,
        )
        .unwrap();

        let chunk_filter = ZarrChunkFilter::new(&expr, Arc::new(schema.clone())).unwrap();
        assert!(evaluate_all(&chunk_filter, &batch));

        // expression: "b < c AND d > b". some rows satisfy "b < c", but no row
        // satisfies "d > b", so the chunk doesn't pass.
        let expr = binary(
            binary(
                col("b", &schema).unwrap(),
                Operator::Lt,
                col("c", &schema).unwrap(),
                &schema,
            )
            .unwrap(),
            Operator::And,
            binary(
                col("d", &schema).unwrap(),
                Operator::Gt,
                col("b", &schema).unwrap(),
                &schema,
            )
            .unwrap(),
            &schema,
        )
        .unwrap();

        let chunk_filter = ZarrChunkFilter::new(&expr, Arc::new(schema.clone())).unwrap();
        assert!(!evaluate_all(&chunk_filter, &batch));

        // expression: "b < c OR d > b". the same conditions as above, but the
        // rows that satisfy "b < c" are enough for the chunk to pass.
        let expr = binary(
            binary(
                col("b", &schema).unwrap(),
                Operator::Lt,
                col("c", &schema).unwrap(),
                &schema,
            )
            .unwrap(),
            Operator::Or,
            binary(
                col("d", &schema).unwrap(),
                Operator::Gt,
                col("b", &schema).unwrap(),
                &schema,
            )
            .unwrap(),
            &schema,
        )
        .unwrap();

        let chunk_filter = ZarrChunkFilter::new(&expr, Arc::new(schema.clone())).unwrap();
        assert!(evaluate_all(&chunk_filter, &batch));

        let expr = binary(
            col("b", &schema).unwrap(),
            Operator::Lt,
            col("c", &schema).unwrap(),
            &schema,
        )
        .unwrap();
        let chunk_filter = ZarrChunkFilter::new(&expr, Arc::new(schema.clone())).unwrap();
        let filter_expr = chunk_filter.predicates.values().next().unwrap();
        assert_eq!(
            vec!["b", "c"],
            filter_expr
                .filter_schema
                .fields()
                .iter()
                .map(|f| f.name())
                .collect::<Vec<_>>()
        );
    }
}
