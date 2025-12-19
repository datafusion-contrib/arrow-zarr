use std::collections::BTreeSet;
use std::sync::Arc;

use arrow_array::{BooleanArray, RecordBatch};
use arrow_schema::{ArrowError, SchemaRef};
use datafusion::common::cast::as_boolean_array;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion, TreeNodeVisitor};
use datafusion::common::Result as DfResult;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::utils::reassign_predicate_columns;
use datafusion::physical_expr::{split_conjunction, PhysicalExpr};
use itertools::Itertools;

/// A predicate operating on [`RecordBatch`].
pub trait ZarrArrowPredicate: Send + 'static {
    /// Evaluate this predicate for the given [`RecordBatch`]. Rows that are `true`
    /// in the returned [`BooleanArray`] satisfy the predicate condition, whereas those
    /// that are `false` do not. The method should not return any `Null` values.
    /// Note that the [`RecordBatch`] is passed by reference and not consumed by
    /// the method.
    fn evaluate(&self, batch: &RecordBatch) -> Result<BooleanArray, ArrowError>;
}

struct ZarrFilterExpression {
    physical_expr: Arc<dyn PhysicalExpr>,
    filter_schema: SchemaRef,
    required_columns: Vec<usize>,
}

impl ZarrFilterExpression {
    fn new(physical_expr: Arc<dyn PhysicalExpr>, table_schema: SchemaRef) -> DfResult<Self> {
        // this part is to make sure that the indices for each column in the
        // predicate match the columns in the filter schema. the physical
        // expressions are created from the full table schema initally, but
        // the record batches will come in with the filter schema, that's why
        // this step is needed.
        let required_columns = pushdown_columns(&physical_expr, table_schema.clone())?;
        let filter_schema = table_schema.project(&required_columns)?;
        let physical_expr = reassign_predicate_columns(physical_expr, &filter_schema, true)?;

        Ok(Self {
            physical_expr,
            filter_schema: Arc::new(filter_schema),
            required_columns,
        })
    }
}

impl ZarrArrowPredicate for ZarrFilterExpression {
    fn evaluate(&self, batch: &RecordBatch) -> Result<BooleanArray, ArrowError> {
        // if there was only one filter expression in the full chunk filter,
        // the record batch would come in with the right schema all the time
        // (because the caller would first check what schema is required for
        // filter, only evaluate those columns and pass in that record batch).
        // but there could be multiple expressions in the final, full chunk
        // filter, so in practice the record batch is a superset of what each
        // individual filter needs, hence we need to project it to make sure
        // field/column indices match between the expression and the record
        // batch schema.
        let batch = batch.project(
            &(self
                .filter_schema
                .fields()
                .iter()
                .map(|f| batch.schema().index_of(f.name()))
                .collect::<Result<Vec<_>, _>>()?[..]),
        )?;

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

/// A struct that implements TreeNodeRewriter to traverse a PhysicalExpr tree structure
/// to determine which columns are required to evaluate it.
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
        if let Some(column) = node.as_any().downcast_ref::<Column>() {
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

/// A collection of one or more objects that implement [`ZarrArrowPredicate`]. The way
/// filters are used for zarr store is by determining whether or not the a chunk needs to be
/// read based on the predicate. First, only the columns needed in the predicate are read,
/// then the predicate is evaluated, and if there is a least one row that satistifes the
/// condition, the other variables that we requested are read.
pub struct ZarrChunkFilter {
    /// A list of [`ZarrArrowPredicate`]
    predicates: Vec<Box<dyn ZarrArrowPredicate>>,
    schema_ref: SchemaRef,
}

impl ZarrChunkFilter {
    pub fn new(expr: &Arc<dyn PhysicalExpr>, table_schema: SchemaRef) -> Result<Self, ArrowError> {
        let predicate_exprs = split_conjunction(expr);
        let mut predicates: Vec<Box<dyn ZarrArrowPredicate>> =
            Vec::with_capacity(predicate_exprs.len());
        let mut schema_indices: Vec<usize> = Vec::new();

        // we don't bother reorganizing filters to start with the cheaper ones
        // before we do the more expensive ones. in terms of the amount of data
        // read, it's the same for all the chunks. some operations might be
        // cheaper to check (computationally), not sure how e.g. the parquet
        // case handles this, I might revisit later to optimize things a bit.
        for pred_expr in predicate_exprs {
            let filter_expr = ZarrFilterExpression::new(pred_expr.clone(), table_schema.clone())?;
            schema_indices.extend(filter_expr.required_columns.clone());
            predicates.push(Box::new(filter_expr));
        }

        schema_indices.sort();
        schema_indices.dedup();
        let table_schema = table_schema.project(&schema_indices)?;

        Ok(Self {
            predicates,
            schema_ref: Arc::new(table_schema),
        })
    }

    pub fn schema_ref(&self) -> &SchemaRef {
        &self.schema_ref
    }

    /// Applies all the filters in the chunk filter and returns true only
    /// if all the filters return true for at least one row in the record
    /// batch.
    pub fn evaluate(&self, rec_batch: &RecordBatch) -> Result<bool, ArrowError> {
        let mut bool_arr: Option<BooleanArray> = None;
        for predicate in self.predicates.iter() {
            let mask = predicate.evaluate(rec_batch)?;
            if let Some(old_bool_arr) = bool_arr {
                bool_arr = Some(BooleanArray::from(
                    old_bool_arr
                        .iter()
                        .zip(mask.iter())
                        .map(|(x, y)| x.unwrap() && y.unwrap())
                        .collect_vec(),
                ));
            } else {
                bool_arr = Some(mask);
            }
        }

        if let Some(bool_arr) = bool_arr {
            Ok(bool_arr.true_count() > 0)
        } else {
            Ok(true)
        }
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

        // expression: "b < c AND a < d"
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

        let chunk_filter = ZarrChunkFilter::new(&Arc::new(expr), Arc::new(schema.clone())).unwrap();
        let filter_passed = chunk_filter.evaluate(&batch).unwrap();
        assert!(!filter_passed);

        // expression: "b < c OR a < d"
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
                col("a", &schema).unwrap(),
                Operator::Lt,
                col("d", &schema).unwrap(),
                &schema,
            )
            .unwrap(),
            &schema,
        )
        .unwrap();

        let chunk_filter = ZarrChunkFilter::new(&Arc::new(expr), Arc::new(schema.clone())).unwrap();
        let filter_passed = chunk_filter.evaluate(&batch).unwrap();
        assert!(filter_passed);

        let expr = binary(
            col("b", &schema).unwrap(),
            Operator::Lt,
            col("c", &schema).unwrap(),
            &schema,
        )
        .unwrap();
        let chunk_filter = ZarrChunkFilter::new(&Arc::new(expr), Arc::new(schema.clone())).unwrap();
        assert_eq!(
            vec!["b", "c"],
            chunk_filter
                .schema_ref()
                .fields()
                .iter()
                .map(|f| f.name())
                .collect::<Vec<_>>()
        );
    }
}
