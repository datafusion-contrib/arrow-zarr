use crate::reader::{ZarrArrowPredicate, ZarrChunkFilter, ZarrProjection};
use arrow::array::BooleanArray;
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use arrow_schema::Schema;
use datafusion_common::cast::as_boolean_array;
use datafusion_common::tree_node::{RewriteRecursion, TreeNode, TreeNodeRewriter, VisitRecursion};
use datafusion_common::Result as DataFusionResult;
use datafusion_common::{internal_err, DataFusionError};
use datafusion_expr::{Expr, ScalarFunctionDefinition, Volatility};
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::utils::reassign_predicate_columns;
use datafusion_physical_expr::{split_conjunction, PhysicalExpr};
use std::collections::BTreeSet;
use std::sync::Arc;

// Checks whether the given expression can be resolved using only the columns `col_names`.
// Copied from datafusion, because it's not accessible from the outside.
pub fn expr_applicable_for_cols(col_names: &[String], expr: &Expr) -> bool {
    let mut is_applicable = true;
    expr.apply(&mut |expr| match expr {
        Expr::Column(datafusion_common::Column { ref name, .. }) => {
            is_applicable &= col_names.contains(name);
            if is_applicable {
                Ok(VisitRecursion::Skip)
            } else {
                Ok(VisitRecursion::Stop)
            }
        }
        Expr::Literal(_)
        | Expr::Alias(_)
        | Expr::OuterReferenceColumn(_, _)
        | Expr::ScalarVariable(_, _)
        | Expr::Not(_)
        | Expr::IsNotNull(_)
        | Expr::IsNull(_)
        | Expr::IsTrue(_)
        | Expr::IsFalse(_)
        | Expr::IsUnknown(_)
        | Expr::IsNotTrue(_)
        | Expr::IsNotFalse(_)
        | Expr::IsNotUnknown(_)
        | Expr::Negative(_)
        | Expr::Cast { .. }
        | Expr::TryCast { .. }
        | Expr::BinaryExpr { .. }
        | Expr::Between { .. }
        | Expr::Like { .. }
        | Expr::SimilarTo { .. }
        | Expr::InList { .. }
        | Expr::Exists { .. }
        | Expr::InSubquery(_)
        | Expr::ScalarSubquery(_)
        | Expr::GetIndexedField { .. }
        | Expr::GroupingSet(_)
        | Expr::Case { .. } => Ok(VisitRecursion::Continue),

        Expr::ScalarFunction(scalar_function) => match &scalar_function.func_def {
            ScalarFunctionDefinition::BuiltIn(fun) => match fun.volatility() {
                Volatility::Immutable => Ok(VisitRecursion::Continue),
                Volatility::Stable | Volatility::Volatile => {
                    is_applicable = false;
                    Ok(VisitRecursion::Stop)
                }
            },
            ScalarFunctionDefinition::UDF(fun) => match fun.signature().volatility {
                Volatility::Immutable => Ok(VisitRecursion::Continue),
                Volatility::Stable | Volatility::Volatile => {
                    is_applicable = false;
                    Ok(VisitRecursion::Stop)
                }
            },
            ScalarFunctionDefinition::Name(_) => {
                internal_err!("Function `Expr` with name should be resolved.")
            }
        },

        Expr::AggregateFunction { .. }
        | Expr::Sort { .. }
        | Expr::WindowFunction { .. }
        | Expr::Wildcard { .. }
        | Expr::Unnest { .. }
        | Expr::Placeholder(_) => {
            is_applicable = false;
            Ok(VisitRecursion::Stop)
        }
    })
    .unwrap();
    is_applicable
}

// Below is all the logic necessary (I think) to convert a PhysicalExpr into a ZarrChunkFilter.
// The logic is mostly copied from datafusion, and is simplified here for the zarr use case.
pub struct ZarrFilterCandidate {
    expr: Arc<dyn PhysicalExpr>,
    projection: Vec<usize>,
}

struct ZarrFilterCandidateBuilder<'a> {
    expr: Arc<dyn PhysicalExpr>,
    file_schema: &'a Schema,
    required_column_indices: BTreeSet<usize>,
}

impl<'a> ZarrFilterCandidateBuilder<'a> {
    pub fn new(expr: Arc<dyn PhysicalExpr>, file_schema: &'a Schema) -> Self {
        Self {
            expr,
            file_schema,
            required_column_indices: BTreeSet::default(),
        }
    }

    pub fn build(mut self) -> DataFusionResult<Option<ZarrFilterCandidate>> {
        let expr = self.expr.clone().rewrite(&mut self)?;

        Ok(Some(ZarrFilterCandidate {
            expr,
            projection: self.required_column_indices.into_iter().collect(),
        }))
    }
}

impl<'a> TreeNodeRewriter for ZarrFilterCandidateBuilder<'a> {
    type N = Arc<dyn PhysicalExpr>;

    fn pre_visit(&mut self, node: &Arc<dyn PhysicalExpr>) -> DataFusionResult<RewriteRecursion> {
        if let Some(column) = node.as_any().downcast_ref::<Column>() {
            if let Ok(idx) = self.file_schema.index_of(column.name()) {
                self.required_column_indices.insert(idx);
            }
        }

        Ok(RewriteRecursion::Continue)
    }

    fn mutate(&mut self, expr: Arc<dyn PhysicalExpr>) -> DataFusionResult<Arc<dyn PhysicalExpr>> {
        Ok(expr)
    }
}

#[derive(Clone)]
pub struct ZarrDatafusionArrowPredicate {
    physical_expr: Arc<dyn PhysicalExpr>,
    projection_mask: ZarrProjection,
    projection: Vec<String>,
}

impl ZarrDatafusionArrowPredicate {
    pub fn new(candidate: ZarrFilterCandidate, schema: &Schema) -> DataFusionResult<Self> {
        let cols: Vec<_> = candidate
            .projection
            .iter()
            .map(|idx| schema.field(*idx).name().to_string())
            .collect();

        let schema = Arc::new(schema.project(&candidate.projection)?);
        let physical_expr = reassign_predicate_columns(candidate.expr, &schema, true)?;

        Ok(Self {
            physical_expr,
            projection_mask: ZarrProjection::keep(cols.clone()),
            projection: cols,
        })
    }
}

impl ZarrArrowPredicate for ZarrDatafusionArrowPredicate {
    fn projection(&self) -> &ZarrProjection {
        &self.projection_mask
    }

    fn evaluate(&mut self, batch: &RecordBatch) -> Result<BooleanArray, ArrowError> {
        let index_projection = self
            .projection
            .iter()
            .map(|col| batch.schema().index_of(col))
            .collect::<Result<Vec<_>, _>>()?;
        let batch = batch.project(&index_projection[..])?;

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

pub(crate) fn build_row_filter(
    expr: &Arc<dyn PhysicalExpr>,
    file_schema: &Schema,
) -> DataFusionResult<Option<ZarrChunkFilter>> {
    let predicates = split_conjunction(expr);
    let candidates: Vec<ZarrFilterCandidate> = predicates
        .into_iter()
        .flat_map(|expr| {
            if let Ok(candidate) =
                ZarrFilterCandidateBuilder::new(expr.clone(), file_schema).build()
            {
                candidate
            } else {
                None
            }
        })
        .collect();

    if candidates.is_empty() {
        Ok(None)
    } else {
        let mut filters: Vec<Box<dyn ZarrArrowPredicate>> = vec![];
        for candidate in candidates {
            let filter = ZarrDatafusionArrowPredicate::new(candidate, file_schema)?;
            filters.push(Box::new(filter));
        }

        let chunk_filter = ZarrChunkFilter::new(filters);

        Ok(Some(chunk_filter))
    }
}
