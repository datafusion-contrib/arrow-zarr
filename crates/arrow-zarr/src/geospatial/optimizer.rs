use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{HashMap, JoinSide, Result};
use datafusion::config::ConfigOptions;
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::expressions::{BinaryExpr, Column};
use datafusion::physical_expr::ScalarFunctionExpr;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::joins::utils::{ColumnIndex, JoinFilter};
use datafusion::physical_plan::joins::NestedLoopJoinExec;
use datafusion::physical_plan::{ExecutionPlan, PhysicalExpr};

use super::exec::BulkSpatialJoinExec;
use super::spatial_predicate::{RelationPredicate, SpatialRelationType};

#[derive(Debug, Default)]
pub struct SpatialJoinPhysicalOptimizer;

impl SpatialJoinPhysicalOptimizer {
    pub fn new() -> Self {
        Self
    }
}

impl PhysicalOptimizerRule for SpatialJoinPhysicalOptimizer {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(plan.transform_up(try_optimize_join)?.data)
    }

    fn name(&self) -> &str {
        "bulk_spatial_join_physical_optimizer"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

fn try_optimize_join(plan: Arc<dyn ExecutionPlan>) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
    let Some(nlj) = plan.as_any().downcast_ref::<NestedLoopJoinExec>() else {
        return Ok(Transformed::no(plan));
    };
    let Some(spatial_join) = try_convert_to_spatial_join(nlj)? else {
        return Ok(Transformed::no(plan));
    };
    Ok(Transformed::yes(spatial_join))
}

fn try_convert_to_spatial_join(nlj: &NestedLoopJoinExec) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let Some(join_filter) = nlj.filter() else {
        return Ok(None);
    };

    let Some((predicate, remainder)) = transform_join_filter(join_filter) else {
        return Ok(None);
    };

    let left = nlj.left();
    let left = if let Some(coalesce) = left.as_any().downcast_ref::<CoalescePartitionsExec>() {
        coalesce.input()
    } else {
        left
    };

    let exec = BulkSpatialJoinExec::try_new(
        left.clone(),
        nlj.right().clone(),
        predicate,
        remainder,
        *nlj.join_type(),
        nlj.projection().cloned(),
    )?;

    Ok(Some(Arc::new(exec)))
}

fn transform_join_filter(jf: &JoinFilter) -> Option<(RelationPredicate, Option<JoinFilter>)> {
    let (predicate, remainder_expr) =
        extract_spatial_predicate(jf.expression(), jf.column_indices())?;

    let remainder = remainder_expr
        .as_ref()
        .map(|expr| replace_join_filter_expr(expr, jf));

    Some((predicate, remainder))
}

fn replace_join_filter_expr(expr: &Arc<dyn PhysicalExpr>, jf: &JoinFilter) -> JoinFilter {
    let column_refs = collect_column_references(expr, jf.column_indices());

    let referenced: Vec<(usize, &ColumnIndex)> = jf
        .column_indices()
        .iter()
        .enumerate()
        .filter(|(_, col_idx)| column_refs.contains(col_idx))
        .collect();

    let pruned_column_indices: Vec<ColumnIndex> =
        referenced.iter().map(|(_, ci)| (*ci).clone()).collect();

    let column_index_mapping: HashMap<usize, usize> = referenced
        .iter()
        .enumerate()
        .map(|(new_idx, (old_idx, _))| (*old_idx, new_idx))
        .collect();

    let old_indices: Vec<usize> = referenced.iter().map(|(old_idx, _)| *old_idx).collect();
    let pruned_schema = jf
        .schema()
        .project(&old_indices)
        .expect("failed to project join filter schema");

    let reprojected = reproject_column_references(expr, &column_index_mapping);

    JoinFilter::new(reprojected, pruned_column_indices, Arc::new(pruned_schema))
}

fn extract_spatial_predicate(
    expr: &Arc<dyn PhysicalExpr>,
    column_indices: &[ColumnIndex],
) -> Option<(RelationPredicate, Option<Arc<dyn PhysicalExpr>>)> {
    if let Some(scalar_fn) = expr.as_any().downcast_ref::<ScalarFunctionExpr>() {
        if let Some(predicate) = match_relation_predicate(scalar_fn, column_indices) {
            return Some((predicate, None));
        }
    }

    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        if !matches!(binary.op(), Operator::And) {
            return None;
        }
        let left = binary.left();
        let right = binary.right();

        if let Some((predicate, left_remainder)) = extract_spatial_predicate(left, column_indices) {
            let remainder = match left_remainder {
                Some(r) => Arc::new(BinaryExpr::new(r, Operator::And, right.clone()))
                    as Arc<dyn PhysicalExpr>,
                None => right.clone(),
            };
            return Some((predicate, Some(remainder)));
        }

        if let Some((predicate, right_remainder)) = extract_spatial_predicate(right, column_indices)
        {
            let remainder = match right_remainder {
                Some(r) => Arc::new(BinaryExpr::new(left.clone(), Operator::And, r))
                    as Arc<dyn PhysicalExpr>,
                None => left.clone(),
            };
            return Some((predicate, Some(remainder)));
        }
    }

    None
}

fn match_relation_predicate(
    scalar_fn: &ScalarFunctionExpr,
    column_indices: &[ColumnIndex],
) -> Option<RelationPredicate> {
    let relation_type = SpatialRelationType::from_name(scalar_fn.fun().name())?;

    let args = scalar_fn.args();
    assert!(args.len() >= 2);
    let arg0 = &args[0];
    let arg1 = &args[1];

    let refs0 = collect_column_references(arg0, column_indices);
    let refs1 = collect_column_references(arg1, column_indices);

    let (side0, side1) = resolve_column_reference_sides(&refs0, &refs1)?;

    let arg0_repr = reproject_column_references_for_side(arg0, column_indices, side0);
    let arg1_repr = reproject_column_references_for_side(arg1, column_indices, side1);

    Some(RelationPredicate::new(arg0_repr, arg1_repr, relation_type))
}

fn collect_column_references(
    expr: &Arc<dyn PhysicalExpr>,
    column_indices: &[ColumnIndex],
) -> Vec<ColumnIndex> {
    let mut result = Vec::new();
    expr.apply(|node| {
        if let Some(col) = node.as_any().downcast_ref::<Column>() {
            result.push(column_indices[col.index()].clone());
        }
        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
    })
    .expect("collect_column_references failed");
    result
}

fn resolve_column_reference_sides(
    refs0: &[ColumnIndex],
    refs1: &[ColumnIndex],
) -> Option<(JoinSide, JoinSide)> {
    let side0 = side_of_column_references(refs0)?;
    let side1 = side_of_column_references(refs1)?;
    if side0 != side1 {
        Some((side0, side1))
    } else {
        None
    }
}

fn side_of_column_references(refs: &[ColumnIndex]) -> Option<JoinSide> {
    let Some(first) = refs.first() else {
        return Some(JoinSide::None);
    };
    let side = first.side;
    if refs.iter().all(|r| r.side == side) {
        Some(side)
    } else {
        None
    }
}

fn reproject_column_references_for_side(
    expr: &Arc<dyn PhysicalExpr>,
    column_indices: &[ColumnIndex],
    side: JoinSide,
) -> Arc<dyn PhysicalExpr> {
    if side == JoinSide::None {
        return expr.clone();
    }
    let index_map: HashMap<usize, usize> = column_indices
        .iter()
        .enumerate()
        .filter_map(|(i, ci)| (ci.side == side).then_some((i, ci.index)))
        .collect();
    reproject_column_references(expr, &index_map)
}

fn reproject_column_references(
    expr: &Arc<dyn PhysicalExpr>,
    index_map: &HashMap<usize, usize>,
) -> Arc<dyn PhysicalExpr> {
    expr.clone()
        .transform_down(|node| {
            if let Some(col) = node.as_any().downcast_ref::<Column>() {
                if let Some(&new_idx) = index_map.get(&col.index()) {
                    return Ok(Transformed::yes(
                        Arc::new(Column::new(col.name(), new_idx)) as Arc<dyn PhysicalExpr>
                    ));
                }
            }
            Ok(Transformed::no(node))
        })
        .unwrap_or_else(|_| Transformed::no(expr.clone()))
        .data
}

#[cfg(test)]
mod optimizer_tests {
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::common::JoinType;
    use datafusion::logical_expr::ScalarUDF;
    use datafusion::physical_plan::empty::EmptyExec;
    use datafusion::physical_plan::joins::NestedLoopJoinExec;

    use super::super::spatial_predicate::SpatialRelationType;
    use super::super::udfs::StWithinUdf;
    use super::*;

    fn within_bulk_expr(left_idx: usize, right_idx: usize) -> Arc<dyn PhysicalExpr> {
        let udf = Arc::new(ScalarUDF::from(StWithinUdf::default()));
        Arc::new(ScalarFunctionExpr::new(
            "st_within_bulk",
            udf,
            vec![
                Arc::new(Column::new("left_geom", left_idx)),
                Arc::new(Column::new("right_geom", right_idx)),
            ],
            Arc::new(Field::new("result", DataType::Boolean, true)),
        ))
    }

    fn two_col_indices() -> Vec<ColumnIndex> {
        vec![
            ColumnIndex {
                index: 0,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 0,
                side: JoinSide::Right,
            },
        ]
    }

    fn four_col_indices() -> Vec<ColumnIndex> {
        vec![
            ColumnIndex {
                index: 0,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 0,
                side: JoinSide::Right,
            },
            ColumnIndex {
                index: 1,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 1,
                side: JoinSide::Right,
            },
        ]
    }

    #[test]
    fn test_only_predicate() {
        let expr = within_bulk_expr(0, 1);
        let (predicate, remainder) = extract_spatial_predicate(&expr, &two_col_indices())
            .expect("should extract spatial predicate");
        assert_eq!(predicate.relation_type, SpatialRelationType::Within);
        assert!(remainder.is_none());
    }

    #[test]
    fn test_predicate_left_of_and() {
        let spatial = within_bulk_expr(0, 1);
        let extra = Arc::new(Column::new("extra", 2)) as Arc<dyn PhysicalExpr>;
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(spatial, Operator::And, extra));
        let (predicate, remainder) = extract_spatial_predicate(&expr, &four_col_indices())
            .expect("should extract spatial predicate");
        assert_eq!(predicate.relation_type, SpatialRelationType::Within);
        assert!(remainder.is_some());
    }

    #[test]
    fn test_predicate_right_of_and() {
        let spatial = within_bulk_expr(0, 1);
        let extra = Arc::new(Column::new("extra", 2)) as Arc<dyn PhysicalExpr>;
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(extra, Operator::And, spatial));
        let (predicate, remainder) = extract_spatial_predicate(&expr, &four_col_indices())
            .expect("should extract spatial predicate");
        assert_eq!(predicate.relation_type, SpatialRelationType::Within);
        assert!(remainder.is_some());
    }

    #[test]
    fn test_no_match_or_condition() {
        let spatial = within_bulk_expr(0, 1);
        let extra = Arc::new(Column::new("extra", 2)) as Arc<dyn PhysicalExpr>;
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(spatial, Operator::Or, extra));
        assert!(extract_spatial_predicate(&expr, &four_col_indices()).is_none());
    }

    #[test]
    fn test_no_match_no_spatial_fn() {
        let left = Arc::new(Column::new("a", 0)) as Arc<dyn PhysicalExpr>;
        let right = Arc::new(Column::new("b", 1)) as Arc<dyn PhysicalExpr>;
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(left, Operator::And, right));
        assert!(extract_spatial_predicate(&expr, &two_col_indices()).is_none());
    }

    fn geom_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]))
    }

    #[test]
    fn test_optimizer_converts_nlj_to_spatial_join() {
        let left = Arc::new(EmptyExec::new(geom_schema())) as Arc<dyn ExecutionPlan>;
        let right = Arc::new(EmptyExec::new(geom_schema())) as Arc<dyn ExecutionPlan>;

        let filter_schema = Arc::new(Schema::new(vec![
            Field::new("left_geom", DataType::Binary, true),
            Field::new("right_geom", DataType::Binary, true),
        ]));
        let column_indices = vec![
            ColumnIndex {
                index: 0,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 0,
                side: JoinSide::Right,
            },
        ];
        let filter = JoinFilter::new(within_bulk_expr(0, 1), column_indices, filter_schema);

        let nlj =
            NestedLoopJoinExec::try_new(left, right, Some(filter), &JoinType::Inner, None).unwrap();

        let optimized = SpatialJoinPhysicalOptimizer::new()
            .optimize(Arc::new(nlj), &ConfigOptions::default())
            .unwrap();

        let spatial = optimized
            .as_any()
            .downcast_ref::<BulkSpatialJoinExec>()
            .expect("should be converted to BulkSpatialJoinExec");

        assert_eq!(spatial.join_type, JoinType::Inner);
        assert_eq!(spatial.predicate.relation_type, SpatialRelationType::Within);
        assert!(spatial.filter.is_none());
    }
}
