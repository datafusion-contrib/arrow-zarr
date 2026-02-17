use std::fmt::Formatter;
use std::sync::{Arc, Mutex};

use arrow_schema::SchemaRef;
use datafusion::common::{project_schema, DataFusionError, JoinType, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::equivalence::{join_equivalence_properties, ProjectionMapping};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::joins::utils::{
    adjust_right_output_partitioning, build_join_schema, check_join_is_valid, ColumnIndex,
    JoinFilter,
};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties, SendableRecordBatchStream,
};
use futures::future::{BoxFuture, FutureExt, Shared};
use futures::TryStreamExt;

use super::build_side::{BuildSide, BuildSideBuilder, ProcessedBatch};
use super::spatial_predicate::RelationPredicate;
use super::stream::SpatialJoinStream;

type BuildSideFut = Shared<BoxFuture<'static, Result<Arc<BuildSide>, Arc<DataFusionError>>>>;

/// Physical execution plan for spatial joins.
#[derive(Debug)]
pub struct BulkSpatialJoinExec {
    pub(crate) left: Arc<dyn ExecutionPlan>,
    pub(crate) right: Arc<dyn ExecutionPlan>,
    pub(crate) predicate: RelationPredicate,
    pub(crate) filter: Option<JoinFilter>,
    pub(crate) join_type: JoinType,
    join_schema: SchemaRef,
    column_indices: Vec<ColumnIndex>,
    projection: Option<Vec<usize>>,
    cache: PlanProperties,

    /// Lazily initialised on the first `execute()` call, then shared across all probe partitions.
    build_side_shared: Arc<Mutex<Option<BuildSideFut>>>,
}

impl BulkSpatialJoinExec {
    /// Create a new [`BulkSpatialJoinExec`].
    ///
    /// `left` is the build side; `right` is the probe side.
    pub fn try_new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        predicate: RelationPredicate,
        filter: Option<JoinFilter>,
        join_type: JoinType,
        projection: Option<Vec<usize>>,
    ) -> Result<Self> {
        match join_type {
            JoinType::Inner | JoinType::Left => {}
            other => {
                return Err(DataFusionError::NotImplemented(format!(
                    "BulkSpatialJoinExec does not support join type {other:?}"
                )));
            }
        }

        let left_schema = left.schema();
        let right_schema = right.schema();
        check_join_is_valid(&left_schema, &right_schema, &[])?;
        let (join_schema, column_indices) =
            build_join_schema(&left_schema, &right_schema, &join_type);
        let join_schema = Arc::new(join_schema);
        let cache = Self::compute_properties(
            &left,
            &right,
            &join_type,
            Arc::clone(&join_schema),
            projection.as_ref(),
        )?;

        Ok(Self {
            left,
            right,
            predicate,
            filter,
            join_type,
            join_schema,
            column_indices,
            projection,
            cache,
            build_side_shared: Arc::new(Mutex::new(None)),
        })
    }

    fn compute_properties(
        left: &Arc<dyn ExecutionPlan>,
        right: &Arc<dyn ExecutionPlan>,
        join_type: &JoinType,
        join_schema: SchemaRef,
        projection: Option<&Vec<usize>>,
    ) -> Result<PlanProperties> {
        let mut eq_properties = join_equivalence_properties(
            left.equivalence_properties().clone(),
            right.equivalence_properties().clone(),
            join_type,
            Arc::clone(&join_schema),
            &[false, false],
            None,
            &[],
        )?;

        let mut output_partitioning = match join_type {
            JoinType::Inner => adjust_right_output_partitioning(
                right.output_partitioning(),
                left.schema().fields().len(),
            )?,
            JoinType::Left => {
                Partitioning::UnknownPartitioning(right.output_partitioning().partition_count())
            }
            _ => unreachable!(),
        };

        let emission_type = if left.boundedness().is_unbounded() {
            EmissionType::Final
        } else if right.pipeline_behavior() == EmissionType::Incremental {
            match join_type {
                JoinType::Inner => EmissionType::Incremental,
                // Left join emits matched rows incrementally, then unmatched build rows at the end.
                JoinType::Left => EmissionType::Both,
                _ => unreachable!(),
            }
        } else {
            right.pipeline_behavior()
        };

        let boundedness = match (left.boundedness(), right.boundedness()) {
            (
                Boundedness::Unbounded {
                    requires_infinite_memory: true,
                },
                _,
            )
            | (
                _,
                Boundedness::Unbounded {
                    requires_infinite_memory: true,
                },
            ) => Boundedness::Unbounded {
                requires_infinite_memory: true,
            },
            (Boundedness::Unbounded { .. }, _) | (_, Boundedness::Unbounded { .. }) => {
                Boundedness::Unbounded {
                    requires_infinite_memory: false,
                }
            }
            _ => Boundedness::Bounded,
        };

        if let Some(projection) = projection {
            let projection_mapping = ProjectionMapping::from_indices(projection, &join_schema)?;
            let out_schema = project_schema(&join_schema, Some(projection))?;
            output_partitioning = output_partitioning.project(&projection_mapping, &eq_properties);
            eq_properties = eq_properties.project(&projection_mapping, out_schema);
        }

        Ok(PlanProperties::new(
            eq_properties,
            output_partitioning,
            emission_type,
            boundedness,
        ))
    }
}

impl DisplayAs for BulkSpatialJoinExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let p = &self.predicate;
                let display_on = format!(", on={:?}({}, {})", p.relation_type, p.left, p.right);
                let display_filter = self.filter.as_ref().map_or_else(
                    || "".to_string(),
                    |f| format!(", filter={}", f.expression()),
                );
                let display_projection = self.projection.as_ref().map_or_else(
                    || "".to_string(),
                    |proj| {
                        format!(
                            ", projection=[{}]",
                            proj.iter()
                                .map(|i| format!(
                                    "{}@{}",
                                    self.join_schema.fields().get(*i).unwrap().name(),
                                    i
                                ))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    },
                );
                write!(
                    f,
                    "BulkSpatialJoinExec: join_type={:?}{}{}{}",
                    self.join_type, display_on, display_filter, display_projection
                )
            }
            DisplayFormatType::TreeRender => {
                if self.join_type != JoinType::Inner {
                    writeln!(f, "join_type={:?}", self.join_type)
                } else {
                    Ok(())
                }
            }
        }
    }
}

impl ExecutionPlan for BulkSpatialJoinExec {
    fn name(&self) -> &str {
        "BulkSpatialJoinExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![false, false]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.left, &self.right]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::try_new(
            children[0].clone(),
            children[1].clone(),
            self.predicate.clone(),
            self.filter.clone(),
            self.join_type,
            self.projection.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // Lazily init the shared build future (once per exec instance).
        let build_side_fut = {
            let mut guard = self
                .build_side_shared
                .lock()
                .map_err(|e| DataFusionError::Internal(e.to_string()))?;
            if guard.is_none() {
                let left = Arc::clone(&self.left);
                let geo_expr = Arc::clone(&self.predicate.left);
                let join_type = self.join_type;
                let probe_partition_count = self.right.output_partitioning().partition_count();
                let ctx = Arc::clone(&context);

                let fut: BoxFuture<'static, Result<Arc<BuildSide>, Arc<DataFusionError>>> =
                    Box::pin(async move {
                        let num_build_partitions = left.output_partitioning().partition_count();
                        let mut builder = BuildSideBuilder::new();
                        for k in 0..num_build_partitions {
                            let mut stream = left.execute(k, Arc::clone(&ctx)).map_err(Arc::new)?;
                            let mut processed = Vec::new();
                            while let Some(batch) = stream.try_next().await.map_err(Arc::new)? {
                                processed
                                    .push(ProcessedBatch::new(batch, &geo_expr).map_err(Arc::new)?);
                            }
                            builder.add_partition(processed);
                        }
                        let needs_visited = matches!(join_type, JoinType::Left);
                        Ok(Arc::new(
                            builder.build(needs_visited, probe_partition_count),
                        ))
                    });
                *guard = Some(fut.shared());
            }
            guard
                .as_ref()
                .ok_or_else(|| {
                    DataFusionError::Internal("build side future not initialized".into())
                })?
                .clone()
        };

        let column_indices = match &self.projection {
            Some(p) => p.iter().map(|&i| self.column_indices[i].clone()).collect(),
            None => self.column_indices.clone(),
        };

        let probe_stream = self.right.execute(partition, context)?;

        Ok(Box::pin(SpatialJoinStream::new(
            self.schema(),
            self.join_type,
            self.filter.clone(),
            probe_stream,
            column_indices,
            build_side_fut,
            self.predicate.clone(),
        )))
    }
}

#[cfg(test)]
mod exec_tests {
    use std::sync::Arc;

    use datafusion::assert_batches_sorted_eq;
    use datafusion::common::JoinType;
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::execution::TaskContext;
    use datafusion::physical_expr::expressions::Column;
    use datafusion::physical_plan::joins::utils::JoinFilter;
    use datafusion::physical_plan::{ExecutionPlan, PhysicalExpr};
    use futures::TryStreamExt;

    use super::BulkSpatialJoinExec;
    use crate::geospatial::spatial_predicate::{RelationPredicate, SpatialRelationType};
    use crate::geospatial::test_utils::{
        build_schema, col4_gt_zero_filter, make_build_batch, make_probe_batches, probe_schema,
    };

    fn make_exec(
        join_type: JoinType,
        projection: Option<Vec<usize>>,
        filter: Option<JoinFilter>,
    ) -> BulkSpatialJoinExec {
        let build_exec =
            MemorySourceConfig::try_new_exec(&[vec![make_build_batch()]], build_schema(), None)
                .unwrap();
        let probe_batches = make_probe_batches();
        let probe_exec = MemorySourceConfig::try_new_exec(
            &[probe_batches.clone(), probe_batches],
            probe_schema(),
            None,
        )
        .unwrap();
        let predicate = RelationPredicate::new(
            Arc::new(Column::new("geo", 0)) as Arc<dyn PhysicalExpr>,
            Arc::new(Column::new("geo", 0)) as Arc<dyn PhysicalExpr>,
            SpatialRelationType::Within,
        );
        BulkSpatialJoinExec::try_new(
            build_exec, probe_exec, predicate, filter, join_type, projection,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn test_inner_join() {
        // projection: col1 (Left index 1 → combined 1), col4 (Right index 2 → combined 5)
        let exec = make_exec(JoinType::Inner, Some(vec![1, 5]), None);
        let ctx = Arc::new(TaskContext::default());

        let stream0 = exec.execute(0, Arc::clone(&ctx)).unwrap();
        let stream1 = exec.execute(1, Arc::clone(&ctx)).unwrap();

        let batches0 = stream0.try_collect::<Vec<_>>().await.unwrap();
        let batches1 = stream1.try_collect::<Vec<_>>().await.unwrap();

        // Both partitions have identical probe data → same matched rows.
        let expected = [
            "+------+------+",
            "| col1 | col4 |",
            "+------+------+",
            "| 1    | 0.0  |",
            "| 2    | 0.1  |",
            "| 3    | 0.0  |",
            "| 4    | 0.1  |",
            "+------+------+",
        ];
        assert_batches_sorted_eq!(expected, &batches0);
        assert_batches_sorted_eq!(expected, &batches1);
    }

    #[tokio::test]
    async fn test_left_join() {
        // projection: col2 (Left index 2 → combined 2), col3 (Right index 1 → combined 4)
        let exec = make_exec(JoinType::Left, Some(vec![2, 4]), None);
        let ctx = Arc::new(TaskContext::default());

        let stream0 = exec.execute(0, Arc::clone(&ctx)).unwrap();
        let stream1 = exec.execute(1, Arc::clone(&ctx)).unwrap();

        // Partition 0 collected first: remaining_probes 2→1 (old=2≠1), no unmatched rows.
        let batches0 = stream0.try_collect::<Vec<_>>().await.unwrap();
        assert_batches_sorted_eq!(
            [
                "+------+------+",
                "| col2 | col3 |",
                "+------+------+",
                "| 1.1  | 100  |",
                "| 2.2  | 101  |",
                "| 3.3  | 104  |",
                "| 4.4  | 105  |",
                "+------+------+",
            ],
            &batches0
        );

        // Partition 1 collected second: remaining_probes 1→0 (old=1==1), emits unmatched build rows.
        let batches1 = stream1.try_collect::<Vec<_>>().await.unwrap();
        assert_batches_sorted_eq!(
            [
                "+------+------+",
                "| col2 | col3 |",
                "+------+------+",
                "| 1.1  | 100  |",
                "| 2.2  | 101  |",
                "| 3.3  | 104  |",
                "| 4.4  | 105  |",
                "| 5.5  |      |",
                "| 6.6  |      |",
                "| 7.7  |      |",
                "| 8.8  |      |",
                "+------+------+",
            ],
            &batches1
        );
    }

    #[tokio::test]
    async fn test_inner_join_with_filter() {
        // projection: col1 (Left index 1 → combined 1), col4 (Right index 2 → combined 5)
        // col4 > 0 removes rows where col4=0.0 (P0→B0 and P4→B2)
        let exec = make_exec(
            JoinType::Inner,
            Some(vec![1, 5]),
            Some(col4_gt_zero_filter()),
        );
        let ctx = Arc::new(TaskContext::default());

        let stream0 = exec.execute(0, Arc::clone(&ctx)).unwrap();
        let stream1 = exec.execute(1, Arc::clone(&ctx)).unwrap();

        let expected = [
            "+------+------+",
            "| col1 | col4 |",
            "+------+------+",
            "| 2    | 0.1  |",
            "| 4    | 0.1  |",
            "+------+------+",
        ];
        let batches0 = stream0.try_collect::<Vec<_>>().await.unwrap();
        let batches1 = stream1.try_collect::<Vec<_>>().await.unwrap();
        assert_batches_sorted_eq!(expected, &batches0);
        assert_batches_sorted_eq!(expected, &batches1);
    }

    #[tokio::test]
    async fn test_left_join_with_filter() {
        // projection: col2 (Left index 2 → combined 2), col3 (Right index 1 → combined 4)
        // col4 > 0 removes P0→B0 (col4=0.0) and P4→B2 (col4=0.0).
        // B0 and B2 are not marked visited → appear as unmatched (null col3).
        let exec = make_exec(
            JoinType::Left,
            Some(vec![2, 4]),
            Some(col4_gt_zero_filter()),
        );
        let ctx = Arc::new(TaskContext::default());

        let stream0 = exec.execute(0, Arc::clone(&ctx)).unwrap();
        let stream1 = exec.execute(1, Arc::clone(&ctx)).unwrap();

        // Partition 0: matched rows only.
        let batches0 = stream0.try_collect::<Vec<_>>().await.unwrap();
        assert_batches_sorted_eq!(
            [
                "+------+------+",
                "| col2 | col3 |",
                "+------+------+",
                "| 2.2  | 101  |",
                "| 4.4  | 105  |",
                "+------+------+",
            ],
            &batches0
        );

        // Partition 1: matched rows + all unmatched build rows (B0, B2, B4, B5, null-geo, empty-geo).
        let batches1 = stream1.try_collect::<Vec<_>>().await.unwrap();
        assert_batches_sorted_eq!(
            [
                "+------+------+",
                "| col2 | col3 |",
                "+------+------+",
                "| 1.1  |      |",
                "| 2.2  | 101  |",
                "| 3.3  |      |",
                "| 4.4  | 105  |",
                "| 5.5  |      |",
                "| 6.6  |      |",
                "| 7.7  |      |",
                "| 8.8  |      |",
                "+------+------+",
            ],
            &batches1
        );
    }
}
