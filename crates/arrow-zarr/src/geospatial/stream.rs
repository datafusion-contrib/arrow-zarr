use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::compute::take;
use arrow_array::cast::AsArray;
use arrow_array::{new_null_array, ArrayRef, RecordBatch, UInt32Array};
use arrow_schema::Schema;
use async_stream::try_stream;
use datafusion::common::{JoinSide, JoinType};
use datafusion::error::{DataFusionError, Result};
use datafusion::physical_plan::joins::utils::{ColumnIndex, JoinFilter};
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::future::{BoxFuture, Shared};
use futures::stream::{BoxStream, Stream};
use futures::TryStreamExt;

use super::build_side::BuildSide;
use super::probe_side::ProbeBatch;
use super::spatial_predicate::RelationPredicate;

// the main stream impmlementation that handles joining record batches.
struct InnerSpatialJoinStream {
    schema: Arc<Schema>,
    join_type: JoinType,
    filter: Option<JoinFilter>,
    probe_stream: SendableRecordBatchStream,
    column_indices: Vec<ColumnIndex>,
    build_side_fut: Shared<BoxFuture<'static, Result<Arc<BuildSide>, Arc<DataFusionError>>>>,
    predicate: RelationPredicate,
    // Populated on first poll by awaiting build_side_fut.
    build_side: Option<Arc<BuildSide>>,
}

impl InnerSpatialJoinStream {
    pub(crate) fn new(
        schema: Arc<Schema>,
        join_type: JoinType,
        filter: Option<JoinFilter>,
        probe_stream: SendableRecordBatchStream,
        column_indices: Vec<ColumnIndex>,
        build_side_fut: Shared<BoxFuture<'static, Result<Arc<BuildSide>, Arc<DataFusionError>>>>,
        predicate: RelationPredicate,
    ) -> Self {
        Self {
            schema,
            join_type,
            filter,
            probe_stream,
            column_indices,
            build_side_fut,
            predicate,
            build_side: None,
        }
    }

    pub(crate) async fn next_batch(&mut self) -> Result<Option<RecordBatch>, DataFusionError> {
        if self.build_side.is_none() {
            self.build_side = Some(
                self.build_side_fut
                    .clone()
                    .await
                    .map_err(|e| DataFusionError::Internal(e.to_string()))?,
            );
        }

        let next_probe_batch = self.probe_stream.try_next().await?;
        if let Some(probe_batch) = next_probe_batch {
            let mut probe_batch = ProbeBatch::new(probe_batch, &self.predicate.right)?;
            let (probe_indices, build_side_indices) = self.traverse(&mut probe_batch)?;
            let rec_batch = self.join_matched_rows(
                &build_side_indices,
                &probe_indices,
                &probe_batch.rec_batch,
                JoinSide::Left,
            )?;
            Ok(Some(rec_batch))
        } else {
            let prob_num = self
                .build_side
                .as_ref()
                .expect("Build side unexpectedly not initialized")
                .decrement_remaining_probes();
            if self.join_type == JoinType::Left && prob_num == 1 {
                return self.build_unmatched_batch(JoinSide::Left);
            }
            Ok(None)
        }
    }

    fn build_batch(
        &self,
        build_positions: &[(u32, u32)],
        probe_batch: &RecordBatch,
        probe_indices: &[u32],
        build_side: JoinSide,
        column_indices: &[ColumnIndex],
        schema: &Arc<Schema>,
    ) -> Result<RecordBatch> {
        let build = self.build_side.as_ref().ok_or_else(|| {
            datafusion::error::DataFusionError::Internal("build_side not initialized".to_string())
        })?;
        let mut columns = Vec::with_capacity(column_indices.len());

        for col in column_indices {
            let array: ArrayRef = if col.side == build_side {
                build.interleave_column(build_positions, col.index)?
            } else {
                take(
                    probe_batch.column(col.index).as_ref(),
                    &UInt32Array::from(probe_indices.to_vec()),
                    None,
                )?
            };
            columns.push(array);
        }

        Ok(RecordBatch::try_new(Arc::clone(schema), columns)?)
    }

    fn join_matched_rows(
        &self,
        build_geo_ids: &[u32],
        probe_indices: &[u32],
        probe_batch: &RecordBatch,
        build_side: JoinSide,
    ) -> Result<RecordBatch> {
        let build = self.build_side.as_ref().ok_or_else(|| {
            datafusion::error::DataFusionError::Internal("build_side not initialized".to_string())
        })?;

        // 1. Resolve geo IDs → (batch_idx, row_idx)
        let mut positions = build.resolve_positions(build_geo_ids);
        let mut probe_indices = probe_indices.to_vec();

        // 2. Apply join filter if present
        if let Some(filter) = &self.filter {
            let filter_batch = self.build_batch(
                &positions,
                probe_batch,
                &probe_indices,
                build_side,
                filter.column_indices(),
                filter.schema(),
            )?;

            let mask = filter
                .expression()
                .evaluate(&filter_batch)?
                .into_array(filter_batch.num_rows())?;
            let mask = mask.as_boolean();

            positions = positions
                .iter()
                .zip(mask.iter())
                .filter_map(|(&pos, keep)| keep.unwrap_or(false).then_some(pos))
                .collect();
            probe_indices = probe_indices
                .iter()
                .zip(mask.iter())
                .filter_map(|(&idx, keep)| keep.unwrap_or(false).then_some(idx))
                .collect();
        }

        // 3. Mark matched build-side rows as visited
        build.mark_visited(&positions);

        // 4. Build and return the final joined batch
        self.build_batch(
            &positions,
            probe_batch,
            &probe_indices,
            build_side,
            &self.column_indices,
            &self.schema,
        )
    }

    fn build_unmatched_batch(&self, build_side: JoinSide) -> Result<Option<RecordBatch>> {
        let build = self
            .build_side
            .as_ref()
            .ok_or_else(|| DataFusionError::Internal("build_side not initialized".to_string()))?;

        let positions = build.unvisited_positions()?;

        if positions.is_empty() {
            return Ok(None);
        }

        let n = positions.len();
        let mut columns = Vec::with_capacity(self.column_indices.len());

        for (out_idx, col) in self.column_indices.iter().enumerate() {
            let array: ArrayRef = if col.side == build_side {
                build.interleave_column(&positions, col.index)?
            } else {
                new_null_array(self.schema.field(out_idx).data_type(), n)
            };
            columns.push(array);
        }

        Ok(Some(RecordBatch::try_new(
            Arc::clone(&self.schema),
            columns,
        )?))
    }

    fn traverse(&self, probe: &mut ProbeBatch) -> Result<(Vec<u32>, Vec<u32>)> {
        let build = self
            .build_side
            .as_ref()
            .ok_or_else(|| DataFusionError::Internal("build_side not initialized".to_string()))?;
        let tree = &build.tree;

        let mut probe_row_ids: Vec<u32> = Vec::new();
        let mut build_geo_ids: Vec<u32> = Vec::new();

        if tree.nodes.is_empty() || probe.boundary == 0 {
            return Ok((probe_row_ids, build_geo_ids));
        }

        // Each entry: (node_idx, boundary to restore before visiting this node).
        // Siblings share the same saved boundary (their parent's post-partition boundary).
        let mut stack: Vec<(usize, usize)> = vec![(0, probe.boundary)];

        while let Some((idx, saved_boundary)) = stack.pop() {
            probe.set_boundary(saved_boundary);
            let node = tree.node(idx);
            let new_boundary = probe.partition(node.min_x, node.min_y, node.max_x, node.max_y);
            if new_boundary == 0 {
                continue;
            }
            if let Some(build_exploded) = build.get_leaf_exploded(idx)? {
                let probe_exploded = probe.get_current_valid_exploded()?;
                let (p_ids, b_ids) =
                    probe_exploded.join(build_exploded, self.predicate.relation_type.clone());
                probe_row_ids.extend(p_ids);
                build_geo_ids.extend(b_ids);
            } else {
                let mut n = 0;
                while let Some(child_idx) = tree.child_idx(idx, n) {
                    stack.push((child_idx, new_boundary));
                    n += 1;
                }
            }
        }

        Ok((probe_row_ids, build_geo_ids))
    }
}

pub(crate) struct SpatialJoinStream {
    schema: Arc<Schema>,
    stream: BoxStream<'static, Result<RecordBatch, DataFusionError>>,
}

impl SpatialJoinStream {
    pub(crate) fn new(
        schema: Arc<Schema>,
        join_type: JoinType,
        filter: Option<JoinFilter>,
        probe_stream: SendableRecordBatchStream,
        column_indices: Vec<ColumnIndex>,
        build_side_fut: Shared<BoxFuture<'static, Result<Arc<BuildSide>, Arc<DataFusionError>>>>,
        predicate: RelationPredicate,
    ) -> Self {
        let mut stream = InnerSpatialJoinStream::new(
            schema.clone(),
            join_type,
            filter,
            probe_stream,
            column_indices,
            build_side_fut,
            predicate,
        );

        let stream = Box::pin(try_stream! {
            while let Some(batch) = stream.next_batch().await? {
                yield batch;
            }
        });
        Self { schema, stream }
    }
}

impl Stream for SpatialJoinStream {
    type Item = Result<RecordBatch, DataFusionError>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().stream.as_mut().poll_next(cx)
    }
}

impl datafusion::physical_plan::RecordBatchStream for SpatialJoinStream {
    fn schema(&self) -> arrow_schema::SchemaRef {
        Arc::clone(&self.schema)
    }
}

#[cfg(test)]
mod stream_tests {
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll};

    use arrow_array::RecordBatch;
    use arrow_schema::SchemaRef;
    use datafusion::assert_batches_sorted_eq;
    use datafusion::common::{JoinSide, JoinType};
    use datafusion::error::{DataFusionError, Result};
    use datafusion::physical_expr::expressions::Column;
    use datafusion::physical_plan::joins::utils::{ColumnIndex, JoinFilter};
    use datafusion::physical_plan::{PhysicalExpr, RecordBatchStream, SendableRecordBatchStream};
    use futures::future::{BoxFuture, FutureExt, Shared};
    use futures::{Stream, TryStreamExt};

    use super::SpatialJoinStream;
    use crate::geospatial::build_side::{BuildSide, BuildSideBuilder, ProcessedBatch};
    use crate::geospatial::spatial_predicate::{RelationPredicate, SpatialRelationType};
    use crate::geospatial::test_utils::{
        build_schema, col4_gt_zero_filter, make_build_batch, make_probe_batches, probe_schema,
    };

    struct MockStream {
        schema: SchemaRef,
        batches: std::vec::IntoIter<RecordBatch>,
    }

    impl Stream for MockStream {
        type Item = datafusion::error::Result<RecordBatch>;
        fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Ready(self.get_mut().batches.next().map(Ok))
        }
    }

    impl RecordBatchStream for MockStream {
        fn schema(&self) -> SchemaRef {
            self.schema.clone()
        }
    }

    fn make_build_side_fut(
        needs_visited: bool,
    ) -> Shared<BoxFuture<'static, Result<Arc<BuildSide>, Arc<DataFusionError>>>> {
        let pb = ProcessedBatch::new(
            make_build_batch(),
            &(Arc::new(Column::new("geo", 0)) as Arc<dyn PhysicalExpr>),
        )
        .unwrap();

        let mut builder = BuildSideBuilder::new();
        builder.add_partition(vec![pb]);
        let build_side = Arc::new(builder.build(needs_visited, 1));
        futures::future::ready(Ok(build_side)).boxed().shared()
    }

    /// Creates a SpatialJoinStream for one probe partition.
    pub(crate) fn make_test_stream(
        join_type: JoinType,
        column_indices: Vec<ColumnIndex>,
        filter: Option<JoinFilter>,
        probe_batches: Vec<RecordBatch>,
        build_side_fut: Shared<BoxFuture<'static, Result<Arc<BuildSide>, Arc<DataFusionError>>>>,
    ) -> SpatialJoinStream {
        use arrow_schema::{Field, Schema};

        let probe_stream: SendableRecordBatchStream = Box::pin(MockStream {
            schema: probe_schema(),
            batches: probe_batches.into_iter(),
        });

        let predicate = RelationPredicate::new(
            Arc::new(Column::new("geo", 0)) as Arc<dyn PhysicalExpr>,
            Arc::new(Column::new("geo", 0)) as Arc<dyn PhysicalExpr>,
            SpatialRelationType::Within,
        );

        let bs = build_schema();
        let ps = probe_schema();
        let output_schema = Arc::new(Schema::new(
            column_indices
                .iter()
                .map(|ci| {
                    let schema = if ci.side == JoinSide::Left { &bs } else { &ps };
                    let f = schema.field(ci.index);
                    Field::new(f.name(), f.data_type().clone(), ci.side == JoinSide::Right)
                })
                .collect::<Vec<_>>(),
        ));

        SpatialJoinStream::new(
            output_schema,
            join_type,
            filter,
            probe_stream,
            column_indices,
            build_side_fut,
            predicate,
        )
    }

    #[tokio::test]
    async fn test_inner_join() {
        let build_fut = make_build_side_fut(false);

        let stream = make_test_stream(
            JoinType::Inner,
            vec![
                ColumnIndex {
                    index: 1,
                    side: JoinSide::Left,
                },
                ColumnIndex {
                    index: 2,
                    side: JoinSide::Right,
                },
            ],
            None,
            make_probe_batches(),
            build_fut,
        );

        // col1 from build (Int32), col4 from probe (Float64)
        // Matched: P0→B0, P1→B1, P4→B2, P5→B3
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        assert_batches_sorted_eq!(
            [
                "+------+------+",
                "| col1 | col4 |",
                "+------+------+",
                "| 1    | 0.0  |",
                "| 2    | 0.1  |",
                "| 3    | 0.0  |",
                "| 4    | 0.1  |",
                "+------+------+",
            ],
            &batches
        );
    }

    #[tokio::test]
    async fn test_left_join() {
        let build_fut = make_build_side_fut(true);

        let stream = make_test_stream(
            JoinType::Left,
            vec![
                ColumnIndex {
                    index: 2,
                    side: JoinSide::Left,
                },
                ColumnIndex {
                    index: 1,
                    side: JoinSide::Right,
                },
            ],
            None,
            make_probe_batches(),
            build_fut,
        );

        // col2 from build (Float64), col3 from probe (Int32, nullable)
        // Matched: P0→B0, P1→B1, P4→B2, P5→B3
        // Unmatched build: B4, B5, null-geo row, empty-geo row → null col3
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
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
            &batches
        );
    }

    #[tokio::test]
    async fn test_inner_join_with_filter() {
        let build_fut = make_build_side_fut(false);

        let stream = make_test_stream(
            JoinType::Inner,
            vec![
                ColumnIndex {
                    index: 1,
                    side: JoinSide::Left,
                },
                ColumnIndex {
                    index: 2,
                    side: JoinSide::Right,
                },
            ],
            Some(col4_gt_zero_filter()),
            make_probe_batches(),
            build_fut,
        );

        // col4 > 0 removes P0→B0 (col4=0.0) and P4→B2 (col4=0.0)
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        assert_batches_sorted_eq!(
            [
                "+------+------+",
                "| col1 | col4 |",
                "+------+------+",
                "| 2    | 0.1  |",
                "| 4    | 0.1  |",
                "+------+------+",
            ],
            &batches
        );
    }

    #[tokio::test]
    async fn test_left_join_with_filter() {
        let build_fut = make_build_side_fut(true);

        let stream = make_test_stream(
            JoinType::Left,
            vec![
                ColumnIndex {
                    index: 2,
                    side: JoinSide::Left,
                },
                ColumnIndex {
                    index: 1,
                    side: JoinSide::Right,
                },
            ],
            Some(col4_gt_zero_filter()),
            make_probe_batches(),
            build_fut,
        );

        // col4 > 0 removes P0→B0 and P4→B2 from matched set.
        // B0 and B2 are not marked visited → appear as unmatched (null col3).
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
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
            &batches
        );
    }
}
