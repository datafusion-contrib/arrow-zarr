use std::collections::HashSet;
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
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::joins::utils::{ColumnIndex, JoinFilter};
use futures::future::{BoxFuture, Shared};
use futures::stream::BoxStream;
use futures::{Stream, TryStreamExt};
use geo_types::Rect;

use super::boxed_geo_batch::{BBoxedGeoBatch, BBoxedGeoStream};
use super::indexed_build_side::IndexedBuildSide;
use super::spatial_predicate::RelationPredicate;

const BUILD_SIDE_JOIN_SIDE: JoinSide = JoinSide::Left;
/// Probe rows per traverse/refine call (bounds the parsed-geometry memory peak).
const PROBE_CHUNK_SIZE: usize = 512;
/// Output rows per emitted batch (bounds the output-assembly memory peak).
const BATCH_SIZE: usize = 8192;

pub(crate) type SharedIndexFuture =
    Shared<BoxFuture<'static, Result<Arc<IndexedBuildSide>, Arc<DataFusionError>>>>;

/// A `(build, probe)` output row. `None` on a side means null that side's columns:
/// matched = `(Some, Some)`, unmatched probe = `(None, Some)`, unmatched build = `(Some, None)`.
type Pair = (Option<usize>, Option<usize>);

enum SpatialJoinState {
    FetchingProbeBatch,
    ProcessingProbeBatch {
        /// Remaining probe rows; drained one chunk at a time via `pop_chunk`.
        geo_batch: BBoxedGeoBatch,
        acc: Vec<Pair>,
    },
    OutputtingBatch {
        /// `Some` = probe processing (resume + assemble against its batch);
        /// `None` = unmatched-build output (terminal, probe columns null).
        geo_batch: Option<BBoxedGeoBatch>,
        acc: Vec<Pair>,
        /// Source fully drained — flush the remaining accumulator.
        last_batch: bool,
    },
    Done,
}

struct InnerSpatialJoinStream {
    schema: Arc<Schema>,
    join_type: JoinType,
    filter: Option<JoinFilter>,
    probe_stream: BBoxedGeoStream,
    column_indices: Vec<ColumnIndex>,
    build_side_fut: SharedIndexFuture,
    predicate: RelationPredicate,
    /// Populated on first poll by awaiting `build_side_fut`.
    build_side: Option<Arc<IndexedBuildSide>>,
    state: SpatialJoinState,
}

impl InnerSpatialJoinStream {
    pub(crate) fn new(
        schema: Arc<Schema>,
        join_type: JoinType,
        filter: Option<JoinFilter>,
        probe_stream: SendableRecordBatchStream,
        column_indices: Vec<ColumnIndex>,
        build_side_fut: SharedIndexFuture,
        predicate: RelationPredicate,
    ) -> Self {
        let probe_stream = BBoxedGeoStream::new(probe_stream, predicate.right.clone());
        Self {
            schema,
            join_type,
            filter,
            probe_stream,
            column_indices,
            build_side_fut,
            predicate,
            build_side: None,
            state: SpatialJoinState::FetchingProbeBatch,
        }
    }

    fn build_side(&self) -> Result<&Arc<IndexedBuildSide>> {
        self.build_side
            .as_ref()
            .ok_or_else(|| DataFusionError::Internal("build_side not initialized".to_string()))
    }

    /// Traverses + refines one chunk of probe rows and returns the output pairs:
    /// surviving matches as `(Some(build), Some(probe))`, plus (for Right/Full) every probe
    /// row in the chunk's `[min, max]` range with no surviving match as `(None, Some(probe))`.
    fn join(
        &self,
        probe_rects: Vec<Rect<f32>>,
        probe_ids: Vec<usize>,
        probe_geo_array: &ArrayRef,
        probe_batch: &RecordBatch,
    ) -> Result<Vec<Pair>> {
        // Record the chunk's contiguous probe-row range before anything consumes the ids.
        let range = match (probe_ids.iter().min(), probe_ids.iter().max()) {
            (Some(&lo), Some(&hi)) => Some((lo, hi)),
            _ => None,
        };

        let build = self.build_side()?;
        let (probe_matched, build_matched) = build.traverse_with_refinement(
            probe_rects,
            probe_ids,
            probe_geo_array,
            &self.predicate.relation_type,
            BUILD_SIDE_JOIN_SIDE,
        );

        // Apply the optional non-spatial filter to the matched pairs.
        let (probe_kept, build_kept) = if let Some(filter) = &self.filter {
            let matched: Vec<Pair> = build_matched
                .iter()
                .zip(probe_matched.iter())
                .map(|(&b, &p)| (Some(b), Some(p)))
                .collect();
            let filter_batch = self.assemble(
                Some(probe_batch),
                &matched,
                filter.column_indices(),
                filter.schema(),
            )?;
            let mask = filter
                .expression()
                .evaluate(&filter_batch)?
                .into_array(filter_batch.num_rows())?;
            let mask = mask.as_boolean();

            let mut probe_kept = Vec::new();
            let mut build_kept = Vec::new();
            for ((&p, &b), keep) in probe_matched
                .iter()
                .zip(build_matched.iter())
                .zip(mask.iter())
            {
                if keep.unwrap_or(false) {
                    probe_kept.push(p);
                    build_kept.push(b);
                }
            }
            (probe_kept, build_kept)
        } else {
            (probe_matched, build_matched)
        };

        build.mark_visited(&build_kept);

        let mut pairs: Vec<Pair> = build_kept
            .iter()
            .zip(probe_kept.iter())
            .map(|(&b, &p)| (Some(b), Some(p)))
            .collect();

        // Right/Full: pad probe rows in the chunk's range that have no surviving match.
        if matches!(self.join_type, JoinType::Right | JoinType::Full) {
            if let Some((lo, hi)) = range {
                let matched: HashSet<usize> = probe_kept.iter().copied().collect();
                for id in lo..=hi {
                    if !matched.contains(&id) {
                        pairs.push((None, Some(id)));
                    }
                }
            }
        }

        Ok(pairs)
    }

    /// All unmatched build rows (Left/Full) as `(Some(build), None)` pairs.
    fn unmatched_build_side(&self) -> Vec<Pair> {
        self.build_side
            .as_ref()
            .and_then(|b| b.unvisited_positions())
            .unwrap_or_default()
            .into_iter()
            .map(|pos| (Some(pos), None))
            .collect()
    }

    /// Assembles an output `RecordBatch` from `(build, probe)` pairs against the given
    /// column layout. `None` on a side yields null columns for that row. `probe_batch` may
    /// be `None` only when every probe entry is `None` (unmatched-build output).
    fn assemble(
        &self,
        probe_batch: Option<&RecordBatch>,
        pairs: &[Pair],
        column_indices: &[ColumnIndex],
        schema: &Arc<Schema>,
    ) -> Result<RecordBatch> {
        let build = self.build_side()?;

        let build_ids: Vec<Option<usize>> = pairs.iter().map(|&(b, _)| b).collect();
        let probe_idx = UInt32Array::from(
            pairs
                .iter()
                .map(|&(_, p)| p.map(|p| p as u32))
                .collect::<Vec<_>>(),
        );

        let mut columns = Vec::with_capacity(column_indices.len());
        for (out_idx, col) in column_indices.iter().enumerate() {
            let array = if col.side == BUILD_SIDE_JOIN_SIDE {
                build.interleave_column_opt(&build_ids, col.index)?
            } else if let Some(probe_batch) = probe_batch {
                take(probe_batch.column(col.index).as_ref(), &probe_idx, None)?
            } else {
                new_null_array(schema.field(out_idx).data_type(), build_ids.len())
            };
            columns.push(array);
        }

        Ok(RecordBatch::try_new(Arc::clone(schema), columns)?)
    }

    /// Assembles a final output batch using the stream's projected schema/columns.
    fn build_batch(
        &self,
        probe_batch: Option<&RecordBatch>,
        pairs: &[Pair],
    ) -> Result<RecordBatch> {
        self.assemble(probe_batch, pairs, &self.column_indices, &self.schema)
    }

    pub(crate) async fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        // Init build side once.
        if self.build_side.is_none() {
            self.build_side = Some(
                self.build_side_fut
                    .clone()
                    .await
                    .map_err(|e| DataFusionError::Internal(e.to_string()))?,
            );
        }

        loop {
            // Take the state out (replaced with Done as a safe sentinel).
            let state = std::mem::replace(&mut self.state, SpatialJoinState::Done);

            match state {
                SpatialJoinState::FetchingProbeBatch => match self.probe_stream.try_next().await? {
                    Some(geo_batch) => {
                        self.state = SpatialJoinState::ProcessingProbeBatch {
                            geo_batch,
                            acc: Vec::new(),
                        };
                    }
                    None => {
                        let build = self.build_side.as_ref().unwrap();
                        let prev = build.decrement_remaining_probes();
                        let acc = if matches!(self.join_type, JoinType::Left | JoinType::Full)
                            && prev == 1
                        {
                            self.unmatched_build_side()
                        } else {
                            Vec::new()
                        };
                        if acc.is_empty() {
                            self.state = SpatialJoinState::Done;
                            return Ok(None);
                        }
                        // Unmatched-build output: no probe batch, terminal once drained.
                        self.state = SpatialJoinState::OutputtingBatch {
                            geo_batch: None,
                            acc,
                            last_batch: true,
                        };
                    }
                },

                SpatialJoinState::ProcessingProbeBatch {
                    mut geo_batch,
                    mut acc,
                } => {
                    let (rects, ids) = geo_batch.pop_chunk(PROBE_CHUNK_SIZE);
                    let pairs = self.join(rects, ids, &geo_batch.geo_array, &geo_batch.batch)?;
                    acc.extend(pairs);

                    let last_batch = geo_batch.is_empty();
                    self.state = SpatialJoinState::OutputtingBatch {
                        geo_batch: Some(geo_batch),
                        acc,
                        last_batch,
                    };
                }

                SpatialJoinState::OutputtingBatch {
                    geo_batch,
                    mut acc,
                    last_batch,
                } => {
                    let probe_batch = geo_batch.as_ref().map(|g| &g.batch);

                    if acc.len() >= BATCH_SIZE {
                        let chunk = acc.split_off(acc.len() - BATCH_SIZE);
                        let out_batch = self.build_batch(probe_batch, &chunk)?;
                        self.state = SpatialJoinState::OutputtingBatch {
                            geo_batch,
                            acc,
                            last_batch,
                        };
                        return Ok(Some(out_batch));
                    } else if last_batch {
                        // Flush the remainder (if any), then move on. A carried probe batch
                        // means more probe batches may follow; its absence is the terminal
                        // unmatched-build output.
                        let out_batch = if acc.is_empty() {
                            None
                        } else {
                            Some(self.build_batch(probe_batch, &acc)?)
                        };
                        self.state = if geo_batch.is_some() {
                            SpatialJoinState::FetchingProbeBatch
                        } else {
                            SpatialJoinState::Done
                        };
                        match out_batch {
                            Some(batch) => return Ok(Some(batch)),
                            None => continue,
                        }
                    } else {
                        // Not the last chunk: resume processing (geo_batch is always Some here).
                        let geo_batch = geo_batch.expect("probe processing has a geo_batch");
                        self.state = SpatialJoinState::ProcessingProbeBatch { geo_batch, acc };
                    }
                }

                SpatialJoinState::Done => {
                    self.state = SpatialJoinState::Done;
                    return Ok(None);
                }
            }
        }
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
        build_side_fut: SharedIndexFuture,
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
mod spatial_join_stream_tests {
    use std::sync::Arc;

    use arrow_array::RecordBatch;
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::assert_batches_sorted_eq;
    use datafusion::common::{JoinSide, JoinType};
    use datafusion::error::DataFusionError;
    use datafusion::physical_expr::expressions::Column;
    use datafusion::physical_plan::joins::utils::{ColumnIndex, JoinFilter};
    use datafusion::physical_plan::PhysicalExpr;
    use futures::{FutureExt, TryStreamExt};

    use super::SpatialJoinStream;
    use crate::geospatial::indexed_build_side::test_helpers::make_indexed_build_side;
    use crate::geospatial::spatial_predicate::{RelationPredicate, SpatialRelationType};
    use crate::geospatial::test_utils::{col1_gte_col2_filter, make_geo_stream};

    // Build side: 4 unit squares across 2 partitions.
    //   build[0] = (1,1)-(2,2)     → inside probe[1]
    //   build[1] = (20,20)-(21,21) → no match
    //   build[2] = (6,6)-(7,7)     → inside probe[2] and probe[3]
    //   build[3] = (30,30)-(31,31) → no match
    const BUILD_WKTS: &[&str] = &[
        "POLYGON((1 1, 2 1, 2 2, 1 2, 1 1))",
        "POLYGON((20 20, 21 20, 21 21, 20 21, 20 20))",
        "POLYGON((6 6, 7 6, 7 7, 6 7, 6 6))",
        "POLYGON((30 30, 31 30, 31 31, 30 31, 30 30))",
    ];

    // Probe side: 4 larger squares across 2 batches.
    //   probe[0] = (50,50)-(60,60) → no match
    //   probe[1] = (0,0)-(4,4)     → contains build[0]
    //   probe[2] = (5,5)-(9,9)     → contains build[2]
    //   probe[3] = (4,4)-(10,10)   → contains build[2]
    const PROBE_WKTS: &[&str] = &[
        "POLYGON((50 50, 60 50, 60 60, 50 60, 50 50))",
        "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))",
        "POLYGON((5 5, 9 5, 9 9, 5 9, 5 5))",
        "POLYGON((4 4, 10 4, 10 10, 4 10, 4 4))",
    ];

    pub(crate) async fn run_join(
        join_type: JoinType,
        filter: Option<JoinFilter>,
    ) -> Vec<RecordBatch> {
        let needs_visited = matches!(join_type, JoinType::Left | JoinType::Full);
        let build_side =
            make_indexed_build_side(BUILD_WKTS, 2, 1, needs_visited, "geometry_1", "col_1");

        let build_side_fut = futures::future::ready(Ok::<_, Arc<DataFusionError>>(build_side))
            .boxed()
            .shared();

        let probe_stream = make_geo_stream(PROBE_WKTS, 2, "geometry_2", "col_2");

        let schema = Arc::new(Schema::new(vec![
            Field::new("col_1", DataType::Int32, true),
            Field::new("col_2", DataType::Int32, true),
        ]));

        // col_1 from build (left, index 1 in build batch); col_2 from probe (right, index 1)
        let column_indices = vec![
            ColumnIndex {
                index: 1,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 1,
                side: JoinSide::Right,
            },
        ];

        let predicate = RelationPredicate::new(
            Arc::new(Column::new("geometry_1", 0)) as Arc<dyn PhysicalExpr>,
            Arc::new(Column::new("geometry_2", 0)) as Arc<dyn PhysicalExpr>,
            SpatialRelationType::Within,
        );

        let stream = SpatialJoinStream::new(
            schema,
            join_type,
            filter,
            probe_stream,
            column_indices,
            build_side_fut,
            predicate,
        );

        stream.try_collect::<Vec<_>>().await.unwrap()
    }

    #[tokio::test]
    async fn test_join_types() {
        let inner = run_join(JoinType::Inner, None).await;
        assert_batches_sorted_eq!(
            [
                "+-------+-------+",
                "| col_1 | col_2 |",
                "+-------+-------+",
                "| 0     | 1     |",
                "| 2     | 2     |",
                "| 2     | 3     |",
                "+-------+-------+",
            ],
            &inner
        );

        let left = run_join(JoinType::Left, None).await;
        assert_batches_sorted_eq!(
            [
                "+-------+-------+",
                "| col_1 | col_2 |",
                "+-------+-------+",
                "| 0     | 1     |",
                "| 1     |       |",
                "| 2     | 2     |",
                "| 2     | 3     |",
                "| 3     |       |",
                "+-------+-------+",
            ],
            &left
        );

        let right = run_join(JoinType::Right, None).await;
        assert_batches_sorted_eq!(
            [
                "+-------+-------+",
                "| col_1 | col_2 |",
                "+-------+-------+",
                "|       | 0     |",
                "| 0     | 1     |",
                "| 2     | 2     |",
                "| 2     | 3     |",
                "+-------+-------+",
            ],
            &right
        );
    }

    #[tokio::test]
    async fn test_join_types_with_filter() {
        let inner = run_join(JoinType::Inner, Some(col1_gte_col2_filter())).await;
        assert_batches_sorted_eq!(
            [
                "+-------+-------+",
                "| col_1 | col_2 |",
                "+-------+-------+",
                "| 2     | 2     |",
                "+-------+-------+",
            ],
            &inner
        );

        let left = run_join(JoinType::Left, Some(col1_gte_col2_filter())).await;
        assert_batches_sorted_eq!(
            [
                "+-------+-------+",
                "| col_1 | col_2 |",
                "+-------+-------+",
                "| 0     |       |",
                "| 1     |       |",
                "| 2     | 2     |",
                "| 3     |       |",
                "+-------+-------+",
            ],
            &left
        );

        let right = run_join(JoinType::Right, Some(col1_gte_col2_filter())).await;
        assert_batches_sorted_eq!(
            [
                "+-------+-------+",
                "| col_1 | col_2 |",
                "+-------+-------+",
                "|       | 0     |",
                "|       | 1     |",
                "| 2     | 2     |",
                "|       | 3     |",
                "+-------+-------+",
            ],
            &right
        );
    }
}
