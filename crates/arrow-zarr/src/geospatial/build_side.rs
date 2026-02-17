use std::collections::VecDeque;
use std::ops::Range;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};

use arrow_array::builder::BooleanBufferBuilder;
use arrow_array::{Array, ArrayRef, RecordBatch};
use datafusion::error::Result;
use datafusion::physical_plan::PhysicalExpr;
use rstar::{RTree, RTreeNode, RTreeObject, AABB};

use super::extension_traits::{ArrayWkbOps, WkbVecOps};
use super::operations::ExplodedSide;

struct GeoItem {
    geo_id: u32,
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
}

impl RTreeObject for GeoItem {
    type Envelope = AABB<[f32; 2]>;
    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners([self.min_x, self.min_y], [self.max_x, self.max_y])
    }
}

// An r-tree to support vectorized traversal, with a flat layout
// suited for batch processing.
pub(crate) enum NodeContent {
    Internal(Range<usize>),
    Leaf(Range<usize>),
}

pub(crate) struct TreeNode {
    pub(crate) min_x: f32,
    pub(crate) min_y: f32,
    pub(crate) max_x: f32,
    pub(crate) max_y: f32,
    pub(crate) content: NodeContent,
}

pub(crate) struct CustomTree {
    pub(crate) nodes: Vec<TreeNode>,
    pub(crate) leaf_geo_ids: Vec<u32>,
}

impl CustomTree {
    pub(crate) fn node(&self, i: usize) -> &TreeNode {
        &self.nodes[i]
    }

    pub(crate) fn child_idx(&self, i: usize, n: usize) -> Option<usize> {
        if let NodeContent::Internal(range) = &self.nodes[i].content {
            let idx = range.start + n;
            if idx < range.end {
                return Some(idx);
            }
        }
        None
    }
}

// A processed build side for a spatial join.
pub(crate) struct BatchWithGeoArray {
    pub(crate) batch: RecordBatch,
    pub(crate) geometry_array: ArrayRef,
}

pub(crate) struct BuildSide {
    pub(crate) tree: CustomTree,
    batches: Vec<BatchWithGeoArray>,
    batch_positions: Vec<(u32, u32)>,

    // None for join types that don't need unmatched-row tracking (e.g. inner join).
    // One BooleanBufferBuilder per batch, all bits false initially.
    visited: Option<Mutex<Vec<BooleanBufferBuilder>>>,
    remaining_probes: AtomicUsize,
}

impl BuildSide {
    pub(crate) fn get_leaf_exploded(&self, node_idx: usize) -> Result<Option<ExplodedSide>> {
        let node = &self.tree.nodes[node_idx];
        match &node.content {
            NodeContent::Internal(_) => Ok(None),
            NodeContent::Leaf(range) => {
                let geo_ids = &self.tree.leaf_geo_ids[range.clone()];
                let wkbs: Vec<_> = geo_ids
                    .iter()
                    .map(|&geo_id| {
                        let (batch_idx, row_idx) = self.batch_positions[geo_id as usize];
                        self.batches[batch_idx as usize]
                            .geometry_array
                            .wkb_at(row_idx)
                    })
                    .collect::<Result<_>>()?;
                Ok(Some(ExplodedSide::new(geo_ids, &wkbs)?))
            }
        }
    }

    pub(crate) fn resolve_positions(&self, geo_ids: &[u32]) -> Vec<(u32, u32)> {
        geo_ids
            .iter()
            .map(|&geo_id| self.batch_positions[geo_id as usize])
            .collect()
    }

    pub(crate) fn interleave_column(
        &self,
        positions: &[(u32, u32)],
        col_idx: usize,
    ) -> Result<ArrayRef> {
        let arrays: Vec<&dyn Array> = self
            .batches
            .iter()
            .map(|b| b.batch.column(col_idx).as_ref())
            .collect();
        let indices: Vec<(usize, usize)> = positions
            .iter()
            .map(|&(batch_idx, row_idx)| (batch_idx as usize, row_idx as usize))
            .collect();
        Ok(arrow::compute::interleave(&arrays, &indices)?)
    }

    /// Marks each (batch_idx, row_idx) position as visited in the bitmap.
    /// No-op if `visited` is `None` (i.e. inner join).
    pub(crate) fn mark_visited(&self, positions: &[(u32, u32)]) {
        if let Some(mutex) = &self.visited {
            let mut bitmaps = mutex.lock().unwrap();
            for &(batch_idx, row_idx) in positions {
                bitmaps[batch_idx as usize].set_bit(row_idx as usize, true);
            }
        }
    }

    /// Returns (batch_idx, row_idx) pairs for all rows that were never visited.
    /// Errors if this join type does not track visited rows.
    pub(crate) fn unvisited_positions(&self) -> Result<Vec<(u32, u32)>> {
        let mutex = self.visited.as_ref().ok_or_else(|| {
            datafusion::error::DataFusionError::Internal(
                "unvisited_positions called on join type without visited tracking".to_string(),
            )
        })?;
        let bitmaps = mutex.lock().unwrap();
        let positions = bitmaps
            .iter()
            .enumerate()
            .flat_map(|(batch_idx, builder)| {
                (0..builder.len())
                    .filter(|&row_idx| !builder.get_bit(row_idx))
                    .map(move |row_idx| (batch_idx as u32, row_idx as u32))
            })
            .collect();
        Ok(positions)
    }

    /// Decrements the remaining probe count and returns the value before decrement.
    pub(crate) fn decrement_remaining_probes(&self) -> usize {
        self.remaining_probes
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst)
    }
}

// The builder that assembles the build side object,
// one partition at a time.
pub(crate) struct ProcessedBatch {
    pub(crate) batch: RecordBatch,
    pub(crate) geometry_array: ArrayRef,
    pub(crate) rects: Vec<Option<(f32, f32, f32, f32)>>,
    pub(crate) row_indices: Vec<u32>,
}

impl ProcessedBatch {
    pub(crate) fn new(batch: RecordBatch, geo_expr: &Arc<dyn PhysicalExpr>) -> Result<Self> {
        let n = batch.num_rows();
        let geometry_array = geo_expr.evaluate(&batch)?.into_array(n)?;

        let non_null_indices: Vec<u32> = (0..n as u32)
            .filter(|&i| !geometry_array.is_null(i as usize))
            .collect();
        let wkbs = geometry_array.as_wkbs(&non_null_indices)?;

        let rects_all = wkbs.as_slice().bounding_rects()?;

        let (row_indices, rects) = non_null_indices
            .into_iter()
            .zip(rects_all)
            .filter_map(|(idx, rect)| rect.map(|r| (idx, Some(r))))
            .unzip();

        Ok(Self {
            batch,
            geometry_array,
            rects,
            row_indices,
        })
    }
}

pub(crate) struct BuildSideBuilder {
    batches: Vec<BatchWithGeoArray>,
    batch_positions: Vec<(u32, u32)>,
    rects: Vec<Option<(f32, f32, f32, f32)>>,
}

impl BuildSideBuilder {
    pub(crate) fn new() -> Self {
        Self {
            batches: Vec::new(),
            batch_positions: Vec::new(),
            rects: Vec::new(),
        }
    }

    pub(crate) fn add_partition(&mut self, processed_batches: Vec<ProcessedBatch>) {
        for pb in processed_batches {
            self.add_batch(pb);
        }
    }

    fn add_batch(&mut self, pb: ProcessedBatch) {
        let batch_idx = self.batches.len() as u32;

        for &row_idx in &pb.row_indices {
            self.batch_positions.push((batch_idx, row_idx));
        }

        self.rects.extend(pb.rects);

        self.batches.push(BatchWithGeoArray {
            batch: pb.batch,
            geometry_array: pb.geometry_array,
        });
    }

    pub(crate) fn build(self, needs_visited: bool, num_probe_partitions: usize) -> BuildSide {
        let items: Vec<GeoItem> = self
            .rects
            .iter()
            .enumerate()
            .filter_map(|(geo_id, rect_opt)| {
                rect_opt.map(|(min_x, min_y, max_x, max_y)| GeoItem {
                    geo_id: geo_id as u32,
                    min_x,
                    min_y,
                    max_x,
                    max_y,
                })
            })
            .collect();

        let rtree = RTree::bulk_load(items);
        let tree = extract_tree(rtree);

        let visited = needs_visited.then(|| {
            let bitmaps = self
                .batches
                .iter()
                .map(|b| {
                    let mut builder = BooleanBufferBuilder::new(b.batch.num_rows());
                    builder.append_n(b.batch.num_rows(), false);
                    builder
                })
                .collect();
            Mutex::new(bitmaps)
        });

        BuildSide {
            tree,
            batches: self.batches,
            batch_positions: self.batch_positions,
            visited,
            remaining_probes: AtomicUsize::new(num_probe_partitions),
        }
    }
}

fn extract_tree(rtree: RTree<GeoItem>) -> CustomTree {
    use rstar::ParentNode;

    let mut nodes = Vec::new();
    let mut leaf_geo_ids = Vec::new();
    let mut n_reserved: usize = 1;

    let mut queue: VecDeque<&ParentNode<GeoItem>> = VecDeque::new();
    queue.push_back(rtree.root());

    while let Some(parent) = queue.pop_front() {
        let children = parent.children();
        let env = parent.envelope();
        let all_leaves = children.iter().all(|c| matches!(c, RTreeNode::Leaf(_)));

        if all_leaves {
            let start = leaf_geo_ids.len();
            for child in children {
                if let RTreeNode::Leaf(item) = child {
                    leaf_geo_ids.push(item.geo_id);
                }
            }
            nodes.push(TreeNode {
                min_x: env.lower()[0],
                min_y: env.lower()[1],
                max_x: env.upper()[0],
                max_y: env.upper()[1],
                content: NodeContent::Leaf(start..leaf_geo_ids.len()),
            });
        } else {
            nodes.push(TreeNode {
                min_x: env.lower()[0],
                min_y: env.lower()[1],
                max_x: env.upper()[0],
                max_y: env.upper()[1],
                content: NodeContent::Internal(n_reserved..n_reserved + children.len()),
            });
            n_reserved += children.len();
            for child in children {
                if let RTreeNode::Parent(p) = child {
                    queue.push_back(p);
                }
            }
        }
    }

    CustomTree {
        nodes,
        leaf_geo_ids,
    }
}

#[cfg(test)]
mod build_side_tests {
    use std::sync::Arc;

    use arrow_array::{BinaryArray, Int32Array};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::physical_expr::expressions::Column;
    use datafusion::physical_plan::PhysicalExpr;

    use super::*;
    use crate::geospatial::extension_traits::GeoComponentType;
    use crate::geospatial::test_utils::make_wkb_bufs;

    fn make_build_side() -> (BuildSide, Vec<(f32, f32, f32, f32)>) {
        let (bufs_0, rects_0) = make_wkb_bufs(0.0, 0.0);
        let (bufs_1, rects_1) = make_wkb_bufs(10.0, 20.0);
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("data", DataType::Int32, false),
        ]));

        let geo_expr = &(Arc::new(Column::new("geometry", 0)) as Arc<dyn PhysicalExpr>);

        // Batch 0: geos 0-2 (shift 0), data 1-3
        let pb_0 = ProcessedBatch::new(
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(BinaryArray::from_iter_values(
                        bufs_0[0..3].iter().map(|b| b.as_slice()),
                    )),
                    Arc::new(Int32Array::from(vec![1, 2, 3])),
                ],
            )
            .unwrap(),
            geo_expr,
        )
        .unwrap();

        // Batch 1: geos 3-5 (shift 0), data 4-6
        let pb_1 = ProcessedBatch::new(
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(BinaryArray::from_iter_values(
                        bufs_0[3..6].iter().map(|b| b.as_slice()),
                    )),
                    Arc::new(Int32Array::from(vec![4, 5, 6])),
                ],
            )
            .unwrap(),
            geo_expr,
        )
        .unwrap();

        // Batch 2: geos 6-8 (x-shift 10, y-shift 20), data 7-9
        let pb_2 = ProcessedBatch::new(
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(BinaryArray::from_iter_values(
                        bufs_1[0..3].iter().map(|b| b.as_slice()),
                    )),
                    Arc::new(Int32Array::from(vec![7, 8, 9])),
                ],
            )
            .unwrap(),
            geo_expr,
        )
        .unwrap();

        // Batch 3: geos 9-11 (x-shift 10, y-shift 20), data 10-12
        let pb_3 = ProcessedBatch::new(
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(BinaryArray::from_iter_values(
                        bufs_1[3..6].iter().map(|b| b.as_slice()),
                    )),
                    Arc::new(Int32Array::from(vec![10, 11, 12])),
                ],
            )
            .unwrap(),
            geo_expr,
        )
        .unwrap();

        let rects: Vec<(f32, f32, f32, f32)> =
            rects_0.iter().chain(rects_1.iter()).copied().collect();

        let mut builder = BuildSideBuilder::new();
        builder.add_partition(vec![pb_0, pb_1]);
        builder.add_partition(vec![pb_2, pb_3]);
        (builder.build(false, 2), rects)
    }

    #[test]
    fn test_build_side_custom_tree() {
        use std::collections::VecDeque;

        use rstar::{ParentNode, RTree, RTreeNode};

        let (build_side, rects) = make_build_side();

        let items: Vec<GeoItem> = rects
            .iter()
            .enumerate()
            .map(|(i, &(min_x, min_y, max_x, max_y))| GeoItem {
                geo_id: i as u32,
                min_x,
                min_y,
                max_x,
                max_y,
            })
            .collect();

        let rtree = RTree::bulk_load(items);
        let custom_tree = &build_side.tree;

        let mut queue: VecDeque<&ParentNode<GeoItem>> = VecDeque::new();
        queue.push_back(rtree.root());
        let mut node_idx = 0usize;

        while let Some(parent) = queue.pop_front() {
            let env = parent.envelope();
            let children = parent.children();
            let custom_node = &custom_tree.nodes[node_idx];

            assert_eq!(custom_node.min_x, env.lower()[0], "node {node_idx} min_x");
            assert_eq!(custom_node.min_y, env.lower()[1], "node {node_idx} min_y");
            assert_eq!(custom_node.max_x, env.upper()[0], "node {node_idx} max_x");
            assert_eq!(custom_node.max_y, env.upper()[1], "node {node_idx} max_y");

            let all_leaves = children.iter().all(|c| matches!(c, RTreeNode::Leaf(_)));

            if all_leaves {
                let ref_geo_ids: Vec<u32> = children
                    .iter()
                    .filter_map(|c| {
                        if let RTreeNode::Leaf(item) = c {
                            Some(item.geo_id)
                        } else {
                            None
                        }
                    })
                    .collect();

                match &custom_node.content {
                    NodeContent::Leaf(range) => {
                        assert_eq!(
                            &custom_tree.leaf_geo_ids[range.clone()],
                            ref_geo_ids.as_slice()
                        );
                    }
                    NodeContent::Internal(_) => panic!("expected Leaf at node {node_idx}"),
                }
            } else {
                assert!(
                    matches!(custom_node.content, NodeContent::Internal(_)),
                    "expected Internal at node {node_idx}"
                );
                for child in children {
                    if let RTreeNode::Parent(p) = child {
                        queue.push_back(p);
                    }
                }
            }

            node_idx += 1;
        }

        assert_eq!(node_idx, custom_tree.nodes.len());
    }

    #[test]
    fn test_build_side_leaf_component_types() {
        let (build_side, _) = make_build_side();

        let mut all_types: Vec<GeoComponentType> = Vec::new();

        for node_idx in 0..build_side.tree.nodes.len() {
            if let Some(side) = build_side.get_leaf_exploded(node_idx).unwrap() {
                all_types.extend(side.component_type.iter().cloned());
            }
        }

        all_types.sort_by_key(|ct| match ct {
            GeoComponentType::Point => 0u8,
            GeoComponentType::Line(_, _) => 1,
            GeoComponentType::EdgeFromPoly => 2,
        });

        // 12 geometries total (6 from shift 0,0 + 6 from shift 10,20):
        //   2× Point             → 2 × 1 =  2 Point
        //   2× MultiPoint(2)     → 2 × 2 =  4 Point
        //   2× LineString(3)     → 2 × 2 =  4 Line
        //   2× MultiLineString   → 2 × 2 =  4 Line
        //   2× Polygon(5)        → 2 × 4 =  8 EdgeFromPoly
        //   2× MultiPolygon(2×5) → 2 × 8 = 16 EdgeFromPoly
        let expected: Vec<GeoComponentType> = [
            vec![GeoComponentType::Point; 6],
            vec![GeoComponentType::default_line(); 8],
            vec![GeoComponentType::EdgeFromPoly; 24],
        ]
        .concat();

        assert_eq!(all_types, expected);
    }
}
