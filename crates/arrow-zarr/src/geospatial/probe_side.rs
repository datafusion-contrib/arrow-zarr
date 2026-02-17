use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch};
use datafusion::error::Result;
use datafusion::physical_plan::PhysicalExpr;

use super::extension_traits::{ArrayWkbOps, WkbVecOps};
use super::operations::ExplodedSide;

pub(crate) struct ProbeBatch {
    pub(crate) rec_batch: RecordBatch,
    pub(crate) geometry_array: ArrayRef,
    pub(crate) min_x: Vec<f32>,
    pub(crate) min_y: Vec<f32>,
    pub(crate) max_x: Vec<f32>,
    pub(crate) max_y: Vec<f32>,
    pub(crate) ids: Vec<u32>, // original row indices for non-null, non-empty geometries only
    pub(crate) boundary: usize, // rows 0..boundary are "active" during tree traversal
}

impl ProbeBatch {
    pub(crate) fn new(rec_batch: RecordBatch, geo_expr: &Arc<dyn PhysicalExpr>) -> Result<Self> {
        let n = rec_batch.num_rows();
        let geometry_array = geo_expr.evaluate(&rec_batch)?.into_array(n)?;

        let non_null_indices: Vec<u32> = (0..n as u32)
            .filter(|&i| !geometry_array.is_null(i as usize))
            .collect();
        let wkbs = geometry_array.as_wkbs(&non_null_indices)?;

        let rects = wkbs.as_slice().bounding_rects()?;

        let (ids, min_x, min_y, max_x, max_y) = non_null_indices
            .into_iter()
            .zip(rects)
            .filter_map(|(idx, rect)| rect.map(|(x0, y0, x1, y1)| (idx, x0, y0, x1, y1)))
            .fold(
                (vec![], vec![], vec![], vec![], vec![]),
                |(mut ids, mut xs0, mut ys0, mut xs1, mut ys1), (idx, x0, y0, x1, y1)| {
                    ids.push(idx);
                    xs0.push(x0);
                    ys0.push(y0);
                    xs1.push(x1);
                    ys1.push(y1);
                    (ids, xs0, ys0, xs1, ys1)
                },
            );

        let boundary = ids.len();
        Ok(Self {
            rec_batch,
            geometry_array,
            min_x,
            min_y,
            max_x,
            max_y,
            ids,
            boundary,
        })
    }

    // Partition rows 0..boundary against a node bbox. Rows whose bbox overlaps the node bbox
    // are shuffled to the front; non-overlapping rows are moved past the new boundary.
    // Returns the new boundary (number of rows that passed).
    pub(crate) fn partition(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) -> usize {
        let mut w = 0;
        for r in 0..self.boundary {
            if self.min_x[r] <= x1
                && self.max_x[r] >= x0
                && self.min_y[r] <= y1
                && self.max_y[r] >= y0
            {
                if w != r {
                    self.min_x.swap(w, r);
                    self.min_y.swap(w, r);
                    self.max_x.swap(w, r);
                    self.max_y.swap(w, r);
                    self.ids.swap(w, r);
                }
                w += 1;
            }
        }
        self.boundary = w;
        w
    }

    pub(crate) fn set_boundary(&mut self, boundary: usize) {
        self.boundary = boundary;
    }

    pub(crate) fn get_current_valid_exploded(&self) -> Result<ExplodedSide> {
        let active_ids = &self.ids[..self.boundary];
        let wkbs = self.geometry_array.as_wkbs(active_ids)?;
        ExplodedSide::new(active_ids, &wkbs)
    }
}

#[cfg(test)]
mod probe_side_tests {
    use std::sync::Arc;

    use arrow_array::BinaryArray;
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::physical_expr::expressions::Column;

    use super::*;
    use crate::geospatial::extension_traits::GeoComponentType;
    use crate::geospatial::test_utils::make_wkb_bufs;

    #[test]
    fn test_probe_batch_all_valid() {
        let (bufs, rects) = make_wkb_bufs(0.0, 0.0);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "geometry",
            DataType::Binary,
            false,
        )]));
        let geo_col = Arc::new(BinaryArray::from_iter_values(
            bufs.iter().map(|b| b.as_slice()),
        ));
        let batch = RecordBatch::try_new(schema, vec![geo_col]).unwrap();

        let probe = ProbeBatch::new(
            batch,
            &(Arc::new(Column::new("geometry", 0)) as Arc<dyn PhysicalExpr>),
        )
        .unwrap();

        assert_eq!(probe.ids, vec![0, 1, 2, 3, 4, 5]);
        for (i, &(exp_x0, exp_y0, exp_x1, exp_y1)) in rects.iter().enumerate() {
            assert_eq!(probe.min_x[i], exp_x0, "min_x at {i}");
            assert_eq!(probe.min_y[i], exp_y0, "min_y at {i}");
            assert_eq!(probe.max_x[i], exp_x1, "max_x at {i}");
            assert_eq!(probe.max_y[i], exp_y1, "max_y at {i}");
        }
    }

    #[test]
    fn test_probe_batch_null_row() {
        let (bufs, rects) = make_wkb_bufs(0.0, 0.0);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "geometry",
            DataType::Binary,
            true,
        )]));
        // rows: valid(0), null, valid(1)
        let geo_col = Arc::new(BinaryArray::from(vec![
            Some(bufs[0].as_slice()),
            None,
            Some(bufs[1].as_slice()),
        ]));
        let batch = RecordBatch::try_new(schema, vec![geo_col]).unwrap();

        let probe = ProbeBatch::new(
            batch,
            &(Arc::new(Column::new("geometry", 0)) as Arc<dyn PhysicalExpr>),
        )
        .unwrap();

        // null row is dropped; ids are original row indices
        assert_eq!(probe.ids, vec![0, 2]);
        assert_eq!(probe.min_x.len(), 2);
        assert_eq!(probe.min_x[0], rects[0].0);
        assert_eq!(probe.min_y[0], rects[0].1);
        assert_eq!(probe.max_x[0], rects[0].2);
        assert_eq!(probe.max_y[0], rects[0].3);
        assert_eq!(probe.min_x[1], rects[1].0);
        assert_eq!(probe.min_y[1], rects[1].1);
        assert_eq!(probe.max_x[1], rects[1].2);
        assert_eq!(probe.max_y[1], rects[1].3);
    }

    #[test]
    fn test_probe_batch_partition_and_explode() {
        let (bufs, _) = make_wkb_bufs(0.0, 0.0);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "geometry",
            DataType::Binary,
            false,
        )]));
        let geo_col = Arc::new(BinaryArray::from_iter_values(
            bufs.iter().map(|b| b.as_slice()),
        ));
        let batch = RecordBatch::try_new(schema, vec![geo_col]).unwrap();
        let mut probe = ProbeBatch::new(
            batch,
            &(Arc::new(Column::new("geometry", 0)) as Arc<dyn PhysicalExpr>),
        )
        .unwrap();

        // MultiPoint bbox is (3,4,5,6): min_x=3 > x1=2, so it is filtered out
        let new_boundary = probe.partition(0.0, 0.0, 2.0, 2.0);
        assert_eq!(new_boundary, 5);

        let exploded = probe.get_current_valid_exploded().unwrap();
        let mut types = exploded.component_type;
        types.sort_by_key(|ct| match ct {
            GeoComponentType::Point => 0u8,
            GeoComponentType::Line(_, _) => 1,
            GeoComponentType::EdgeFromPoly => 2,
        });

        // 5 active geometries (MultiPoint excluded):
        //   1× Point             →  1 Point
        //   1× LineString(3pts)  →  2 Line
        //   1× MultiLineString   →  2 Line
        //   1× Polygon(5pts)     →  4 EdgeFromPoly
        //   1× MultiPolygon(2×5) →  8 EdgeFromPoly
        let expected: Vec<GeoComponentType> = [
            vec![GeoComponentType::Point; 1],
            vec![GeoComponentType::default_line(); 4],
            vec![GeoComponentType::EdgeFromPoly; 12],
        ]
        .concat();

        assert_eq!(types, expected);
    }
}
