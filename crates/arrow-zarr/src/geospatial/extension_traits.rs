use datafusion::error::{DataFusionError, Result};

const _EPS: f64 = 1e-10;

use arrow_array::{Array, ArrayAccessor, BinaryArray, BinaryViewArray};
use arrow_schema::DataType;
use geo_traits::{
    CoordTrait, GeometryTrait, GeometryType, LineStringTrait, MultiLineStringTrait,
    MultiPointTrait, MultiPolygonTrait, PointTrait, PolygonTrait,
};
use wkb::reader::Wkb;

// A couple enums to keep track of the geometry type (without any
// othe information, just the type) and of the different basic components
// that make up the geometry.
// We make a distinction between a line as a standalone geometry and
// a line that is an edge from a polygon, as this will be useful to
// decide which checks to apply based on the spatial predicate.
#[derive(Debug, Clone, Copy)]
pub(crate) enum GeoComponentType {
    Point,
    Line(bool, bool),
    EdgeFromPoly,
}

impl PartialEq for GeoComponentType {
    fn eq(&self, other: &Self) -> bool {
        self.discriminant() == other.discriminant()
    }
}

impl Eq for GeoComponentType {}

impl PartialOrd for GeoComponentType {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GeoComponentType {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.discriminant().cmp(&other.discriminant())
    }
}

impl std::hash::Hash for GeoComponentType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.discriminant().hash(state);
    }
}

impl GeoComponentType {
    pub(crate) fn default_line() -> Self {
        Self::Line(false, false)
    }

    fn discriminant(&self) -> u8 {
        match self {
            Self::Point => 0,
            Self::Line(_, _) => 1,
            Self::EdgeFromPoly => 2,
        }
    }
}

pub(crate) type GeoComponents = (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<GeoComponentType>,
);

// A few builders to help with bounding boxes and exploding
// geometries into its components.
type Bbox = Option<(f64, f64, f64, f64)>;
#[derive(Debug, Clone)]
struct BboxBuilder {
    x_min: Option<f64>,
    y_min: Option<f64>,
    x_max: Option<f64>,
    y_max: Option<f64>,
}

impl BboxBuilder {
    fn new() -> Self {
        Self {
            x_min: None,
            y_min: None,
            x_max: None,
            y_max: None,
        }
    }

    fn add(&mut self, x: f64, y: f64) {
        self.x_min = Some(self.x_min.map_or(x, |v| v.min(x)));
        self.y_min = Some(self.y_min.map_or(y, |v| v.min(y)));
        self.x_max = Some(self.x_max.map_or(x, |v| v.max(x)));
        self.y_max = Some(self.y_max.map_or(y, |v| v.max(y)));
    }

    fn into_bbox(self) -> Bbox {
        if let (Some(x_min), Some(y_min), Some(x_max), Some(y_max)) =
            (self.x_min, self.y_min, self.x_max, self.y_max)
        {
            return Some((x_min, y_min, x_max, y_max));
        }

        None
    }
}

struct GeoComponetsBuilder {
    x1: Vec<f64>,
    y1: Vec<f64>,
    x2: Vec<f64>,
    y2: Vec<f64>,
    types: Vec<GeoComponentType>,
}

impl GeoComponetsBuilder {
    fn new() -> Self {
        Self {
            x1: Vec::new(),
            y1: Vec::new(),
            x2: Vec::new(),
            y2: Vec::new(),
            types: Vec::new(),
        }
    }

    fn add_point(&mut self, pt: &impl PointTrait<T = f64>) {
        if let Some(c) = pt.coord() {
            self.x1.push(c.x());
            self.x2.push(c.x());
            self.y1.push(c.y());
            self.y2.push(c.y());
            self.types.push(GeoComponentType::Point);
        }
    }

    // Scans already-pushed segments for any endpoint whose coordinate matches (x, y).
    // Toggles the boundary flag on every match and counts total occurrences (including
    // the new endpoint being added). Returns true when the count is even (mod-2 cancellation).
    fn find_and_cancel_boundary(&mut self, x: f64, y: f64) -> bool {
        let mut count = 1; // count the new endpoint being added
        for i in 0..self.types.len() {
            if let GeoComponentType::Line(mut left, mut right) = self.types[i] {
                if (self.x1[i] - x).abs() < _EPS && (self.y1[i] - y).abs() < _EPS {
                    left = !left;
                    count += 1;
                }
                if (self.x2[i] - x).abs() < _EPS && (self.y2[i] - y).abs() < _EPS {
                    right = !right;
                    count += 1;
                }
                self.types[i] = GeoComponentType::Line(left, right);
            }
        }
        count % 2 == 0
    }

    fn add_line(&mut self, ls: &impl LineStringTrait<T = f64>) {
        let coords: Vec<_> = ls.coords().collect();
        let n = coords.len();
        if n < 2 {
            return;
        }
        let closed = (coords[0].x() - coords[n - 1].x()).abs() < _EPS
            && (coords[0].y() - coords[n - 1].y()).abs() < _EPS;
        let num_segments = n - 1;

        // Resolve boundary flags before pushing any segments.
        // For each endpoint that would be a boundary, check whether an existing
        // segment already carries a flag at the same coordinate. If so, cancel
        // both (mod-2 rule): the existing flag is cleared and the new one is not set.
        let left_boundary = !closed && !self.find_and_cancel_boundary(coords[0].x(), coords[0].y());
        let right_boundary =
            !closed && !self.find_and_cancel_boundary(coords[n - 1].x(), coords[n - 1].y());

        for (i, cs) in coords.windows(2).enumerate() {
            self.x1.push(cs[0].x());
            self.y1.push(cs[0].y());
            self.x2.push(cs[1].x());
            self.y2.push(cs[1].y());
            let l = left_boundary && i == 0;
            let r = right_boundary && i == num_segments - 1;
            self.types.push(GeoComponentType::Line(l, r));
        }
    }

    fn add_edge(&mut self, ls: &impl LineStringTrait<T = f64>) {
        for cs in ls.coords().collect::<Vec<_>>().windows(2) {
            self.x1.push(cs[0].x());
            self.y1.push(cs[0].y());
            self.x2.push(cs[1].x());
            self.y2.push(cs[1].y());
            self.types.push(GeoComponentType::EdgeFromPoly);
        }
    }

    fn add_poly(&mut self, poly: &impl PolygonTrait<T = f64>) {
        if let Some(ext) = poly.exterior() {
            self.add_edge(&ext);
        }
        for int in poly.interiors() {
            self.add_edge(&int);
        }
    }

    fn into_geo_components(self) -> GeoComponents {
        (self.x1, self.y1, self.x2, self.y2, self.types)
    }
}

// Some functionalities that we implement directly on the Wkb struct so
// that we can extract bounding boxes and geometry types.
pub(crate) trait WkbOps {
    fn bounding_rect(&self) -> Result<Bbox>;
    fn explode(&self) -> Result<GeoComponents>;
}

impl WkbOps for Wkb<'_> {
    fn bounding_rect(&self) -> Result<Bbox> {
        let mut bbox_builder = BboxBuilder::new();
        match self.as_type() {
            GeometryType::Point(p) => {
                if let Some(c) = p.coord() {
                    bbox_builder.add(c.x(), c.y());
                }
            }
            GeometryType::MultiPoint(mp) => {
                for p in mp.points() {
                    if let Some(c) = p.coord() {
                        bbox_builder.add(c.x(), c.y());
                    }
                }
            }
            GeometryType::LineString(ls) => {
                for c in ls.coords() {
                    bbox_builder.add(c.x(), c.y());
                }
            }
            GeometryType::Polygon(poly) => {
                if let Some(ext) = poly.exterior() {
                    for c in ext.coords() {
                        bbox_builder.add(c.x(), c.y());
                    }
                }
            }
            GeometryType::MultiLineString(mls) => {
                for ls in mls.line_strings() {
                    for c in ls.coords() {
                        bbox_builder.add(c.x(), c.y());
                    }
                }
            }
            GeometryType::MultiPolygon(mp) => {
                for poly in mp.polygons() {
                    if let Some(ext) = poly.exterior() {
                        for c in ext.coords() {
                            bbox_builder.add(c.x(), c.y());
                        }
                    }
                }
            }
            _ => {
                return Err(DataFusionError::Internal(
                    "Unsupported geometry type in wkb object".into(),
                ))
            }
        }

        Ok(bbox_builder.into_bbox())
    }

    fn explode(&self) -> Result<GeoComponents> {
        let mut component_builder = GeoComponetsBuilder::new();

        match self.as_type() {
            GeometryType::Point(p) => {
                component_builder.add_point(p);
            }
            GeometryType::MultiPoint(mp) => {
                for p in mp.points() {
                    component_builder.add_point(&p);
                }
            }
            GeometryType::LineString(ls) => {
                component_builder.add_line(ls);
            }
            GeometryType::MultiLineString(mls) => {
                for ls in mls.line_strings() {
                    component_builder.add_line(ls);
                }
            }
            GeometryType::Polygon(poly) => {
                component_builder.add_poly(poly);
            }
            GeometryType::MultiPolygon(mp) => {
                for poly in mp.polygons() {
                    component_builder.add_poly(poly);
                }
            }
            _ => {
                return Err(DataFusionError::Internal(
                    "Unsupported geometry type in wkb object".into(),
                ))
            }
        }

        Ok(component_builder.into_geo_components())
    }
}

fn wkbs_from_array<'a, B>(arr: B, indices: &[u32]) -> Result<Vec<Wkb<'a>>>
where
    B: ArrayAccessor<Item = &'a [u8]> + Copy,
{
    indices
        .iter()
        .map(|&i| {
            Wkb::try_new(arr.value(i as usize))
                .map_err(|e| DataFusionError::Internal(e.to_string()))
        })
        .collect()
}

pub(crate) trait ArrayWkbOps {
    fn as_wkbs<'a>(&'a self, indices: &[u32]) -> Result<Vec<Wkb<'a>>>;
    fn wkb_at<'a>(&'a self, index: u32) -> Result<Wkb<'a>>;
}

impl ArrayWkbOps for dyn Array {
    fn as_wkbs<'a>(&'a self, indices: &[u32]) -> Result<Vec<Wkb<'a>>> {
        match self.data_type() {
            DataType::Binary => {
                let arr = self.as_any().downcast_ref::<BinaryArray>().ok_or_else(|| {
                    DataFusionError::Internal("failed to downcast geometry array".into())
                })?;
                wkbs_from_array(arr, indices)
            }
            DataType::BinaryView => {
                let arr = self
                    .as_any()
                    .downcast_ref::<BinaryViewArray>()
                    .ok_or_else(|| {
                        DataFusionError::Internal("failed to downcast geometry array".into())
                    })?;
                wkbs_from_array(arr, indices)
            }
            other => Err(DataFusionError::Internal(format!(
                "geometry column must be Binary, LargeBinary, or BinaryView, got {other}"
            ))),
        }
    }

    fn wkb_at<'a>(&'a self, index: u32) -> Result<Wkb<'a>> {
        self.as_wkbs(&[index]).map(|mut v| v.remove(0))
    }
}

// Batch versions of WkbOps for slices of Wkb objects.
// Any individual error short-circuits the entire operation.
type BboxBatch = Vec<Option<(f32, f32, f32, f32)>>;
pub(crate) trait WkbVecOps {
    fn bounding_rects(&self) -> Result<BboxBatch>;
    fn explode(&self, indices: &[usize]) -> Result<(GeoComponents, Vec<usize>)>;
}

impl WkbVecOps for [Wkb<'_>] {
    fn bounding_rects(&self) -> Result<BboxBatch> {
        self.iter()
            .map(|w| {
                Ok(w.bounding_rect()?
                    .map(|(x0, y0, x1, y1)| (x0 as f32, y0 as f32, x1 as f32, y1 as f32)))
            })
            .collect()
    }

    fn explode(&self, indices: &[usize]) -> Result<(GeoComponents, Vec<usize>)> {
        let mut total = 0usize;
        let components: Vec<GeoComponents> = self
            .iter()
            .map(|w| {
                let c = w.explode()?;
                total += c.0.len();
                Ok(c)
            })
            .collect::<Result<_>>()?;

        let mut x1 = Vec::with_capacity(total);
        let mut y1 = Vec::with_capacity(total);
        let mut x2 = Vec::with_capacity(total);
        let mut y2 = Vec::with_capacity(total);
        let mut types = Vec::with_capacity(total);
        let mut exploded_indices = Vec::with_capacity(total);

        for ((cx1, cy1, cx2, cy2, ctypes), &idx) in components.into_iter().zip(indices.iter()) {
            let n = cx1.len();
            x1.extend(cx1);
            y1.extend(cy1);
            x2.extend(cx2);
            y2.extend(cy2);
            types.extend(ctypes);
            exploded_indices.extend(std::iter::repeat_n(idx, n));
        }

        Ok(((x1, y1, x2, y2, types), exploded_indices))
    }
}

#[cfg(test)]
mod wkb_tests {
    use geo_types::{line_string, point, polygon, MultiLineString, MultiPoint, MultiPolygon};
    use wkb::writer::{
        write_line_string, write_multi_line_string, write_multi_point, write_multi_polygon,
        write_point, write_polygon, WriteOptions,
    };

    use super::*;

    fn make_points() -> Vec<Vec<u8>> {
        let opts = WriteOptions::default();
        let mut p = Vec::new();
        write_point(&mut p, &point!(x: 1.0, y: 2.0), &opts).unwrap();

        vec![p]
    }

    fn make_multi_points() -> Vec<Vec<u8>> {
        let opts = WriteOptions::default();
        let mut empty = Vec::new();
        write_multi_point(&mut empty, &MultiPoint::new(vec![]), &opts).unwrap();

        let mut mp = Vec::new();
        write_multi_point(
            &mut mp,
            &MultiPoint::new(vec![point!(x: 1.0, y: 2.0), point!(x: 3.0, y: 4.0)]),
            &opts,
        )
        .unwrap();

        vec![empty, mp]
    }

    fn make_line_strings() -> Vec<Vec<u8>> {
        let opts = WriteOptions::default();
        let mut empty = Vec::new();
        write_line_string(&mut empty, &line_string![], &opts).unwrap();

        let mut ls = Vec::new();
        write_line_string(
            &mut ls,
            &line_string![(x: 0.0, y: 0.0), (x: 1.0, y: 1.0), (x: 2.0, y: 0.0)],
            &opts,
        )
        .unwrap();

        vec![empty, ls]
    }

    fn make_multi_line_strings() -> Vec<Vec<u8>> {
        let opts = WriteOptions::default();
        let mut empty = Vec::new();
        write_multi_line_string(&mut empty, &MultiLineString::new(vec![]), &opts).unwrap();

        let mut mls = Vec::new();
        write_multi_line_string(
            &mut mls,
            &MultiLineString::new(vec![
                line_string![(x: 0.0, y: 0.0), (x: 1.0, y: 1.0)],
                line_string![(x: 2.0, y: 2.0), (x: 3.0, y: 3.0)],
            ]),
            &opts,
        )
        .unwrap();

        vec![empty, mls]
    }

    fn make_polygons() -> Vec<Vec<u8>> {
        let opts = WriteOptions::default();
        let mut empty = Vec::new();
        write_polygon(
            &mut empty,
            &geo_types::Polygon::new(line_string![], vec![]),
            &opts,
        )
        .unwrap();

        let mut no_hole = Vec::new();
        write_polygon(
            &mut no_hole,
            &polygon![
                (x: 0.0, y: 0.0),
                (x: 4.0, y: 0.0),
                (x: 4.0, y: 4.0),
                (x: 0.0, y: 4.0),
                (x: 0.0, y: 0.0),
            ],
            &opts,
        )
        .unwrap();

        let mut with_hole = Vec::new();
        write_polygon(
            &mut with_hole,
            &geo_types::Polygon::new(
                line_string![
                    (x: 0.0, y: 0.0),
                    (x: 10.0, y: 0.0),
                    (x: 10.0, y: 10.0),
                    (x: 0.0, y: 10.0),
                    (x: 0.0, y: 0.0),
                ],
                vec![line_string![
                    (x: 2.0, y: 2.0),
                    (x: 8.0, y: 2.0),
                    (x: 8.0, y: 8.0),
                    (x: 2.0, y: 8.0),
                    (x: 2.0, y: 2.0),
                ]],
            ),
            &opts,
        )
        .unwrap();

        vec![empty, no_hole, with_hole]
    }

    fn make_multi_polygons() -> Vec<Vec<u8>> {
        let opts = WriteOptions::default();
        let mut empty = Vec::new();
        write_multi_polygon(&mut empty, &MultiPolygon::new(vec![]), &opts).unwrap();

        let mut two_simple = Vec::new();
        write_multi_polygon(
            &mut two_simple,
            &MultiPolygon::new(vec![
                polygon![
                    (x: 0.0, y: 0.0),
                    (x: 1.0, y: 0.0),
                    (x: 1.0, y: 1.0),
                    (x: 0.0, y: 1.0),
                    (x: 0.0, y: 0.0),
                ],
                polygon![
                    (x: 5.0, y: 5.0),
                    (x: 6.0, y: 5.0),
                    (x: 6.0, y: 6.0),
                    (x: 5.0, y: 6.0),
                    (x: 5.0, y: 5.0),
                ],
            ]),
            &opts,
        )
        .unwrap();

        let mut one_simple_one_hole = Vec::new();
        write_multi_polygon(
            &mut one_simple_one_hole,
            &MultiPolygon::new(vec![
                polygon![
                    (x: 0.0, y: 0.0),
                    (x: 1.0, y: 0.0),
                    (x: 1.0, y: 1.0),
                    (x: 0.0, y: 1.0),
                    (x: 0.0, y: 0.0),
                ],
                geo_types::Polygon::new(
                    line_string![
                        (x: 10.0, y: 10.0),
                        (x: 20.0, y: 10.0),
                        (x: 20.0, y: 20.0),
                        (x: 10.0, y: 20.0),
                        (x: 10.0, y: 10.0),
                    ],
                    vec![line_string![
                        (x: 12.0, y: 12.0),
                        (x: 18.0, y: 12.0),
                        (x: 18.0, y: 18.0),
                        (x: 12.0, y: 18.0),
                        (x: 12.0, y: 12.0),
                    ]],
                ),
            ]),
            &opts,
        )
        .unwrap();

        vec![empty, two_simple, one_simple_one_hole]
    }

    #[test]
    fn test_bounding_rects() {
        let bufs: Vec<Vec<u8>> = [
            make_points(),
            make_multi_points(),
            make_line_strings(),
            make_multi_line_strings(),
            make_polygons(),
            make_multi_polygons(),
        ]
        .into_iter()
        .flatten()
        .collect();

        let wkbs: Vec<_> = bufs.iter().map(|b| Wkb::try_new(b).unwrap()).collect();
        let rects = wkbs.bounding_rects().unwrap();

        // None for empty geometries
        assert!(rects[1].is_none()); // empty MultiPoint
        assert!(rects[3].is_none()); // empty LineString
        assert!(rects[5].is_none()); // empty MultiLineString
        assert!(rects[7].is_none()); // empty Polygon
        assert!(rects[10].is_none()); // empty MultiPolygon

        // Some with correct bbox for non-empty geometries
        assert_eq!(rects[0], Some((1.0, 2.0, 1.0, 2.0))); // Point(1,2)
        assert_eq!(rects[2], Some((1.0, 2.0, 3.0, 4.0))); // MultiPoint (1,2),(3,4)
        assert_eq!(rects[4], Some((0.0, 0.0, 2.0, 1.0))); // LineString
        assert_eq!(rects[6], Some((0.0, 0.0, 3.0, 3.0))); // MultiLineString
        assert_eq!(rects[8], Some((0.0, 0.0, 4.0, 4.0))); // Polygon no hole
        assert_eq!(rects[9], Some((0.0, 0.0, 10.0, 10.0))); // Polygon with hole
        assert_eq!(rects[11], Some((0.0, 0.0, 6.0, 6.0))); // MultiPolygon two simple
        assert_eq!(rects[12], Some((0.0, 0.0, 20.0, 20.0))); // MultiPolygon simple+hole
    }

    #[test]
    fn test_explode() {
        let bufs: Vec<Vec<u8>> = [
            make_points(),
            make_multi_points(),
            make_line_strings(),
            make_multi_line_strings(),
            make_polygons(),
            make_multi_polygons(),
        ]
        .into_iter()
        .flatten()
        .collect();

        let wkbs: Vec<_> = bufs.iter().map(|b| Wkb::try_new(b).unwrap()).collect();
        let indices: Vec<usize> = (0..wkbs.len()).collect();
        let ((x1, y1, x2, y2, types), exploded_indices) = wkbs.explode(&indices).unwrap();

        // Expected component counts per input:
        //  0: Point(1,2)              → 1 point
        //  1: empty MultiPoint        → 0
        //  2: MultiPoint (1,2),(3,4)  → 2 points
        //  3: empty LineString        → 0
        //  4: LineString 3 coords     → 2 lines
        //  5: empty MultiLineString   → 0
        //  6: MultiLineString 2×2     → 2 lines
        //  7: empty Polygon           → 0
        //  8: Polygon no hole 5 coords→ 4 edges
        //  9: Polygon with hole 5+5   → 8 edges
        // 10: empty MultiPolygon      → 0
        // 11: MultiPoly 2 simple 5+5  → 8 edges
        // 12: MultiPoly simple+hole   → 12 edges
        // Total: 1+2+2+2+4+8+8+12 = 39

        assert_eq!(x1.len(), 39);

        // Check exploded indices map components back to their source
        let expected_indices: Vec<usize> = [
            vec![0; 1],
            vec![2; 2],
            vec![4; 2],
            vec![6; 2],
            vec![8; 4],
            vec![9; 8],
            vec![11; 8],
            vec![12; 12],
        ]
        .into_iter()
        .flatten()
        .collect();
        assert_eq!(exploded_indices, expected_indices);

        // Check component types
        let expected_types: Vec<GeoComponentType> = [
            vec![GeoComponentType::Point; 1],
            vec![GeoComponentType::Point; 2],
            vec![GeoComponentType::default_line(); 2],
            vec![GeoComponentType::default_line(); 2],
            vec![GeoComponentType::EdgeFromPoly; 4],
            vec![GeoComponentType::EdgeFromPoly; 8],
            vec![GeoComponentType::EdgeFromPoly; 8],
            vec![GeoComponentType::EdgeFromPoly; 12],
        ]
        .into_iter()
        .flatten()
        .collect();
        assert_eq!(types, expected_types);

        // Spot-check some coordinate values:
        // First component: Point(1,2) -> x1=1, y1=2, x2=1, y2=2
        assert_eq!((x1[0], y1[0], x2[0], y2[0]), (1.0, 2.0, 1.0, 2.0));

        // Components 1-2: MultiPoint points -> (1,2) and (3,4)
        assert_eq!((x1[1], y1[1]), (1.0, 2.0));
        assert_eq!((x1[2], y1[2]), (3.0, 4.0));

        // Components 3-4: LineString segments (0,0)->(1,1) and (1,1)->(2,0)
        assert_eq!((x1[3], y1[3], x2[3], y2[3]), (0.0, 0.0, 1.0, 1.0));
        assert_eq!((x1[4], y1[4], x2[4], y2[4]), (1.0, 1.0, 2.0, 0.0));

        // Components 5-6: MultiLineString segments (0,0)->(1,1) and (2,2)->(3,3)
        assert_eq!((x1[5], y1[5], x2[5], y2[5]), (0.0, 0.0, 1.0, 1.0));
        assert_eq!((x1[6], y1[6], x2[6], y2[6]), (2.0, 2.0, 3.0, 3.0));

        // Components 7-10: Polygon no hole edges
        assert_eq!((x1[7], y1[7], x2[7], y2[7]), (0.0, 0.0, 4.0, 0.0));
        assert_eq!((x1[8], y1[8], x2[8], y2[8]), (4.0, 0.0, 4.0, 4.0));
        assert_eq!((x1[9], y1[9], x2[9], y2[9]), (4.0, 4.0, 0.0, 4.0));
        assert_eq!((x1[10], y1[10], x2[10], y2[10]), (0.0, 4.0, 0.0, 0.0));
    }

    // Helper: returns (types, x1, y1, x2, y2) for a MultiLineString WKB.
    fn explode_multiline(lines: Vec<geo_types::LineString>) -> Vec<GeoComponentType> {
        use geo_types::MultiLineString;
        use wkb::writer::{write_multi_line_string, WriteOptions};

        let mut buf = Vec::new();
        write_multi_line_string(
            &mut buf,
            &MultiLineString::new(lines),
            &WriteOptions::default(),
        )
        .unwrap();
        let wkb = Wkb::try_new(&buf).unwrap();
        let ((_, _, _, _, types), _) = [wkb].explode(&[0]).unwrap();
        types
    }

    #[test]
    fn test_boundary_connected_path() {
        // Two components A→B, B→C. B is shared → both flags at B cancel.
        // Only A (left of seg 0) and C (right of seg 1) remain as boundary.
        let types = explode_multiline(vec![
            line_string![(x: 0.0, y: 0.0), (x: 1.0, y: 1.0)], // A→B
            line_string![(x: 1.0, y: 1.0), (x: 2.0, y: 0.0)], // B→C
        ]);
        assert_eq!(types.len(), 2);
        assert!(
            matches!(types[0], GeoComponentType::Line(true, false)),
            "A left-boundary, B right-boundary cancelled"
        );
        assert!(
            matches!(types[1], GeoComponentType::Line(false, true)),
            "B left-boundary cancelled, C right-boundary"
        );
    }

    #[test]
    fn test_boundary_three_way_junction() {
        // Three components A→B, C→B, D→B. B appears 3 times as an endpoint (odd) → boundary.
        // All segments at B carry the boundary flag after toggling.
        let types = explode_multiline(vec![
            line_string![(x: 0.0, y: 0.0), (x: 1.0, y: 1.0)], // A→B
            line_string![(x: 2.0, y: 0.0), (x: 1.0, y: 1.0)], // C→B
            line_string![(x: 3.0, y: 2.0), (x: 1.0, y: 1.0)], // D→B
        ]);
        assert_eq!(types.len(), 3);
        // All segments carry boundary at both endpoints (A/C/D are unique, B is odd-count)
        assert!(matches!(types[0], GeoComponentType::Line(true, true)));
        assert!(matches!(types[1], GeoComponentType::Line(true, true)));
        assert!(matches!(types[2], GeoComponentType::Line(true, true)));
    }

    #[test]
    fn test_boundary_disconnected() {
        // Two components with no shared endpoints. All 4 endpoints remain as boundaries.
        let types = explode_multiline(vec![
            line_string![(x: 0.0, y: 0.0), (x: 1.0, y: 1.0)], // A→B
            line_string![(x: 5.0, y: 5.0), (x: 6.0, y: 6.0)], // C→D
        ]);
        assert_eq!(types.len(), 2);
        assert!(matches!(types[0], GeoComponentType::Line(true, true)));
        assert!(matches!(types[1], GeoComponentType::Line(true, true)));
    }

    #[test]
    fn test_boundary_closed_ring_and_open_line() {
        // The open line makes (0,0) a boundary of the whole MultiLineString.
        // Ring segments touching (0,0) get the boundary flag toggled on.
        let types = explode_multiline(vec![
            line_string![(x: 0.0, y: 0.0), (x: 1.0, y: 0.0), (x: 1.0, y: 1.0), (x: 0.0, y: 0.0)], // closed
            line_string![(x: 0.0, y: 0.0), (x: 2.0, y: 0.0)], // open, shares ring vertex
        ]);
        // Ring seg 0: (0,0)→(1,0), boundary toggled at (0,0)
        assert!(matches!(types[0], GeoComponentType::Line(true, false)));
        // Ring seg 1: (1,0)→(1,1), untouched
        assert!(matches!(types[1], GeoComponentType::Line(false, false)));
        // Ring seg 2: (1,1)→(0,0), boundary toggled at (0,0)
        assert!(matches!(types[2], GeoComponentType::Line(false, true)));
        // Open line: (0,0)→(2,0), boundary at both endpoints
        assert!(matches!(types[3], GeoComponentType::Line(true, true)));
    }
}
