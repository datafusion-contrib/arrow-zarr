use geo_traits::{
    CoordTrait, GeometryTrait, GeometryType, LineStringTrait, MultiLineStringTrait,
    MultiPointTrait, MultiPolygonTrait, PointTrait, PolygonTrait,
};
use itertools::Itertools;
use wkb::reader::Wkb;

const EPS: f64 = 1e-10;
const N_STRIPES: usize = 64;

//*************************************************
// Base component that all geometries are built from.
// all the basic checks are implemented here.
//*************************************************
#[derive(Copy, Clone, Debug)]
pub(crate) struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn offset(self, d: Dir) -> Point {
        Point {
            x: self.x + d.x,
            y: self.y + d.y,
        }
    }

    pub(crate) fn xy(&self) -> (f64, f64) {
        (self.x, self.y)
    }

    pub(crate) fn is_equal(&self, other: &Point) -> bool {
        self.xy() == other.xy()
    }

    pub(crate) fn diff(&self, other: &Point) -> Dir {
        Dir {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

#[derive(Copy, Clone)]
pub(crate) struct Dir {
    x: f64,
    y: f64,
}

impl Dir {
    fn cross(&self, other: &Dir) -> f64 {
        self.x * other.y - self.y * other.x
    }

    fn dot(&self, other: &Dir) -> f64 {
        self.x * other.x + self.y * other.y
    }

    fn turn_to(&self, other: &Dir) -> f64 {
        self.cross(other).atan2(self.dot(other))
    }

    fn into_edge(self, p1: Point, raw_turn: f64, interior_on_left: bool) -> Edge {
        let angle = if interior_on_left {
            std::f64::consts::PI - raw_turn
        } else {
            std::f64::consts::PI + raw_turn
        };
        Edge {
            p1,
            p2: p1.offset(self),
            angle,
            interior_on_left,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct LineSegment {
    p1: Point,
    p2: Point,
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct Edge {
    p1: Point,
    p2: Point,
    angle: f64,
    interior_on_left: bool,
}

impl Edge {
    pub(crate) fn interior_on_left(&self) -> bool {
        self.interior_on_left
    }

    pub(crate) fn is_reflex(&self) -> bool {
        self.angle > std::f64::consts::PI
    }

    pub(crate) fn crossing_at_vertex(&self, seg: &impl ToDir) -> Option<bool> {
        let raw_turn = self.to_dir().turn_to(&seg.to_dir());
        let seg_angle = if self.interior_on_left {
            std::f64::consts::PI - raw_turn
        } else {
            std::f64::consts::PI + raw_turn
        };
        let flipped = if seg_angle < std::f64::consts::PI {
            seg_angle + std::f64::consts::PI
        } else {
            seg_angle - std::f64::consts::PI
        };
        let seg_in_wedge = seg_angle > EPS && seg_angle < self.angle - EPS;
        let flipped_in_wedge = flipped > EPS && flipped < self.angle - EPS;
        if seg_in_wedge != flipped_in_wedge {
            Some(seg_in_wedge)
        } else {
            None
        }
    }
}

pub(crate) trait ToPoints {
    fn to_points(&self) -> (&Point, &Point);
    fn xs(&self) -> (f64, f64);
    fn ys(&self) -> (f64, f64);
    fn midpoint(&self) -> Point {
        let (p1, p2) = self.to_points();
        Point {
            x: (p1.x + p2.x) / 2.0,
            y: (p1.y + p2.y) / 2.0,
        }
    }
}

impl ToPoints for LineSegment {
    fn to_points(&self) -> (&Point, &Point) {
        (&self.p1, &self.p2)
    }

    fn xs(&self) -> (f64, f64) {
        (self.p1.x, self.p2.x)
    }

    fn ys(&self) -> (f64, f64) {
        (self.p1.y, self.p2.y)
    }
}

impl ToPoints for Edge {
    fn to_points(&self) -> (&Point, &Point) {
        (&self.p1, &self.p2)
    }

    fn xs(&self) -> (f64, f64) {
        (self.p1.x, self.p2.x)
    }

    fn ys(&self) -> (f64, f64) {
        (self.p1.y, self.p2.y)
    }
}

pub(crate) trait ToDir {
    fn to_dir(&self) -> Dir;
}

impl ToDir for Dir {
    fn to_dir(&self) -> Dir {
        *self
    }
}

impl ToDir for LineSegment {
    fn to_dir(&self) -> Dir {
        self.p2.diff(&self.p1)
    }
}

impl ToDir for Edge {
    fn to_dir(&self) -> Dir {
        self.p2.diff(&self.p1)
    }
}

pub(crate) trait SegmentTrait: ToPoints + ToDir {
    fn norm_sq(&self) -> f64 {
        let d = self.to_dir();
        d.dot(&d)
    }

    fn cross(&self, other: &impl ToDir) -> f64 {
        self.to_dir().cross(&other.to_dir())
    }

    fn dot(&self, other: &impl ToDir) -> f64 {
        self.to_dir().dot(&other.to_dir())
    }

    fn y_range_contains(&self, p: &Point) -> bool {
        let (y1, y2) = self.ys();
        (y1 > p.y) != (y2 > p.y)
    }

    fn x_ranges_overlap(&self, other: &impl ToPoints) -> bool {
        let (ax1, ax2) = self.xs();
        let (bx1, bx2) = other.xs();
        ax1.min(ax2) <= bx1.max(bx2) && ax1.max(ax2) >= bx1.min(bx2)
    }

    fn x_range_contains_point(&self, p: &Point) -> bool {
        let (x1, x2) = self.xs();
        let px = p.x;
        x1.min(x2) <= px && x1.max(x2) >= px
    }

    fn x_intercept_at_point(&self, p: &Point) -> f64 {
        let (p1, p2) = self.to_points();
        p1.x + (p.y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y)
    }

    fn is_collinear_with(&self, p: &Point) -> bool {
        let (p1, _) = self.to_points();
        let dp = p.diff(p1);
        let c = self.cross(&dp);
        c * c < EPS * EPS * self.norm_sq()
    }

    fn projection_t(&self, p: &Point) -> f64 {
        let (p1, _) = self.to_points();
        let dp = p.diff(p1);
        self.dot(&dp) / self.norm_sq()
    }
}

impl<T: ToPoints + ToDir> SegmentTrait for T {}

pub(crate) struct YStripes {
    buckets: Vec<Vec<usize>>,
    y_min: f64,
    stripe_h: f64, // (y_max - y_min) / N_STRIPES; 0.0 when all components share the same y
}

impl YStripes {
    fn from_points(points: &[Point]) -> Self {
        let n = points.len();
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); N_STRIPES];

        if n == 0 {
            return YStripes {
                buckets,
                y_min: 0.0,
                stripe_h: 0.0,
            };
        }

        let (y_min, y_max) = points
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
                (lo.min(p.y), hi.max(p.y))
            });

        if y_max - y_min < EPS {
            for i in 0..n {
                buckets[0].push(i);
            }
            return YStripes {
                buckets,
                y_min,
                stripe_h: 0.0,
            };
        }

        let stripe_h = (y_max - y_min) / N_STRIPES as f64;
        for (i, p) in points.iter().enumerate() {
            let s = ((p.y - y_min) / stripe_h).floor() as usize;
            buckets[s.min(N_STRIPES - 1)].push(i);
        }
        YStripes {
            buckets,
            y_min,
            stripe_h,
        }
    }

    fn from_segments(segments: &[impl ToPoints]) -> Self {
        let n = segments.len();
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); N_STRIPES];

        if n == 0 {
            return YStripes {
                buckets,
                y_min: 0.0,
                stripe_h: 0.0,
            };
        }

        let (y_min, y_max) =
            segments
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), seg| {
                    let (y1, y2) = seg.ys();
                    (lo.min(y1).min(y2), hi.max(y1).max(y2))
                });

        if y_max - y_min < EPS {
            for i in 0..n {
                buckets[0].push(i);
            }
            return YStripes {
                buckets,
                y_min,
                stripe_h: 0.0,
            };
        }

        let stripe_h = (y_max - y_min) / N_STRIPES as f64;
        for (i, seg) in segments.iter().enumerate() {
            let (y1, y2) = seg.ys();
            let lo = y1.min(y2);
            let hi = y1.max(y2);
            let s = ((lo - y_min) / stripe_h).floor() as usize;
            let e = ((hi - y_min) / stripe_h).floor() as usize;
            for bucket in buckets
                .iter_mut()
                .take(e.min(N_STRIPES - 1) + 1)
                .skip(s.min(N_STRIPES - 1))
            {
                bucket.push(i);
            }
        }
        YStripes {
            buckets,
            y_min,
            stripe_h,
        }
    }

    pub(crate) fn candidate_pairs(&self, other: &YStripes) -> Vec<(usize, usize)> {
        let mut pairs: Vec<(usize, usize)> = Vec::new();

        for i in 0..N_STRIPES {
            if self.buckets[i].is_empty() {
                // Degenerate left: all content is in bucket 0, nothing more after that.
                if self.stripe_h < EPS {
                    break;
                }
                continue;
            }

            let left_lo = self.y_min + i as f64 * self.stripe_h;
            let left_hi = left_lo + self.stripe_h;

            let (j_start, j_end) = if other.stripe_h < EPS {
                // Right is a single y value: check if it falls in this left bucket.
                // EPS padding on both sides handles the both-degenerate case
                // (left_lo == left_hi == self.y_min) and exact boundary points.
                if other.y_min < left_lo - EPS {
                    break; // right y is below this and all future left buckets
                }
                if other.y_min > left_hi + EPS {
                    continue;
                }
                (0, 0)
            } else {
                let j_lo = ((left_lo - other.y_min) / other.stripe_h).floor() as isize;
                let j_hi = ((left_hi - other.y_min) / other.stripe_h).floor() as isize;
                let j_end = j_hi.min(N_STRIPES as isize - 1);
                if j_end < 0 {
                    continue;
                }
                // Clamp j_start to j_end: handles the case where left_lo lands exactly
                // at other's y_max (j_lo == N_STRIPES) due to the same min(N_STRIPES-1)
                // clamp used during stripe construction.
                (j_lo.max(0).min(j_end) as usize, j_end as usize)
            };

            for j in j_start..=j_end {
                for &li in &self.buckets[i] {
                    for &ri in &other.buckets[j] {
                        pairs.push((li, ri));
                    }
                }
            }
        }

        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }
}

//*************************************************
// Struct that holds all the relevant data from a
// geometry and then exposes a view that will be used
// in geospatial checks.
//*************************************************
pub(crate) enum GeoOwned {
    Point {
        points: Vec<Point>,
        stripes: YStripes,
    },
    Line {
        lines: Vec<LineSegment>,
        stripes: YStripes,

        // true = that endpoint is a linestring boundary point (mod-2 rule)
        b1s: Vec<bool>,
        b2s: Vec<bool>,

        // All-false flags of length n, for not_exterior() for which no endpoint
        // is a boundary. Since we return GeoViews as references to the GeoOwned
        // class, we need false flas to live here.
        false_flags: Vec<bool>,
    },
    Poly {
        edges: Vec<Edge>,
        stripes: YStripes,

        // Index where hole edges begin; edges [0..holes_start) are exterior,
        // [holes_start..) are holes.
        holes_start: usize,
        holes_stripes: YStripes,
        poly_ids: Vec<usize>,
    },
}

impl GeoOwned {
    pub(crate) fn num_components(&self) -> usize {
        match self {
            GeoOwned::Point { points, .. } => points.len(),
            GeoOwned::Line { lines, .. } => lines.len(),
            GeoOwned::Poly { edges, .. } => edges.len(),
        }
    }

    pub(crate) fn x_bounds(&self) -> (f64, f64) {
        match self {
            GeoOwned::Point { points, .. } => points
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
                    (lo.min(p.x), hi.max(p.x))
                }),
            GeoOwned::Line { lines, .. } => {
                lines
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), seg| {
                        let (x1, x2) = seg.xs();
                        (lo.min(x1).min(x2), hi.max(x1).max(x2))
                    })
            }
            GeoOwned::Poly { edges, .. } => {
                edges
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), e| {
                        let (x1, x2) = e.xs();
                        (lo.min(x1).min(x2), hi.max(x1).max(x2))
                    })
            }
        }
    }

    pub(crate) fn y_bounds(&self) -> (f64, f64) {
        match self {
            GeoOwned::Point { points, .. } => points
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
                    (lo.min(p.y), hi.max(p.y))
                }),
            GeoOwned::Line { lines, .. } => {
                lines
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), seg| {
                        let (y1, y2) = seg.ys();
                        (lo.min(y1).min(y2), hi.max(y1).max(y2))
                    })
            }
            GeoOwned::Poly { edges, .. } => {
                edges
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), e| {
                        let (y1, y2) = e.ys();
                        (lo.min(y1).min(y2), hi.max(y1).max(y2))
                    })
            }
        }
    }

    fn from_points(points: Vec<Point>) -> Self {
        let stripes = YStripes::from_points(&points);
        GeoOwned::Point { points, stripes }
    }

    fn from_line(lines: Vec<LineSegment>, b1s: Vec<bool>, b2s: Vec<bool>) -> Self {
        let n = lines.len();
        assert_eq!(b1s.len(), n, "b1s length mismatch");
        assert_eq!(b2s.len(), n, "b2s length mismatch");
        let stripes = YStripes::from_segments(&lines);
        let false_flags = vec![false; n];
        GeoOwned::Line {
            lines,
            stripes,
            b1s,
            b2s,
            false_flags,
        }
    }

    fn from_poly(edges: Vec<Edge>, holes_start: usize, poly_ids: Vec<usize>) -> Self {
        let stripes = YStripes::from_segments(&edges);
        let holes_stripes = YStripes::from_segments(&edges[holes_start..]);
        GeoOwned::Poly {
            edges,
            stripes,
            holes_start,
            holes_stripes,
            poly_ids,
        }
    }

    pub(crate) fn from_wkb(bytes: &[u8]) -> Self {
        let wkb = Wkb::try_new(bytes).expect("invalid WKB bytes");
        match wkb.as_type() {
            GeometryType::Point(p) => {
                let mut points = Vec::new();
                if let Some(coord) = p.coord() {
                    points.push(Point {
                        x: coord.x(),
                        y: coord.y(),
                    });
                }
                GeoOwned::from_points(points)
            }
            GeometryType::MultiPoint(mp) => {
                let mut points = Vec::new();
                for p in mp.points() {
                    if let Some(coord) = p.coord() {
                        points.push(Point {
                            x: coord.x(),
                            y: coord.y(),
                        });
                    }
                }
                GeoOwned::from_points(points)
            }
            GeometryType::LineString(ls) => build_line_geo(std::iter::once(ls)),
            GeometryType::MultiLineString(mls) => build_line_geo(mls.line_strings()),
            GeometryType::Polygon(poly) => build_poly_geo(std::iter::once(poly)),
            GeometryType::MultiPolygon(mp) => build_poly_geo(mp.polygons()),
            _ => todo!("from_wkb: unsupported geometry type"),
        }
    }

    // Returns the interior of the geometry. Right now, for polygons, we are
    // returning the same thing as "not exterior", which is not quite correct,
    // but it's not a problem for now.
    pub(crate) fn interior(&self) -> (GeoView<'_>, &YStripes) {
        match self {
            GeoOwned::Point { points, stripes } => (GeoView::Points { points }, stripes),
            GeoOwned::Line {
                lines,
                b1s,
                b2s,
                stripes,
                ..
            } => (GeoView::LineSegments { lines, b1s, b2s }, stripes),
            GeoOwned::Poly {
                edges,
                poly_ids,
                stripes,
                ..
            } => (GeoView::PolyEdges { edges, poly_ids }, stripes),
        }
    }

    pub(crate) fn not_exterior(&self) -> (GeoView<'_>, &YStripes) {
        match self {
            GeoOwned::Point { points, stripes } => (GeoView::Points { points }, stripes),
            GeoOwned::Line {
                lines,
                false_flags,
                stripes,
                ..
            } => (
                GeoView::LineSegments {
                    lines,
                    b1s: false_flags,
                    b2s: false_flags,
                },
                stripes,
            ),
            // TODO: will need to add some flag to indicate that the edges
            // themselves do count (as opposed to interior above).
            GeoOwned::Poly {
                edges,
                poly_ids,
                stripes,
                ..
            } => (GeoView::PolyEdges { edges, poly_ids }, stripes),
        }
    }

    pub(crate) fn holes(&self) -> Option<(GeoView<'_>, &YStripes)> {
        match self {
            GeoOwned::Point { .. } | GeoOwned::Line { .. } => None,
            GeoOwned::Poly {
                edges,
                poly_ids,
                holes_start,
                holes_stripes,
                ..
            } => Some((
                GeoView::PolyEdges {
                    edges: &edges[*holes_start..],
                    poly_ids: &poly_ids[*holes_start..],
                },
                holes_stripes,
            )),
        }
    }
}

fn build_line_geo<LS: LineStringTrait<T = f64>>(linestrings: impl Iterator<Item = LS>) -> GeoOwned {
    let mut lines: Vec<LineSegment> = Vec::new();
    let mut b1s: Vec<bool> = Vec::new();
    let mut b2s: Vec<bool> = Vec::new();

    for ls in linestrings {
        let n = ls.coords().len();
        assert!(
            n >= 2,
            "degenerate linestring with fewer than 2 coordinates"
        );

        // This part handles flipping the endpoints to figure out
        // what's a boundary (odd number of appearances) and what's
        // not (even number).
        for (c1, c2) in ls
            .coords()
            .map(|c| Point { x: c.x(), y: c.y() })
            .tuple_windows::<(_, _)>()
        {
            let mut nb1 = true;
            let mut nb2 = true;
            for (j, seg) in lines.iter().enumerate() {
                if c1.is_equal(&seg.p1) {
                    nb1 = !nb1;
                    b1s[j] = !b1s[j];
                }
                if c1.is_equal(&seg.p2) {
                    nb1 = !nb1;
                    b2s[j] = !b2s[j];
                }
                if c2.is_equal(&seg.p1) {
                    nb2 = !nb2;
                    b1s[j] = !b1s[j];
                }
                if c2.is_equal(&seg.p2) {
                    nb2 = !nb2;
                    b2s[j] = !b2s[j];
                }
            }
            lines.push(LineSegment { p1: c1, p2: c2 });
            b1s.push(nb1);
            b2s.push(nb2);
        }
    }

    GeoOwned::from_line(lines, b1s, b2s)
}

// Handles one ring from one poly, filling in the edges and
// computing angles and which side of the edge the poly's
// interior is on.
fn process_ring(
    coords: impl Iterator<Item = impl CoordTrait<T = f64>>,
    is_exterior: bool,
    edges: &mut Vec<Edge>,
    poly_ids: &mut Vec<usize>,
    poly_idx: usize,
) {
    let mut entries: Vec<(Point, Dir, f64)> = Vec::new();
    let mut signed_area = 0.0f64;

    let mut push_window = |a: Point, b: Point, c: Point| -> (Dir, Point) {
        let dir_ab = b.diff(&a);
        let dir_bc = c.diff(&b);
        signed_area += a.x * b.y - b.x * a.y;
        entries.push((a, dir_ab, dir_ab.turn_to(&dir_bc)));
        (dir_bc, b)
    };

    let mut coords = coords.map(|c| Point { x: c.x(), y: c.y() });
    let Some(c0) = coords.next() else {
        return;
    };
    let mut windows = std::iter::once(c0)
        .chain(coords)
        .tuple_windows::<(_, _, _)>();
    let (a, b, c) = windows.next().expect("Degenerate polyon ring");
    let first_ab_dir = b.diff(&a);
    let (mut last_bc_dir, mut last_b) = push_window(a, b, c);

    for (a, b, c) in windows {
        (last_bc_dir, last_b) = push_window(a, b, c);
    }

    // This part here is why c0 and the first dir need to be extracted
    // before getting into the iteration on windows, because the last edge
    // connects to the first edge.
    signed_area += last_b.x * c0.y - c0.x * last_b.y;
    entries.push((last_b, last_bc_dir, last_bc_dir.turn_to(&first_ab_dir)));

    // The below logic checks which direction the ring is going in
    // (ccw or cw), and determines which side the interior is on,
    // depending of whether this is an exterior ring or an interior
    // ring (i.e. a "hole").
    assert!(entries.len() >= 3, "Degenerate polygon ring");
    let ring_is_ccw = signed_area > 0.0;
    let interior_on_left = if is_exterior {
        ring_is_ccw
    } else {
        !ring_is_ccw
    };

    for (p1, dir, raw_turn) in entries {
        edges.push(dir.into_edge(p1, raw_turn, interior_on_left));
        poly_ids.push(poly_idx);
    }
}

fn build_poly_geo<P: PolygonTrait<T = f64>>(polygons: impl Iterator<Item = P>) -> GeoOwned {
    let mut edges: Vec<Edge> = Vec::new();
    let mut poly_ids: Vec<usize> = Vec::new();
    let polys: Vec<P> = polygons.collect();
    for (poly_idx, poly) in polys.iter().enumerate() {
        let Some(exterior) = poly.exterior() else {
            continue;
        };
        process_ring(exterior.coords(), true, &mut edges, &mut poly_ids, poly_idx);
    }
    let holes_start = edges.len();
    for (poly_idx, poly) in polys.iter().enumerate() {
        for hole in poly.interiors() {
            process_ring(hole.coords(), false, &mut edges, &mut poly_ids, poly_idx);
        }
    }
    GeoOwned::from_poly(edges, holes_start, poly_ids)
}

pub(crate) enum GeoView<'a> {
    Points {
        points: &'a [Point],
    },
    LineSegments {
        lines: &'a [LineSegment],
        b1s: &'a [bool],
        b2s: &'a [bool],
    },
    PolyEdges {
        edges: &'a [Edge],
        poly_ids: &'a [usize],
    },
}
