use geo_traits::{
    CoordTrait, GeometryTrait, GeometryType, LineStringTrait, MultiLineStringTrait,
    MultiPointTrait, MultiPolygonTrait, PointTrait, PolygonTrait,
};
use itertools::Itertools;
use wkb::reader::Wkb;

pub(crate) const EPS: f64 = 1e-10;

// Cheap geometry-validity errors caught at construction (not full DE-9IM validation — just the
// checks that prevent later misbehavior like divide-by-zero). Kept free of any DataFusion
// dependency; convert at the boundary via `DataFusionError::External`.
#[derive(Debug)]
pub(crate) enum GeoError {
    // The WKB bytes could not be parsed.
    MalformedWkb(String),
    // A segment/edge with identical endpoints (zero length).
    DegenerateSegment,
    // A polygon ring whose first and last vertices differ (or that produced no edges).
    UnclosedRing,
}

impl std::fmt::Display for GeoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeoError::MalformedWkb(msg) => write!(f, "malformed WKB: {msg}"),
            GeoError::DegenerateSegment => write!(f, "degenerate segment (identical endpoints)"),
            GeoError::UnclosedRing => write!(f, "polygon ring is not closed"),
        }
    }
}

impl std::error::Error for GeoError {}

//*************************************************
// Base component that all geometries are built from.
// all the basic checks are implemented here.
//*************************************************
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
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct Point {
    x: f64,
    y: f64,
}

impl Point {
    pub(crate) fn x(&self) -> f64 {
        self.x
    }

    pub(crate) fn is_equal(&self, other: &Point) -> bool {
        self.x == other.x && self.y == other.y
    }

    pub(crate) fn diff(&self, other: &Point) -> Dir {
        Dir {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct LineSegment {
    p1: Point,
    p2: Point,
    p1_boundary: bool,
    p2_boundary: bool,
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct Edge {
    p1: Point,
    p2: Point,
    /// Index (in the geometry's edge vector) of the next edge in this ring; wraps to the
    /// ring's first edge at the ring end. Used to recompute the interior angle at this
    /// edge's end vertex on demand instead of precomputing it at parse time.
    next: usize,
    interior_on_left: bool,
}

// Endpoints of a segment-like component. (Renamed from geo_owned's `ToPoints`; `xs`/`ys` are
// dropped for now.)
pub(crate) trait AsPoints {
    fn as_points(&self) -> (&Point, &Point);
}

impl AsPoints for LineSegment {
    fn as_points(&self) -> (&Point, &Point) {
        (&self.p1, &self.p2)
    }
}

impl AsPoints for Edge {
    fn as_points(&self) -> (&Point, &Point) {
        (&self.p1, &self.p2)
    }
}

// Geometric primitives shared by any segment-like component. No `ToDir`: a component's direction
// is derived on demand from its endpoints, and `cross`/`dot` take a `Dir` directly (the value
// `is_collinear_with`/`projection_t` build from `point.diff`).
pub(crate) trait SegmentTrait: AsPoints {
    fn dir(&self) -> Dir {
        let (p1, p2) = self.as_points();
        p2.diff(p1)
    }

    fn norm_sq(&self) -> f64 {
        let d = self.dir();
        d.dot(&d)
    }

    fn cross(&self, other: &Dir) -> f64 {
        self.dir().cross(other)
    }

    fn dot(&self, other: &Dir) -> f64 {
        self.dir().dot(other)
    }

    fn is_collinear_with(&self, p: &Point) -> bool {
        let (p1, _) = self.as_points();
        let dp = p.diff(p1);
        let c = self.cross(&dp);
        c * c < EPS * EPS * self.norm_sq()
    }

    fn projection_t(&self, p: &Point) -> f64 {
        let (p1, _) = self.as_points();
        let dp = p.diff(p1);
        self.dot(&dp) / self.norm_sq()
    }

    // x of the intersection between the segment's supporting line and the horizontal line
    // through `p`. Only meaningful when `p.y` is within the segment's y-span.
    fn x_intercept_at_point(&self, p: &Point) -> f64 {
        let (p1, p2) = self.as_points();
        p1.x + (p.y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y)
    }

    fn midpoint(&self) -> Point {
        let (p1, p2) = self.as_points();
        Point {
            x: (p1.x + p2.x) / 2.0,
            y: (p1.y + p2.y) / 2.0,
        }
    }
}

impl<T: AsPoints> SegmentTrait for T {}

impl Edge {
    pub(crate) fn interior_on_left(&self) -> bool {
        self.interior_on_left
    }

    // Index of the next edge in this ring (in the geometry's global edge vector).
    pub(crate) fn next(&self) -> usize {
        self.next
    }

    // Interior angle of the polygon at this edge's end vertex (p2), from this edge's direction
    // and the next edge's direction.
    fn vertex_angle(&self, next: &Edge) -> f64 {
        let raw_turn = self.dir().turn_to(&next.dir());
        if self.interior_on_left {
            std::f64::consts::PI - raw_turn
        } else {
            std::f64::consts::PI + raw_turn
        }
    }

    // Whether this edge's end vertex (p2) is reflex (interior angle > π). For a segment running
    // collinearly along the boundary into this vertex, the straight continuation past it (the
    // "peel") lands in the interior iff the vertex is reflex, exterior iff convex.
    pub(crate) fn is_reflex(&self, next: &Edge) -> bool {
        self.vertex_angle(next) > std::f64::consts::PI
    }

    // Core wedge test at this edge's end vertex (p2, interior wedge formed with `next`): for a
    // line through the vertex in direction `seg_dir`, whether the forward ray (`seg_dir`) and
    // its opposite each lie strictly inside the interior wedge. Shared by the crossing test and
    // the midpoint-on-vertex inside test.
    fn wedge_membership(&self, seg_dir: &Dir, next: &Edge) -> (bool, bool) {
        let angle = self.vertex_angle(next);
        let raw_turn = self.dir().turn_to(seg_dir);
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
        let seg_in_wedge = seg_angle > EPS && seg_angle < angle - EPS;
        let flipped_in_wedge = flipped > EPS && flipped < angle - EPS;
        (seg_in_wedge, flipped_in_wedge)
    }

    // Decides whether a ray/segment in direction `seg_dir` properly crosses the polygon boundary
    // at this edge's end vertex (p2, whose interior wedge is formed with `next`). `Some(_)` = a
    // proper crossing (the bool is the entering side); `None` = tangent / not a crossing.
    pub(crate) fn crossing_at_vertex(&self, seg_dir: &Dir, next: &Edge) -> Option<bool> {
        let (seg_in_wedge, flipped_in_wedge) = self.wedge_membership(seg_dir, next);
        if seg_in_wedge != flipped_in_wedge {
            Some(seg_in_wedge)
        } else {
            None
        }
    }

    // Whether a line passing through this edge's end vertex (p2) in direction `seg_dir` lies
    // *inside* the polygon there — i.e. both the ray and its opposite fall within the interior
    // wedge. A crossing (one in, one out) or a fully-exterior pass both return false, so the
    // caller treats them identically (invalidate parity, do nothing else).
    pub(crate) fn segment_inside_at_vertex(&self, seg_dir: &Dir, next: &Edge) -> bool {
        let (seg_in_wedge, flipped_in_wedge) = self.wedge_membership(seg_dir, next);
        seg_in_wedge && flipped_in_wedge
    }
}

//*************************************************
// Natural hierarchical bbox index.
//
// A flat tree of aggregate bounding boxes over a geometry's ordered components (polygon edges or
// line segments). Only the *internal* levels are stored; leaf boxes are derived from the
// components on the fly during descent (at the leaf level you call `ray_candidate` against the
// `Edge`/`LineSegment` directly, since those are `Bboxable` too). Levels are stored bottom-up:
// the finest internal level (parents of leaves) first, the root level last. The layout is
// implicit/positional — a node at (level, i) has children at (level+1, i*INDEX_FANOUT ..).
// `geo_id_offset` is added to leaf positions so a holes-only index (built over
// `edges[holes_start..]`) still emits *global* edge indices, keeping `Edge.next` resolvable
// through a single uniform descent.
//*************************************************
const INDEX_FANOUT: usize = 16; // tg's default index spread; tunable.

#[derive(Copy, Clone, Debug)]
pub(crate) struct IndexBox {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

// Implemented by both internal index nodes (`IndexBox`) and leaf components (`Edge`,
// `LineSegment`), so index construction (parent = union of children) and descent
// (`query.ray_candidate(node)`) are uniform and fully monomorphized (no `dyn`).
pub(crate) trait Bboxable {
    fn bbox(&self) -> IndexBox;

    // Rightward-ray descent test: `self` is the casting component/node, `other` a candidate
    // node. True when their y-bands overlap and `other` is not entirely left of `self`, so a
    // +x ray from `self` could reach it. Degenerates to the point ray test for point boxes.
    // Comparisons are slackened by `EPS` so the prune is never tighter than the EPS-tolerant
    // exact tests downstream (a candidate within tolerance must not be pruned away).
    fn ray_candidate(&self, other: &impl Bboxable) -> bool {
        let a = self.bbox();
        let b = other.bbox();
        a.min_y <= b.max_y + EPS && a.max_y >= b.min_y - EPS && b.max_x >= a.min_x - EPS
    }

    // Symmetric bbox overlap (EPS-slackened). The prune for order-independent checks (line
    // coverage, segment connectivity), where the rightward-ray test would wrongly admit
    // everything to the right.
    fn box_overlap(&self, other: &impl Bboxable) -> bool {
        let a = self.bbox();
        let b = other.bbox();
        a.min_x <= b.max_x + EPS
            && a.max_x >= b.min_x - EPS
            && a.min_y <= b.max_y + EPS
            && a.max_y >= b.min_y - EPS
    }
}

impl Bboxable for IndexBox {
    fn bbox(&self) -> IndexBox {
        *self
    }
}

impl Bboxable for Point {
    fn bbox(&self) -> IndexBox {
        IndexBox {
            min_x: self.x,
            min_y: self.y,
            max_x: self.x,
            max_y: self.y,
        }
    }
}

impl Bboxable for LineSegment {
    fn bbox(&self) -> IndexBox {
        IndexBox {
            min_x: self.p1.x.min(self.p2.x),
            min_y: self.p1.y.min(self.p2.y),
            max_x: self.p1.x.max(self.p2.x),
            max_y: self.p1.y.max(self.p2.y),
        }
    }
}

impl Bboxable for Edge {
    fn bbox(&self) -> IndexBox {
        IndexBox {
            min_x: self.p1.x.min(self.p2.x),
            min_y: self.p1.y.min(self.p2.y),
            max_x: self.p1.x.max(self.p2.x),
            max_y: self.p1.y.max(self.p2.y),
        }
    }
}

// Bounding box covering every component in the slice. Used over `&[Edge]` for the finest
// internal level and over `&[IndexBox]` for higher levels.
fn union(items: &[impl Bboxable]) -> IndexBox {
    let mut it = items.iter();
    let first = it.next().expect("union of empty slice").bbox();
    it.fold(first, |acc, item| {
        let b = item.bbox();
        IndexBox {
            min_x: acc.min_x.min(b.min_x),
            min_y: acc.min_y.min(b.min_y),
            max_x: acc.max_x.max(b.max_x),
            max_y: acc.max_y.max(b.max_y),
        }
    })
}

pub(crate) struct NaturalIndex {
    // Internal aggregate levels only, concatenated bottom-up (finest internal level first, root
    // last). Empty for a degenerate geometry (<= 1 component; nothing to prune).
    boxes: Vec<IndexBox>,
    // Start offset of each internal level in `boxes`, finest level first. `len()` is the number
    // of levels; the root is the last entry. Empty when `boxes` is empty. Precomputed traversal
    // metadata so the descent never has to recompute the tree shape.
    level_bases: Vec<usize>,
    geo_id_offset: usize,
}

impl NaturalIndex {
    fn from_components<C: Bboxable>(comps: &[C], geo_id_offset: usize) -> NaturalIndex {
        let n_leaves = comps.len();
        let mut boxes: Vec<IndexBox> = Vec::new();
        let mut level_bases: Vec<usize> = Vec::new();

        if n_leaves > 1 {
            // Finest internal level (starts at 0): union groups of up to INDEX_FANOUT leaves.
            level_bases.push(0);
            let level1_count = n_leaves.div_ceil(INDEX_FANOUT);
            for p in 0..level1_count {
                let lo = p * INDEX_FANOUT;
                let hi = (lo + INDEX_FANOUT).min(n_leaves);
                boxes.push(union(&comps[lo..hi]));
            }

            // Higher levels: union groups of the previous (already-stored) internal level.
            let mut level_start = 0;
            let mut level_len = level1_count;
            while level_len > 1 {
                let parents_start = boxes.len();
                level_bases.push(parents_start);
                let parent_count = level_len.div_ceil(INDEX_FANOUT);
                for p in 0..parent_count {
                    let lo = level_start + p * INDEX_FANOUT;
                    let hi = (lo + INDEX_FANOUT).min(level_start + level_len);
                    let parent = union(&boxes[lo..hi]);
                    boxes.push(parent);
                }
                level_start = parents_start;
                level_len = parent_count;
            }
        }

        NaturalIndex {
            boxes,
            level_bases,
            geo_id_offset,
        }
    }
}

//*************************************************
// A represenation of a geometry as its components
//*************************************************
pub(crate) enum JoinableGeo {
    Point {
        points: Vec<Point>,
    },
    Line {
        lines: Vec<LineSegment>,
        index: NaturalIndex,
    },
    Poly {
        edges: Vec<Edge>,
        poly_ids: Vec<usize>,
        // Index over all edges (exteriors + holes), geo_id_offset 0.
        index: NaturalIndex,
        // Index over hole edges only, geo_id_offset = the first hole edge's global index.
        holes_index: NaturalIndex,
    },
}

impl JoinableGeo {
    pub(crate) fn from_wkb(bytes: &[u8]) -> Result<Self, GeoError> {
        let wkb = Wkb::try_new(bytes).map_err(|e| GeoError::MalformedWkb(e.to_string()))?;
        match wkb.as_type() {
            GeometryType::Point(p) => {
                let mut points = Vec::new();
                if let Some(coord) = p.coord() {
                    points.push(Point {
                        x: coord.x(),
                        y: coord.y(),
                    });
                }
                Ok(JoinableGeo::Point { points })
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
                Ok(JoinableGeo::Point { points })
            }
            GeometryType::LineString(ls) => build_line_geo(std::iter::once(ls)),
            GeometryType::MultiLineString(mls) => build_line_geo(mls.line_strings()),
            GeometryType::Polygon(poly) => build_poly_geo(std::iter::once(poly)),
            GeometryType::MultiPolygon(mp) => build_poly_geo(mp.polygons()),
            _ => todo!("from_wkb: unsupported geometry type"),
        }
    }
}

fn build_line_geo<LS: LineStringTrait<T = f64>>(
    linestrings: impl Iterator<Item = LS>,
) -> Result<JoinableGeo, GeoError> {
    let mut lines: Vec<LineSegment> = Vec::new();

    for ls in linestrings {
        // A valid WKB linestring has >= 2 coordinates; we only guard against degenerate
        // (zero-length) segments, which would later divide by zero. Boundary flags are dummies
        // here — the real mod-2 values are computed by the self-join below.
        for (c1, c2) in ls
            .coords()
            .map(|c| Point { x: c.x(), y: c.y() })
            .tuple_windows::<(_, _)>()
        {
            if c1.is_equal(&c2) {
                return Err(GeoError::DegenerateSegment);
            }
            lines.push(LineSegment {
                p1: c1,
                p2: c2,
                p1_boundary: false,
                p2_boundary: false,
            });
        }
    }

    let index = NaturalIndex::from_components(&lines, 0);

    // Compute the mod-2 endpoint boundary flags via a cheap, faithful self-join over the segment
    // *endpoints* as points. Each segment contributes its two endpoints, laid out so endpoint
    // `2*i` is segment `i`'s `p1` and `2*i + 1` is its `p2`; the tag is implicit in that layout.
    // The point boxes are degenerate, so the EPS-slack `box_overlap` prune admits only points
    // within tolerance — candidates collapse to ≈ the real coincidences (no fat-box blow-up). A
    // `flags` entry is `true` (boundary) unless an odd number of coincident endpoints cancel it.
    let mut endpoints: Vec<Point> = Vec::with_capacity(lines.len() * 2);
    for seg in &lines {
        endpoints.push(seg.p1);
        endpoints.push(seg.p2);
    }
    let flags = {
        let point_index = NaturalIndex::from_components(&endpoints, 0);
        let mut acc = PointCoincidence::new(&endpoints);
        run_dual(&endpoints, &point_index, &endpoints, &point_index, &mut acc);
        acc.flags
    };
    for (i, seg) in lines.iter_mut().enumerate() {
        seg.p1_boundary = flags[2 * i];
        seg.p2_boundary = flags[2 * i + 1];
    }

    Ok(JoinableGeo::Line { lines, index })
}

// Accumulator for the construction-time endpoint self-join that computes the mod-2 boundary
// flags. Never ready (full traversal); `finish` is unused. Each endpoint's flag starts `true`
// (a boundary point) and flips once per coincident neighbor, so the final flag is `true ^ (k-1)`
// for a coincidence cluster of size `k` — i.e. boundary iff `k` is odd (mod-2 vertex degree).
// This is uniform across single/multi linestrings, lollipops (`k=3` → boundary), and closed rings
// (`k=2` ends → not boundary), with no special-casing.
struct PointCoincidence<'a> {
    points: &'a [Point],
    flags: Vec<bool>,
}

impl<'a> PointCoincidence<'a> {
    fn new(points: &'a [Point]) -> Self {
        PointCoincidence {
            points,
            flags: vec![true; points.len()],
        }
    }
}

impl Accumulator for PointCoincidence<'_> {
    // Symmetric EPS-slack bbox-overlap. For the degenerate point boxes this admits only points
    // within EPS (L∞), so the descent does work proportional to real coincidences, not to fat
    // overlapping segment boxes.
    fn prune(a: &impl Bboxable, b: &impl Bboxable) -> bool {
        a.box_overlap(b)
    }

    fn update(&mut self, li: usize, ri: usize) {
        // A point never matches itself. The symmetric dual traversal hands each point its turn as
        // `li` against every coincident neighbor, so flipping only `flags[li]` flips each endpoint
        // exactly once per coincident neighbor — both members of a pair end up correct, no dedup.
        if li == ri {
            return;
        }
        if self.points[li].is_equal(&self.points[ri]) {
            self.flags[li] ^= true;
        }
    }

    fn ready(&self) -> bool {
        false
    }

    fn finish(self) -> bool {
        false
    }
}

// Handles one ring from one poly, appending its directed edges. Each edge records the index of
// its in-ring successor (`next`), patched at the ring end to wrap back to the ring's first edge,
// so the interior angle at a vertex can be recomputed on demand from an edge and its `next`.
fn process_ring(
    coords: impl Iterator<Item = impl CoordTrait<T = f64>>,
    is_exterior: bool,
    edges: &mut Vec<Edge>,
    poly_ids: &mut Vec<usize>,
    poly_idx: usize,
) -> Result<(), GeoError> {
    let ring_start = edges.len();
    let mut signed_area = 0.0f64;

    for (a, b) in coords
        .map(|c| Point { x: c.x(), y: c.y() })
        .tuple_windows::<(_, _)>()
    {
        if a.is_equal(&b) {
            return Err(GeoError::DegenerateSegment);
        }
        signed_area += a.x * b.y - b.x * a.y;
        let idx = edges.len();
        edges.push(Edge {
            p1: a,
            p2: b,
            next: idx + 1,
            interior_on_left: false, // patched below once orientation is known
        });
        poly_ids.push(poly_idx);
    }

    // A closed ring (first vertex == last vertex) produces at least one edge ending where the
    // first edge starts. No edges, or a mismatched end, means it isn't a valid closed ring.
    // (Combined with the degenerate-segment check, this also rules out too-few-edge rings.)
    if edges.len() == ring_start || !edges[edges.len() - 1].p2.is_equal(&edges[ring_start].p1) {
        return Err(GeoError::UnclosedRing);
    }
    let last = edges.len() - 1;
    edges[last].next = ring_start; // wrap the last edge back to the ring's first edge

    // Determine ring orientation, then which side the polygon interior is on.
    let ring_is_ccw = signed_area > 0.0;
    let interior_on_left = if is_exterior {
        ring_is_ccw
    } else {
        !ring_is_ccw
    };
    for edge in &mut edges[ring_start..] {
        edge.interior_on_left = interior_on_left;
    }
    Ok(())
}

fn build_poly_geo<P: PolygonTrait<T = f64>>(
    polygons: impl Iterator<Item = P>,
) -> Result<JoinableGeo, GeoError> {
    let mut edges: Vec<Edge> = Vec::new();
    let mut poly_ids: Vec<usize> = Vec::new();
    let polys: Vec<P> = polygons.collect();

    // Exterior rings first, then holes — so `[0..holes_start)` are exterior edges and
    // `[holes_start..)` are hole edges, addressable as one global edge vector.
    for (poly_idx, poly) in polys.iter().enumerate() {
        let Some(exterior) = poly.exterior() else {
            continue;
        };
        process_ring(exterior.coords(), true, &mut edges, &mut poly_ids, poly_idx)?;
    }
    let holes_start = edges.len();
    for (poly_idx, poly) in polys.iter().enumerate() {
        for hole in poly.interiors() {
            process_ring(hole.coords(), false, &mut edges, &mut poly_ids, poly_idx)?;
        }
    }

    let index = NaturalIndex::from_components(&edges, 0);
    let holes_index = NaturalIndex::from_components(&edges[holes_start..], holes_start);
    Ok(JoinableGeo::Poly {
        edges,
        poly_ids,
        index,
        holes_index,
    })
}

// Fuses candidate generation with predicate evaluation: a descent calls `update(li, ri)` at each
// surviving leaf pair, with global component ids in the accumulator's (left, right) orientation.
// The accumulator captures the concrete component slices it needs up front, so `update` only
// indexes them — it never touches the `JoinableGeo` enum.
pub(crate) trait Accumulator {
    // Conservative box filter that drives index pruning: true means the descent must look
    // inside `b` for a possible match against `a`. Default is the rightward-ray test, correct
    // for ray-cast / point-in-polygon checks; crossing checks override it with a symmetric
    // bbox-overlap test (a ray-cast prune would wrongly discard candidates to the left).
    fn prune(a: &impl Bboxable, b: &impl Bboxable) -> bool {
        a.ray_candidate(b)
    }

    // Folds one candidate component pair into the running state.
    fn update(&mut self, li: usize, ri: usize);

    // Checked right after each `update`; true ends the descent early.
    fn ready(&self) -> bool;

    // The verdict, once the descent has finished (or early-exited).
    fn finish(self) -> bool;
}

// Single-descent kernel: iterate the left components, and for each descend the right index,
// folding every surviving leaf pair into the accumulator (left ids arrive grouped/contiguous).
// `right_index` is `None` for a Point side (no index), in which case — like an index with no
// internal levels — we linear-scan the right leaves. The leaves themselves are passed
// separately because `NaturalIndex` only stores aggregate boxes, not the components.
fn single_descend<LC: Bboxable, RC: Bboxable, A: Accumulator>(
    left: &[LC],
    right_leaves: &[RC],
    right_index: Option<&NaturalIndex>,
    acc: &mut A,
) {
    // Descend when there's an index with internal levels.
    if let Some(idx) = right_index {
        if !idx.boxes.is_empty() {
            let root_level = idx.level_bases.len() - 1;
            for li in 0..left.len() {
                if descend_node::<LC, RC, A>(
                    &left[li],
                    li,
                    right_leaves,
                    &idx.boxes,
                    &idx.level_bases,
                    idx.geo_id_offset,
                    root_level,
                    0,
                    acc,
                ) {
                    return;
                }
            }
            return;
        }
    }

    // Otherwise — no index, or a degenerate one with no internal levels (a Point side, a
    // ≤1-leaf geometry, or a holes index with no hole edges) — scan the right leaves.
    let offset = right_index.map_or(0, |idx| idx.geo_id_offset);
    for li in 0..left.len() {
        if scan_leaves::<LC, RC, A>(
            &left[li],
            li,
            right_leaves,
            0,
            right_leaves.len(),
            offset,
            acc,
        ) {
            return;
        }
    }
}

// Tests one left query against a contiguous run of right leaves; folds survivors. Returns true
// if the accumulator asked to stop.
fn scan_leaves<LC: Bboxable, RC: Bboxable, A: Accumulator>(
    query: &LC,
    li: usize,
    right_leaves: &[RC],
    lo: usize,
    hi: usize,
    right_offset: usize,
    acc: &mut A,
) -> bool {
    for ri in lo..hi {
        if A::prune(query, &right_leaves[ri]) {
            acc.update(li, right_offset + ri);
            if acc.ready() {
                return true;
            }
        }
    }
    false
}

// Recursively descends one internal node of the right index for a single left query, pruning
// subtrees and bottoming out into `scan_leaves` at the finest level. Returns true to stop.
#[allow(clippy::too_many_arguments)]
fn descend_node<LC: Bboxable, RC: Bboxable, A: Accumulator>(
    query: &LC,
    li: usize,
    right_leaves: &[RC],
    right_boxes: &[IndexBox],
    bases: &[usize],
    right_offset: usize,
    level: usize,
    pos: usize,
    acc: &mut A,
) -> bool {
    if !A::prune(query, &right_boxes[bases[level] + pos]) {
        return false;
    }
    let lo = pos * INDEX_FANOUT;
    if level == 0 {
        // Children are leaf components.
        let hi = (lo + INDEX_FANOUT).min(right_leaves.len());
        scan_leaves::<LC, RC, A>(query, li, right_leaves, lo, hi, right_offset, acc)
    } else {
        // Levels are contiguous in `boxes`, so the child level's node count is the gap between
        // its base and this level's base.
        let child_level_size = bases[level] - bases[level - 1];
        let hi = (lo + INDEX_FANOUT).min(child_level_size);
        for c in lo..hi {
            if descend_node::<LC, RC, A>(
                query,
                li,
                right_leaves,
                right_boxes,
                bases,
                right_offset,
                level - 1,
                c,
                acc,
            ) {
                return true;
            }
        }
        false
    }
}

// Dual-descent kernel: both sides indexed, descended in lockstep so they bottom out together.
// For order-independent ("any hit") checks whose prune is symmetric. If either side has no
// internal levels (a degenerate ≤1-component geometry), a plain single descent is correct and
// keeps the left/right orientation, so we fall back to it.
fn run_dual<AC: Bboxable, BC: Bboxable, A: Accumulator>(
    a_leaves: &[AC],
    a_index: &NaturalIndex,
    b_leaves: &[BC],
    b_index: &NaturalIndex,
    acc: &mut A,
) {
    if a_index.boxes.is_empty() || b_index.boxes.is_empty() {
        single_descend(a_leaves, b_leaves, Some(b_index), acc);
        return;
    }
    dual_recurse::<AC, BC, A>(
        a_leaves,
        &a_index.boxes,
        &a_index.level_bases,
        a_index.geo_id_offset,
        b_leaves,
        &b_index.boxes,
        &b_index.level_bases,
        b_index.geo_id_offset,
        a_index.level_bases.len() - 1,
        0,
        b_index.level_bases.len() - 1,
        0,
        acc,
    );
}

// `level`/`pos` count internal levels with 0 = the leaf-parent level (its children are leaves).
// "More levels left" = higher `level`. Prune the node pair, then: both at the leaf-parent level
// → pair leaves via `scan_leaves`; equal level above that → cartesian over child pairs; unequal
// → descend only the deeper side so the levels re-sync. Returns true to stop (early exit).
#[allow(clippy::too_many_arguments)]
fn dual_recurse<AC: Bboxable, BC: Bboxable, A: Accumulator>(
    a_leaves: &[AC],
    a_boxes: &[IndexBox],
    a_bases: &[usize],
    a_offset: usize,
    b_leaves: &[BC],
    b_boxes: &[IndexBox],
    b_bases: &[usize],
    b_offset: usize,
    a_level: usize,
    a_pos: usize,
    b_level: usize,
    b_pos: usize,
    acc: &mut A,
) -> bool {
    let a_box = &a_boxes[a_bases[a_level] + a_pos];
    let b_box = &b_boxes[b_bases[b_level] + b_pos];
    if !A::prune(a_box, b_box) {
        return false;
    }

    if a_level == 0 && b_level == 0 {
        // Both at the leaf-parent level: pair a's leaf range against b's leaf range.
        let a_lo = a_pos * INDEX_FANOUT;
        let a_hi = (a_lo + INDEX_FANOUT).min(a_leaves.len());
        let b_lo = b_pos * INDEX_FANOUT;
        let b_hi = (b_lo + INDEX_FANOUT).min(b_leaves.len());
        for ai in a_lo..a_hi {
            if scan_leaves::<AC, BC, A>(
                &a_leaves[ai],
                a_offset + ai,
                b_leaves,
                b_lo,
                b_hi,
                b_offset,
                acc,
            ) {
                return true;
            }
        }
        return false;
    }

    // Child position ranges for whichever side(s) we descend. A level's child count is the
    // contiguous gap to the next-finer level's base.
    let a_lo = a_pos * INDEX_FANOUT;
    let a_hi = (a_lo + INDEX_FANOUT).min(if a_level > 0 {
        a_bases[a_level] - a_bases[a_level - 1]
    } else {
        a_leaves.len()
    });
    let b_lo = b_pos * INDEX_FANOUT;
    let b_hi = (b_lo + INDEX_FANOUT).min(if b_level > 0 {
        b_bases[b_level] - b_bases[b_level - 1]
    } else {
        b_leaves.len()
    });

    if a_level > b_level {
        // Descend only A; keep B's node fixed until the levels re-sync.
        for ac in a_lo..a_hi {
            if dual_recurse::<AC, BC, A>(
                a_leaves,
                a_boxes,
                a_bases,
                a_offset,
                b_leaves,
                b_boxes,
                b_bases,
                b_offset,
                a_level - 1,
                ac,
                b_level,
                b_pos,
                acc,
            ) {
                return true;
            }
        }
    } else if b_level > a_level {
        // Descend only B.
        for bc in b_lo..b_hi {
            if dual_recurse::<AC, BC, A>(
                a_leaves,
                a_boxes,
                a_bases,
                a_offset,
                b_leaves,
                b_boxes,
                b_bases,
                b_offset,
                a_level,
                a_pos,
                b_level - 1,
                bc,
                acc,
            ) {
                return true;
            }
        }
    } else {
        // Equal level (> 0): descend both — cartesian over child pairs.
        for ac in a_lo..a_hi {
            for bc in b_lo..b_hi {
                if dual_recurse::<AC, BC, A>(
                    a_leaves,
                    a_boxes,
                    a_bases,
                    a_offset,
                    b_leaves,
                    b_boxes,
                    b_bases,
                    b_offset,
                    a_level - 1,
                    ac,
                    b_level - 1,
                    bc,
                    acc,
                ) {
                    return true;
                }
            }
        }
    }
    false
}

//*************************************************
// Predicate-facing traversal dispatch. The caller picks the method by what the *check* needs —
// `fold_single` when it needs `self`'s components grouped (containment/parity), `fold_dual` for
// order-independent "any hit" checks — and never has to reason about whether a side is a Point.
// The accumulator (built by the caller for these geo types) is fed `update(self_id, other_id)`.
//*************************************************
impl JoinableGeo {
    // Grouped descent: iterate `self`'s components, descend (or scan) `other`. `self`'s ids
    // arrive contiguous, so accumulating checks can finalize per group and early-exit.
    pub(crate) fn fold_for_grouped_check(&self, other: &JoinableGeo, acc: &mut impl Accumulator) {
        match self {
            JoinableGeo::Point { points } => single_into(points, other, acc),
            JoinableGeo::Line { lines, .. } => single_into(lines, other, acc),
            JoinableGeo::Poly { edges, .. } => single_into(edges, other, acc),
        }
    }

    // Order-independent descent: lockstep dual descent when both sides are indexed, otherwise a
    // single descent (iterate `self`, descend/scan `other`) — same traversal a `fold_single`
    // call would produce, so a Point side needs no special handling at the call site.
    pub(crate) fn fold_for_unordered_check(&self, other: &JoinableGeo, acc: &mut impl Accumulator) {
        match self {
            JoinableGeo::Point { points } => dual_into(points, None, other, acc),
            JoinableGeo::Line { lines, index, .. } => dual_into(lines, Some(index), other, acc),
            JoinableGeo::Poly { edges, index, .. } => dual_into(edges, Some(index), other, acc),
        }
    }

    // Index into this poly's `edges` where the hole edges begin (exterior rings are stored first).
    // Equal to `edges.len()` when the poly has no holes. `fold_holes_into` delivers hole edges to an
    // accumulator with *hole-relative* left ids, so the accumulator adds this offset to recover the
    // edge in the full vector. Zero for non-poly geometries (which have no holes).
    pub(crate) fn holes_start(&self) -> usize {
        match self {
            JoinableGeo::Poly { holes_index, .. } => holes_index.geo_id_offset,
            _ => 0,
        }
    }

    // Grouped descent over **only this poly's hole edges** as the query side, descending `other`.
    // Holes occupy `edges[holes_index.geo_id_offset..]` (an empty slice when the poly has none, so
    // the descent is then a no-op). Poly-only: the caller (poly×poly within) is the sole user.
    pub(crate) fn fold_holes_into(&self, other: &JoinableGeo, acc: &mut impl Accumulator) {
        match self {
            JoinableGeo::Poly {
                edges, holes_index, ..
            } => single_into(&edges[holes_index.geo_id_offset..], other, acc),
            _ => unreachable!("fold_holes_into is poly-only"),
        }
    }
}

// Resolves `other` for a single descent: descend its index (Line/Poly) or scan it (Point).
fn single_into<LC: Bboxable, A: Accumulator>(left: &[LC], other: &JoinableGeo, acc: &mut A) {
    match other {
        JoinableGeo::Point { points } => single_descend(left, points, None, acc),
        JoinableGeo::Line { lines, index, .. } => single_descend(left, lines, Some(index), acc),
        JoinableGeo::Poly { edges, index, .. } => single_descend(left, edges, Some(index), acc),
    }
}

// Resolves `other` for an order-independent descent: dual descent when both sides are indexed,
// else fall back to a single descent (iterating `left`).
fn dual_into<LC: Bboxable, A: Accumulator>(
    left: &[LC],
    left_index: Option<&NaturalIndex>,
    other: &JoinableGeo,
    acc: &mut A,
) {
    match other {
        JoinableGeo::Point { points } => single_descend(left, points, None, acc),
        JoinableGeo::Line { lines, index, .. } => match left_index {
            Some(li) => run_dual(left, li, lines, index, acc),
            None => single_descend(left, lines, Some(index), acc),
        },
        JoinableGeo::Poly { edges, index, .. } => match left_index {
            Some(li) => run_dual(left, li, edges, index, acc),
            None => single_descend(left, edges, Some(index), acc),
        },
    }
}

//************************************************************
// Various helper functions for spatial predicate checks.
//************************************************************
pub(crate) fn point_on_segment_t(p: &Point, seg: &impl SegmentTrait) -> Option<f64> {
    if !seg.is_collinear_with(p) {
        return None;
    }
    let t = seg.projection_t(p);
    if (-EPS..=(1.0 + EPS)).contains(&t) {
        Some(t.clamp(0.0, 1.0))
    } else {
        None
    }
}

// The portion of `left` (in `left`'s parameter, clamped to `[0, 1]`) covered by `right`. Returns
// `None` when the two aren't collinear or the clamped overlap is empty. Collinearity is checked
// here rather than assumed: every `right` is a mere bbox candidate from the index, so it may cross
// `left` instead of lying along it; the gate keeps only the collinear (coverage-contributing) ones.
pub(crate) fn covered_range(left: &LineSegment, right: &LineSegment) -> Option<(f64, f64)> {
    if !segments_collinear(left, right) {
        return None;
    }
    let (r1, r2) = right.as_points();
    let ta = left.projection_t(r1);
    let tb = left.projection_t(r2);
    let t_lo = ta.min(tb).max(0.0);
    let t_hi = ta.max(tb).min(1.0);
    (t_lo <= t_hi + EPS).then_some((t_lo, t_hi))
}
// If `p` coincides with an endpoint of `seg`, returns `Some(is_boundary)` — whether that endpoint
// is a linestring boundary vertex (odd-degree under the mod-2 rule). Returns `None` otherwise.
// Boundary endpoints are not part of the line's interior, so a point landing on one is not within
// it (and, for a self-touching line, that boundary hit overrides any interior match elsewhere).
pub(crate) fn point_at_line_endpoint(p: &Point, seg: &LineSegment) -> Option<bool> {
    if p.is_equal(&seg.p1) {
        Some(seg.p1_boundary)
    } else if p.is_equal(&seg.p2) {
        Some(seg.p2_boundary)
    } else {
        None
    }
}

pub(crate) fn ray_crosses_edge(p: &Point, edge: &Edge, next: &Edge) -> bool {
    let (p1, p2) = edge.as_points();
    // Check the owned end vertex (p2) first: for a horizontal edge sitting on the ray
    // (`p1.y == p2.y == p.y`) this resolves its p2 vertex via the wedge test instead of bailing
    // on the p1 hand-off below.
    if p.y == p2.y {
        // End vertex: only a rightward crossing if the vertex is strictly to the right.
        if p2.x <= p.x {
            return false;
        }
        return edge.crossing_at_vertex(&p2.diff(p), next).is_some();
    }
    if p.y == p1.y {
        return false; // first endpoint: owned by the previous edge
    }
    edge.x_intercept_at_point(p) > p.x
}

// The ray-cast / midpoint classification of one left segment against one poly edge (`edge`,
// whose interior wedge is formed with `next`). Returns `(ray_crossed, inside_at_vertex,
// parity_valid)`, folded across the left segment's group by the caller:
//   * `ray_crossed` — the rightward ray from the segment's midpoint crosses `edge` (parity flip);
//   * `inside_at_vertex` — the midpoint sits on this edge's owned vertex and the segment lies
//     inside the polygon there (an independent "qualifying" signal, not a parity flip);
//   * `parity_valid` — false when the midpoint sits on the boundary (on the vertex or on the
//     edge body), since ray parity through a boundary point is unreliable.
// The midpoint-on-vertex check comes first (cheap exact match); a crossing or fully-exterior
// pass at the vertex both yield `inside_at_vertex = false`, so the caller treats them like
// "outside" — parity invalidated, nothing else.
pub(crate) fn midpoint_ray_check(
    left: &impl SegmentTrait,
    edge: &Edge,
    next: &Edge,
) -> (bool, bool, bool) {
    let mid = left.midpoint();
    let (_, p2) = edge.as_points();

    if mid.is_equal(p2) {
        let inside = edge.segment_inside_at_vertex(&left.dir(), next);
        return (false, inside, false);
    }
    if point_on_segment_t(&mid, edge).is_some() {
        return (false, false, false);
    }
    (ray_crosses_edge(&mid, edge, next), false, true)
}

// Classifies the intersection of one left segment with one poly edge (`edge`, whose interior
// wedge is formed with `next`). Returns `(qualifying, signed_t)`:
//   * `qualifying` — the segment touches the polygon interior here.
//   * `signed_t` — a recorded crossing that must cancel against co-located ones: the crossing's
//     parameter on the left segment, `+` entering / `−` exiting. `None` when nothing is recorded.
// No case is an unconditional disqualify — even a transversal edge-*body* crossing records a
// signed `t`, because a vertex of another polygon may touch this edge's interior and fill the far
// side (a valid multipolygon). A *genuine* exit to the exterior simply leaves the accumulator's
// `s_sum` unbalanced and fails there. Only this edge's owned vertex (`p2`) is the classified
// vertex; its `p1` belongs to the previous edge (exclude-start). A crossing only counts when it
// lies strictly inside the left segment.
pub(crate) fn segment_crossing_check(
    left: &LineSegment,
    edge: &Edge,
    next: &Edge,
) -> (bool, Option<f64>) {
    let (e1, e2) = edge.as_points(); // e2 is this edge's owned vertex
    let collinear_edge = left.is_collinear_with(e1) && left.is_collinear_with(e2);
    let (n1, n2) = next.as_points();
    let collinear_next = left.is_collinear_with(n1) && left.is_collinear_with(n2);

    if collinear_edge || collinear_next {
        // 180° pass-through (boundary continues straight through the vertex): ignore.
        if collinear_edge && collinear_next {
            return (false, None);
        }
        // The vertex must lie strictly within the left segment, else no crossing here (the edge
        // fully contains the segment, or they only touch outside (0, 1)).
        let t = left.projection_t(e2);
        if !(t > EPS && t < 1.0 - EPS) {
            return (false, None);
        }
        if edge.is_reflex(next) {
            // Peel into the interior: an interior contact, nothing to cancel.
            return (true, None);
        }
        // Convex peel into the exterior: record a signed crossing (not an interior contact).
        // Exit (−t) when the exterior peel sits on the segment's +t side. The peel is +edge.dir()
        // when the vertex is `edge`'s 2nd endpoint, or −next.dir() when it's `next`'s 1st.
        let exit = if collinear_edge {
            left.dir().dot(&edge.dir()) > 0.0
        } else {
            left.dir().dot(&next.dir()) < 0.0
        };
        return (false, Some(if exit { -t } else { t }));
    }

    // Non-collinear: solve the line intersection for `t` (on the left) and `u` (on the edge).
    let cross = left.dir().cross(&edge.dir());
    if cross.abs() < EPS {
        return (false, None); // parallel but not collinear → no intersection
    }
    let (a1, _) = left.as_points();
    let f = e1.diff(a1); // edge.p1 − left.p1
    let t = -edge.dir().cross(&f) / cross;
    let u = -left.dir().cross(&f) / cross;
    if !(t > EPS && t < 1.0 - EPS) {
        return (false, None);
    }
    if u > EPS && u < 1.0 - EPS {
        // Transversal crossing through the edge body: an interior contact, recorded so it can
        // cancel against another polygon's vertex touching this edge here; an unmatched one is a
        // genuine exit (caught by the unbalanced `s_sum`). Entering side from the orientation sign.
        let is_entering = (cross < 0.0) == edge.interior_on_left();
        return (true, Some(if is_entering { t } else { -t }));
    }
    if (1.0 - EPS..=1.0 + EPS).contains(&u) {
        // Crossing at the owned vertex (p2); exclude-start drops u ≈ 0 (the previous edge's).
        if let Some(is_entering) = edge.crossing_at_vertex(&left.dir(), next) {
            return (true, Some(if is_entering { t } else { -t }));
        }
    }
    (false, None)
}

fn segments_collinear(a: &LineSegment, b: &LineSegment) -> bool {
    let (q1, q2) = b.as_points();
    a.is_collinear_with(q1) && a.is_collinear_with(q2)
}

// How one left polygon edge relates to one right polygon edge (`edge`, interior wedge formed with
// `next`), for the poly×poly within check. Like `segment_crossing_check` but for an `Edge` left side
// and with no signed-`t` bookkeeping (a single container, nothing to cancel). Returns
// `(disqualifies, collinear_inside)`:
//   * `disqualifies` — a transversal/owned-vertex crossing, or a collinear overlap with the
//     interiors on opposite sides (adjacent disjoint polys): the left poly pokes outside here.
//   * `collinear_inside` — a collinear overlap with interiors on the *same* side: the left edge lies
//     on the right boundary, inside → qualifies (the caller takes the container from this edge).
// Both false = no contact here (disjoint, parallel, a graze, or contact only at an endpoint).
pub(crate) fn poly_edge_relation(left: &Edge, edge: &Edge, next: &Edge) -> (bool, bool) {
    let (e1, e2) = edge.as_points();
    if left.is_collinear_with(e1) && left.is_collinear_with(e2) {
        // Collinear: do they share more than a point (a real overlap along the boundary line)?
        let t1 = left.projection_t(e1);
        let t2 = left.projection_t(e2);
        let lo = t1.min(t2).max(0.0);
        let hi = t1.max(t2).min(1.0);
        if hi - lo <= EPS {
            return (false, false); // collinear but touching at most at a point
        }
        // `interior_on_left` is relative to each edge's own direction, so a flip is needed when the
        // edges run opposite ways.
        let same_dir = left.dir().dot(&edge.dir()) > 0.0;
        let same_side = (left.interior_on_left() == edge.interior_on_left()) == same_dir;
        return (!same_side, same_side);
    }

    // Non-collinear: solve the line intersection for `t` (on the left) and `u` (on the edge).
    let cross = left.dir().cross(&edge.dir());
    if cross.abs() < EPS {
        return (false, false); // parallel but not collinear
    }
    let (a1, _) = left.as_points();
    let f = e1.diff(a1);
    let t = -edge.dir().cross(&f) / cross;
    let u = -left.dir().cross(&f) / cross;
    if !(t > EPS && t < 1.0 - EPS) {
        return (false, false); // intersection not strictly inside the left edge
    }
    if u > EPS && u < 1.0 - EPS {
        return (true, false); // transversal crossing through the edge body
    }
    if (1.0 - EPS..=1.0 + EPS).contains(&u) && edge.crossing_at_vertex(&left.dir(), next).is_some() {
        // Enter/exit crossing at the edge's owned vertex (exclude-start drops u ≈ 0); a graze
        // (`crossing_at_vertex` → None) is ignored.
        return (true, false);
    }
    (false, false)
}
