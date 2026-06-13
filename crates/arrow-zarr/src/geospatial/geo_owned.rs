use std::time::Instant;

use geo_traits::{
    CoordTrait, GeometryTrait, GeometryType, LineStringTrait, MultiLineStringTrait,
    MultiPointTrait, MultiPolygonTrait, PointTrait, PolygonTrait,
};
use itertools::Itertools;
use wkb::reader::Wkb;

const EPS: f64 = 1e-10;
const N_STRIPES: usize = 64;
const INDEX_FANOUT: usize = 8;
// Max tree levels per ring block (INDEX_FANOUT^16 components is far beyond any real geometry).
const MAX_INDEX_LEVELS: usize = 16;

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
    /// Index (in the geometry's edge vector) of the next edge in this ring; wraps to the
    /// ring's first edge at the ring end. Used to recompute the interior angle at this
    /// edge's end vertex on demand instead of precomputing it at parse time.
    next: usize,
    interior_on_left: bool,
}

impl Edge {
    pub(crate) fn interior_on_left(&self) -> bool {
        self.interior_on_left
    }

    /// Interior angle of the polygon at this edge's end vertex, recomputed on demand from
    /// this edge's direction and the next edge's direction (replaces the precomputed `angle`).
    fn vertex_angle(&self, next: &Edge) -> f64 {
        let raw_turn = self.to_dir().turn_to(&next.to_dir());
        if self.interior_on_left {
            std::f64::consts::PI - raw_turn
        } else {
            std::f64::consts::PI + raw_turn
        }
    }

    pub(crate) fn is_reflex(&self, next: &Edge) -> bool {
        self.vertex_angle(next) > std::f64::consts::PI
    }

    pub(crate) fn crossing_at_vertex(&self, seg: &impl ToDir, next: &Edge) -> Option<bool> {
        let angle = self.vertex_angle(next);
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
        let seg_in_wedge = seg_angle > EPS && seg_angle < angle - EPS;
        let flipped_in_wedge = flipped > EPS && flipped < angle - EPS;
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

// Returns the t parameter in [0.0, 1.0] where the point lies on the segment,
// or None if it doesn't.
pub(crate) fn point_on_segment_t(p: &Point, seg: &impl SegmentTrait) -> Option<f64> {
    if !seg.x_range_contains_point(p) {
        return None;
    }
    let ns = seg.norm_sq();
    if ns < EPS * EPS {
        let (p1, _) = seg.to_points();
        return if p.is_equal(p1) { Some(0.0) } else { None };
    }
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

/// Result of testing whether a left segment's midpoint lands on a right polygon edge.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum MidpointResult {
    // midpoint lies collinearly on the right edge
    OnEdge,
    // midpoint is at a reflex end-vertex of the right edge → inside
    MidpointInside,
    // midpoint is at a convex end-vertex of the right edge → outside
    MidpointOutside,
    // midpoint not on right edge
    NoContact,
}

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
        Self::from_segments_range(segments, 0)
    }

    /// Builds a stripe index over `segments[start..]`, but stores each segment's **global**
    /// index (its position in the full `segments` slice). The y range is derived from the
    /// `[start..]` subset only, so a hole stripe can be tight to the holes' y-extent while
    /// its indices still address the full edge vector — keeping all poly views in one index
    /// space and `Edge::next` resolvable.
    fn from_segments_range(segments: &[impl ToPoints], start: usize) -> Self {
        let sub = &segments[start..];
        let n = sub.len();
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); N_STRIPES];

        if n == 0 {
            return YStripes {
                buckets,
                y_min: 0.0,
                stripe_h: 0.0,
            };
        }

        let (y_min, y_max) =
            sub.iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), seg| {
                    let (y1, y2) = seg.ys();
                    (lo.min(y1).min(y2), hi.max(y1).max(y2))
                });

        if y_max - y_min < EPS {
            for i in start..segments.len() {
                buckets[0].push(i);
            }
            return YStripes {
                buckets,
                y_min,
                stripe_h: 0.0,
            };
        }

        let stripe_h = (y_max - y_min) / N_STRIPES as f64;
        for (i, seg) in segments.iter().enumerate().skip(start) {
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
// Natural hierarchical bbox index (tg-style): a flat, per-group tree of bounding boxes over
// the ordered components (polygon ring edges, linestring segments, or points). Cheap O(N)
// build with no replication; queried by descending and pruning subtrees. Built but NOT yet
// wired into parsing or the predicate dispatch.
//*************************************************
#[derive(Clone, Copy)]
struct IndexBox {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

impl IndexBox {
    fn from_segment(seg: &impl ToPoints) -> Self {
        let (x1, x2) = seg.xs();
        let (y1, y2) = seg.ys();
        IndexBox {
            min_x: x1.min(x2),
            min_y: y1.min(y2),
            max_x: x1.max(x2),
            max_y: y1.max(y2),
        }
    }

    fn from_point(p: &Point) -> Self {
        IndexBox {
            min_x: p.x,
            min_y: p.y,
            max_x: p.x,
            max_y: p.y,
        }
    }

    // Parent box: the union of all its children, computed in one fold.
    fn union(boxes: &[IndexBox]) -> Self {
        let mut it = boxes.iter();
        let first = *it.next().expect("union of empty box slice");
        it.fold(first, |acc, b| IndexBox {
            min_x: acc.min_x.min(b.min_x),
            min_y: acc.min_y.min(b.min_y),
            max_x: acc.max_x.max(b.max_x),
            max_y: acc.max_y.max(b.max_y),
        })
    }

    // Descent test for a rightward ray cast from `self` (the left/casting component or node)
    // against `other` (a right-side candidate node): their y-bands overlap and `other` is not
    // entirely left of `self`, so a +x ray from `self` could reach it. Degenerates to the
    // point ray test when `self` is a point box.
    fn ray_candidate(&self, other: &IndexBox) -> bool {
        self.min_y <= other.max_y && self.max_y >= other.min_y && other.max_x >= self.min_x
    }
}

// One entry per component-group: a polygon ring, a linestring, or a points set.
struct Ring {
    leaf_start: usize, // index of this group's first leaf in `boxes` (front leaf region)
    n_edges: usize,
    internal_start: usize, // index of this group's first internal node in `boxes` (tail region)
}

pub(crate) struct NaturalIndex {
    // Contiguous layout: [ leaf boxes in component order ][ per-ring internal-node blocks ].
    // A leaf's index in `boxes` equals its component index, so a leaf's global geo-id is
    // `geo_id_offset + leaf_index`. Each ring's internal nodes (level 1 upward, root last)
    // live in the tail starting at `internal_start`.
    boxes: Vec<IndexBox>,
    rings: Vec<Ring>,
    geo_id_offset: usize, // global geo-id of component 0 (e.g. holes_start for a holes index)
}

impl NaturalIndex {
    fn num_components(&self) -> usize {
        self.rings.iter().map(|r| r.n_edges).sum()
    }

    // Candidate component pairs between `self` (left / ray-casting side) and `other` (right),
    // mirroring `YStripes::candidate_pairs`. Returns `(self_component, other_component)` in
    // global geo-ids; `flipped_output` swaps the tuple to `(other, self)`. To always iterate
    // the smaller side, when `other` has fewer components we recurse with the sides swapped
    // and the flip toggled — so the body only handles self-descends-into-other.
    pub(crate) fn candidate_pairs(
        &self,
        other: &NaturalIndex,
        flipped_output: bool,
    ) -> Vec<(usize, usize)> {
        if other.num_components() < self.num_components() {
            return other.candidate_pairs(self, !flipped_output);
        }

        let mut pairs = Vec::new();
        for sr in &self.rings {
            for k in 0..sr.n_edges {
                let self_geo = self.geo_id_offset + sr.leaf_start + k;
                let fb = self.boxes[sr.leaf_start + k]; // self leaf box (left/caster)
                for or in &other.rings {
                    other.descend_ring(or, &fb, self_geo, flipped_output, &mut pairs);
                }
            }
        }
        pairs
    }

    // Descends one ring's bbox tree top-down, pruning subtrees via `query.ray_candidate(node)`
    // and pushing a candidate pair for each surviving leaf. `self_geo` is the geo-id of the
    // querying (left) component; `flipped` swaps the emitted tuple order. Level 0 (leaves)
    // lives in the front leaf region at `ring.leaf_start..`; levels >= 1 live in the tail at
    // `ring.internal_start..`. Level sizes/bases go in fixed-size arrays (no heap alloc).
    fn descend_ring(
        &self,
        ring: &Ring,
        query: &IndexBox,
        self_geo: usize,
        flipped: bool,
        pairs: &mut Vec<(usize, usize)>,
    ) {
        let n = ring.n_edges;
        if n == 0 {
            return;
        }

        // sizes[0] = leaves (n), then ceil/INDEX_FANOUT up to 1.
        let mut sizes = [0usize; MAX_INDEX_LEVELS];
        let mut nlevels = 0;
        let mut s = n;
        loop {
            sizes[nlevels] = s;
            nlevels += 1;
            if s == 1 {
                break;
            }
            s = s.div_ceil(INDEX_FANOUT);
        }
        // base[0] = leaf region; base[k>=1] = internal region, cumulative across internal levels.
        let mut bases = [0usize; MAX_INDEX_LEVELS];
        bases[0] = ring.leaf_start;
        let mut acc = ring.internal_start;
        for k in 1..nlevels {
            bases[k] = acc;
            acc += sizes[k];
        }

        let mut stack: Vec<(usize, usize)> = vec![(nlevels - 1, 0)]; // (level, position), root
        while let Some((level, pos)) = stack.pop() {
            let abs = bases[level] + pos;
            if !query.ray_candidate(&self.boxes[abs]) {
                continue;
            }
            if level == 0 {
                // Contiguous leaves: `abs` is the leaf's component index, so geo-id = offset + abs.
                let o_geo = self.geo_id_offset + abs;
                pairs.push(if flipped {
                    (o_geo, self_geo)
                } else {
                    (self_geo, o_geo)
                });
            } else {
                let child_lo = pos * INDEX_FANOUT;
                let child_hi = (child_lo + INDEX_FANOUT).min(sizes[level - 1]);
                for cp in child_lo..child_hi {
                    stack.push((level - 1, cp));
                }
            }
        }
    }
}

// Builds a `NaturalIndex` one component-group at a time. Leaf boxes accumulate contiguously in
// `leaves` (component order); each group's internal nodes accumulate in `internals`. `finish`
// moves `leaves` into `boxes` (no copy) then appends `internals`, so the leaf region — the
// large part — is never moved and a leaf's box index stays equal to its component index.
struct NaturalIndexBuilder {
    leaves: Vec<IndexBox>,
    internals: Vec<IndexBox>,
    rings: Vec<Ring>,
}

impl NaturalIndexBuilder {
    fn new() -> Self {
        NaturalIndexBuilder {
            leaves: Vec::new(),
            internals: Vec::new(),
            rings: Vec::new(),
        }
    }

    fn push_segments<S: ToPoints>(&mut self, segs: &[S]) {
        let leaf_start = self.leaves.len();
        for s in segs {
            self.leaves.push(IndexBox::from_segment(s));
        }
        self.build_internals(leaf_start, segs.len());
    }

    fn push_points(&mut self, points: &[Point]) {
        let leaf_start = self.leaves.len();
        for p in points {
            self.leaves.push(IndexBox::from_point(p));
        }
        self.build_internals(leaf_start, points.len());
    }

    // Builds this group's internal levels into `internals`: level 1 unions the group's leaves
    // (in `leaves`), higher levels union the previous internal level (in `internals`).
    fn build_internals(&mut self, leaf_start: usize, n: usize) {
        let internal_start = self.internals.len();

        if n > 1 {
            // Level 1: union groups of this ring's leaves.
            let level1_count = n.div_ceil(INDEX_FANOUT);
            for p in 0..level1_count {
                let lo = leaf_start + p * INDEX_FANOUT;
                let hi = (lo + INDEX_FANOUT).min(leaf_start + n);
                let parent = IndexBox::union(&self.leaves[lo..hi]);
                self.internals.push(parent);
            }
            // Levels 2.. : union groups of the previous internal level.
            let mut level_start = internal_start;
            let mut level_len = level1_count;
            while level_len > 1 {
                let parents_start = self.internals.len();
                let parent_count = level_len.div_ceil(INDEX_FANOUT);
                for p in 0..parent_count {
                    let lo = level_start + p * INDEX_FANOUT;
                    let hi = (lo + INDEX_FANOUT).min(level_start + level_len);
                    let parent = IndexBox::union(&self.internals[lo..hi]);
                    self.internals.push(parent);
                }
                level_start = parents_start;
                level_len = parent_count;
            }
        }

        self.rings.push(Ring {
            leaf_start,
            n_edges: n,
            internal_start,
        });
    }

    fn finish(mut self, geo_id_offset: usize) -> NaturalIndex {
        let base = self.leaves.len();
        let mut boxes = self.leaves; // moved, not copied
        boxes.extend(self.internals); // only the (small) internals region is copied
        for r in &mut self.rings {
            r.internal_start += base; // shift into the combined `boxes` vector
        }
        NaturalIndex {
            boxes,
            rings: self.rings,
            geo_id_offset,
        }
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
        index: NaturalIndex,
    },
    Line {
        lines: Vec<LineSegment>,
        index: NaturalIndex,

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
        // Index over all rings (exteriors + holes), geo_id_offset 0.
        index: NaturalIndex,
        // Index over hole rings only, geo_id_offset = the first hole edge's global index.
        holes_index: NaturalIndex,
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
        let mut builder = NaturalIndexBuilder::new();
        builder.push_points(&points);
        let index = builder.finish(0);
        GeoOwned::Point { points, index }
    }

    fn from_line(lines: Vec<LineSegment>, b1s: Vec<bool>, b2s: Vec<bool>) -> Self {
        let n = lines.len();
        assert_eq!(b1s.len(), n, "b1s length mismatch");
        assert_eq!(b2s.len(), n, "b2s length mismatch");
        // Single ring over all segments (MultiLineStrings share one bbox tree).
        let mut builder = NaturalIndexBuilder::new();
        builder.push_segments(&lines);
        let index = builder.finish(0);
        let false_flags = vec![false; n];
        GeoOwned::Line {
            lines,
            index,
            b1s,
            b2s,
            false_flags,
        }
    }

    fn from_poly(
        edges: Vec<Edge>,
        holes_start: usize,
        poly_ids: Vec<usize>,
        ring_lengths: Vec<usize>,
    ) -> Self {
        let t_index = Instant::now();

        // Global index over every ring (exteriors then holes). Contiguous leaves mean a leaf's
        // index equals its global edge index, so geo_id_offset = 0.
        let mut global = NaturalIndexBuilder::new();
        // Holes index over only the hole rings (those starting at/after holes_start). Its
        // geo_id_offset = holes_start, so leaf i addresses global edge holes_start + i.
        let mut holes = NaturalIndexBuilder::new();

        let mut start = 0;
        for &len in &ring_lengths {
            let ring = &edges[start..start + len];
            global.push_segments(ring);
            if start >= holes_start {
                holes.push_segments(ring);
            }
            start += len;
        }

        let index = global.finish(0);
        let holes_index = holes.finish(holes_start);
        // println!(
        //     "[GEO] index creation: {:?} ({} edges)",
        //     t_index.elapsed(),
        //     edges.len()
        // );
        GeoOwned::Poly {
            edges,
            index,
            holes_index,
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
    pub(crate) fn interior(&self) -> (GeoView<'_>, &NaturalIndex) {
        match self {
            GeoOwned::Point { points, index } => (GeoView::Points { points }, index),
            GeoOwned::Line {
                lines,
                b1s,
                b2s,
                index,
                ..
            } => (GeoView::LineSegments { lines, b1s, b2s }, index),
            GeoOwned::Poly {
                edges,
                poly_ids,
                index,
                ..
            } => (GeoView::PolyEdges { edges, poly_ids }, index),
        }
    }

    pub(crate) fn not_exterior(&self) -> (GeoView<'_>, &NaturalIndex) {
        match self {
            GeoOwned::Point { points, index } => (GeoView::Points { points }, index),
            GeoOwned::Line {
                lines,
                false_flags,
                index,
                ..
            } => (
                GeoView::LineSegments {
                    lines,
                    b1s: false_flags,
                    b2s: false_flags,
                },
                index,
            ),
            // TODO: will need to add some flag to indicate that the edges
            // themselves do count (as opposed to interior above).
            GeoOwned::Poly {
                edges,
                poly_ids,
                index,
                ..
            } => (GeoView::PolyEdges { edges, poly_ids }, index),
        }
    }

    pub(crate) fn holes(&self) -> Option<(GeoView<'_>, &NaturalIndex)> {
        match self {
            GeoOwned::Point { .. } | GeoOwned::Line { .. } => None,
            GeoOwned::Poly {
                edges,
                poly_ids,
                holes_index,
                ..
            } => Some((
                // Full edge slice (global index space); `holes_index` covers only the hole
                // edges and carries their global geo-ids via its offset.
                GeoView::PolyEdges { edges, poly_ids },
                holes_index,
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
    ring_lengths: &mut Vec<usize>,
    poly_idx: usize,
) {
    // WKB/WKT rings are closed (first coord == last coord), so iterating consecutive coord
    // pairs yields every edge including the wraparound edge directly. Each edge records the
    // index of its in-ring successor (`next`), patched at the end so the last edge wraps to
    // the ring's first edge. The interior angle is no longer precomputed — it's recomputed
    // on demand from an edge and its `next` edge.
    let t_ring = Instant::now();
    let ring_start = edges.len();
    let mut signed_area = 0.0f64;

    for (a, b) in coords
        .map(|c| Point { x: c.x(), y: c.y() })
        .tuple_windows::<(_, _)>()
    {
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

    let n_edges = edges.len() - ring_start;
    assert!(n_edges >= 3, "Degenerate polygon ring");
    ring_lengths.push(n_edges);

    // The closing duplicate coord makes the last edge end where the first edge starts.
    let last = edges.len() - 1;
    assert!(
        edges[last].p2.is_equal(&edges[ring_start].p1),
        "polygon ring must be closed (first coord == last coord)"
    );
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

    // println!(
    //     "[GEO] process_ring: {:?} ({} edges)",
    //     t_ring.elapsed(),
    //     n_edges
    // );
}

fn build_poly_geo<P: PolygonTrait<T = f64>>(polygons: impl Iterator<Item = P>) -> GeoOwned {
    let mut edges: Vec<Edge> = Vec::new();
    let mut poly_ids: Vec<usize> = Vec::new();
    let mut ring_lengths: Vec<usize> = Vec::new();
    let polys: Vec<P> = polygons.collect();
    for (poly_idx, poly) in polys.iter().enumerate() {
        let Some(exterior) = poly.exterior() else {
            continue;
        };
        process_ring(
            exterior.coords(),
            true,
            &mut edges,
            &mut poly_ids,
            &mut ring_lengths,
            poly_idx,
        );
    }
    let holes_start = edges.len();
    for (poly_idx, poly) in polys.iter().enumerate() {
        for hole in poly.interiors() {
            process_ring(
                hole.coords(),
                false,
                &mut edges,
                &mut poly_ids,
                &mut ring_lengths,
                poly_idx,
            );
        }
    }
    GeoOwned::from_poly(edges, holes_start, poly_ids, ring_lengths)
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

impl GeoView<'_> {
    // Returns (t, is_entering) where t is the parameter on the left segment where it crosses
    // the right polygon edge `right_idx`, and is_entering is true if the left segment is
    // entering the polygon interior. Uses exclude-start/include-end convention on u to avoid
    // double-counting shared poly vertices. At the edge's end vertex (u ≈ 1) the interior
    // angle is recomputed from the edge and its `next` edge.
    pub(crate) fn segment_crossing_t(
        &self,
        left: &impl SegmentTrait,
        right_idx: usize,
    ) -> Option<(f64, bool)> {
        match self {
            GeoView::PolyEdges { edges, .. } => {
                let right = &edges[right_idx];
                if !left.x_ranges_overlap(right) {
                    return None;
                }
                let cross = left.cross(right);
                let (a1, _) = left.to_points();
                let (b1, b2) = right.to_points();
                let f = b1.diff(a1);
                let (t, u);
                if cross.abs() >= EPS {
                    t = -right.cross(&f) / cross;
                    u = -left.cross(&f) / cross;
                } else {
                    if left.norm_sq() < EPS {
                        return None;
                    }
                    if !left.is_collinear_with(b1) {
                        return None;
                    }
                    u = 1.0;
                    t = left.projection_t(b2);
                }
                if !(t > EPS && t < 1.0 - EPS) {
                    return None;
                }

                // exclude-start/include-end convention: only check u = 1, not u = 0.
                if u > EPS && u < 1.0 - EPS {
                    let is_entering = (cross < 0.0) == right.interior_on_left();
                    return Some((t, is_entering));
                }
                if (1.0 - EPS..=1.0 + EPS).contains(&u) {
                    let next = &edges[right.next];
                    if let Some(is_entering) = right.crossing_at_vertex(left, next) {
                        return Some((t, is_entering));
                    }
                }
                None
            }
            _ => unimplemented!(),
        }
    }

    // Checks if the left segment's midpoint falls on the right edge `right_idx`
    // (this check does not include ray casting).
    pub(crate) fn segment_midpoint_check(
        &self,
        left: &impl SegmentTrait,
        right_idx: usize,
    ) -> MidpointResult {
        match self {
            GeoView::PolyEdges { edges, .. } => {
                let right = &edges[right_idx];
                let mid = left.midpoint();
                if point_on_segment_t(&mid, right).is_some() {
                    if left.cross(right).abs() < EPS {
                        return MidpointResult::OnEdge;
                    }
                    let u = right.projection_t(&mid);
                    if u >= 1.0 - EPS && right.is_reflex(&edges[right.next]) {
                        return MidpointResult::MidpointInside;
                    }
                    return MidpointResult::MidpointOutside;
                }
                MidpointResult::NoContact
            }
            _ => unimplemented!(),
        }
    }
}
