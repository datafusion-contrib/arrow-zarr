use super::joinable_geo::{
    covered_range, midpoint_ray_check, point_at_line_endpoint, point_on_segment_t,
    poly_edge_relation, ray_crosses_edge, segment_crossing_check, Accumulator, Bboxable, Edge,
    JoinableGeo, LineSegment, Point, SegmentTrait, EPS,
};

// `within(a, b)` — true if `a` is contained in `b`.
pub(crate) fn st_within(a: &JoinableGeo, b: &JoinableGeo) -> bool {
    match (a, b) {
        // A higher-dimensional geometry can't be within a lower-dimensional one.
        (JoinableGeo::Line { .. } | JoinableGeo::Poly { .. }, JoinableGeo::Point { .. })
        | (JoinableGeo::Poly { .. }, JoinableGeo::Line { .. }) => false,

        // Point(s) within polygon(s): each point must be strictly inside.
        (JoinableGeo::Point { points }, JoinableGeo::Poly { edges, .. }) => {
            let mut acc = PointInPoly::new(points, edges);
            a.fold_for_grouped_check(b, &mut acc);
            acc.finish()
        }

        // Point(s) within point(s): every left point must equal some right point.
        (JoinableGeo::Point { points: a_pts }, JoinableGeo::Point { points: b_pts }) => {
            let mut acc = PointInPoint::new(a_pts, b_pts);
            a.fold_for_grouped_check(b, &mut acc);
            acc.finish()
        }

        // Point(s) within line(s): every left point must lie on the line's interior (on a segment
        // but not at a boundary endpoint).
        (JoinableGeo::Point { points }, JoinableGeo::Line { lines, .. }) => {
            let mut acc = PointInLine::new(points, lines);
            a.fold_for_grouped_check(b, &mut acc);
            acc.finish()
        }

        // Line(s) within line(s): every left segment must be fully covered by right segments,
        // stitched across collinear overlaps via a per-segment interval-union sweep.
        (JoinableGeo::Line { lines: l_lines, .. }, JoinableGeo::Line { lines: r_lines, .. }) => {
            let mut acc = LineInLine::new(l_lines, r_lines);
            a.fold_for_grouped_check(b, &mut acc);
            acc.finish()
        }

        // Line(s) within polygon(s): every left segment must lie in interior∪boundary, and at
        // least one must touch the interior.
        (JoinableGeo::Line { lines, .. }, JoinableGeo::Poly { edges, .. }) => {
            let mut acc = LineInPoly::new(lines, edges);
            a.fold_for_grouped_check(b, &mut acc);
            acc.finish()
        }

        // Polygon(s) within polygon(s). First a reverse pre-check: if any right hole lies entirely
        // inside the left poly (the "donut" — left poly swallows a right hole), the left poly's
        // interior covers right-exterior, so it is not within. This case crosses no boundary, so
        // the forward within-check can't see it. The forward check itself is not yet implemented.
        (
            JoinableGeo::Poly {
                edges: l_edges,
                poly_ids: l_ids,
                ..
            },
            JoinableGeo::Poly {
                edges: r_edges,
                poly_ids: r_ids,
                ..
            },
        ) => {
            let mut hole_acc = HolesInLeftPoly::new(r_edges, b.holes_start(), l_edges);
            b.fold_holes_into(a, &mut hole_acc);
            if hole_acc.finish() {
                return false; // an engulfed right hole ⇒ left poly not within
            }
            let mut acc = PolyInPoly::new(l_edges, l_ids, r_edges, r_ids);
            a.fold_for_grouped_check(b, &mut acc);
            acc.finish()
        }

        _ => unimplemented!("st_within: geometry combination not yet supported"),
    }
}

pub(crate) fn st_contains(a: &JoinableGeo, b: &JoinableGeo) -> bool {
    st_within(b, a)
}

// Accumulator for `within(points, polygon)`. Each point is classified inside / boundary /
// outside (odd ray-cast parity over all edges = inside; on an edge = boundary; else outside).
// The (multi)point is within iff **no point is outside** and **at least one is strictly inside**
// (boundary points are tolerated, but don't satisfy the interior requirement).
pub(crate) struct PointInPoly<'a> {
    points: &'a [Point],
    edges: &'a [Edge],
    // Current point's group and its running parity / on-edge (boundary) state.
    current_group: Option<usize>,
    parity: bool,
    on_edge: bool,
    // Points not yet seen, decremented as each new group starts. A point that produces no
    // candidate edge is never seen — and is necessarily outside — so a nonzero remainder fails.
    left_to_see: usize,
    // Whether any finished point was strictly inside / outside.
    pt_inside: bool,
    pt_outside: bool,
}

impl<'a> PointInPoly<'a> {
    pub(crate) fn new(points: &'a [Point], edges: &'a [Edge]) -> Self {
        PointInPoly {
            left_to_see: points.len(),
            points,
            edges,
            current_group: None,
            parity: false,
            on_edge: false,
            pt_inside: false,
            pt_outside: false,
        }
    }

    // Classify the point whose group just ended (boundary points set neither flag).
    fn finalize_group(&mut self) {
        if !self.on_edge {
            if self.parity {
                self.pt_inside = true;
            } else {
                self.pt_outside = true;
            }
        }
    }
}

impl Accumulator for PointInPoly<'_> {
    fn update(&mut self, li: usize, ri: usize) {
        if self.current_group != Some(li) {
            if self.current_group.is_some() {
                self.finalize_group();
                if self.pt_outside {
                    return; // a point in the exterior decides it
                }
            }
            // Start the new point's group.
            self.left_to_see -= 1;
            self.current_group = Some(li);
            self.parity = false;
            self.on_edge = false;
        }

        let p = &self.points[li];
        let edge = &self.edges[ri];
        if point_on_segment_t(p, edge).is_some() {
            self.on_edge = true;
        } else if ray_crosses_edge(p, edge, &self.edges[edge.next()]) {
            self.parity = !self.parity;
        }
    }

    fn ready(&self) -> bool {
        self.pt_outside
    }

    // Within iff every point was seen, none outside, and at least one strictly inside.
    fn finish(mut self) -> bool {
        if self.current_group.is_some() && !self.pt_outside {
            self.finalize_group(); // finalize the last point's group
        }
        self.left_to_see == 0 && !self.pt_outside && self.pt_inside
    }
}

// Accumulator for `within(points, points)`. Each left point (a group) must equal at least one
// right point. The (multi)point is within iff every left point matched (and it is non-empty).
pub(crate) struct PointInPoint<'a> {
    left: &'a [Point],
    right: &'a [Point],
    current_group: Option<usize>,
    // Whether the current group has matched (resets per group; also gates the skip-rest guard).
    point_matched: bool,
    // Whether some finished group ended without a match (the (multi)point is not within).
    point_unmatched: bool,
    // Left points not yet matched; decremented once per matched group.
    count: usize,
}

impl<'a> PointInPoint<'a> {
    pub(crate) fn new(left: &'a [Point], right: &'a [Point]) -> Self {
        PointInPoint {
            count: left.len(),
            left,
            right,
            current_group: None,
            point_matched: false,
            point_unmatched: false,
        }
    }
}

impl Accumulator for PointInPoint<'_> {
    fn update(&mut self, li: usize, ri: usize) {
        if self.current_group == Some(li) {
            if self.point_matched {
                return; // group already matched — skip the rest
            }
        } else {
            if self.current_group.is_some() && !self.point_matched {
                self.point_unmatched = true; // previous group ended unmatched
            }
            self.current_group = Some(li);
            self.point_matched = false;
        }

        if self.left[li].is_equal(&self.right[ri]) {
            self.point_matched = true;
            self.count -= 1;
        }
    }

    fn ready(&self) -> bool {
        self.count == 0 || self.point_unmatched
    }

    // Within iff every left point matched. `point_matched` only guards the empty (n == 0) case;
    // `count == 0` already implies the last group matched.
    fn finish(self) -> bool {
        self.point_matched && self.count == 0
    }
}

// Accumulator for `within(points, line)`. Each point is classified interior / boundary / exterior:
// on a segment body (or a non-boundary interior vertex) = interior; coinciding with a boundary
// endpoint = boundary; not on the line at all = exterior. The (multi)point is within iff **no point
// is exterior** and **at least one is interior** (boundary points are tolerated but don't satisfy
// the interior requirement). Mirrors `PointInPoly`: boundary overrides interior within a group, so
// the whole group must be scanned (no skip-on-match).
pub(crate) struct PointInLine<'a> {
    points: &'a [Point],
    lines: &'a [LineSegment],
    current_group: Option<usize>,
    // Current point's running interior / boundary state (reset per group).
    on_interior: bool,
    on_boundary: bool,
    // Points not yet seen; a point with no candidate segment is exterior, so a nonzero remainder
    // fails (matches `PointInPoly`).
    left_to_see: usize,
    // Whether any finished point was interior / exterior.
    pt_inside: bool,
    pt_outside: bool,
}

impl<'a> PointInLine<'a> {
    pub(crate) fn new(points: &'a [Point], lines: &'a [LineSegment]) -> Self {
        PointInLine {
            left_to_see: points.len(),
            points,
            lines,
            current_group: None,
            on_interior: false,
            on_boundary: false,
            pt_inside: false,
            pt_outside: false,
        }
    }

    // Classify the point whose group just ended (boundary overrides interior; sets neither flag).
    fn finalize_group(&mut self) {
        if !self.on_boundary {
            if self.on_interior {
                self.pt_inside = true;
            } else {
                self.pt_outside = true;
            }
        }
    }
}

impl Accumulator for PointInLine<'_> {
    fn update(&mut self, li: usize, ri: usize) {
        if self.current_group != Some(li) {
            if self.current_group.is_some() {
                self.finalize_group();
                if self.pt_outside {
                    return; // a point in the exterior decides it
                }
            }
            self.left_to_see -= 1;
            self.current_group = Some(li);
            self.on_interior = false;
            self.on_boundary = false;
        }

        let p = &self.points[li];
        let seg = &self.lines[ri];
        if let Some(is_boundary) = point_at_line_endpoint(p, seg) {
            if is_boundary {
                self.on_boundary = true;
            } else {
                self.on_interior = true;
            }
        } else if point_on_segment_t(p, seg).is_some() {
            self.on_interior = true;
        }
    }

    fn ready(&self) -> bool {
        self.pt_outside
    }

    // Within iff every point was seen, none exterior, and at least one interior.
    fn finish(mut self) -> bool {
        if self.current_group.is_some() && !self.pt_outside {
            self.finalize_group();
        }
        self.left_to_see == 0 && !self.pt_outside && self.pt_inside
    }
}

// Accumulator for `within(line, line)`. Every left segment must be fully covered by right
// segments. Per left segment (a contiguous descent group) we collect the covered sub-interval
// `covered_range` gives for each candidate right segment (in the left segment's own parameter,
// clamped to `[0, 1]`) into `ranges`, then at group end sort by start and sweep: the union must
// tile `[0, 1]`. A single right segment that covers the whole left segment is the common case and
// short-circuits without touching the vector. Only the *current* group's coverage is ever live
// (groups are contiguous, processed once), so a single `group_covered` bool plus a count of
// covered groups suffices — no per-segment bitmap. `covered_count == n` is the verdict (coverage
// is 1-D, so there's no separate "touches the interior" requirement to track).
pub(crate) struct LineInLine<'a> {
    left: &'a [LineSegment],
    right: &'a [LineSegment],
    current_group: Option<usize>,
    // Whether the current group is already fully covered (set by the single-segment fast path).
    group_covered: bool,
    // Number of left segments (groups) confirmed fully covered.
    covered_count: usize,
    // Covered sub-intervals of the current left segment, in its own parameter. Reused across
    // groups (`clear` per group); `Vec::new()` so a no-coverage call never allocates.
    ranges: Vec<(f64, f64)>,
    // A left segment was found provably uncoverable (its group/gap ended uncovered) — fail early.
    line_unmatched: bool,
}

impl<'a> LineInLine<'a> {
    pub(crate) fn new(left: &'a [LineSegment], right: &'a [LineSegment]) -> Self {
        LineInLine {
            left,
            right,
            current_group: None,
            group_covered: false,
            covered_count: 0,
            ranges: Vec::new(),
            line_unmatched: false,
        }
    }

    // Decide coverage of the current left segment: the fast path may have already covered it,
    // otherwise sweep the accumulated `ranges`. If neither tiles `[0, 1]` the segment is provably
    // uncoverable → fail the whole `within`.
    fn finalize_group(&mut self) {
        if self.group_covered || fully_covered(&mut self.ranges) {
            self.covered_count += 1;
        } else {
            self.line_unmatched = true;
        }
    }
}

// Does the union of `[0, 1]`-clamped intervals tile `[0, 1]`? Sort by start, sweep a frontier from
// 0: every interval must start at or before the running frontier (else a gap) and may extend it.
fn fully_covered(ranges: &mut [(f64, f64)]) -> bool {
    if ranges.is_empty() {
        return false;
    }
    ranges.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let mut frontier = 0.0;
    for &(lo, hi) in ranges.iter() {
        if lo > frontier + EPS {
            return false; // gap before this interval starts
        }
        if hi > frontier {
            frontier = hi;
        }
        if frontier >= 1.0 - EPS {
            return true;
        }
    }
    frontier >= 1.0 - EPS
}

impl Accumulator for LineInLine<'_> {
    // Coverage is order-independent, so a symmetric bbox overlap (not the rightward ray) is the
    // correct, tighter prune.
    fn prune(a: &impl Bboxable, b: &impl Bboxable) -> bool {
        a.box_overlap(b)
    }

    fn update(&mut self, li: usize, ri: usize) {
        if self.current_group != Some(li) {
            // Finalize the previous group, then detect skipped (candidate-less, hence uncoverable)
            // segments — either fails the whole `within` early.
            match self.current_group {
                Some(prev) => {
                    self.finalize_group();
                    if self.line_unmatched || li > prev + 1 {
                        self.line_unmatched = true;
                        return;
                    }
                }
                None if li > 0 => {
                    self.line_unmatched = true;
                    return;
                }
                None => {}
            }
            self.ranges.clear();
            self.group_covered = false;
            self.current_group = Some(li);
        }
        if self.group_covered {
            return;
        }
        if let Some((t_lo, t_hi)) = covered_range(&self.left[li], &self.right[ri]) {
            // Fast path: this single right segment already covers the whole left segment.
            if t_lo <= EPS && t_hi >= 1.0 - EPS {
                self.group_covered = true;
            } else {
                self.ranges.push((t_lo, t_hi));
            }
        }
    }

    fn ready(&self) -> bool {
        self.line_unmatched || self.covered_count == self.left.len()
    }

    fn finish(mut self) -> bool {
        if self.current_group.is_some() {
            self.finalize_group();
        }
        // A trailing candidate-less segment leaves `covered_count < n`, so no explicit gap check is
        // needed here.
        !self.left.is_empty() && self.covered_count == self.left.len()
    }
}

// Accumulator for `within(line, polygon)`. Each left segment (a group) must lie in the
// polygon's interior∪boundary, and at least one segment must touch the interior. Per segment we
// fold two per-edge checks (default `ray_candidate` prune, a superset of `box_overlap`, feeds
// both):
//   * `midpoint_ray_check` — the interior anchor: ray-cast parity (trustworthy only when the
//     midpoint is off the boundary) plus a midpoint-on-vertex "inside" flag that holds even when
//     parity has been invalidated by another edge (kissing polygons at a shared vertex).
//   * `segment_crossing_check` — boundary contacts: every crossing (transversal edge-body, vertex,
//     or convex-collinear peel) contributes a signed `t` that must net to zero over the segment;
//     an unbalanced one is a genuine exit to the exterior, while a vertex of another ring touching
//     an edge interior cancels against the edge-body crossing there. Interior contacts set
//     `qualified`.
pub(crate) struct LineInPoly<'a> {
    left: &'a [LineSegment],
    edges: &'a [Edge],
    current_group: Option<usize>,
    // Per-group running state (reset on each group change).
    parity: bool,
    parity_valid: bool,
    mid_inside: bool,
    s_sum: f64,
    qualified: bool,
    // Global verdict state.
    failed: bool,
    any_interior: bool,
}

impl<'a> LineInPoly<'a> {
    pub(crate) fn new(left: &'a [LineSegment], edges: &'a [Edge]) -> Self {
        LineInPoly {
            left,
            edges,
            current_group: None,
            parity: false,
            parity_valid: true,
            mid_inside: false,
            s_sum: 0.0,
            qualified: false,
            failed: false,
            any_interior: false,
        }
    }

    fn reset_group(&mut self, li: usize) {
        self.current_group = Some(li);
        self.parity = false;
        self.parity_valid = true;
        self.mid_inside = false;
        self.s_sum = 0.0;
        self.qualified = false;
    }

    // Classify the segment whose group just ended. Unbalanced crossings = a genuine exterior
    // excursion → fail; any interior evidence flags the line's interior contact; a trustworthy
    // outside reading with no interior contact = a strictly exterior segment → fail; otherwise
    // the segment is boundary-neutral (on the boundary, contributing nothing).
    fn finalize_group(&mut self) {
        if self.s_sum.abs() > EPS {
            self.failed = true;
        } else if self.mid_inside || self.qualified || (self.parity_valid && self.parity) {
            self.any_interior = true;
        } else if self.parity_valid && !self.parity {
            self.failed = true;
        }
    }
}

impl Accumulator for LineInPoly<'_> {
    fn update(&mut self, li: usize, ri: usize) {
        if self.current_group != Some(li) {
            // Finalize the previous group, then detect a gap: a skipped left segment had no
            // candidate edges, so it lies strictly in the exterior → fail.
            match self.current_group {
                Some(prev) => {
                    self.finalize_group();
                    if li > prev + 1 {
                        self.failed = true;
                    }
                }
                None if li > 0 => self.failed = true, // leading gap
                None => {}
            }
            if self.failed {
                return;
            }
            self.reset_group(li);
        }

        let edge = &self.edges[ri];
        let next = &self.edges[edge.next()];
        let seg = &self.left[li];

        let (ray, inside, parity_valid) = midpoint_ray_check(seg, edge, next);
        if ray {
            self.parity ^= true;
        }
        if inside {
            self.mid_inside = true;
        }
        if !parity_valid {
            self.parity_valid = false;
        }

        let (qualifying, signed_t) = segment_crossing_check(seg, edge, next);
        if qualifying {
            self.qualified = true;
        }
        if let Some(t) = signed_t {
            self.s_sum += t;
        }
    }

    fn ready(&self) -> bool {
        self.failed
    }

    // Within iff every segment is interior∪boundary (no group failed, no gap) and at least one
    // segment touched the interior.
    fn finish(mut self) -> bool {
        if self.failed {
            return false;
        }
        match self.current_group {
            Some(prev) => {
                self.finalize_group();
                // Trailing gap: left segments after the last candidate are strictly exterior.
                if prev + 1 < self.left.len() {
                    return false;
                }
            }
            // No left segment had any candidate edge → the whole line is exterior (or empty).
            None => return false,
        }
        !self.failed && self.any_interior
    }
}

// Reverse pre-check for `within(left_poly, right_poly)`: is any right hole engulfed by the left
// poly? Each right hole edge (a group) is a point query — its midpoint is ray-cast against the left
// poly's edges for parity. No crossing logic (boundary crossings are the forward check's job): we
// only ask "is this hole point strictly inside the left poly". `found_inside` (any hole edge inside)
// ⇒ the left poly swallows a right hole ⇒ not within. `right` is the right poly's full edge vector
// and `li` arrives hole-relative, so `right[holes_start + li]` is the hole edge.
pub(crate) struct HolesInLeftPoly<'a> {
    right: &'a [Edge],
    holes_start: usize,
    left: &'a [Edge],
    current_group: Option<usize>,
    parity: bool,
    parity_valid: bool,
    mid_inside: bool,
    found_inside: bool,
}

impl<'a> HolesInLeftPoly<'a> {
    pub(crate) fn new(right: &'a [Edge], holes_start: usize, left: &'a [Edge]) -> Self {
        HolesInLeftPoly {
            right,
            holes_start,
            left,
            current_group: None,
            parity: false,
            parity_valid: true,
            mid_inside: false,
            found_inside: false,
        }
    }

    // Classify the hole edge whose group just ended: inside (engulfed) iff a midpoint landed inside
    // or the completed ray-cast parity is odd.
    fn finalize_group(&mut self) {
        if self.mid_inside || (self.parity_valid && self.parity) {
            self.found_inside = true;
        }
    }
}

impl Accumulator for HolesInLeftPoly<'_> {
    fn update(&mut self, li: usize, ri: usize) {
        if self.current_group != Some(li) {
            if self.current_group.is_some() {
                self.finalize_group();
                if self.found_inside {
                    return;
                }
            }
            self.current_group = Some(li);
            self.parity = false;
            self.parity_valid = true;
            self.mid_inside = false;
        }
        let hole = &self.right[self.holes_start + li];
        let edge = &self.left[ri];
        let next = &self.left[edge.next()];
        let (ray, inside, parity_valid) = midpoint_ray_check(hole, edge, next);
        if ray {
            self.parity ^= true;
        }
        if inside {
            self.mid_inside = true;
            self.found_inside = true; // midpoint-inside is decisive on its own — early exit
        }
        if !parity_valid {
            self.parity_valid = false;
        }
    }

    fn ready(&self) -> bool {
        self.found_inside
    }

    fn finish(mut self) -> bool {
        if self.current_group.is_some() {
            self.finalize_group();
        }
        self.found_inside
    }
}

// Forward check for `within(left_poly, right_poly)`. Every left edge (grouped, in WKB order) must be
// *contained* by the right poly — strictly inside (odd parity ray-cast), `mid_inside`, or a same-side
// collinear overlap. Anything else (strictly outside, a crossing, an opposite-side overlap, or a
// candidate-less gap) disqualifies. A polygon edge has a sided interior, so there is no
// boundary-only limbo and no `any_interior` flag (`A == B` is within). Additionally every edge of one
// left poly must resolve to a single right container component — the **nearest rightward crossing**'s
// poly id (disjoint interiors put nothing between the point and its own poly's boundary), or a
// boundary contact's poly id; a second differing container within one left poly disqualifies. The
// container resets when the left poly id changes. Verdict: no disqualification.
pub(crate) struct PolyInPoly<'a> {
    left: &'a [Edge],
    left_poly_ids: &'a [usize],
    right: &'a [Edge],
    right_poly_ids: &'a [usize],
    current_group: Option<usize>,
    // The left poly we're currently in, and the single right container all its edges must share.
    current_left_poly: Option<usize>,
    container: Option<usize>,
    // Per-left-edge running state (reset on each group change).
    parity: bool,
    parity_valid: bool,
    mid_inside: bool,
    qualified: bool,
    // This edge's container candidate as `(x-distance from the midpoint, right poly id)`: the nearest
    // contact wins (a rightward crossing at `xi - mid.x`, or a boundary contact at distance 0).
    edge_container: Option<(f64, usize)>,
    failed: bool,
}

impl<'a> PolyInPoly<'a> {
    pub(crate) fn new(
        left: &'a [Edge],
        left_poly_ids: &'a [usize],
        right: &'a [Edge],
        right_poly_ids: &'a [usize],
    ) -> Self {
        PolyInPoly {
            left,
            left_poly_ids,
            right,
            right_poly_ids,
            current_group: None,
            current_left_poly: None,
            container: None,
            parity: false,
            parity_valid: true,
            mid_inside: false,
            qualified: false,
            edge_container: None,
            failed: false,
        }
    }

    fn reset_group(&mut self, li: usize) {
        self.current_group = Some(li);
        self.parity = false;
        self.parity_valid = true;
        self.mid_inside = false;
        self.qualified = false;
        self.edge_container = None;
    }

    // Record a container candidate at x-distance `dist` (≥ 0) from the midpoint; the nearest wins. A
    // boundary contact passes `dist == 0`, so it beats any interior crossing (which is strictly > 0).
    fn consider_container(&mut self, dist: f64, rp: usize) {
        if self.edge_container.map_or(true, |(d, _)| dist < d) {
            self.edge_container = Some((dist, rp));
        }
    }

    // Classify the left edge whose group just ended: it must be contained, and its container must
    // agree with the rest of its left poly.
    fn finalize_group(&mut self) {
        self.qualified |= (self.parity_valid && self.parity) || self.mid_inside;
        if !self.qualified {
            self.failed = true; // a strictly-exterior edge
            return;
        }
        // A qualified edge always recorded a container (a parity-odd edge crossed something; a
        // boundary contact set distance 0).
        let c = self
            .edge_container
            .expect("qualified edge has a container")
            .1;
        match self.container {
            None => self.container = Some(c),
            Some(q) if q != c => self.failed = true, // edges of one left poly in two right polys
            Some(_) => {}
        }
    }
}

impl Accumulator for PolyInPoly<'_> {
    fn update(&mut self, li: usize, ri: usize) {
        if self.current_group != Some(li) {
            match self.current_group {
                Some(prev) => {
                    self.finalize_group();
                    if li > prev + 1 {
                        self.failed = true; // gap: a skipped left edge is strictly exterior
                    }
                }
                None if li > 0 => self.failed = true, // leading gap
                None => {}
            }
            if self.failed {
                return;
            }
            // A new left poly starts a fresh container.
            let lp = self.left_poly_ids[li];
            if self.current_left_poly != Some(lp) {
                self.current_left_poly = Some(lp);
                self.container = None;
            }
            self.reset_group(li);
        }

        let edge = &self.right[ri];
        let next = &self.right[edge.next()];
        let seg = &self.left[li];
        let rp = self.right_poly_ids[ri];

        let (ray, inside, parity_valid) = midpoint_ray_check(seg, edge, next);
        if ray {
            self.parity ^= true;
            let mid = seg.midpoint();
            let xi = edge.x_intercept_at_point(&mid);
            self.consider_container(xi - mid.x(), rp);
        }
        if inside {
            self.mid_inside = true;
            self.consider_container(0.0, rp); // boundary contact: nearest possible
        }
        if !parity_valid {
            self.parity_valid = false;
        }

        let (disqualifies, collinear_inside) = poly_edge_relation(seg, edge, next);
        if disqualifies {
            self.failed = true;
        }
        if collinear_inside {
            self.qualified = true;
            self.consider_container(0.0, rp); // on the boundary of rp
        }
    }

    fn ready(&self) -> bool {
        self.failed
    }

    // Within iff no edge disqualified, no gap, and the left poly was non-empty.
    fn finish(mut self) -> bool {
        if self.failed {
            return false;
        }
        match self.current_group {
            Some(prev) => {
                self.finalize_group();
                if prev + 1 < self.left.len() {
                    return false; // trailing gap
                }
            }
            None => return false, // no candidates at all → exterior (or empty)
        }
        !self.failed && !self.left.is_empty()
    }
}

#[cfg(test)]
mod test_helpers {
    use geos::Geom;

    pub(super) use super::super::test_utils::wkt_to_wkb;
    use super::JoinableGeo;
    use crate::geospatial::st_within::{st_contains, st_within};

    pub(super) fn compare_within(label: &str, wkt_a: &str, wkt_b: &str) {
        let bytes_a = wkt_to_wkb(wkt_a);
        let bytes_b = wkt_to_wkb(wkt_b);

        let geos_a = geos::Geometry::new_from_wkb(&bytes_a).unwrap();
        let geos_b = geos::Geometry::new_from_wkb(&bytes_b).unwrap();
        let mut geos_result = geos_a.within(&geos_b).unwrap();

        let joinable_a = JoinableGeo::from_wkb(&bytes_a).unwrap();
        let joinable_b = JoinableGeo::from_wkb(&bytes_b).unwrap();
        let mut joinable_result = st_within(&joinable_a, &joinable_b);

        assert_eq!(geos_result, joinable_result, "{label}");

        geos_result = geos_b.contains(&geos_a).unwrap();
        joinable_result = st_contains(&joinable_b, &joinable_a);

        assert_eq!(geos_result, joinable_result, "{label}, st_contains version");
    }
}

#[cfg(test)]
mod within_tests {
    use super::test_helpers::compare_within;

    #[test]
    fn point_left() {
        compare_within("point left, case 1", "POINT (1 1)", "POINT (1 1)");
        compare_within("point left, case 2", "POINT (1 1)", "POINT (2 2)");
        compare_within("point left, case 3", "POINT (1 1)", "MULTIPOINT (0 0, 1 1)");
        compare_within("point left, case 4", "POINT (1 1)", "MULTIPOINT (0 0, 2 2)");

        compare_within(
            "point left, case 5",
            "POINT (0.5 0)",
            "LINESTRING (0 0, 1 0)",
        );
        compare_within("point left, case 6", "POINT (2 0)", "LINESTRING (0 0, 1 0)");
        compare_within("point left, case 7", "POINT (0 0)", "LINESTRING (0 0, 1 0)");

        compare_within(
            "point left, case 8",
            "POINT (1 0)",
            "LINESTRING (0 0, 1 0, 2 0)",
        );
        compare_within(
            "point left, case 9",
            "POINT (0 0)",
            "LINESTRING (0 0, 1 0, 2 0)",
        );

        compare_within(
            "point left, case 10",
            "POINT (0 0)",
            "LINESTRING (0 0, 1 0, 1 1, 0 1, 0 0)",
        );

        compare_within(
            "point left, case 11",
            "POINT (0 1)",
            "LINESTRING (0 3, 0 0, 1 0, 0 1)",
        );

        compare_within(
            "point left, case 12",
            "POINT (0 2)",
            "LINESTRING (0 4, 0 2, 0 0, 2 0, 2 2, 0 2)",
        );

        compare_within(
            "point left, case 13",
            "POINT (0 2)",
            "LINESTRING (0 4, 0 2, 0 0, 2 0, 2 2, -1 2)",
        );

        compare_within(
            "point left, case 14",
            "POINT (0 2)",
            "LINESTRING (0 4, 0 2, 0 0, 2 0, 2 2, 0 2, -1 2)",
        );

        compare_within(
            "point left, case 15",
            "POINT (1 1)",
            "MULTILINESTRING ((0 0, 2 2), (0 2, 2 0))",
        );

        compare_within(
            "point left, case 16",
            "POINT (1 1)",
            "MULTILINESTRING ((0 0, 1 1), (1 1, 2 0))",
        );

        compare_within(
            "point left, case 17",
            "POINT (1 1)",
            "MULTILINESTRING ((0 0, 1 1), (1 1, 2 0), (1 1, 1 2))",
        );

        compare_within(
            "point left, case 18",
            "POINT (1 1)",
            "MULTILINESTRING ((0 0, 1 1, 2 2), (0 2, 1 1, 2 0))",
        );

        compare_within(
            "point left, case 19",
            "POINT (1 1)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );

        compare_within(
            "point left, case 20",
            "POINT (3 1)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );

        compare_within(
            "point left, case 21",
            "POINT (0 0)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );

        compare_within(
            "point left, case 22",
            "POINT (1 0)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );

        compare_within(
            "point left, case 23",
            "POINT (0.5 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1))",
        );

        compare_within(
            "point left, case 24",
            "POINT (2 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1))",
        );

        compare_within(
            "point left, case 25",
            "POINT (1 1)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1))",
        );

        compare_within(
            "point left, case 26",
            "POINT (1 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1))",
        );

        compare_within(
            "point left, case 27",
            "POINT (-1 1)",
            "POLYGON ((0 0, 2 0, 2 1, 2 2, 0 2, 0 0))",
        );

        compare_within(
            "point left, case 28",
            "POINT (1 1)",
            "POLYGON ((0 0, 2 0, 2 1, 2 2, 0 2, 0 0))",
        );

        compare_within(
            "point left, case 29",
            "POINT (-1 0)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );

        compare_within(
            "point left, case 30",
            "POINT (2 1)",
            "POLYGON ((0 0, 2 0, 2 1, 2 2, 0 2, 0 0))",
        );

        compare_within(
            "point left, case 31",
            "POINT (2 3)",
            "POLYGON ((0 0, 4 0, 4 4, 2 2, 0 4, 0 0))",
        );

        compare_within(
            "point left, case 32",
            "POINT (2 1)",
            "POLYGON ((0 0, 4 0, 4 4, 2 2, 0 4, 0 0))",
        );

        compare_within(
            "point left, case 33",
            "POINT (1 1)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((3 0, 5 0, 5 2, 3 2, 3 0)))",
        );

        compare_within(
            "point left, case 34",
            "POINT (2.5 1)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((3 0, 5 0, 5 2, 3 2, 3 0)))",
        );

        compare_within(
            "point left, case 35",
            "POINT (2 0)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((2 0, 4 0, 4 2, 2 2, 2 0)))",
        );

        compare_within(
            "point left, case 36",
            "POINT (-1 4)",
            "POLYGON ((0 0, 0 4, 4 2, 0 0))",
        );

        compare_within(
            "point left, case 37",
            "POINT (1 2)",
            "POLYGON ((0 0, 0 4, 4 2, 0 0))",
        );
    }

    #[test]
    fn multi_point_left() {
        compare_within(
            "multi point left, case 1",
            "MULTIPOINT (1 1, 2 2)",
            "POINT (1 1)",
        );

        compare_within(
            "multi point left, case 2",
            "MULTIPOINT (1 1, 2 2)",
            "MULTIPOINT (1 1, 2 2)",
        );

        compare_within(
            "multi point left, case 3",
            "MULTIPOINT (1 1, 2 2, 3 3)",
            "MULTIPOINT (1 1, 2 2)",
        );

        compare_within(
            "multi point left, case 4",
            "MULTIPOINT (1 1, 2 2)",
            "MULTIPOINT (1 1, 2 2, 3 3)",
        );

        compare_within(
            "multi point left, case 5",
            "MULTIPOINT (0 1, 1 1)",
            "LINESTRING (0 0, 2 0)",
        );

        compare_within(
            "multi point left, case 6",
            "MULTIPOINT (1 0, 1 1)",
            "LINESTRING (0 0, 2 0)",
        );

        compare_within(
            "multi point left, case 7",
            "MULTIPOINT (0.5 0, 1.5 0)",
            "LINESTRING (0 0, 2 0)",
        );

        compare_within(
            "multi point left, case 8",
            "MULTIPOINT (1 0, 0 0)",
            "LINESTRING (0 0, 2 0)",
        );

        compare_within(
            "multi point left, case 9",
            "MULTIPOINT (2 0, 1 0)",
            "LINESTRING (0 0, 2 0, 4 0)",
        );

        compare_within(
            "multi point left, case 10",
            "MULTIPOINT (0 0, 2 0)",
            "LINESTRING (0 0, 2 0, 4 0)",
        );

        compare_within(
            "multi point left, case 11",
            "MULTIPOINT (1 0, 2 0)",
            "LINESTRING (0 0, 1 0, 2 0, 3 0)",
        );

        compare_within(
            "multi point left, case 12",
            "MULTIPOINT (0 0, 3 0)",
            "LINESTRING (0 0, 1 0, 2 0, 3 0)",
        );

        compare_within(
            "multi point left, case 13",
            "MULTIPOINT (2 2, 0 0)",
            "MULTILINESTRING ((0 0, 2 2), (4 0, 2 2), (2 4, 2 2))",
        );

        compare_within(
            "multi point left, case 14",
            "MULTIPOINT (2 0, 4 0)",
            "MULTILINESTRING ((0 2, 2 0, 0 -2), (6 2, 4 0, 6 -2), (2 0, 4 0))",
        );

        compare_within(
            "multi point left, case 15",
            "MULTIPOINT (2 0, 4 0)",
            "MULTILINESTRING ((0 2, 2 0, 0 -2), (6 2, 4 0, 6 -2), (2 0, 4 0), (2 0, 2 2))",
        );

        compare_within(
            "multi point left, case 16",
            "MULTIPOINT (-1 2, 5 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "multi point left, case 17",
            "MULTIPOINT (2 2, 5 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "multi point left, case 18",
            "MULTIPOINT (0 2, 4 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "multi point left, case 19",
            "MULTIPOINT (0 2, 2 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "multi point left, case 20",
            "MULTIPOINT (0 0, 4 4)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "multi point left, case 21",
            "MULTIPOINT (0 0, 2 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "multi point left, case 22",
            "MULTIPOINT (3 5, 5 7)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "multi point left, case 23",
            "MULTIPOINT (3 3, 7 7)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "multi point left, case 24",
            "MULTIPOINT (5 5, 1 1)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "multi point left, case 25",
            "MULTIPOINT (1 1, 9 9)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "multi point left, case 26",
            "MULTIPOINT (1 3, 1 7)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "multi point left, case 27",
            "MULTIPOINT (-1 2, 5 2)",
            "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), ((6 0, 10 0, 10 4, 6 4, 6 0)))",
        );

        compare_within(
            "multi point left, case 28",
            "MULTIPOINT (2 2, 5 2)",
            "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), ((6 0, 10 0, 10 4, 6 4, 6 0)))",
        );

        compare_within(
            "multi point left, case 29",
            "MULTIPOINT (2 2, 8 2)",
            "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), ((6 0, 10 0, 10 4, 6 4, 6 0)))",
        );
    }

    #[test]
    fn line_left() {
        compare_within("line left, case 1", "LINESTRING (0 0, 4 0)", "POINT (2 0)");

        compare_within(
            "line left, case 2",
            "LINESTRING (0 0, 4 0)",
            "MULTIPOINT (1 0, 3 0)",
        );

        compare_within(
            "line left, case 3",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (0 0, 4 0)",
        );

        compare_within(
            "line left, case 4",
            "LINESTRING (0 0, 4 4)",
            "LINESTRING (0 4, 4 0)",
        );

        compare_within(
            "line left, case 5",
            "LINESTRING (0 0, 2 2, 4 0)",
            "LINESTRING (0 4, 2 2, 4 4)",
        );

        compare_within(
            "line left, case 6",
            "LINESTRING (0 0, 2 0)",
            "LINESTRING (5 0, 7 0)",
        );

        compare_within(
            "line left, case 7",
            "LINESTRING (0 0, 2 0)",
            "LINESTRING (3 0, 5 0)",
        );

        compare_within(
            "line left, case 8",
            "LINESTRING (0 0, 3 0)",
            "LINESTRING (2 0, 5 0)",
        );

        compare_within(
            "line left, case 9",
            "LINESTRING (1 0, 3 0)",
            "LINESTRING (0 0, 4 0)",
        );

        compare_within(
            "line left, case 10",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (1 0, 3 0)",
        );

        compare_within(
            "line left, case 11",
            "LINESTRING (0 0, 4 0, 2 2)",
            "LINESTRING (0 0, 4 0, 4 4)",
        );

        compare_within(
            "line left, case 12",
            "LINESTRING (0 0, 4 0, 4 4, 0 4, 0 0)",
            "LINESTRING (0 0, 4 0, 4 4, 0 4, 0 0)",
        );

        compare_within(
            "line left, case 13",
            "LINESTRING (0 2, 4 2)",
            "LINESTRING (0 0, 2 2, 4 0)",
        );

        compare_within(
            "line left, case 14",
            "LINESTRING (0 0, 2 0, 2 2, 4 2)",
            "MULTILINESTRING ((0 0, 2 0), (2 0, 2 2), (2 2, 4 2))",
        );

        compare_within(
            "line left, case 15",
            "LINESTRING (1 0, 3 0)",
            "LINESTRING (0 0, 2 0, 4 0)",
        );

        compare_within(
            "line left, case 16",
            "LINESTRING (1 0, 3 0)",
            "MULTILINESTRING ((0 0, 2 0), (2 0, 4 0))",
        );

        compare_within(
            "line left, case 17",
            "LINESTRING (0 0, 4 0)",
            "MULTILINESTRING ((0 0, 4 0), (0 2, 4 2))",
        );

        compare_within(
            "line left, case 18",
            "LINESTRING (0 0, 2 0, 2 2)",
            "MULTILINESTRING ((-1 0, 3 0), (2 -1, 2 3))",
        );

        compare_within(
            "line left, case 19",
            "LINESTRING (0 0, 6 0)",
            "MULTILINESTRING ((0 0, 3 0), (2 0, 5 0), (4 0, 6 0))",
        );

        compare_within(
            "line left, case 20",
            "LINESTRING (1 0, 5 0)",
            "LINESTRING (0 0, 4 0, 4 2, 5 2, 5 0, 2 0)",
        );

        compare_within(
            "line left, case 21",
            "LINESTRING (0 0, 6 0)",
            "MULTILINESTRING ((0 0, 3 0), (2 0, 5 0))",
        );

        compare_within(
            "line left, case 18",
            "LINESTRING (-2 3, -1 3)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );

        compare_within(
            "line left, case 19",
            "LINESTRING (1 3, 5 3)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );

        compare_within(
            "line left, case 20",
            "LINESTRING (-1 3, 3 3)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );

        compare_within(
            "line left, case 21",
            "LINESTRING (-2 -2, 2 2)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );

        compare_within(
            "line left, case 22",
            "LINESTRING (-1 0, 3 0)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );

        compare_within(
            "line left, case 23",
            "LINESTRING (1 0, 5 0)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );

        compare_within(
            "line left, case 24",
            "LINESTRING (0 0, 6 0, 6 6, 0 6, 0 0)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );

        compare_within(
            "line left, case 25",
            "LINESTRING (1 5, 2 5)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "line left, case 26",
            "LINESTRING (4 5, 6 5)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "line left, case 27",
            "LINESTRING (1 5, 5 5)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "line left, case 28",
            "LINESTRING (1 1, 5 5)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "line left, case 29",
            "LINESTRING (3 3, 3 7)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "line left, case 30",
            "LINESTRING (0.5 2, 3.5 2)",
            "MULTIPOLYGON (((0 0, 2 2, 0 4, 0 0)), ((2 2, 4 0, 4 4, 2 2)))",
        );

        compare_within(
            "line left, case 31",
            "LINESTRING (0.5 2, 3.5 2)",
            "MULTIPOLYGON (((0 0, 2 2, 0 4, 0 0)), ((2 2, 4 0, 4 4, 2 2)), ((2 2, 1 4, 3 4, 2 2)))",
        );

        compare_within(
            "line left, case 32",
            "LINESTRING (0.5 2, 7.5 2)",
            "MULTIPOLYGON (((0 0, 2 2, 0 4, 0 0)), ((2 2, 4 0, 6 2, 4 4, 2 2)), ((6 2, 8 0, 8 4, 6 2)))",
        );

        compare_within(
            "line left, case 33",
            "LINESTRING (0.5 2, 3.5 2)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((2 2, 4 2, 4 4, 2 4, 2 2)))",
        );

        compare_within(
            "line left, case 34",
            "LINESTRING (1 6, 4 6)",
            "POLYGON ((0 0, 8 0, 8 8, 5 8, 5 6, 3 6, 3 8, 0 8, 0 0))",
        );

        compare_within(
            "line left, case 35",
            "LINESTRING (-1 6, 4 6)",
            "POLYGON ((0 0, 8 0, 8 8, 5 8, 5 6, 3 6, 3 8, 0 8, 0 0))",
        );

        compare_within(
            "line left, case 36",
            "LINESTRING (1 5, 7 5)",
            "POLYGON ((0 0, 8 0, 8 8, 4 5, 0 8, 0 0))",
        );

        compare_within(
            "line left, case 37",
            "LINESTRING (-1 5, 9 5)",
            "POLYGON ((0 0, 8 0, 8 8, 4 5, 0 8, 0 0))",
        );

        compare_within(
            "line left, case 38",
            "LINESTRING (1 5, 7 5)",
            "MULTIPOLYGON (((4 5, 3 7, 5 7, 4 5)), ((0 0, 8 0, 8 8, 4 5, 0 8, 0 0)))",
        );

        compare_within(
            "line left, case 39",
            "LINESTRING (1 6, 7 6)",
            "POLYGON ((0 0, 8 0, 8 8, 5 8, 5 6, 3 6, 3 8, 0 8, 0 0))",
        );

        compare_within(
            "line left, case 40",
            "LINESTRING (1 5, 5 5)",
            "POLYGON ((0 0, 8 0, 8 8, 4 5, 0 8, 0 0))",
        );

        compare_within(
            "line left, case 41",
            "LINESTRING (1 6, 7 6)",
            "POLYGON ((0 0, 8 0, 8 8, 4 5, 0 8, 0 0))",
        );

        compare_within(
            "line left, case 42",
            "LINESTRING (0 -2, 0 2)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );

        compare_within(
            "line left, case 43",
            "LINESTRING (-1 4, 6 4)",
            "POLYGON ((0 0, 8 0, 8 8, 4 8, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "line left, case 44",
            "LINESTRING (1 4, 6 4)",
            "POLYGON ((0 0, 8 0, 8 8, 4 8, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "line left, case 45",
            "LINESTRING (6 4, -1 4)",
            "POLYGON ((0 0, 8 0, 8 8, 4 8, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "line left, case 46",
            "LINESTRING (6 4, 1 4)",
            "POLYGON ((0 0, 8 0, 8 8, 4 8, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "line left, case 47",
            "LINESTRING (2 1, 4 1)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((2 1, 3 2, 4 1, 3 0, 2 1)), ((4 0, 6 0, 6 2, 4 2, 4 0)))",
        );

        compare_within(
            "line left, case 48",
            "LINESTRING (0.5 2, 4 2)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((2 2, 4 2, 4 4, 2 4, 2 2)))",
        );

        compare_within(
            "line left, case 49",
            "LINESTRING (2 1, 4 1)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((2 1, 3 0, 4 1, 3 2, 2 1)), ((4 0, 6 0, 6 2, 4 2, 4 0)))",
        );

        compare_within(
            "line left, case 50",
            "LINESTRING (0.5 2, 7.5 2)",
            "MULTIPOLYGON (((0 0, 0 4, 2 2, 0 0)), ((2 2, 4 0, 6 2, 4 4, 2 2)), ((6 2, 8 0, 8 4, 6 2)))",
        );

        compare_within(
            "line left, case 51",
            "LINESTRING (1 5, 7 5)",
            "POLYGON ((0 0, 0 8, 4 5, 8 8, 8 0, 0 0))",
        );

        compare_within(
            "line left, case 52",
            "LINESTRING (5 5, 1 5)",
            "POLYGON ((0 0, 8 0, 8 8, 4 5, 0 8, 0 0))",
        );

        compare_within(
            "line left, case 53",
            "LINESTRING (1 6, 4 6)",
            "POLYGON ((0 0, 0 8, 3 8, 3 6, 5 6, 5 8, 8 8, 8 0, 0 0))",
        );

        compare_within(
            "line left, case 54",
            "LINESTRING (4 6, 1 6)",
            "POLYGON ((0 0, 8 0, 8 8, 5 8, 5 6, 3 6, 3 8, 0 8, 0 0))",
        );

        compare_within(
            "line left, case 55",
            "LINESTRING (4 6, 1 6)",
            "POLYGON ((0 0, 0 8, 3 8, 3 6, 5 6, 5 8, 8 8, 8 0, 0 0))",
        );

        compare_within(
            "line left, case 56",
            "LINESTRING (1 4, 7 4)",
            "MULTIPOLYGON (((0 0, 3 0, 3 4, 0 4, 0 0)), ((5 0, 8 0, 8 4, 5 4, 5 0)))",
        );

        compare_within(
            "line left, case 57",
            "LINESTRING (0 4, 5.5 4)",
            "MULTIPOLYGON (((0 0, 3 0, 3 4, 0 4, 0 0)), ((5 0, 8 0, 8 4, 5 4, 5 0)))",
        );

        compare_within(
            "line left, case 58",
            "LINESTRING (1 2, 7 2)",
            "MULTIPOLYGON (((0 0, 3 0, 3 4, 0 4, 0 0)), ((5 0, 8 0, 8 4, 5 4, 5 0)))",
        );

        compare_within(
            "line left, case 59",
            "LINESTRING (0 2, 5.5 2)",
            "MULTIPOLYGON (((0 0, 3 0, 3 4, 0 4, 0 0)), ((5 0, 8 0, 8 4, 5 4, 5 0)))",
        );

        compare_within(
            "line left, case 60",
            "LINESTRING (0 2, 8 2)",
            "POLYGON ((4 0, 8 0, 8 4, 4 4, 4 0))",
        );

        compare_within(
            "line left, case 61",
            "LINESTRING (0.5 2, 5.5 2)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((4 0, 6 0, 6 2, 4 2, 4 0)), ((2 2, 3 1, 4 2, 3 3, 2 2)))",
        );

        compare_within(
            "line left, case 62",
            "LINESTRING (1 1, 5 1)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((4 0, 6 0, 6 2, 4 2, 4 0)), ((2 1, 3 0, 4 1, 3 2, 2 1)))",
        );

        compare_within(
            "line left, case 63",
            "LINESTRING (1 6, 7 6)",
            "POLYGON ((0 0, 8 0, 8 8, 5 8, 5 6, 4 6, 3 6, 3 8, 0 8, 0 0))",
        );
    }

    #[test]
    fn multi_line_left() {
        compare_within(
            "multi line left, case 1",
            "MULTILINESTRING ((1 0, 3 0), (1 2, 3 2))",
            "MULTILINESTRING ((0 0, 4 0), (0 2, 4 2))",
        );

        compare_within(
            "multi line left, case 2",
            "MULTILINESTRING ((1 0, 3 0), (1 1, 3 1))",
            "MULTILINESTRING ((0 0, 4 0), (0 2, 4 2))",
        );

        compare_within(
            "multi line left, case 3",
            "MULTILINESTRING ((1 1, 3 1), (1 3, 3 3))",
            "MULTILINESTRING ((0 0, 4 0), (0 2, 4 2))",
        );

        compare_within(
            "multi line left, case 4",
            "MULTILINESTRING ((1 1, 3 1), (6 1, 8 1))",
            "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), ((5 0, 9 0, 9 4, 5 4, 5 0)))",
        );

        compare_within(
            "multi line left, case 5",
            "MULTILINESTRING ((1 1, 3 1), (10 1, 12 1))",
            "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), ((5 0, 9 0, 9 4, 5 4, 5 0)))",
        );

        compare_within(
            "multi line left, case 6",
            "MULTILINESTRING ((10 1, 12 1), (10 3, 12 3))",
            "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), ((5 0, 9 0, 9 4, 5 4, 5 0)))",
        );

        compare_within(
            "multi line left, case 7",
            "MULTILINESTRING ((0 0, 2 0), (2 0, 4 0))",
            "LINESTRING (0 0, 2 0, 4 0)",
        );
    }

    #[test]
    fn poly_left() {
        compare_within(
            "poly left, case 1",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            "POINT (2 2)",
        );

        compare_within(
            "poly left, case 2",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            "LINESTRING (0 0, 4 4)",
        );

        compare_within(
            "poly left, case 3",
            "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "poly left, case 4",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "poly left, case 5",
            "POLYGON ((2 2, 6 2, 6 6, 2 6, 2 2))",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "poly left, case 6",
            "POLYGON ((5 5, 9 5, 9 9, 5 9, 5 5))",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "poly left, case 7",
            "POLYGON ((1 1, 9 1, 9 9, 1 9, 1 1))",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "poly left, case 8",
            "POLYGON ((1 1, 9 1, 9 9, 1 9, 1 1), (2 2, 2 8, 8 8, 8 2, 2 2))",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "poly left, case 9",
            "POLYGON ((4 4, 6 4, 6 6, 4 6, 4 4))",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "poly left, case 10",
            "POLYGON ((2 2, 5 2, 5 5, 2 5, 2 2))",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "poly left, case 11",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            "POLYGON ((0 2, 4 2, 4 6, 0 6, 0 2))",
        );

        compare_within(
            "poly left, case 12",
            "POLYGON ((2 2, 4 2, 4 4, 2 4, 2 2))",
            "MULTIPOLYGON (((0 2, 2 2, 2 4, 0 4, 0 2)), ((2 4, 4 4, 4 6, 2 6, 2 4)), ((4 2, 6 2, 6 4, 4 4, 4 2)), ((2 0, 4 0, 4 2, 2 2, 2 0)))",
        );

        compare_within(
            "poly left, case 13",
            "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))",
            "MULTIPOLYGON (
                ((1 1, 2 0.5, 3 1, 2 1.5, 1 1)),
                ((3 1, 3.5 2, 3 3, 2.5 2, 3 1)),
                ((3 3, 2 3.5, 1 3, 2 2.5, 3 3)),
                ((1 3, 0.5 2, 1 1, 1.5 2, 1 3)))
            ",
        );

        compare_within(
            "poly left, case 14",
            "POLYGON ((1 4, 7 4, 7 5, 1 5, 1 4))",
            "MULTIPOLYGON (((4 5, 3 7, 5 7, 4 5)), ((0 0, 8 0, 8 8, 4 5, 0 8, 0 0)))",
        );
    }

    #[test]
    fn multi_poly_left() {
        compare_within(
            "multi poly left, case 1",
            "MULTIPOLYGON (((1 1, 2 1, 2 2, 1 2, 1 1)), ((3 1, 4 1, 4 2, 3 2, 3 1)))",
            "POLYGON ((0 0, 5 0, 5 3, 0 3, 0 0))",
        );

        compare_within(
            "multi poly left, case 2",
            "MULTIPOLYGON (((1 1, 2 1, 2 2, 1 2, 1 1)), ((6 1, 7 1, 7 2, 6 2, 6 1)))",
            "MULTIPOLYGON (((0 0, 4 0, 4 3, 0 3, 0 0)), ((5 0, 9 0, 9 3, 5 3, 5 0)))",
        );

        compare_within(
            "multi poly left, case 3",
            "MULTIPOLYGON (((1 1, 2 1, 2 2, 1 2, 1 1)), ((6 1, 7 1, 7 2, 6 2, 6 1)))",
            "MULTIPOLYGON (((0 0, 4 0, 4 3, 0 3, 0 0)), ((5 0, 9 0, 9 3, 5 3, 5 0)), ((10 0, 14 0, 14 3, 10 3, 10 0)))",
        );

        compare_within(
            "multi poly left, case 4",
            "MULTIPOLYGON (((1 1, 2 1, 2 2, 1 2, 1 1)), ((6 1, 7 1, 7 2, 6 2, 6 1)))",
            "POLYGON ((0 0, 4 0, 4 3, 0 3, 0 0))",
        );

        compare_within(
            "multi poly left, case 5",
            "MULTIPOLYGON (((1 1, 2 1, 2 2, 1 2, 1 1)), ((10 1, 11 1, 11 2, 10 2, 10 1)))",
            "MULTIPOLYGON (((0 0, 4 0, 4 3, 0 3, 0 0)), ((5 0, 9 0, 9 3, 5 3, 5 0)))",
        );

        compare_within(
            "multi poly left, case 6",
            "MULTIPOLYGON (((6 1, 7 1, 7 2, 6 2, 6 1)), ((8 1, 9 1, 9 2, 8 2, 8 1)))",
            "POLYGON ((0 0, 4 0, 4 3, 0 3, 0 0))",
        );
    }

    #[test]
    fn empty_geo() {
        let sq = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";
        // empty on the left
        compare_within("empty multipoint left", "MULTIPOINT EMPTY", sq);
        compare_within("empty multilinestring left", "MULTILINESTRING EMPTY", sq);
        compare_within("empty polygon left", "POLYGON EMPTY", sq);
        // empty on the right
        compare_within("empty multipoint right", sq, "MULTIPOINT EMPTY");
        compare_within("empty multilinestring right", sq, "MULTILINESTRING EMPTY");
        compare_within("empty polygon right", sq, "POLYGON EMPTY");
    }
}
