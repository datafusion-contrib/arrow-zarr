use datafusion::common::JoinSide;

use super::geo_owned::{Dir, Edge, GeoOwned, GeoView, LineSegment, Point, SegmentTrait, ToPoints};
use super::spatial_predicate::SpatialRelationType;

//*************************************************
// Various helpers to check different conditions on
// the components that make up teh geometries.
//*************************************************
const EPS: f64 = 1e-10;
const N_STRIPES: usize = 64;

enum LineIntersection {
    NoCrossing,
    Overlap,
    // t parameter on the left segment, clamped to [0.0, 1.0]
    Crossing(f64),
}

// Returns the t parameter in [0.0, 1.0] where the point lies on the segment,
// or None if it doesn't.
fn point_on_segment_t(p: &Point, seg: &impl SegmentTrait) -> Option<f64> {
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

// Check for an intersection between 2 line segments.
fn line_intersects_line(seg_a: &LineSegment, seg_b: &LineSegment) -> LineIntersection {
    if !seg_a.x_ranges_overlap(seg_b) {
        return LineIntersection::NoCrossing;
    }
    let (a1, _) = seg_a.to_points();
    let (b1, b2) = seg_b.to_points();
    let cross = seg_a.cross(seg_b);

    if cross.abs() < EPS {
        let len_sq = seg_a.norm_sq();
        assert!(len_sq >= EPS * EPS, "Degenerate zero-length line segment");
        if !seg_a.is_collinear_with(b1) {
            return LineIntersection::NoCrossing;
        }
        let t1 = seg_a.projection_t(b1);
        let t2 = seg_a.projection_t(b2);
        let lo = t1.min(t2);
        let hi = t1.max(t2);
        if lo <= 1.0 + EPS && hi >= -EPS {
            LineIntersection::Overlap
        } else {
            LineIntersection::NoCrossing
        }
    } else {
        let f = b1.diff(a1);
        let t = -seg_b.cross(&f) / cross;
        let u = -seg_a.cross(&f) / cross;
        if (-EPS..=(1.0 + EPS)).contains(&t) && (-EPS..=(1.0 + EPS)).contains(&u) {
            LineIntersection::Crossing(t.clamp(0.0, 1.0))
        } else {
            LineIntersection::NoCrossing
        }
    }
}

// Checks for a ray going from a point and extending at constant y,
// along x, to its right, and see if it crosses a segment.
fn ray_crosses_segment(p: &Point, seg: &impl SegmentTrait) -> (bool, bool) {
    let (px, _) = p.xy();
    let (x1, x2) = seg.xs();
    if x1.max(x2) < px {
        return (false, false);
    }
    if point_on_segment_t(p, seg).is_some() {
        return (false, true);
    }
    if !seg.y_range_contains(p) {
        return (false, false);
    }
    (seg.x_intercept_at_point(p) > px, false)
}

// Checks if the 2 polygon edges have their interior on the
// same side, meaning the corresponding polygons have at least
// some intersection between their interiors.
fn interiors_on_same_side(left: &Edge, right: &Edge) -> bool {
    let same_dir = left.dot(right) > 0.0;
    same_dir == (left.interior_on_left() == right.interior_on_left())
}

enum MidpointResult {
    // midpoint on right edge AND collinear with left segment
    OnEdge,
    // reflex end-vertex (u≈1, angle > π), not collinear
    MidpointInside,
    // on boundary but not collinear, not reflex
    MidpointOutside,
    // midpoint not on right edge
    NoContact,
}

// Checks if the left segment's midpoint falls on the right edge
// (this check does not include ray casting).
fn segment_midpoint_check(left: &impl SegmentTrait, right: &Edge) -> MidpointResult {
    let mid = left.midpoint();
    if point_on_segment_t(&mid, right).is_some() {
        if left.cross(right).abs() < EPS {
            return MidpointResult::OnEdge;
        }
        let u = right.projection_t(&mid);
        if u >= 1.0 - EPS && right.is_reflex() {
            return MidpointResult::MidpointInside;
        }
        return MidpointResult::MidpointOutside;
    }
    MidpointResult::NoContact
}

// Returns the interval [t_lo, t_hi] on the left segment that is covered by the right segment,
// if the two segments are collinear and overlapping. Returns None if not collinear, not
// overlapping, or the overlap is degenerate.
fn segment_collinear_ts(
    seg_a: &impl SegmentTrait,
    seg_b: &impl SegmentTrait,
) -> Option<(f64, f64)> {
    if !seg_a.x_ranges_overlap(seg_b) {
        return None;
    }
    assert!(
        seg_a.norm_sq() >= EPS * EPS,
        "Degenerate zero-length line segment"
    );
    if seg_a.cross(seg_b).abs() >= EPS {
        return None; // not parallel
    }
    let (b1, b2) = seg_b.to_points();
    if !seg_a.is_collinear_with(b1) {
        return None; // parallel but not collinear
    }
    let t1 = seg_a.projection_t(b1);
    let t2 = seg_a.projection_t(b2);
    let t_lo = t1.min(t2).max(0.0);
    let t_hi = t1.max(t2).min(1.0);
    if t_lo >= t_hi - EPS {
        return None; // degenerate or no overlap
    }
    Some((t_lo, t_hi))
}

// Like `segment_collinear_ts`, but also verifies that the two polygon edges'
// interiors face the same physical side.
fn segment_collinear_ts_same_interior(left: &Edge, right: &Edge) -> Option<(f64, f64)> {
    let (t_lo, t_hi) = segment_collinear_ts(left, right)?;
    if !interiors_on_same_side(left, right) {
        return None;
    }
    Some((t_lo, t_hi))
}

// Returns (t, is_entering) where t is the parameter on the left segment where
// it crosses the right polygon edge, and is_entering is true if the left segment
// is entering the polygon interior. Uses exclude-start/include-end convention on
// u to avoid double-counting shared poly vertices.
fn segment_crossing_t(left: &impl SegmentTrait, right: &Edge) -> Option<(f64, bool)> {
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

    // this is the exclude-start/include-end convention, only check of
    // u = 1, not u = 0.
    if u > EPS && u < 1.0 - EPS {
        let is_entering = (cross < 0.0) == right.interior_on_left();
        return Some((t, is_entering));
    }
    if (1.0 - EPS..=1.0 + EPS).contains(&u) {
        if let Some(is_entering) = right.crossing_at_vertex(left) {
            return Some((t, is_entering));
        }
    }
    None
}

//*************************************************
// Helpers to validate some conditions on each component group,
// i.e. rows that share the same component on either the left
// or the right, when iterating over all the component pairs
// to check.
//*************************************************

// Checks whether the sorted t-values (t_lo/t_hi pairs from segment_collinear_ts)
// cover [0, 1], by cancelling t values in pairs.
fn collinear_ts_cover(ts: &mut [f64]) -> bool {
    if ts.is_empty() {
        return false;
    }
    ts.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let n = ts.len();
    if ts[0] > EPS || ts[n - 1] < 1.0 - EPS {
        return false;
    }
    ts[1..n - 1].chunks(2).all(|c| (c[0] - c[1]).abs() < EPS)
}

// Checks whether a left line segment is fully covered by the union
// of directed polygon crossings and collinear boundary intervals.
// Crossing pairs are (t, is_entering); collinear_ts are [t_lo, t_hi].
fn crossing_and_collinear_contained(
    crossing_pairs: &mut [(f64, bool)],
    collinear_ts: &mut Vec<f64>,
) -> bool {
    crossing_pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));

    // Convert directed crossings into [t_lo, t_hi] intervals and push
    // into collinear_ts.
    let mut start_t: Option<f64> = None;
    for &(t, is_entering) in crossing_pairs.iter() {
        if is_entering {
            start_t = Some(t);
        } else if let Some(st) = start_t {
            collinear_ts.push(st);
            collinear_ts.push(t);
            start_t = None;
        } else {
            // Exiting without a prior entry — segment started inside.
            collinear_ts.push(0.0);
            collinear_ts.push(t);
        }
    }
    if let Some(st) = start_t {
        // Still inside at end of segment.
        collinear_ts.push(st);
        collinear_ts.push(1.0);
    }

    collinear_ts_cover(collinear_ts)
}

// Checks that all left component groups have matched with
// something from the right side.
fn all_groups_matched(pairs: &mut [(usize, usize)], check: impl Fn(usize, usize) -> bool) -> bool {
    pairs.sort_unstable_by_key(|&(i, _)| i);
    let mut matched = false;
    let mut prev_li = pairs[0].0;
    for &(li, ri) in pairs.iter() {
        if li != prev_li {
            if !matched {
                return false;
            }
            matched = false;
            prev_li = li;
        }
        if !matched && check(li, ri) {
            matched = true;
        }
    }
    matched
}

//*************************************************
// Main primitive checks that will be used for geospatial
// operations, intersections and full containment
// (of left side by right side).
//*************************************************
fn intersects(
    left: &GeoView<'_>,
    right: &GeoView<'_>,
    pairs: &mut [(usize, usize)],
    flip_pairs: bool,
) -> bool {
    if pairs.is_empty() {
        return false;
    }
    macro_rules! lr {
        ($i:expr, $j:expr) => {
            if flip_pairs {
                ($j, $i)
            } else {
                ($i, $j)
            }
        };
    }
    match (left, right) {
        // Point × *
        (GeoView::Points { points: lpoints }, GeoView::Points { points: rpoints }) => {
            pairs.iter().any(|&(i, j)| {
                let (li, ri) = lr!(i, j);
                lpoints[li].is_equal(&rpoints[ri])
            })
        }
        // check for point on the line, but exclude boundary points. Boundaries
        // exclude everything, so if a point is on a boundary point, it cancels
        // any other intersections it might have.
        (GeoView::Points { points: lpoints }, GeoView::LineSegments { lines, b1s, b2s }) => {
            pairs.sort_unstable_by_key(|&(i, j)| lr!(i, j).0);
            let mut on_segment = false;
            let mut on_boundary = false;
            let mut prev_li = lr!(pairs[0].0, pairs[0].1).0;
            for &(i, j) in pairs.iter() {
                let (li, ri) = lr!(i, j);
                if li != prev_li {
                    if on_segment && !on_boundary {
                        return true;
                    }
                    on_segment = false;
                    on_boundary = false;
                    prev_li = li;
                }
                if on_boundary {
                    continue;
                }
                let p = &lpoints[li];
                let (seg_p1, seg_p2) = lines[ri].to_points();
                if (b1s[ri] && p.is_equal(seg_p1)) || (b2s[ri] && p.is_equal(seg_p2)) {
                    on_boundary = true;
                    continue;
                }
                if point_on_segment_t(p, &lines[ri]).is_some() {
                    on_segment = true;
                }
            }
            on_segment && !on_boundary
        }
        // TODO: this part assumes that the edges themselves are not part of the
        // right side geo, which won't always be true. It's fine for now, but
        // the poly edges should contain a flag indicates if the edges are included
        // or not.
        (GeoView::Points { points: lpoints }, GeoView::PolyEdges { edges, .. }) => {
            pairs.sort_unstable_by_key(|&(i, j)| lr!(i, j).0);
            let mut parity = false;
            let mut on_edge = false;
            let mut prev_li = lr!(pairs[0].0, pairs[0].1).0;
            for &(i, j) in pairs.iter() {
                let (li, ri) = lr!(i, j);
                if li != prev_li {
                    if parity && !on_edge {
                        return true;
                    }
                    parity = false;
                    on_edge = false;
                    prev_li = li;
                }
                let p = &lpoints[li];
                if !on_edge {
                    let (crosses, edge) = ray_crosses_segment(p, &edges[ri]);
                    if edge {
                        on_edge = true;
                    } else if crosses {
                        parity = !parity;
                    }
                }
            }
            parity && !on_edge
        }

        // Line × *
        (GeoView::LineSegments { .. }, GeoView::Points { .. }) => {
            intersects(right, left, pairs, true)
        }
        (
            GeoView::LineSegments {
                lines: alines,
                b1s: ab1s,
                b2s: ab2s,
            },
            GeoView::LineSegments {
                lines: blines,
                b1s: bb1s,
                b2s: bb2s,
            },
        ) => {
            pairs.sort_unstable_by_key(|&(i, j)| lr!(i, j).0);
            let mut crossings: Vec<f64> = Vec::new();
            let mut boundaries: Vec<f64> = Vec::new();
            let first_li = lr!(pairs[0].0, pairs[0].1).0;
            let mut prev_li = first_li;
            if ab1s[first_li] {
                boundaries.push(0.0);
            }
            if ab2s[first_li] {
                boundaries.push(1.0);
            }

            // Here we check all the crossings, but remove any crossings at a
            // boundary point.
            macro_rules! check_group {
                () => {{
                    crossings.retain(|&tc| !boundaries.iter().any(|&tb| (tc - tb).abs() < EPS));
                    if !crossings.is_empty() {
                        return true;
                    }
                    crossings.clear();
                    boundaries.clear();
                }};
            }

            for &(i, j) in pairs.iter() {
                let (li, ri) = lr!(i, j);
                if li != prev_li {
                    check_group!();
                    prev_li = li;
                    if ab1s[li] {
                        boundaries.push(0.0);
                    }
                    if ab2s[li] {
                        boundaries.push(1.0);
                    }
                }
                match line_intersects_line(&alines[li], &blines[ri]) {
                    LineIntersection::Overlap => return true,
                    LineIntersection::Crossing(t) => crossings.push(t),
                    LineIntersection::NoCrossing => {}
                }

                let (bp1, bp2) = blines[ri].to_points();
                if bb1s[ri] {
                    if let Some(t) = point_on_segment_t(bp1, &alines[li]) {
                        boundaries.push(t);
                    }
                }
                if bb2s[ri] {
                    if let Some(t) = point_on_segment_t(bp2, &alines[li]) {
                        boundaries.push(t);
                    }
                }
            }
            check_group!();
            false
        }
        // TODO: right now in this function edges are assumed to represent the
        // interior of the poly, not the boundary itself (i.e. we are only checking
        // for proper crossings). If we wanted to represent the poly's "not exterior",
        // and intersect something with it, this would not be correct. It's not a
        // problem for now, but I do need to do this correctly at some point.
        (GeoView::LineSegments { lines: alines, .. }, GeoView::PolyEdges { edges, .. }) => {
            pairs.sort_unstable_by_key(|&(i, j)| lr!(i, j).0);
            let first_li = lr!(pairs[0].0, pairs[0].1).0;
            let mut parity = false;
            let mut parity_valid = true;
            let mut midpoint_ok = false;
            let mut prev_li = first_li;
            for &(i, j) in pairs.iter() {
                let (li, ri) = lr!(i, j);
                if li != prev_li {
                    if (parity_valid && parity) || midpoint_ok {
                        return true;
                    }
                    parity = false;
                    parity_valid = true;
                    midpoint_ok = false;
                    prev_li = li;
                }
                let mid = alines[li].midpoint();
                if segment_crossing_t(&alines[li], &edges[ri]).is_some() {
                    return true;
                }
                match segment_midpoint_check(&alines[li], &edges[ri]) {
                    MidpointResult::MidpointInside => {
                        midpoint_ok = true;
                        parity_valid = false;
                    }
                    MidpointResult::MidpointOutside => parity_valid = false,
                    MidpointResult::OnEdge => parity_valid = false,
                    _ => {}
                }
                if !midpoint_ok && parity_valid {
                    let (crosses, _) = ray_crosses_segment(&mid, &edges[ri]);
                    if crosses {
                        parity = !parity;
                    }
                }
            }
            (parity_valid && parity) || midpoint_ok
        }

        // Poly × *
        (GeoView::PolyEdges { .. }, GeoView::Points { .. }) => intersects(right, left, pairs, true),
        (GeoView::PolyEdges { .. }, GeoView::LineSegments { .. }) => {
            intersects(right, left, pairs, true)
        }
        // TODO: similar issue to the line x poly case above, the edges themselves are excluded
        // from the check, that's not correct, but not a problem for now.
        (GeoView::PolyEdges { edges: aedges, .. }, GeoView::PolyEdges { edges, .. }) => {
            // Part 1: check edge interactions and whether B contains A,
            // based on direct intersections and midpoitn ray casting.
            pairs.sort_unstable_by_key(|&(i, j)| lr!(i, j).0);
            let first_li = lr!(pairs[0].0, pairs[0].1).0;
            let mut prev_li = first_li;
            let mut parity = false;
            let mut parity_valid = true;
            for &(i, j) in pairs.iter() {
                let (li, ri) = lr!(i, j);
                if li != prev_li {
                    if parity_valid && parity {
                        return true;
                    }
                    parity = false;
                    parity_valid = true;
                    prev_li = li;
                }
                let mid = aedges[li].midpoint();
                if segment_crossing_t(&aedges[li], &edges[ri]).is_some() {
                    return true;
                }
                match segment_midpoint_check(&aedges[li], &edges[ri]) {
                    MidpointResult::OnEdge => {
                        if interiors_on_same_side(&aedges[li], &edges[ri]) {
                            return true;
                        }
                    }
                    MidpointResult::MidpointInside => return true,
                    MidpointResult::MidpointOutside => parity_valid = false,
                    _ => {}
                }
                if parity_valid {
                    let (crosses, _) = ray_crosses_segment(&mid, &edges[ri]);
                    if crosses {
                        parity = !parity;
                    }
                }
            }
            if parity_valid && parity {
                return true;
            }

            // Part 2: Check whether A contains B, same as part 1 but with the left
            // and right side flipped. We all the direct intesections and overlaps
            // since those would have been caught in step 1.
            pairs.sort_unstable_by_key(|&(i, j)| lr!(i, j).1);
            let first_ri = lr!(pairs[0].0, pairs[0].1).1;
            let mut prev_ri = first_ri;
            let mut b_parity = false;
            let mut b_parity_valid = true;
            for &(i, j) in pairs.iter() {
                let (li, ri) = lr!(i, j);
                if ri != prev_ri {
                    if b_parity_valid && b_parity {
                        return true;
                    }
                    b_parity = false;
                    b_parity_valid = true;
                    prev_ri = ri;
                }
                let mid = edges[ri].midpoint();
                match segment_midpoint_check(&edges[ri], &aedges[li]) {
                    MidpointResult::MidpointInside => return true,
                    MidpointResult::MidpointOutside => b_parity_valid = false,
                    _ => {}
                }
                if b_parity_valid {
                    let (crosses, _) = ray_crosses_segment(&mid, &aedges[li]);
                    if crosses {
                        b_parity = !b_parity;
                    }
                }
            }
            b_parity_valid && b_parity
        }
    }
}

fn fully_contained(left: &GeoView<'_>, right: &GeoView<'_>, pairs: &mut [(usize, usize)]) -> bool {
    let n_left = match left {
        GeoView::Points { points } => points.len(),
        GeoView::LineSegments { lines, .. } => lines.len(),
        GeoView::PolyEdges { edges, .. } => edges.len(),
    };
    // vacuously true: nothing to contain
    if n_left == 0 {
        return true;
    }
    if pairs.is_empty() {
        return false;
    }

    let mut seen = vec![false; n_left];
    for &(i, _) in pairs.iter() {
        seen[i] = true;
    }
    // If any of the left components don't appear in the pairs to check, then
    // the left side can't be fully contained.
    if seen.iter().any(|&s| !s) {
        return false;
    }

    match (left, right) {
        // --- Point × * ---
        (GeoView::Points { points: lpoints }, GeoView::Points { points: rpoints }) => {
            all_groups_matched(pairs, |li, ri| lpoints[li].is_equal(&rpoints[ri]))
        }
        (GeoView::Points { points: lpoints }, GeoView::LineSegments { lines, b1s, b2s }) => {
            pairs.sort_unstable_by_key(|&(i, _)| i);
            let mut on_segment = false;
            let mut on_boundary = false;
            let mut prev_li = pairs[0].0;
            for &(li, ri) in pairs.iter() {
                if li != prev_li {
                    if !on_segment || on_boundary {
                        return false;
                    }
                    on_segment = false;
                    on_boundary = false;
                    prev_li = li;
                }
                if on_boundary {
                    continue;
                }
                let p = &lpoints[li];
                let (seg_p1, seg_p2) = lines[ri].to_points();
                if (b1s[ri] && p.is_equal(seg_p1)) || (b2s[ri] && p.is_equal(seg_p2)) {
                    on_boundary = true;
                    continue;
                }
                if point_on_segment_t(p, &lines[ri]).is_some() {
                    on_segment = true;
                }
            }
            on_segment && !on_boundary
        }
        // TODO: this is assuming that the edges are part of the geometry, which
        // is fine for now, but there should be a flag to indicate if the edges
        // theselves count or not.
        (GeoView::Points { points: lpoints }, GeoView::PolyEdges { edges, .. }) => {
            pairs.sort_unstable_by_key(|&(i, _)| i);
            let mut parity = false;
            let mut on_edge = false;
            let mut crosses;
            let mut prev_li = pairs[0].0;
            for &(li, ri) in pairs.iter() {
                if li != prev_li {
                    if !parity && !on_edge {
                        return false;
                    }
                    parity = false;
                    on_edge = false;
                    prev_li = li;
                }
                if !on_edge {
                    (crosses, on_edge) = ray_crosses_segment(&lpoints[li], &edges[ri]);
                    if crosses {
                        parity = !parity;
                    }
                }
            }
            parity || on_edge
        }

        // --- Line × * ---
        (GeoView::LineSegments { .. }, GeoView::Points { .. }) => false,
        (
            // TODO: ignoring the endpoint boundaries on the line is not
            // correct here, but it doesn't really impact anything for now.
            // I still need to fix this at some point though.
            GeoView::LineSegments { lines: alines, .. },
            GeoView::LineSegments { lines: blines, .. },
        ) => {
            pairs.sort_unstable_by_key(|&(i, _)| i);
            let first_li = pairs[0].0;
            let mut prev_li = first_li;
            let mut ts: Vec<f64> = Vec::new();

            for &(li, ri) in pairs.iter() {
                if li != prev_li {
                    if !collinear_ts_cover(&mut ts) {
                        return false;
                    }
                    ts.clear();
                    prev_li = li;
                }
                if let Some((t_lo, t_hi)) = segment_collinear_ts(&alines[li], &blines[ri]) {
                    ts.push(t_lo);
                    ts.push(t_hi);
                }
            }
            collinear_ts_cover(&mut ts)
        }
        (
            // TODO: ignoring the endpoint boundaries on the line is not
            // correct here, but it doesn't really impact anything for now.
            // I still need to fix this at some point though. And also,
            // this is again assuming the right side edges are included in
            // the geo, which is not necessarily true (it is for the function
            // call we are making for st_within, but that won't always be the case.)
            GeoView::LineSegments { lines: alines, .. },
            GeoView::PolyEdges { edges, .. },
        ) => {
            pairs.sort_unstable_by_key(|&(i, _)| i);
            let mut prev_li = pairs[0].0;
            let mut crossing_pairs: Vec<(f64, bool)> = Vec::new();
            let mut collinear_ts: Vec<f64> = Vec::new();
            let mut parity = false;
            let mut parity_valid = true;
            let mut midpoint_inside = false;

            macro_rules! check_group {
                () => {{
                    if !crossing_pairs.is_empty() {
                        crossing_and_collinear_contained(&mut crossing_pairs, &mut collinear_ts)
                    } else if midpoint_inside || (parity_valid && parity) {
                        true
                    } else if !collinear_ts.is_empty() {
                        crossing_and_collinear_contained(&mut crossing_pairs, &mut collinear_ts)
                    } else {
                        false
                    }
                }};
            }

            for &(li, ri) in pairs.iter() {
                if li != prev_li {
                    if !check_group!() {
                        return false;
                    }
                    crossing_pairs.clear();
                    collinear_ts.clear();
                    parity = false;
                    parity_valid = true;
                    midpoint_inside = false;
                    prev_li = li;
                }
                let mid = alines[li].midpoint();
                if let Some(pair) = segment_crossing_t(&alines[li], &edges[ri]) {
                    crossing_pairs.push(pair);
                }
                if let Some((t_lo, t_hi)) = segment_collinear_ts(&alines[li], &edges[ri]) {
                    collinear_ts.push(t_lo);
                    collinear_ts.push(t_hi);
                }
                if !midpoint_inside {
                    match segment_midpoint_check(&alines[li], &edges[ri]) {
                        MidpointResult::MidpointInside => midpoint_inside = true,
                        MidpointResult::MidpointOutside => parity_valid = false,
                        MidpointResult::OnEdge => parity_valid = false,
                        _ => {}
                    }
                }
                if !midpoint_inside && parity_valid {
                    let (crosses, _) = ray_crosses_segment(&mid, &edges[ri]);
                    if crosses {
                        parity = !parity;
                    }
                }
            }
            check_group!()
        }

        // --- Poly × * ---
        (GeoView::PolyEdges { .. }, GeoView::Points { .. }) => false,
        (GeoView::PolyEdges { .. }, GeoView::LineSegments { .. }) => false,
        (
            GeoView::PolyEdges {
                edges: aedges,
                poly_ids: left_poly_ids,
            },
            GeoView::PolyEdges {
                edges,
                poly_ids: right_poly_ids,
            },
        ) => {
            let n_right_polys = right_poly_ids
                .iter()
                .copied()
                .max()
                .map(|m| m + 1)
                .unwrap_or(0);
            let n_left_polys = left_poly_ids
                .iter()
                .copied()
                .max()
                .map(|m| m + 1)
                .unwrap_or(0);
            // Maps left_poly_id → right_poly_id that contains it.
            let mut left_right_poly_map: Vec<Option<usize>> = vec![None; n_left_polys];

            pairs.sort_unstable_by_key(|&(i, _)| i);
            let mut prev_li = pairs[0].0;
            let mut collinear_ts: Vec<f64> = Vec::new();
            let mut collinear_poly_id: Option<usize> = None;
            let mut parity_vec: Vec<(bool, bool)> = vec![(false, true); n_right_polys];
            let mut midpoint_poly_id: Option<usize> = None;

            // The complexity here comes from the fact that all the dges on the left
            // being contained doesn't mean the poly is contained. The latter is only
            // true if the containment is from a single polygon on the right, which
            // we need to check for the multipoly on the right case. And then there can
            // be mulit polys on the left, so we need to keep track of left and right
            // side polygon indices.
            macro_rules! check_group {
                () => {{
                    let right_pid =
                        if let Some(rpid) = midpoint_poly_id {
                            Some(rpid)
                        } else if let Some(rpid) = parity_vec.iter().enumerate().find_map(
                            |(rpid, (parity, parity_valid))| {
                                (*parity_valid && *parity).then_some(rpid)
                            },
                        ) {
                            Some(rpid)
                        } else if !collinear_ts.is_empty() && collinear_ts_cover(&mut collinear_ts)
                        {
                            collinear_poly_id
                        } else {
                            None
                        };
                    match right_pid {
                        None => false,
                        Some(rpid) => {
                            let left_pid = left_poly_ids[prev_li];
                            match left_right_poly_map[left_pid] {
                                None => {
                                    left_right_poly_map[left_pid] = Some(rpid);
                                    true
                                }
                                Some(existing) if existing == rpid => true,
                                _ => false,
                            }
                        }
                    }
                }};
            }

            for &(li, ri) in pairs.iter() {
                if li != prev_li {
                    if !check_group!() {
                        return false;
                    }
                    collinear_ts.clear();
                    collinear_poly_id = None;
                    parity_vec.fill((false, true));
                    midpoint_poly_id = None;
                    prev_li = li;
                }
                let mid = aedges[li].midpoint();
                let rpid = right_poly_ids[ri];
                if segment_crossing_t(&aedges[li], &edges[ri]).is_some() {
                    return false;
                }
                if let Some((t_lo, t_hi)) =
                    segment_collinear_ts_same_interior(&aedges[li], &edges[ri])
                {
                    match collinear_poly_id {
                        None => collinear_poly_id = Some(rpid),
                        Some(existing) if existing != rpid => return false,
                        _ => {}
                    }
                    collinear_ts.push(t_lo);
                    collinear_ts.push(t_hi);
                }
                if midpoint_poly_id.is_none() {
                    match segment_midpoint_check(&aedges[li], &edges[ri]) {
                        MidpointResult::MidpointInside => midpoint_poly_id = Some(rpid),
                        MidpointResult::MidpointOutside => parity_vec[rpid].1 = false,
                        MidpointResult::OnEdge => parity_vec[rpid].1 = false,
                        _ => {}
                    }
                }
                if midpoint_poly_id.is_none() && parity_vec[rpid].1 {
                    let (crosses, _) = ray_crosses_segment(&mid, &edges[ri]);
                    if crosses {
                        parity_vec[rpid].0 ^= true;
                    }
                }
            }
            check_group!()
        }
    }
}

//*************************************************
// The main struct that is exposed from this module, that
// handles the interface for join refinement.
//*************************************************
pub struct ParsedGeometry {
    geo: GeoOwned,
    pub(crate) x_min: f64,
    pub(crate) x_max: f64,
    pub(crate) y_min: f64,
    pub(crate) y_max: f64,
}

impl ParsedGeometry {
    pub fn from_wkb(bytes: &[u8]) -> Self {
        let geo = GeoOwned::from_wkb(bytes);
        let (x_min, x_max) = geo.x_bounds();
        let (y_min, y_max) = geo.y_bounds();
        Self {
            geo,
            x_min,
            x_max,
            y_min,
            y_max,
        }
    }

    pub fn join_with_known_bbox(
        &self,
        other: &ParsedGeometry,
        predicate: &SpatialRelationType,
        side: JoinSide,
    ) -> bool {
        match predicate {
            SpatialRelationType::Within => {
                let (lx0, lx1, ly0, ly1, rx0, rx1, ry0, ry1) = match side {
                    JoinSide::Left => (
                        self.x_min,
                        self.x_max,
                        self.y_min,
                        self.y_max,
                        other.x_min,
                        other.x_max,
                        other.y_min,
                        other.y_max,
                    ),
                    JoinSide::Right => (
                        other.x_min,
                        other.x_max,
                        other.y_min,
                        other.y_max,
                        self.x_min,
                        self.x_max,
                        self.y_min,
                        self.y_max,
                    ),
                    JoinSide::None => panic!("JoinSide::None is not valid for spatial join"),
                };
                if !(lx0 >= rx0 && lx1 <= rx1 && ly0 >= ry0 && ly1 <= ry1) {
                    return false;
                }
            }
        }

        self.join(other, predicate, side)
    }

    pub fn join(
        &self,
        other: &ParsedGeometry,
        predicate: &SpatialRelationType,
        side: JoinSide,
    ) -> bool {
        if self.geo.num_components() == 0 || other.geo.num_components() == 0 {
            return false;
        }
        match predicate {
            SpatialRelationType::Within => {
                let (left, right) = match side {
                    JoinSide::Left => (&self.geo, &other.geo),
                    JoinSide::Right => (&other.geo, &self.geo),
                    JoinSide::None => panic!("JoinSide::None is not valid for spatial join"),
                };

                // 1a. left must not intersect right's holes
                if let Some((right_holes, right_holes_stripes)) = right.holes() {
                    let (left_nex, left_nex_stripes) = left.not_exterior();
                    let mut pairs = left_nex_stripes.candidate_pairs(right_holes_stripes);
                    if intersects(&left_nex, &right_holes, &mut pairs, false) {
                        return false;
                    }
                }

                // 1b. not_exterior(left) ⊆ not_exterior(right)
                let (left_nex, left_nex_stripes) = left.not_exterior();
                let (right_nex, right_nex_stripes) = right.not_exterior();
                let mut pairs = left_nex_stripes.candidate_pairs(right_nex_stripes);
                if !fully_contained(&left_nex, &right_nex, &mut pairs) {
                    return false;
                }

                // 2. I(left) ∩ I(right) ≠ ∅
                let (left_int, left_int_stripes) = left.interior();
                let (right_int, right_int_stripes) = right.interior();
                let mut pairs = left_int_stripes.candidate_pairs(right_int_stripes);
                if !intersects(&left_int, &right_int, &mut pairs, false) {
                    return false;
                }

                true
            }
        }
    }
}

#[cfg(test)]
mod test_helpers {
    use datafusion::common::JoinSide;
    use geos::Geom;

    pub(super) use super::super::test_utils::wkt_to_wkb;
    use super::*;

    pub(super) fn compare_within(label: &str, wkt_a: &str, wkt_b: &str) {
        let bytes_a = wkt_to_wkb(wkt_a);
        let bytes_b = wkt_to_wkb(wkt_b);

        let geos_a = geos::Geometry::new_from_wkb(&bytes_a).unwrap();
        let geos_b = geos::Geometry::new_from_wkb(&bytes_b).unwrap();
        let geos_result = geos_a.within(&geos_b).unwrap();

        let parsed = ParsedGeometry::from_wkb(&bytes_a);
        let parsed_b = ParsedGeometry::from_wkb(&bytes_b);
        let parsed_result = parsed.join(&parsed_b, &SpatialRelationType::Within, JoinSide::Left);

        assert_eq!(geos_result, parsed_result, "{label}")
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
