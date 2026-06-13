use std::time::Instant;

use datafusion::common::JoinSide;

use super::geo_owned::{
    point_on_segment_t, Dir, Edge, GeoOwned, GeoView, LineSegment, MidpointResult, Point,
    SegmentTrait, ToPoints,
};
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
                if right.segment_crossing_t(&alines[li], ri).is_some() {
                    return true;
                }
                match right.segment_midpoint_check(&alines[li], ri) {
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
                if right.segment_crossing_t(&aedges[li], ri).is_some() {
                    return true;
                }
                match right.segment_midpoint_check(&aedges[li], ri) {
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
                match left.segment_midpoint_check(&edges[ri], li) {
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
                if let Some(pair) = right.segment_crossing_t(&alines[li], ri) {
                    crossing_pairs.push(pair);
                }
                if let Some((t_lo, t_hi)) = segment_collinear_ts(&alines[li], &edges[ri]) {
                    collinear_ts.push(t_lo);
                    collinear_ts.push(t_hi);
                }
                if !midpoint_inside {
                    match right.segment_midpoint_check(&alines[li], ri) {
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
                if right.segment_crossing_t(&aedges[li], ri).is_some() {
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
                    match right.segment_midpoint_check(&aedges[li], ri) {
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
                if let Some((right_holes, right_holes_index)) = right.holes() {
                    let (left_nex, left_nex_index) = left.not_exterior();
                    let t = Instant::now();
                    let mut pairs = left_nex_index.candidate_pairs(right_holes_index, false);
                    let t_stripe = t.elapsed();
                    let t = Instant::now();
                    let hit = intersects(&left_nex, &right_holes, &mut pairs, false);
                    // println!(
                    //     "[JOIN] holes:     stripe={:?} check={:?} pairs={}",
                    //     t_stripe,
                    //     t.elapsed(),
                    //     pairs.len()
                    // );
                    if hit {
                        return false;
                    }
                }

                // 1b. not_exterior(left) ⊆ not_exterior(right)
                let (left_nex, left_nex_index) = left.not_exterior();
                let (right_nex, right_nex_index) = right.not_exterior();
                let t = Instant::now();
                let mut pairs = left_nex_index.candidate_pairs(right_nex_index, false);
                let t_stripe = t.elapsed();
                let t = Instant::now();
                let contained = fully_contained(&left_nex, &right_nex, &mut pairs);
                // println!(
                //     "[JOIN] contained: stripe={:?} check={:?} pairs={}",
                //     t_stripe,
                //     t.elapsed(),
                //     pairs.len()
                // );
                if !contained {
                    return false;
                }

                // 2. I(left) ∩ I(right) ≠ ∅
                let (left_int, left_int_index) = left.interior();
                let (right_int, right_int_index) = right.interior();
                let t = Instant::now();
                let mut pairs = left_int_index.candidate_pairs(right_int_index, false);
                let t_stripe = t.elapsed();
                let t = Instant::now();
                let touches = intersects(&left_int, &right_int, &mut pairs, false);
                // println!(
                //     "[JOIN] interior:  stripe={:?} check={:?} pairs={}",
                //     t_stripe,
                //     t.elapsed(),
                //     pairs.len()
                // );
                if !touches {
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
mod deprecated_tests {
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

    #[test]
    fn custom_geo() {
        compare_within(
            "custom_geo",
            "POINT(-71.514748191 -40.785340657)",
            "POLYGON((-71.7011279 -40.427553,-71.703853 -40.4260625,-71.7085356 -40.4227014,-71.7138571 -40.4250536,-71.717977 -40.4270138,-71.7215819 -40.4262297,-71.7248434 -40.4272751,-71.7272467 -40.4297579,-71.7303366 -40.4308032,-71.7353148 -40.4344618,-71.7418379 -40.4365523,-71.7463011 -40.4357684,-71.748876 -40.4351151,-71.7511076 -40.4326325,-71.7538542 -40.4305419,-71.7552275 -40.4287125,-71.7543692 -40.425707,-71.7545409 -40.4221787,-71.7550558 -40.4189115,-71.7566008 -40.4176046,-71.7593474 -40.4156442,-71.7638106 -40.416167,-71.7693037 -40.4190422,-71.7708487 -40.4219173,-71.7734236 -40.4238775,-71.7765135 -40.42244,-71.7782301 -40.419957,-71.7797751 -40.4179967,-71.7806334 -40.4151214,-71.7816633 -40.4126381,-71.782865 -40.4101547,-71.7862982 -40.4098933,-71.7887015 -40.4089783,-71.7916197 -40.4074098,-71.7962546 -40.4076712,-71.7977995 -40.4072791,-71.8005461 -40.4070177,-71.803636 -40.4059719,-71.8065542 -40.4061027,-71.8089575 -40.4075405,-71.8103308 -40.4098933,-71.8127341 -40.4114618,-71.8154806 -40.4138144,-71.8173689 -40.4157749,-71.8197722 -40.4177353,-71.8220038 -40.4190422,-71.8245787 -40.4204798,-71.8247504 -40.4228321,-71.8242354 -40.4245309,-71.8221754 -40.4272751,-71.8209738 -40.4302806,-71.8209738 -40.4326325,-71.8201155 -40.4348538,-71.8214888 -40.4353764,-71.824922 -40.4364217,-71.8281836 -40.4379895,-71.8297285 -40.4408638,-71.8304152 -40.4429542,-71.8321318 -40.4443912,-71.8357367 -40.4458282,-71.8372816 -40.4479184,-71.8395132 -40.4500085,-71.8414015 -40.4520986,-71.8412298 -40.4552335,-71.8403286 -40.4586295,-71.8389866 -40.459613,-71.8372173 -40.4619601,-71.8357367 -40.4632498,-71.8394274 -40.4651925,-71.8394274 -40.4664495,-71.840738 -40.4684,-71.8423661 -40.4698979,-71.8445119 -40.4699958,-71.8455526 -40.4702978,-71.8459529 -40.4709038,-71.8450054 -40.4718567,-71.844823 -40.4736849,-71.8461797 -40.4748837,-71.8468106 -40.4789446,-71.8454024 -40.4809319,-71.8443135 -40.4821845,-71.8425749 -40.4827834,-71.8404934 -40.48577,-71.840987 -40.489809,-71.8420121 -40.4909197,-71.8438145 -40.4911645,-71.8461534 -40.4933022,-71.8459222 -40.4956591,-71.8447513 -40.4965434,-71.8466303 -40.4992327,-71.8481968 -40.4997548,-71.8502995 -40.5047964,-71.853529 -40.5069663,-71.8551789 -40.5085187,-71.8556189 -40.5094184,-71.8555481 -40.512598,-71.8530076 -40.5166207,-71.852698 -40.5192431,-71.8508404 -40.5211293,-71.8455097 -40.5197487,-71.8443344 -40.520154,-71.8428796 -40.5216807,-71.8429096 -40.5253245,-71.8461963 -40.5260774,-71.8511751 -40.525605,-71.8528616 -40.5271056,-71.8531492 -40.5301163,-71.8552177 -40.5329345,-71.8585179 -40.5342229,-71.8595049 -40.537142,-71.8608997 -40.539637,-71.8601272 -40.5431429,-71.8602345 -40.5471377,-71.8586466 -40.5496813,-71.8587325 -40.5525019,-71.8583676 -40.5548333,-71.8591782 -40.5553641,-71.8633458 -40.5553224,-71.8644573 -40.5560919,-71.8669465 -40.5572918,-71.8684185 -40.5572168,-71.8748257 -40.5541649,-71.879525 -40.5536333,-71.8863914 -40.5528052,-71.8876788 -40.552078,-71.8888354 -40.5518872,-71.8881509 -40.5531214,-71.8904984 -40.5560038,-71.8855631 -40.5630464,-71.8810124 -40.5625494,-71.8791511 -40.5634504,-71.8777026 -40.5645035,-71.8772161 -40.5656545,-71.8759501 -40.5673008,-71.8752763 -40.5703522,-71.8741879 -40.5711273,-71.8722138 -40.5709643,-71.8697531 -40.5723667,-71.8691668 -40.5739959,-71.8669422 -40.5762686,-71.8675044 -40.5798019,-71.8655045 -40.5838239,-71.8627923 -40.5861477,-71.8612945 -40.5890384,-71.8584379 -40.5913516,-71.8550847 -40.5926624,-71.8527887 -40.5916326,-71.8507785 -40.5889139,-71.848426 -40.5873778,-71.8472078 -40.5848103,-71.8455145 -40.582543,-71.8449291 -40.582965,-71.8450152 -40.5847922,-71.8446748 -40.5859392,-71.84221 -40.5868582,-71.8402745 -40.5869755,-71.8355238 -40.5843095,-71.8307688 -40.5816304,-71.825051 -40.5781297,-71.8257001 -40.5805986,-71.8265232 -40.5814136,-71.8299846 -40.5846944,-71.8304942 -40.5860531,-71.8319269 -40.5884593,-71.8339547 -40.5896733,-71.8349444 -40.5909189,-71.8354675 -40.5929321,-71.835438 -40.5942851,-71.836991 -40.5954576,-71.8390386 -40.5960709,-71.8406694 -40.5977948,-71.8411415 -40.6033145,-71.8396571 -40.6057961,-71.8390171 -40.6077903,-71.8376133 -40.6099082,-71.8385906 -40.6107876,-71.8401587 -40.6137239,-71.841947 -40.6164607,-71.8426538 -40.6185266,-71.8430073 -40.6200155,-71.8437849 -40.6214911,-71.8446155 -40.6240665,-71.8453754 -40.6263467,-71.8466655 -40.6288012,-71.8473724 -40.6299949,-71.8477788 -40.6306655,-71.8517287 -40.6348244,-71.8535225 -40.6383415,-71.854618 -40.641126,-71.8541232 -40.642306,-71.8568094 -40.6440224,-71.858559 -40.6432581,-71.8609977 -40.6428022,-71.8632245 -40.6439286,-71.8647796 -40.6451622,-71.8654688 -40.6483535,-71.8649917 -40.6497211,-71.8670496 -40.6518782,-71.8674337 -40.6538307,-71.8675489 -40.6556082,-71.8661086 -40.6582453,-71.8673224 -40.6605138,-71.8700456 -40.6637959,-71.8724271 -40.6652818,-71.8739059 -40.6648157,-71.8759032 -40.6645389,-71.8771323 -40.6643495,-71.8794369 -40.6644223,-71.8844879 -40.6661122,-71.8874068 -40.6691077,-71.8893468 -40.6705988,-71.8926117 -40.6742841,-71.8944449 -40.6756175,-71.897222 -40.679167,-71.9 -40.681389,-71.9037146 -40.6831681,-71.9090361 -40.683819,-71.9145293 -40.6872034,-71.9205367 -40.6890681,-71.9225973 -40.690848,-71.9246573 -40.6934512,-71.9260306 -40.6963145,-71.9301504 -40.697616,-71.932382 -40.6993079,-71.9328748 -40.6997291,-71.934442 -40.7020409,-71.9406438 -40.7081492,-71.9411737 -40.7093672,-71.9428324 -40.7120488,-71.9429643 -40.7121916,-71.9431471 -40.7123133,-71.9433368 -40.7123928,-71.9435406 -40.7124492,-71.9437455 -40.7124904,-71.9439383 -40.7125142,-71.9444516 -40.7125374,-71.945517 -40.7125851,-71.9455018 -40.7129323,-71.9461149 -40.7151835,-71.9490761 -40.7182645,-71.9495482 -40.7190868,-71.9516527 -40.7183718,-71.9544416 -40.7178675,-71.9589717 -40.7173632,-71.9625817 -40.7184576,-71.96387 -40.7195521,-71.9632811 -40.7237704,-71.9646544 -40.7276732,-71.9632811 -40.7309253,-71.9637961 -40.733657,-71.9656843 -40.7363885,-71.9649977 -40.7383396,-71.9636244 -40.7405507,-71.9628118 -40.7415677,-71.9634527 -40.7427617,-71.9666079 -40.7451389,-71.9650376 -40.7466247,-71.9640139 -40.7470921,-71.962875 -40.7471161,-71.9596009 -40.7456062,-71.9583671 -40.7458219,-71.9564146 -40.7497845,-71.9560038 -40.7526674,-71.958184 -40.7568068,-71.959359 -40.7583939,-71.9560713 -40.7612275,-71.9546664 -40.7619063,-71.9543392 -40.762207,-71.9539851 -40.7626905,-71.953674 -40.7629099,-71.9535077 -40.7638606,-71.9532019 -40.7647626,-71.9527137 -40.765173,-71.9520807 -40.7654777,-71.9514477 -40.7655671,-71.9504017 -40.7661278,-71.9490767 -40.7670541,-71.9481164 -40.7675701,-71.9472796 -40.7674645,-71.9420815 -40.7658028,-71.9424999 -40.766981,-71.9425267 -40.7672532,-71.9426984 -40.7679967,-71.9428043 -40.7687382,-71.9428159 -40.7689316,-71.9428311 -40.7693608,-71.9425321 -40.7702109,-71.9421673 -40.7706578,-71.9418615 -40.7712144,-71.9414216 -40.7715678,-71.9406438 -40.7720025,-71.940279 -40.7723275,-71.9400537 -40.7725794,-71.9397372 -40.7732253,-71.9390506 -40.7738712,-71.9383854 -40.7740987,-71.9381172 -40.7742775,-71.9371516 -40.7744969,-71.9364059 -40.7747609,-71.9359338 -40.7746594,-71.9353867 -40.7748381,-71.9351399 -40.7751021,-71.9333267 -40.7754393,-71.9317389 -40.7760771,-71.9310361 -40.7763737,-71.930666 -40.776983,-71.9303763 -40.7779295,-71.9301403 -40.7784332,-71.9299257 -40.779075,-71.9290727 -40.7793716,-71.9287884 -40.7796153,-71.9279999 -40.7799199,-71.9269431 -40.7801027,-71.9263691 -40.7802733,-71.9258219 -40.7808989,-71.9245881 -40.7822637,-71.9235098 -40.7826618,-71.9231719 -40.7828771,-71.9228125 -40.7829827,-71.922394 -40.7842825,-71.9213641 -40.7857447,-71.9213855 -40.7867317,-71.9215572 -40.787544,-71.922217 -40.7882142,-71.9230431 -40.7887584,-71.923306 -40.7889899,-71.9249475 -40.7904805,-71.9260418 -40.7913252,-71.9264871 -40.7917639,-71.9270128 -40.7922512,-71.9276565 -40.7933153,-71.9273883 -40.793957,-71.9275975 -40.7942331,-71.9277316 -40.7944118,-71.9280803 -40.7950778,-71.9281393 -40.7960972,-71.9282895 -40.7975023,-71.9281071 -40.7979043,-71.927796 -40.7983145,-71.9270665 -40.7997683,-71.9269216 -40.8007348,-71.9269377 -40.8012627,-71.92734 -40.8020829,-71.9273776 -40.8027976,-71.9271094 -40.804093,-71.9270611 -40.8047061,-71.927104 -40.8051893,-71.927383 -40.8058471,-71.9276995 -40.8064359,-71.9279087 -40.8070003,-71.9279891 -40.8074347,-71.9279945 -40.8081697,-71.9278282 -40.8085391,-71.9276619 -40.808734,-71.9259292 -40.809741,-71.9257522 -40.8097978,-71.925602 -40.8100577,-71.9254947 -40.8103744,-71.9256502 -40.8106992,-71.9259936 -40.8109266,-71.9263315 -40.8109672,-71.926766 -40.8109509,-71.927442 -40.8109753,-71.9284022 -40.8111499,-71.9290406 -40.8113366,-71.9292927 -40.8114584,-71.9306874 -40.8125669,-71.9311166 -40.8130662,-71.9311971 -40.8136793,-71.9308537 -40.8141421,-71.9306767 -40.8144872,-71.9300491 -40.8156159,-71.9295824 -40.8165334,-71.9296306 -40.8169841,-71.9295716 -40.8174672,-71.9296145 -40.8180355,-71.9298613 -40.8184659,-71.9301885 -40.8187338,-71.9305426 -40.818953,-71.9306874 -40.819152,-71.9311112 -40.819566,-71.931417 -40.8198056,-71.9317013 -40.8198624,-71.9322217 -40.8198502,-71.9329029 -40.8197447,-71.9332194 -40.8198259,-71.9334984 -40.82011,-71.9339758 -40.8203901,-71.9346839 -40.8209504,-71.9351506 -40.8213685,-71.9351453 -40.8219125,-71.9347805 -40.8221439,-71.9343138 -40.8222575,-71.9341153 -40.8225133,-71.9339758 -40.8225579,-71.9335467 -40.8231953,-71.9334179 -40.8232765,-71.9325596 -40.8233779,-71.932388 -40.823658,-71.9324631 -40.8239503,-71.9329244 -40.8243278,-71.9329941 -40.8245552,-71.9328815 -40.8247541,-71.9326937 -40.8249246,-71.9323987 -40.8250139,-71.9320929 -40.8248637,-71.9317764 -40.8248718,-71.9314492 -40.8248312,-71.9310254 -40.8246729,-71.9303978 -40.8245146,-71.9300544 -40.8245511,-71.9299847 -40.8246079,-71.9281501 -40.8249043,-71.927045 -40.8249651,-71.9262403 -40.8250869,-71.9257736 -40.8253954,-71.925543 -40.8259637,-71.9252747 -40.8263981,-71.9248724 -40.8265564,-71.9242662 -40.8266782,-71.9240034 -40.8270841,-71.9238639 -40.8280299,-71.9238585 -40.8286265,-71.9240302 -40.8293937,-71.9244486 -40.8302461,-71.9255215 -40.8307535,-71.9260901 -40.8307941,-71.9265515 -40.831062,-71.9266748 -40.8314557,-71.9266856 -40.8318007,-71.9271845 -40.8321741,-71.9277155 -40.8326206,-71.9276834 -40.8329534,-71.9275224 -40.8335054,-71.9269323 -40.8339559,-71.9268143 -40.8340249,-71.9260633 -40.8347961,-71.9255805 -40.8357093,-71.9243681 -40.8365291,-71.9231129 -40.8369147,-71.9219381 -40.8368863,-71.9209939 -40.8370364,-71.920495 -40.8374342,-71.919964 -40.8380592,-71.9195509 -40.8384285,-71.9190735 -40.8387248,-71.918419 -40.8389115,-71.9178504 -40.8388465,-71.9172657 -40.8388344,-71.9161552 -40.838599,-71.9153559 -40.8385422,-71.9147658 -40.8384691,-71.9144225 -40.8387897,-71.9141758 -40.8398774,-71.9139129 -40.8404374,-71.9126415 -40.8410868,-71.9124699 -40.8414601,-71.9126362 -40.841939,-71.9131136 -40.8422353,-71.9133925 -40.8426127,-71.9134462 -40.8432904,-71.9130814 -40.8434203,-71.9129902 -40.8439965,-71.9126952 -40.844309,-71.9121641 -40.8446215,-71.9116652 -40.8451247,-71.9111127 -40.8453722,-71.9107264 -40.845494,-71.9102597 -40.8458024,-71.9099486 -40.8459119,-71.9094122 -40.8459201,-71.9084787 -40.8462975,-71.9082266 -40.8463867,-71.9078136 -40.8463867,-71.9076312 -40.8464395,-71.9075668 -40.8466627,-71.9083339 -40.8475757,-71.9094873 -40.8488296,-71.9100452 -40.8498968,-71.9105762 -40.8511993,-71.9105923 -40.8525587,-71.9104046 -40.8533864,-71.9101095 -40.853569,-71.9093478 -40.8541371,-71.9090903 -40.8549526,-71.908983 -40.8562388,-71.9082964 -40.8580322,-71.9075882 -40.858515,-71.9065046 -40.8590586,-71.9062632 -40.8591438,-71.9054532 -40.8596307,-71.9050294 -40.8596591,-71.9043911 -40.8597808,-71.9039995 -40.8599431,-71.903801 -40.8605841,-71.9036079 -40.8606937,-71.9034094 -40.8608195,-71.902948 -40.8609371,-71.9012046 -40.8621015,-71.9011188 -40.8626208,-71.9007862 -40.862848,-71.9006145 -40.8630103,-71.9005394 -40.8634565,-71.899429 -40.8643531,-71.8992037 -40.8643896,-71.8983507 -40.8648846,-71.8971223 -40.8655093,-71.896677 -40.8659799,-71.8959153 -40.8666614,-71.8952608 -40.866986,-71.894778 -40.8674647,-71.8946117 -40.8678906,-71.8944937 -40.8684302,-71.8937856 -40.8689494,-71.8933135 -40.8690752,-71.8928897 -40.8696999,-71.8921924 -40.8701218,-71.8918169 -40.8704909,-71.8914628 -40.8707019,-71.8906689 -40.871643,-71.8900895 -40.8730749,-71.8896872 -40.8734034,-71.8893868 -40.8734846,-71.8889254 -40.8735576,-71.8883032 -40.8736103,-71.8876165 -40.8737969,-71.8872303 -40.8741336,-71.8866992 -40.874884,-71.8862969 -40.8754559,-71.8855459 -40.8758534,-71.8849772 -40.8762225,-71.8847144 -40.8771798,-71.8843764 -40.8778774,-71.882456 -40.8790009,-71.8821877 -40.8792159,-71.8813862 -40.8803281,-71.8800902 -40.8810533,-71.8794787 -40.8814386,-71.8789208 -40.8817468,-71.8782502 -40.881759,-71.87751 -40.8816738,-71.8767482 -40.8816373,-71.8761206 -40.8815521,-71.8756324 -40.8812155,-71.874243 -40.8818036,-71.8724567 -40.8818279,-71.8717486 -40.8815521,-71.8713999 -40.8812479,-71.871078 -40.8804408,-71.8705845 -40.8796986,-71.8701714 -40.8795282,-71.8694043 -40.8793051,-71.868825 -40.8791186,-71.8681061 -40.8788671,-71.867618 -40.8788671,-71.8668509 -40.8789239,-71.8662232 -40.8788387,-71.8654883 -40.8784818,-71.86544 -40.8784088,-71.8649089 -40.8781208,-71.8636483 -40.8779951,-71.863423 -40.8779951,-71.8628758 -40.8779302,-71.8626559 -40.8778653,-71.8622053 -40.8778125,-71.8619263 -40.8779139,-71.8612772 -40.8780681,-71.8611485 -40.8781289,-71.8607301 -40.8781208,-71.8603653 -40.8781654,-71.8593299 -40.8780924,-71.8591476 -40.8781857,-71.8588418 -40.8784534,-71.8587184 -40.8785913,-71.858182 -40.8794674,-71.8581605 -40.8801204,-71.8584797 -40.8808394,-71.858713 -40.8813453,-71.8588311 -40.8821199,-71.8584234 -40.8825823,-71.8586755 -40.8833042,-71.8591958 -40.883868,-71.8597913 -40.8848008,-71.8599951 -40.8849752,-71.8610305 -40.8873436,-71.8619102 -40.888175,-71.8644583 -40.8891158,-71.8657565 -40.8896998,-71.8670923 -40.890665,-71.8675965 -40.8909529,-71.8686801 -40.8918369,-71.8690556 -40.8925142,-71.8690395 -40.8929683,-71.8688518 -40.8938321,-71.8671727 -40.8960461,-71.8666363 -40.8973599,-71.8666363 -40.8992089,-71.8657994 -40.899955,-71.865499 -40.9009768,-71.8648247 -40.9039299,-71.8645758 -40.9057852,-71.8654341 -40.9079907,-71.8657774 -40.9092881,-71.8640608 -40.911234,-71.8635458 -40.9129205,-71.8614859 -40.9147366,-71.8599409 -40.9170716,-71.857881 -40.9196659,-71.8556745 -40.9227398,-71.8606747 -40.9248102,-71.8593316 -40.9267864,-71.8613142 -40.9305608,-71.8668503 -40.9339327,-71.8635458 -40.9345811,-71.8621284 -40.9349636,-71.8601126 -40.9367857,-71.857881 -40.9383418,-71.8571381 -40.9410899,-71.8571985 -40.9417653,-71.8549628 -40.9443066,-71.8551132 -40.9451826,-71.8556433 -40.9474919,-71.856739 -40.949147,-71.8577899 -40.9537434,-71.8576662 -40.954958,-71.8572398 -40.9559886,-71.8574894 -40.9563994,-71.8595041 -40.957,-71.8612006 -40.956306,-71.8645761 -40.9547311,-71.8661843 -40.9543374,-71.8693476 -40.9549046,-71.8745433 -40.9588951,-71.879475 -40.9618918,-71.8815678 -40.9630107,-71.8852037 -40.9659355,-71.8877752 -40.9665011,-71.8909509 -40.9652697,-71.8919491 -40.96438,-71.8928392 -40.9635846,-71.8935258 -40.9626772,-71.8959291 -40.9616402,-71.8980749 -40.9602143,-71.8984182 -40.958529,-71.8997915 -40.9600846,-71.9024522 -40.9607328,-71.9030758 -40.9613997,-71.9037397 -40.9619643,-71.9057138 -40.9628717,-71.9059713 -40.9641679,-71.9057138 -40.9662419,-71.9064004 -40.9672789,-71.9064863 -40.9690935,-71.9078596 -40.9710377,-71.908632 -40.972917,-71.9076021 -40.9735651,-71.9067438 -40.9746019,-71.905628 -40.9766755,-71.9054563 -40.9794619,-71.9044263 -40.9836736,-71.9051988 -40.9851638,-71.9022806 -40.9847751,-71.9004781 -40.983868,-71.898504 -40.9839976,-71.8957574 -40.9842567,-71.8920667 -40.9861357,-71.8906934 -40.9874315,-71.8882902 -40.9883385,-71.887346 -40.990347,-71.8854578 -40.9920962,-71.8836587 -40.9947571,-71.8783071 -40.9974587,-71.8763597 -40.9981209,-71.8750722 -40.9984448,-71.8751467 -40.9992388,-71.873699 -41.0002586,-71.8739564 -41.001878,-71.8740423 -41.0051815,-71.8740423 -41.009262,-71.8740397 -41.0106004,-71.8698366 -41.0123708,-71.8678625 -41.0136661,-71.8662317 -41.0151556,-71.8656309 -41.0161918,-71.8657167 -41.0179403,-71.8634367 -41.017308,-71.8626535 -41.017138,-71.8620313 -41.0170814,-71.8614948 -41.0171542,-71.8611408 -41.0169842,-71.8604541 -41.016798,-71.8596173 -41.0167818,-71.8591237 -41.0169275,-71.8587375 -41.0170975,-71.8582333 -41.0172999,-71.8573857 -41.0173566,-71.8569994 -41.017389,-71.856581 -41.0177209,-71.8559266 -41.0180042,-71.8553579 -41.0182147,-71.8548966 -41.0185546,-71.8541563 -41.0189432,-71.8529547 -41.0193479,-71.8519462 -41.0197931,-71.8516243 -41.0201412,-71.8509269 -41.0207888,-71.8502081 -41.021153,-71.8490172 -41.0211611,-71.8482554 -41.0210802,-71.8472255 -41.0210559,-71.8464101 -41.020975,-71.8456143 -41.021047,-71.8453372 -41.0210721,-71.8442536 -41.0213554,-71.8431592 -41.0216873,-71.8423224 -41.0220758,-71.8418396 -41.0223025,-71.8412495 -41.0225615,-71.8406809 -41.0227477,-71.8404234 -41.0230471,-71.8400908 -41.0235085,-71.8398655 -41.0236623,-71.8396241 -41.023804,-71.8395919 -41.0240589,-71.8393827 -41.0242613,-71.8390876 -41.0244151,-71.8387282 -41.0245162,-71.8382186 -41.024581,-71.8374193 -41.0246336,-71.8369151 -41.0248562,-71.8366254 -41.0250424,-71.8362016 -41.0253378,-71.8360514 -41.0255563,-71.8360246 -41.0257789,-71.8360675 -41.0260541,-71.8361801 -41.0262646,-71.8361265 -41.02656,-71.8356169 -41.0267866,-71.8353969 -41.0268878,-71.835236 -41.0270901,-71.8352628 -41.0272277,-71.8352199 -41.0273775,-71.8350053 -41.0273694,-71.8347156 -41.0275596,-71.8345547 -41.0276648,-71.8341577 -41.0279359,-71.8339754 -41.0282192,-71.8337554 -41.0284175,-71.8335569 -41.028721,-71.8335784 -41.0290002,-71.8335462 -41.0292835,-71.8333799 -41.0294535,-71.8330151 -41.0294859,-71.832645 -41.0294454,-71.8323338 -41.0294656,-71.8321032 -41.0295547,-71.8317277 -41.0296396,-71.8314112 -41.0296437,-71.831041 -41.0295789,-71.8306816 -41.029502,-71.8303758 -41.0293887,-71.8301505 -41.029154,-71.8299091 -41.0289841,-71.8296892 -41.0288869,-71.8296087 -41.0286927,-71.8295229 -41.0285187,-71.8295497 -41.0283366,-71.8295604 -41.0281221,-71.8294639 -41.0278833,-71.8291903 -41.0277214,-71.8288953 -41.0275312,-71.8287933 -41.0273572,-71.8287397 -41.0270982,-71.8286914 -41.0267097,-71.8284232 -41.0263738,-71.8281067 -41.0260703,-71.8278277 -41.0258841,-71.8274254 -41.0257668,-71.8269265 -41.0256777,-71.8266476 -41.0255725,-71.8264062 -41.0254592,-71.8260575 -41.0252933,-71.8257463 -41.0250464,-71.8255532 -41.0248643,-71.8253762 -41.0247793,-71.8251348 -41.0245446,-71.8248827 -41.024322,-71.8246144 -41.0240508,-71.8245125 -41.0239416,-71.8243623 -41.0238161,-71.8240405 -41.0236542,-71.8238205 -41.0235449,-71.8236649 -41.0234195,-71.8234718 -41.0232738,-71.8232465 -41.0231928,-71.8229837 -41.0232455,-71.8227852 -41.0232859,-71.8226135 -41.0234235,-71.8224687 -41.0235491,-71.8223775 -41.0235652,-71.8223174 -41.0235627,-71.8220724 -41.0235527,-71.816822 -41.0230122,-71.8126739 -41.0226634,-71.8016701 -41.021738,-71.7435019 -41.0222433,-71.7084569 -41.0253601,-71.6134602 -41.0338081,-71.5516111 -41.0230409,-71.4920307 -41.038793,-71.47954 -41.0383837,-71.461258 -41.0405848,-71.4569058 -41.0416113,-71.4480401 -41.0457313,-71.4401436 -41.0502948,-71.4225913 -41.0622036,-71.36108 -41.0884588,-71.3430692 -41.0942302,-71.3062478 -41.1001809,-71.2882635 -41.1006358,-71.2527663 -41.0980014,-71.2387792 -41.0950086,-71.2212936 -41.0876557,-71.2196193 -41.0865262,-71.174989 -41.0562553,-71.170249 -41.0548741,-71.1606151 -41.0535673,-71.1568809 -41.0543078,-71.1546064 -41.0549874,-71.1529391 -41.0551152,-71.1523761 -41.0551827,-71.1519932 -41.0553236,-71.1515075 -41.0555701,-71.1509658 -41.0558307,-71.1503587 -41.0560067,-71.1498263 -41.056042,-71.1490555 -41.0562156,-71.1486425 -41.0563385,-71.1478955 -41.0564942,-71.1465919 -41.0567814,-71.1458382 -41.0569311,-71.1455056 -41.0569634,-71.1452508 -41.0569493,-71.1450389 -41.056927,-71.144717 -41.056834,-71.1444193 -41.0567288,-71.1441323 -41.0566419,-71.1438078 -41.0565872,-71.143384 -41.0565286,-71.1427081 -41.0564012,-71.1421555 -41.0562839,-71.1419436 -41.0562272,-71.14178 -41.0561403,-71.1415949 -41.0560169,-71.1413965 -41.055847,-71.141257 -41.0557539,-71.1410022 -41.0556023,-71.1405489 -41.0553211,-71.1400929 -41.0550076,-71.1390415 -41.0541662,-71.1383307 -41.0535594,-71.1375958 -41.0528454,-71.1369011 -41.052186,-71.13648 -41.0517956,-71.1361662 -41.0515529,-71.1356887 -41.051203,-71.1351389 -41.0508106,-71.1342672 -41.0502301,-71.1335081 -41.0497143,-71.1332881 -41.0495504,-71.1330575 -41.0493522,-71.1326498 -41.0489517,-71.1319122 -41.0482073,-71.1314079 -41.0477542,-71.1308876 -41.0472909,-71.1304128 -41.0468176,-71.1297423 -41.0461501,-71.1294258 -41.0457576,-71.1292273 -41.045436,-71.1290824 -41.0451467,-71.1288786 -41.0447037,-71.1286774 -41.0441535,-71.1284763 -41.0434778,-71.1282529 -41.0427874,-71.1279425 -41.0419747,-71.1277199 -41.0413557,-71.1274597 -41.0407144,-71.1272129 -41.0402106,-71.1268777 -41.0395633,-71.1266001 -41.0390523,-71.1264378 -41.0385719,-71.1263332 -41.0379488,-71.1262393 -41.0372529,-71.1262071 -41.0365873,-71.1262232 -41.0358205,-71.1263144 -41.0354421,-71.1263895 -41.0351832,-71.1264968 -41.0347583,-71.1265934 -41.0342039,-71.1267006 -41.0330547,-71.1269957 -41.0309545,-71.1271191 -41.0300723,-71.1271191 -41.0295746,-71.1270547 -41.0291699,-71.1269689 -41.0287936,-71.1267972 -41.028397,-71.12654 -41.027815,-71.1263412 -41.0272719,-71.126191 -41.0266082,-71.1260301 -41.0258312,-71.1258638 -41.0250582,-71.1256492 -41.0241153,-71.1253917 -41.0230468,-71.1252737 -41.0226705,-71.1251289 -41.0224195,-71.1247963 -41.0220067,-71.1241847 -41.0213592,-71.1237824 -41.0209989,-71.1232996 -41.0206145,-71.122849 -41.0202988,-71.1222428 -41.0198657,-71.1216581 -41.0195379,-71.1210626 -41.0192424,-71.1202312 -41.01887,-71.1197591 -41.018615,-71.1194855 -41.0184248,-71.1192548 -41.0182144,-71.1190081 -41.0178865,-71.1188954 -41.0175951,-71.1188042 -41.0172551,-71.1186755 -41.0168463,-71.1185306 -41.0164861,-71.1182678 -41.0160327,-71.1178333 -41.0153811,-71.1173719 -41.0146727,-71.1168751 -41.0139224,-71.1163902 -41.012997,-71.1159986 -41.0122481,-71.1157519 -41.0116248,-71.1156339 -41.0108233,-71.1155695 -41.009957,-71.1155534 -41.0089815,-71.1155266 -41.0084593,-71.1154568 -41.0080464,-71.1153013 -41.0076942,-71.1151189 -41.007427,-71.1148346 -41.0071396,-71.1144537 -41.0068076,-71.1139977 -41.006504,-71.1134666 -41.006249,-71.1129141 -41.0060992,-71.1123347 -41.0060425,-71.1115944 -41.0060668,-71.1108112 -41.0062044,-71.1100227 -41.0063502,-71.1095184 -41.0064069,-71.1090571 -41.0063988,-71.1084884 -41.0063461,-71.1081129 -41.0062571,-71.1077857 -41.0061559,-71.1073726 -41.0058968,-71.1069542 -41.0056094,-71.1063588 -41.0050709,-71.1054736 -41.0040184,-71.1048675 -41.0031682,-71.1044705 -41.0027391,-71.103918 -41.0023059,-71.1034137 -41.0019658,-71.1028022 -41.0016298,-71.1017186 -41.0011359,-71.1003989 -41.0006137,-71.0991973 -41.0001926,-71.0970349 -40.9996969,-71.0959464 -40.9996218,-71.0948199 -40.999561,-71.0935754 -40.9994234,-71.09254 -40.9992857,-71.0918212 -40.9991926,-71.0911292 -40.9990347,-71.0904586 -40.9987715,-71.0900939 -40.9985813,-71.0897774 -40.9983707,-71.0895252 -40.9981399,-71.0892087 -40.9977513,-71.0889351 -40.9973262,-71.0887581 -40.9969051,-71.0886884 -40.9965366,-71.0886937 -40.9962492,-71.0887688 -40.9958645,-71.0889083 -40.9955244,-71.0891122 -40.9952532,-71.0894448 -40.9949697,-71.0897291 -40.994828,-71.0899866 -40.9947329,-71.0902306 -40.9946701,-71.0904881 -40.9946418,-71.0907215 -40.99466,-71.0910058 -40.9946985,-71.0913304 -40.9947693,-71.0917085 -40.9948442,-71.0920867 -40.994909,-71.0923845 -40.9949434,-71.0927948 -40.9949313,-71.0931999 -40.9948948,-71.0935244 -40.994828,-71.0938007 -40.994739,-71.094439 -40.9945244,-71.0951874 -40.9943158,-71.0958955 -40.9941458,-71.0964587 -40.9940506,-71.097199 -40.993913,-71.0985562 -40.9936215,-71.0992616 -40.9934696,-71.099782 -40.9933198,-71.1002058 -40.99317,-71.1006939 -40.9929432,-71.1012382 -40.9925924,-71.1015737 -40.9921436,-71.1017346 -40.9918682,-71.1017883 -40.991682,-71.1018205 -40.9914066,-71.101799 -40.9911353,-71.10174 -40.9908398,-71.1016327 -40.9906332,-71.1014611 -40.9904713,-71.1011193 -40.9902823,-71.1005759 -40.9899996,-71.0995111 -40.9894995,-71.0985696 -40.9891391,-71.097706 -40.9887828,-71.0960537 -40.9879506,-71.0954019 -40.9876753,-71.0948011 -40.9874748,-71.094211 -40.9873453,-71.0936451 -40.9872825,-71.0908824 -40.9870537,-71.0903996 -40.9870051,-71.089941 -40.9869241,-71.0886133 -40.9866164,-71.0878006 -40.9863491,-71.0870335 -40.986017,-71.086092 -40.9854886,-71.0840508 -40.9840692,-71.0833937 -40.9836744,-71.0827929 -40.9833909,-71.0821277 -40.9831439,-71.081602 -40.9830083,-71.0810789 -40.9829455,-71.0806417 -40.9829212,-71.0801509 -40.9829455,-71.0797781 -40.9830204,-71.0788447 -40.9833727,-71.0774553 -40.9840368,-71.0769886 -40.98413,-71.0764521 -40.9841462,-71.0744834 -40.9837412,-71.0730243 -40.983154,-71.0719782 -40.9824534,-71.0716134 -40.9817569,-71.0714632 -40.981198,-71.0712862 -40.9808741,-71.0710609 -40.980623,-71.0707176 -40.9803921,-71.0698271 -40.9799062,-71.0689366 -40.9795133,-71.0681802 -40.9792987,-71.0676062 -40.9792096,-71.0667847 -40.9792566,-71.0643027 -40.9806324,-71.0636509 -40.9809017,-71.0628532 -40.9810876,-71.061825 -40.981197,-71.0609585 -40.9810038,-71.0601619 -40.980517,-71.0598507 -40.9798557,-71.0585431 -40.9771433,-71.0566876 -40.9735599,-71.0559983 -40.9724335,-71.0549435 -40.9713868,-71.054517 -40.9708967,-71.0541415 -40.970358,-71.0539699 -40.9700705,-71.0538706 -40.9698619,-71.0538223 -40.9696209,-71.0538358 -40.9694528,-71.0539001 -40.9692847,-71.0540235 -40.9689951,-71.0541509 -40.9686569,-71.0542099 -40.968351,-71.0542274 -40.9680878,-71.054218 -40.9678599,-71.0542032 -40.967706,-71.0541643 -40.9675511,-71.0541187 -40.9674397,-71.054053 -40.9673364,-71.0539109 -40.9671764,-71.0526995 -40.9664655,-71.0524437 -40.9662893,-71.0523257 -40.9661232,-71.052264 -40.9659288,-71.0521942 -40.9655338,-71.0520789 -40.9649505,-71.051977 -40.9645657,-71.051816 -40.9641343,-71.0515693 -40.9637434,-71.0511428 -40.9631277,-71.0507566 -40.9626639,-71.050432 -40.9623236,-71.0499117 -40.961876,-71.0492787 -40.9613534,-71.0488334 -40.9609827,-71.0484123 -40.9605513,-71.0482004 -40.9603022,-71.0479912 -40.9600409,-71.0478732 -40.9599599,-71.0477042 -40.9599031,-71.0474628 -40.9598809,-71.0470954 -40.959889,-71.0467118 -40.9599558,-71.0464248 -40.9599963,-71.046221 -40.9600307,-71.0460681 -40.9600854,-71.045984 -40.9601418,-71.0458943 -40.9602238,-71.0457677 -40.9602698,-71.0455638 -40.9603163,-71.0452876 -40.9603184,-71.0448369 -40.9602414,-71.0443595 -40.9601219,-71.0438177 -40.9599558,-71.043359 -40.9597634,-71.0428011 -40.9594879,-71.0424927 -40.9592773,-71.0421896 -40.9589471,-71.0419831 -40.9586088,-71.0417953 -40.958224,-71.0415566 -40.957912,-71.0412776 -40.9576669,-71.0409826 -40.9574968,-71.0406822 -40.9574117,-71.0402906 -40.9573671,-71.0396791 -40.957353,-71.0391024 -40.9573064,-71.0384157 -40.9571545,-71.0378605 -40.9569742,-71.0375655 -40.9568202,-71.0372114 -40.9565589,-71.0368359 -40.9562065,-71.036616 -40.9559168,-71.0364819 -40.9556271,-71.0364202 -40.9551329,-71.0364336 -40.9547824,-71.0365221 -40.9541808,-71.0367635 -40.9533644,-71.0370612 -40.95254,-71.0373643 -40.9517925,-71.0375413 -40.9511422,-71.0378337 -40.9503076,-71.0384882 -40.948934,-71.0388154 -40.9484397,-71.0391694 -40.9482331,-71.0396093 -40.9481358,-71.03984 -40.9481358,-71.0401082 -40.9482047,-71.0404033 -40.9483384,-71.040768 -40.9485046,-71.0411543 -40.9486221,-71.0415351 -40.9486383,-71.0419428 -40.948618,-71.0424417 -40.9484397,-71.0427958 -40.948229,-71.0429031 -40.9480629,-71.0429353 -40.947755,-71.0428977 -40.9474673,-71.042726 -40.9471837,-71.0422432 -40.9467582,-71.041621 -40.9462436,-71.0409719 -40.945571,-71.0397863 -40.9441609,-71.0390353 -40.9435005,-71.0381824 -40.9427427,-71.0372382 -40.9416446,-71.0367608 -40.9409963,-71.0365248 -40.9405586,-71.0364658 -40.940356,-71.0365087 -40.9400845,-71.0368359 -40.9397036,-71.0372114 -40.9393551,-71.0374045 -40.9389985,-71.0374904 -40.9386865,-71.0374582 -40.9383947,-71.0372919 -40.9380543,-71.037088 -40.9377949,-71.0365301 -40.9373573,-71.0362408 -40.9371391,-71.0360526 -40.9369453,-71.0354412 -40.9363158,-71.034852 -40.935733,-71.0343093 -40.935408,-71.0336173 -40.9350757,-71.032995 -40.9347718,-71.0325766 -40.9345165,-71.0323352 -40.9343017,-71.0321957 -40.9340423,-71.0320187 -40.9336006,-71.0319489 -40.9331791,-71.0319382 -40.9325509,-71.0320938 -40.9313554,-71.0322976 -40.930407,-71.0327482 -40.9291141,-71.0329843 -40.9287413,-71.0336173 -40.9279064,-71.0345292 -40.9268769,-71.0353392 -40.9259609,-71.0358113 -40.9253367,-71.0362244 -40.9245869,-71.0363048 -40.9241613,-71.0362941 -40.9236465,-71.0361868 -40.9232777,-71.0358918 -40.9228115,-71.0353446 -40.9222643,-71.0342717 -40.9213158,-71.0334081 -40.9205092,-71.0321618 -40.9191261,-71.0320026 -40.9187621,-71.0318846 -40.9183689,-71.0318631 -40.9180163,-71.031965 -40.9177609,-71.0322172 -40.9174974,-71.0324854 -40.9173028,-71.0331398 -40.9168853,-71.0336494 -40.9165245,-71.0341805 -40.9160503,-71.0345668 -40.9155314,-71.0347813 -40.9150814,-71.0352212 -40.9142017,-71.0354894 -40.9135896,-71.035586 -40.913172,-71.0356075 -40.9127748,-71.0355324 -40.9123532,-71.0353768 -40.911968,-71.0351622 -40.9115424,-71.0349745 -40.9111572,-71.0347277 -40.9105815,-71.034261 -40.9090491,-71.0340571 -40.9077153,-71.0340732 -40.9063287,-71.0341537 -40.9048286,-71.0341752 -40.9039609,-71.0342825 -40.9033811,-71.034497 -40.9029676,-71.0347545 -40.9026189,-71.035248 -40.902181,-71.0359079 -40.9016052,-71.0366589 -40.9009768,-71.0369754 -40.9007051,-71.0373455 -40.9003402,-71.0376835 -40.8999712,-71.0379249 -40.8995455,-71.0380751 -40.8991643,-71.0382199 -40.8987142,-71.038354 -40.8982398,-71.0383756 -40.8981817,-71.038633 -40.8974897,-71.0389119 -40.8967841,-71.0391533 -40.8961637,-71.0393438 -40.8955676,-71.0394082 -40.8952432,-71.0394189 -40.8949594,-71.0394082 -40.8947262,-71.0393572 -40.8944363,-71.0392606 -40.8940511,-71.0391426 -40.8936739,-71.0390192 -40.8931995,-71.0389334 -40.8925628,-71.0388261 -40.891545,-71.0387349 -40.8907826,-71.038751 -40.8903406,-71.03881 -40.8899472,-71.038987 -40.8894038,-71.0392928 -40.8888157,-71.0397112 -40.8882155,-71.0404944 -40.8873558,-71.0418275 -40.8861006,-71.0425141 -40.8855531,-71.0429969 -40.8852773,-71.0434127 -40.8851191,-71.0438284 -40.8849792,-71.0444105 -40.8847683,-71.044786 -40.8845615,-71.045081 -40.8843709,-71.0453171 -40.8842208,-71.045596 -40.884087,-71.045934 -40.8839815,-71.0462934 -40.8839329,-71.046685 -40.8839369,-71.0472536 -40.8839653,-71.0481387 -40.8839734,-71.0488629 -40.8839815,-71.0497212 -40.8839288,-71.0505045 -40.8837869,-71.0514003 -40.8835354,-71.0521835 -40.8831907,-71.0528916 -40.8828256,-71.0532457 -40.8825904,-71.0534549 -40.8823917,-71.05353 -40.8821605,-71.05353 -40.8819334,-71.0534549 -40.8816089,-71.0532028 -40.8810451,-71.0528594 -40.880307,-71.0523766 -40.87937,-71.0521352 -40.8789969,-71.0519153 -40.8787981,-71.0516578 -40.8787008,-71.0514003 -40.8786805,-71.0511321 -40.8787292,-71.0507459 -40.8788387,-71.0500538 -40.8790861,-71.049276 -40.8793214,-71.0486805 -40.8794309,-71.0481119 -40.8794106,-71.0477793 -40.879297,-71.0474253 -40.8790821,-71.0471195 -40.8788428,-71.0469264 -40.8785751,-71.0468084 -40.8783074,-71.0466984 -40.8780316,-71.0468888 -40.8774434,-71.0469478 -40.8771068,-71.047039 -40.8766647,-71.0470712 -40.8762104,-71.0471141 -40.8757601,-71.0470766 -40.8752328,-71.0470015 -40.8749205,-71.046862 -40.8745067,-71.0467011 -40.873878,-71.0466152 -40.873371,-71.0466099 -40.872799,-71.0465938 -40.8722271,-71.0467118 -40.8717565,-71.0468352 -40.8714361,-71.0470498 -40.8711116,-71.047436 -40.8708398,-71.0479778 -40.8706086,-71.048643 -40.870422,-71.0493189 -40.8703084,-71.0501933 -40.8702394,-71.0507888 -40.8702354,-71.051572 -40.8703003,-71.0527253 -40.8703449,-71.0535514 -40.8703084,-71.0540718 -40.8702719,-71.0544527 -40.8701826,-71.0547209 -40.8700366,-71.0549516 -40.8698175,-71.0552251 -40.8694971,-71.0555846 -40.8690589,-71.0558474 -40.8686898,-71.0559976 -40.8684302,-71.0560513 -40.8681868,-71.0560834 -40.867919,-71.0560566 -40.8676432,-71.0559976 -40.8673024,-71.0560191 -40.8668156,-71.0560834 -40.8663653,-71.0561854 -40.8658298,-71.0563839 -40.8651036,-71.0566896 -40.8643288,-71.0568827 -40.8640407,-71.0571027 -40.8638947,-71.0573924 -40.8637973,-71.0577625 -40.8637567,-71.0581112 -40.863777,-71.0585135 -40.8638014,-71.0588408 -40.8637851,-71.0590714 -40.8637121,-71.059227 -40.8635782,-71.0593182 -40.8634403,-71.0593611 -40.8632496,-71.0593772 -40.8630265,-71.0593772 -40.8627993,-71.0593557 -40.8625559,-71.0593611 -40.8623409,-71.0594201 -40.8620406,-71.0596401 -40.86157,-71.0598707 -40.8609371,-71.0600638 -40.8603285,-71.0603213 -40.8595009,-71.0605198 -40.858872,-71.0606217 -40.8584825,-71.0606432 -40.8582026,-71.0606486 -40.8579754,-71.0606915 -40.8577847,-71.0607934 -40.8576061,-71.0609973 -40.8574479,-71.0612118 -40.8573627,-71.0615337 -40.8572734,-71.0618502 -40.8571639,-71.062215 -40.8569975,-71.0627407 -40.8567298,-71.0635614 -40.8562226,-71.0645646 -40.8555004,-71.0655034 -40.8547822,-71.0663295 -40.8541736,-71.0667962 -40.8537556,-71.067161 -40.8533661,-71.0673648 -40.8529685,-71.0677457 -40.8521813,-71.0681319 -40.8515483,-71.0685504 -40.8510127,-71.0691136 -40.8505907,-71.0698915 -40.8501606,-71.0705406 -40.84984,-71.0711145 -40.8496817,-71.0717315 -40.8496046,-71.0723698 -40.8496046,-71.0727292 -40.8496533,-71.0730082 -40.8497994,-71.0732496 -40.8500469,-71.0734749 -40.8503919,-71.0738504 -40.8509518,-71.0742313 -40.8514225,-71.0747033 -40.8518405,-71.0750735 -40.852092,-71.0757226 -40.8524126,-71.0765755 -40.8527128,-71.0775518 -40.8529685,-71.0782814 -40.853074,-71.0791504 -40.8531673,-71.0806525 -40.8531673,-71.0825622 -40.8530659,-71.084075 -40.8529157,-71.0850406 -40.8527615,-71.0855824 -40.8526155,-71.0861886 -40.8523477,-71.0864621 -40.852088,-71.0865802 -40.851812,-71.086607 -40.8515361,-71.0865372 -40.8510979,-71.0864514 -40.850757,-71.086371 -40.850335,-71.0863602 -40.8498156,-71.0864085 -40.8495519,-71.0865158 -40.8492638,-71.0867679 -40.8487566,-71.0871112 -40.8480951,-71.0875833 -40.8472592,-71.0878837 -40.8467398,-71.0882378 -40.8462528,-71.088742 -40.8456401,-71.0895735 -40.8446215,-71.0912472 -40.8429049,-71.0922182 -40.84187,-71.0929263 -40.8412532,-71.0935378 -40.8408311,-71.0943693 -40.8404374,-71.0949433 -40.8402223,-71.0956192 -40.8400641,-71.0961664 -40.8400438,-71.0966706 -40.8400925,-71.0972527 -40.8402,-71.0977489 -40.8403542,-71.0981888 -40.8404456,-71.0986823 -40.8405145,-71.0992026 -40.8405348,-71.0995889 -40.8404821,-71.0998195 -40.8403766,-71.0999268 -40.8402548,-71.1000824 -40.8399829,-71.1003399 -40.8394066,-71.1006725 -40.8388384,-71.1011982 -40.8382134,-71.1018419 -40.8373449,-71.1021209 -40.8368497,-71.1022282 -40.836241,-71.1021745 -40.8357945,-71.1019492 -40.8354292,-71.101327 -40.8348285,-71.1005223 -40.8344145,-71.0995339 -40.8339808,-71.0984978 -40.8335203,-71.0971212 -40.8330427,-71.0955012 -40.8325556,-71.0934091 -40.8319874,-71.0925293 -40.8317195,-71.0921645 -40.8315084,-71.091789 -40.8312162,-71.0916495 -40.8309077,-71.0916174 -40.8305505,-71.09173 -40.8302217,-71.0920358 -40.8294911,-71.0922751 -40.8288735,-71.0926634 -40.8277701,-71.0929477 -40.8267228,-71.0931248 -40.8259881,-71.093393 -40.8256025,-71.0936719 -40.8253833,-71.0939509 -40.8252737,-71.0944873 -40.8251559,-71.0948253 -40.8250788,-71.0952115 -40.8249408,-71.0954905 -40.8247297,-71.0958499 -40.8243765,-71.096102 -40.8242182,-71.0964078 -40.8241249,-71.0966867 -40.8241046,-71.0971212 -40.824068,-71.0979742 -40.82393,-71.0993528 -40.8236012,-71.1003667 -40.8233455,-71.1009461 -40.823041,-71.1014557 -40.8226513,-71.1018151 -40.8222535,-71.1019653 -40.821742,-71.1019653 -40.8212386,-71.1017454 -40.820719,-71.101386 -40.8202643,-71.1007614 -40.8197749,-71.1003882 -40.8194991,-71.1001736 -40.8192738,-71.100018 -40.8190342,-71.0999376 -40.8186526,-71.0998839 -40.8183116,-71.0998356 -40.8178468,-71.0998356 -40.8174631,-71.0998464 -40.8170044,-71.0998571 -40.8164624,-71.099892 -40.8161579,-71.0999617 -40.8159102,-71.1000636 -40.8155956,-71.1001655 -40.8152444,-71.1002272 -40.8149663,-71.100246 -40.8147207,-71.1003077 -40.8143512,-71.1004713 -40.8137463,-71.1005813 -40.8134012,-71.1007208 -40.813188,-71.1009139 -40.8130155,-71.1010882 -40.8128978,-71.1013806 -40.8127678,-71.1016783 -40.8126521,-71.1020458 -40.8125384,-71.1024535 -40.8123882,-71.1027136 -40.8122522,-71.1028907 -40.8121182,-71.1030275 -40.8119863,-71.1032769 -40.8117406,-71.103521 -40.8115112,-71.1037275 -40.8113772,-71.1039689 -40.8112636,-71.1042157 -40.81116,-71.1046556 -40.8109956,-71.1059833 -40.810419,-71.1078125 -40.8094852,-71.1089256 -40.8088802,-71.1095157 -40.8085128,-71.1099637 -40.8082001,-71.110763 -40.8075809,-71.1113825 -40.8071586,-71.11175 -40.8069434,-71.1122301 -40.8067668,-71.1127827 -40.8065841,-71.1136946 -40.8062674,-71.1147782 -40.8059588,-71.1157687 -40.8057113,-71.1171815 -40.8053355,-71.1180988 -40.8050208,-71.1186326 -40.8048096,-71.1190027 -40.8046188,-71.1191851 -40.804495,-71.1193433 -40.8043224,-71.1194211 -40.8041254,-71.119456 -40.8039123,-71.1194828 -40.8036158,-71.1194667 -40.8031529,-71.1194399 -40.8027326,-71.1194238 -40.8023144,-71.119405 -40.8020606,-71.1193809 -40.8018596,-71.1192897 -40.8015327,-71.1191475 -40.8011469,-71.1187747 -40.8002759,-71.1185655 -40.7998231,-71.1182919 -40.7991246,-71.1180532 -40.7986231,-71.117844 -40.7980932,-71.1177152 -40.7975409,-71.1176509 -40.7969683,-71.1176536 -40.7965439,-71.1176965 -40.7959124,-71.1178789 -40.7948565,-71.1179647 -40.7943834,-71.1181363 -40.7938311,-71.118249 -40.7935732,-71.1183617 -40.7933965,-71.118595 -40.7931894,-71.1188793 -40.7929437,-71.1191878 -40.7926127,-71.1193782 -40.7923365,-71.1194855 -40.7920218,-71.1195445 -40.7916339,-71.1195874 -40.7912095,-71.119574 -40.7907648,-71.1195177 -40.790383,-71.1194265 -40.7899931,-71.1192843 -40.7895463,-71.1190939 -40.7891057,-71.1189008 -40.7886548,-71.1187157 -40.7883055,-71.1184609 -40.7878608,-71.1182544 -40.7875887,-71.1179701 -40.7871602,-71.1178279 -40.7868982,-71.117785 -40.7866443,-71.1178038 -40.78636,-71.1178172 -40.7860188,-71.1178064 -40.7856452,-71.1177903 -40.7853324,-71.1177903 -40.7841809,-71.1179888 -40.7833807,-71.1182034 -40.7826374,-71.1184716 -40.782089,-71.1191958 -40.781037,-71.1196089 -40.7806877,-71.1199683 -40.7804033,-71.1203116 -40.7802287,-71.1205798 -40.7801352,-71.1209661 -40.7800702,-71.1216474 -40.7800337,-71.1225379 -40.7799931,-71.1232567 -40.7799606,-71.1235946 -40.779859,-71.1239219 -40.779729,-71.1241043 -40.7795868,-71.1242008 -40.7794325,-71.1242759 -40.7792213,-71.1242598 -40.7790466,-71.1241365 -40.7789044,-71.1239487 -40.7787988,-71.1237288 -40.7787054,-71.1234337 -40.7785794,-71.1232084 -40.778417,-71.122967 -40.778161,-71.1227578 -40.7778604,-71.1226076 -40.7775233,-71.1225754 -40.7772064,-71.1226183 -40.7770236,-71.1227417 -40.7768489,-71.1229026 -40.7766783,-71.1229992 -40.7765321,-71.1230636 -40.7763899,-71.1231226 -40.7761705,-71.1231977 -40.7757237,-71.1231601 -40.775354,-71.1230314 -40.7749437,-71.1228865 -40.7744847,-71.1226291 -40.7737169,-71.1222428 -40.7728028,-71.1218941 -40.7721569,-71.1214972 -40.7715191,-71.120891 -40.7707065,-71.1203599 -40.7700646,-71.119979 -40.7696096,-71.1189222 -40.7683258,-71.1183697 -40.7677164,-71.1181337 -40.7674157,-71.1180532 -40.7671638,-71.1180371 -40.7669851,-71.1179888 -40.7666519,-71.1179888 -40.7662578,-71.118021 -40.7658353,-71.1180961 -40.7655427,-71.1182088 -40.7652664,-71.1184126 -40.7650064,-71.1187184 -40.7647017,-71.1190027 -40.7644417,-71.1193031 -40.7642791,-71.1195928 -40.7642019,-71.1199254 -40.764141,-71.1202687 -40.7640638,-71.1206818 -40.7639175,-71.1210036 -40.7637713,-71.121127 -40.7636087,-71.121186 -40.763434,-71.121186 -40.7631781,-71.121127 -40.7626783,-71.121068 -40.7620932,-71.1209607 -40.761435,-71.1208159 -40.7605776,-71.1208105 -40.7601144,-71.1207944 -40.7595943,-71.1208695 -40.7590336,-71.1208963 -40.7584403,-71.1208373 -40.7579649,-71.1207408 -40.7575057,-71.120553 -40.7570141,-71.1203438 -40.7567215,-71.1201346 -40.7565102,-71.1198878 -40.756372,-71.1195552 -40.7562664,-71.1191583 -40.756242,-71.1185789 -40.7562704,-71.1180049 -40.7563151,-71.1176187 -40.7563273,-71.11727 -40.7563192,-71.1169538 -40.7562688,-71.116342 -40.7560469,-71.115548 -40.7555918,-71.1153603 -40.7553561,-71.1151725 -40.7550189,-71.115033 -40.7545881,-71.1148077 -40.7539623,-71.1146253 -40.7535641,-71.1144215 -40.7531699,-71.114105 -40.752796,-71.1137402 -40.7524303,-71.113236 -40.7520768,-71.1127532 -40.7518248,-71.1123186 -40.7516663,-71.1118949 -40.7515525,-71.1114872 -40.7514225,-71.1109883 -40.7512396,-71.1102963 -40.7509348,-71.1088747 -40.7503334,-71.1077535 -40.7498335,-71.1070293 -40.7494881,-71.1066163 -40.749228,-71.1062139 -40.7489232,-71.1058706 -40.7485493,-71.1056614 -40.748212,-71.1055273 -40.7479926,-71.1053395 -40.7477772,-71.1051035 -40.7475658,-71.1048675 -40.7474317,-71.1045456 -40.7473139,-71.1040145 -40.7471594,-71.1033869 -40.7470009,-71.1029041 -40.7468627,-71.102609 -40.7467083,-71.1023891 -40.7465092,-71.1022282 -40.7463222,-71.1020941 -40.7460296,-71.1020565 -40.7457289,-71.1020297 -40.7453509,-71.1019868 -40.7449282,-71.1020726 -40.7443308,-71.1022979 -40.7435667,-71.1026466 -40.742957,-71.1029524 -40.7423474,-71.1030757 -40.7419288,-71.1030757 -40.741567,-71.1029685 -40.7412053,-71.1027914 -40.740807,-71.1024106 -40.7402745,-71.1022764 -40.7400347,-71.1021477 -40.7397421,-71.102078 -40.7394413,-71.1020404 -40.7389901,-71.1019921 -40.7384983,-71.1019278 -40.7381935,-71.1017668 -40.7379374,-71.1015576 -40.7377545,-71.101166 -40.737535,-71.100651 -40.7373521,-71.0996801 -40.7371001,-71.0992026 -40.7369456,-71.0987145 -40.7367261,-71.0982531 -40.7364253,-71.097942 -40.7361368,-71.0977864 -40.7357628,-71.0977542 -40.7354254,-71.0977382 -40.7349661,-71.0977542 -40.7345718,-71.0978079 -40.7339377,-71.0979688 -40.7333239,-71.0981244 -40.7327101,-71.098178 -40.7323361,-71.0981995 -40.732015,-71.0981298 -40.7317914,-71.0979152 -40.7315109,-71.0972496 -40.7307102,-71.0959303 -40.7295109,-71.0951042 -40.7282995,-71.0945678 -40.7275352,-71.0940206 -40.7269986,-71.0936022 -40.7267303,-71.0930014 -40.7265677,-71.09195 -40.7262993,-71.0904908 -40.7259416,-71.0885596 -40.7252993,-71.0878408 -40.7250106,-71.0874063 -40.7248683,-71.0870791 -40.7247057,-71.0867143 -40.7244902,-71.0862851 -40.7243358,-71.0856199 -40.7242138,-71.0845685 -40.7242179,-71.0838443 -40.7242992,-71.0827446 -40.7244537,-71.0813499 -40.7248033,-71.0797405 -40.7252749,-71.0779488 -40.7259253,-71.0751915 -40.7277466,-71.0712112 -40.7296156,-71.0590834 -40.7281411,-71.0473616 -40.7297522,-71.0357071 -40.7263378,-71.027674 -40.7244314,-71.0260599 -40.7221857,-71.0249537 -40.7206467,-71.0276571 -40.7177782,-71.0268761 -40.7121775,-71.0207156 -40.706487,-71.0143822 -40.6983708,-71.0143515 -40.6963516,-71.0153032 -40.6943381,-71.0162473 -40.6930203,-71.0166153 -40.6914813,-71.016376 -40.6889202,-71.0151744 -40.6840063,-71.0116864 -40.6801126,-71.0054541 -40.675398,-70.9970476 -40.669388,-70.987334 -40.6591461,-70.9868822 -40.6584674,-70.9793923 -40.6501003,-70.9637574 -40.6429964,-70.9490408 -40.6410003,-70.9329321 -40.6416423,-70.9243602 -40.6360491,-70.9169147 -40.6302075,-70.9023728 -40.6242455,-70.8901546 -40.6242558,-70.8879824 -40.6242576,-70.8804694 -40.6227812,-70.8762606 -40.6202175,-70.8745852 -40.6134423,-70.8734742 -40.6038984,-71.0107681 -40.598733,-71.1478594 -40.5935749,-71.1554645 -40.5998744,-71.1596101 -40.5981482,-71.1691049 -40.5965235,-71.1687037 -40.5908369,-71.1742421 -40.5884765,-71.1887633 -40.5874856,-71.1941125 -40.5852513,-71.1992682 -40.5843494,-71.2067233 -40.5883056,-71.2392101 -40.5880642,-71.24625 -40.5906753,-71.2653247 -40.589497,-71.2799972 -40.5932632,-71.2857476 -40.5929586,-71.2908142 -40.591315,-71.3038012 -40.5821936,-71.3094179 -40.5793497,-71.3152022 -40.5733729,-71.3231921 -40.5714269,-71.3280063 -40.5681762,-71.3348088 -40.5590407,-71.3441877 -40.5508028,-71.3617706 -40.5407129,-71.3728319 -40.53449,-71.3857778 -40.5373887,-71.4292084 -40.5277538,-71.4626071 -40.5266989,-71.4699633 -40.5207012,-71.4795661 -40.5186113,-71.4923319 -40.5104159,-71.4980056 -40.5096251,-71.5064216 -40.5058145,-71.5133246 -40.5015724,-71.5180527 -40.4976175,-71.5363661 -40.4959889,-71.5363598 -40.4940736,-71.5375829 -40.4914953,-71.5401364 -40.4843963,-71.539836 -40.479549,-71.5406299 -40.4784228,-71.5419817 -40.477215,-71.543752 -40.4758929,-71.5446639 -40.4756807,-71.546638 -40.4750645,-71.5471315 -40.4739424,-71.5478504 -40.4726039,-71.5486658 -40.4704573,-71.5498352 -40.4685311,-71.5509081 -40.4675761,-71.5528286 -40.4665477,-71.5542448 -40.4663681,-71.5556932 -40.4667354,-71.558075 -40.4670864,-71.5597165 -40.4666538,-71.5616155 -40.4654213,-71.5635252 -40.4648172,-71.5649521 -40.4638622,-71.5658856 -40.462997,-71.5670443 -40.4588012,-71.5699947 -40.4557889,-71.5717328 -40.4542867,-71.5742326 -40.4526703,-71.576314 -40.4517232,-71.5813136 -40.4517151,-71.5833736 -40.45196,-71.5861631 -40.4527764,-71.5886307 -40.4538949,-71.5901435 -40.4542704,-71.5934479 -40.4550705,-71.5975142 -40.4552092,-71.5991342 -40.4555358,-71.6003144 -40.456197,-71.6037262 -40.4587522,-71.6043913 -40.4598134,-71.6045952 -40.4608909,-71.6051102 -40.4614215,-71.6057968 -40.4619929,-71.6079104 -40.4607277,-71.6092891 -40.4591277,-71.6091442 -40.4572093,-71.6092086 -40.4548827,-71.6070963 -40.4514232,-71.6071487 -40.4496454,-71.60945 -40.447539,-71.6115099 -40.4466899,-71.6129315 -40.4457223,-71.6144563 -40.444964,-71.6172713 -40.4451773,-71.6192883 -40.4458019,-71.6213107 -40.4471838,-71.6232634 -40.4476308,-71.6262245 -40.4480819,-71.6294164 -40.4486657,-71.6317391 -40.4482207,-71.6328549 -40.4466695,-71.6342658 -40.4454692,-71.6375307 -40.44464,-71.6423285 -40.4442077,-71.6449034 -40.4451141,-71.6489321 -40.4469675,-71.6504288 -40.448531,-71.6523171 -40.4501312,-71.6553104 -40.4504619,-71.6573542 -40.4497965,-71.6594584 -40.4482452,-71.6608411 -40.4461673,-71.663636 -40.442934,-71.665535 -40.4423502,-71.6680241 -40.4428646,-71.670717 -40.4437464,-71.6728413 -40.4437464,-71.6751051 -40.4427217,-71.6785008 -40.4374957,-71.6809309 -40.4360585,-71.6960693 -40.4334412,-71.697979 -40.4308851,-71.6977751 -40.4294192,-71.6982526 -40.4286229,-71.7011279 -40.427553))"
        );
    }
}
