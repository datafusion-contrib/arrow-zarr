// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::joinable_geo::{
    covered_range, point_on_segment_t, ray_crosses_edge, segments_cross, Accumulator, Bboxable,
    Edge, JoinableGeo, LineSegment, Point, SegmentTrait,
};

// *************************************************************
// st_intersects(a, b) -> true if a and b share at least one point. Unlike
// st_within, this is existential (the first genuine touch wins) and symmetric.
// *************************************************************
pub(crate) fn st_intersects(a: &JoinableGeo, b: &JoinableGeo) -> bool {
    match (a, b) {
        (JoinableGeo::Point { points: a_pts }, JoinableGeo::Point { points: b_pts }) => {
            let mut acc = PointIntersectsPoint::new(a_pts, b_pts);
            a.fold_for_unordered_check(b, &mut acc);
            acc.finish()
        }

        (JoinableGeo::Point { points }, JoinableGeo::Line { lines, .. }) => {
            let mut acc = PointIntersectsLine::new(points, lines);
            a.fold_for_unordered_check(b, &mut acc);
            acc.finish()
        }
        (JoinableGeo::Point { points }, JoinableGeo::Poly { edges, .. }) => {
            let mut acc = PointIntersectsPoly::new(points, edges);
            a.fold_for_grouped_check(b, &mut acc);
            acc.finish()
        }

        (JoinableGeo::Line { lines: l_lines, .. }, JoinableGeo::Line { lines: r_lines, .. }) => {
            let mut acc = LineIntersectsLine::new(l_lines, r_lines);
            a.fold_for_unordered_check(b, &mut acc);
            acc.finish()
        }

        (JoinableGeo::Line { lines, .. }, JoinableGeo::Poly { edges, .. }) => {
            let mut acc = SegmentsIntersectPoly::new(lines, edges, true);
            a.fold_for_grouped_check(b, &mut acc);
            acc.finish()
        }

        (JoinableGeo::Poly { edges: a_edges, .. }, JoinableGeo::Poly { edges: b_edges, .. }) => {
            // First pass (A against B): boundary crossings and A contained in B.
            let mut acc = SegmentsIntersectPoly::new(a_edges, b_edges, true);
            a.fold_for_grouped_check(b, &mut acc);
            if acc.finish() {
                return true;
            }
            // Second pass (B against A) catches the remaining case, B contained in
            // A. No crossings exist by now, so skip the boundary-contact test.
            let mut acc = SegmentsIntersectPoly::new(b_edges, a_edges, false);
            b.fold_for_grouped_check(a, &mut acc);
            acc.finish()
        }

        // Symmetric: forward to the point-on-left orientation.
        (JoinableGeo::Line { .. }, JoinableGeo::Point { .. })
        | (JoinableGeo::Poly { .. }, JoinableGeo::Point { .. })
        | (JoinableGeo::Poly { .. }, JoinableGeo::Line { .. }) => st_intersects(b, a),
    }
}

// *************************************************************
// Accumulators. Each carries a `found` flag flipped on the first genuine
// geometric touch; `ready` and `finish` both just report it.
// *************************************************************

// Two point sets intersect if any left point equals any right point. Exact
// equality, matching PointInPoint in st_within.
pub(crate) struct PointIntersectsPoint<'a> {
    left: &'a [Point],
    right: &'a [Point],
    found: bool,
}

impl<'a> PointIntersectsPoint<'a> {
    pub(crate) fn new(left: &'a [Point], right: &'a [Point]) -> Self {
        PointIntersectsPoint {
            left,
            right,
            found: false,
        }
    }
}

impl Accumulator for PointIntersectsPoint<'_> {
    // No ray casting, so a symmetric bbox overlap is the correct, tighter prune.
    fn prune(a: &impl Bboxable, b: &impl Bboxable) -> bool {
        a.box_overlap(b)
    }

    fn update(&mut self, li: usize, ri: usize) {
        if self.left[li].is_equal(&self.right[ri]) {
            self.found = true;
        }
    }

    fn ready(&self) -> bool {
        self.found
    }

    fn finish(self) -> bool {
        self.found
    }
}

// A point set and a line intersect if any point lies on any segment (interior
// or endpoint — intersects doesn't distinguish). No ray casting, so box_overlap.
pub(crate) struct PointIntersectsLine<'a> {
    points: &'a [Point],
    lines: &'a [LineSegment],
    found: bool,
}

impl<'a> PointIntersectsLine<'a> {
    pub(crate) fn new(points: &'a [Point], lines: &'a [LineSegment]) -> Self {
        PointIntersectsLine {
            points,
            lines,
            found: false,
        }
    }
}

impl Accumulator for PointIntersectsLine<'_> {
    fn prune(a: &impl Bboxable, b: &impl Bboxable) -> bool {
        a.box_overlap(b)
    }

    fn update(&mut self, li: usize, ri: usize) {
        if point_on_segment_t(&self.points[li], &self.lines[ri]).is_some() {
            self.found = true;
        }
    }

    fn ready(&self) -> bool {
        self.found
    }

    fn finish(self) -> bool {
        self.found
    }
}

// A point set intersects a poly if any point is on the boundary or inside.
// Grouped single descent (point as query) so ray-parity sees all of a point's
// candidate edges contiguously; default ray_candidate prune. A point on an edge
// is an immediate hit; otherwise odd ray-crossing parity at group end means
// inside. Points that match nothing are outside and simply ignored.
pub(crate) struct PointIntersectsPoly<'a> {
    points: &'a [Point],
    edges: &'a [Edge],
    current_group: Option<usize>,
    parity: bool,
    found: bool,
}

impl<'a> PointIntersectsPoly<'a> {
    pub(crate) fn new(points: &'a [Point], edges: &'a [Edge]) -> Self {
        PointIntersectsPoly {
            points,
            edges,
            current_group: None,
            parity: false,
            found: false,
        }
    }

    fn finalize_group(&mut self) {
        if self.parity {
            self.found = true;
        }
    }
}

impl Accumulator for PointIntersectsPoly<'_> {
    fn update(&mut self, li: usize, ri: usize) {
        if self.current_group != Some(li) {
            if self.current_group.is_some() {
                self.finalize_group();
                if self.found {
                    return;
                }
            }
            self.current_group = Some(li);
            self.parity = false;
        }

        let p = &self.points[li];
        let edge = &self.edges[ri];
        if point_on_segment_t(p, edge).is_some() {
            self.found = true;
        } else if ray_crosses_edge(p, edge, &self.edges[edge.next()]) {
            self.parity = !self.parity;
        }
    }

    fn ready(&self) -> bool {
        self.found
    }

    fn finish(mut self) -> bool {
        if self.current_group.is_some() {
            self.finalize_group();
        }
        self.found
    }
}

// Two line sets intersect if any left segment touches any right segment: either a
// non-parallel crossing (`segments_cross`) or a collinear overlap (`covered_range`).
// Purely pairwise, so dual descent + box_overlap.
pub(crate) struct LineIntersectsLine<'a> {
    left: &'a [LineSegment],
    right: &'a [LineSegment],
    found: bool,
}

impl<'a> LineIntersectsLine<'a> {
    pub(crate) fn new(left: &'a [LineSegment], right: &'a [LineSegment]) -> Self {
        LineIntersectsLine {
            left,
            right,
            found: false,
        }
    }
}

impl Accumulator for LineIntersectsLine<'_> {
    fn prune(a: &impl Bboxable, b: &impl Bboxable) -> bool {
        a.box_overlap(b)
    }

    fn update(&mut self, li: usize, ri: usize) {
        let (l, r) = (&self.left[li], &self.right[ri]);
        if covered_range(l, r).is_some() || segments_cross(l, r) {
            self.found = true;
        }
    }

    fn ready(&self) -> bool {
        self.found
    }

    fn finish(self) -> bool {
        self.found
    }
}

// A set of query segments intersects a poly if any segment touches the boundary,
// or any segment lies inside (fully contained, boundaries never crossed). Grouped
// single descent so ray-parity sees all of a segment's candidate edges
// contiguously. Boundary contact uses the inclusive segments_cross/covered_range.
// Generic over the query segment type so it serves both line×poly (query =
// LineSegment) and poly×poly (query = Edge).
//
// `check_crossings` skips the boundary-contact test when the caller already knows
// there are none. the second poly×poly pass (B against A) runs only after the
// first pass found no contact, so it only needs the containment parity check.
pub(crate) struct SegmentsIntersectPoly<'a, L: SegmentTrait> {
    left: &'a [L],
    edges: &'a [Edge],
    check_crossings: bool,
    current_group: Option<usize>,
    parity: bool,
    found: bool,
}

impl<'a, L: SegmentTrait> SegmentsIntersectPoly<'a, L> {
    pub(crate) fn new(left: &'a [L], edges: &'a [Edge], check_crossings: bool) -> Self {
        SegmentsIntersectPoly {
            left,
            edges,
            check_crossings,
            current_group: None,
            parity: false,
            found: false,
        }
    }

    fn finalize_group(&mut self) {
        if self.parity {
            self.found = true;
        }
    }
}

impl<L: SegmentTrait> Accumulator for SegmentsIntersectPoly<'_, L> {
    fn update(&mut self, li: usize, ri: usize) {
        if self.current_group != Some(li) {
            if self.current_group.is_some() {
                self.finalize_group();
                if self.found {
                    return;
                }
            }
            self.current_group = Some(li);
            self.parity = false;
        }

        let seg = &self.left[li];
        let edge = &self.edges[ri];
        // Boundary contact (inclusive of endpoints/vertices) is an immediate hit.
        if self.check_crossings && (covered_range(seg, edge).is_some() || segments_cross(seg, edge))
        {
            self.found = true;
            return;
        }
        // Otherwise cast a rightward ray from the segment midpoint for the
        // fully-inside case. The midpoint can't be on the boundary here, that
        // would be a point shared with this same edge, already caught above (or,
        // when crossings aren't checked, ruled out by the prior pass).
        if ray_crosses_edge(&seg.midpoint(), edge, &self.edges[edge.next()]) {
            self.parity = !self.parity;
        }
    }

    fn ready(&self) -> bool {
        self.found
    }

    fn finish(mut self) -> bool {
        if self.current_group.is_some() {
            self.finalize_group();
        }
        self.found
    }
}

#[cfg(test)]
mod test_helpers {
    use geos::Geom;

    pub(super) use super::super::test_utils::wkt_to_wkb;
    use super::JoinableGeo;
    use crate::geospatial::st_intersects::st_intersects;

    pub(super) fn compare_intersects(label: &str, wkt_a: &str, wkt_b: &str) {
        let bytes_a = wkt_to_wkb(wkt_a);
        let bytes_b = wkt_to_wkb(wkt_b);

        let geos_a = geos::Geometry::new_from_wkb(&bytes_a).unwrap();
        let geos_b = geos::Geometry::new_from_wkb(&bytes_b).unwrap();
        let geos_result = geos_a.intersects(&geos_b).unwrap();

        let joinable_a = JoinableGeo::from_wkb(&bytes_a).unwrap();
        let joinable_b = JoinableGeo::from_wkb(&bytes_b).unwrap();

        // Intersects is symmetric, so both orderings must match GEOS.
        assert_eq!(
            geos_result,
            st_intersects(&joinable_a, &joinable_b),
            "{label}"
        );
        assert_eq!(
            geos_result,
            st_intersects(&joinable_b, &joinable_a),
            "{label}, swapped"
        );
    }
}

#[cfg(test)]
mod intersects_tests {
    use super::test_helpers::compare_intersects;

    #[test]
    fn point_left() {
        compare_intersects("point left, case 1", "POINT (1 1)", "POINT (1 1)");
        compare_intersects("point left, case 2", "POINT (1 1)", "POINT (2 2)");
        compare_intersects("point left, case 3", "POINT (1 1)", "MULTIPOINT (0 0, 1 1)");
        compare_intersects("point left, case 4", "POINT (1 1)", "MULTIPOINT (0 0, 2 2)");

        compare_intersects(
            "point left, case 5",
            "POINT (0.5 0)",
            "LINESTRING (0 0, 1 0)",
        );
        compare_intersects("point left, case 6", "POINT (2 0)", "LINESTRING (0 0, 1 0)");
        compare_intersects("point left, case 7", "POINT (0 0)", "LINESTRING (0 0, 1 0)");
        compare_intersects(
            "point left, case 8",
            "POINT (0.5 0)",
            "MULTILINESTRING ((0 0, 1 0), (0 1, 1 1))",
        );
        compare_intersects(
            "point left, case 9",
            "POINT (5 5)",
            "MULTILINESTRING ((0 0, 1 0), (0 1, 1 1))",
        );
        compare_intersects(
            "point left, case 10",
            "POINT (1 0)",
            "MULTILINESTRING ((0 0, 1 0), (1 0, 2 0))",
        );

        compare_intersects(
            "point left, case 11",
            "POINT (1 1)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );
        compare_intersects(
            "point left, case 12",
            "POINT (2 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1))",
        );
        compare_intersects(
            "point left, case 13",
            "POINT (5 5)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );
        compare_intersects(
            "point left, case 14",
            "POINT (1 1)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((3 0, 5 0, 5 2, 3 2, 3 0)))",
        );
        compare_intersects(
            "point left, case 15",
            "POINT (1 0)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );
        compare_intersects(
            "point left, case 16",
            "POINT (0 0)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );
        compare_intersects(
            "point left, case 17",
            "POINT (2 2)",
            "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1)), ((5 0, 7 0, 7 2, 5 2, 5 0)))",
        );
    }

    #[test]
    fn multi_point_left() {
        compare_intersects(
            "multi point left, case 1",
            "MULTIPOINT (1 1, 2 2)",
            "POINT (1 1)",
        );
        compare_intersects(
            "multi point left, case 2",
            "MULTIPOINT (1 1, 2 2)",
            "POINT (3 3)",
        );

        compare_intersects(
            "multi point left, case 3",
            "MULTIPOINT (0.5 0, 5 5)",
            "LINESTRING (0 0, 1 0)",
        );
        compare_intersects(
            "multi point left, case 4",
            "MULTIPOINT (5 5, 6 6)",
            "LINESTRING (0 0, 1 0)",
        );
        compare_intersects(
            "multi point left, case 5",
            "MULTIPOINT (1 0, 5 5)",
            "LINESTRING (0 0, 1 0, 2 0)",
        );
        compare_intersects(
            "multi point left, case 6",
            "MULTIPOINT (5 5, 6 6)",
            "MULTILINESTRING ((0 0, 1 0), (0 1, 1 1))",
        );
        compare_intersects(
            "multi point left, case 7",
            "MULTIPOINT (0.5 0, 0.5 1)",
            "MULTILINESTRING ((0 0, 1 0), (0 1, 1 1))",
        );

        compare_intersects(
            "multi point left, case 8",
            "MULTIPOINT (1 1, 5 5)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );
        compare_intersects(
            "multi point left, case 9",
            "MULTIPOINT (5 5, 6 6)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );
        compare_intersects(
            "multi point left, case 10",
            "MULTIPOINT (2 2, 2.5 2.5)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1))",
        );
        compare_intersects(
            "multi point left, case 11",
            "MULTIPOINT (1 1, 8 8)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((3 0, 5 0, 5 2, 3 2, 3 0)))",
        );
    }

    #[test]
    fn line_left() {
        compare_intersects("line left, case 1", "LINESTRING (0 0, 4 0)", "POINT (2 0)");
        compare_intersects("line left, case 2", "LINESTRING (0 0, 4 0)", "POINT (2 2)");

        compare_intersects(
            "line left, case 3",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (2 -2, 2 2)",
        );
        compare_intersects(
            "line left, case 4",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (2 0, 6 0)",
        );
        compare_intersects(
            "line left, case 5",
            "LINESTRING (1 0, 3 0)",
            "LINESTRING (0 0, 4 0)",
        );
        compare_intersects(
            "line left, case 6",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (1 0, 3 0)",
        );
        compare_intersects(
            "line left, case 7",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (2 0, 2 2)",
        );
        compare_intersects(
            "line left, case 8",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (4 0, 6 2)",
        );
        compare_intersects(
            "line left, case 9",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (0 2, 4 2)",
        );
        compare_intersects(
            "line left, case 10",
            "LINESTRING (0 2, 4 2)",
            "MULTILINESTRING ((0 0, 2 2), (2 2, 4 0))",
        );

        compare_intersects(
            "line left, case 11",
            "LINESTRING (-1 1, 3 1)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );
        compare_intersects(
            "line left, case 12",
            "LINESTRING (0.5 0.5, 1.5 1.5)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );
        compare_intersects(
            "line left, case 13",
            "LINESTRING (0.5 0, 1.5 0)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );
        compare_intersects(
            "line left, case 14",
            "LINESTRING (5 5, 6 6)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );
        compare_intersects(
            "line left, case 15",
            "LINESTRING (1.5 2, 2.5 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1))",
        );
        compare_intersects(
            "line left, case 16",
            "LINESTRING (2 2, 2 0.5)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1))",
        );
        compare_intersects(
            "line left, case 17",
            "LINESTRING (-1 1, 1 1)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((3 0, 5 0, 5 2, 3 2, 3 0)))",
        );
        compare_intersects(
            "line left, case 18",
            "LINESTRING (10 10, 11 11)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((3 0, 5 0, 5 2, 3 2, 3 0)))",
        );
    }

    #[test]
    fn multi_line_left() {
        compare_intersects(
            "multi line left, case 1",
            "MULTILINESTRING ((0 0, 2 2), (2 2, 4 0))",
            "MULTILINESTRING ((0 4, 2 2), (2 2, 4 4))",
        );
        compare_intersects(
            "multi line left, case 2",
            "MULTILINESTRING ((2 2, 4 2), (4 2, 6 2))",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );
        compare_intersects(
            "multi line left, case 3",
            "MULTILINESTRING ((3 3, 3 2), (3 2, 3 1))",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))",
        );
        compare_intersects(
            "multi line left, case 4",
            "MULTILINESTRING ((-1 1, 1 1), (10 10, 11 11))",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((3 0, 5 0, 5 2, 3 2, 3 0)))",
        );
        compare_intersects(
            "multi line left, case 5",
            "MULTILINESTRING ((10 10, 11 11), (12 12, 13 13))",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((3 0, 5 0, 5 2, 3 2, 3 0)))",
        );
    }

    #[test]
    fn poly_left() {
        compare_intersects(
            "poly left, case 1",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
            "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))",
        );
        compare_intersects(
            "poly left, case 2",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
            "POLYGON ((5 5, 7 5, 7 7, 5 7, 5 5))",
        );
        compare_intersects(
            "poly left, case 3",
            "POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1))",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );
        compare_intersects(
            "poly left, case 4",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            "POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1))",
        );
        compare_intersects(
            "poly left, case 5",
            "POLYGON ((2 2, 4 2, 4 4, 2 4, 2 2))",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0), (1 1, 5 1, 5 5, 1 5, 1 1))",
        );
        compare_intersects(
            "poly left, case 6",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0), (1 1, 5 1, 5 5, 1 5, 1 1))",
            "POLYGON ((2 2, 4 2, 4 4, 2 4, 2 2))",
        );
        compare_intersects(
            "poly left, case 7",
            "POLYGON ((1 1, 4 1, 4 4, 1 4, 1 1))",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((5 0, 7 0, 7 2, 5 2, 5 0)))",
        );
        compare_intersects(
            "poly left, case 8",
            "POLYGON ((10 10, 11 10, 11 11, 10 11, 10 10))",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((5 0, 7 0, 7 2, 5 2, 5 0)))",
        );
    }

    #[test]
    fn multi_poly_left() {
        compare_intersects(
            "multi poly left, case 1",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((10 10, 12 10, 12 12, 10 12, 10 10)))",
            "MULTIPOLYGON (((1 1, 3 1, 3 3, 1 3, 1 1)), ((20 20, 22 20, 22 22, 20 22, 20 20)))",
        );
        compare_intersects(
            "multi poly left, case 2",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((10 10, 12 10, 12 12, 10 12, 10 10)))",
            "MULTIPOLYGON (((30 30, 32 30, 32 32, 30 32, 30 30)), ((40 40, 42 40, 42 42, 40 42, 40 40)))",
        );
        compare_intersects(
            "multi poly left, case 3",
            "MULTIPOLYGON (((4 4, 6 4, 6 6, 4 6, 4 4)), ((30 30, 32 30, 32 32, 30 32, 30 30)))",
            "MULTIPOLYGON (((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 8 2, 8 8, 2 8, 2 2)), ((20 0, 22 0, 22 2, 20 2, 20 0)))",
        );
        compare_intersects(
            "multi poly left, case 4",
            "MULTIPOLYGON (((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 8 2, 8 8, 2 8, 2 2)), ((30 30, 32 30, 32 32, 30 32, 30 30)))",
            "MULTIPOLYGON (((4 4, 6 4, 6 6, 4 6, 4 4)), ((50 50, 52 50, 52 52, 50 52, 50 50)))",
        );
    }

    #[test]
    fn empty_geo() {
        let sq = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";
        compare_intersects("empty multipoint left", "MULTIPOINT EMPTY", sq);
        compare_intersects("empty multilinestring left", "MULTILINESTRING EMPTY", sq);
        compare_intersects("empty polygon left", "POLYGON EMPTY", sq);
        compare_intersects("empty multipolygon left", "MULTIPOLYGON EMPTY", sq);
    }
}
