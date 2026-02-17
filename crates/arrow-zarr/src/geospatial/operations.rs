use std::collections::HashMap;

use datafusion::error::Result;
use wkb::reader::Wkb;

use super::extension_traits::{GeoComponentType, WkbVecOps};
use super::spatial_predicate::SpatialRelationType;

const _N_Y_STRIPES: usize = 64;
const _EPS: f64 = 1e-10;
// Segment parameter range that includes endpoints (with tolerance): used by point_on_edge.
const _UNIT_RANGE: std::ops::RangeInclusive<f64> = -_EPS..=1.0 + _EPS;
// Segment parameter range that excludes endpoints: used by edge_crosses_edge.
const _UNIT_INTERIOR: std::ops::RangeInclusive<f64> = _EPS..=1.0 - _EPS;

pub(crate) struct GeoComponentGrouping {
    groups: HashMap<(GeoComponentType, GeoComponentType), (usize, usize)>,
}

impl GeoComponentGrouping {
    fn new() -> Self {
        Self {
            groups: HashMap::new(),
        }
    }

    fn add(
        &mut self,
        geo_t1: GeoComponentType,
        geo_t2: GeoComponentType,
        start: usize,
        end: usize,
    ) {
        self.groups.insert((geo_t1, geo_t2), (start, end));
    }

    fn get_start_and_end(
        &self,
        geo_t1: GeoComponentType,
        geo_t2: GeoComponentType,
    ) -> Option<&(usize, usize)> {
        self.groups.get(&(geo_t1, geo_t2))
    }
}

#[derive(Debug)]
pub(crate) struct ExplodedSide {
    x1: Vec<f64>,
    y1: Vec<f64>,
    x2: Vec<f64>,
    y2: Vec<f64>,
    geo_index: Vec<u32>,
    pub(crate) component_type: Vec<GeoComponentType>,
}

impl ExplodedSide {
    pub(crate) fn new(geo_ids: &[u32], wkbs: &[Wkb<'_>]) -> Result<Self> {
        let indices: Vec<usize> = geo_ids.iter().map(|&id| id as usize).collect();
        let ((x1, y1, x2, y2, component_type), exploded_indices) = wkbs.explode(&indices)?;
        Ok(Self {
            x1,
            y1,
            x2,
            y2,
            geo_index: exploded_indices.into_iter().map(|i| i as u32).collect(),
            component_type,
        })
    }

    fn cross_join(
        &self,
        other: &ExplodedSide,
        left_range: std::ops::Range<usize>,
        right_side_stripes: Option<&(Vec<Vec<usize>>, f64, f64)>,
        rel_type: &SpatialRelationType,
    ) -> (Vec<usize>, Vec<usize>) {
        use std::collections::HashSet;
        let mut seen: HashSet<(u32, u32)> = HashSet::new();
        let mut left_idx = Vec::new();
        let mut right_idx = Vec::new();
        let n_right = other.x1.len();

        for li in left_range {
            let mut candidates = Vec::new();
            match right_side_stripes {
                None => candidates.extend(0..n_right),
                Some((stripes, y_min, stripe_height)) => {
                    let n_stripes = stripes.len();
                    let s_lo = (((self.y1[li].min(self.y2[li]) - y_min) / stripe_height) as usize)
                        .min(n_stripes - 1);
                    let s_hi = (((self.y1[li].max(self.y2[li]) - y_min) / stripe_height) as usize)
                        .min(n_stripes - 1);
                    candidates.extend(stripes[s_lo..=s_hi].iter().flatten());
                }
            }
            for ri in candidates {
                if should_drop(
                    &self.component_type[li],
                    &other.component_type[ri],
                    rel_type,
                ) {
                    continue;
                }
                if seen.insert((li as u32, ri as u32)) {
                    left_idx.push(li);
                    right_idx.push(ri);
                }
            }
        }
        (left_idx, right_idx)
    }

    fn group_by_component_type(
        &self,
        other: &ExplodedSide,
        left_indices: &mut [usize],
        right_indices: &mut [usize],
    ) -> GeoComponentGrouping {
        let mut pairs: Vec<_> = left_indices
            .iter()
            .copied()
            .zip(right_indices.iter().copied())
            .collect();
        pairs.sort_by_key(|&(li, ri)| (self.component_type[li], other.component_type[ri]));
        for (i, (l, r)) in pairs.into_iter().enumerate() {
            left_indices[i] = l;
            right_indices[i] = r;
        }

        let n = left_indices.len();
        let mut grouping = GeoComponentGrouping::new();
        let mut i = 0;
        while i < n {
            let lt = self.component_type[left_indices[i]];
            let rt = other.component_type[right_indices[i]];
            let start = i;
            while i < n
                && self.component_type[left_indices[i]] == lt
                && other.component_type[right_indices[i]] == rt
            {
                i += 1;
            }
            grouping.add(lt, rt, start, i);
        }
        grouping
    }

    pub(crate) fn join_on_y_stripes(
        &self,
        other: &ExplodedSide,
        rel_type: SpatialRelationType,
        require_all_left: bool,
    ) -> (Vec<usize>, Vec<usize>) {
        use std::collections::{HashMap, HashSet};

        let n_left = self.x1.len();
        let n_right = other.x1.len();

        if n_left == 0 || n_right == 0 {
            return (Vec::new(), Vec::new());
        }

        // Compute global y-extent across both sides.
        let y_extent = |y1: &[f64], y2: &[f64]| -> (f64, f64) {
            y1.iter()
                .zip(y2.iter())
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), (&a, &b)| {
                    (lo.min(a.min(b)), hi.max(a.max(b)))
                })
        };
        let (left_lo, left_hi) = y_extent(&self.y1, &self.y2);
        let (right_lo, right_hi) = y_extent(&other.y1, &other.y2);
        let y_min = left_lo.min(right_lo);
        let y_max = left_hi.max(right_hi);

        let n_stripes = _N_Y_STRIPES.min(n_left.max(n_right));
        let stripe_height = (y_max - y_min) / n_stripes as f64;

        // for the degenerate case, just do a simple cross join and
        // return.
        if stripe_height <= 0.0 || !stripe_height.is_finite() {
            return self.cross_join(other, 0..n_left, None, &rel_type);
        }

        // aggregate the right side indices per y stripe.
        let mut right_side_stripes: Vec<Vec<usize>> = vec![Vec::new(); n_stripes];
        (0..n_right).for_each(|ri| {
            let s_lo = (((other.y1[ri].min(other.y2[ri]) - y_min) / stripe_height) as usize)
                .min(n_stripes - 1);
            let s_hi = (((other.y1[ri].max(other.y2[ri]) - y_min) / stripe_height) as usize)
                .min(n_stripes - 1);
            (s_lo..=s_hi).for_each(|s| right_side_stripes[s].push(ri));
        });
        let stripes_arg = Some((right_side_stripes, y_min, stripe_height));

        let mut left_out_indices: Vec<usize> = Vec::new();
        let mut right_out_indices: Vec<usize> = Vec::new();
        let mut gstart = 0;
        while gstart < n_left {
            let gi = self.geo_index[gstart];
            let mut gend = gstart + 1;
            while gend < n_left && self.geo_index[gend] == gi {
                gend += 1;
            }

            let (mut grp_left, mut grp_right) =
                self.cross_join(other, gstart..gend, stripes_arg.as_ref(), &rel_type);

            if require_all_left {
                let group_size = gend - gstart;
                let mut coverage: HashMap<u32, HashSet<usize>> = HashMap::new();
                grp_left
                    .iter()
                    .zip(grp_right.iter())
                    .for_each(|(&li, &ri)| {
                        coverage.entry(other.geo_index[ri]).or_default().insert(li);
                    });
                let complete_right: HashSet<u32> = coverage
                    .into_iter()
                    .filter(|(_, s)| s.len() == group_size)
                    .map(|(rg, _)| rg)
                    .collect();

                let keep: Vec<bool> = grp_right
                    .iter()
                    .map(|&ri| complete_right.contains(&other.geo_index[ri]))
                    .collect();
                grp_left = grp_left
                    .into_iter()
                    .zip(keep.iter())
                    .filter(|(_, &k)| k)
                    .map(|(li, _)| li)
                    .collect();
                grp_right = grp_right
                    .into_iter()
                    .zip(keep.iter())
                    .filter(|(_, &k)| k)
                    .map(|(ri, _)| ri)
                    .collect();
            }

            left_out_indices.extend(grp_left);
            right_out_indices.extend(grp_right);
            gstart = gend;
        }

        (left_out_indices, right_out_indices)
    }

    pub(crate) fn join(
        self,
        other: ExplodedSide,
        predicate: SpatialRelationType,
    ) -> (Vec<u32>, Vec<u32>) {
        // Predicate conversion map: Contains(A,B) = Within(B,A).
        let conversions =
            HashMap::from([(SpatialRelationType::Contains, SpatialRelationType::Within)]);

        let (left, right, predicate, swapped) = match conversions.get(&predicate) {
            Some(converted) => (&other, &self, converted.clone(), true),
            None => (&self, &other, predicate, false),
        };

        let require_left_side = [SpatialRelationType::Within].contains(&predicate);
        let (mut left_indices, mut right_indices) =
            left.join_on_y_stripes(right, predicate.clone(), require_left_side);
        let grouping = left.group_by_component_type(right, &mut left_indices, &mut right_indices);

        let mut out_left: Vec<u32> = Vec::new();
        let mut out_right: Vec<u32> = Vec::new();

        if predicate == SpatialRelationType::Within {
            if let Some(&(start, end)) =
                grouping.get_start_and_end(GeoComponentType::Point, GeoComponentType::Point)
            {
                let (l, r) = point_point_within(
                    left,
                    right,
                    &left_indices[start..end],
                    &right_indices[start..end],
                );
                out_left.extend(l);
                out_right.extend(r);
            }
            if let Some(&(start, end)) = grouping
                .get_start_and_end(GeoComponentType::Point, GeoComponentType::default_line())
            {
                let (l, r) = point_line_within(
                    left,
                    right,
                    &left_indices[start..end],
                    &right_indices[start..end],
                );
                out_left.extend(l);
                out_right.extend(r);
            }
            if let Some(&(start, end)) = grouping.get_start_and_end(
                GeoComponentType::default_line(),
                GeoComponentType::default_line(),
            ) {
                let (l, r) = line_line_within(
                    left,
                    right,
                    &left_indices[start..end],
                    &right_indices[start..end],
                );
                out_left.extend(l);
                out_right.extend(r);
            }
            if let Some(&(start, end)) =
                grouping.get_start_and_end(GeoComponentType::Point, GeoComponentType::EdgeFromPoly)
            {
                let (l, r) = point_poly_within(
                    left,
                    right,
                    &left_indices[start..end],
                    &right_indices[start..end],
                );
                out_left.extend(l);
                out_right.extend(r);
            }
            if let Some(&(start, end)) = grouping.get_start_and_end(
                GeoComponentType::default_line(),
                GeoComponentType::EdgeFromPoly,
            ) {
                let (l, r) = line_poly_within(
                    left,
                    right,
                    &left_indices[start..end],
                    &right_indices[start..end],
                    false,
                );
                out_left.extend(l);
                out_right.extend(r);
            }
            if let Some(&(start, end)) = grouping.get_start_and_end(
                GeoComponentType::EdgeFromPoly,
                GeoComponentType::EdgeFromPoly,
            ) {
                let (l, r) = line_poly_within(
                    left,
                    right,
                    &left_indices[start..end],
                    &right_indices[start..end],
                    true,
                );
                out_left.extend(l);
                out_right.extend(r);
            }
        }

        if swapped {
            (out_right, out_left)
        } else {
            (out_left, out_right)
        }
    }
}

/****************************************************************/
// Helper functions
/****************************************************************/
fn should_drop(
    left_type: &GeoComponentType,
    right_type: &GeoComponentType,
    rel_type: &SpatialRelationType,
) -> bool {
    match rel_type {
        SpatialRelationType::Contains => matches!(
            (left_type, right_type),
            (GeoComponentType::Point, GeoComponentType::Line(_, _))
                | (GeoComponentType::Point, GeoComponentType::EdgeFromPoly)
                | (GeoComponentType::Line(_, _), GeoComponentType::EdgeFromPoly)
        ),
        SpatialRelationType::Within => matches!(
            (left_type, right_type),
            (GeoComponentType::Line(_, _), GeoComponentType::Point)
                | (GeoComponentType::EdgeFromPoly, GeoComponentType::Point)
                | (GeoComponentType::EdgeFromPoly, GeoComponentType::Line(_, _))
        ),
    }
}

/****************************************************************/
// Primitive checks that get combined to check various conditions
/****************************************************************/
fn point_point_equality(ax: &[f64], ay: &[f64], bx: &[f64], by: &[f64]) -> Vec<bool> {
    ax.iter()
        .zip(ay)
        .zip(bx.iter().zip(by))
        .map(|((ax, ay), (bx, by))| (ax - bx).abs() < _EPS && (ay - by).abs() < _EPS)
        .collect()
}

fn point_on_edge(
    px: &[f64],
    py: &[f64],
    sx1: &[f64],
    sy1: &[f64],
    sx2: &[f64],
    sy2: &[f64],
) -> Vec<bool> {
    px.iter()
        .zip(py)
        .zip(sx1.iter().zip(sy1))
        .zip(sx2.iter().zip(sy2))
        .map(|(((&px, &py), (&sx1, &sy1)), (&sx2, &sy2))| {
            let dx = sx2 - sx1;
            let dy = sy2 - sy1;
            let len_sq = dx * dx + dy * dy;

            // Degenerate segment (zero length) → point equality
            if len_sq < _EPS * _EPS {
                return (px - sx1).abs() < _EPS && (py - sy1).abs() < _EPS;
            }

            // Project point onto line: t = dot(P-A, B-A) / |B-A|^2
            let dot = (px - sx1) * dx + (py - sy1) * dy;
            let t = dot / len_sq;
            if !_UNIT_RANGE.contains(&t) {
                return false;
            }

            // Check distance from point to projected point
            let proj_x = sx1 + t * dx;
            let proj_y = sy1 + t * dy;
            (px - proj_x) * (px - proj_x) + (py - proj_y) * (py - proj_y) < _EPS * _EPS
        })
        .collect()
}

// Returns true if a horizontal ray cast in the +x direction from (px, py)
// crosses the edge (sx1,sy1)→(sx2,sy2).
//
// Convention: the lower endpoint is included and the upper endpoint is
// excluded ("lower-open" rule). This ensures each vertex is counted exactly
// once when a ray passes through it, avoiding double-counting.
//
// Horizontal edges (sy1 == sy2) are never counted — the ray is collinear
// with them and the crossing count is undefined; they are harmless to skip.
fn ray_crosses_edge(
    px: &[f64],
    py: &[f64],
    sx1: &[f64],
    sy1: &[f64],
    sx2: &[f64],
    sy2: &[f64],
) -> Vec<bool> {
    px.iter()
        .zip(py)
        .zip(sx1.iter().zip(sy1))
        .zip(sx2.iter().zip(sy2))
        .map(|(((&px, &py), (&sx1, &sy1)), (&sx2, &sy2))| {
            // Check that the edge straddles y=py with the lower-open convention:
            // one endpoint strictly above py, the other at or below py.
            if (sy1 > py) == (sy2 > py) {
                return false;
            }
            // x-coordinate where the edge crosses y=py
            let x_intercept = sx1 + (py - sy1) * (sx2 - sx1) / (sy2 - sy1);
            x_intercept > px
        })
        .collect()
}

// Returns true if the two segments properly cross — i.e., their intersection
// lies strictly in the interior of both (t ∈ (0,1) and u ∈ (0,1)).
// Endpoint touches, T-intersections, and collinear overlaps all return false.
#[allow(clippy::too_many_arguments)]
fn edge_crosses_edge(
    lx1: &[f64],
    ly1: &[f64],
    lx2: &[f64],
    ly2: &[f64],
    rx1: &[f64],
    ry1: &[f64],
    rx2: &[f64],
    ry2: &[f64],
) -> Vec<bool> {
    lx1.iter()
        .zip(ly1)
        .zip(lx2.iter().zip(ly2))
        .zip(rx1.iter().zip(ry1))
        .zip(rx2.iter().zip(ry2))
        .map(|((((lx1, ly1), (lx2, ly2)), (rx1, ry1)), (rx2, ry2))| {
            let (lx1, ly1, lx2, ly2) = (*lx1, *ly1, *lx2, *ly2);
            let (rx1, ry1, rx2, ry2) = (*rx1, *ry1, *rx2, *ry2);
            let dx = lx2 - lx1;
            let dy = ly2 - ly1;
            let ex = rx2 - rx1;
            let ey = ry2 - ry1;
            // 2-D cross product of direction vectors; zero means parallel/collinear.
            let cross = dx * ey - dy * ex;
            if cross.abs() < _EPS {
                return false;
            }
            let fx = rx1 - lx1;
            let fy = ry1 - ly1;
            let t = (fx * ey - fy * ex) / cross;
            let u = (fx * dy - fy * dx) / cross;
            _UNIT_INTERIOR.contains(&t) && _UNIT_INTERIOR.contains(&u)
        })
        .collect()
}

/****************************************************************/
// Predicate implementations, which are expressed in terms of exploded
// geometry components. for a given predicate there are different
// implementations for each type of geometry components (that effectively
// trace back to the actual geometries that were exploded).
/****************************************************************/

fn aggregate_hits(
    left: &ExplodedSide,
    right: &ExplodedSide,
    left_indices: &[usize],
    right_indices: &[usize],
    hits: &[bool],
) -> (Vec<u32>, Vec<u32>) {
    use std::collections::{HashMap, HashSet};

    let mut total: HashMap<(u32, u32), HashSet<usize>> = HashMap::new();
    let mut covered: HashMap<(u32, u32), HashSet<usize>> = HashMap::new();

    for (i, &hit) in hits.iter().enumerate() {
        let li = left_indices[i];
        let ri = right_indices[i];
        let key = (left.geo_index[li], right.geo_index[ri]);
        total.entry(key).or_default().insert(li);
        if hit {
            covered.entry(key).or_default().insert(li);
        }
    }

    total
        .into_iter()
        .filter(|(key, all)| covered.get(key).is_some_and(|cov| cov.len() == all.len()))
        .map(|(key, _)| key)
        .unzip()
}

fn point_point_within(
    left: &ExplodedSide,
    right: &ExplodedSide,
    left_indices: &[usize],
    right_indices: &[usize],
) -> (Vec<u32>, Vec<u32>) {
    let lx: Vec<f64> = left_indices.iter().map(|&i| left.x1[i]).collect();
    let ly: Vec<f64> = left_indices.iter().map(|&i| left.y1[i]).collect();
    let rx: Vec<f64> = right_indices.iter().map(|&i| right.x1[i]).collect();
    let ry: Vec<f64> = right_indices.iter().map(|&i| right.y1[i]).collect();
    let eq = point_point_equality(&lx, &ly, &rx, &ry);

    aggregate_hits(left, right, left_indices, right_indices, &eq)
}

fn point_line_within(
    left: &ExplodedSide,
    right: &ExplodedSide,
    left_indices: &[usize],
    right_indices: &[usize],
) -> (Vec<u32>, Vec<u32>) {
    let px: Vec<f64> = left_indices.iter().map(|&i| left.x1[i]).collect();
    let py: Vec<f64> = left_indices.iter().map(|&i| left.y1[i]).collect();
    let sx1: Vec<f64> = right_indices.iter().map(|&i| right.x1[i]).collect();
    let sy1: Vec<f64> = right_indices.iter().map(|&i| right.y1[i]).collect();
    let sx2: Vec<f64> = right_indices.iter().map(|&i| right.x2[i]).collect();
    let sy2: Vec<f64> = right_indices.iter().map(|&i| right.y2[i]).collect();
    let mut on_edge = point_on_edge(&px, &py, &sx1, &sy1, &sx2, &sy2);

    // Exclude hits at linestring boundary endpoints.
    for (i, hit) in on_edge.iter_mut().enumerate() {
        if !*hit {
            continue;
        }
        let ri = right_indices[i];
        if let GeoComponentType::Line(left_boundary, right_boundary) = right.component_type[ri] {
            let at_left =
                left_boundary && (px[i] - sx1[i]).abs() < _EPS && (py[i] - sy1[i]).abs() < _EPS;
            let at_right =
                right_boundary && (px[i] - sx2[i]).abs() < _EPS && (py[i] - sy2[i]).abs() < _EPS;
            if at_left || at_right {
                *hit = false;
            }
        }
    }

    aggregate_hits(left, right, left_indices, right_indices, &on_edge)
}

// Two-level aggregation for point-in-polygon via ray casting.
//
// Inner level: for each left primitive li (individual point), count how many
// right primitives ri (polygon edges) produce a ray crossing. li is inside the
// polygon iff the count is odd.
//
// Outer level: for each (left_geo_id, right_geo_id) pair, all left primitives
// must be inside. Relies on require_all_left=true in join_on_y_stripes so that
// every li for a given geo pair is guaranteed to appear in left_indices.
fn aggregate_odd_crossings(
    left: &ExplodedSide,
    right: &ExplodedSide,
    left_indices: &[usize],
    right_indices: &[usize],
    hits: &[bool],
) -> (Vec<u32>, Vec<u32>) {
    use std::collections::HashMap;

    let mut crossing_counts: HashMap<(u32, u32), HashMap<usize, usize>> = HashMap::new();
    for (i, &hit) in hits.iter().enumerate() {
        let li = left_indices[i];
        let ri = right_indices[i];
        let key = (left.geo_index[li], right.geo_index[ri]);
        let count = crossing_counts
            .entry(key)
            .or_default()
            .entry(li)
            .or_insert(0);
        if hit {
            *count += 1;
        }
    }

    crossing_counts
        .into_iter()
        .filter(|(_, li_counts)| li_counts.values().all(|&c| c % 2 == 1))
        .map(|(key, _)| key)
        .unzip()
}

fn line_line_within(
    left: &ExplodedSide,
    right: &ExplodedSide,
    left_indices: &[usize],
    right_indices: &[usize],
) -> (Vec<u32>, Vec<u32>) {
    let lx1: Vec<f64> = left_indices.iter().map(|&i| left.x1[i]).collect();
    let ly1: Vec<f64> = left_indices.iter().map(|&i| left.y1[i]).collect();
    let lx2: Vec<f64> = left_indices.iter().map(|&i| left.x2[i]).collect();
    let ly2: Vec<f64> = left_indices.iter().map(|&i| left.y2[i]).collect();
    let rx1: Vec<f64> = right_indices.iter().map(|&i| right.x1[i]).collect();
    let ry1: Vec<f64> = right_indices.iter().map(|&i| right.y1[i]).collect();
    let rx2: Vec<f64> = right_indices.iter().map(|&i| right.x2[i]).collect();
    let ry2: Vec<f64> = right_indices.iter().map(|&i| right.y2[i]).collect();

    let e1 = point_on_edge(&lx1, &ly1, &rx1, &ry1, &rx2, &ry2);
    let e2 = point_on_edge(&lx2, &ly2, &rx1, &ry1, &rx2, &ry2);
    let hits: Vec<bool> = e1.into_iter().zip(e2).map(|(a, b)| a && b).collect();

    aggregate_hits(left, right, left_indices, right_indices, &hits)
}

fn point_poly_within(
    left: &ExplodedSide,
    right: &ExplodedSide,
    left_indices: &[usize],
    right_indices: &[usize],
) -> (Vec<u32>, Vec<u32>) {
    use std::collections::HashSet;

    let px: Vec<f64> = left_indices.iter().map(|&i| left.x1[i]).collect();
    let py: Vec<f64> = left_indices.iter().map(|&i| left.y1[i]).collect();
    let sx1: Vec<f64> = right_indices.iter().map(|&i| right.x1[i]).collect();
    let sy1: Vec<f64> = right_indices.iter().map(|&i| right.y1[i]).collect();
    let sx2: Vec<f64> = right_indices.iter().map(|&i| right.x2[i]).collect();
    let sy2: Vec<f64> = right_indices.iter().map(|&i| right.y2[i]).collect();

    // Inner aggregation: odd crossing count per point → inside the polygon.
    let crossings = ray_crosses_edge(&px, &py, &sx1, &sy1, &sx2, &sy2);
    let (inside_left, inside_right) =
        aggregate_odd_crossings(left, right, left_indices, right_indices, &crossings);
    let inside_set: HashSet<(u32, u32)> = inside_left.into_iter().zip(inside_right).collect();

    // Any point on any polygon edge → on the boundary (not within the interior).
    let on_edge = point_on_edge(&px, &py, &sx1, &sy1, &sx2, &sy2);
    let boundary_set: HashSet<(u32, u32)> = on_edge
        .iter()
        .enumerate()
        .filter(|(_, &h)| h)
        .map(|(i, _)| {
            (
                left.geo_index[left_indices[i]],
                right.geo_index[right_indices[i]],
            )
        })
        .collect();

    inside_set
        .into_iter()
        .filter(|key| !boundary_set.contains(key))
        .unzip()
}

// A line segment is within a polygon iff (DE-9IM):
//   1. Interior(A) ∩ Interior(B) ≠ ∅  — at least one segment's interior is strictly inside
//   2. Interior(A) ∩ Exterior(B) = ∅  — no segment's interior is outside
//   3. Boundary(A) ∩ Exterior(B) = ∅  — no endpoint is outside (implied by 1+2+no-crossing)
//
// Strategy: for each left segment, compute its midpoint and classify it:
//   - Odd ray crossings + not on any polygon edge → strictly inside  (qualifying)
//   - Even ray crossings + not on any polygon edge → strictly outside (disqualifying)
//   - On a polygon edge → boundary (neither qualifying nor disqualifying)
// Additionally, any proper crossing of a left segment with a polygon edge disqualifies the pair.
//
// Per (left_geo, right_geo): result iff has_inside && !has_outside && !has_crossing.
// When `emit_all_boundary_as_within` is true, pairs where every left segment midpoint falls on
// the right polygon boundary (has_inside=false, has_outside=false) are also emitted. This handles
// the identical-polygon case (EdgeFromPoly × EdgeFromPoly) where all edge midpoints land on the
// shared boundary — a valid DE-9IM within. For Line × EdgeFromPoly the flag should be false,
// since a line on the boundary is correctly not within.
fn line_poly_within(
    left: &ExplodedSide,
    right: &ExplodedSide,
    left_indices: &[usize],
    right_indices: &[usize],
    emit_all_boundary_as_within: bool,
) -> (Vec<u32>, Vec<u32>) {
    use std::collections::{HashMap, HashSet};

    let mx: Vec<f64> = left_indices
        .iter()
        .map(|&i| (left.x1[i] + left.x2[i]) / 2.0)
        .collect();
    let my: Vec<f64> = left_indices
        .iter()
        .map(|&i| (left.y1[i] + left.y2[i]) / 2.0)
        .collect();
    let lx1: Vec<f64> = left_indices.iter().map(|&i| left.x1[i]).collect();
    let ly1: Vec<f64> = left_indices.iter().map(|&i| left.y1[i]).collect();
    let lx2: Vec<f64> = left_indices.iter().map(|&i| left.x2[i]).collect();
    let ly2: Vec<f64> = left_indices.iter().map(|&i| left.y2[i]).collect();
    let rx1: Vec<f64> = right_indices.iter().map(|&i| right.x1[i]).collect();
    let ry1: Vec<f64> = right_indices.iter().map(|&i| right.y1[i]).collect();
    let rx2: Vec<f64> = right_indices.iter().map(|&i| right.x2[i]).collect();
    let ry2: Vec<f64> = right_indices.iter().map(|&i| right.y2[i]).collect();

    let mid_cross = ray_crosses_edge(&mx, &my, &rx1, &ry1, &rx2, &ry2);
    let mid_on_bnd = point_on_edge(&mx, &my, &rx1, &ry1, &rx2, &ry2);
    let seg_cross = edge_crosses_edge(&lx1, &ly1, &lx2, &ly2, &rx1, &ry1, &rx2, &ry2);

    // crossing_counts[key][li] = ray-crossing count for the midpoint of segment li.
    let mut crossing_counts: HashMap<(u32, u32), HashMap<usize, usize>> = HashMap::new();
    // on_boundary_segs[key]: li values whose midpoints lie on a polygon edge.
    let mut on_boundary_segs: HashMap<(u32, u32), HashSet<usize>> = HashMap::new();
    // crossing_pairs: (left_geo, right_geo) pairs where any segment properly crosses a poly edge.
    let mut crossing_pairs: HashSet<(u32, u32)> = HashSet::new();

    for i in 0..left_indices.len() {
        let li = left_indices[i];
        let ri = right_indices[i];
        let key = (left.geo_index[li], right.geo_index[ri]);

        let cnt = crossing_counts
            .entry(key)
            .or_default()
            .entry(li)
            .or_insert(0);
        if mid_cross[i] {
            *cnt += 1;
        }
        if mid_on_bnd[i] {
            on_boundary_segs.entry(key).or_default().insert(li);
        }
        if seg_cross[i] {
            crossing_pairs.insert(key);
        }
    }

    let mut out_left = Vec::new();
    let mut out_right = Vec::new();
    for (key, counts) in &crossing_counts {
        if crossing_pairs.contains(key) {
            continue;
        }
        let on_bnd = on_boundary_segs.get(key);
        let mut has_inside = false;
        let mut has_outside = false;
        for (&li, &cnt) in counts {
            let on_b = on_bnd.is_some_and(|s| s.contains(&li));
            if !on_b {
                if cnt % 2 == 1 {
                    has_inside = true;
                } else {
                    has_outside = true;
                }
            }
        }
        if (has_inside || emit_all_boundary_as_within) && !has_outside {
            out_left.push(key.0);
            out_right.push(key.1);
        }
    }
    (out_left, out_right)
}

#[cfg(test)]
mod test_helpers {
    use wkb::reader::Wkb;

    use super::*;

    pub(super) fn wkt_to_wkb(wkt_str: &str) -> Vec<u8> {
        use geo_types::Geometry;
        use wkb::writer::{
            write_line_string, write_multi_line_string, write_multi_point, write_multi_polygon,
            write_point, write_polygon, WriteOptions,
        };
        use wkt::TryFromWkt;
        let geom = Geometry::<f64>::try_from_wkt_str(wkt_str).unwrap();
        let opts = WriteOptions::default();
        let mut buf = Vec::new();
        match geom {
            Geometry::Point(g) => write_point(&mut buf, &g, &opts).unwrap(),
            Geometry::LineString(g) => write_line_string(&mut buf, &g, &opts).unwrap(),
            Geometry::Polygon(g) => write_polygon(&mut buf, &g, &opts).unwrap(),
            Geometry::MultiPoint(g) => write_multi_point(&mut buf, &g, &opts).unwrap(),
            Geometry::MultiLineString(g) => write_multi_line_string(&mut buf, &g, &opts).unwrap(),
            Geometry::MultiPolygon(g) => write_multi_polygon(&mut buf, &g, &opts).unwrap(),
            _ => panic!("unsupported geometry: {wkt_str}"),
        }
        buf
    }

    pub(super) fn spatial_join(
        left_wkts: &[&str],
        right_wkts: &[&str],
        pred: SpatialRelationType,
    ) -> (Vec<u32>, Vec<u32>) {
        let left_bufs: Vec<Vec<u8>> = left_wkts.iter().map(|s| wkt_to_wkb(s)).collect();
        let right_bufs: Vec<Vec<u8>> = right_wkts.iter().map(|s| wkt_to_wkb(s)).collect();
        let left_wkbs: Vec<_> = left_bufs.iter().map(|b| Wkb::try_new(b).unwrap()).collect();
        let right_wkbs: Vec<_> = right_bufs
            .iter()
            .map(|b| Wkb::try_new(b).unwrap())
            .collect();
        let left_ids: Vec<u32> = (0..left_wkts.len() as u32).collect();
        let right_ids: Vec<u32> = (0..right_wkts.len() as u32).collect();
        let left_side = ExplodedSide::new(&left_ids, &left_wkbs).unwrap();
        let right_side = ExplodedSide::new(&right_ids, &right_wkbs).unwrap();
        let (out_left, out_right) = left_side.join(right_side, pred);

        // Sort by (left, right) pairs for deterministic comparison.
        let mut pairs: Vec<_> = out_left.iter().zip(out_right.iter()).collect();
        pairs.sort();
        let out_left = pairs.iter().map(|(&l, _)| l).collect();
        let out_right = pairs.iter().map(|(_, &r)| r).collect();

        (out_left, out_right)
    }
}

#[cfg(test)]
mod within_tests {
    use super::test_helpers::spatial_join;
    use super::SpatialRelationType;

    #[test]
    fn test_within_degenerate_y_coords() {
        let left = &[
            "POINT(1 5)",
            "MULTIPOINT((2 5), (3 5))",
            "MULTIPOINT((4 5), (6 5))",
        ];
        let right = &["POINT(1 5)", "MULTIPOINT((2 5), (3 5))"];

        let (out_left, out_right) = spatial_join(left, right, SpatialRelationType::Within);
        assert_eq!(out_left, vec![0, 1]);
        assert_eq!(out_right, vec![0, 1]);
    }

    #[test]
    fn test_within_points_and_multipoints() {
        let left = &[
            "POINT(1 2)",
            "POINT(3 4)",
            "MULTIPOINT((5 6), (7 8))",
            "MULTIPOINT((9 10), (50 50))",
        ];
        let right = &[
            "POINT(1 2)",
            "POINT(99 99)",
            "MULTIPOINT((5 6), (7 8))",
            "MULTIPOINT((3 4), (11 12))",
        ];

        let (out_left, out_right) = spatial_join(left, right, SpatialRelationType::Within);
        assert_eq!(out_left, vec![0, 1, 2]);
        assert_eq!(out_right, vec![0, 3, 2]);
    }

    #[test]
    fn test_within_point_and_multipoint_in_linestrings() {
        let left = &["POINT(1 1)", "MULTIPOINT((3 3), (4 4))"];
        let right = &[
            "LINESTRING(0 0, 2 2)",            // 0: point (1,1) interior → match left 0
            "LINESTRING(2 2, 5 5)",            // 1: (3,3),(4,4) interior → match left 1
            "MULTILINESTRING((0 0, 2 2))",     // 2: same as 0 → match left 0
            "MULTILINESTRING((2 2, 5 5))",     // 3: same as 1 → match left 1
            "LINESTRING(3 3, 6 0)",            // 4: (3,3) endpoint, (1,1) not on it → no match
            "MULTILINESTRING((10 10, 12 12))", // 5: nothing on it → no match
        ];

        let (out_left, out_right) = spatial_join(left, right, SpatialRelationType::Within);
        assert_eq!(out_left, vec![0, 0, 1, 1]);
        assert_eq!(out_right, vec![0, 2, 1, 3]);
    }

    #[test]
    fn test_within_point_at_linestring_boundary_positions() {
        let left = &["POINT(0 0)"];
        let right = &[
            "LINESTRING(0 0, 1 1)",                      // 0: endpoint → not within
            "LINESTRING(-1 -1, 0 0, 1 0)",               // 1: connecting vertex → within
            "LINESTRING(0 0, 1 0, 1 1, 0 0)",            // 2: closed, closure point → within
            "MULTILINESTRING((-1 0, 0 0))",              // 3: endpoint → not within
            "MULTILINESTRING((-1 -1, 0 0), (0 0, 1 0))", // 4: connecting vertex → within
            "MULTILINESTRING((0 0, 1 0), (1 0, 1 1), (1 1, 0 0))", // 5: closed, closure point → within
            "MULTILINESTRING((-1 0, 0 0), (0 0, 1 0), (0 0, 0 1))", // 6: 3-way junction, (0,0) endpoint (odd count) → not within
        ];

        let (out_left, out_right) = spatial_join(left, right, SpatialRelationType::Within);
        assert_eq!(out_left, vec![0, 0, 0, 0]);
        assert_eq!(out_right, vec![1, 2, 4, 5]);
    }

    #[test]
    fn test_within_multipoint_at_linestring_boundary_positions() {
        let left = &["MULTIPOINT((0 0), (1 0))"];
        let right = &[
            "LINESTRING(0 0, 1 0)",                      // 0: both on endpoints → no match
            "LINESTRING(-1 -1, 0 0, 1 0)", // 1: (0,0) interior, (1,0) endpoint → no match
            "LINESTRING(0 0, 1 0, 1 1, 0 0)", // 2: closed, both interior → match
            "MULTILINESTRING((-1 0, 0 0), (0 0, 1 0))", // 3: (0,0) interior, (1,0) endpoint → no match
            "MULTILINESTRING((-1 -1, 0 0), (0 0, 1 0))", // 4: (0,0) interior, (1,0) endpoint → no match
            "MULTILINESTRING((0 0, 1 0), (1 0, 1 1), (1 1, 0 0))", // 5: closed, both interior → match
        ];

        let (out_left, out_right) = spatial_join(left, right, SpatialRelationType::Within);
        assert_eq!(out_left, vec![0, 0]);
        assert_eq!(out_right, vec![2, 5]);
    }

    #[test]
    fn test_within_line_in_collinear_lines() {
        let left = &["LINESTRING(2 2, 4 4)"];
        let right = &[
            "LINESTRING(5 5, 7 7)", // 0: collinear but shifted, no overlap → no match
            "LINESTRING(2 2, 4 4)", // 1: exact match → match
            "LINESTRING(1 1, 5 5)", // 2: extended, fully contains left → match
            "LINESTRING(3 3, 6 6)", // 3: collinear, partial overlap → no match
        ];

        let (out_left, out_right) = spatial_join(left, right, SpatialRelationType::Within);
        assert_eq!(out_left, vec![0, 0]);
        assert_eq!(out_right, vec![1, 2]);
    }

    #[test]
    fn test_within_line_in_polygons() {
        let left = &["LINESTRING(2 2, 4 2)", "MULTILINESTRING((2 2, 4 2))"];
        let right = &[
            "POLYGON((2.5 1, 6 1, 6 3, 2.5 3, 2.5 1))", // 0: partial overlap → no match
            "POLYGON((10 10, 12 10, 12 12, 10 12, 10 10))", // 1: far away → no match
            "POLYGON((0 0, 8 0, 8 8, 0 8, 0 0), (1 1, 5 1, 5 3, 1 3, 1 1))", // 2: line in hole → no match
            "POLYGON((0 0, 8 0, 8 8, 0 8, 0 0), (3 1, 5 1, 5 3, 3 3, 3 1))", // 3: line crosses hole boundary → no match
            "POLYGON((0 0, 8 0, 8 8, 0 8, 0 0), (1 4, 5 4, 5 7, 1 7, 1 4))", // 4: hole above line, line in body → match
            "POLYGON((0 0, 6 0, 6 4, 0 4, 0 0))", // 5: normal polygon containing line → match
        ];

        let (out_left, out_right) = spatial_join(left, right, SpatialRelationType::Within);
        assert_eq!(out_left, vec![0, 0, 1, 1]);
        assert_eq!(out_right, vec![4, 5, 4, 5]);
    }

    #[test]
    fn test_within_lines_in_multipolygons() {
        let left = &["LINESTRING(2 2, 4 2)", "LINESTRING(7 7, 9 7)"];
        let right = &[
            // 0: poly A has hole containing line 0, poly B contains line 1
            "MULTIPOLYGON(((0 0, 10 0, 10 5, 0 5, 0 0), (1 1, 5 1, 5 3, 1 3, 1 1)), ((5 5, 10 5, 10 10, 5 10, 5 5)))",
            // 1: poly A (no hole) contains line 0, poly B has hole that line 1 partially crosses
            "MULTIPOLYGON(((0 0, 10 0, 10 5, 0 5, 0 0)), ((5 5, 10 5, 10 10, 5 10, 5 5), (6 6, 8 6, 8 8, 6 8, 6 6)))",
        ];

        let (out_left, out_right) = spatial_join(left, right, SpatialRelationType::Within);
        assert_eq!(out_left, vec![0, 1]);
        assert_eq!(out_right, vec![1, 0]);
    }

    #[test]
    fn test_within_line_on_polygon_boundary() {
        let left = &[
            "LINESTRING(0 0, 4 0, 4 4, 0 4, 0 0)",
            "MULTILINESTRING((10 10, 14 10, 14 14, 10 14, 10 10), (11 11, 13 11, 13 13, 11 13, 11 11))",
            "LINESTRING(1 0, 3 0)"
        ];
        let right = &[
            "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))",
            "POLYGON((10 10, 14 10, 14 14, 10 14, 10 10), (11 11, 13 11, 13 13, 11 13, 11 11))",
        ];

        let (out_left, out_right) = spatial_join(left, right, SpatialRelationType::Within);
        assert!(out_left.is_empty());
        assert!(out_right.is_empty());
    }

    #[test]
    fn test_within_multiline_in_multipolygons() {
        let left = &["MULTILINESTRING((2 2, 4 2), (7 7, 9 7))"];
        let right = &[
            // 0: first poly contains one line, second poly far away → no match
            "MULTIPOLYGON(((0 0, 5 0, 5 5, 0 5, 0 0)), ((20 20, 25 20, 25 25, 20 25, 20 20)))",
            // 1: first poly contains one line, second poly contains the other → match
            "MULTIPOLYGON(((0 0, 5 0, 5 5, 0 5, 0 0)), ((6 6, 10 6, 10 10, 6 10, 6 6)))",
        ];

        let (out_left, out_right) = spatial_join(left, right, SpatialRelationType::Within);
        assert_eq!(out_left, vec![0]);
        assert_eq!(out_right, vec![1]);
    }

    #[test]
    fn test_within_point_in_polygons() {
        let left = &["POINT(5 5)"];
        let right = &[
            // 0: no hole, point inside → match
            "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))",
            // 1: no hole, point far away → no match
            "POLYGON((20 20, 30 20, 30 30, 20 30, 20 20))",
            // 2: hole, point on outer edge at y=5 → no match
            "POLYGON((0 0, 10 0, 10 5, 0 5, 0 0), (1 1, 9 1, 9 4, 1 4, 1 1))",
            // 3: hole, point on hole edge at y=5 → no match
            "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (3 5, 7 5, 7 8, 3 8, 3 5))",
            // 4: hole, point on hole corner vertex (5,5) → no match
            "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (5 5, 8 5, 8 8, 5 8, 5 5))",
            // 5: hole, point inside hole → no match
            "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 7 3, 7 7, 3 7, 3 3))",
            // 6: hole elsewhere, point in polygon body → match
            "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1))",
            // 7: no hole, point on edge (not vertex) → no match
            "POLYGON((5 0, 10 0, 10 10, 5 10, 5 0))",
        ];

        let (out_left, out_right) = spatial_join(left, right, SpatialRelationType::Within);
        assert_eq!(out_left, vec![0, 0]);
        assert_eq!(out_right, vec![0, 6]);
    }

    #[test]
    fn test_within_multipoint_in_polygons_and_multipolygons() {
        let left = &["MULTIPOINT((2 2), (5 5), (8 8), (15 15))"];
        let right = &[
            // Polygons: first 3 points inside, 4th varies
            // 0: (15,15) outside → no match
            "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))",
            // 1: (15,15) on corner → no match
            "POLYGON((0 0, 15 0, 15 15, 0 15, 0 0))",
            // 2: (15,15) inside → match
            "POLYGON((0 0, 20 0, 20 20, 0 20, 0 0))",
            // Multipolygons: first poly has (2,2),(5,5),(8,8), second poly varies for (15,15)
            // 3: (15,15) outside second poly → no match
            "MULTIPOLYGON(((0 0, 10 0, 10 10, 0 10, 0 0)), ((12 12, 14 12, 14 14, 12 14, 12 12)))",
            // 4: (15,15) on corner of second poly → no match
            "MULTIPOLYGON(((0 0, 10 0, 10 10, 0 10, 0 0)), ((12 12, 15 12, 15 15, 12 15, 12 12)))",
            // 5: (15,15) inside second poly → match
            "MULTIPOLYGON(((0 0, 10 0, 10 10, 0 10, 0 0)), ((12 12, 18 12, 18 18, 12 18, 12 12)))",
        ];

        let (out_left, out_right) = spatial_join(left, right, SpatialRelationType::Within);
        assert_eq!(out_left, vec![0, 0]);
        assert_eq!(out_right, vec![2, 5]);
    }

    #[test]
    fn test_within_poly_in_polys() {
        let left = &[
            "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))",
            "POLYGON((9 4, 12 4, 12 6, 9 6, 9 4))",
        ];
        let right = &[
            // 0: identical to left 0 → match
            "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))",
            // 1: contains left 1; left 0 overlaps but extends beyond → only left 1 matches
            "POLYGON((8 3, 15 3, 15 7, 8 7, 8 3))",
        ];

        let (out_left, out_right) = spatial_join(left, right, SpatialRelationType::Within);
        assert_eq!(out_left, vec![0, 1]);
        assert_eq!(out_right, vec![0, 1]);
    }

    #[test]
    fn test_within_multipoly_in_multipolys() {
        let left = &[
            "MULTIPOLYGON(((0 0, 4 0, 4 4, 0 4, 0 0)), ((10 10, 14 10, 14 14, 10 14, 10 10)))",
            "MULTIPOLYGON(((22 22, 23 22, 23 23, 22 23, 22 22)), ((32 32, 33 32, 33 33, 32 33, 32 32)))",
        ];
        let right = &[
            // 0: identical to left 0 → match
            "MULTIPOLYGON(((0 0, 4 0, 4 4, 0 4, 0 0)), ((10 10, 14 10, 14 14, 10 14, 10 10)))",
            // 1: first poly contains left 1's first poly, second poly far away → no match
            "MULTIPOLYGON(((20 20, 25 20, 25 25, 20 25, 20 20)), ((40 40, 45 40, 45 45, 40 45, 40 40)))",
            // 2: first poly contains left 1's first poly, second poly partially intersects left 1's second → no match
            "MULTIPOLYGON(((20 20, 25 20, 25 25, 20 25, 20 20)), ((32.5 31, 35 31, 35 34, 32.5 34, 32.5 31)))",
            // 3: both polys contain left 1's polys → match
            "MULTIPOLYGON(((20 20, 25 20, 25 25, 20 25, 20 20)), ((30 30, 35 30, 35 35, 30 35, 30 30)))",
        ];

        let (out_left, out_right) = spatial_join(left, right, SpatialRelationType::Within);
        assert_eq!(out_left, vec![0, 1]);
        assert_eq!(out_right, vec![0, 3]);
    }

    #[test]
    fn test_within_line_and_poly_cannot_be_within_point() {
        let left = &["LINESTRING(0 0, 2 2)", "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))"];
        let right = &[
            "POINT(1 1)", // on the line and inside the poly
            "POINT(2 2)", // on the line and inside the poly
        ];

        let (out_left, out_right) = spatial_join(left, right, SpatialRelationType::Within);
        assert!(out_left.is_empty());
        assert!(out_right.is_empty());
    }

    #[test]
    fn test_within_and_contains_symmetry() {
        let a = &[
            "POINT(1 1)",
            "LINESTRING(12 12, 14 14)",
            "POLYGON((30 30, 34 30, 34 34, 30 34, 30 30))",
            "POLYGON((52 52, 53 52, 53 53, 52 53, 52 52))",
        ];
        let b = &[
            "LINESTRING(0 0, 3 3)",                         // contains a[0]
            "POLYGON((10 10, 20 10, 20 20, 10 20, 10 10))", // contains a[1]
            "POLYGON((40 40, 44 40, 44 44, 40 44, 40 40))", // a[2] outside
            "POLYGON((50 50, 55 50, 55 55, 50 55, 50 50))", // contains a[3]
        ];

        // a Within b
        let (within_left, within_right) = spatial_join(a, b, SpatialRelationType::Within);
        assert_eq!(within_left, vec![0, 1, 3]);
        assert_eq!(within_right, vec![0, 1, 3]);

        // b Contains a → flip sides, same result
        let (contains_left, contains_right) = spatial_join(b, a, SpatialRelationType::Contains);
        assert_eq!(contains_left, within_right);
        assert_eq!(contains_right, within_left);
    }
}
