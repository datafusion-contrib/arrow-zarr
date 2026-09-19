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

use std::sync::Arc;

use arrow_array::builder::Float64Builder;
use arrow_schema::{DataType, Field, FieldRef};
use datafusion::common::{exec_datafusion_err, Result};
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature,
    Volatility,
};
use geo_traits::{
    CoordTrait, GeometryTrait, GeometryType, LineStringTrait, MultiPolygonTrait, PolygonTrait,
};
use wkb::reader::Wkb;

use super::spatial_predicate_udfs::wkb_at;
use super::st_geomfrom::require_geometry;

// *************************************************************
// area(geo) -> the planar area of a geometry. Only polygons have area; points
// and lines are 0. A polygon's area is its shell area minus the area of each
// hole. Ring role (shell vs hole) comes from the WKB structure, not winding, so
// each ring is measured by the absolute value of its shoelace sum.
// *************************************************************
fn st_area(wkb: &Wkb) -> f64 {
    match wkb.as_type() {
        GeometryType::Polygon(poly) => polygon_area(&poly),
        GeometryType::MultiPolygon(mp) => mp.polygons().map(|p| polygon_area(&p)).sum(),
        _ => 0.0,
    }
}

fn polygon_area(poly: &impl PolygonTrait<T = f64>) -> f64 {
    let shell = poly.exterior().map(|r| ring_area(&r)).unwrap_or(0.0);
    let holes: f64 = poly.interiors().map(|h| ring_area(&h)).sum();
    shell - holes
}

// Absolute area of a single ring via the shoelace formula. Coordinates are
// shifted by the first vertex before the cross products to keep precision when
// the ring sits far from the origin. A WKB ring is closed (last == first), so
// summing consecutive pairs in order already covers the wraparound term.
fn ring_area(ring: &impl LineStringTrait<T = f64>) -> f64 {
    let mut coords = ring.coords();
    let Some(origin) = coords.next() else {
        return 0.0;
    };
    let (ox, oy) = (origin.x(), origin.y());

    let mut sum = 0.0;
    let (mut px, mut py) = (0.0, 0.0);
    for c in coords {
        let (cx, cy) = (c.x() - ox, c.y() - oy);
        sum += px * cy - cx * py;
        (px, py) = (cx, cy);
    }
    (sum / 2.0).abs()
}

/// `st_area(geom)` returns the planar area of a geometry as a `Float64`,
/// row-wise over a WKB column.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StAreaUdf {
    signature: Signature,
}

impl Default for StAreaUdf {
    fn default() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![DataType::Binary]),
                    TypeSignature::Exact(vec![DataType::BinaryView]),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for StAreaUdf {
    fn name(&self) -> &str {
        "st_area"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    // Reject bare binary: the operand must be tagged geometry.
    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        require_geometry(args.arg_fields[0].as_ref())?;
        Ok(Arc::new(Field::new(self.name(), DataType::Float64, true)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        let geoms = &arrays[0];
        let n = geoms.len();

        let mut builder = Float64Builder::with_capacity(n);
        for i in 0..n {
            match wkb_at(geoms, i) {
                Some(bytes) => {
                    let wkb = Wkb::try_new(bytes)
                        .map_err(|e| exec_datafusion_err!("st_area: malformed WKB: {e}"))?;
                    builder.append_value(st_area(&wkb));
                }
                None => builder.append_null(),
            }
        }
        Ok(ColumnarValue::Array(Arc::new(builder.finish())))
    }
}

#[cfg(test)]
mod test_helpers {
    use geos::Geom;

    pub(super) use super::super::super::test_utils::wkt_to_wkb;
    use super::{st_area, Wkb};

    // Checks st_area against GEOS's area.
    pub(super) fn compare_area(label: &str, wkt: &str) {
        let bytes = wkt_to_wkb(wkt);

        let geos_result = geos::Geometry::new_from_wkb(&bytes)
            .unwrap()
            .area()
            .unwrap();
        let ours = st_area(&Wkb::try_new(&bytes).unwrap());

        assert!((geos_result - ours).abs() < 1e-6, "{label}");
    }
}

#[cfg(test)]
mod area_tests {
    use super::test_helpers::compare_area;

    // Vertices of a regular decagon of radius 10 centered at the origin.
    const DECAGON: [(f64, f64); 10] = [
        (10.0, 0.0),
        (8.0902, 5.8779),
        (3.0902, 9.5106),
        (-3.0902, 9.5106),
        (-8.0902, 5.8779),
        (-10.0, 0.0),
        (-8.0902, -5.8779),
        (-3.0902, -9.5106),
        (3.0902, -9.5106),
        (8.0902, -5.8779),
    ];

    // A decagon ring (closed) as a WKT coordinate list, shifted by (dx, dy).
    fn decagon_ring(dx: f64, dy: f64) -> String {
        DECAGON
            .iter()
            .chain(std::iter::once(&DECAGON[0]))
            .map(|(x, y)| format!("{} {}", x + dx, y + dy))
            .collect::<Vec<_>>()
            .join(", ")
    }

    #[test]
    fn decagon() {
        compare_area(
            "decagon",
            &format!("POLYGON (({}))", decagon_ring(0.0, 0.0)),
        );
    }

    #[test]
    fn multipolygon() {
        compare_area(
            "two non-overlapping decagons",
            &format!(
                "MULTIPOLYGON ((({})), (({})))",
                decagon_ring(0.0, 0.0),
                decagon_ring(100.0, 0.0),
            ),
        );
    }

    #[test]
    fn decagon_with_hole() {
        compare_area(
            "decagon with square hole",
            &format!(
                "POLYGON (({}), (-2 -2, 2 -2, 2 2, -2 2, -2 -2))",
                decagon_ring(0.0, 0.0),
            ),
        );
    }
}
