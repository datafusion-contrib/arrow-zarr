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

use std::io::Cursor;
use std::sync::Arc;

use arrow_array::builder::BinaryBuilder;
use arrow_array::Array;
use arrow_schema::DataType;
use datafusion::common::cast::as_float64_array;
use datafusion::common::{not_impl_err, Result};
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature, Volatility,
};
use geo_types::Point;
use wkb::writer::{write_point, WriteOptions};

// For now th stubs only exist so that the query optimizer has something
// to reference before it converts the nested join loop to a spatial join.
// Obviously a TODO to actually implement this.
macro_rules! spatial_predicate_stub {
    ($($struct_name:ident => $udf_name:literal),* $(,)?) => {
        $(
            #[derive(Debug, PartialEq, Eq, Hash)]
            pub struct $struct_name {
                signature: Signature,
            }

            impl Default for $struct_name {
                fn default() -> Self {
                    Self {
                        signature: Signature::one_of(
                            vec![
                                TypeSignature::Exact(vec![DataType::Binary, DataType::Binary]),
                                TypeSignature::Exact(vec![DataType::BinaryView, DataType::BinaryView]),
                            ],
                            Volatility::Immutable,
                        ),
                    }
                }
            }

            impl ScalarUDFImpl for $struct_name {
                fn name(&self) -> &str { $udf_name }
                fn signature(&self) -> &Signature {
                    &self.signature
                }
                fn return_type(&self, _: &[DataType]) -> Result<DataType> {
                    Ok(DataType::Boolean)
                }
                fn invoke_with_args(&self, _: ScalarFunctionArgs) -> Result<ColumnarValue> {
                    not_impl_err!("{} is only supported as a join condition", $udf_name)
                }
            }
        )*
    }
}

spatial_predicate_stub! {
StWithinUdf => "st_within",
StContainsUdf => "st_contains",
StIntersectsUdf => "st_intersects",}

/// Byte length of a 2D WKB point: 1 (byte order) + 4 (geometry type) + 2 * 8 (x, y).
const WKB_POINT_2D_LEN: usize = 21;

/// `st_point(x, y)` constructs a WKB point from two `Float64` coordinate columns,
/// returning a binary array.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StPointUdf {
    signature: Signature,
}

impl Default for StPointUdf {
    fn default() -> Self {
        Self {
            // Exact Float64 args; DataFusion coerces other numeric inputs to Float64.
            signature: Signature::exact(
                vec![DataType::Float64, DataType::Float64],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for StPointUdf {
    fn name(&self) -> &str {
        "st_point"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Binary)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        let xs = as_float64_array(&arrays[0])?;
        let ys = as_float64_array(&arrays[1])?;

        let opts = WriteOptions::default();
        // Every point is exactly WKB_POINT_2D_LEN bytes, so a fixed buffer is fully
        // overwritten each row via a cursor.
        let mut buf = [0u8; WKB_POINT_2D_LEN];
        let mut builder = BinaryBuilder::with_capacity(xs.len(), xs.len() * WKB_POINT_2D_LEN);
        for i in 0..xs.len() {
            if xs.is_null(i) || ys.is_null(i) {
                builder.append_null();
                continue;
            }
            let pt = Point::new(xs.value(i), ys.value(i));
            write_point(&mut Cursor::new(&mut buf[..]), &pt, &opts)
                .map_err(|e| datafusion::common::exec_datafusion_err!("st_point: {e}"))?;
            builder.append_value(buf);
        }

        Ok(ColumnarValue::Array(Arc::new(builder.finish())))
    }
}

#[cfg(test)]
mod udf_tests {
    use arrow_array::{BinaryArray, Float64Array};
    use arrow_schema::Field;
    use geo_traits::{CoordTrait, GeometryTrait, GeometryType, PointTrait};
    use wkb::reader::Wkb;

    use super::*;

    #[test]
    fn test_udf_names() {
        assert_eq!(StWithinUdf::default().name(), "st_within");
        assert_eq!(StContainsUdf::default().name(), "st_contains");
        assert_eq!(StIntersectsUdf::default().name(), "st_intersects");
        assert_eq!(StPointUdf::default().name(), "st_point");
    }

    fn point_coords(bytes: &[u8]) -> (f64, f64) {
        let wkb = Wkb::try_new(bytes).unwrap();
        match wkb.as_type() {
            GeometryType::Point(p) => {
                let c = p.coord().unwrap();
                (c.x(), c.y())
            }
            _ => panic!("expected a point"),
        }
    }

    #[test]
    fn test_st_point_invoke() {
        let xs = Float64Array::from(vec![Some(1.0), Some(-3.5), None]);
        let ys = Float64Array::from(vec![Some(2.0), Some(4.25), Some(9.0)]);
        let n = xs.len();

        let udf = StPointUdf::default();
        let args = ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Array(Arc::new(xs)),
                ColumnarValue::Array(Arc::new(ys)),
            ],
            arg_fields: vec![
                Arc::new(Field::new("x", DataType::Float64, true)),
                Arc::new(Field::new("y", DataType::Float64, true)),
            ],
            number_rows: n,
            return_field: Arc::new(Field::new("out", DataType::Binary, true)),
            config_options: Arc::new(datafusion::config::ConfigOptions::default()),
        };

        let out = match udf.invoke_with_args(args).unwrap() {
            ColumnarValue::Array(a) => a,
            _ => panic!("expected array"),
        };
        let out = out.as_any().downcast_ref::<BinaryArray>().unwrap();

        assert_eq!(point_coords(out.value(0)), (1.0, 2.0));
        assert_eq!(point_coords(out.value(1)), (-3.5, 4.25));
        assert!(out.is_null(2));
    }
}
