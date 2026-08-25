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

use arrow_schema::DataType;
use datafusion::common::{not_impl_err, Result};
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature, Volatility,
};

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
StContainsUdf => "st_contains",}

#[cfg(test)]
mod udf_tests {
    use super::*;

    #[test]
    fn test_udf_names() {
        assert_eq!(StWithinUdf::default().name(), "st_within");
        assert_eq!(StContainsUdf::default().name(), "st_contains");
    }
}
