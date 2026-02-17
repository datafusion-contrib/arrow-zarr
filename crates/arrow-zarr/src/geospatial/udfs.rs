use arrow_schema::DataType;
use datafusion::common::{not_impl_err, Result};
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature, Volatility,
};

macro_rules! spatial_predicate_stub {
    ($($struct_name:ident => $udf_name:literal),* $(,)?) => {
        $(
            #[derive(Debug)]
            pub struct $struct_name {
                signature: Signature,
            }

            impl Default for $struct_name {
                fn default() -> Self {
                    Self {
                        signature: Signature::one_of(
                            vec![
                                TypeSignature::Exact(vec![DataType::Binary, DataType::Binary]),
                                TypeSignature::Exact(vec![DataType::LargeBinary, DataType::LargeBinary]),
                                TypeSignature::Exact(vec![DataType::BinaryView, DataType::BinaryView]),
                            ],
                            Volatility::Immutable,
                        ),
                    }
                }
            }

            impl ScalarUDFImpl for $struct_name {
                fn as_any(&self) -> &dyn std::any::Any { self }
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
    StWithinUdf => "st_within_bulk",
    StContainsUdf => "st_contains_bulk",
}

#[cfg(test)]
mod udf_tests {
    use super::*;

    #[test]
    fn test_udf_names() {
        assert_eq!(StWithinUdf::default().name(), "st_within_bulk");
        assert_eq!(StContainsUdf::default().name(), "st_contains_bulk");
    }
}
