use super::config::ZarrConfig;
use super::scanner::ZarrScan;
use async_trait::async_trait;
use datafusion::arrow::datatypes::{Schema, SchemaRef};
use datafusion::catalog::Session;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::ExecutionPlan;
use std::fmt::Debug;
use std::sync::Arc;
use zarrs_storage::AsyncReadableListableStorageTraits;
pub struct ZarrTable {
    table_schema: Schema,
    zarr_storage: Arc<dyn AsyncReadableListableStorageTraits + Unpin + Send>,
}

impl Debug for ZarrTable {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl ZarrTable {
    pub fn new(
        table_schema: Schema,
        zarr_storage: Arc<dyn AsyncReadableListableStorageTraits + Unpin + Send>,
    ) -> Self {
        Self {
            table_schema,
            zarr_storage,
        }
    }
}

#[async_trait]
impl TableProvider for ZarrTable {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::new(self.table_schema.clone())
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        _projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        let config = ZarrConfig::new(self.zarr_storage.clone(), self.table_schema.clone());
        let scanner = ZarrScan::new(config, None);

        Ok(Arc::new(scanner))
    }
}

#[cfg(test)]
mod table_provider_tests {
    use std::collections::HashMap;

    use super::*;
    use crate::table::table_provider::ZarrTable;
    use crate::test_utils::{
        get_lat_lon_data_store, validate_names_and_types, validate_primitive_column,
    };
    use arrow::datatypes::Float64Type;
    use arrow_schema::DataType;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionContext;
    use futures_util::TryStreamExt;

    #[tokio::test]
    async fn read_data_test() {
        let (wrapper, schema) =
            get_lat_lon_data_store(true, 0.0, "lat_lon_data_for_provider").await;
        let store = wrapper.get_store();

        let table_provider = ZarrTable::new(schema, store);
        let state = SessionStateBuilder::new().build();
        let session = SessionContext::new();

        let scan = table_provider
            .scan(&state, None, &Vec::new(), None)
            .await
            .unwrap();
        let records: Vec<_> = scan
            .execute(0, session.task_ctx())
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        let target_types = HashMap::from([
            ("lat".to_string(), DataType::Float64),
            ("lon".to_string(), DataType::Float64),
            ("data".to_string(), DataType::Float64),
        ]);
        validate_names_and_types(&target_types, &records[0]);
        assert_eq!(records.len(), 9);

        // the top left chunk, full 3x3
        validate_primitive_column::<Float64Type, f64>(
            "lat",
            &records[0],
            &[35., 35., 35., 36., 36., 36., 37., 37., 37.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            &records[0],
            &[
                -120.0, -119.0, -118.0, -120.0, -119.0, -118.0, -120.0, -119.0, -118.0,
            ],
        );
        validate_primitive_column::<Float64Type, f64>(
            "data",
            &records[0],
            &[0.0, 1.0, 2.0, 8.0, 9.0, 10.0, 16.0, 17.0, 18.0],
        );
    }
}
