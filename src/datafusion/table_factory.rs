// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::{
    datasource::{listing::ListingTableUrl, provider::TableProviderFactory, TableProvider},
    error::DataFusionError,
    execution::context::SessionState,
    logical_expr::CreateExternalTable,
};

use super::table_provider::{ListingZarrTableConfig, ListingZarrTableOptions, ZarrTableProvider};

struct ZarrListingTableFactory {}

#[async_trait]
impl TableProviderFactory for ZarrListingTableFactory {
    async fn create(
        &self,
        state: &SessionState,
        cmd: &CreateExternalTable,
    ) -> datafusion::common::Result<Arc<dyn TableProvider>> {
        if cmd.file_type != "ZARR" {
            return Err(datafusion::error::DataFusionError::Execution(
                "Invalid file type".to_string(),
            ));
        }

        let table_path = ListingTableUrl::parse(&cmd.location)?;

        let options = ListingZarrTableOptions {};
        let schema = options
            .infer_schema(state, &table_path)
            .await
            .map_err(|e| DataFusionError::Execution(format!("infer error: {:?}", e)))?;

        let table_provider =
            ZarrTableProvider::new(ListingZarrTableConfig::new(table_path), schema);

        Ok(Arc::new(table_provider))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::execution::{
        config::SessionConfig,
        context::{SessionContext, SessionState},
        runtime_env::RuntimeEnv,
    };

    use crate::tests::get_test_v2_data_path;

    #[tokio::test]
    async fn test_create() -> Result<(), Box<dyn std::error::Error>> {
        let mut state = SessionState::new_with_config_rt(
            SessionConfig::default(),
            Arc::new(RuntimeEnv::default()),
        );

        state
            .table_factories_mut()
            .insert("ZARR".into(), Arc::new(super::ZarrListingTableFactory {}));

        let test_data = get_test_v2_data_path("lat_lon_example.zarr".to_string());

        let sql = format!(
            "CREATE EXTERNAL TABLE zarr_table STORED AS ZARR LOCATION '{}'",
            test_data.display(),
        );

        let session = SessionContext::new_with_state(state);
        session.sql(&sql).await?;

        let sql = "SELECT lat, lon FROM zarr_table LIMIT 10";
        let df = session.sql(sql).await?;

        let batches = df.collect().await?;
        assert_eq!(batches.len(), 1);

        let batch = &batches[0];

        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 10);

        Ok(())
    }
}
