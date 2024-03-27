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

use arrow_schema::{Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::{
    common::Statistics,
    datasource::{
        listing::{ListingTableConfig, ListingTableUrl, PartitionedFile},
        physical_plan::FileScanConfig,
        TableProvider, TableType,
    },
    execution::context::SessionState,
    logical_expr::{Expr, TableProviderFilterPushDown},
    physical_plan::ExecutionPlan,
};
use object_store::{ObjectMeta, ObjectStore};

use crate::{
    async_reader::{ZarrPath, ZarrReadAsync},
    reader::ZarrResult,
};

use super::scanner::ZarrScan;

pub struct ListingZarrTableOptions {}

impl ListingZarrTableOptions {
    pub async fn infer_schema_from_object(
        &self,
        store: Arc<dyn ObjectStore>,
        object_meta: &ObjectMeta,
    ) -> ZarrResult<Schema> {
        let zarr_path = ZarrPath::new(store, object_meta.location.clone());
        let schema = zarr_path.get_zarr_metadata().await?.arrow_schema()?;

        Ok(schema)
    }
}

pub struct ListingZarrTableConfig {
    /// The inner listing table configuration
    inner: ListingTableConfig,

    options: ListingZarrTableOptions,
}

impl ListingZarrTableConfig {
    /// Create a new ListingZarrTableConfig
    pub fn new(table_path: ListingTableUrl, options: ListingZarrTableOptions) -> Self {
        Self {
            inner: ListingTableConfig::new(table_path),
            options,
        }
    }

    pub fn table_paths(&self) -> Vec<ListingTableUrl> {
        self.inner.table_paths.to_vec()
    }
}

pub struct ZarrTableProvider {
    table_schema: Schema,
    config: ListingZarrTableConfig,
}

impl ZarrTableProvider {
    pub fn new(config: ListingZarrTableConfig, table_schema: Schema) -> Self {
        Self {
            table_schema,
            config,
        }
    }
}

#[async_trait]
impl TableProvider for ZarrTableProvider {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::new(self.table_schema.clone())
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion::error::Result<Vec<TableProviderFilterPushDown>> {
        // TODO: which filters can we push down?
        Ok(filters
            .iter()
            .map(|_| TableProviderFilterPushDown::Unsupported)
            .collect())
    }

    async fn scan(
        &self,
        state: &SessionState,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        limit: Option<usize>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        if self.config.table_paths().is_empty() {
            return Err(datafusion::error::DataFusionError::Execution(
                "No table paths found".to_string(),
            ));
        }

        if self.config.table_paths().len() > 1 {
            return Err(datafusion::error::DataFusionError::Execution(
                "Multiple table paths not supported".to_string(),
            ));
        }

        let table_path = self.config.table_paths();
        let table_path = table_path.first().unwrap();

        let object_store_url = table_path.object_store();

        let pf = PartitionedFile::from_path(table_path.to_string())?;

        let file_groups = vec![vec![pf]];

        let file_scan_config = FileScanConfig {
            object_store_url,
            file_schema: Arc::new(self.table_schema.clone()), // TODO differentiate between file and table schema
            file_groups,
            statistics: Statistics::new_unknown(&self.table_schema),
            projection: projection.cloned(),
            limit,
            table_partition_cols: vec![],
            output_ordering: vec![],
        };

        let scanner = ZarrScan::new(file_scan_config);

        Ok(Arc::new(scanner))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::{
        datasource::{listing::ListingTableUrl, physical_plan::FileMeta},
        error::DataFusionError,
        execution::context::SessionContext,
    };
    use object_store::{local::LocalFileSystem, path::Path, ObjectMeta};

    use crate::{
        datafusion::table_provider::{
            ListingZarrTableConfig, ListingZarrTableOptions, ZarrTableProvider,
        },
        tests::get_test_v2_data_path,
    };

    #[tokio::test]
    async fn test_table_provider() -> Result<(), Box<dyn std::error::Error>> {
        let local_fs = Arc::new(LocalFileSystem::new());

        let test_data = get_test_v2_data_path("lat_lon_example.zarr".to_string());
        let location = Path::from_filesystem_path(&test_data)?;

        let ctx = SessionContext::new();

        let file_meta = FileMeta {
            object_meta: ObjectMeta {
                location: location.clone(),
                last_modified: chrono::Utc::now(),
                size: 0,
                e_tag: None,
                version: None,
            },
            range: None,
            extensions: None,
        };

        let options = ListingZarrTableOptions {};
        let schema = options
            .infer_schema_from_object(local_fs, &file_meta.object_meta)
            .await
            .map_err(|e| DataFusionError::Execution(format!("infer error: {:?}", e)))?;

        let listing_table_url = ListingTableUrl::parse(location)?;

        let table_provider = ZarrTableProvider::new(
            ListingZarrTableConfig::new(
                ListingTableUrl::parse(location.to_string())?,
                ListingZarrTableOptions {},
            ),
            schema,
        );

        ctx.register_table("zarr", Arc::new(table_provider))?;

        let df = ctx.sql("SELECT lon, lat FROM zarr LIMIT 10").await?;

        let results = df.collect().await?;
        assert_eq!(results.len(), 1);

        let batch = &results[0];
        assert_eq!(batch.num_columns(), 2);

        Ok(())
    }
}
