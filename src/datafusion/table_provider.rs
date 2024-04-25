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
    common::{Statistics, ToDFSchema},
    datasource::{
        listing::{ListingTableUrl, PartitionedFile},
        physical_plan::FileScanConfig,
        TableProvider, TableType,
    },
    execution::context::SessionState,
    logical_expr::{utils::conjunction, Expr, TableProviderFilterPushDown},
    physical_plan::ExecutionPlan,
};
use datafusion_physical_expr::create_physical_expr;

use crate::{
    async_reader::{ZarrPath, ZarrReadAsync},
    reader::ZarrResult,
};

use super::helpers::expr_applicable_for_cols;
use super::scanner::ZarrScan;

pub struct ListingZarrTableOptions {}

impl ListingZarrTableOptions {
    pub async fn infer_schema(
        &self,
        state: &SessionState,
        table_path: &ListingTableUrl,
    ) -> ZarrResult<Schema> {
        let store = state.runtime_env().object_store(table_path)?;

        let zarr_path = ZarrPath::new(store, table_path.prefix().clone());
        let schema = zarr_path.get_zarr_metadata().await?.arrow_schema()?;

        Ok(schema)
    }
}

pub struct ListingZarrTableConfig {
    /// The inner listing table configuration
    table_path: ListingTableUrl,
}

impl ListingZarrTableConfig {
    /// Create a new ListingZarrTableConfig
    pub fn new(table_path: ListingTableUrl) -> Self {
        Self { table_path }
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
        Ok(filters
            .iter()
            .map(|filter| {
                if expr_applicable_for_cols(
                    &self
                        .table_schema
                        .fields
                        .iter()
                        .map(|field| field.name().to_string())
                        .collect::<Vec<_>>(),
                    filter,
                ) {
                    TableProviderFilterPushDown::Exact
                } else {
                    TableProviderFilterPushDown::Unsupported
                }
            })
            .collect())
    }

    async fn scan(
        &self,
        state: &SessionState,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        let object_store_url = self.config.table_path.object_store();

        let pf = PartitionedFile::new(self.config.table_path.prefix().clone(), 0);
        let file_groups = vec![vec![pf]];

        let filters = if let Some(expr) = conjunction(filters.to_vec()) {
            let table_df_schema = self.table_schema.clone().to_dfschema()?;
            let filters = create_physical_expr(&expr, &table_df_schema, state.execution_props())?;
            Some(filters)
        } else {
            None
        };

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

        let scanner = ZarrScan::new(file_scan_config, filters);

        Ok(Arc::new(scanner))
    }
}
