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

use arrow::datatypes::DataType;
use arrow_schema::{Field, Schema, SchemaBuilder, SchemaRef};
use async_trait::async_trait;
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, Statistics, ToDFSchema},
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
use futures::StreamExt;

use crate::{
    async_reader::{ZarrPath, ZarrReadAsync},
    reader::ZarrResult,
};

use super::helpers::{expr_applicable_for_cols, pruned_partition_list, split_files};
use super::scanner::ZarrScan;

pub struct ListingZarrTableOptions {
    pub table_partition_cols: Vec<(String, DataType)>,
    pub target_partitions: usize,
}

impl ListingZarrTableOptions {
    pub fn new() -> Self {
        Self {
            table_partition_cols: vec![],
            target_partitions: 1,
        }
    }

    pub fn with_partition_cols(mut self, table_partition_cols: Vec<(String, DataType)>) -> Self {
        self.table_partition_cols = table_partition_cols;
        self
    }

    pub fn with_target_partitions(mut self, target_partitions: usize) -> Self {
        self.target_partitions = target_partitions;
        self
    }

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
    table_path: ListingTableUrl,
    pub file_schema: Schema,
    pub options: Option<ListingZarrTableOptions>,
}

impl ListingZarrTableConfig {
    /// Create a new ListingZarrTableConfig
    pub fn new(
        table_path: ListingTableUrl,
        file_schema: Schema,
        options: Option<ListingZarrTableOptions>
    ) -> Self {
        Self { table_path, file_schema, options }
    }
}

pub struct ZarrTableProvider {
    file_schema: Schema,
    table_schema: Schema,
    table_path: ListingTableUrl,
    options: ListingZarrTableOptions,
}

impl ZarrTableProvider {
    pub fn try_new(config: ListingZarrTableConfig) -> DataFusionResult<Self> {
        // TODO does options need to be an Option?
        let options = config.options.ok_or_else(|| {
            DataFusionError::Internal("No ListingOptions provided".into())
        })?;

        let mut builder = SchemaBuilder::from(config.file_schema.clone());
        for (part_col_name, part_col_type) in &options.table_partition_cols {
            builder.push(Field::new(part_col_name, part_col_type.clone(), false));
        }
        let table_schema = builder.finish();

        Ok(Self {
            file_schema: config.file_schema,
            table_schema,
            table_path: config.table_path,
            options,
        })
    }

    async fn list_stores_for_scan<'a>(
        &'a self,
        ctx: &'a SessionState,
        filters: &'a [Expr],
    ) -> datafusion::error::Result<Vec<Vec<PartitionedFile>>> {
        let store = ctx.runtime_env().object_store(&self.table_path)?;
        let mut partition_stream = pruned_partition_list(
                store.as_ref(),
                &self.table_path,
                filters,
                &self.options.table_partition_cols,
        ).await?;

        let mut partition_list = vec![];
        while let Some(partition) = partition_stream.next().await {
            partition_list.push(partition?);
        }

        Ok(split_files(partition_list, self.options.target_partitions))
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
        // TODO handle predicates on partition columns as Exact.
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
                    TableProviderFilterPushDown::Inexact
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
        let object_store_url = self.table_path.object_store();

        let pf = PartitionedFile::new(self.table_path.prefix().clone(), 0);
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
            file_schema: Arc::new(self.file_schema.clone()),
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
