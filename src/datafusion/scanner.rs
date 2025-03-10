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

use std::{any::Any, sync::Arc};

use arrow::datatypes::SchemaRef;
use datafusion::{
    common::Statistics,
    datasource::physical_plan::{FileScanConfig, FileStream},
    physical_plan::{
        metrics::ExecutionPlanMetricsSet, DisplayAs, DisplayFormatType, ExecutionPlan,
        Partitioning, PhysicalExpr, SendableRecordBatchStream,
    },
};

use super::{config::ZarrConfig, file_opener::ZarrFileOpener};

#[derive(Debug, Clone)]
/// Implements a DataFusion `ExecutionPlan` for Zarr files.
pub struct ZarrScan {
    /// The base configuration for the file scan.
    base_config: FileScanConfig,

    /// The projected schema for the scan.
    projected_schema: SchemaRef,

    /// Metrics for the execution plan.
    metrics: ExecutionPlanMetricsSet,

    /// The statistics for the scan.
    statistics: Statistics,

    /// Filters that will be pushed down to the Zarr stream reader.
    filters: Option<Arc<dyn PhysicalExpr>>,
}

impl ZarrScan {
    /// Create a new Zarr scan.
    pub fn new(base_config: FileScanConfig, filters: Option<Arc<dyn PhysicalExpr>>) -> Self {
        let (projected_schema, statistics, _lex_sorting) = base_config.project();

        Self {
            base_config,
            projected_schema,
            metrics: ExecutionPlanMetricsSet::new(),
            statistics,
            filters,
        }
    }
}

impl DisplayAs for ZarrScan {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "ZarrScan")
    }
}

impl ExecutionPlan for ZarrScan {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.projected_schema.clone()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn statistics(&self) -> datafusion::error::Result<Statistics> {
        Ok(self.statistics.clone())
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> datafusion::error::Result<datafusion::physical_plan::SendableRecordBatchStream> {
        let object_store = context
            .runtime_env()
            .object_store(&self.base_config.object_store_url)?;

        // This is just replicating the `file_column_projection_indices` method on
        // `FileScanConfig`, which is only pub within the datafusion crate. We need
        // to remove column indices that correspond to partitions, since we can't
        // pass those to the zarr reader.
        let projection = self.base_config.projection.as_ref().map(|p| {
            p.iter()
                .filter(|col_idx| **col_idx < self.base_config.file_schema.fields().len())
                .copied()
                .collect()
        });

        let config = ZarrConfig::new(object_store).with_projection(projection);
        let opener = ZarrFileOpener::new(config, self.filters.clone());
        let stream = FileStream::new(&self.base_config, partition, opener, &self.metrics)?;

        Ok(Box::pin(stream) as SendableRecordBatchStream)
    }

    fn output_partitioning(&self) -> datafusion::physical_plan::Partitioning {
        Partitioning::UnknownPartitioning(self.base_config.file_groups.len())
    }

    fn output_ordering(&self) -> Option<&[datafusion::physical_expr::PhysicalSortExpr]> {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::{error::Error, sync::Arc};

    use datafusion::{
        datasource::{listing::PartitionedFile, physical_plan::FileMeta},
        execution::object_store::ObjectStoreUrl,
    };
    use futures::TryStreamExt;
    use object_store::{local::LocalFileSystem, path::Path, ObjectMeta};

    use super::*;
    use crate::async_reader::{ZarrPath, ZarrReadAsync};
    use crate::test_utils::{store_lat_lon, StoreWrapper};
    use rstest::*;

    #[rstest]
    #[tokio::test]
    async fn test_scanner_open(
        #[with("test_scanner_open".to_string())] store_lat_lon: StoreWrapper,
    ) -> Result<(), Box<dyn Error>> {
        let local_fs = Arc::new(LocalFileSystem::new());

        let test_data_pathbuf = store_lat_lon.store_path();
        let test_data = test_data_pathbuf.to_str().unwrap();
        let location = Path::from_filesystem_path(test_data)?;

        let file_meta = FileMeta {
            object_meta: ObjectMeta {
                location,
                last_modified: chrono::Utc::now(),
                size: 0,
                e_tag: None,
                version: None,
            },
            range: None,
            extensions: None,
        };

        let zarr_path = ZarrPath::new(local_fs, file_meta.object_meta.location);
        let schema = zarr_path.get_zarr_metadata().await?.arrow_schema()?;

        let test_file = Path::from_filesystem_path(test_data)?;
        let scan_config = FileScanConfig {
            object_store_url: ObjectStoreUrl::local_filesystem(),
            file_schema: Arc::new(schema.clone()),
            file_groups: vec![vec![PartitionedFile::new(test_file.to_string(), 10)]],
            statistics: Statistics::new_unknown(&schema),
            projection: Some(vec![1, 2]),
            limit: Some(10),
            table_partition_cols: vec![],
            output_ordering: vec![],
        };

        let scanner = ZarrScan::new(scan_config, None);

        let session = datafusion::execution::context::SessionContext::new();

        let batch_stream = scanner.execute(0, session.task_ctx())?;
        let batches: Vec<_> = batch_stream.try_collect().await?;

        assert_eq!(batches.len(), 1);

        let first_batch = &batches[0];
        assert_eq!(first_batch.num_columns(), 2);
        assert_eq!(first_batch.num_rows(), 10);

        let schema = first_batch.schema();

        let names = schema.fields().iter().map(|f| f.name()).collect::<Vec<_>>();
        assert_eq!(names, vec!["lat", "lon"]);

        Ok(())
    }
}
