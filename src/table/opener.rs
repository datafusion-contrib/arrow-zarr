use std::any::Any;
use std::fmt;
use std::fmt::Formatter;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion::common::Statistics;
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::{
    FileMeta, FileOpenFuture, FileOpener, FileScanConfig, FileSource,
};
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion::physical_plan::DisplayFormatType;
use futures::StreamExt;
use object_store::ObjectStore;

use super::config::ZarrTableConfig;
use crate::ZarrRecordBatchStream;

/// Implementation of [`FileOpener`] for zarr.
pub(crate) struct ZarrOpener {
    config: ZarrTableConfig,
    n_partitions: usize,
    partition: usize,
}

impl ZarrOpener {
    fn new(config: ZarrTableConfig, n_partitions: usize, partition: usize) -> Self {
        Self {
            config,
            n_partitions,
            partition,
        }
    }
}

impl FileOpener for ZarrOpener {
    // We don't actually need any information about the file partitions, as those
    // don't really make sense for zarr. there's a high level store, and the data
    // is retrieved one chunk at a time, with all the logic inside the zarr stream.
    // There is the option to split the zarr chunks between some number of partitions,
    // but this is again handled inside the zarr stream. We are only implementing
    // this to re-use some of the datafusion functionalities.
    fn open(&self, _file_meta: FileMeta, _file: PartitionedFile) -> DfResult<FileOpenFuture> {
        let config = self.config.clone();
        let (n_partitions, partition) = (self.n_partitions, self.partition);
        let stream = Box::pin(async move {
            let (store, prefix) = config.get_store_pointer_and_prefix().await?;
            let inner_stream = ZarrRecordBatchStream::try_new(
                store,
                config.get_schema_ref(),
                prefix,
                config.get_projection(),
                n_partitions,
                partition,
            )
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
            Ok(inner_stream.boxed())
        });

        Ok(stream)
    }
}

/// Implementation of [`FileSource`] for zarr.
#[derive(Clone)]
pub(crate) struct ZarrSource {
    config: ZarrTableConfig,
    n_partitions: usize,
    exec_plan_metrics: ExecutionPlanMetricsSet,
}

impl ZarrSource {
    pub(crate) fn new(config: ZarrTableConfig, n_partitions: usize) -> Self {
        Self {
            config,
            n_partitions,
            exec_plan_metrics: ExecutionPlanMetricsSet::default(),
        }
    }
}

impl FileSource for ZarrSource {
    // Once again, we don't really need this, it's only so that
    // we can re-use some stuff from datafusion.
    fn create_file_opener(
        &self,
        _object_store: Arc<dyn ObjectStore>,
        _base_config: &FileScanConfig,
        partition: usize,
    ) -> Arc<dyn FileOpener> {
        let file_opener = ZarrOpener::new(self.config.clone(), self.n_partitions, partition);
        Arc::new(file_opener)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    // We don't really need most of the below functions, since we're
    // barely using this struct, but they are required by the trait.
    fn with_batch_size(&self, _batch_size: usize) -> Arc<dyn FileSource> {
        Arc::new(self.clone())
    }

    fn with_schema(&self, _schema: SchemaRef) -> Arc<dyn FileSource> {
        Arc::new(self.clone())
    }

    fn with_projection(&self, _config: &FileScanConfig) -> Arc<dyn FileSource> {
        Arc::new(self.clone())
    }

    fn with_statistics(&self, _statistics: Statistics) -> Arc<dyn FileSource> {
        Arc::new(self.clone())
    }

    fn metrics(&self) -> &ExecutionPlanMetricsSet {
        &self.exec_plan_metrics
    }

    fn statistics(&self) -> DfResult<Statistics> {
        Ok(Statistics::default())
    }

    /// String representation of file source
    fn file_type(&self) -> &str {
        "zarr"
    }

    /// Format FileType specific information
    fn fmt_extra(&self, _t: DisplayFormatType, _f: &mut Formatter) -> fmt::Result {
        Ok(())
    }
}

#[cfg(test)]
mod file_opener_tests {
    use std::collections::HashMap;

    use arrow::datatypes::Float64Type;
    use arrow_schema::DataType;
    use datafusion::datasource::listing::ListingTableUrl;
    use datafusion::datasource::physical_plan::{FileGroup, FileScanConfigBuilder, FileStream};
    use datafusion::execution::object_store::ObjectStoreUrl;
    use futures_util::TryStreamExt;
    use object_store::local::LocalFileSystem;

    use super::*;
    use crate::table::config::ZarrTableUrl;
    use crate::test_utils::{
        get_local_zarr_store, validate_names_and_types, validate_primitive_column,
    };

    #[tokio::test]
    async fn filestream_tests() {
        let (wrapper, schema) = get_local_zarr_store(true, 0.0, "data_for_file_opener").await;
        let path = wrapper.get_store_path();
        let table_url = ZarrTableUrl::ZarrStore(ListingTableUrl::parse(path).unwrap());

        let zarr_config = ZarrTableConfig::new(table_url, schema.clone());
        let zarr_souce = ZarrSource::new(zarr_config, 1);

        let file_groups = vec![FileGroup::new(vec![PartitionedFile::new("", 0)])];
        let file_scan_config = FileScanConfigBuilder::new(
            ObjectStoreUrl::parse("file://").unwrap(),
            schema,
            Arc::new(zarr_souce.clone()),
        )
        .with_file_groups(file_groups)
        .build();
        let dummy_object_store = Arc::new(LocalFileSystem::new());
        let file_opener = zarr_souce.create_file_opener(dummy_object_store, &file_scan_config, 0);

        let metrics = ExecutionPlanMetricsSet::default();
        let file_stream = FileStream::new(&file_scan_config, 0, file_opener, &metrics).unwrap();
        let records: Vec<_> = file_stream.try_collect().await.unwrap();

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
