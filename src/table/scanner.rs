use std::any::Any;
use std::sync::Arc;

use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::{
    FileGroup, FileScanConfigBuilder, FileSource, FileStream,
};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PhysicalExpr, PlanProperties,
    SendableRecordBatchStream,
};
use object_store::local::LocalFileSystem;

use super::config::ZarrTableConfig;
use super::opener::ZarrSource;

#[derive(Debug, Clone)]
pub struct ZarrScan {
    zarr_config: ZarrTableConfig,
    _filters: Option<Arc<dyn PhysicalExpr>>,
    plan_properties: PlanProperties,
}

impl ZarrScan {
    pub(crate) fn new(
        zarr_config: ZarrTableConfig,
        _filters: Option<Arc<dyn PhysicalExpr>>,
    ) -> Self {
        let plan_properties = PlanProperties::new(
            EquivalenceProperties::new(zarr_config.get_projected_schema_ref()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Self {
            zarr_config,
            _filters,
            plan_properties,
        }
    }
}

impl DisplayAs for ZarrScan {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "ZarrScan")
    }
}

impl ExecutionPlan for ZarrScan {
    fn name(&self) -> &str {
        "ZarrScan"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn properties(&self) -> &PlanProperties {
        &self.plan_properties
    }

    fn repartitioned(
        &self,
        target_partitions: usize,
        _config: &datafusion::config::ConfigOptions,
    ) -> datafusion::error::Result<Option<Arc<dyn ExecutionPlan>>> {
        let mut new_plan = self.clone();
        new_plan.plan_properties = new_plan
            .plan_properties
            .with_partitioning(Partitioning::UnknownPartitioning(target_partitions));
        Ok(Some(Arc::new(new_plan)))
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<datafusion::execution::TaskContext>,
    ) -> datafusion::error::Result<SendableRecordBatchStream> {
        let n_partitions = match self.plan_properties.partitioning {
            Partitioning::UnknownPartitioning(n) => n,
            _ => {
                return Err(datafusion::error::DataFusionError::Execution(
                    "Only Unknown partitioning support for zarr scans".into(),
                ));
            }
        };

        let zarr_source = ZarrSource::new(self.zarr_config.clone(), n_partitions);
        let file_groups = vec![FileGroup::new(vec![PartitionedFile::new("", 0)])];
        let file_scan_config = FileScanConfigBuilder::new(
            ObjectStoreUrl::parse("file://").unwrap(),
            self.zarr_config.get_schema_ref(),
            Arc::new(zarr_source.clone()),
        )
        .with_file_groups(file_groups)
        .with_projection(self.zarr_config.get_projection())
        .build();

        let dummy_object_store = Arc::new(LocalFileSystem::new());
        let file_opener =
            zarr_source.create_file_opener(dummy_object_store, &file_scan_config, partition);
        let metrics = ExecutionPlanMetricsSet::default();

        // Note: the "partition" argument is hardcoded to 0 here. We are not making
        // use of most of the logic in the file stream, for example the partitioning
        // logic is handled in the zarr stream object, so we need to effectively
        // "disable" it in the file stream obejct by always setting it to 0.
        let file_stream = FileStream::new(&file_scan_config, 0, file_opener, &metrics).unwrap();

        Ok(Box::pin(file_stream))
    }
}

#[cfg(test)]
mod scanner_tests {
    use std::collections::HashMap;

    use arrow::datatypes::Float64Type;
    use arrow_schema::DataType;
    use datafusion::config::ConfigOptions;
    use datafusion::datasource::listing::ListingTableUrl;
    use datafusion::prelude::SessionContext;
    use futures_util::TryStreamExt;

    use super::*;
    use crate::table::config::ZarrTableUrl;
    use crate::test_utils::{
        get_local_zarr_store, validate_names_and_types, validate_primitive_column,
    };

    #[tokio::test]
    async fn read_data_test() {
        let (wrapper, schema) = get_local_zarr_store(true, 0.0, "lat_lon_data_for_scan").await;
        let path = wrapper.get_store_path();
        let table_url = ZarrTableUrl::ZarrStore(ListingTableUrl::parse(path).unwrap());
        let config = ZarrTableConfig::new(table_url, schema);

        let session = SessionContext::new();
        let scan = ZarrScan::new(config, None);
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

    #[tokio::test]
    async fn read_partition_test() {
        let (wrapper, schema) =
            get_local_zarr_store(true, 0.0, "lat_lon_data_for_scan_with_partition").await;
        let path = wrapper.get_store_path();
        let table_url = ZarrTableUrl::ZarrStore(ListingTableUrl::parse(path).unwrap());
        let config = ZarrTableConfig::new(table_url, schema);

        let session = SessionContext::new();
        let scan = ZarrScan::new(config, None);
        let scan = scan
            .repartitioned(2, &ConfigOptions::default())
            .unwrap()
            .unwrap();

        let records: Vec<_> = scan
            .execute(1, session.task_ctx())
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
        assert_eq!(records.len(), 4);
    }
}
