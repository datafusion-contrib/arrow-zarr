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

use datafusion::common::Statistics;
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::{
    FileGroup, FileScanConfigBuilder, FileSource, FileStreamBuilder,
};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
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
    filters: Option<Arc<dyn PhysicalExpr>>,
    plan_properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl ZarrScan {
    pub(crate) fn new(
        zarr_config: ZarrTableConfig,
        filters: Option<Arc<dyn PhysicalExpr>>,
    ) -> Self {
        let plan_properties = PlanProperties::new(
            EquivalenceProperties::new(zarr_config.get_projected_schema_ref()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Self {
            zarr_config,
            filters,
            plan_properties: Arc::new(plan_properties),
            metrics: ExecutionPlanMetricsSet::default(),
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

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.plan_properties
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn partition_statistics(
        &self,
        partition: Option<usize>,
    ) -> datafusion::error::Result<Arc<Statistics>> {
        match partition {
            // whole-plan statistics (all partitions combined).
            None => Ok(Arc::new(self.zarr_config.statistics()?)),
            // we can't cheaply give an exact per-partition row count (the
            // reader slices the chunk grid into contiguous ranges and edge
            // chunks vary in size), so we report unknown for a specific
            // partition.
            Some(_) => Ok(Arc::new(Statistics::new_unknown(
                &self.zarr_config.get_projected_schema_ref(),
            ))),
        }
    }

    fn repartitioned(
        &self,
        target_partitions: usize,
        _config: &datafusion::config::ConfigOptions,
    ) -> datafusion::error::Result<Option<Arc<dyn ExecutionPlan>>> {
        let mut new_plan = self.clone();
        let props = new_plan
            .plan_properties
            .as_ref()
            .clone()
            .with_partitioning(Partitioning::UnknownPartitioning(target_partitions));
        new_plan.plan_properties = Arc::new(props);
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

        let zarr_source = ZarrSource::new(
            self.zarr_config.clone(),
            n_partitions,
            self.filters.clone(),
            self.metrics.clone(),
        );
        // dummy file group, it's needed to re-use some of the datafusion code,
        // but it doesn't really apply for a zarr store.
        let file_groups = vec![FileGroup::new(vec![PartitionedFile::new("", 0)])];
        let file_scan_config = FileScanConfigBuilder::new(
            ObjectStoreUrl::parse("file://").unwrap(),
            Arc::new(zarr_source.clone()),
        )
        .with_file_groups(file_groups)
        .build();

        let dummy_object_store = Arc::new(LocalFileSystem::new());
        let file_opener =
            zarr_source.create_file_opener(dummy_object_store, &file_scan_config, partition)?;

        // Note: the "partition" argument is hardcoded to 0 here. We are not making
        // use of most of the logic in the file stream, for example the partitioning
        // logic is handled in the zarr stream object, so we need to effectively
        // "disable" it in the file stream obejct by always setting it to 0.
        // Passing in dummy metrics because the real metrics are computed directly
        // from the data stream, don't want to create any confusion around that.
        let dummy_metrics = ExecutionPlanMetricsSet::default();
        let file_stream = FileStreamBuilder::new(&file_scan_config)
            .with_partition(0)
            .with_file_opener(file_opener)
            .with_metrics(&dummy_metrics)
            .build()?;

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
    use crate::table::config::ZarrUrlBuilder;
    use crate::test_utils::{
        get_local_zarr_store, validate_names_and_types, validate_primitive_column,
    };

    #[tokio::test]
    async fn read_data_test() {
        let (wrapper, schema) = get_local_zarr_store(true, 0.0, "lat_lon_data_for_scan").await;
        let path = wrapper.get_store_path();
        let table_url = ZarrUrlBuilder::try_new(ListingTableUrl::parse(path).unwrap(), None)
            .unwrap()
            .build()
            .unwrap();
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
        let table_url = ZarrUrlBuilder::try_new(ListingTableUrl::parse(path).unwrap(), None)
            .unwrap()
            .build()
            .unwrap();
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

    // reads the row-count statistics off the exec plan for different sets of selected
    // columns (by passing a different schema into the config each time). the icechunk
    // store has 1D `lat`/`lon` coordinates and a 2D `data` array, so `lat` alone spans a
    // single dimension (8 rows) while any projection including a second dimension (e.g.
    // `lon` or `data`) broadcasts to the full 2D grid (64 rows).
    #[cfg(feature = "icechunk")]
    #[tokio::test]
    async fn statistics_test() {
        use arrow_schema::{Field, Schema};
        use datafusion::common::stats::Precision;

        use crate::table::config::IcechunkUrlBuilder;
        use crate::test_utils::get_local_icechunk_repo;

        let (wrapper, _schema) =
            get_local_icechunk_repo(true, 0.0, "lat_lon_data_for_scan_stats").await;
        let path = wrapper.get_store_path();
        let table_url = IcechunkUrlBuilder::try_new(ListingTableUrl::parse(path).unwrap(), None)
            .unwrap()
            .build()
            .unwrap();
        let (inferred_schema, stats_base) = table_url.infer_schema().await.unwrap();

        let num_rows = |names: &[&str]| {
            let fields: Vec<Field> = names
                .iter()
                .map(|n| inferred_schema.field_with_name(n).unwrap().clone())
                .collect();
            let config = ZarrTableConfig::new(table_url.clone(), Arc::new(Schema::new(fields)))
                .with_stats_base(stats_base.clone());
            ZarrScan::new(config, None)
                .partition_statistics(None)
                .unwrap()
                .num_rows
        };

        assert_eq!(num_rows(&["lat"]), Precision::Exact(8));
        assert_eq!(num_rows(&["lat", "lon"]), Precision::Exact(64));
        assert_eq!(num_rows(&["data", "lat", "lon"]), Precision::Exact(64));
    }

    // runs a full scan (all three columns, no filter) and reads the runtime metrics back
    // off the exec plan. the store is a 8x8 grid split into 3x3 chunks -> 9 chunks, all of
    // which are read (no filter), for 64 rows.
    #[cfg(feature = "icechunk")]
    #[tokio::test]
    async fn metrics_test() {
        use crate::table::config::IcechunkUrlBuilder;
        use crate::test_utils::get_local_icechunk_repo;

        let (wrapper, schema) =
            get_local_icechunk_repo(true, 0.0, "lat_lon_data_for_scan_metrics").await;
        let path = wrapper.get_store_path();
        let table_url = IcechunkUrlBuilder::try_new(ListingTableUrl::parse(path).unwrap(), None)
            .unwrap()
            .build()
            .unwrap();
        let config = ZarrTableConfig::new(table_url, schema);

        let session = SessionContext::new();
        let scan = ZarrScan::new(config, None);
        let records: Vec<_> = scan
            .execute(0, session.task_ctx())
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert_eq!(records.len(), 9);

        // the scan owns the metrics set the reader registered into, so we can read it back.
        let snapshot = scan.metrics().unwrap();
        for name in [
            "io_time",
            "decode_time",
            "total_time",
            "chunks_looked_at",
            "chunks_read",
            "rows_produced",
        ] {
            assert!(
                snapshot.sum_by_name(name).is_some(),
                "metric {name} missing from the metrics set"
            );
        }

        let value = |name: &str| snapshot.sum_by_name(name).unwrap().as_usize();
        assert_eq!(value("chunks_looked_at"), 9);
        assert_eq!(value("chunks_read"), 9);
        assert_eq!(value("rows_produced"), 64);
    }
}
