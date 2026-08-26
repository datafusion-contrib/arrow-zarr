// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Arc;

use datafusion::common::Statistics;
use datafusion::error::DataFusionError;
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PhysicalExpr, PlanProperties,
    SendableRecordBatchStream,
};
use futures::{StreamExt, TryFutureExt};

use super::config::ZarrTableConfig;
use super::datafusion_filters::create_zarr_chunk_filter;
use crate::zarr_store_opener::metrics::ZarrMetrics;
use crate::ZarrRecordBatchStream;

#[derive(Debug, Clone)]
pub struct ZarrScan {
    zarr_config: ZarrTableConfig,
    filters: Option<Arc<dyn PhysicalExpr>>,
    plan_properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
    // an optional per-partition row limit, taken from the advisory `limit`
    // datafusion passes to TableProvider::scan. it's best-effort (a chunk
    // is emitted whole, so the count can overshoot), which is fine because
    // datafusion will enforce the actual limit later.
    limit: Option<usize>,
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
            limit: None,
        }
    }

    pub(crate) fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
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

    // the only per-execution state we hold is the metrics set.
    fn reset_state(self: Arc<Self>) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        let mut new_plan = self.as_ref().clone();
        new_plan.metrics = ExecutionPlanMetricsSet::default();
        Ok(Arc::new(new_plan))
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<datafusion::execution::TaskContext>,
    ) -> datafusion::error::Result<SendableRecordBatchStream> {
        let n_partitions = match self.plan_properties.partitioning {
            Partitioning::UnknownPartitioning(n) => n,
            _ => {
                return Err(DataFusionError::Execution(
                    "Only Unknown partitioning support for zarr scans".into(),
                ));
            }
        };

        // the zarr reader does all the real work (store access, projection,
        // partitioning, chunk-level filtering), so there's no file layer to
        // manage here. we just build the reader stream and hand it to
        // datafusion. `execute` has to return synchronously, but opening the
        // store is async, so we wrap the async construction in a future that
        // resolves to the stream and flatten it.
        let config = self.zarr_config.clone();
        let filter_expr = self.filters.clone();
        let metrics = ZarrMetrics::new(&self.metrics, partition);
        let schema = config.get_projected_schema_ref();
        let limit = self.limit;

        let stream_fut = async move {
            let filter = filter_expr
                .as_ref()
                .map(|f| create_zarr_chunk_filter(f, config.get_schema_ref()))
                .transpose()?;
            let (store, prefix) = config.get_store_pointer_and_prefix().await?;
            let inner = ZarrRecordBatchStream::try_new(
                store,
                config.get_schema_ref(),
                prefix,
                config.get_projection(),
                n_partitions,
                partition,
                filter,
                metrics,
                limit,
            )
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
            Ok::<_, DataFusionError>(inner.map(|batch| batch.map_err(DataFusionError::from)))
        };

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream_fut.try_flatten_stream(),
        )))
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

    // pushes a limit of 10 onto a single-partition scan. the store's chunks hold
    // 9 rows each (for the full chunks), and the reader only stops between chunks,
    // so it reads two chunks (9 + 9) to reach the limit -> 18 rows.
    #[tokio::test]
    async fn limit_test() {
        let (wrapper, schema) =
            get_local_zarr_store(true, 0.0, "lat_lon_data_for_scan_limit").await;
        let path = wrapper.get_store_path();
        let table_url = ZarrUrlBuilder::try_new(ListingTableUrl::parse(path).unwrap(), None)
            .unwrap()
            .build()
            .unwrap();
        let config = ZarrTableConfig::new(table_url, schema);

        let session = SessionContext::new();
        let scan = ZarrScan::new(config, None).with_limit(10);
        let records: Vec<_> = scan
            .execute(0, session.task_ctx())
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        let total: usize = records.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 18);
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
