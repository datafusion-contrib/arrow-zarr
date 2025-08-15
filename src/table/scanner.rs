use super::config::ZarrConfig;
use crate::zarr_store_opener::zarr_data_stream::ZarrRecordBatchStream;
use arrow::record_batch::RecordBatch;
use arrow_schema::ArrowError;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::datasource::physical_plan::FileOpenFuture;
use datafusion::error::DataFusionError;
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::{DisplayAs, DisplayFormatType};
use datafusion::physical_plan::{
    ExecutionPlan, PhysicalExpr, PlanProperties, RecordBatchStream, SendableRecordBatchStream,
};
use futures::{ready, stream::BoxStream, FutureExt, StreamExt};
use futures_util::{
    task::{Context, Poll},
    Stream,
};
use std::{any::Any, pin::Pin, sync::Arc};

enum StreamWrapperState {
    OpeningInnerStream(FileOpenFuture),
    InnerStreamReady(BoxStream<'static, Result<RecordBatch, ArrowError>>),
    Error,
}
struct StreamWrapper {
    schema_ref: SchemaRef,
    state: StreamWrapperState,
}

impl StreamWrapper {
    fn new(future: FileOpenFuture, schema_ref: SchemaRef) -> Self {
        Self {
            schema_ref,
            state: StreamWrapperState::OpeningInnerStream(future),
        }
    }
}

impl Stream for StreamWrapper {
    type Item = Result<RecordBatch, DataFusionError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match &mut self.state {
                StreamWrapperState::OpeningInnerStream(future) => {
                    match ready!(future.poll_unpin(cx)) {
                        Ok(stream) => {
                            self.state = StreamWrapperState::InnerStreamReady(stream);
                        }
                        Err(e) => {
                            return Poll::Ready(Some(Err(e)));
                        }
                    }
                }
                StreamWrapperState::InnerStreamReady(stream) => {
                    match ready!(stream.poll_next_unpin(cx)) {
                        Some(Ok(batch)) => return Poll::Ready(Some(Ok(batch))),
                        Some(Err(e)) => {
                            self.state = StreamWrapperState::Error;
                            return Poll::Ready(Some(Err(e.into())));
                        }
                        None => return Poll::Ready(None),
                    }
                }
                StreamWrapperState::Error => return Poll::Ready(None),
            }
        }
    }
}

impl RecordBatchStream for StreamWrapper {
    fn schema(&self) -> SchemaRef {
        self.schema_ref.clone()
    }
}

#[derive(Debug)]
pub struct ZarrScan {
    zarr_config: ZarrConfig,
    projected_schema: SchemaRef,
    filters: Option<Arc<dyn PhysicalExpr>>,
    plan_properties: PlanProperties,
}

impl ZarrScan {
    fn new(
        zarr_config: ZarrConfig,
        filters: Option<Arc<dyn PhysicalExpr>>,
        schema_ref: SchemaRef,
    ) -> Self {
        let plan_properties = PlanProperties::new(
            EquivalenceProperties::new(schema_ref.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Self {
            zarr_config,
            projected_schema: schema_ref,
            filters,
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

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> datafusion::error::Result<SendableRecordBatchStream> {
        let zarr_store = self.zarr_config.zarr_store.clone();
        let stream: FileOpenFuture = Box::pin(async move {
            let inner_stream = ZarrRecordBatchStream::new(zarr_store, None, None)
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            Ok(inner_stream.boxed())
        });
        let stream = StreamWrapper::new(stream, self.projected_schema.clone());

        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod scanner_tests {
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::SessionContext;

    use super::*;
    use crate::test_utils::{
        validate_names_and_types, validate_primitive_column, write_1D_float_array,
        write_2D_float_array, StoreWrapper,
    };

    async fn get_lat_lon_data_store(
        write_data: bool,
        fillvalue: f64,
        dir_name: &str,
    ) -> StoreWrapper {
        let wrapper = StoreWrapper::new(dir_name.into());
        let store = wrapper.get_store();

        let lats = vec![35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0];
        write_1D_float_array(
            lats,
            8,
            3,
            store.clone(),
            "/lat",
            Some(["lat".into()].to_vec()),
            true,
        )
        .await;

        let lons = vec![
            -120.0, -119.0, -118.0, -117.0, -116.0, -115.0, -114.0, -113.0,
        ];
        write_1D_float_array(
            lons,
            8,
            3,
            store.clone(),
            "/lon",
            Some(["lon".into()].to_vec()),
            true,
        )
        .await;

        let data: Option<Vec<_>> = if write_data {
            Some((0..64).map(|i| i as f64).collect())
        } else {
            None
        };
        write_2D_float_array(
            data,
            fillvalue,
            (8, 8),
            (3, 3),
            store.clone(),
            "/data",
            Some(["lat".into(), "lon".into()].to_vec()),
            true,
        )
        .await;

        wrapper
    }

    #[tokio::test]
    async fn read_data_test() {
        let wrapper = get_lat_lon_data_store(true, 0.0, "lat_lon_data").await;
        let store = wrapper.get_store();
        let config = ZarrConfig::new(store);
        let schema_ref = Arc::new(Schema::new(vec![
            Field::new("lat", DataType::Float64, false),
            Field::new("lon", DataType::Float64, false),
            Field::new("data", DataType::Float64, false),
        ]));

        let session = SessionContext::new();
        let scan = ZarrScan::new(config, None, schema_ref);
        let records: Vec<_> = scan.execute(0, session.task_ctx()).unwrap().collect().await;

        println!("{:?}", records);
    }
}
