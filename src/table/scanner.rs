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
    filters: Option<Arc<dyn PhysicalExpr>>,
    plan_properties: PlanProperties,
}

impl ZarrScan {
    pub(crate) fn new(zarr_config: ZarrConfig, filters: Option<Arc<dyn PhysicalExpr>>) -> Self {
        let schema_ref = Arc::new(zarr_config.schema.clone());
        let plan_properties = PlanProperties::new(
            EquivalenceProperties::new(schema_ref),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Self {
            zarr_config,
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
        let schema_ref = Arc::new(self.zarr_config.schema.clone());
        let schema_ref_for_future = schema_ref.clone();
        let stream: FileOpenFuture = Box::pin(async move {
            let inner_stream =
                ZarrRecordBatchStream::new(zarr_store, schema_ref_for_future, None, None)
                    .await
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
            Ok(inner_stream.boxed())
        });
        let stream = StreamWrapper::new(stream, schema_ref);

        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod scanner_tests {
    use std::collections::HashMap;

    use arrow::datatypes::Float64Type;
    use arrow_schema::DataType;
    use datafusion::prelude::SessionContext;
    use futures_util::TryStreamExt;

    use super::*;
    use crate::test_utils::{
        get_lat_lon_data_store, validate_names_and_types, validate_primitive_column,
    };

    #[tokio::test]
    async fn read_data_test() {
        let (wrapper, schema) = get_lat_lon_data_store(true, 0.0, "lat_lon_data_for_scan").await;
        let store = wrapper.get_store();
        let config = ZarrConfig::new(store, schema);

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
}
