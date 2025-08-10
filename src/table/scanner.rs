use super::config::ZarrConfig;
use crate::zarr_store_opener::zarr_data_stream::ZarrRecordBatchStream;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::{DisplayAs, DisplayFormatType};
use datafusion::physical_plan::{
    ExecutionPlan, PhysicalExpr, PlanProperties, RecordBatchStream, SendableRecordBatchStream,
};
use std::{any::Any, sync::Arc};
use zarrs_storage::AsyncReadableListableStorageTraits;
impl<T: AsyncReadableListableStorageTraits + Unpin + 'static> RecordBatchStream
    for ZarrRecordBatchStream<T>
{
    fn schema(&self) -> SchemaRef {
        self.get_schema_ref()
    }
}

#[derive(Debug)]
pub struct ZarrScan {
    zarr_config: ZarrConfig,
    projected_schema: SchemaRef,
    filters: Option<Arc<dyn PhysicalExpr>>,
    plan_properties: PlanProperties,
}

impl DisplayAs for ZarrScan {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "ZarrScan")
    }
}

// impl ExecutionPlan for ZarrScan {
//     fn name(&self) -> &str {
//         "ZarrScan"
//     }

//     fn as_any(&self) -> &dyn Any {
//         self
//     }

//     fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
//         vec![]
//     }

//     fn with_new_children(
//         self: Arc<Self>,
//         _children: Vec<Arc<dyn ExecutionPlan>>,
//     ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
//         Ok(self)
//     }

//     fn properties(&self) -> &PlanProperties {
//         &self.plan_properties
//     }

//     fn execute(
//         &self,
//         partition: usize,
//         context: Arc<datafusion::execution::TaskContext>,
//     ) -> datafusion::error::Result<SendableRecordBatchStream> {
//         let stream = ZarrRecordBatchStream::new(self.zarr_config.zarr_store.clone(), None, None);
//         Ok(Box::pin(stream) as SendableRecordBatchStream)
//     }
// }
