use datafusion::prelude::SessionContext;
use pyo3::prelude::*;

#[pyclass(
    name = "GeospatialSessionContext",
    module = "zarr_datafusion._internal"
)]
pub struct PyGeospatialSessionContext {
    #[allow(dead_code)] // retained for upcoming geospatial session methods
    ctx: SessionContext,
}

// #[pymethods]
// impl PyGeospatialSessionContext {
//     #[new]
//     #[allow(clippy::new_without_default)]
//     pub fn new() -> Self {
//         let state = SessionStateBuilder::new()
//             .with_default_features()
//             .with_physical_optimizer_rule(Arc::new(SpatialJoinPhysicalOptimizer))
//             .build();
//         let ctx = SessionContext::new_with_state(state);
//         ctx.register_udf(ScalarUDF::from(StWithinUdf::default()));
//         ctx.register_udf(ScalarUDF::from(StContainsUdf::default()));
//         Self { ctx }
//     }

//     pub fn register_parquet(&self, name: &str, path: &str) -> PyResult<()> {
//         let name = name.to_string();
//         let path = path.to_string();
//         get_tokio_runtime()
//             .block_on(async {
//                 self.ctx
//                     .register_parquet(&name, &path, Default::default())
//                     .await
//             })
//             .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
//     }

//     pub fn sql(&self, py: Python<'_>, query: &str) -> PyResult<Vec<Py<PyAny>>> {
//         let query = query.to_string();
//         let batches = py.allow_threads(|| {
//             get_tokio_runtime()
//                 .block_on(async {
//                     let df = self.ctx.sql(&query).await?;
//                     df.collect().await
//                 })
//                 .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
//         })?;
//         batches.iter().map(|b| b.to_pyarrow(py)).collect()
//     }
// }

#[pymodule]
fn _internal(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    //m.add_class::<PyGeospatialSessionContext>()?;
    Ok(())
}
