use std::ffi::CString;
use std::sync::Arc;

use arrow_zarr::table::ZarrTable;
use datafusion::prelude::SessionContext;
use datafusion_ffi::table_provider::FFI_TableProvider;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

fn get_tokio_runtime() -> &'static tokio::runtime::Runtime {
    use std::sync::OnceLock;
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime"))
}

#[pyclass(name = "ZarrTableProvider", module = "zarr_datafusion._internal")]
pub struct PyZarrTableProvider {
    table: Arc<ZarrTable>,
}

#[pymethods]
impl PyZarrTableProvider {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        let path = path.to_string();
        let table = get_tokio_runtime().block_on(async { ZarrTable::from_path(path).await });
        Ok(Self {
            table: Arc::new(table),
        })
    }

    fn __datafusion_table_provider__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let name = CString::new("datafusion_table_provider").unwrap();
        let runtime_handle = get_tokio_runtime().handle().clone();
        let provider = FFI_TableProvider::new(self.table.clone(), true, Some(runtime_handle));
        PyCapsule::new(py, provider, Some(name))
    }
}

#[pyclass(name = "IcechunkTableProvider", module = "zarr_datafusion._internal")]
pub struct PyIcechunkTableProvider {
    table: Arc<ZarrTable>,
}

#[pymethods]
impl PyIcechunkTableProvider {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        let path = path.to_string();
        let table =
            get_tokio_runtime().block_on(async { ZarrTable::from_path_to_icechunk(path).await });
        Ok(Self {
            table: Arc::new(table),
        })
    }

    fn __datafusion_table_provider__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let name = CString::new("datafusion_table_provider").unwrap();
        let runtime_handle = get_tokio_runtime().handle().clone();
        let provider = FFI_TableProvider::new(self.table.clone(), true, Some(runtime_handle));
        PyCapsule::new(py, provider, Some(name))
    }
}

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
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyZarrTableProvider>()?;
    m.add_class::<PyIcechunkTableProvider>()?;
    //m.add_class::<PyGeospatialSessionContext>()?;
    Ok(())
}
