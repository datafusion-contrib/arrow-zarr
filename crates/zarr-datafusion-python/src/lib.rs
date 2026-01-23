use std::ffi::CString;
use std::sync::Arc;

use arrow_zarr::table::ZarrTable;
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

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyZarrTableProvider>()?;
    m.add_class::<PyIcechunkTableProvider>()?;
    Ok(())
}
