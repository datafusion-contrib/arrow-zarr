pub mod async_reader;
pub mod reader;

#[cfg(feature = "datafusion")]
pub mod datafusion;

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    pub(crate) fn get_test_data_path(zarr_store: String) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test-data/data/zarr/v2_data")
            .join(zarr_store)
    }
}
