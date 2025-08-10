use std::fmt;
use std::sync::Arc;
use zarrs_storage::AsyncReadableListableStorageTraits;
/// Configuration for Zarr DataFusion processing.
pub struct ZarrConfig {
    /// The zarr store.
    pub zarr_store: Arc<dyn AsyncReadableListableStorageTraits>,

    /// The projection for the scan.
    pub projection: Option<Vec<usize>>,
}

impl fmt::Debug for ZarrConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ZarrConfig")
            .field("projection", &self.projection)
            .finish()
    }
}

impl ZarrConfig {
    /// Create a new ZarrConfig.
    pub fn new(zarr_store: Arc<dyn AsyncReadableListableStorageTraits>) -> Self {
        Self {
            zarr_store,
            projection: None,
        }
    }

    /// Set the projection for the scan.
    pub fn with_projection(mut self, projection: Option<Vec<usize>>) -> Self {
        self.projection = projection;
        self
    }
}
