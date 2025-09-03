use arrow_schema::SchemaRef;
use std::fmt;
use std::sync::Arc;
use zarrs_storage::AsyncReadableListableStorageTraits;

/// Configuration for Zarr DataFusion processing.
#[derive(Clone)]
pub struct ZarrConfig {
    /// The zarr store.
    pub zarr_store: Arc<dyn AsyncReadableListableStorageTraits + Unpin + Send>,

    /// The schema for the entire table (regardless of what columns
    /// are selected).
    pub schema: SchemaRef,

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
    pub fn new(
        zarr_store: Arc<dyn AsyncReadableListableStorageTraits + Unpin + Send>,
        schema: SchemaRef,
    ) -> Self {
        Self {
            zarr_store,
            schema,
            projection: None,
        }
    }

    /// Set the projection for the scan.
    pub fn with_projection(mut self, projection: Vec<usize>) -> Self {
        self.projection = Some(projection);
        self
    }
}
