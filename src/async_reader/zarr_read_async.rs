use async_trait::async_trait;
use object_store::{ObjectStore, path::Path};
use std::sync::Arc;
use futures_util::{pin_mut, StreamExt};

use crate::reader::{ZarrInMemoryChunk, ZarrStoreMetadata};
use crate::reader::{ZarrResult, ZarrError};

/// A trait that exposes methods to get data from a zarr store asynchronously.
#[async_trait]
pub trait ZarrReadAsync {
    /// Method to retrieve the metadata from a zarr store asynchronously.
    async fn get_zarr_metadata(&self) -> ZarrResult<ZarrStoreMetadata>;
    
    /// Method to retrive the data in a zarr chunk asynchronously, which is really
    /// the data contained into one or more chunk files, one per zarr array in 
    /// the store.
    async fn get_zarr_chunk(
        &self,
        position: &Vec<usize>,
        cols: &Vec<String>,
        real_dims: Vec<usize>,
    ) -> ZarrResult<ZarrInMemoryChunk>;
}


/// A wrapper around a pointer to an [`ObjectStore`] an a path that points
/// to a zarr store.
#[derive(Debug, Clone)]
pub struct ZarrPath {
    store: Arc<dyn ObjectStore>,
    location: Path,
}

impl ZarrPath{
    pub fn new(store: Arc<dyn ObjectStore>, location: Path) -> Self {
        Self {store, location}
    }
}

#[async_trait]
impl ZarrReadAsync for ZarrPath {
    async fn get_zarr_metadata(&self) -> ZarrResult<ZarrStoreMetadata> {
        let mut meta = ZarrStoreMetadata::new();
        let stream = self.store.list(Some(&self.location));

        pin_mut!(stream);
        while let Some(p) = stream.next().await {
            let p = p?.location;
            if let Some(s) = p.filename() {
                if s == ".zarray"{
                    if let Some(mut dir_name) = p.prefix_match(&self.location) {
                        let array_name = dir_name.next().unwrap().as_ref().to_string();
                        let meta_bytes = self.store.get(&p).await?.bytes().await?;
                        let meta_str = std::str::from_utf8(&meta_bytes)?;
                        meta.add_column(array_name, meta_str)?;
                    }
                }
            }
        }

        if meta.get_num_columns() == 0 {
            return Err(ZarrError::InvalidMetadata("Could not find valid metadata in zarr store".to_string()))
        }
        Ok(meta)
    }

    async fn get_zarr_chunk(
        &self,
        position: &Vec<usize>,
        cols: &Vec<String>,
        real_dims: Vec<usize>,
    ) -> ZarrResult<ZarrInMemoryChunk> {
        let mut chunk = ZarrInMemoryChunk::new(real_dims);
        for var in cols {
            let s: Vec<String> = position.into_iter().map(|i| i.to_string()).collect();
            let s = s.join(".");

            let p = self.location.child(var.to_string()).child(s);
            let data = self.store.get(&p).await?.bytes().await?;
            chunk.add_array(var.to_string(), data.to_vec());
        }

        Ok(chunk)
    }
}



#[cfg(test)]
mod zarr_read_async_tests {
    use object_store::{path::Path, local::LocalFileSystem};
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::collections::HashSet;

    use super::*;
    use crate::reader::metadata::{ZarrArrayMetadata, ChunkSeparator};
    use crate::reader::codecs::{ZarrCodec, ZarrDataType, Endianness};
    use crate::reader::ZarrProjection;

    fn get_test_data_file_system() -> LocalFileSystem {
        LocalFileSystem::new_with_prefix(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testing/data/zarr/v2_data")
        ).unwrap()
    }

    #[tokio::test]
    async fn read_metadata() {
        let file_sys = get_test_data_file_system();
        let p = Path::parse("raw_bytes_example.zarr").unwrap();

        let store = ZarrPath::new(Arc::new(file_sys), p);
        let meta = store.get_zarr_metadata().await.unwrap();

        assert_eq!(meta.get_columns(), &vec!["byte_data", "float_data"]);
        assert_eq!(
            meta.get_array_meta("byte_data").unwrap(),
            &ZarrArrayMetadata::new(
                2,
                ZarrDataType::UInt(1),
                ChunkSeparator::Period,
                None,
                vec![ZarrCodec::Bytes(Endianness::Little)],
            )
        );
        assert_eq!(
            meta.get_array_meta("float_data").unwrap(),
            &ZarrArrayMetadata::new(
                2,
                ZarrDataType::Float(8),
                ChunkSeparator::Period,
                None,
                vec![ZarrCodec::Bytes(Endianness::Little)],
            )
        );
    }

    #[tokio::test]
    async fn read_raw_chunks() {
        let file_sys = get_test_data_file_system();
        let p = Path::parse("raw_bytes_example.zarr").unwrap();

        let store = ZarrPath::new(Arc::new(file_sys), p);
        let meta = store.get_zarr_metadata().await.unwrap();

        // test read from an array where the data is just raw bytes
        let pos = vec![1, 2];
        let chunk = store.get_zarr_chunk(
            &pos, meta.get_columns(), meta.get_real_dims(&pos)
        ).await.unwrap();
        assert_eq!(
            chunk.data.keys().collect::<HashSet<&String>>(),
            HashSet::from([&"float_data".to_string(), &"byte_data".to_string()])
        );
        assert_eq!(
            chunk.data.get("byte_data").unwrap().data,
            vec![33, 34, 35, 42, 43, 44, 51, 52, 53],
        );

        // test selecting only one of the 2 columns
        let col_proj = ZarrProjection::skip(vec!["float_data".to_string()]);
        let cols = col_proj.apply_selection(meta.get_columns()).unwrap();
        let chunk = store.get_zarr_chunk(&pos, &cols, meta.get_real_dims(&pos)).await.unwrap();
        assert_eq!(chunk.data.keys().collect::<Vec<&String>>(), vec!["byte_data"]);

        // same as above, but specify columsn to keep instead of to skip
        let col_proj = ZarrProjection::keep(vec!["float_data".to_string()]);
        let cols = col_proj.apply_selection(meta.get_columns()).unwrap();
        let chunk = store.get_zarr_chunk(
            &pos, &cols, meta.get_real_dims(&pos)).await.unwrap();
        assert_eq!(chunk.data.keys().collect::<Vec<&String>>(), vec!["float_data"]);
    }
}