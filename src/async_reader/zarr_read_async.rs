// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use async_trait::async_trait;
use futures_util::{pin_mut, StreamExt};
use object_store::{path::Path, GetResultPayload, ObjectStore};
use std::collections::HashMap;
use std::fs::{read, read_to_string};
use std::sync::Arc;

use crate::reader::metadata::{ChunkPattern, ChunkSeparator};
use crate::reader::{ZarrError, ZarrResult};
use crate::reader::{ZarrInMemoryChunk, ZarrStoreMetadata};

/// A trait that exposes methods to get data from a zarr store asynchronously.
#[async_trait]
pub trait ZarrReadAsync<'a> {
    /// Method to retrieve the metadata from a zarr store asynchronously.
    async fn get_zarr_metadata(&self) -> ZarrResult<ZarrStoreMetadata>;

    /// Method to retrive the data in a zarr chunk asynchronously, which is really
    /// the data contained into one or more chunk files, one per zarr array in
    /// the store.
    async fn get_zarr_chunk(
        &self,
        position: &'a [usize],
        cols: &'a [String],
        real_dims: Vec<usize>,
        patterns: HashMap<String, ChunkPattern>,
    ) -> ZarrResult<ZarrInMemoryChunk>;
}

/// A wrapper around a pointer to an [`ObjectStore`] an a path that points
/// to a zarr store.
#[derive(Debug, Clone)]
pub struct ZarrPath {
    store: Arc<dyn ObjectStore>,
    location: Path,
}

impl ZarrPath {
    pub fn new(store: Arc<dyn ObjectStore>, location: Path) -> Self {
        Self { store, location }
    }
}

/// Implementation of the [`ZarrReadAsync`] trait for a [`ZarrPath`] which contains the
/// object store that the data will be read from.
#[async_trait]
impl<'a> ZarrReadAsync<'a> for ZarrPath {
    async fn get_zarr_metadata(&self) -> ZarrResult<ZarrStoreMetadata> {
        let mut meta = ZarrStoreMetadata::new();
        let stream = self.store.list(Some(&self.location));

        pin_mut!(stream);
        while let Some(p) = stream.next().await {
            let p = p?.location;
            if let Some(s) = p.filename() {
                if s == ".zarray" || s == "zarr.json" {
                    if let Some(mut dir_name) = p.prefix_match(&self.location) {
                        let array_name = dir_name.next().unwrap().as_ref().to_string();
                        let get_res = self.store.get(&p).await?;
                        let meta_str = match get_res.payload {
                            GetResultPayload::File(_, p) => read_to_string(p)?,
                            GetResultPayload::Stream(_) => {
                                std::str::from_utf8(&get_res.bytes().await?)?.to_string()
                            }
                        };
                        meta.add_column(array_name, &meta_str)?;
                    }
                }
            }
        }

        if meta.get_num_columns() == 0 {
            return Err(ZarrError::InvalidMetadata(
                "Could not find valid metadata in zarr store".to_string(),
            ));
        }
        Ok(meta)
    }

    async fn get_zarr_chunk(
        &self,
        position: &'a [usize],
        cols: &'a [String],
        real_dims: Vec<usize>,
        patterns: HashMap<String, ChunkPattern>,
    ) -> ZarrResult<ZarrInMemoryChunk> {
        let mut chunk = ZarrInMemoryChunk::new(real_dims);
        for var in cols {
            let s: Vec<String> = position.iter().map(|i| i.to_string()).collect();
            let pattern = patterns
                .get(var.as_str())
                .ok_or(ZarrError::InvalidMetadata(
                    "Could not find separator for column".to_string(),
                ))?;

            let p = match pattern {
                ChunkPattern {
                    separator: sep,
                    c_prefix: false,
                } => match sep {
                    ChunkSeparator::Period => {
                        self.location.child(var.to_string()).child(s.join("."))
                    }
                    ChunkSeparator::Slash => {
                        let mut path = self.location.child(var.to_string());
                        for idx in s {
                            path = path.child(idx);
                        }
                        path
                    }
                },
                ChunkPattern {
                    separator: sep,
                    c_prefix: true,
                } => match sep {
                    ChunkSeparator::Period => self
                        .location
                        .child(var.to_string())
                        .child("c.".to_string() + &s.join(".")),
                    ChunkSeparator::Slash => {
                        let mut path = self.location.child(var.to_string()).child("c");
                        for idx in s {
                            path = path.child(idx);
                        }
                        path
                    }
                },
            };
            let get_res = self.store.get(&p).await?;
            let data = match get_res.payload {
                GetResultPayload::File(_, p) => read(p)?,
                GetResultPayload::Stream(_) => get_res.bytes().await?.to_vec(),
            };
            chunk.add_array(var.to_string(), data);
        }

        Ok(chunk)
    }
}

#[cfg(test)]
mod zarr_read_async_tests {
    use object_store::{local::LocalFileSystem, path::Path};
    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::sync::Arc;

    use super::*;
    use crate::reader::codecs::{Endianness, ZarrCodec, ZarrDataType};
    use crate::reader::metadata::{ChunkSeparator, ZarrArrayMetadata};
    use crate::reader::ZarrProjection;

    fn get_test_data_file_system() -> LocalFileSystem {
        LocalFileSystem::new_with_prefix(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-data/data/zarr/v2_data"),
        )
        .unwrap()
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
                ChunkPattern {
                    separator: ChunkSeparator::Period,
                    c_prefix: false
                },
                None,
                vec![ZarrCodec::Bytes(Endianness::Little)],
            )
        );
        assert_eq!(
            meta.get_array_meta("float_data").unwrap(),
            &ZarrArrayMetadata::new(
                2,
                ZarrDataType::Float(8),
                ChunkPattern {
                    separator: ChunkSeparator::Period,
                    c_prefix: false
                },
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
        let chunk = store
            .get_zarr_chunk(
                &pos,
                meta.get_columns(),
                meta.get_real_dims(&pos),
                meta.get_chunk_patterns(),
            )
            .await
            .unwrap();
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
        let chunk = store
            .get_zarr_chunk(
                &pos,
                &cols,
                meta.get_real_dims(&pos),
                meta.get_chunk_patterns(),
            )
            .await
            .unwrap();
        assert_eq!(
            chunk.data.keys().collect::<Vec<&String>>(),
            vec!["byte_data"]
        );

        // same as above, but specify columsn to keep instead of to skip
        let col_proj = ZarrProjection::keep(vec!["float_data".to_string()]);
        let cols = col_proj.apply_selection(meta.get_columns()).unwrap();
        let chunk = store
            .get_zarr_chunk(
                &pos,
                &cols,
                meta.get_real_dims(&pos),
                meta.get_chunk_patterns(),
            )
            .await
            .unwrap();
        assert_eq!(
            chunk.data.keys().collect::<Vec<&String>>(),
            vec!["float_data"]
        );
    }
}
