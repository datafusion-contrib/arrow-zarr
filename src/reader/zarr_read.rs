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

use itertools::Itertools;
use std::collections::HashMap;
use std::fs::{read, read_to_string};
use std::path::PathBuf;

use crate::reader::metadata::{ChunkPattern, ChunkSeparator};
use crate::reader::ZarrStoreMetadata;
use crate::reader::{ZarrError, ZarrResult};

/// An in-memory representation of the data contained in one chunk
/// of one zarr array (a single variable).
#[derive(Debug, Clone)]
pub struct ZarrInMemoryArray {
    pub(crate) data: Vec<u8>,
}

impl ZarrInMemoryArray {
    pub(crate) fn new(data: Vec<u8>) -> Self {
        Self { data }
    }

    pub(crate) fn take_data(self) -> Vec<u8> {
        self.data
    }
}

/// An in-memory representation of the data contained in one chunk of one
/// whole zarr store with one or more zarr arrays (one or more variables).
#[derive(Debug, Default)]
pub struct ZarrInMemoryChunk {
    pub(crate) data: HashMap<String, ZarrInMemoryArray>,
    real_dims: Vec<usize>,
}

impl ZarrInMemoryChunk {
    pub(crate) fn new(real_dims: Vec<usize>) -> Self {
        Self {
            data: HashMap::new(),
            real_dims,
        }
    }

    pub(crate) fn add_array(&mut self, col_name: String, data: Vec<u8>) {
        self.data.insert(col_name, ZarrInMemoryArray::new(data));
    }

    pub(crate) fn get_cols_in_chunk(&self) -> Vec<String> {
        let mut cols = self.data.keys().map(|s| s.to_string()).collect_vec();

        // the sort below is important, because within a zarr store, the different arrays are
        // not ordered, so there is no predefined order for the different columns. we effectively
        // define one here, by ordering the columns alphabetically.
        cols.sort();
        cols
    }

    pub(crate) fn get_real_dims(&self) -> &Vec<usize> {
        &self.real_dims
    }

    pub(crate) fn take_array(&mut self, col: &str) -> ZarrResult<ZarrInMemoryArray> {
        self.data
            .remove(col)
            .ok_or(ZarrError::MissingArray(col.to_string()))
    }
}

#[derive(Clone, PartialEq, Debug)]
pub(crate) enum ProjectionType {
    Select,
    SelectByIndex,
    Skip,
    Null,
}

/// A structure to handle skipping or selecting specific columns (zarr arrays) from
/// a zarr store.
#[derive(Clone, Debug)]
pub struct ZarrProjection {
    projection_type: ProjectionType,
    col_names: Option<Vec<String>>,
    col_indices: Option<Vec<usize>>,
}

impl From<Vec<usize>> for ZarrProjection {
    fn from(indices: Vec<usize>) -> Self {
        Self::keep_by_index(indices)
    }
}

impl From<Option<&Vec<usize>>> for ZarrProjection {
    fn from(indices: Option<&Vec<usize>>) -> Self {
        match indices {
            Some(i) => Self::keep_by_index(i.to_vec()),
            None => Self::all(),
        }
    }
}

impl ZarrProjection {
    /// Create a projection that keeps all columns.
    pub fn all() -> Self {
        Self {
            projection_type: ProjectionType::Null,
            col_names: None,
            col_indices: None,
        }
    }

    /// Create a projection that skips certain columns (and keeps all the other columns).
    pub fn skip(col_names: Vec<String>) -> Self {
        Self {
            projection_type: ProjectionType::Skip,
            col_names: Some(col_names),
            col_indices: None,
        }
    }

    /// Create a projection that keeps certain columns (and skips all the other columns).
    pub fn keep(col_names: Vec<String>) -> Self {
        Self {
            projection_type: ProjectionType::Select,
            col_names: Some(col_names),
            col_indices: None,
        }
    }

    /// Create a projection that keeps certain columns by index (and skips all the other columns).
    pub fn keep_by_index(col_indices: Vec<usize>) -> Self {
        Self {
            projection_type: ProjectionType::SelectByIndex,
            col_names: None,
            col_indices: Some(col_indices),
        }
    }

    pub(crate) fn apply_selection(&self, all_cols: &[String]) -> ZarrResult<Vec<String>> {
        match self.projection_type {
            ProjectionType::Null => Ok(all_cols.to_owned()),
            ProjectionType::Skip => {
                let col_names = self.col_names.as_ref().unwrap();
                return Ok(all_cols
                    .iter()
                    .filter(|x| !col_names.contains(x))
                    .map(|x| x.to_string())
                    .collect());
            }
            ProjectionType::Select => {
                let col_names = self.col_names.as_ref().unwrap();
                for col in col_names {
                    if !all_cols.contains(col) {
                        return Err(ZarrError::MissingArray(
                            "Column in projection missing columns to select from".to_string(),
                        ));
                    }
                }
                Ok(col_names.clone())
            }
            ProjectionType::SelectByIndex => {
                let col_indices = self.col_indices.as_ref().unwrap();
                let col_names: Vec<String> = col_indices
                    .iter()
                    .map(|i| all_cols[*i].to_string())
                    .collect();
                Ok(col_names)
            }
        }
    }

    pub(crate) fn update(&mut self, other_proj: ZarrProjection) -> ZarrResult<()> {
        match (&self.projection_type, &other_proj.projection_type) {
            (_, ProjectionType::Null) => (),
            (ProjectionType::Null, _) => {
                self.projection_type = other_proj.projection_type;
                self.col_names = other_proj.col_names;
                self.col_indices = other_proj.col_indices;
            }
            (ProjectionType::SelectByIndex, ProjectionType::SelectByIndex) => {
                let mut indices = self.col_indices.take().ok_or(ZarrError::InvalidPredicate(
                    "ZarrProjection missing indices".to_string(),
                ))?;
                for i in other_proj.col_indices.ok_or(ZarrError::InvalidPredicate(
                    "ZarrProjection update missing indices".to_string(),
                ))? {
                    if !indices.contains(&i) {
                        indices.push(i);
                    }
                }
                self.col_indices = Some(indices);
            }
            (ProjectionType::Select, ProjectionType::Select) => {
                let mut col_names = self.col_names.take().ok_or(ZarrError::InvalidPredicate(
                    "ZarrProjection missing col_names".to_string(),
                ))?;
                for col in other_proj.col_names.ok_or(ZarrError::InvalidPredicate(
                    "ZarrProjection update missing col_names".to_string(),
                ))? {
                    if !col_names.contains(&col) {
                        col_names.push(col);
                    }
                }
                self.col_names = Some(col_names);
            }
            (ProjectionType::Skip, ProjectionType::Select)
            | (ProjectionType::Select, ProjectionType::Skip) => {
                let mut col_names = self.col_names.take().ok_or(ZarrError::InvalidPredicate(
                    "ZarrProjection missing col_names".to_string(),
                ))?;
                for col in other_proj.col_names.ok_or(ZarrError::InvalidPredicate(
                    "ZarrProjection update missing col_names".to_string(),
                ))? {
                    if let Some(index) = col_names.iter().position(|value| value == &col) {
                        col_names.remove(index);
                    }
                }
            }
            _ => {
                return Err(ZarrError::InvalidPredicate(
                    "Invalid ZarrProjection update".to_string(),
                ))
            }
        };

        Ok(())
    }
}

/// A trait that exposes methods to get data from a zarr store.
pub trait ZarrRead {
    /// Method to retrieve the metadata from a zarr store.
    fn get_zarr_metadata(&self) -> ZarrResult<ZarrStoreMetadata>;

    /// Method to retrive the data in a zarr chunk, which is really the data
    /// contained into one or more chunk files, one per zarr array in the store.
    fn get_zarr_chunk(
        &self,
        position: &[usize],
        cols: &[String],
        real_dims: Vec<usize>,
        patterns: HashMap<String, ChunkPattern>,
    ) -> ZarrResult<ZarrInMemoryChunk>;
}

/// Implementation of the [`ZarrRead`] trait for a [`PathBuf`] which contains the
/// path to a zarr store.
impl ZarrRead for PathBuf {
    fn get_zarr_metadata(&self) -> ZarrResult<ZarrStoreMetadata> {
        let mut meta = ZarrStoreMetadata::new();
        let dir = self.read_dir().unwrap();

        for dir_entry in dir {
            let dir_entry = dir_entry?;

            // try both v2 (.zarray) and v3 (zarr.json)
            let mut p = dir_entry.path().join(".zarray");
            if !p.exists() {
                p = dir_entry.path().join("zarr.json");
            }

            if p.exists() {
                let meta_str = read_to_string(p)?;
                meta.add_column(
                    dir_entry
                        .path()
                        .file_name()
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_string(),
                    &meta_str,
                )?;
            }
        }

        if meta.get_num_columns() == 0 {
            return Err(ZarrError::InvalidMetadata(
                "Could not find valid metadata in zarr store".to_string(),
            ));
        }
        Ok(meta)
    }

    fn get_zarr_chunk(
        &self,
        position: &[usize],
        cols: &[String],
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

            let chunk_file = match pattern {
                ChunkPattern {
                    separator: sep,
                    c_prefix: false,
                } => match sep {
                    ChunkSeparator::Period => s.join("."),
                    ChunkSeparator::Slash => s.join("/"),
                },
                ChunkPattern {
                    separator: sep,
                    c_prefix: true,
                } => match sep {
                    ChunkSeparator::Period => "c.".to_string() + &s.join("."),
                    ChunkSeparator::Slash => "c/".to_string() + &s.join("/"),
                },
            };

            let path = self.join(var).join(chunk_file);

            if !path.exists() {
                return Err(ZarrError::MissingChunk(position.to_vec()));
            }
            let data = read(path)?;
            chunk.add_array(var.to_string(), data);
        }

        Ok(chunk)
    }
}

#[cfg(test)]
mod zarr_read_tests {
    use std::collections::HashSet;
    use std::path::PathBuf;

    use super::*;
    use crate::reader::codecs::{Endianness, ZarrCodec, ZarrDataType};
    use crate::reader::metadata::{ChunkSeparator, ZarrArrayMetadata};

    fn get_test_data_path(zarr_store: String) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test-data/data/zarr/v2_data")
            .join(zarr_store)
    }

    // read the store metadata, given a path to a zarr store.
    #[test]
    fn read_metadata() {
        let p = get_test_data_path("raw_bytes_example.zarr".to_string());
        let meta = p.get_zarr_metadata().unwrap();

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

    // read the raw data contained into a zarr store. one of the variables contains
    // byte data, which we explicitly check here.
    #[test]
    fn read_raw_chunks() {
        let p = get_test_data_path("raw_bytes_example.zarr".to_string());
        let meta = p.get_zarr_metadata().unwrap();

        // test read from an array where the data is just raw bytes
        let pos = vec![1, 2];
        let chunk = p
            .get_zarr_chunk(
                &pos,
                meta.get_columns(),
                meta.get_real_dims(&pos),
                meta.get_chunk_patterns(),
            )
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
        let chunk = p
            .get_zarr_chunk(
                &pos,
                &cols,
                meta.get_real_dims(&pos),
                meta.get_chunk_patterns(),
            )
            .unwrap();
        assert_eq!(
            chunk.data.keys().collect::<Vec<&String>>(),
            vec!["byte_data"]
        );

        // same as above, but specify columsn to keep instead of to skip
        let col_proj = ZarrProjection::keep(vec!["float_data".to_string()]);
        let cols = col_proj.apply_selection(meta.get_columns()).unwrap();
        let chunk = p
            .get_zarr_chunk(
                &pos,
                &cols,
                meta.get_real_dims(&pos),
                meta.get_chunk_patterns(),
            )
            .unwrap();
        assert_eq!(
            chunk.data.keys().collect::<Vec<&String>>(),
            vec!["float_data"]
        );
    }
}
