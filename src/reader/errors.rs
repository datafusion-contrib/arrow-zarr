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

use arrow_schema::ArrowError;
#[cfg(feature = "datafusion")]
use datafusion::error::DataFusionError;
#[cfg(all(feature = "io-uring", target_os = "linux"))]
use io_uring::squeue::PushError;
use object_store::Error as ObjStoreError;
use rayon::ThreadPoolBuildError;
use std::error::Error;
use std::ffi::NulError;
use std::io;
use std::result::Result;
use std::str::Utf8Error;

/// An error enumeration for the zarr operations in the crate.
#[derive(Debug)]
pub enum ZarrError {
    InvalidMetadata(String),
    InvalidPredicate(String),
    MissingChunk(Vec<usize>),
    MissingArray(String),
    Read(String),
    InvalidChunkRange(usize, usize, usize),
    Io(Box<dyn Error + Send + Sync>),
    Arrow(Box<dyn Error + Send + Sync>),
    DataFusion(Box<dyn Error + Send + Sync>),
    ObjectStore(Box<dyn Error + Send + Sync>),
    #[cfg(all(feature = "io-uring", target_os = "linux"))]
    IoUring(Box<dyn Error + Send + Sync>),
    Rayon(Box<dyn Error + Send + Sync>),
    Ffi(Box<dyn Error + Send + Sync>),
}

impl std::fmt::Display for ZarrError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        match &self {
            ZarrError::InvalidMetadata(msg) => write!(fmt, "Invalid zarr metadata: {msg}"),
            ZarrError::InvalidPredicate(msg) => write!(fmt, "Invalid zarr predicate: {msg}"),
            ZarrError::MissingChunk(pos) => {
                let s: Vec<String> = pos.iter().map(|i| i.to_string()).collect();
                let s = s.join(".");
                write!(fmt, "Missing zarr chunk file: {s}")
            }
            ZarrError::MissingArray(arr_name) => write!(fmt, "Missing zarr chunk file: {arr_name}"),
            ZarrError::Read(msg) => write!(fmt, "error while reading files: {msg}"),
            ZarrError::InvalidChunkRange(start, end, l) => {
                write!(
                    fmt,
                    "Invalid chunk range {start}, {end} with store on length {l}"
                )
            }
            ZarrError::Io(e) => write!(fmt, "IO error: {e}"),
            ZarrError::Arrow(e) => write!(fmt, "Arrow error: {e}"),
            ZarrError::ObjectStore(e) => write!(fmt, "ObjectStore error: {e}"),
            ZarrError::DataFusion(e) => write!(fmt, "DataFusion error: {e}"),
            #[cfg(all(feature = "io-uring", target_os = "linux"))]
            ZarrError::IoUring(e) => write!(fmt, "IoUring error: {e}"),
            ZarrError::Rayon(e) => write!(fmt, "Rayon error: {e}"),
            ZarrError::Ffi(e) => write!(fmt, "Ffi error: {e}"),
        }
    }
}

impl Error for ZarrError {}

impl From<io::Error> for ZarrError {
    fn from(e: io::Error) -> ZarrError {
        ZarrError::Io(Box::new(e))
    }
}

impl From<ArrowError> for ZarrError {
    fn from(e: ArrowError) -> ZarrError {
        ZarrError::Arrow(Box::new(e))
    }
}

impl From<ObjStoreError> for ZarrError {
    fn from(e: ObjStoreError) -> ZarrError {
        ZarrError::ObjectStore(Box::new(e))
    }
}

#[cfg(all(feature = "io-uring", target_os = "linux"))]
impl From<PushError> for ZarrError {
    fn from(e: PushError) -> ZarrError {
        ZarrError::IoUring(Box::new(e))
    }
}

impl From<Utf8Error> for ZarrError {
    fn from(e: Utf8Error) -> ZarrError {
        ZarrError::InvalidMetadata(e.to_string())
    }
}

impl From<ThreadPoolBuildError> for ZarrError {
    fn from(e: ThreadPoolBuildError) -> ZarrError {
        ZarrError::Rayon(Box::new(e))
    }
}

impl From<NulError> for ZarrError {
    fn from(e: NulError) -> ZarrError {
        ZarrError::Ffi(Box::new(e))
    }
}

#[cfg(feature = "datafusion")]
impl From<DataFusionError> for ZarrError {
    fn from(e: DataFusionError) -> ZarrError {
        ZarrError::DataFusion(Box::new(e))
    }
}

/// A specialized [`Result`] for [`ZarrError`]s.
pub type ZarrResult<T, E = ZarrError> = Result<T, E>;

// a helper to raise invalid metadata errors
pub(crate) fn throw_invalid_meta(err_str: &str) -> ZarrError {
    ZarrError::InvalidMetadata(err_str.to_string())
}
