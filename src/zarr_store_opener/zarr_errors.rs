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

use std::error::Error;

use arrow::error::ArrowError;
use zarrs::array::codec::CodecError;
use zarrs::array::{ArrayCreateError, ArrayError};
use zarrs_storage::{StorageError, StorePrefixError};

#[derive(Debug)]
pub enum ZarrQueryError {
    InvalidColumnRequest(String),
    InvalidType(String),
    InvalidMetadata(String),
    InvalidCompute(String),
    Zarrs(Box<dyn Error + Send + Sync>),
    External(Box<dyn Error + Send + Sync>),
}

impl std::fmt::Display for ZarrQueryError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        match &self {
            Self::InvalidColumnRequest(msg) => write!(fmt, "Invalid projection: {msg}"),
            Self::InvalidType(msg) => write!(fmt, "Invaild type: {msg}"),
            Self::InvalidMetadata(msg) => write!(fmt, "Invaild meta data: {msg}"),
            Self::InvalidCompute(msg) => write!(fmt, "Invaild compute: {msg}"),
            Self::Zarrs(e) => write!(fmt, "A zarrs call returned an error: {e}"),
            Self::External(e) => write!(fmt, "External error: {e}"),
        }
    }
}

impl Error for ZarrQueryError {}

/// [`ZarrQueryError`]` from various extrenal errors.
impl From<StorageError> for ZarrQueryError {
    fn from(e: StorageError) -> ZarrQueryError {
        ZarrQueryError::Zarrs(Box::new(e))
    }
}

impl From<StorePrefixError> for ZarrQueryError {
    fn from(e: StorePrefixError) -> ZarrQueryError {
        ZarrQueryError::Zarrs(Box::new(e))
    }
}

impl From<ArrayCreateError> for ZarrQueryError {
    fn from(e: ArrayCreateError) -> ZarrQueryError {
        ZarrQueryError::Zarrs(Box::new(e))
    }
}

impl From<CodecError> for ZarrQueryError {
    fn from(e: CodecError) -> ZarrQueryError {
        ZarrQueryError::Zarrs(Box::new(e))
    }
}

impl From<ArrayError> for ZarrQueryError {
    fn from(e: ArrayError) -> ZarrQueryError {
        ZarrQueryError::Zarrs(Box::new(e))
    }
}

impl From<ArrowError> for ZarrQueryError {
    fn from(e: ArrowError) -> ZarrQueryError {
        ZarrQueryError::External(Box::new(e))
    }
}

impl From<std::io::Error> for ZarrQueryError {
    fn from(e: std::io::Error) -> ZarrQueryError {
        ZarrQueryError::External(Box::new(e))
    }
}

/// A specialized [`Result`] for [`ZarrError`]s.
pub type ZarrQueryResult<T, E = ZarrQueryError> = Result<T, E>;

/// [`ArrowError`]` from a [`ZarrQueryError`]`
impl From<ZarrQueryError> for ArrowError {
    fn from(e: ZarrQueryError) -> ArrowError {
        ArrowError::ExternalError(Box::new(e))
    }
}
