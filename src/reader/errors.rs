use std::error::Error;
use std::io;
use std::result::Result;
use arrow_schema::ArrowError;
use object_store::Error as ObjStoreError;
use std::str::Utf8Error;

/// An error enumeration for the zarr operations in the crate.
#[derive(Debug)]
pub enum ZarrError {
    InvalidMetadata(String),
    InvalidPredicate(String),
    MissingChunk(Vec<usize>),
    MissingArray(String),
    InvalidChunkRange(usize, usize, usize),
    Io(Box<dyn Error + Send + Sync>),
    Arrow(Box<dyn Error + Send + Sync>),
    ObjectStore(Box<dyn Error + Send + Sync>),
}

impl std::fmt::Display for ZarrError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        match &self {
            ZarrError::InvalidMetadata(msg) => write!(fmt, "Invalid zarr metadata: {msg}"),
            ZarrError::InvalidPredicate(msg) => write!(fmt, "Invalid zarr predicate: {msg}"),
            ZarrError::MissingChunk(pos) => {
                let s: Vec<String> = pos.into_iter().map(|i| i.to_string()).collect();
                let s = s.join(".");
                write!(fmt, "Missing zarr chunk file: {s}")
            },
            ZarrError::MissingArray(arr_name) => write!(fmt, "Missing zarr chunk file: {arr_name}"),
            ZarrError::InvalidChunkRange(start, end, l) => {
                write!(fmt, "Invalid chunk range {start}, {end} with store on length {l}")
            },
            ZarrError::Io(e) => write!(fmt, "IO error: {e}"),
            ZarrError::Arrow(e) => write!(fmt, "Arrow error: {e}"),
            ZarrError::ObjectStore(e) => write!(fmt, "Arrow error: {e}"),
        }
    }
}

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

impl From<Utf8Error> for ZarrError {
    fn from(e: Utf8Error) -> ZarrError {
        ZarrError::InvalidMetadata(e.to_string())
    }
}

/// A specialized [`Result`] for [`ZarrError`]s.
pub type ZarrResult<T, E = ZarrError> = Result<T, E>;