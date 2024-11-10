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

//! A module tha provides an asychronous reader for zarr store, to generate [`RecordBatch`]es.
//!
//! ```
//! # #[tokio::main(flavor="current_thread")]
//! # async fn main() {
//! #
//! # use arrow_zarr::async_reader::{ZarrPath, ZarrRecordBatchStreamBuilder};
//! # use arrow_zarr::reader::ZarrProjection;
//! # use arrow_cast::pretty::pretty_format_batches;
//! # use arrow_array::RecordBatch;
//! # use object_store::{path::Path, local::LocalFileSystem};
//! # use std::path::PathBuf;
//! # use std::sync::Arc;
//! # use futures::stream::Stream;
//! # use futures_util::TryStreamExt;
//! #
//! # fn get_test_data_path(zarr_store: String) -> ZarrPath {
//! #   let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
//! #                   .join("test-data/data/zarr/v2_data")
//! #                   .join(zarr_store);
//! #   ZarrPath::new(
//! #       Arc::new(LocalFileSystem::new()),
//! #       Path::from_absolute_path(p).unwrap()
//! #   )
//! # }
//! #
//! # fn assert_batches_eq(batches: &[RecordBatch], expected_lines: &[&str]) {
//! #     let formatted = pretty_format_batches(batches).unwrap().to_string();
//! #     let actual_lines: Vec<_> = formatted.trim().lines().collect();
//! #     assert_eq!(
//! #          &actual_lines, expected_lines,
//! #          "\n\nexpected:\n\n{:#?}\nactual:\n\n{:#?}\n\n",
//! #          expected_lines, actual_lines
//! #      );
//! #  }
//!
//! // The ZarrReadAsync trait is implemented for the ZarrPath struct.
//! let p: ZarrPath = get_test_data_path("lat_lon_example.zarr".to_string());
//!
//! let proj = ZarrProjection::keep(vec!["lat".to_string(), "float_data".to_string()]);
//! let builder = ZarrRecordBatchStreamBuilder::new(p).with_projection(proj);
//! let mut stream = builder.build().await.unwrap();
//! let mut rec_batches: Vec<_> = stream.try_collect().await.unwrap();
//!
//! assert_batches_eq(
//!     &[rec_batches.remove(0)],
//!     &[
//!         "+------------+------+",
//!         "| float_data | lat  |",
//!         "+------------+------+",
//!         "| 1001.0     | 38.0 |",
//!         "| 1002.0     | 38.1 |",
//!         "| 1003.0     | 38.2 |",
//!         "| 1004.0     | 38.3 |",
//!         "| 1012.0     | 38.0 |",
//!         "| 1013.0     | 38.1 |",
//!         "| 1014.0     | 38.2 |",
//!         "| 1015.0     | 38.3 |",
//!         "| 1023.0     | 38.0 |",
//!         "| 1024.0     | 38.1 |",
//!         "| 1025.0     | 38.2 |",
//!         "| 1026.0     | 38.3 |",
//!         "| 1034.0     | 38.0 |",
//!         "| 1035.0     | 38.1 |",
//!         "| 1036.0     | 38.2 |",
//!         "| 1037.0     | 38.3 |",
//!         "+------------+------+",
//!     ],
//!  );
//! # }
//! ```

use arrow_array::{BooleanArray, RecordBatch};
use async_trait::async_trait;
use futures::stream::{BoxStream, Stream};
use futures::{ready, FutureExt};
use futures_util::future::BoxFuture;
use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::async_reader::io_uring_utils::WorkerPool;
use crate::reader::ZarrChunkFilter;
use crate::reader::{unwrap_or_return, ZarrIterator, ZarrRecordBatchReader};
use crate::reader::{ZarrError, ZarrResult};
use crate::reader::{ZarrInMemoryChunk, ZarrProjection, ZarrStoreMetadata};

pub use crate::async_reader::zarr_read_async::{ZarrPath, ZarrReadAsync};

pub mod io_uring_utils;
pub mod zarr_read_async;

const _IO_URING_SIZE: u32 = 64;
const _IO_URING_N_WORKERS: usize = 1;

/// A zarr store that holds an async reader for all the zarr data.
pub struct ZarrStoreAsync<T: for<'a> ZarrReadAsync<'a>> {
    meta: ZarrStoreMetadata,
    chunk_positions: Vec<Vec<usize>>,
    zarr_reader: T,
    projection: ZarrProjection,
    curr_chunk: usize,
    io_uring_worker_pool: WorkerPool,
    broadcastable_array_axes: HashMap<String, Option<usize>>,
}

impl<T: for<'a> ZarrReadAsync<'a>> ZarrStoreAsync<T> {
    async fn new(
        zarr_reader: T,
        chunk_positions: Vec<Vec<usize>>,
        projection: ZarrProjection,
    ) -> ZarrResult<Self> {
        let meta = zarr_reader.get_zarr_metadata().await?;
        let mut bdc_axes: HashMap<String, Option<usize>> = HashMap::new();
        for col in meta.get_columns() {
            let mut axis = None;
            if let Some(params) = meta.get_array_meta(col)?.get_ond_d_array_params() {
                axis = Some(params.1);
            }
            bdc_axes.insert(col.to_string(), axis);
        }
        Ok(Self {
            meta,
            chunk_positions,
            zarr_reader,
            projection,
            curr_chunk: 0,
            io_uring_worker_pool: WorkerPool::new(_IO_URING_SIZE, _IO_URING_N_WORKERS)?,
            broadcastable_array_axes: bdc_axes,
        })
    }
}

/// A trait exposing a method to asynchronously get zarr chunk data, but also to
/// skip a chunk if needed.
#[async_trait]
pub trait ZarrStream {
    async fn poll_next_chunk(&mut self) -> Option<ZarrResult<ZarrInMemoryChunk>>;
    fn skip_next_chunk(&mut self);
}

/// Implementation of the [`ZarrStream`] trait for the [`ZarrStoreAsync`] struct, which
/// itself holds an asynchronous reader for the zarr data.
#[async_trait]
impl<T> ZarrStream for ZarrStoreAsync<T>
where
    T: for<'a> ZarrReadAsync<'a> + Unpin + Send + 'static,
{
    async fn poll_next_chunk(&mut self) -> Option<ZarrResult<ZarrInMemoryChunk>> {
        if self.curr_chunk == self.chunk_positions.len() {
            return None;
        }

        let cols = self.projection.apply_selection(self.meta.get_columns());
        let cols = unwrap_or_return!(cols);

        let pos = &self.chunk_positions[self.curr_chunk];
        let chnk = self
            .zarr_reader
            .get_zarr_chunk(
                pos,
                &cols,
                self.meta.get_real_dims(pos),
                self.meta.get_chunk_patterns(),
                &mut self.io_uring_worker_pool,
                &self.broadcastable_array_axes,
            )
            .await;

        self.curr_chunk += 1;
        Some(chnk)
    }

    fn skip_next_chunk(&mut self) {
        if self.curr_chunk < self.chunk_positions.len() {
            self.curr_chunk += 1;
        }
    }
}

// a simple struct to expose the zarr iterator trait for a single,
// preprocessed in memory chunk.
struct ZarrInMemoryChunkContainer {
    data: ZarrInMemoryChunk,
    done: bool,
}

impl ZarrInMemoryChunkContainer {
    fn new(data: ZarrInMemoryChunk) -> Self {
        Self { data, done: false }
    }
}

impl ZarrIterator for ZarrInMemoryChunkContainer {
    fn next_chunk(&mut self) -> Option<ZarrResult<ZarrInMemoryChunk>> {
        if self.done {
            return None;
        }
        self.done = true;
        Some(Ok(std::mem::take(&mut self.data)))
    }

    fn skip_chunk(&mut self) {
        self.done = true;
    }
}

// struct to bundle the store and the chunk data it returns together
// in a future so that that future's lifetime is static.
struct ZarrStoreWrapper<T: ZarrStream> {
    store: T,
}

impl<T: ZarrStream> ZarrStoreWrapper<T> {
    fn new(store: T) -> Self {
        Self { store }
    }

    async fn get_next(mut self) -> (Self, Option<ZarrResult<ZarrInMemoryChunk>>) {
        let next = self.store.poll_next_chunk().await;
        (self, next)
    }
}
type StoreReadResults<T> = (ZarrStoreWrapper<T>, Option<ZarrResult<ZarrInMemoryChunk>>);

enum ZarrStreamState<T: ZarrStream> {
    Init,
    ReadingPredicateData(BoxFuture<'static, StoreReadResults<T>>),
    ProcessingPredicate(ZarrRecordBatchReader<ZarrInMemoryChunkContainer>),
    Reading(BoxFuture<'static, StoreReadResults<T>>),
    Decoding(ZarrRecordBatchReader<ZarrInMemoryChunkContainer>),
    Error,
}

/// A struct to read all the requested content from a zarr store, through the implementation
/// of the [`Stream`] trait, with [`Item = ZarrResult<RecordBatch>`]. Can only be created
/// through a [`ZarrRecordBatchStreamBuilder`]. The data is read asynchronously.
///
/// For a sync API see [`crate::reader::ZarrRecordBatchReader`].
pub struct ZarrRecordBatchStream<T: ZarrStream> {
    meta: ZarrStoreMetadata,
    filter: Option<ZarrChunkFilter>,
    state: ZarrStreamState<T>,

    // an option so that we can "take" the wrapper and bundle it
    // in a future when polling the stream.
    store_wrapper: Option<ZarrStoreWrapper<T>>,

    // this one is an option because it may or may not be present, not
    // just so that we can take it later (but it's useful for that too)
    predicate_store_wrapper: Option<ZarrStoreWrapper<T>>,
}

impl<T: ZarrStream> ZarrRecordBatchStream<T> {
    fn new(
        meta: ZarrStoreMetadata,
        zarr_store: T,
        filter: Option<ZarrChunkFilter>,
        mut predicate_store: Option<T>,
    ) -> Self {
        let mut predicate_store_wrapper = None;
        if predicate_store.is_some() {
            predicate_store_wrapper = Some(ZarrStoreWrapper::new(predicate_store.take().unwrap()));
        }
        Self {
            meta,
            filter,
            predicate_store_wrapper,
            store_wrapper: Some(ZarrStoreWrapper::new(zarr_store)),
            state: ZarrStreamState::Init,
        }
    }
}

const LOST_STORE_ERR: &str = "unexpectedly lost store wrapper in zarr record batch stream";
/// The [`Stream`] trait implementation for a [`ZarrRecordBatchStream`]. Provides the interface
/// through which the record batches can be retrieved.
impl<T> Stream for ZarrRecordBatchStream<T>
where
    T: ZarrStream + Unpin + Send + 'static,
{
    type Item = ZarrResult<RecordBatch>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match &mut self.state {
                ZarrStreamState::Init => {
                    if self.predicate_store_wrapper.is_none() {
                        let wrapper = self.store_wrapper.take().expect(LOST_STORE_ERR);
                        let fut = wrapper.get_next().boxed();
                        self.state = ZarrStreamState::Reading(fut);
                    } else {
                        let wrapper = self.predicate_store_wrapper.take().unwrap();
                        let fut = wrapper.get_next().boxed();
                        self.state = ZarrStreamState::ReadingPredicateData(fut);
                    }
                }
                ZarrStreamState::ReadingPredicateData(f) => {
                    let (wrapper, chunk) = ready!(f.poll_unpin(cx));
                    self.predicate_store_wrapper = Some(wrapper);

                    if let Some(chunk) = chunk {
                        if let Err(e) = chunk {
                            self.state = ZarrStreamState::Error;
                            return Poll::Ready(Some(Err(e)));
                        }

                        let chunk = chunk?;
                        let container = ZarrInMemoryChunkContainer::new(chunk);

                        if self.filter.is_none() {
                            self.state = ZarrStreamState::Error;
                            return Poll::Ready(Some(Err(ZarrError::InvalidMetadata(
                                "predicate store provided with no filter in zarr record batch stream"
                                    .to_string(),
                            ))));
                        }
                        let zarr_reader = ZarrRecordBatchReader::new(
                            self.meta.clone(),
                            None,
                            self.filter.as_ref().cloned(),
                            Some(container),
                        );
                        self.state = ZarrStreamState::ProcessingPredicate(zarr_reader);
                    } else {
                        // if the predicate store returns none, it's the end and it's
                        // time to return
                        self.state = ZarrStreamState::Init;
                        return Poll::Ready(None);
                    }
                }
                ZarrStreamState::ProcessingPredicate(reader) => {
                    // this call should always return something, we should never get a None because
                    // if we're here it means we provided a filter and some predicate data to evaluate.
                    let mask = reader
                        .next()
                        .expect("could not get mask in zarr record batch stream");
                    if let Err(e) = mask {
                        self.state = ZarrStreamState::Error;
                        return Poll::Ready(Some(Err(e)));
                    }

                    // here we know that mask will have a single boolean array column because of the
                    // way the reader was created in the previous state.
                    let mask = mask
                        .unwrap()
                        .column(0)
                        .as_any()
                        .downcast_ref::<BooleanArray>()
                        .expect("could not cast mask to boolean array in zarr record batch stream")
                        .clone();
                    if mask.true_count() == 0 {
                        self.store_wrapper
                            .as_mut()
                            .expect(LOST_STORE_ERR)
                            .store
                            .skip_next_chunk();
                        self.state = ZarrStreamState::Init;
                    } else {
                        let wrapper = self.store_wrapper.take().expect(LOST_STORE_ERR);
                        let fut = wrapper.get_next().boxed();
                        self.state = ZarrStreamState::Reading(fut);
                    }
                }
                ZarrStreamState::Reading(f) => {
                    let (wrapper, chunk) = ready!(f.poll_unpin(cx));
                    self.store_wrapper = Some(wrapper);

                    if let Some(chunk) = chunk {
                        if let Err(e) = chunk {
                            self.state = ZarrStreamState::Error;
                            return Poll::Ready(Some(Err(e)));
                        }

                        let chunk = chunk?;
                        let container = ZarrInMemoryChunkContainer::new(chunk);
                        let zarr_reader = ZarrRecordBatchReader::new(
                            self.meta.clone(),
                            Some(container),
                            None,
                            None,
                        );

                        self.state = ZarrStreamState::Decoding(zarr_reader);
                    } else {
                        // if store returns none, it's the end and it's time to return
                        self.state = ZarrStreamState::Init;
                        return Poll::Ready(None);
                    }
                }
                ZarrStreamState::Decoding(reader) => {
                    // this call should always return something, we should never get a None because
                    // if we're here it means we provided store with a zarr in memory chunk to the reader
                    let rec_batch = reader
                        .next()
                        .expect("could not get record batch in zarr record batch stream");

                    if let Err(e) = rec_batch {
                        self.state = ZarrStreamState::Error;
                        return Poll::Ready(Some(Err(e)));
                    }

                    self.state = ZarrStreamState::Init;
                    return Poll::Ready(Some(rec_batch));
                }
                ZarrStreamState::Error => return Poll::Ready(None),
            }
        }
    }
}

/// A builder used to construct a [`ZarrRecordBatchStream`] for a zarr store.
///
/// To build the equivalent synchronous reader see [`crate::reader::ZarrRecordBatchReaderBuilder`].
pub struct ZarrRecordBatchStreamBuilder<T: for<'a> ZarrReadAsync<'a> + Clone + Unpin + Send> {
    zarr_reader_async: T,
    projection: ZarrProjection,
    filter: Option<ZarrChunkFilter>,
}

impl<T: for<'a> ZarrReadAsync<'a> + Clone + Unpin + Send + 'static>
    ZarrRecordBatchStreamBuilder<T>
{
    /// Create a [`ZarrRecordBatchStreamBuilder`] from a [`ZarrReadAsync`] struct.
    pub fn new(zarr_reader_async: T) -> Self {
        Self {
            zarr_reader_async,
            projection: ZarrProjection::all(),
            filter: None,
        }
    }

    /// Adds a column projection to the builder, so that the resulting reader will only
    /// read some of the columns (zarr arrays) from the zarr store.
    pub fn with_projection(self, projection: ZarrProjection) -> Self {
        Self { projection, ..self }
    }

    /// Adds a row filter to the builder, so that the resulting reader will only
    /// read rows that satisfy some conditions from the zarr store.
    pub fn with_filter(self, filter: ZarrChunkFilter) -> Self {
        Self {
            filter: Some(filter),
            ..self
        }
    }

    /// Build a [`ZarrRecordBatchStream`], consuming the builder. The option range
    /// argument controls the start and end chunk (following the way zarr chunks are
    /// named and numbered).
    pub async fn build_partial_reader(
        self,
        chunk_range: Option<(usize, usize)>,
    ) -> ZarrResult<ZarrRecordBatchStream<ZarrStoreAsync<T>>> {
        let meta = self.zarr_reader_async.get_zarr_metadata().await?;
        let mut chunk_pos: Vec<Vec<usize>> = meta.get_chunk_positions();
        if let Some(chunk_range) = chunk_range {
            if (chunk_range.0 > chunk_range.1) | (chunk_range.1 > chunk_pos.len()) {
                return Err(ZarrError::InvalidChunkRange(
                    chunk_range.0,
                    chunk_range.1,
                    chunk_pos.len(),
                ));
            }
            chunk_pos = chunk_pos[chunk_range.0..chunk_range.1].to_vec();
        }

        let mut predicate_stream: Option<ZarrStoreAsync<T>> = None;
        if let Some(filter) = &self.filter {
            let predicate_proj = filter.get_all_projections()?;
            predicate_stream = Some(
                ZarrStoreAsync::new(
                    self.zarr_reader_async.clone(),
                    chunk_pos.clone(),
                    predicate_proj.clone(),
                )
                .await?,
            );
        }

        let zarr_stream =
            ZarrStoreAsync::new(self.zarr_reader_async, chunk_pos, self.projection.clone()).await?;
        Ok(ZarrRecordBatchStream::new(
            meta,
            zarr_stream,
            self.filter,
            predicate_stream,
        ))
    }

    /// Build a [`ZarrRecordBatchStream`], consuming the builder. The resulting reader
    /// will read all the chunks in the zarr store.
    pub async fn build(self) -> ZarrResult<ZarrRecordBatchStream<ZarrStoreAsync<T>>> {
        self.build_partial_reader(None).await
    }
}

//***********************************************************
// implementation of an async stream that doesn't block when decompressing chunks. for now, that doesn't
// to help at all, I need to revisit this at some point, won't be used for now. Also, for now filters are
// not supported, i needed to add a "wrapper" for the mask stream, that packagaes the stream and the returned
// chunk into a future, to avoid static lifetime issues.
//************************************************************

type InterleavedResults<T> = (
    Result<StoreReadResults<T>, tokio::task::JoinError>,
    Result<ZarrResult<RecordBatch>, tokio::task::JoinError>,
);
enum ZarrStreamStateNonBlocking<T: ZarrStream> {
    Init,
    Reading(BoxFuture<'static, StoreReadResults<T>>),
    _Processing(ZarrRecordBatchReader<ZarrInMemoryChunkContainer>),
    Interleaving(Option<ZarrRecordBatchReader<ZarrInMemoryChunkContainer>>),
    ProcessingInterleaved(BoxFuture<'static, InterleavedResults<T>>),
    Done,
    Error,
}

pub struct ZarrRecordBatchStreamNonBlocking<'a, T: ZarrStream> {
    meta: ZarrStoreMetadata,
    filter: Option<ZarrChunkFilter>,
    store_wrapper: Option<ZarrStoreWrapper<T>>,
    state: ZarrStreamStateNonBlocking<T>,

    // this is an optional record batch stream that will provide masks
    // to apply to the data.
    _mask_stream: Option<BoxStream<'a, ZarrResult<RecordBatch>>>,
    _mask: Option<BooleanArray>,
}

impl<'a, T: ZarrStream> ZarrRecordBatchStreamNonBlocking<'a, T> {
    fn new(meta: ZarrStoreMetadata, filter: Option<ZarrChunkFilter>, store: T) -> Self {
        Self {
            meta,
            filter,
            store_wrapper: Some(ZarrStoreWrapper::new(store)),
            state: ZarrStreamStateNonBlocking::Init,
            _mask_stream: None,
            _mask: None,
        }
    }
}

impl<'a, T> Stream for ZarrRecordBatchStreamNonBlocking<'a, T>
where
    T: ZarrStream + Unpin + Send + 'static,
{
    type Item = ZarrResult<RecordBatch>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match &mut self.state {
                ZarrStreamStateNonBlocking::Init => {
                    let wrapper = self.store_wrapper.take().expect(LOST_STORE_ERR);
                    let fut = wrapper.get_next().boxed();
                    self.state = ZarrStreamStateNonBlocking::Reading(fut);
                }
                ZarrStreamStateNonBlocking::Reading(f) => {
                    let (wrapper, chunk) = ready!(f.poll_unpin(cx));
                    self.store_wrapper = Some(wrapper);

                    // if store returns none, it's the end and it's time to return
                    if let Some(chunk) = chunk {
                        let chunk = chunk?;
                        let container = ZarrInMemoryChunkContainer::new(chunk);
                        let reader = ZarrRecordBatchReader::new(
                            self.meta.clone(),
                            Some(container),
                            None,
                            None,
                        );
                        self.state = ZarrStreamStateNonBlocking::Interleaving(Some(reader))
                    } else {
                        self.state = ZarrStreamStateNonBlocking::Done;
                        return Poll::Ready(None);
                    }
                }
                ZarrStreamStateNonBlocking::_Processing(reader) => {
                    let rec_batch = reader
                        .next()
                        .expect("could not get record batch in zarr record batch stream");

                    if let Err(e) = rec_batch {
                        self.state = ZarrStreamStateNonBlocking::Error;
                        return Poll::Ready(Some(Err(e)));
                    }

                    self.state = ZarrStreamStateNonBlocking::Init;
                    return Poll::Ready(Some(rec_batch));
                }
                ZarrStreamStateNonBlocking::Interleaving(reader) => {
                    let mut reader = reader.take().unwrap();
                    let wrapper = self.store_wrapper.take().unwrap();

                    let io_fut =
                        tokio::task::spawn(async move { wrapper.get_next().await }).boxed();
                    let compute_fut = tokio::task::spawn_blocking(move || {
                        reader
                            .next()
                            .expect("reader should always produce Some data here")
                    })
                    .boxed();
                    let fut = async { tokio::join!(io_fut, compute_fut) }.boxed();

                    self.state = ZarrStreamStateNonBlocking::ProcessingInterleaved(fut);
                }
                ZarrStreamStateNonBlocking::ProcessingInterleaved(f) => {
                    let (io_out, rec_batch) = ready!(f.poll_unpin(cx));
                    let (wrapper, chnk) = io_out.unwrap();
                    self.store_wrapper = Some(wrapper);

                    if let Some(chnk) = chnk {
                        let chnk = chnk?;
                        let container = ZarrInMemoryChunkContainer::new(chnk);
                        let reader = if self.filter.is_none() {
                            ZarrRecordBatchReader::new(
                                self.meta.clone(),
                                Some(container),
                                None,
                                None,
                            )
                        } else {
                            ZarrRecordBatchReader::new(
                                self.meta.clone(),
                                None,
                                self.filter.as_ref().cloned(),
                                Some(container),
                            )
                        };
                        self.state = ZarrStreamStateNonBlocking::Interleaving(Some(reader));
                    } else {
                        self.state = ZarrStreamStateNonBlocking::Done;
                    }

                    let rec_batch = rec_batch.unwrap();
                    return Poll::Ready(Some(rec_batch));
                }
                ZarrStreamStateNonBlocking::Done => {
                    return Poll::Ready(None);
                }
                ZarrStreamStateNonBlocking::Error => {
                    return Poll::Ready(None);
                }
            }
        }
    }
}

pub struct ZarrRecordBatchStreamBuilderNonBlocking<
    T: for<'a> ZarrReadAsync<'a> + Clone + Unpin + Send,
> {
    zarr_reader_async: T,
    projection: ZarrProjection,
    _filter: Option<ZarrChunkFilter>,
}

impl<T: for<'a> ZarrReadAsync<'a> + Clone + Unpin + Send + 'static>
    ZarrRecordBatchStreamBuilderNonBlocking<T>
{
    pub fn new(zarr_reader_async: T) -> Self {
        Self {
            zarr_reader_async,
            projection: ZarrProjection::all(),
            _filter: None,
        }
    }

    pub fn with_projection(self, projection: ZarrProjection) -> Self {
        Self { projection, ..self }
    }

    pub fn with_filter(self, filter: ZarrChunkFilter) -> Self {
        Self {
            _filter: Some(filter),
            ..self
        }
    }

    pub async fn build<'a>(
        self,
    ) -> ZarrResult<ZarrRecordBatchStreamNonBlocking<'a, ZarrStoreAsync<T>>> {
        let meta = self.zarr_reader_async.get_zarr_metadata().await?;
        let chunk_pos: Vec<Vec<usize>> = meta.get_chunk_positions();

        let zarr_stream =
            ZarrStoreAsync::new(self.zarr_reader_async, chunk_pos, self.projection.clone()).await?;
        Ok(ZarrRecordBatchStreamNonBlocking::new(
            meta,
            None,
            zarr_stream,
        ))
    }
}

#[cfg(test)]
mod zarr_async_reader_tests {
    use arrow::compute::kernels::cmp::{gt_eq, lt};
    use arrow_array::cast::AsArray;
    use arrow_array::types::*;
    use arrow_array::*;
    use arrow_schema::DataType;
    use futures_util::TryStreamExt;
    use itertools::enumerate;
    use object_store::{local::LocalFileSystem, path::Path};
    use std::sync::Arc;
    use std::{collections::HashMap, fmt::Debug};

    use super::*;
    use crate::async_reader::zarr_read_async::ZarrPath;
    use crate::reader::{ZarrArrowPredicate, ZarrArrowPredicateFn};
    use crate::tests::{get_test_v2_data_path, get_test_v3_data_path};

    fn get_v2_test_zarr_path(zarr_store: String) -> ZarrPath {
        ZarrPath::new(
            Arc::new(LocalFileSystem::new()),
            Path::from_absolute_path(get_test_v2_data_path(zarr_store)).unwrap(),
        )
    }

    fn validate_names_and_types(targets: &HashMap<String, DataType>, rec: &RecordBatch) {
        let mut target_cols: Vec<&String> = targets.keys().collect();
        let schema = rec.schema();
        let from_rec: Vec<&String> = schema.fields.iter().map(|f| f.name()).collect();

        target_cols.sort();
        assert_eq!(from_rec, target_cols);

        for field in schema.fields.iter() {
            assert_eq!(field.data_type(), targets.get(field.name()).unwrap());
        }
    }

    fn validate_bool_column(col_name: &str, rec: &RecordBatch, targets: &[bool]) {
        let mut matched = false;
        for (idx, col) in enumerate(rec.schema().fields.iter()) {
            if col.name().as_str() == col_name {
                assert_eq!(
                    rec.column(idx).as_boolean(),
                    &BooleanArray::from(targets.to_vec()),
                );
                matched = true;
            }
        }
        assert!(matched);
    }

    fn validate_primitive_column<T, U>(col_name: &str, rec: &RecordBatch, targets: &[U])
    where
        T: ArrowPrimitiveType,
        [U]: AsRef<[<T as arrow_array::ArrowPrimitiveType>::Native]>,
        U: Debug,
    {
        let mut matched = false;
        for (idx, col) in enumerate(rec.schema().fields.iter()) {
            if col.name().as_str() == col_name {
                assert_eq!(rec.column(idx).as_primitive::<T>().values(), targets,);
                matched = true;
            }
        }
        assert!(matched);
    }

    // create a test filter
    fn create_filter() -> ZarrChunkFilter {
        let mut filters: Vec<Box<dyn ZarrArrowPredicate>> = Vec::new();
        let f = ZarrArrowPredicateFn::new(
            ZarrProjection::keep(vec!["lat".to_string()]),
            move |batch| {
                gt_eq(
                    batch.column_by_name("lat").unwrap(),
                    &Scalar::new(&Float64Array::from(vec![38.6])),
                )
            },
        );
        filters.push(Box::new(f));
        let f = ZarrArrowPredicateFn::new(
            ZarrProjection::keep(vec!["lon".to_string()]),
            move |batch| {
                gt_eq(
                    batch.column_by_name("lon").unwrap(),
                    &Scalar::new(&Float64Array::from(vec![-109.7])),
                )
            },
        );
        filters.push(Box::new(f));
        let f = ZarrArrowPredicateFn::new(
            ZarrProjection::keep(vec!["lon".to_string()]),
            move |batch| {
                lt(
                    batch.column_by_name("lon").unwrap(),
                    &Scalar::new(&Float64Array::from(vec![-109.2])),
                )
            },
        );
        filters.push(Box::new(f));

        ZarrChunkFilter::new(filters)
    }

    #[tokio::test]
    async fn projection_tests() {
        let zp = get_v2_test_zarr_path("compression_example.zarr".to_string());
        let proj = ZarrProjection::keep(vec!["bool_data".to_string(), "int_data".to_string()]);
        let stream_builder = ZarrRecordBatchStreamBuilder::new(zp).with_projection(proj);

        let stream = stream_builder.build().await.unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        let target_types = HashMap::from([
            ("bool_data".to_string(), DataType::Boolean),
            ("int_data".to_string(), DataType::Int64),
        ]);

        // center chunk
        let rec = &records[4];
        validate_names_and_types(&target_types, rec);
        validate_bool_column(
            "bool_data",
            rec,
            &[false, true, false, false, true, false, false, true, false],
        );
        validate_primitive_column::<Int64Type, i64>(
            "int_data",
            rec,
            &[-4, -3, -2, 4, 5, 6, 12, 13, 14],
        );
    }

    #[tokio::test]
    async fn filters_tests() {
        let zp = get_v2_test_zarr_path("lat_lon_example.zarr".to_string());
        let stream_builder = ZarrRecordBatchStreamBuilder::new(zp).with_filter(create_filter());
        let stream = stream_builder.build().await.unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        let target_types = HashMap::from([
            ("lat".to_string(), DataType::Float64),
            ("lon".to_string(), DataType::Float64),
            ("float_data".to_string(), DataType::Float64),
        ]);

        // check the values in a chunk. the predicate pushdown only takes care of
        // skipping whole chunks, so there is no guarantee that the values in the
        // record batch fully satisfy the predicate, here we are only checking that
        // the first chunk that was read is the first one with some values that
        // satisfy the predicate.
        let rec = &records[0];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Float64Type, f64>(
            "lat",
            rec,
            &[
                38.4, 38.5, 38.6, 38.7, 38.4, 38.5, 38.6, 38.7, 38.4, 38.5, 38.6, 38.7, 38.4, 38.5,
                38.6, 38.7,
            ],
        );
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            rec,
            &[
                -110.0, -110.0, -110.0, -110.0, -109.9, -109.9, -109.9, -109.9, -109.8, -109.8,
                -109.8, -109.8, -109.7, -109.7, -109.7, -109.7,
            ],
        );
        validate_primitive_column::<Float64Type, f64>(
            "float_data",
            rec,
            &[
                1005.0, 1006.0, 1007.0, 1008.0, 1016.0, 1017.0, 1018.0, 1019.0, 1027.0, 1028.0,
                1029.0, 1030.0, 1038.0, 1039.0, 1040.0, 1041.0,
            ],
        );
    }

    #[tokio::test]
    async fn multiple_readers_tests() {
        let zp = get_v2_test_zarr_path("compression_example.zarr".to_string());
        let stream1 = ZarrRecordBatchStreamBuilder::new(zp.clone())
            .build_partial_reader(Some((0, 5)))
            .await
            .unwrap();
        let stream2 = ZarrRecordBatchStreamBuilder::new(zp)
            .build_partial_reader(Some((5, 9)))
            .await
            .unwrap();

        let records1: Vec<_> = stream1.try_collect().await.unwrap();
        let records2: Vec<_> = stream2.try_collect().await.unwrap();

        let target_types = HashMap::from([
            ("bool_data".to_string(), DataType::Boolean),
            ("uint_data".to_string(), DataType::UInt64),
            ("int_data".to_string(), DataType::Int64),
            ("float_data".to_string(), DataType::Float64),
            ("float_data_no_comp".to_string(), DataType::Float64),
        ]);

        // center chunk
        let rec = &records1[4];
        validate_names_and_types(&target_types, rec);
        validate_bool_column(
            "bool_data",
            rec,
            &[false, true, false, false, true, false, false, true, false],
        );
        validate_primitive_column::<Int64Type, i64>(
            "int_data",
            rec,
            &[-4, -3, -2, 4, 5, 6, 12, 13, 14],
        );
        validate_primitive_column::<UInt64Type, u64>(
            "uint_data",
            rec,
            &[27, 28, 29, 35, 36, 37, 43, 44, 45],
        );
        validate_primitive_column::<Float64Type, f64>(
            "float_data",
            rec,
            &[127., 128., 129., 135., 136., 137., 143., 144., 145.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "float_data_no_comp",
            rec,
            &[227., 228., 229., 235., 236., 237., 243., 244., 245.],
        );

        // bottom edge chunk
        let rec = &records2[2];
        validate_names_and_types(&target_types, rec);
        validate_bool_column("bool_data", rec, &[false, true, false, false, true, false]);
        validate_primitive_column::<Int64Type, i64>("int_data", rec, &[20, 21, 22, 28, 29, 30]);
        validate_primitive_column::<UInt64Type, u64>("uint_data", rec, &[51, 52, 53, 59, 60, 61]);
        validate_primitive_column::<Float64Type, f64>(
            "float_data",
            rec,
            &[151.0, 152.0, 153.0, 159.0, 160.0, 161.0],
        );
        validate_primitive_column::<Float64Type, f64>(
            "float_data_no_comp",
            rec,
            &[251.0, 252.0, 253.0, 259.0, 260.0, 261.0],
        );
    }

    #[tokio::test]
    async fn empty_query_tests() {
        let zp = get_v2_test_zarr_path("lat_lon_example.zarr".to_string());
        let mut builder = ZarrRecordBatchStreamBuilder::new(zp);

        // set a filter that will filter out all the data, there should be nothing left after
        // we apply it.
        let mut filters: Vec<Box<dyn ZarrArrowPredicate>> = Vec::new();
        let f = ZarrArrowPredicateFn::new(
            ZarrProjection::keep(vec!["lat".to_string()]),
            move |batch| {
                gt_eq(
                    batch.column_by_name("lat").unwrap(),
                    &Scalar::new(&Float64Array::from(vec![100.0])),
                )
            },
        );
        filters.push(Box::new(f));

        builder = builder.with_filter(ZarrChunkFilter::new(filters));
        let stream = builder.build().await.unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        // there should be no records, because of the filter.
        assert_eq!(records.len(), 0);
    }

    #[tokio::test]
    async fn array_broadcast_tests() {
        // reference that doesn't broadcast a 1D array
        let zp = get_v2_test_zarr_path("lat_lon_example.zarr".to_string());
        let mut builder = ZarrRecordBatchStreamBuilder::new(zp);

        builder = builder.with_filter(create_filter());
        let stream = builder.build().await.unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        // v2 format with array broadcast
        let zp = get_v2_test_zarr_path("lat_lon_example_broadcastable.zarr".to_string());
        let mut builder = ZarrRecordBatchStreamBuilder::new(zp);

        builder = builder.with_filter(create_filter());
        let stream = builder.build().await.unwrap();
        let records_from_one_d_repr: Vec<_> = stream.try_collect().await.unwrap();

        assert_eq!(records_from_one_d_repr.len(), records.len());
        for (rec, rec_from_one_d_repr) in records.iter().zip(records_from_one_d_repr.iter()) {
            assert_eq!(rec, rec_from_one_d_repr);
        }

        // v3 format with array broadcast
        let zp = get_v3_test_zarr_path("with_broadcastable_array.zarr".to_string());
        let mut builder = ZarrRecordBatchStreamBuilder::new(zp);

        builder = builder.with_filter(create_filter());
        let stream = builder.build().await.unwrap();
        let records_from_one_d_repr: Vec<_> = stream.try_collect().await.unwrap();

        assert_eq!(records_from_one_d_repr.len(), records.len());
        for (rec, rec_from_one_d_repr) in records.iter().zip(records_from_one_d_repr.iter()) {
            assert_eq!(rec, rec_from_one_d_repr);
        }
    }

    fn get_v3_test_zarr_path(zarr_store: String) -> ZarrPath {
        ZarrPath::new(
            Arc::new(LocalFileSystem::new()),
            Path::from_absolute_path(get_test_v3_data_path(zarr_store)).unwrap(),
        )
    }

    #[tokio::test]
    async fn with_sharding_tests() {
        let zp = get_v3_test_zarr_path("with_sharding.zarr".to_string());
        let stream_builder = ZarrRecordBatchStreamBuilder::new(zp);

        let stream = stream_builder.build().await.unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        let target_types = HashMap::from([
            ("float_data".to_string(), DataType::Float64),
            ("int_data".to_string(), DataType::Int64),
        ]);

        let rec = &records[2];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Float64Type, f64>(
            "float_data",
            rec,
            &[
                32.0, 33.0, 40.0, 41.0, 34.0, 35.0, 42.0, 43.0, 48.0, 49.0, 56.0, 57.0, 50.0, 51.0,
                58.0, 59.0,
            ],
        );
        validate_primitive_column::<Int64Type, i64>(
            "int_data",
            rec,
            &[
                32, 33, 40, 41, 34, 35, 42, 43, 48, 49, 56, 57, 50, 51, 58, 59,
            ],
        );
    }

    #[tokio::test]
    async fn three_dims_with_sharding_with_edge_tests() {
        let zp = get_v3_test_zarr_path("with_sharding_with_edge_3d.zarr".to_string());
        let stream_builder = ZarrRecordBatchStreamBuilder::new(zp);

        let stream = stream_builder.build().await.unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        let target_types = HashMap::from([("float_data".to_string(), DataType::Float64)]);

        let rec = &records[23];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Float64Type, f64>(
            "float_data",
            rec,
            &[
                1020.0, 1021.0, 1031.0, 1032.0, 1141.0, 1142.0, 1152.0, 1153.0, 1022.0, 1033.0,
                1143.0, 1154.0, 1042.0, 1043.0, 1053.0, 1054.0, 1163.0, 1164.0, 1174.0, 1175.0,
                1044.0, 1055.0, 1165.0, 1176.0, 1262.0, 1263.0, 1273.0, 1274.0, 1264.0, 1275.0,
                1284.0, 1285.0, 1295.0, 1296.0, 1286.0, 1297.0,
            ],
        );
    }

    #[tokio::test]
    async fn no_sharding_tests() {
        let zp = get_v3_test_zarr_path("no_sharding.zarr".to_string());
        let stream_builder = ZarrRecordBatchStreamBuilder::new(zp);

        let stream = stream_builder.build().await.unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        let target_types = HashMap::from([("int_data".to_string(), DataType::Int32)]);

        let rec = &records[1];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Int32Type, i32>(
            "int_data",
            rec,
            &[4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 39, 52, 53, 54, 55],
        );
    }
}
