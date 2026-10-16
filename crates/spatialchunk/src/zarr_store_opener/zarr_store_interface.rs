// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::{BTreeSet, HashMap};
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::*;
use arrow::compute::kernels::aggregate::{max as arrow_max, min as arrow_min};
use arrow::compute::{concat, take};
use arrow::datatypes::*;
use arrow::record_batch::{RecordBatch, RecordBatchOptions};
use bytes::Bytes;
use datafusion::common::ScalarValue;
use datafusion::logical_expr::ColumnarValue;
use datafusion::physical_expr::utils::{collect_columns, reassign_expr_columns};
use datafusion::physical_plan::PhysicalExpr;
use futures::future::{BoxFuture, Shared};
use tokio::sync::mpsc::Receiver;
use tokio::task::JoinSet;
use zarrs::array::codec::api::{ArrayToBytesCodecTraits, CodecOptions};
use zarrs::array::data_type::{
    BoolDataType, Float32DataType, Float64DataType, Int16DataType, Int32DataType, Int64DataType,
    Int8DataType, NumpyDateTime64DataType, NumpyTimeDelta64DataType, StringDataType,
    UInt16DataType, UInt32DataType, UInt64DataType, UInt8DataType,
};
use zarrs::array::{
    Array, ArrayBytes, ArraySubset, ChunkShapeTraits, DataType as zDataType, ElementOwned,
};
use zarrs::metadata_ext::data_type::NumpyTimeUnit;
use zarrs_storage::AsyncReadableListableStorageTraits;

use super::filter::ZarrChunkFilter;
use super::io_runtime::IoRuntime;
use super::metrics::ZarrMetrics;
use super::util::{ColumnNames, NumElements};
use super::zarr_errors::{ZarrQueryError, ZarrQueryResult};

//***************************************************
// A helper struct that handles the special case where an
// array is a 1D coordinate (that can be broadcasted)
//***************************************************
struct BroadcastableCoord {
    field: FieldRef,
    values: ArrayRef,

    // the position of the coordinate in the full dimensionality.
    position: usize,
}

impl BroadcastableCoord {
    // the range of the coordinate's dimension, out of one optional range per
    // dimension of the full dimensionality.
    fn range<'a>(&self, ranges: &'a [Option<Range<u64>>]) -> ZarrQueryResult<&'a Range<u64>> {
        let name = self.field.name();
        let range = ranges
            .get(self.position)
            .ok_or_else(|| {
                ZarrQueryError::InvalidCompute(format!(
                    "Coordinate {name} is outside of the broadcast dimensions"
                ))
            })?
            .as_ref()
            .ok_or_else(|| {
                ZarrQueryError::InvalidCompute(format!(
                    "No range provided for the dimension of coordinate {name}"
                ))
            })?;
        if range.end > self.values.len() as u64 {
            return Err(ZarrQueryError::InvalidCompute(format!(
                "Range {range:?} is out of bounds for coordinate {name}"
            )));
        }
        Ok(range)
    }

    // the coordinate values over the range of its dimension, not broadcast.
    fn slice(&self, ranges: &[Option<Range<u64>>]) -> ZarrQueryResult<ArrayRef> {
        let range = self.range(ranges)?;
        Ok(self.values.slice(
            range.start as usize,
            range.end.saturating_sub(range.start) as usize,
        ))
    }

    // the coordinate values broadcast over the dimensions with a range, see
    // the `Broadcastable` trait.
    fn broadcast(&self, ranges: &[Option<Range<u64>>]) -> ZarrQueryResult<ArrayRef> {
        let sliced = self.slice(ranges)?;

        let lens: Vec<u64> = ranges
            .iter()
            .flatten()
            .map(|r| r.end.saturating_sub(r.start))
            .collect();
        if lens.num_elements() == 0 {
            return Ok(new_empty_array(self.field.data_type()));
        }

        // the output is each value of the range repeated `stride` times (the
        // axes after the coordinate's), with that block tiled `outer` times
        // (the axes before it).
        let axis = ranges[..self.position].iter().flatten().count();
        let len = lens[axis];
        let stride = lens[axis + 1..].num_elements();
        let outer = lens[..axis].num_elements();

        let block = if stride == 1 {
            sliced
        } else {
            let indices: UInt64Array = (0..len * stride).map(|i| i / stride).collect();
            take(&sliced, &indices, None)?
        };
        if outer == 1 {
            Ok(block)
        } else {
            Ok(concat(&vec![block.as_ref(); outer as usize])?)
        }
    }
}

trait Broadcastable {
    // takes one optional range per dimension of the full dimensionality.
    // the output spans the dimensions with a range, in C order. dimensions
    // without a coordinate just repeat the values along that axis.
    fn broadcast(&self, ranges: &[Option<Range<u64>>]) -> ZarrQueryResult<RecordBatch>;

    // takes the ranges of a chunk along every dimension, and broadcasts only
    // over the dimensions of the coordinates.
    fn broadcast_chunk(&self, chunk_ranges: &[Option<Range<u64>>]) -> ZarrQueryResult<RecordBatch>;
}

impl Broadcastable for [&BroadcastableCoord] {
    fn broadcast(&self, ranges: &[Option<Range<u64>>]) -> ZarrQueryResult<RecordBatch> {
        let num_rows: u64 = ranges
            .iter()
            .flatten()
            .map(|r| r.end.saturating_sub(r.start))
            .product();

        let mut fields = Vec::with_capacity(self.len());
        let mut columns = Vec::with_capacity(self.len());
        for coord in self {
            fields.push(coord.field.clone());
            columns.push(coord.broadcast(ranges)?);
        }

        Ok(RecordBatch::try_new_with_options(
            Arc::new(Schema::new(fields)),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(num_rows as usize)),
        )?)
    }

    fn broadcast_chunk(&self, chunk_ranges: &[Option<Range<u64>>]) -> ZarrQueryResult<RecordBatch> {
        let mut ranges = vec![None; chunk_ranges.len()];
        for coord in self {
            ranges[coord.position] = chunk_ranges[coord.position].clone();
        }
        self.broadcast(&ranges)
    }
}

//***************************************************
// A helper struct for the geometry of the chunk grid.
//***************************************************
#[derive(Clone, Default)]
struct ChunkGrid {
    array_shape: Vec<u64>,
    grid_shape: Vec<u64>,

    // the chunk shape in the metadata, the chunks at the array edges can be
    // smaller.
    chunk_shape: Vec<u64>,
}

impl ChunkGrid {
    // technically getting the chunk shape requires a chunk index, but zarrs
    // doesn't return a chunk size that depends on the index for regular
    // grids, so we retrieve "chunk 0" and adjust for array edges when reading.
    // so only regular grids are supported here, for now.
    fn from_array<T: ?Sized>(arr: &Array<T>) -> ZarrQueryResult<Self> {
        Ok(Self {
            array_shape: arr.shape().to_vec(),
            chunk_shape: arr
                .chunk_shape(&vec![0; arr.dimensionality()])?
                .to_array_shape(),
            grid_shape: arr.chunk_grid_shape().to_vec(),
        })
    }

    // appends the dimensions of another grid after this grid's dimensions.
    fn extend(&mut self, other: ChunkGrid) {
        self.array_shape.extend(other.array_shape);
        self.chunk_shape.extend(other.chunk_shape);
        self.grid_shape.extend(other.grid_shape);
    }

    // the real chunk shape for a given index, adjusted for the array edges.
    fn chunk_shape(&self, chk_idx: &[u64]) -> Vec<u64> {
        chk_idx
            .iter()
            .zip(self.array_shape.iter())
            .zip(self.chunk_shape.iter())
            .map(|((i, a), c)| std::cmp::min(a - i * c, *c))
            .collect()
    }

    // the range of values the chunk covers along every dimension.
    fn ranges(&self, chk_idx: &[u64]) -> Vec<Option<Range<u64>>> {
        chk_idx
            .iter()
            .zip(self.chunk_shape.iter())
            .zip(self.chunk_shape(chk_idx))
            .map(|((i, c), real_c)| Some(i * c..i * c + real_c))
            .collect()
    }

    // whether the chunk is at the edge of the array along any dimension.
    fn is_edge(&self, chk_idx: &[u64]) -> bool {
        chk_idx
            .iter()
            .zip(self.grid_shape.iter())
            .any(|(i, g)| *i == g - 1)
    }
}

impl FromIterator<ChunkGrid> for ChunkGrid {
    fn from_iter<I: IntoIterator<Item = ChunkGrid>>(iter: I) -> Self {
        let mut grid = ChunkGrid::default();
        for other in iter {
            grid.extend(other);
        }
        grid
    }
}

//***************************************************
// A helper struct to resolve data arrays and 1D coordinates
// and validate everything.
//***************************************************
struct DataArraysMeta {
    dim_names: Option<Vec<String>>,
    grid: ChunkGrid,
}

impl DataArraysMeta {
    fn new<T: ?Sized>(arr: &Array<T>) -> ZarrQueryResult<Self> {
        Ok(Self {
            dim_names: dim_names(arr)?,
            grid: ChunkGrid::from_array(arr)?,
        })
    }

    fn validate<T: ?Sized>(&self, name: &str, arr: &Array<T>) -> ZarrQueryResult<()> {
        let other = Self::new(arr)?;
        if other.grid.array_shape != self.grid.array_shape {
            return Err(ZarrQueryError::InvalidMetadata(format!(
                "Shape of array {name} doesn't match the shape of the other data arrays"
            )));
        }
        if other.grid.chunk_shape != self.grid.chunk_shape {
            return Err(ZarrQueryError::InvalidMetadata(format!(
                "Chunk shape of array {name} doesn't match the chunk shape of the other data arrays"
            )));
        }
        if other.dim_names != self.dim_names {
            return Err(ZarrQueryError::InvalidMetadata(format!(
                "Dimension names of array {name} don't match the dimension names of the other data arrays"
            )));
        }
        Ok(())
    }

    // returns the position of the coordinate in the data dimensionality.
    // also validate that the coord exists and has the right size.
    fn validate_coord(&self, name: &str, values: &ArrayRef) -> ZarrQueryResult<usize> {
        let dim_names = self
            .dim_names
            .as_ref()
            .ok_or(ZarrQueryError::InvalidMetadata(
                "Dimension names are required on data arrays when there are coordinates".into(),
            ))?;
        let position = dim_names.iter().position(|d| d == name).ok_or_else(|| {
            ZarrQueryError::InvalidMetadata(format!(
                "Coordinate {name} is not a dimension of the data arrays"
            ))
        })?;
        if values.len() as u64 != self.grid.array_shape[position] {
            return Err(ZarrQueryError::InvalidMetadata(format!(
                "Length of coordinate {name} doesn't match the data size along its dimension"
            )));
        }
        Ok(position)
    }
}

// when only coordinates are requested, each one is its own dimension, in the
// order they were requested, with its own chunking.
fn coords_only_meta(
    coords: Vec<(String, FieldRef, ArrayRef, ChunkGrid)>,
) -> (HashMap<String, BroadcastableCoord>, ChunkGrid) {
    let (coords, grids): (Vec<_>, Vec<_>) = coords
        .into_iter()
        .map(|(name, field, values, grid)| ((name, field, values), grid))
        .unzip();

    let broadcastable_coords = coords
        .into_iter()
        .enumerate()
        .map(|(position, (name, field, values))| {
            (
                name,
                BroadcastableCoord {
                    position,
                    field,
                    values,
                },
            )
        })
        .collect();
    (broadcastable_coords, grids.into_iter().collect())
}

//***************************************************
// The interface to an array in a zarr store.
//***************************************************
struct ArrayInterface<T: AsyncReadableListableStorageTraits + ?Sized> {
    name: String,
    arr: Arc<Array<T>>,
    chk_index: Vec<u64>,

    // the real chunk shape, adjusted for the array edges, and a flag
    // to specify if the chunk idx maps to an edge chunk.
    chunk_shape: Vec<u64>,
    is_edge: bool,
}

enum BytesFromArray {
    // the static lifetime is there because the ArrayBytes comes from calling
    // into_owned() on the ArrayBytes we get from reading the zarr data.
    Decoded(ArrayBytes<'static>),
    Encoded(Option<Bytes>),
}

impl<T: AsyncReadableListableStorageTraits + ?Sized + 'static> ArrayInterface<T> {
    async fn read_bytes(&self) -> ZarrQueryResult<BytesFromArray> {
        // handling edges is easier if we use zarrs built in method, but it
        // reads and decodes, so we don't split between runtimes here.
        if self.is_edge {
            let ranges: Vec<_> = self.chunk_shape.iter().map(|c| 0..*c).collect();
            let array_subset = ArraySubset::new_with_ranges(&ranges);
            let data = self
                .arr
                .async_retrieve_chunk_subset::<ArrayBytes>(&self.chk_index, &array_subset)
                .await?;
            Ok(BytesFromArray::Decoded(data.into_owned()))
        } else {
            let data = self
                .arr
                .async_retrieve_encoded_chunk(&self.chk_index)
                .await?;
            Ok(BytesFromArray::Encoded(data))
        }
    }

    // separate from the read so that decoding can happen on the compute
    // runtime while other reads are still in flight.
    fn decode_data(&self, bytes: BytesFromArray) -> ZarrQueryResult<ArrayRef> {
        let decoded_bytes = match bytes {
            BytesFromArray::Encoded(Some(bytes)) => {
                let chunk_shape = self.arr.chunk_shape(&self.chk_index)?;
                self.arr.codecs().decode(
                    // move the raw bytes into the Vec (reusing the allocation)
                    // rather than copying them; the chain decode wants raw bytes
                    // (ArrayBytesRaw = Cow<[u8]>), which a Vec coerces into.
                    Vec::from(bytes).into(),
                    &chunk_shape,
                    self.arr.data_type(),
                    self.arr.fill_value(),
                    &CodecOptions::default(),
                )?
            }
            BytesFromArray::Encoded(None) => ArrayBytes::new_fill_value(
                self.arr.data_type(),
                self.chunk_shape.num_elements(),
                self.arr.fill_value(),
            )?,
            BytesFromArray::Decoded(bytes) => bytes,
        };
        array_bytes_to_arrow(self.arr.data_type(), decoded_bytes)
    }
}

type ZarrReceiver<T> = Receiver<(ZarrQueryResult<BytesFromArray>, ArrayInterface<T>)>;

// The mutable state for reading chunks, owned by the reader so that the
// store interface itself can be shared.
pub(crate) struct ChunkReadState<T: AsyncReadableListableStorageTraits + ?Sized> {
    join_set: JoinSet<()>,
    metrics: ZarrMetrics,

    // the prefetched reads, with their chunk index and columns.
    state: Option<(ZarrReceiver<T>, Vec<u64>, Vec<String>)>,
}

impl<T: AsyncReadableListableStorageTraits + ?Sized> ChunkReadState<T> {
    pub(crate) fn new(metrics: ZarrMetrics) -> Self {
        Self {
            join_set: JoinSet::new(),
            state: None,
            metrics,
        }
    }

    pub(crate) fn metrics(&self) -> &ZarrMetrics {
        &self.metrics
    }
}

// A future resolving to a store interface, shared between the streams of a
// scan. the error is wrapped in an Arc because a shared future's output must
// be cloneable.
pub(crate) type SharedZarrStoreInterface<T> =
    Shared<BoxFuture<'static, Result<Arc<ZarrStoreInterface<T>>, Arc<ZarrQueryError>>>>;

//***************************************************
// The interface to the zarr store
//***************************************************
pub(crate) struct ZarrStoreInterface<T: AsyncReadableListableStorageTraits + ?Sized> {
    schema_ref: SchemaRef,
    coords: HashMap<String, BroadcastableCoord>,
    grid: ChunkGrid,

    // non 1D coordinate arrays only.
    arrays: HashMap<String, Arc<Array<T>>>,

    // offloads reading data to a dedicated i/o runtime.
    io_runtime: IoRuntime,
}

impl<T: AsyncReadableListableStorageTraits + ?Sized + 'static> ZarrStoreInterface<T> {
    pub(crate) async fn new(
        store: Arc<T>,
        prefix: Option<String>,
        schema_ref: SchemaRef,
        metrics: &ZarrMetrics,
    ) -> ZarrQueryResult<Self> {
        if schema_ref.fields().is_empty() {
            return Err(ZarrQueryError::InvalidColumnRequest(
                "No arrays to open in the zarr store".into(),
            ));
        }

        // some remote stores (e.g. s3) don't work off of the url, just the
        // bucket, so the path to the zarr store is provided separately.
        let prefix = match prefix {
            Some(prefix) => format!("/{prefix}"),
            None => "/".to_string(),
        };
        let cols = schema_ref.fields().column_names();

        // each coordinate's name, field, array, and its own 1D chunk grid.
        let mut coord_arrays: Vec<(String, FieldRef, Array<T>, ChunkGrid)> = Vec::new();

        let mut data_arrays: HashMap<String, Arc<Array<T>>> = HashMap::new();
        let mut data_meta: Option<DataArraysMeta> = None;

        // for each col in the schema, determine if it's a 1d coord or not.
        // data arrays are stored for future reads, but coords are fully read
        // and cached, just outside of the below for loop.
        for col in &cols {
            let path = PathBuf::from(&prefix)
                .join(col)
                .to_str()
                .ok_or(ZarrQueryError::InvalidCompute(
                    "Could not form path from group and column name".into(),
                ))?
                .to_string();
            let arr = Array::async_open(store.clone(), &path).await?;
            if is_1d_coordinate(col, &arr)? {
                let (_, field) = schema_ref.fields().find(col).ok_or_else(|| {
                    ZarrQueryError::InvalidColumnRequest(format!(
                        "Coordinate {col} is not in the schema"
                    ))
                })?;
                let grid = ChunkGrid::from_array(&arr)?;
                coord_arrays.push((col.to_string(), field.clone(), arr, grid));
            } else {
                match &data_meta {
                    Some(meta) => meta.validate(col, &arr)?,
                    None => data_meta = Some(DataArraysMeta::new(&arr)?),
                }
                data_arrays.insert(col.to_string(), Arc::new(arr));
            }
        }

        let reads = coord_arrays
            .into_iter()
            .map(|(name, field, arr, grid)| async move {
                // here the full 1d coordinate data is read.
                let values = read_full_array(&arr).await?;
                Ok::<_, ZarrQueryError>((name, field, values, grid))
            });
        let read_coords = metrics
            .time_io(futures::future::try_join_all(reads))
            .await?;

        let (coords, grid) = match data_meta {
            Some(meta) => {
                let coords = read_coords
                    .into_iter()
                    .map(|(name, field, values, _)| {
                        let position = meta.validate_coord(&name, &values)?;
                        Ok((
                            name,
                            BroadcastableCoord {
                                position,
                                field,
                                values,
                            },
                        ))
                    })
                    .collect::<ZarrQueryResult<_>>()?;
                (coords, meta.grid)
            }
            None => coords_only_meta(read_coords),
        };

        // this runtime will handle the i/o. i/o tasks spawned in that runtime
        // will not share a thead pool with other (probably compute heavy,
        // blocking) tasks.
        let io_runtime = IoRuntime::try_new()?;

        Ok(Self {
            schema_ref,
            arrays: data_arrays,
            coords,
            grid,
            io_runtime,
        })
    }

    pub(crate) fn schema_ref(&self) -> &SchemaRef {
        &self.schema_ref
    }

    pub(crate) fn chunk_grid_shape(&self) -> &[u64] {
        &self.grid.grid_shape
    }

    fn is_coord(&self, name: &str) -> bool {
        self.coords.contains_key(name)
    }

    fn is_data_array(&self, name: &str) -> bool {
        self.arrays.contains_key(name)
    }

    // whether every column is a 1D coordinate.
    fn only_coords<S: AsRef<str>>(&self, cols: impl IntoIterator<Item = S>) -> bool {
        cols.into_iter().all(|col| self.is_coord(col.as_ref()))
    }

    // the columns that are data arrays, in order.
    fn data_cols<S: AsRef<str>>(&self, cols: impl IntoIterator<Item = S>) -> Vec<String> {
        cols.into_iter()
            .filter(|col| self.is_data_array(col.as_ref()))
            .map(|col| col.as_ref().to_string())
            .collect()
    }

    fn get_array_interfaces(
        &self,
        cols: &[String],
        chk_idx: &[u64],
    ) -> ZarrQueryResult<Vec<ArrayInterface<T>>> {
        let chunk_shape = self.grid.chunk_shape(chk_idx);
        let is_edge = self.grid.is_edge(chk_idx);
        cols.iter()
            .map(|col| {
                let arr = self
                    .arrays
                    .get(col)
                    .ok_or_else(|| ZarrQueryError::array_not_in_store(col))?
                    .clone();
                Ok(ArrayInterface {
                    name: col.to_string(),
                    arr,
                    chk_index: chk_idx.to_vec(),
                    chunk_shape: chunk_shape.clone(),
                    is_edge,
                })
            })
            .collect()
    }

    // spawns a read task per array interface on the i/o runtime, returning the
    // receiver that yields each (read result, interface) pair as it completes.
    fn spawn_reads(
        &self,
        interfaces: Vec<ArrayInterface<T>>,
        read_state: &mut ChunkReadState<T>,
    ) -> ZarrReceiver<T> {
        let (tx, rx) = tokio::sync::mpsc::channel(interfaces.len());
        for arr_interface in interfaces {
            let tx = tx.clone();
            let metrics = read_state.metrics.clone();
            let io_task = async move {
                let b = metrics.time_io(arr_interface.read_bytes()).await;
                let _ = tx.send((b, arr_interface)).await;
            };
            read_state
                .join_set
                .spawn_on(io_task, self.io_runtime.handle());
        }
        rx
    }

    // decodes each array as its read completes, which overlaps decoding one
    // array with the still in flight reads of the others.
    async fn drain_decode(
        mut rx: ZarrReceiver<T>,
        metrics: &ZarrMetrics,
    ) -> ZarrQueryResult<HashMap<String, ArrayRef>> {
        let mut data = HashMap::new();
        while let Some((bytes, arr_interface)) = rx.recv().await {
            let bytes = bytes?;
            let arr = metrics.time_decode(|| arr_interface.decode_data(bytes))?;
            data.insert(arr_interface.name, arr);
        }
        Ok(data)
    }

    // reads a chunk without using or setting any prefetched reads.
    async fn get_chunk_uncached(
        &self,
        cols: &[String],
        chk_idx: &[u64],
        read_state: &mut ChunkReadState<T>,
    ) -> ZarrQueryResult<ZarrInMemoryChunk> {
        self.get_chunk(cols, chk_idx.to_vec(), false, None, read_state)
            .await
    }

    // returns the requested columns for a chunk. the 1D coordinates are always
    // broadcast over every dimension, so that the number of rows is the same
    // regardless of the columns requested. this can also trigger the next
    // cbunk to be prefetched, optionally.
    pub(crate) async fn get_chunk(
        &self,
        cols: &[String],
        chk_idx: Vec<u64>,
        use_cached_value: bool,
        next_chunk_idx: Option<Vec<u64>>,
        read_state: &mut ChunkReadState<T>,
    ) -> ZarrQueryResult<ZarrInMemoryChunk> {
        if cols.is_empty() {
            return Err(ZarrQueryError::InvalidColumnRequest(
                "No columns when polling zarr store for chunks".into(),
            ));
        }

        let mut data_cols = Vec::new();
        let mut coords = Vec::new();
        for col in cols {
            if self.is_data_array(col) {
                data_cols.push(col.clone());
            } else if let Some(coord) = self.coords.get(col) {
                coords.push(coord);
            } else {
                return Err(ZarrQueryError::array_not_in_store(col));
            }
        }
        let coords_batch = coords.as_slice().broadcast(&self.grid.ranges(&chk_idx))?;

        let mut chunk = ZarrInMemoryChunk::new();
        chunk.add_batch(&coords_batch);

        if !data_cols.is_empty() {
            // either use the reads prefetched by the previous call, or spawn
            // them now.
            let rx = if use_cached_value && read_state.state.is_some() {
                let (rx, cached_idx, cached_cols) = read_state
                    .state
                    .take()
                    .expect("cache hit implies state is present");
                if cached_idx != chk_idx {
                    return Err(ZarrQueryError::InvalidCompute(
                        "Cached zarr chunk index doesn't match requested chunk index".into(),
                    ));
                }
                if cached_cols != data_cols {
                    return Err(ZarrQueryError::InvalidCompute(
                        "Cached zarr chunk columns don't match requested columns".into(),
                    ));
                }
                rx
            } else {
                let interfaces = self.get_array_interfaces(&data_cols, &chk_idx)?;
                self.spawn_reads(interfaces, read_state)
            };
            for (name, arr) in Self::drain_decode(rx, &read_state.metrics).await? {
                chunk.add_data(name, arr);
            }

            // prefetch the next chunk, so that its data is fetched while other
            // operations run between now and the next call.
            if let Some(next_chunk_idx) = next_chunk_idx {
                let interfaces = self.get_array_interfaces(&data_cols, &next_chunk_idx)?;
                let rx = self.spawn_reads(interfaces, read_state);
                read_state.state = Some((rx, next_chunk_idx, data_cols));
            }
        }

        Ok(chunk)
    }

    // evaluates each of the filter's predicates on a record batch with only the
    // predicate's columns, stopping at the first one that no row passes. the 1D
    // coordinates are only broadcast over the dimensions a predicate needs. returns
    // None if the chunk can be skipped, otherwise the full chunk columns that
    // were read.
    pub(crate) async fn evaluate_filter(
        &self,
        filter: &ZarrChunkFilter,
        chk_idx: &[u64],
        read_state: &mut ChunkReadState<T>,
    ) -> ZarrQueryResult<Option<ZarrInMemoryChunk>> {
        // the predicates on only 1D coordinates are cheap, so they go first,
        // then the predicates with fewer columns.
        let mut keys: Vec<&BTreeSet<String>> = filter.keys().collect();
        keys.sort_by_key(|key| (!self.only_coords(key.iter()), key.len()));

        let chunk_ranges = self.grid.ranges(chk_idx);
        let mut read_chunk = ZarrInMemoryChunk::new();
        for key in keys {
            let batch = if self.only_coords(key.iter()) {
                let coords: Vec<&BroadcastableCoord> =
                    key.iter().map(|col| &self.coords[col]).collect();
                coords.as_slice().broadcast_chunk(&chunk_ranges)?
            } else {
                let cols: Vec<String> = key.iter().cloned().collect();
                let chunk = self.get_chunk_uncached(&cols, chk_idx, read_state).await?;
                let batch = chunk.to_record_batch()?;

                // the columns read are kept so that they don't need to be read
                // again after the filter.
                read_chunk.combine(chunk);
                batch
            };

            if !filter.evaluate(&batch)? {
                return Ok(None);
            }
        }

        Ok(Some(read_chunk))
    }

    // evaluates the chunk-pruning expression on the min/max of each pruning
    // column. the min/max of a 1D coordinate is the same whether it's
    // broadcast or not, so they are computed from the raw coordinate values.
    pub(crate) async fn keep_chunk(
        &self,
        pruning: &ChunkPruning,
        chk_idx: &[u64],
        read_state: &mut ChunkReadState<T>,
    ) -> ZarrQueryResult<bool> {
        let data_cols = self.data_cols(pruning.schema.fields().iter_column_names());
        let mut chunk = if data_cols.is_empty() {
            ZarrInMemoryChunk::new()
        } else {
            self.get_chunk_uncached(&data_cols, chk_idx, read_state)
                .await?
        };

        let ranges = self.grid.ranges(chk_idx);
        for field in pruning.schema.fields() {
            let name = field.name();
            if let Some(coord) = self.coords.get(name) {
                chunk.add_data(name.clone(), coord.slice(&ranges)?);
            }
        }

        let min_max = chunk.to_min_max_record_batch(&pruning.schema)?;
        chunk_kept(&pruning.expr, &min_max)
    }
}

//***************************************************
// A structure to accumulate zarr array data until we can output
// the whole chunk as a record batch.
//***************************************************
pub(crate) struct ZarrInMemoryChunk {
    data: HashMap<String, ArrayRef>,
}

impl ZarrInMemoryChunk {
    pub(crate) fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    fn add_data(&mut self, arr_name: String, data: ArrayRef) {
        self.data.insert(arr_name, data);
    }

    fn add_batch(&mut self, batch: &RecordBatch) {
        for (field, arr) in batch.schema().fields().iter().zip(batch.columns()) {
            self.add_data(field.name().clone(), arr.clone());
        }
    }

    pub(crate) fn combine(&mut self, other: ZarrInMemoryChunk) {
        self.data.extend(other.data);
    }

    pub(crate) fn get(&self, arr_name: &str) -> Option<&ArrayRef> {
        self.data.get(arr_name)
    }

    // gathers the chunk's arrays in schema-field order, erroring if any
    // field's array is missing from the chunk.
    fn columns_in_schema_order(&self, schema: &Schema) -> ZarrQueryResult<Vec<ArrayRef>> {
        schema
            .fields()
            .iter()
            .map(|f| {
                self.data
                    .get(f.name())
                    .cloned()
                    .ok_or_else(|| ZarrQueryError::array_not_in_chunk(f.name()))
            })
            .collect()
    }

    // a record batch with all the arrays in the chunk, in no particular order.
    fn to_record_batch(&self) -> ZarrQueryResult<RecordBatch> {
        Ok(RecordBatch::try_from_iter(
            self.data.iter().map(|(k, v)| (k.clone(), v.clone())),
        )?)
    }

    pub(crate) fn into_record_batch(self, schema: &SchemaRef) -> ZarrQueryResult<RecordBatch> {
        let columns = self.columns_in_schema_order(schema)?;
        Ok(RecordBatch::try_new(schema.clone(), columns)?)
    }

    // a 2-row record batch holding, per column, that column's min (row 0) and
    // max (row 1) in schema-field order.
    fn to_min_max_record_batch(&self, schema: &SchemaRef) -> ZarrQueryResult<RecordBatch> {
        let min_max = self
            .columns_in_schema_order(schema)?
            .iter()
            .zip(schema.fields())
            .map(|(arr, f)| Ok((f.name().clone(), min_max_array(arr)?)))
            .collect::<ZarrQueryResult<Vec<_>>>()?;
        Ok(RecordBatch::try_from_iter(min_max)?)
    }
}

//***************************************************
// A resolved chunk-pruning expression and the subset schema it reads.
// This is currently used for pruning based on spatial joins, and the logic
// relies on the pruning only depending on the min/max of all the columns.
//***************************************************
pub(crate) struct ChunkPruning {
    expr: Arc<dyn PhysicalExpr>,
    schema: SchemaRef,
}

impl ChunkPruning {
    // Selects the columns `expr` references from `projected_schema` (by name)
    // into a subset schema, and reassigns the expression's column indices to it
    // so it evaluates against a min/max batch built from that subset.
    pub(crate) fn try_new(
        expr: Arc<dyn PhysicalExpr>,
        projected_schema: &SchemaRef,
    ) -> ZarrQueryResult<Self> {
        let mut indices = collect_columns(&expr)
            .iter()
            .map(|c| projected_schema.index_of(c.name()))
            .collect::<Result<Vec<_>, _>>()?;
        indices.sort_unstable();
        indices.dedup();

        let schema = Arc::new(projected_schema.project(&indices)?);
        let expr = reassign_expr_columns(expr, &schema)?;
        Ok(Self { expr, schema })
    }
}

// Evaluates a chunk-pruning expression against a chunk's min/max record batch,
// returning whether the chunk should be kept (a null verdict keeps it).
fn chunk_kept(expr: &Arc<dyn PhysicalExpr>, min_max: &RecordBatch) -> ZarrQueryResult<bool> {
    let value = expr.evaluate(min_max)?;
    match value {
        ColumnarValue::Scalar(ScalarValue::Boolean(Some(keep))) => Ok(keep),
        ColumnarValue::Scalar(ScalarValue::Boolean(None)) => Ok(true),
        other => Err(ZarrQueryError::InvalidCompute(format!(
            "chunk-pruning expression must return a boolean scalar, got {other:?}"
        ))),
    }
}

// Returns a length-2 array `[min, max]` of the same numeric type as `array`.
// `None` min/max (empty or all-null input) become nulls.
fn min_max_array(array: &ArrayRef) -> ZarrQueryResult<ArrayRef> {
    macro_rules! min_max {
        ($ty:ty) => {{
            let a = array
                .as_any()
                .downcast_ref::<PrimitiveArray<$ty>>()
                .expect("array data type matched");
            Arc::new(PrimitiveArray::<$ty>::from(vec![
                arrow_min(a),
                arrow_max(a),
            ])) as ArrayRef
        }};
    }

    let out = match array.data_type() {
        DataType::Float64 => min_max!(Float64Type),
        DataType::Float32 => min_max!(Float32Type),
        DataType::Int64 => min_max!(Int64Type),
        DataType::Int32 => min_max!(Int32Type),
        DataType::Int16 => min_max!(Int16Type),
        DataType::Int8 => min_max!(Int8Type),
        DataType::UInt64 => min_max!(UInt64Type),
        DataType::UInt32 => min_max!(UInt32Type),
        DataType::UInt16 => min_max!(UInt16Type),
        DataType::UInt8 => min_max!(UInt8Type),
        other => {
            return Err(ZarrQueryError::InvalidColumnRequest(format!(
                "min/max not supported for data type {other}"
            )))
        }
    };
    Ok(out)
}

fn dim_names<T: ?Sized>(arr: &Array<T>) -> ZarrQueryResult<Option<Vec<String>>> {
    arr.dimension_names()
        .clone()
        .map(|names| {
            names
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .ok_or(ZarrQueryError::InvalidMetadata(
                    "Null dimension names not supported".into(),
                ))
        })
        .transpose()
}

fn is_1d_coordinate<T: ?Sized>(name: &str, arr: &Array<T>) -> ZarrQueryResult<bool> {
    Ok(is_1d_coordinate_from_meta(
        name,
        dim_names(arr)?.as_deref(),
        arr.dimensionality(),
    ))
}

// a 1-D coordinate is a 1-D array that names its own single dimension.
// everything else, including already broadcast N-D coordinates, is a data array.
pub(crate) fn is_1d_coordinate_from_meta(
    name: &str,
    dim_names: Option<&[String]>,
    dimensionality: usize,
) -> bool {
    dimensionality == 1 && matches!(dim_names, Some([dim]) if dim == name)
}

async fn read_full_array<T: AsyncReadableListableStorageTraits + ?Sized + 'static>(
    arr: &Array<T>,
) -> ZarrQueryResult<ArrayRef> {
    let bytes = arr
        .async_retrieve_array_subset::<ArrayBytes>(&arr.subset_all())
        .await?;
    array_bytes_to_arrow(arr.data_type(), bytes)
}

fn array_bytes_to_arrow(t: &zDataType, bytes: ArrayBytes<'_>) -> ZarrQueryResult<ArrayRef> {
    macro_rules! to_array_ref {
        ($array_t: ty, $prim_type: ty) => {{
            let arr: $array_t = <$prim_type>::from_array_bytes(t, bytes)?.into();
            Ok(Arc::new(arr) as ArrayRef)
        }};
    }

    if t.is::<BoolDataType>() {
        to_array_ref!(BooleanArray, bool)
    } else if t.is::<UInt8DataType>() {
        to_array_ref!(PrimitiveArray<UInt8Type>, u8)
    } else if t.is::<UInt16DataType>() {
        to_array_ref!(PrimitiveArray<UInt16Type>, u16)
    } else if t.is::<UInt32DataType>() {
        to_array_ref!(PrimitiveArray<UInt32Type>, u32)
    } else if t.is::<UInt64DataType>() {
        to_array_ref!(PrimitiveArray<UInt64Type>, u64)
    } else if t.is::<Int8DataType>() {
        to_array_ref!(PrimitiveArray<Int8Type>, i8)
    } else if t.is::<Int16DataType>() {
        to_array_ref!(PrimitiveArray<Int16Type>, i16)
    } else if t.is::<Int32DataType>() {
        to_array_ref!(PrimitiveArray<Int32Type>, i32)
    } else if t.is::<Int64DataType>() {
        to_array_ref!(PrimitiveArray<Int64Type>, i64)
    } else if t.is::<Float32DataType>() {
        to_array_ref!(PrimitiveArray<Float32Type>, f32)
    } else if t.is::<Float64DataType>() {
        to_array_ref!(PrimitiveArray<Float64Type>, f64)
    } else if t.is::<StringDataType>() {
        to_array_ref!(StringArray, String)
    } else if let Some(dt) = t.downcast_ref::<NumpyDateTime64DataType>() {
        ensure_unit_scale(t, dt.scale_factor.get())?;
        match dt.unit {
            NumpyTimeUnit::Second => to_array_ref!(TimestampSecondArray, i64),
            NumpyTimeUnit::Millisecond => to_array_ref!(TimestampMillisecondArray, i64),
            NumpyTimeUnit::Microsecond => to_array_ref!(TimestampMicrosecondArray, i64),
            NumpyTimeUnit::Nanosecond => to_array_ref!(TimestampNanosecondArray, i64),
            _ => Err(ZarrQueryError::InvalidType(format!(
                "Unsupported datetime64 unit {} from zarr metadata",
                dt.unit
            ))),
        }
    } else if let Some(dt) = t.downcast_ref::<NumpyTimeDelta64DataType>() {
        ensure_unit_scale(t, dt.scale_factor.get())?;
        match dt.unit {
            NumpyTimeUnit::Second => to_array_ref!(DurationSecondArray, i64),
            NumpyTimeUnit::Millisecond => to_array_ref!(DurationMillisecondArray, i64),
            NumpyTimeUnit::Microsecond => to_array_ref!(DurationMicrosecondArray, i64),
            NumpyTimeUnit::Nanosecond => to_array_ref!(DurationNanosecondArray, i64),
            _ => Err(ZarrQueryError::InvalidType(format!(
                "Unsupported timedelta64 unit {} from zarr metadata",
                dt.unit
            ))),
        }
    } else {
        Err(ZarrQueryError::InvalidType(format!(
            "Unsupported type {t} from zarr metadata"
        )))
    }
}

// arrow's timestamp/duration types can't represent a scale factor other than 1,
// so we reject anything else rather than silently reading wrong values.
fn ensure_unit_scale(t: &zDataType, scale_factor: u32) -> ZarrQueryResult<()> {
    if scale_factor != 1 {
        return Err(ZarrQueryError::InvalidType(format!(
            "Unsupported scale factor {scale_factor} for type {t} from zarr metadata"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod zarr_store_interface_tests {
    use super::*;
    use crate::test_utils::get_local_zarr_store;

    #[tokio::test]
    async fn data_and_coords_get_chunk_test() {
        let (wrapper, schema) =
            get_local_zarr_store(true, 0.0, "store_interface_data_and_coords").await;
        let store = wrapper.get_store();

        let names: Vec<String> = vec!["data".into(), "lat".into(), "lon".into()];
        let metrics = ZarrMetrics::disconnected();
        let interface = ZarrStoreInterface::new(store, None, schema, &metrics)
            .await
            .unwrap();
        let mut read_state = ChunkReadState::new(metrics.clone());

        let batch = interface
            .get_chunk(
                &["lat".to_string()],
                vec![0, 0],
                false,
                None,
                &mut read_state,
            )
            .await
            .unwrap()
            .to_record_batch()
            .unwrap();
        assert_eq!(batch.num_columns(), 1);
        assert_eq!(batch.num_rows(), 9);

        let batch = interface
            .get_chunk(&names, vec![0, 0], false, None, &mut read_state)
            .await
            .unwrap()
            .to_record_batch()
            .unwrap();
        assert_eq!(batch.num_columns(), 3);
        assert_eq!(batch.num_rows(), 9);
    }

    #[tokio::test]
    async fn coord_only_get_chunk_test() {
        let (wrapper, schema) = get_local_zarr_store(true, 0.0, "store_interface_coord_only").await;
        let store = wrapper.get_store();

        let names: Vec<String> = vec!["lat".into()];
        let schema = Arc::new(schema.project(&[schema.index_of("lat").unwrap()]).unwrap());
        let metrics = ZarrMetrics::disconnected();
        let interface = ZarrStoreInterface::new(store, None, schema, &metrics)
            .await
            .unwrap();
        let mut read_state = ChunkReadState::new(metrics.clone());

        let batch = interface
            .get_chunk(&names, vec![0], false, None, &mut read_state)
            .await
            .unwrap()
            .to_record_batch()
            .unwrap();
        assert_eq!(batch.num_columns(), 1);
        assert_eq!(batch.num_rows(), 3);
    }
}
