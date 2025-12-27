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

use std::borrow::Cow;
use std::cmp::min;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::*;
use arrow::datatypes::*;
use arrow::record_batch::RecordBatch;
use arrow_schema::ArrowError;
use async_stream::try_stream;
use bytes::Bytes;
use futures::stream::{BoxStream, Stream};
use itertools::iproduct;
use tokio::sync::mpsc::Receiver;
use tokio::task::JoinSet;
use zarrs::array::codec::{ArrayToBytesCodecTraits, CodecOptions};
use zarrs::array::{Array, ArrayBytes, ArraySize, DataType as zDataType, ElementOwned};
use zarrs::array_subset::ArraySubset;
use zarrs_storage::AsyncReadableListableStorageTraits;

use super::filter::ZarrChunkFilter;
use super::io_runtime::IoRuntime;
use super::zarr_errors::{ZarrQueryError, ZarrQueryResult};

/// this function handles having multiple values for a given vector,
/// one per array, including some arrays that might be lower
/// dimension coordinates.
fn resolve_vector(
    coords: &ZarrCoordinates,
    vecs: HashMap<String, Vec<u64>>,
) -> ZarrQueryResult<Vec<u64>> {
    let mut final_vec: Option<Vec<u64>> = None;
    for (k, vec) in vecs.iter() {
        if let Some(final_vec) = &final_vec {
            if let Some(pos) = coords.get_coord_position(k) {
                // if we have a the final vector (from a previous non
                // coordinate array), and this current array is a coordinate
                // array, its one vector element must match the array element
                // in the final vector at the position of the cooridnate.
                if final_vec[pos] != vec[0] {
                    return Err(ZarrQueryError::InvalidMetadata(
                        "Mismatch between vectors for different arrays".into(),
                    ));
                }
            // if the current array is not a coordinate, it must match the
            // final vector we have extracted from a previous array.
            } else if final_vec != vec {
                return Err(ZarrQueryError::InvalidMetadata(
                    "Mismatch between vectors for different arrays".into(),
                ));
            }
        } else if !coords.is_coordinate(k) {
            final_vec = Some(vec.clone());
        }
    }

    if let Some(final_vec) = final_vec {
        Ok(final_vec)
    // the else branch here would happen if all the arrays are coordinates.
    } else {
        let mut final_vec: Vec<u64> = vec![0; coords.coord_positions.len()];
        for (k, p) in coords.coord_positions.iter() {
            final_vec[*p] = vecs.get(k).ok_or(ZarrQueryError::InvalidMetadata(
                "Array is missing from array map".into(),
            ))?[0];
        }
        Ok(final_vec)
    }
}

/// A struct to handle coordinate variables, and "broadcasting" them when reading multidimensional
/// data.
#[derive(Debug)]
struct ZarrCoordinates {
    /// the position of each coordinate in the overall chunk shape.
    /// the coordinates are arrays that contain data that characterises
    /// a dimension, such as time, or a longitude or latitude.
    coord_positions: HashMap<String, usize>,
}

impl ZarrCoordinates {
    fn new<T: ?Sized>(
        arrays: &HashMap<String, Array<T>>,
        schema_ref: SchemaRef,
    ) -> ZarrQueryResult<Self> {
        // the goal of these "coordinates" is to determine what needs
        // to be broadcasted from 1D to ND depending on what columns
        // were selected. based on what is a broadcastable coordinate
        // and its position in the overall chunk dimensionality, we
        // can combine a 1D array with ND arrays later on.
        let mut coord_positions: HashMap<String, usize> = HashMap::new();

        // this is pretty messy, but essentially for each array we
        // extract it's dimentionality and its dimension. at this stage,
        // we allow for an array to not have dimensions, but not to have
        // dimensions without a name.
        let arr_dims = arrays
            .iter()
            .map(|(k, v)| {
                (
                    k,
                    v.dimensionality(),
                    v.dimension_names()
                        .clone()
                        .map(|vec| {
                            vec.into_iter().collect::<Option<Vec<_>>>().ok_or(
                                ZarrQueryError::InvalidMetadata(
                                    "Null dimension names not supported".into(),
                                ),
                            )
                        })
                        .transpose(),
                )
            })
            .map(|(k, d, res)| res.map(|names| (k, d, names)))
            .collect::<ZarrQueryResult<Vec<(&String, usize, Option<Vec<String>>)>>>()?;

        // first case to check, do all the arrays have the same
        // dimensionality.
        let mut ordered_dim_names: Option<Vec<String>> = None;
        if arr_dims.windows(2).all(|w| w[0].1 == w[1].1) {
            // this is the case where all the arrays are coordinates,
            // so we determine the broadcasting order from the schema.
            if arr_dims.iter().all(|d| Some(vec![d.0.clone()]) == d.2) {
                ordered_dim_names = Some(
                    schema_ref
                        .fields
                        .into_iter()
                        .map(|f| f.name().to_string())
                        .collect(),
                );
            // this is the case where there is a mix of data and
            // coordinates, but all the coordinates are already
            // stored as broadcasted ararys, so there is no need
            // to do anything later on.
            } else {
                return Ok(Self { coord_positions });
            }
        }

        // if we didn't hit the above conditions, then the arrays
        // have mixed dimensionality. we extract the chunk dimension
        // names, which must be consistent across all arrays (that
        // are not broadcastable coordinates).
        if ordered_dim_names.is_none() {
            for d in &arr_dims {
                if d.1 != 1 {
                    let d = d.2.clone();
                    let arr_dim_names: Vec<_> = d.ok_or(ZarrQueryError::InvalidMetadata(
                        "With mixed array dimensionality, dimension names are required".into(),
                    ))?;

                    if let Some(ordered_dim_names) = &ordered_dim_names {
                        if *ordered_dim_names != arr_dim_names {
                            return Err(ZarrQueryError::InvalidMetadata(
                                "Dimension names must be consistent across arrays".into(),
                            ));
                        }
                    } else {
                        ordered_dim_names = Some(arr_dim_names);
                    }
                }
            }
        }

        // for each 1D array, we check that it is a coordinate, it has
        // to be at this point in the function, and find its position
        // in the chunk dimension names.
        let ordered_dim_names = ordered_dim_names.ok_or(ZarrQueryError::InvalidMetadata(
            "With mixed array dimensionality, dimension names are required".into(),
        ))?;
        for d in arr_dims {
            if d.1 == 1 {
                if Some(vec![d.0.clone()]) != d.2 {
                    return Err(ZarrQueryError::InvalidMetadata(
                        "With mixed array dimensionality, 1D arrays must be coordinates".into(),
                    ));
                }
                let pos = ordered_dim_names.iter().position(|dim| dim == d.0).ok_or(
                    ZarrQueryError::InvalidMetadata(
                        "Could not find coordinate in dimension names".into(),
                    ),
                )?;
                coord_positions.insert(d.0.clone(), pos);
            }
        }

        Ok(Self { coord_positions })
    }

    /// checks if a column name corresponds to a coordinate.
    fn is_coordinate(&self, col: &str) -> bool {
        self.coord_positions.contains_key(col)
    }

    /// returns the position of a coordinate within the chunk
    /// dimensionality if the column is a coordinate, if not
    /// returns None.
    fn get_coord_position(&self, col: &str) -> Option<usize> {
        self.coord_positions.get(col).cloned()
    }

    /// return the vector element that corresponds to a coordinate's
    /// position within the dimensionality (if the variable is a coordinate).
    fn reduce_if_coord(&self, vec: Vec<u64>, col: &str) -> Vec<u64> {
        if let Some(pos) = self.coord_positions.get(col) {
            return vec![vec[*pos]];
        }

        vec
    }

    /// broadacast a 1D array to a nD array if the variable is a coordinate.
    /// note that we return a 1D vector, but this is just because we map all
    /// the chunk to columnar data, so a m x n array gets mapped to a 1D
    /// vector of length m x n.
    fn broadcast_if_coord<T: Clone>(
        &self,
        coord_name: &str,
        data: Vec<T>,
        full_chunk_shape: &[u64],
    ) -> ZarrQueryResult<Vec<T>> {
        let dim_idx = self.get_coord_position(coord_name);
        if dim_idx.is_none() || full_chunk_shape.len() == 1 {
            return Ok(data);
        }
        let dim_idx = dim_idx.unwrap();

        match (full_chunk_shape.len(), dim_idx) {
            (2, 0) => Ok(data
                .into_iter()
                .flat_map(|v| std::iter::repeat_n(v, full_chunk_shape[1] as usize))
                .collect()),
            (2, 1) => Ok(vec![&data[..]; full_chunk_shape[0] as usize].concat()),
            (3, 0) => Ok(data
                .into_iter()
                .flat_map(|v| {
                    std::iter::repeat_n(v, (full_chunk_shape[1] * full_chunk_shape[2]) as usize)
                })
                .collect()),
            (3, 1) => {
                let v: Vec<_> = data
                    .into_iter()
                    .flat_map(|v| std::iter::repeat_n(v, full_chunk_shape[2] as usize))
                    .collect();
                Ok(vec![&v[..]; full_chunk_shape[0] as usize].concat())
            }
            (3, 2) => {
                Ok(vec![&data[..]; (full_chunk_shape[0] * full_chunk_shape[1]) as usize].concat())
            }
            _ => Err(ZarrQueryError::InvalidCompute(
                "Invalid dimensionality when trying to broadcast dimension".into(),
            )),
        }
    }
}

/// An interface to a zarr array that can be used to retrieve
/// data and then decode it.
///
/// the chunk index corresponds to the chunk that is being read, the
/// coords to the coordinates for the full chunk (which can be made up
/// of one or more arrays) and the full chunk shape is relevant when
/// the chunk has some coordinate arrays, which are 1 dimensional, while
/// the non coordinate arrays can be multi dimensional. the full chunk
/// size is used to broadcast the coordinates to the full size.
struct ArrayInterface<T: AsyncReadableListableStorageTraits + ?Sized> {
    name: String,
    arr: Arc<Array<T>>,
    coords: Arc<ZarrCoordinates>,
    full_chunk_shape: Vec<u64>,
    chk_index: Vec<u64>,
}

// T doesn't need to be Clone, but deriving apparently requires
// that, so I have implement manually.
impl<T: AsyncReadableListableStorageTraits + ?Sized> Clone for ArrayInterface<T> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.to_string(),
            arr: self.arr.clone(),
            coords: self.coords.clone(),
            full_chunk_shape: self.full_chunk_shape.clone(),
            chk_index: self.chk_index.clone(),
        }
    }
}

/// in most cases, we will read encoded bytes and decode them after,
/// but in the case of a missing chunk the result of the read operation
/// will be done.vin a few cases though we will read pre-decoded bytes,
/// hence why we have this enum.
enum BytesFromArray {
    Decoded(Bytes),
    Encoded(Option<Bytes>),
}

impl<T: AsyncReadableListableStorageTraits + ?Sized + 'static> ArrayInterface<T> {
    fn new(
        name: String,
        arr: Arc<Array<T>>,
        coords: Arc<ZarrCoordinates>,
        full_chunk_shape: Vec<u64>,
        mut chk_index: Vec<u64>,
    ) -> Self {
        chk_index = coords.reduce_if_coord(chk_index, &name);
        Self {
            name,
            arr,
            coords,
            full_chunk_shape,
            chk_index,
        }
    }

    /// read the bytes from the chunk the interface was built for.
    async fn read_bytes(&self) -> ZarrQueryResult<BytesFromArray> {
        let chunk_grid = self.arr.chunk_grid_shape();
        let is_edge_grid = self
            .chk_index
            .iter()
            .zip(chunk_grid.iter())
            .any(|(i, g)| i == &(g - 1));
        // handling edges is easier if we just read a subset of the array
        // from the start, but to do that we need to read decoded bytes.
        if is_edge_grid {
            let arr_shape = self.arr.shape();
            let chunk_shape = self.arr.chunk_shape(&self.chk_index)?.to_array_shape();

            // determine the real size for each of the dimensions (at least
            // one of which will be at the edge of the array.)
            let ranges: Vec<_> = self
                .chk_index
                .iter()
                .zip(arr_shape.iter())
                .zip(chunk_shape.iter())
                .map(|((i, a), c)| 0..(std::cmp::min(a - i * c, *c)))
                .collect();

            let array_subset = ArraySubset::new_with_ranges(&ranges);
            let data = self
                .arr
                .async_retrieve_chunk_subset(&self.chk_index, &array_subset)
                .await?;
            let data = data.into_fixed()?;
            Ok(BytesFromArray::Decoded(data.into_owned().into()))
        // this will be the more common case, everything except edge chunks.
        } else {
            let data = self
                .arr
                .async_retrieve_encoded_chunk(&self.chk_index)
                .await?;
            Ok(BytesFromArray::Encoded(data))
        }
    }

    /// decode the chunk that was read previously read from this interface.
    /// the reason the 2 functionalities are separated is that we want to
    /// interleave the async part (reading data) with the compute part
    /// (decoding the data, creating the record batch) so that we can make
    /// progress on the latter while the former is running.
    fn decode_data(&self, bytes: BytesFromArray) -> ZarrQueryResult<ArrayRef> {
        let decoded_bytes = match bytes {
            BytesFromArray::Encoded(bytes) => {
                if let Some(bytes) = bytes {
                    self.arr.codecs().decode(
                        Cow::Owned(bytes.into()),
                        &self.arr.chunk_array_representation(&self.chk_index)?,
                        &CodecOptions::default(),
                    )?
                } else {
                    let chk_shp = self
                        .coords
                        .reduce_if_coord(self.full_chunk_shape.clone(), &self.name);
                    let num_elems = chk_shp.iter().fold(1, |mut acc, x| {
                        acc *= x;
                        acc
                    });
                    let array_size = ArraySize::new(self.arr.data_type().size(), num_elems);
                    ArrayBytes::new_fill_value(array_size, self.arr.fill_value())
                }
            }
            BytesFromArray::Decoded(bytes) => ArrayBytes::Fixed(Cow::Owned(bytes.into())),
        };

        let t = self.arr.data_type();
        macro_rules! return_array_ref {
            ($array_t: ty, $prim_type: ty) => {{
                let arr_ref: $array_t = self
                    .coords
                    .broadcast_if_coord(
                        &self.name,
                        <$prim_type>::from_array_bytes(t, decoded_bytes)?,
                        &self.full_chunk_shape,
                    )?
                    .into();
                return Ok(Arc::new(arr_ref) as ArrayRef);
            }};
        }

        match t {
            zDataType::Bool => return_array_ref!(BooleanArray, bool),
            zDataType::UInt8 => return_array_ref!(PrimitiveArray<UInt8Type>, u8),
            zDataType::UInt16 => return_array_ref!(PrimitiveArray<UInt16Type>, u16),
            zDataType::UInt32 => return_array_ref!(PrimitiveArray<UInt32Type>, u32),
            zDataType::UInt64 => return_array_ref!(PrimitiveArray<UInt64Type>, u64),
            zDataType::Int8 => return_array_ref!(PrimitiveArray<Int8Type>, i8),
            zDataType::Int16 => return_array_ref!(PrimitiveArray<Int16Type>, i16),
            zDataType::Int32 => return_array_ref!(PrimitiveArray<Int32Type>, i32),
            zDataType::Int64 => return_array_ref!(PrimitiveArray<Int64Type>, i64),
            zDataType::Float32 => return_array_ref!(PrimitiveArray<Float32Type>, f32),
            zDataType::Float64 => return_array_ref!(PrimitiveArray<Float64Type>, f64),
            zDataType::String => return_array_ref!(StringArray, String),
            _ => Err(ZarrQueryError::InvalidType(format!(
                "Unsupported type {t} from zarr metadata"
            ))),
        }
    }
}

/// A structure to accumulate zarr array data until we can output
/// the whole chunk as a record batch.
struct ZarrInMemoryChunk {
    data: HashMap<String, ArrayRef>,
}

impl ZarrInMemoryChunk {
    fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    fn add_data(&mut self, arr_name: String, data: ArrayRef) {
        self.data.insert(arr_name, data);
    }

    fn combine(&mut self, other: ZarrInMemoryChunk) {
        self.data.extend(other.data);
    }

    fn check_filter(&self, filter: &ZarrChunkFilter) -> Result<bool, ArrowError> {
        let array_refs: Vec<(String, ArrayRef)> = filter
            .schema_ref()
            .fields()
            .iter()
            .map(|f| self.data.get(f.name()).cloned())
            .collect::<Option<Vec<ArrayRef>>>()
            .ok_or(ZarrQueryError::InvalidProjection(
                "Array missing from array map".into(),
            ))?
            .into_iter()
            .zip(filter.schema_ref().fields.iter())
            .map(|(ar, f)| (f.name().to_string(), ar))
            .collect();

        let rec_batch = RecordBatch::try_from_iter(array_refs)?;
        filter.evaluate(&rec_batch)
    }

    /// the columns in the record batch will be ordered following
    /// the field names in the schema.
    fn into_record_batch(mut self, schema: &SchemaRef) -> ZarrQueryResult<RecordBatch> {
        let array_refs: Vec<(String, ArrayRef)> = schema
            .fields()
            .iter()
            .map(|f| self.data.remove(f.name()))
            .collect::<Option<Vec<ArrayRef>>>()
            .ok_or(ZarrQueryError::InvalidProjection(
                "Array missing from array map".into(),
            ))?
            .into_iter()
            .zip(schema.fields.iter())
            .map(|(ar, f)| (f.name().to_string(), ar))
            .collect();

        RecordBatch::try_from_iter(array_refs)
            .map_err(|e| ZarrQueryError::RecordBatchError(Box::new(e)))
    }
}

/// A wrapper for a map of arrays, which will handle interleaving
/// reading and decoding data from zarr storage.
type ZarrReceiver<T> = Receiver<(ZarrQueryResult<BytesFromArray>, ArrayInterface<T>)>;
struct ZarrStore<T: AsyncReadableListableStorageTraits + ?Sized> {
    arrays: HashMap<String, Arc<Array<T>>>,
    coordinates: Arc<ZarrCoordinates>,
    chunk_shape: Vec<u64>,
    chunk_grid_shape: Vec<u64>,
    array_shape: Vec<u64>,
    io_runtime: IoRuntime,
    join_set: JoinSet<()>,
    state: Option<(ZarrReceiver<T>, Vec<u64>, Vec<String>)>,
}

impl<T: AsyncReadableListableStorageTraits + ?Sized + 'static> ZarrStore<T> {
    fn new(arrays: HashMap<String, Array<T>>, schema_ref: SchemaRef) -> ZarrQueryResult<Self> {
        let coordinates = ZarrCoordinates::new(&arrays, schema_ref)?;

        // technically getting the chunk shape requires a chunk
        // index, but it seems the zarrs library doesn't actually
        // return a chunk size that depends on the index, at least
        // for regular grids (it ignores edges in other words).
        // so here we just retrive "chunk 0", store that, and adjust
        // for array edges in a separate function.
        let mut chk_shapes: HashMap<String, Vec<u64>> = HashMap::new();
        for (k, arr) in arrays.iter() {
            let chk_idx = vec![0; arr.shape().len()];
            chk_shapes.insert(k.to_owned(), arr.chunk_shape(&chk_idx)?.to_array_shape());
        }
        let chunk_shape = resolve_vector(&coordinates, chk_shapes)?;

        let mut chk_grid_shapes: HashMap<String, Vec<u64>> = HashMap::new();
        for (k, arr) in arrays.iter() {
            chk_grid_shapes.insert(k.to_owned(), arr.chunk_grid_shape().clone());
        }
        let chunk_grid_shape = resolve_vector(&coordinates, chk_grid_shapes)?;

        let mut arr_shapes: HashMap<String, Vec<u64>> = HashMap::new();
        for (k, arr) in arrays.iter() {
            arr_shapes.insert(k.to_owned(), arr.shape().to_vec());
        }
        let array_shape = resolve_vector(&coordinates, arr_shapes)?;

        // this runtime will handle the i/o. i/o tasks spawned in
        // that runtime will not share a thead pool with out (probably
        // compute heavy, blocking) tasks.
        let io_runtime = IoRuntime::try_new()?;

        Ok(Self {
            arrays: arrays.into_iter().map(|(k, a)| (k, Arc::new(a))).collect(),
            coordinates: Arc::new(coordinates),
            chunk_shape,
            chunk_grid_shape,
            array_shape,
            io_runtime,
            join_set: JoinSet::new(),
            state: None,
        })
    }

    /// return the chunk shape for a given index, taking into account
    /// the array edges where the "real" chunk is smaller than the
    /// chunk size in the metadata.
    fn get_chunk_shape(&self, chk_idx: &[u64]) -> ZarrQueryResult<Vec<u64>> {
        let is_edge_grid = chk_idx
            .iter()
            .zip(self.chunk_grid_shape.iter())
            .any(|(i, g)| i == &(g - 1));

        let mut chunk_shape = self.chunk_shape.clone();
        if is_edge_grid {
            chunk_shape = chk_idx
                .iter()
                .zip(self.array_shape.iter())
                .zip(chunk_shape.iter())
                .map(|((i, a), c)| std::cmp::min(a - i * c, *c))
                .collect();
        }

        Ok(chunk_shape)
    }

    fn get_array_interfaces(
        &self,
        cols: Vec<String>,
        chk_idx: Vec<u64>,
    ) -> ZarrQueryResult<Vec<ArrayInterface<T>>> {
        let full_chunk_shape = self.get_chunk_shape(&chk_idx)?;
        let arr_interfaces = cols
            .iter()
            .map(|col| {
                let arr = self
                    .arrays
                    .get(col)
                    .ok_or_else(|| ZarrQueryError::InvalidCompute("".into()))?
                    .clone();
                Ok(ArrayInterface::new(
                    col.to_string(),
                    arr,
                    self.coordinates.clone(),
                    full_chunk_shape.clone(),
                    chk_idx.clone(),
                ))
            })
            .collect::<ZarrQueryResult<Vec<_>>>()
            .unwrap();
        Ok(arr_interfaces)
    }

    /// this is the main function that does the heavy lifting, getting
    /// the data from the zarr store and decoding it.
    async fn get_chunk(
        &mut self,
        cols: Vec<String>,
        chk_idx: Vec<u64>,
        use_cached_value: bool,
        next_chunk_idx: Option<Vec<u64>>,
    ) -> ZarrQueryResult<ZarrInMemoryChunk> {
        if cols.is_empty() {
            return Err(ZarrQueryError::InvalidProjection(
                "No columns when polling zarr store for chunks".into(),
            ));
        }
        let mut chk_data = ZarrInMemoryChunk::new();

        // if there is a cached zarr chunk that was triggered from the
        // previous call, we use that.
        if use_cached_value & self.state.is_some() {
            let (mut rx, cached_idx, cached_cols) = self
                .state
                .take()
                .expect("Cached zarr received unexpectedly available");
            if chk_idx != cached_idx {
                return Err(ZarrQueryError::InvalidCompute(
                    "Cached zarr chunk index doesn't match requested chunk index".into(),
                ));
            }

            if cols != cached_cols {
                return Err(ZarrQueryError::InvalidCompute(
                    "Cached zarr chunk columns don't match requested columns".into(),
                ));
            }

            while let Some((data, arr_interface)) = rx.recv().await {
                let data = data?;
                let data = arr_interface.decode_data(data)?;
                chk_data.add_data(arr_interface.name, data);
            }
        }
        // if we are either not pre reading data, or we are but
        // this is the first call and there is no cached data yet,
        // we read the data now and wait for it to be ready.
        else {
            let arr_interfaces = self.get_array_interfaces(cols.clone(), chk_idx)?;
            let (tx, mut rx) = tokio::sync::mpsc::channel(arr_interfaces.len());
            for arr_interface in arr_interfaces {
                let tx_copy = tx.clone();
                let io_task = async move {
                    let b = arr_interface.read_bytes().await;
                    let _ = tx_copy.send((b, arr_interface)).await;
                };
                self.join_set.spawn_on(io_task, self.io_runtime.handle());

                if let Some((Ok(d), arr_int)) = rx.recv().await {
                    let data = arr_int.decode_data(d)?;
                    chk_data.add_data(arr_int.name, data);
                } else {
                    return Err(ZarrQueryError::InvalidCompute(
                        "Unable to retrieve decoded chunk".into(),
                    ));
                }
            }
        };

        // if the call was made with an index for the next chunk, we
        // submit a job to read that next chunk before returning,
        // so that we can fetch the data while other operations run
        // between now and the next call to this function.
        if let Some(next_chunk_idx) = next_chunk_idx {
            let arr_interfaces = self.get_array_interfaces(cols.clone(), next_chunk_idx.clone())?;
            let (tx, rx) = tokio::sync::mpsc::channel(arr_interfaces.len());
            for arr_interface in arr_interfaces {
                let tx_copy = tx.clone();
                let io_task = async move {
                    let b = arr_interface.read_bytes().await;
                    let _ = tx_copy.send((b, arr_interface)).await;
                };
                self.join_set.spawn_on(io_task, self.io_runtime.handle());
            }
            self.state = Some((rx, next_chunk_idx, cols));
        };

        Ok(chk_data)
    }
}

/// A stream of RecordBatches read from a Zarr store.
///
/// This struct is separate from `ZarrRecordBatchStream`, so that we can avoid manually
/// implementing [`Stream`]. Instead, we use the `async-stream` crate to convert an async iterable
/// into a stream.
struct ZarrRecordBatchStreamInner<T: AsyncReadableListableStorageTraits + ?Sized> {
    zarr_store: Arc<ZarrStore<T>>,
    projected_schema_ref: SchemaRef,
    schema_without_filter_cols: Option<SchemaRef>,
    filter: Option<ZarrChunkFilter>,
    chunk_indices: VecDeque<Vec<u64>>,
}

impl<T: AsyncReadableListableStorageTraits + ?Sized + 'static> ZarrRecordBatchStreamInner<T> {
    /// Create a new ZarrRecordBatchStreamInner.
    ///
    /// This function is intentionally private, as all users should call
    /// [`ZarrRecordBatchStream::new`] instead.
    async fn new(
        store: Arc<T>,
        schema_ref: SchemaRef,
        prefix: Option<String>,
        projection: Option<Vec<usize>>,
        n_partitions: usize,
        partition: usize,
    ) -> ZarrQueryResult<Self> {
        // quick check to make sure the partition we're reading from does
        // not exceed the number of partitions.
        if partition >= n_partitions {
            return Err(ZarrQueryError::InvalidCompute(
                "Parition number exceeds number of partition in zarr stream".into(),
            ));
        }

        // if there is a projection provided, modify the schema.
        let projected_schema_ref = match projection {
            Some(proj) => Arc::new(schema_ref.project(&proj)?),
            None => schema_ref.clone(),
        };

        // the prefix is necessary when reading from some remote
        // stores that don't work off of the url and require a
        // prefix. for example aws s3 object store doesn't seem
        // to use the url, just the bucket, so the path to the
        // actual zarr store needs to be provided separately.
        let prefix = if let Some(prefix) = prefix {
            ["/".into(), prefix].join("")
        } else {
            "/".to_string()
        };

        // this will extract column (i.e. array) names based (possibly
        // projected) schema.
        let cols: Vec<_> = projected_schema_ref
            .fields()
            .iter()
            .map(|f| f.name())
            .collect();

        // open all the arrays based on the column names.
        let mut arrays: HashMap<String, Array<T>> = HashMap::new();
        for col in &cols {
            let path = PathBuf::from(&prefix)
                .join(col)
                .into_os_string()
                .to_str()
                .ok_or(ZarrQueryError::InvalidMetadata(
                    "could not form path from group and column name".into(),
                ))?
                .to_string();
            let arr = Array::async_open(store.clone(), &path).await?;
            arrays.insert(col.to_string(), arr);
        }

        // store all the zarr arrays in a struct that we can use
        // to access them later.
        let zarr_store = Arc::new(ZarrStore::new(arrays, projected_schema_ref.clone())?);

        // this creates all the chunk indices we will be reading from.
        let chk_grid_shape = &zarr_store.chunk_grid_shape;
        let mut chunk_indices: Vec<_> = match chk_grid_shape.len() {
            1 => (0..chk_grid_shape[0]).map(|i| vec![i]).collect(),
            2 => {
                let d0: Vec<_> = (0..chk_grid_shape[0]).collect();
                let d1: Vec<_> = (0..chk_grid_shape[1]).collect();
                iproduct!(d0, d1).map(|(x, y)| vec![x, y]).collect()
            }
            3 => {
                let d0: Vec<_> = (0..chk_grid_shape[0]).collect();
                let d1: Vec<_> = (0..chk_grid_shape[1]).collect();
                let d2: Vec<_> = (0..chk_grid_shape[2]).collect();
                iproduct!(d0, d1, d2)
                    .map(|(x, y, z)| vec![x, y, z])
                    .collect()
            }
            _ => {
                return Err(ZarrQueryError::InvalidMetadata(
                    "Only 1, 2 or 3D arrays supported".into(),
                ))
            }
        };
        let chunks_per_partitions = chunk_indices.len().div_ceil(n_partitions);
        let max_idx = chunk_indices.len();
        let start = chunks_per_partitions * partition;
        let end = min(chunks_per_partitions * (partition + 1), max_idx);

        // this is to handle cases where more partitions than there are
        // chunks to read were requested.
        if end <= start {
            chunk_indices = Vec::new();
        } else {
            chunk_indices = chunk_indices[start..end].to_vec();
        }
        let chunk_indices = VecDeque::from(chunk_indices);

        Ok(Self {
            zarr_store,
            projected_schema_ref,
            filter: None,
            chunk_indices,
            schema_without_filter_cols: None,
        })
    }

    /// Fetch the next chunk, returning None if there are no more chunks.
    pub(crate) async fn next_chunk(&mut self) -> Result<Option<RecordBatch>, ArrowError> {
        // the logic here is not trivial so it wararnts a few explanations.
        // if there is a filter to apply, we read whatever data is needed to
        // evaluate it. we do pre fetch chunks here, when calling get_chunk,
        // and we keep going through the chunk indices until we find a chunk
        // where the filter condition is satisfied.
        //
        // when we do find such a chunk, we move on to the next stage, which
        // is to read the data for the actual query. we do save the data we
        // read to evaluate the filter, because some of it might also be
        // requested in the query. we don't request columns if they are already
        // present in the filter data, and then combine the filter data with
        // the data for the chunk for the main query. if there is a filter,
        // we can't pre fetch the data when reading the data for the main
        // query, because the next time we read some data it would be for the
        // fitler, not for the main query.

        let mut chunk_index: Option<Vec<u64>> = None;
        let mut filter_zarr_chunk: Option<ZarrInMemoryChunk> = None;

        let filter = self.filter.take();
        if let Some(filter) = filter {
            let mut filter_passed = false;
            while !filter_passed {
                chunk_index = self.pop_chunk_idx();
                if let Some(chunk_index) = &chunk_index {
                    let next_chnk_idx = self.see_chunk_idx();
                    let column_names: Vec<_> = filter
                        .schema_ref()
                        .fields()
                        .iter()
                        .map(|f| f.name().to_owned())
                        .collect();
                    let zarr_chunk = Arc::get_mut(&mut self.zarr_store)
                        .expect("Zarr store pointer unexpectedly not unique")
                        .get_chunk(column_names, chunk_index.clone(), true, next_chnk_idx)
                        .await?;
                    filter_passed = zarr_chunk.check_filter(&filter)?;
                    filter_zarr_chunk = Some(zarr_chunk);
                } else {
                    filter_passed = true;
                }
            }
            self.filter = Some(filter);
        } else {
            chunk_index = self.pop_chunk_idx();
        }

        if let Some(chunk_index) = chunk_index {
            let mut zarr_chunk;
            if self.filter.is_some() {
                let filter_zarr_chunk = filter_zarr_chunk.expect("Filter zarr chunk missing.");
                let schema = self
                    .schema_without_filter_cols
                    .as_ref()
                    .expect("Schema without filter columns is missing.");

                let column_names: Vec<_> = schema
                    .fields()
                    .iter()
                    .map(|f| f.name().to_owned())
                    .collect();
                zarr_chunk = Arc::get_mut(&mut self.zarr_store)
                    .expect("Zarr store pointer unexpectedly not unique")
                    .get_chunk(column_names, chunk_index, false, None)
                    .await?;
                zarr_chunk.combine(filter_zarr_chunk);
            } else {
                let next_chnk_idx = self.see_chunk_idx();
                let column_names: Vec<_> = self
                    .projected_schema_ref
                    .fields()
                    .iter()
                    .map(|f| f.name().to_owned())
                    .collect();

                zarr_chunk = Arc::get_mut(&mut self.zarr_store)
                    .expect("Zarr store pointer unexpectedly not unique")
                    .get_chunk(column_names, chunk_index, true, next_chnk_idx)
                    .await?;
            }

            let record_batch = zarr_chunk.into_record_batch(&self.projected_schema_ref)?;
            Ok(Some(record_batch))
        } else {
            Ok(None)
        }
    }

    /// Convert this into a `ZarrRecordBatchStream`, using the `async-stream` crate to handle the
    /// low-level specifics of stream polling.
    fn into_stream(mut self) -> ZarrRecordBatchStream {
        let schema = self.projected_schema_ref.clone();
        let stream = Box::pin(try_stream! {
            while let Some(batch) = self.next_chunk().await? {
                yield batch;
            }
        });
        ZarrRecordBatchStream { stream, schema }
    }

    fn pop_chunk_idx(&mut self) -> Option<Vec<u64>> {
        self.chunk_indices.pop_front()
    }

    fn see_chunk_idx(&self) -> Option<Vec<u64>> {
        self.chunk_indices.front().cloned()
    }

    /// adds a filter to avoid reading whole chunks if no values
    /// in the corresponding arrays pass the check. this is not to
    /// filter out values within a chunk, we rely on datafusion's
    /// default filtering for that. basically this here is to handle
    /// filter pushdowns.
    fn with_filter(mut self, filter: ZarrChunkFilter) -> ZarrQueryResult<Self> {
        // because we'll need to read the filter data first, evaluate
        // the filter, then read the data for the main query, we want
        // to re-use the filter data if it's also requested in the
        // query, so here we build the schema for the columns that
        // are requested in the query, but not in the filter predicate.
        let fields: Vec<_> = self
            .projected_schema_ref
            .fields()
            .iter()
            .filter(|f| filter.schema_ref().index_of(f.name()).is_err())
            .cloned()
            .collect();
        let schema = Schema::new(fields);
        self.schema_without_filter_cols = Some(Arc::new(schema));

        // set the filter on the inner stream.
        self.filter = Some(filter);

        Ok(self)
    }
}

/// An async stream of record batches read from the Zarr store.
///
/// This implementation is modeled to be used with the DataFusion [`RecordBatchStream`] trait.
///
/// [`RecordBatchStream`]: https://docs.rs/datafusion/latest/datafusion/execution/trait.RecordBatchStream.html
pub struct ZarrRecordBatchStream {
    stream: BoxStream<'static, Result<RecordBatch, ArrowError>>,
    schema: SchemaRef,
}

impl ZarrRecordBatchStream {
    /// Create a new ZarrRecordBatchStream.
    pub async fn try_new<T: AsyncReadableListableStorageTraits + ?Sized + 'static>(
        store: Arc<T>,
        schema_ref: SchemaRef,
        prefix: Option<String>,
        projection: Option<Vec<usize>>,
        n_partitions: usize,
        partition: usize,
        filter: Option<ZarrChunkFilter>,
    ) -> ZarrQueryResult<Self> {
        let mut inner = ZarrRecordBatchStreamInner::new(
            store,
            schema_ref,
            prefix,
            projection,
            n_partitions,
            partition,
        )
        .await?;

        if let Some(filter) = filter {
            inner = inner.with_filter(filter)?;
        }

        Ok(Self {
            schema: inner.projected_schema_ref.clone(),
            stream: inner.into_stream().stream,
        })
    }

    /// A reference to the schema of the record batches produced by this stream.
    pub fn schema_ref(&self) -> &SchemaRef {
        &self.schema
    }

    /// The schema of the record batches produced by this stream.
    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl Stream for ZarrRecordBatchStream {
    type Item = Result<RecordBatch, ArrowError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.stream).poll_next(cx)
    }
}

#[cfg(test)]
mod zarr_stream_tests {
    use futures_util::TryStreamExt;

    use super::*;
    use crate::test_utils::{
        extract_col, get_local_zarr_store, get_local_zarr_store_mix_dims, validate_names_and_types,
        validate_primitive_column,
    };
    use crate::zarr_store_opener::ZarrArrowPredicate;

    // this is just to help with testing the filter pushdown
    // functionality, since the full implementation is not done
    // in this module.
    struct DummyPredicate {}

    impl ZarrArrowPredicate for DummyPredicate {
        fn evaluate(&self, batch: &RecordBatch) -> Result<BooleanArray, ArrowError> {
            let lat_values = extract_col::<Float64Type>("lat", batch);
            let lon_values = extract_col::<Float64Type>("lon", batch);
            let bools: Vec<_> = lat_values
                .iter()
                .zip(lon_values.iter())
                .map(|(lat, lon)| *lat < 41.0 && *lon > -118.0)
                .collect();
            Ok(bools.into())
        }
    }

    #[tokio::test]
    async fn read_data_test() {
        let (wrapper, schema) = get_local_zarr_store(true, 0.0, "lat_lon_data").await;
        let store = wrapper.get_store();

        let stream = ZarrRecordBatchStream::try_new(store, schema, None, None, 1, 0, None)
            .await
            .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        let target_types = HashMap::from([
            ("lat".to_string(), DataType::Float64),
            ("lon".to_string(), DataType::Float64),
            ("data".to_string(), DataType::Float64),
        ]);
        validate_names_and_types(&target_types, &records[0]);
        assert_eq!(records.len(), 9);

        // the top left chunk, full 3x3
        validate_primitive_column::<Float64Type, f64>(
            "lat",
            &records[0],
            &[35., 35., 35., 36., 36., 36., 37., 37., 37.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            &records[0],
            &[
                -120.0, -119.0, -118.0, -120.0, -119.0, -118.0, -120.0, -119.0, -118.0,
            ],
        );
        validate_primitive_column::<Float64Type, f64>(
            "data",
            &records[0],
            &[0.0, 1.0, 2.0, 8.0, 9.0, 10.0, 16.0, 17.0, 18.0],
        );

        // the top right chunk, 3 x 2
        validate_primitive_column::<Float64Type, f64>(
            "lat",
            &records[2],
            &[35., 35., 36., 36., 37., 37.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            &records[2],
            &[-114.0, -113.0, -114.0, -113.0, -114.0, -113.0],
        );
        validate_primitive_column::<Float64Type, f64>(
            "data",
            &records[2],
            &[6.0, 7.0, 14.0, 15.0, 22.0, 23.0],
        );

        // the bottom right chunk, 2 x 2
        validate_primitive_column::<Float64Type, f64>(
            "lat",
            &records[8],
            &[41.0, 41.0, 42.0, 42.0],
        );
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            &records[8],
            &[-114.0, -113.0, -114.0, -113.0],
        );
        validate_primitive_column::<Float64Type, f64>(
            "data",
            &records[8],
            &[54.0, 55.0, 62.0, 63.0],
        );
    }

    #[tokio::test]
    async fn filter_test() {
        let (wrapper, schema) = get_local_zarr_store(true, 0.0, "lat_lon_data_with_filter").await;
        let store = wrapper.get_store();

        // note: we need to project the schema to match what the filter
        // predicate checks, since here we're manually creating a chunk
        // filter. a proper impmlementation of the chunk filter creation
        // (e.g. see the [`create_zarr_chunk_filter`]) should handle this.
        let filter = Some(
            ZarrChunkFilter::new(
                vec![Box::new(DummyPredicate {})],
                Arc::new(schema.project(&[1, 2]).unwrap()),
            )
            .unwrap(),
        );

        let stream = ZarrRecordBatchStream::try_new(store, schema, None, None, 1, 0, filter)
            .await
            .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        let target_types = HashMap::from([
            ("lat".to_string(), DataType::Float64),
            ("lon".to_string(), DataType::Float64),
            ("data".to_string(), DataType::Float64),
        ]);
        validate_names_and_types(&target_types, &records[0]);

        // this tests for the filter push down, which doesn't completely
        // filter out the results, it only drops chunks of data where
        // not a single "row" passes the filter, so the condition weinto()
        // are checking lines up with the data in the chunks, and is
        // a bit different from the WHERE clause.
        assert_eq!(records.len(), 4);
        for batch in records {
            let lat_values = extract_col::<Float64Type>("lat", &batch);
            let lon_values = extract_col::<Float64Type>("lon", &batch);
            assert!(lat_values
                .iter()
                .zip(lon_values.iter())
                .all(|(lat, lon)| *lat < 41.0 && *lon > -118.0));
        }
    }

    #[tokio::test]
    async fn dimension_tests() {
        // this store will have 2d lat coordinates and 1d lon coordinates.
        // that shoudl effecitvely given the same as 1d and 1d.
        let (wrapper, schema) = get_local_zarr_store_mix_dims(0.0, "lat_lon_mixed_dims_data").await;
        let store = wrapper.get_store();

        let stream = ZarrRecordBatchStream::try_new(store, schema, None, None, 1, 0, None)
            .await
            .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        let target_types = HashMap::from([
            ("lat".to_string(), DataType::Float64),
            ("lon".to_string(), DataType::Float64),
            ("data".to_string(), DataType::Float64),
        ]);
        validate_names_and_types(&target_types, &records[0]);
        assert_eq!(records.len(), 9);

        // the top left chunk, full 3x3
        validate_primitive_column::<Float64Type, f64>(
            "lat",
            &records[0],
            &[35., 35., 35., 36., 36., 36., 37., 37., 37.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            &records[0],
            &[
                -120.0, -119.0, -118.0, -120.0, -119.0, -118.0, -120.0, -119.0, -118.0,
            ],
        );
        validate_primitive_column::<Float64Type, f64>(
            "data",
            &records[0],
            &[0.0, 1.0, 2.0, 8.0, 9.0, 10.0, 16.0, 17.0, 18.0],
        );
    }

    #[tokio::test]
    async fn read_missing_chunks_test() {
        let fillvalue = 1234.0;
        let (wrapper, schema) = get_local_zarr_store(false, fillvalue, "lat_lon_empty_data").await;
        let store = wrapper.get_store();

        let stream = ZarrRecordBatchStream::try_new(store, schema, None, None, 1, 0, None)
            .await
            .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        let target_types = HashMap::from([
            ("lat".to_string(), DataType::Float64),
            ("lon".to_string(), DataType::Float64),
            ("data".to_string(), DataType::Float64),
        ]);
        validate_names_and_types(&target_types, &records[0]);
        assert_eq!(records.len(), 9);

        // the top left chunk, full 3x3, but "data" is missing.
        validate_primitive_column::<Float64Type, f64>(
            "lat",
            &records[0],
            &[35., 35., 35., 36., 36., 36., 37., 37., 37.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            &records[0],
            &[
                -120.0, -119.0, -118.0, -120.0, -119.0, -118.0, -120.0, -119.0, -118.0,
            ],
        );
        validate_primitive_column::<Float64Type, f64>("data", &records[0], &[fillvalue; 9]);
    }

    #[tokio::test]
    async fn read_with_partition_test() {
        let (wrapper, schema) =
            get_local_zarr_store(true, 0.0, "lat_lon_data_with_partition").await;
        let store = wrapper.get_store();

        let target_types = HashMap::from([
            ("lat".to_string(), DataType::Float64),
            ("lon".to_string(), DataType::Float64),
            ("data".to_string(), DataType::Float64),
        ]);

        let stream =
            ZarrRecordBatchStream::try_new(store.clone(), schema.clone(), None, None, 2, 0, None)
                .await
                .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();
        validate_names_and_types(&target_types, &records[0]);
        assert_eq!(records.len(), 5);

        let stream = ZarrRecordBatchStream::try_new(store, schema, None, None, 2, 1, None)
            .await
            .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();
        validate_names_and_types(&target_types, &records[0]);
        assert_eq!(records.len(), 4);

        // the full data has 3x3 chunks, the first partition would
        // read the first 5, the second one the last 4, so the first
        // chunk of the second stream would effectively be the middle
        // right chunk of the full data.
        validate_primitive_column::<Float64Type, f64>(
            "lat",
            &records[0],
            &[38., 38., 39., 39., 40., 40.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            &records[0],
            &[-114.0, -113.0, -114.0, -113.0, -114.0, -113.0],
        );
        validate_primitive_column::<Float64Type, f64>(
            "data",
            &records[0],
            &[30.0, 31.0, 38.0, 39.0, 46.0, 47.0],
        );
    }

    #[tokio::test]
    async fn read_too_many_partitions_test() {
        let (wrapper, schema) =
            get_local_zarr_store(true, 0.0, "lat_lon_data_too_many_partition").await;
        let store = wrapper.get_store();

        // there are only 9 chunks, asking for 20 partitions, so each partition up to
        // the 9th parittion should have one batch in them, after that there should be
        // no data returned by the streams.
        let stream =
            ZarrRecordBatchStream::try_new(store.clone(), schema.clone(), None, None, 20, 0, None)
                .await
                .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();
        assert_eq!(records.len(), 1);

        let stream =
            ZarrRecordBatchStream::try_new(store.clone(), schema.clone(), None, None, 20, 8, None)
                .await
                .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();
        assert_eq!(records.len(), 1);

        let stream =
            ZarrRecordBatchStream::try_new(store.clone(), schema.clone(), None, None, 20, 10, None)
                .await
                .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();
        assert_eq!(records.len(), 0);

        let stream = ZarrRecordBatchStream::try_new(store, schema, None, None, 20, 19, None)
            .await
            .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();
        assert_eq!(records.len(), 0);
    }
}
