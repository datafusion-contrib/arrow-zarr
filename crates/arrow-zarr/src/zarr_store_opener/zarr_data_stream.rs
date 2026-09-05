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

use std::cmp::min;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

use arrow::array::*;
use arrow::compute::kernels::aggregate::{max as arrow_max, min as arrow_min};
use arrow::datatypes::*;
use arrow::record_batch::RecordBatch;
use arrow_schema::ArrowError;
use async_stream::try_stream;
use bytes::Bytes;
use datafusion::common::ScalarValue;
use datafusion::logical_expr::ColumnarValue;
use datafusion::physical_expr::utils::{collect_columns, reassign_expr_columns};
use datafusion::physical_plan::PhysicalExpr;
use futures::stream::{BoxStream, Stream};
use itertools::{izip, Itertools};
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
use super::zarr_errors::{ZarrQueryError, ZarrQueryResult};
use crate::geospatial::probe_pruning_expr::ProbePruningExpr;

/// A struct to handle 1-D coordinate arrays stored in reduced form, and
/// "broadcasting" them up to the full chunk shape when reading multidimensional
/// data. An already-broadcast N-D coordinate (e.g. a 2-D `lat` with dimensions
/// `[lat, lon]`) is not tracked here.
#[derive(Debug)]
pub(crate) struct ZarrBroadcastableCoordinates {
    // the position of each broadcastable coordinate in the overall chunk shape.
    // the coordinates are arrays that contain data that characterizes a
    // dimension, such as time, or a longitude or latitude.
    coord_positions: HashMap<String, usize>,
}

impl ZarrBroadcastableCoordinates {
    /// Build the coordinate map from each array's dimensionality and (optional)
    /// dimension names. Taking cached `(dim_names, ndim)` metadata rather than the
    /// live arrays lets the planning-time statistics reuse this classification.
    pub(crate) fn new(
        array_dims: &HashMap<String, (Option<Vec<String>>, usize)>,
        schema_ref: SchemaRef,
    ) -> ZarrQueryResult<Self> {
        // classify each array. a "broadcastable coordinate" is a 1-D array that
        // names its own single dimension; it is stored reduced and must be
        // broadcast up to the full chunk shape. every other array (data, or an
        // already-broadcast N-D coordinate) is a "full" array that already spans
        // the chunk. the full arrays that carry dimension names must all agree,
        // and that shared order is what we position the coordinates within.
        let mut coords: Vec<String> = Vec::new();
        let mut dim_order: Option<Vec<String>> = None;
        let mut unnamed_multidim = false;

        for (name, (names, ndim)) in array_dims {
            if *ndim == 1 && names.as_deref() == Some(std::slice::from_ref(name)) {
                coords.push(name.clone());
            } else if let Some(names) = names {
                match &dim_order {
                    Some(existing) if existing != names => {
                        return Err(ZarrQueryError::InvalidMetadata(
                            "Dimension names must be consistent across arrays".into(),
                        ));
                    }
                    _ => dim_order = Some(names.clone()),
                }
            } else if *ndim > 1 {
                // a multi-dim array without dimension names leaves us no way to
                // place coordinates onto its axes.
                unnamed_multidim = true;
            }
        }

        let mut coord_positions: HashMap<String, usize> = HashMap::new();
        if coords.is_empty() {
            // nothing to broadcast.
        } else if unnamed_multidim {
            return Err(ZarrQueryError::InvalidMetadata(
                "With mixed array dimensionality, dimension names are required".into(),
            ));
        } else if coords.len() == array_dims.len() {
            // every array is a coordinate: the chunk dimensionality equals the
            // number of coordinates, so we take their order from the schema.
            for (pos, field) in schema_ref.fields().iter().enumerate() {
                coord_positions.insert(field.name().to_string(), pos);
            }
        } else if let Some(dim_order) = dim_order {
            // a named full array is present, so the shared dimension order is
            // known. place each coordinate within it.
            for coord in coords {
                let pos = dim_order.iter().position(|d| d == &coord).ok_or(
                    ZarrQueryError::InvalidMetadata(
                        "Could not find coordinate in dimension names".into(),
                    ),
                )?;
                coord_positions.insert(coord, pos);
            }
        }
        // otherwise the only non-coordinate arrays are 1-D and unnamed, so the
        // output is 1-D and nothing needs broadcasting.

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

    /// broadacast a 1D array to a ND array if the variable is a coordinate.
    /// note that we return a 1D vector, but this is just because we map all
    /// the chunk to columnar data, so a m x n array gets mapped to a 1D
    /// vector of length m x n.
    fn broadcast_if_coord<T: Clone>(
        &self,
        coord_name: &str,
        data: Vec<T>,
        full_chunk_shape: &[u64],
    ) -> ZarrQueryResult<Vec<T>> {
        let dim_idx = match self.get_coord_position(coord_name) {
            Some(dim_idx) if full_chunk_shape.len() > 1 => dim_idx,
            // not a coordinate, or 1D data: nothing to broadcast.
            _ => return Ok(data),
        };

        // in a row-major (C-order) flattened array, a coordinate along dimension
        // dim_idx has a value that depends only on that dimension's index. so we
        // repeat each value across all the faster-varying (inner) dimensions, then
        // tile that whole block across all the slower-varying (outer) dimensions.
        let inner = full_chunk_shape[dim_idx + 1..].iter().product::<u64>() as usize;
        let outer = full_chunk_shape[..dim_idx].iter().product::<u64>() as usize;

        let block: Vec<T> = data
            .into_iter()
            .flat_map(|v| std::iter::repeat_n(v, inner))
            .collect();
        Ok(vec![&block[..]; outer].concat())
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
    coords: Arc<ZarrBroadcastableCoordinates>,
    full_chunk_shape: Vec<u64>,
    chk_index: Vec<u64>,
}

/// in most cases, we will read encoded bytes and decode them after,
/// but in the case of a missing chunk the result of the read operation
/// will be none. in a few cases though we will read pre-decoded bytes,
/// hence why we have this enum.
enum BytesFromArray {
    // since this is used to basically transfer decoded bytes between the
    // read and decode stages, it's easier if the enum just owns the data.
    // ArrayBytes wraps a Cow<u8>, and because it borrows data there's a
    // lifetime associated with it. so long story short, the static lifetime
    // is there because the ArrayBytes comes from calling into_owned() on
    // the ArrayBytes we get from reading the zarr data.
    Decoded(ArrayBytes<'static>),
    Encoded(Option<Bytes>),
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

impl<T: AsyncReadableListableStorageTraits + ?Sized + 'static> ArrayInterface<T> {
    fn new(
        name: String,
        arr: Arc<Array<T>>,
        coords: Arc<ZarrBroadcastableCoordinates>,
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
            let ranges: Vec<_> = izip!(&self.chk_index, arr_shape.iter(), chunk_shape.iter())
                .map(|(i, a, c)| 0..min(a - i * c, *c))
                .collect();

            let array_subset = ArraySubset::new_with_ranges(&ranges);
            let data = self
                .arr
                .async_retrieve_chunk_subset::<ArrayBytes>(&self.chk_index, &array_subset)
                .await?;
            Ok(BytesFromArray::Decoded(data.into_owned()))
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
                } else {
                    let chk_shp = self
                        .coords
                        .reduce_if_coord(self.full_chunk_shape.clone(), &self.name);
                    let num_elems = chk_shp.iter().product::<u64>();
                    ArrayBytes::new_fill_value(
                        self.arr.data_type(),
                        num_elems,
                        self.arr.fill_value(),
                    )
                    .map_err(|e| ZarrQueryError::External(Box::new(e)))?
                }
            }
            BytesFromArray::Decoded(bytes) => bytes,
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

        if t.is::<BoolDataType>() {
            return_array_ref!(BooleanArray, bool)
        } else if t.is::<UInt8DataType>() {
            return_array_ref!(PrimitiveArray<UInt8Type>, u8)
        } else if t.is::<UInt16DataType>() {
            return_array_ref!(PrimitiveArray<UInt16Type>, u16)
        } else if t.is::<UInt32DataType>() {
            return_array_ref!(PrimitiveArray<UInt32Type>, u32)
        } else if t.is::<UInt64DataType>() {
            return_array_ref!(PrimitiveArray<UInt64Type>, u64)
        } else if t.is::<Int8DataType>() {
            return_array_ref!(PrimitiveArray<Int8Type>, i8)
        } else if t.is::<Int16DataType>() {
            return_array_ref!(PrimitiveArray<Int16Type>, i16)
        } else if t.is::<Int32DataType>() {
            return_array_ref!(PrimitiveArray<Int32Type>, i32)
        } else if t.is::<Int64DataType>() {
            return_array_ref!(PrimitiveArray<Int64Type>, i64)
        } else if t.is::<Float32DataType>() {
            return_array_ref!(PrimitiveArray<Float32Type>, f32)
        } else if t.is::<Float64DataType>() {
            return_array_ref!(PrimitiveArray<Float64Type>, f64)
        } else if t.is::<StringDataType>() {
            return_array_ref!(StringArray, String)
        } else if let Some(dt) = t.downcast_ref::<NumpyDateTime64DataType>() {
            ensure_unit_scale(t, dt.scale_factor.get())?;
            match dt.unit {
                NumpyTimeUnit::Second => return_array_ref!(TimestampSecondArray, i64),
                NumpyTimeUnit::Millisecond => return_array_ref!(TimestampMillisecondArray, i64),
                NumpyTimeUnit::Microsecond => return_array_ref!(TimestampMicrosecondArray, i64),
                NumpyTimeUnit::Nanosecond => return_array_ref!(TimestampNanosecondArray, i64),
                _ => Err(ZarrQueryError::InvalidType(format!(
                    "Unsupported datetime64 unit {} from zarr metadata",
                    dt.unit
                ))),
            }
        } else if let Some(dt) = t.downcast_ref::<NumpyTimeDelta64DataType>() {
            ensure_unit_scale(t, dt.scale_factor.get())?;
            match dt.unit {
                NumpyTimeUnit::Second => return_array_ref!(DurationSecondArray, i64),
                NumpyTimeUnit::Millisecond => return_array_ref!(DurationMillisecondArray, i64),
                NumpyTimeUnit::Microsecond => return_array_ref!(DurationMicrosecondArray, i64),
                NumpyTimeUnit::Nanosecond => return_array_ref!(DurationNanosecondArray, i64),
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

    /// gather the chunk's arrays in schema-field order, each paired with its
    /// name, erroring if any field's array is missing from the chunk.
    fn columns_in_schema_order(&self, schema: &Schema) -> ZarrQueryResult<Vec<(String, ArrayRef)>> {
        schema
            .fields()
            .iter()
            .map(|f| {
                let ar = self.data.get(f.name()).cloned().ok_or(
                    ZarrQueryError::InvalidColumnRequest("Array missing from array map".into()),
                )?;
                Ok((f.name().to_string(), ar))
            })
            .collect()
    }

    /// this checks if any "row" in the chunk passes the filter
    /// condition. this function does not consume the chunk.
    fn check_filter(&self, filter: &ZarrChunkFilter) -> Result<bool, ArrowError> {
        let array_refs = self.columns_in_schema_order(filter.schema_ref())?;
        let rec_batch = RecordBatch::try_from_iter(array_refs)?;
        filter.evaluate(&rec_batch)
    }

    /// the columns in the record batch will be ordered following
    /// the field names in the schema.
    fn into_record_batch(self, schema: &SchemaRef) -> ZarrQueryResult<RecordBatch> {
        let array_refs = self.columns_in_schema_order(schema)?;
        RecordBatch::try_from_iter(array_refs).map_err(|e| ZarrQueryError::External(Box::new(e)))
    }

    /// Builds a 2-row record batch holding, per column, that column's min (row 0)
    /// and max (row 1) in schema-field order. An all-null/empty column yields
    /// `[null, null]`. Used to cheaply test a chunk's coordinate envelope against
    /// a pushed-down spatial filter without materializing the whole chunk.
    fn to_min_max_record_batch(&self, schema: &SchemaRef) -> ZarrQueryResult<RecordBatch> {
        let min_max = self
            .columns_in_schema_order(schema)?
            .into_iter()
            .map(|(name, arr)| Ok((name, min_max_array(&arr)?)))
            .collect::<ZarrQueryResult<Vec<_>>>()?;
        RecordBatch::try_from_iter(min_max).map_err(|e| ZarrQueryError::External(Box::new(e)))
    }
}

/// Returns a length-2 array `[min, max]` of the same numeric type as `array`.
/// `None` min/max (empty or all-null input) become nulls.
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

/// A wrapper for a map of arrays, which will handle interleaving
/// reading and decoding data from zarr storage.
type ZarrReceiver<T> = Receiver<(ZarrQueryResult<BytesFromArray>, ArrayInterface<T>)>;

struct ZarrStore<T: AsyncReadableListableStorageTraits + ?Sized> {
    arrays: HashMap<String, Arc<Array<T>>>,
    coordinates: Arc<ZarrBroadcastableCoordinates>,
    chunk_shape: Vec<u64>,
    chunk_grid_shape: Vec<u64>,
    array_shape: Vec<u64>,
    io_runtime: IoRuntime,
    join_set: JoinSet<()>,
    state: Option<(ZarrReceiver<T>, Vec<u64>, Vec<String>)>,
}

/// this function handles vectors that have 1 value for each dimension
/// of the array. some of those vectors can correspond to coordinates,
/// which are 1D arrays and therefore will only have 1 element in this
/// vector.
pub(crate) fn resolve_vector(
    coords: &ZarrBroadcastableCoordinates,
    vecs: HashMap<String, Vec<u64>>,
) -> ZarrQueryResult<Vec<u64>> {
    let mismatch =
        || ZarrQueryError::InvalidMetadata("Mismatch between vectors for different arrays".into());

    // the full vector comes from the non-coordinate arrays; they must all agree.
    let mut final_vec: Option<&Vec<u64>> = None;
    for (k, vec) in &vecs {
        if !coords.is_coordinate(k) {
            match final_vec {
                Some(fv) if fv != vec => return Err(mismatch()),
                _ => final_vec = Some(vec),
            }
        }
    }

    if let Some(final_vec) = final_vec {
        // each coordinate is 1D; its single element must match the full vector
        // at the coordinate's position in the chunk dimensionality. for example,
        // if the full vector is [l, m, n] and this array is the coordinate for the
        // second dimension, it must be a 1D array whose one element equals m.
        for (k, vec) in &vecs {
            if let Some(pos) = coords.get_coord_position(k) {
                if final_vec[pos] != vec[0] {
                    return Err(mismatch());
                }
            }
        }
        Ok(final_vec.clone())
    // the else branch here happens if all the arrays are coordinates.
    } else {
        let mut final_vec: Vec<u64> = vec![0; coords.coord_positions.len()];
        for (k, p) in coords.coord_positions.iter() {
            final_vec[*p] = vecs.get(k).ok_or(ZarrQueryError::InvalidColumnRequest(
                "Array is missing from array map".into(),
            ))?[0];
        }
        Ok(final_vec)
    }
}

/// collect an array's dimension names, returning `None` when the array carries
/// no dimension-name metadata and erroring if any individual name is null.
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

impl<T: AsyncReadableListableStorageTraits + ?Sized + 'static> ZarrStore<T> {
    async fn new(
        cols: &Vec<&String>,
        prefix: &str,
        store: Arc<T>,
        schema_ref: SchemaRef,
    ) -> ZarrQueryResult<Self> {
        // open all the arrays based on the column names.
        let mut arrays: HashMap<String, Array<T>> = HashMap::new();
        for col in cols {
            let path = PathBuf::from(&prefix)
                .join(col)
                .to_str()
                .ok_or(ZarrQueryError::InvalidCompute(
                    "Could not form path from group and column name".into(),
                ))?
                .to_string();
            let arr = Array::async_open(store.clone(), &path).await?;
            arrays.insert(col.to_string(), arr);
        }

        // determine which column, if any, represents a coordinate. the
        // classification only needs each array's dimension names and its
        // dimensionality, so we hand it that cached metadata.
        let arr_dims: HashMap<String, (Option<Vec<String>>, usize)> = arrays
            .iter()
            .map(|(k, arr)| Ok((k.clone(), (dim_names(arr)?, arr.dimensionality()))))
            .collect::<ZarrQueryResult<_>>()?;
        let coordinates = ZarrBroadcastableCoordinates::new(&arr_dims, schema_ref)?;

        // technically getting the chunk shape requires a chunk
        // index, but it seems the zarrs library doesn't actually
        // return a chunk size that depends on the index, at least
        // for regular grids (it ignores edges in other words).
        // so here we just retrieve "chunk 0", store that, and adjust
        // for array edges in a separate function.
        let mut chk_shapes: HashMap<String, Vec<u64>> = HashMap::new();
        let mut chk_grid_shapes: HashMap<String, Vec<u64>> = HashMap::new();
        let mut arr_shapes: HashMap<String, Vec<u64>> = HashMap::new();
        for (k, arr) in arrays.iter() {
            let chk_idx = vec![0; arr.shape().len()];
            chk_shapes.insert(k.to_owned(), arr.chunk_shape(&chk_idx)?.to_array_shape());
            chk_grid_shapes.insert(k.to_owned(), arr.chunk_grid_shape().to_vec());
            arr_shapes.insert(k.to_owned(), arr.shape().to_vec());
        }
        let chunk_shape = resolve_vector(&coordinates, chk_shapes)?;
        let chunk_grid_shape = resolve_vector(&coordinates, chk_grid_shapes)?;
        let array_shape = resolve_vector(&coordinates, arr_shapes)?;

        // this runtime will handle the i/o. i/o tasks spawned in
        // that runtime will not share a thead pool with other
        // (probably compute heavy, blocking) tasks.
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

    /// this method will create interfaces (with pointers to the arrays)
    /// and return them. those can than be packaged with the (encoded)
    /// data so that those can be tracked together across async calls,
    /// and the array interfaces can then be used to decode the encoded
    /// data. this is all so that data can be read and decoded in different
    /// thread pools and in parallel.
    fn get_array_interfaces(
        &self,
        cols: Vec<String>,
        chk_idx: Vec<u64>,
    ) -> ZarrQueryResult<Vec<ArrayInterface<T>>> {
        let full_chunk_shape = self.get_chunk_shape(&chk_idx)?;
        cols.iter()
            .map(|col| {
                let arr = self
                    .arrays
                    .get(col)
                    .ok_or_else(|| {
                        ZarrQueryError::InvalidCompute(format!("Array '{col}' not found in store"))
                    })?
                    .clone();
                Ok(ArrayInterface::new(
                    col.to_string(),
                    arr,
                    self.coordinates.clone(),
                    full_chunk_shape.clone(),
                    chk_idx.clone(),
                ))
            })
            .collect()
    }

    /// spawn a read task per array interface on the I/O runtime, returning the
    /// receiver that yields each (read result, interface) pair as it completes.
    /// the reads run in parallel with each other and with the decoding the
    /// caller does while draining the receiver.
    fn spawn_reads(
        &mut self,
        interfaces: Vec<ArrayInterface<T>>,
        metrics: &ZarrMetrics,
    ) -> ZarrReceiver<T> {
        let (tx, rx) = tokio::sync::mpsc::channel(interfaces.len());
        for arr_interface in interfaces {
            let tx = tx.clone();
            let metrics = metrics.clone();
            let io_task = async move {
                let read_start = Instant::now();
                let b = arr_interface.read_bytes().await;
                metrics.add_io_time(read_start.elapsed());
                let _ = tx.send((b, arr_interface)).await;
            };
            self.join_set.spawn_on(io_task, self.io_runtime.handle());
        }
        rx
    }

    /// drain a receiver of read results, decoding each array on the compute side
    /// as it arrives and accumulating it into the chunk. decoding one array
    /// overlaps with the still-in-flight reads of the others.
    async fn drain_decode(
        mut rx: ZarrReceiver<T>,
        metrics: &ZarrMetrics,
        chk_data: &mut ZarrInMemoryChunk,
    ) -> ZarrQueryResult<()> {
        while let Some((bytes, arr_interface)) = rx.recv().await {
            let bytes = bytes?;
            let decode_start = Instant::now();
            let data = arr_interface.decode_data(bytes)?;
            metrics.add_decode_time(decode_start.elapsed());
            chk_data.add_data(arr_interface.name, data);
        }
        Ok(())
    }

    /// this is the main function that does the heavy lifting, getting
    /// the data from the zarr store and decoding it.
    async fn get_chunk(
        &mut self,
        cols: Vec<String>,
        chk_idx: Vec<u64>,
        use_cached_value: bool,
        next_chunk_idx: Option<Vec<u64>>,
        metrics: &ZarrMetrics,
    ) -> ZarrQueryResult<ZarrInMemoryChunk> {
        if cols.is_empty() {
            return Err(ZarrQueryError::InvalidColumnRequest(
                "No columns when polling zarr store for chunks".into(),
            ));
        }
        let mut chk_data = ZarrInMemoryChunk::new();

        // obtain the receiver for this chunk's reads: either the one prefetched
        // by the previous call (if it matches what we're asked for), or a fresh
        // batch of reads spawned now. spawning all the reads up front lets the
        // decoding below overlap with the still-in-flight reads.
        let cache_hit = use_cached_value && self.state.is_some();
        let rx = if cache_hit {
            let (rx, cached_idx, cached_cols) = self
                .state
                .take()
                .expect("cache hit implies state is present");
            if cached_idx != chk_idx {
                return Err(ZarrQueryError::InvalidCompute(
                    "Cached zarr chunk index doesn't match requested chunk index".into(),
                ));
            }
            if cached_cols != cols {
                return Err(ZarrQueryError::InvalidCompute(
                    "Cached zarr chunk columns don't match requested columns".into(),
                ));
            }
            rx
        } else {
            let interfaces = self.get_array_interfaces(cols.clone(), chk_idx.clone())?;
            self.spawn_reads(interfaces, metrics)
        };
        Self::drain_decode(rx, metrics, &mut chk_data).await?;

        // if the call was made with an index for the next chunk, we submit a job
        // to read that next chunk before returning, so that we can fetch the data
        // while other operations run between now and the next call.
        if let Some(next_chunk_idx) = next_chunk_idx {
            let interfaces = self.get_array_interfaces(cols.clone(), next_chunk_idx.clone())?;
            let rx = self.spawn_reads(interfaces, metrics);
            self.state = Some((rx, next_chunk_idx, cols));
        }

        Ok(chk_data)
    }
}

/// A stream of RecordBatches read from a Zarr store.
///
/// This struct is separate from `ZarrRecordBatchStream`, so that we
/// can avoid manually implementing [`Stream`]. Instead, we use the
/// `async-stream` crate to convert an async iterable into a stream.
struct ZarrRecordBatchStreamInner<T: AsyncReadableListableStorageTraits + ?Sized> {
    zarr_store: ZarrStore<T>,
    projected_schema_ref: SchemaRef,
    // when a filter is set it is bundled with the schema of the
    // projected columns that are *not* in the filter predicate (i.e.
    // the columns still to read once a chunk passes). either both
    // are present or neither is.
    filter: Option<(ZarrChunkFilter, SchemaRef)>,
    // Dynamic filters pushed down from the scan (e.g. the spatial join's
    // chunk-pruning filter). Drained into `pruning` on the first `next_chunk`.
    dynamic_filters: Vec<Arc<dyn PhysicalExpr>>,
    // Resolved chunk-pruning expression and the coordinate subset schema it reads.
    pruning: Option<ChunkPruning>,
    chunk_indices: VecDeque<Vec<u64>>,
    // the number of rows this partition may still produce, if a limit
    // is set. the last chunk can push the running total slightly over
    // the original limit.
    limit: Option<usize>,
    metrics: ZarrMetrics,
}

/// A resolved chunk-pruning expression and the coordinate subset
/// schema it reads.
struct ChunkPruning {
    expr: Arc<dyn PhysicalExpr>,
    schema: SchemaRef,
}

impl ChunkPruning {
    /// Selects the columns `expr` references from `projected_schema` (by name)
    /// into a subset schema, and reassigns the expression's column indices to it
    /// so it evaluates against a min/max batch built from that subset.
    fn try_new(expr: Arc<dyn PhysicalExpr>, projected_schema: &SchemaRef) -> ZarrQueryResult<Self> {
        let mut indices = collect_columns(&expr)
            .iter()
            .map(|c| projected_schema.index_of(c.name()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ZarrQueryError::External(Box::new(e)))?;
        indices.sort_unstable();
        indices.dedup();

        let schema = Arc::new(
            projected_schema
                .project(&indices)
                .map_err(|e| ZarrQueryError::External(Box::new(e)))?,
        );
        let expr = reassign_expr_columns(expr, &schema)
            .map_err(|e| ZarrQueryError::External(Box::new(e)))?;
        Ok(Self { expr, schema })
    }
}

/// Evaluates a chunk-pruning expression against a chunk's min/max record batch,
/// returning whether the chunk should be kept (a null verdict keeps it).
fn chunk_kept(expr: &Arc<dyn PhysicalExpr>, min_max: &RecordBatch) -> ZarrQueryResult<bool> {
    let value = expr
        .evaluate(min_max)
        .map_err(|e| ZarrQueryError::External(Box::new(e)))?;
    match value {
        ColumnarValue::Scalar(ScalarValue::Boolean(Some(keep))) => Ok(keep),
        ColumnarValue::Scalar(ScalarValue::Boolean(None)) => Ok(true),
        other => Err(ZarrQueryError::InvalidCompute(format!(
            "chunk-pruning expression must return a boolean scalar, got {other:?}"
        ))),
    }
}

/// collect the column (array) names from a schema's fields.
fn column_names(schema: &Schema) -> Vec<String> {
    schema
        .fields()
        .iter()
        .map(|f| f.name().to_owned())
        .collect()
}

impl<T: AsyncReadableListableStorageTraits + ?Sized + 'static> ZarrRecordBatchStreamInner<T> {
    /// Create a new ZarrRecordBatchStreamInner.
    ///
    /// This function is intentionally private, as all users should call
    /// [`ZarrRecordBatchStream::new`] instead.
    #[allow(clippy::too_many_arguments)]
    async fn new(
        store: Arc<T>,
        schema_ref: SchemaRef,
        prefix: Option<String>,
        projection: Option<Vec<usize>>,
        n_partitions: usize,
        partition: usize,
        dynamic_filters: Vec<Arc<dyn PhysicalExpr>>,
        metrics: ZarrMetrics,
        limit: Option<usize>,
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

        // this will extract column (i.e. array) names based on the
        // (possibly projected) schema.
        let cols: Vec<_> = projected_schema_ref
            .fields()
            .iter()
            .map(|f| f.name())
            .collect();

        // create the zarr store object that will have access to all
        // the data stored in it.
        let zarr_store =
            ZarrStore::new(&cols, &prefix, store.clone(), projected_schema_ref.clone()).await?;

        // this creates all the chunk indices we will be reading from.
        let chk_grid_shape = &zarr_store.chunk_grid_shape;
        // enumerate every chunk index over the N-D grid (last dimension varies
        // fastest, matching the row-major layout of the data).
        let mut chunk_indices: Vec<Vec<u64>> = chk_grid_shape
            .iter()
            .map(|&n| 0..n)
            .multi_cartesian_product()
            .collect();
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
            dynamic_filters,
            pruning: None,
            chunk_indices,
            limit,
            metrics,
        })
    }

    /// Resolves the pushed dynamic filters into a chunk-pruning expression, on
    /// the first `next_chunk` call. By this point the spatial join has finished
    /// its build side, so the shared dynamic filter holds the real
    /// `ProbePruningExpr`.
    fn resolve_pruning(&mut self) -> ZarrQueryResult<()> {
        if self.dynamic_filters.is_empty() {
            return Ok(());
        }

        let mut pruning_expr: Option<Arc<dyn PhysicalExpr>> = None;
        for filter in self.dynamic_filters.drain(..) {
            let Some(inner) = filter
                .snapshot()
                .map_err(|e| ZarrQueryError::External(Box::new(e)))?
            else {
                continue;
            };
            if inner.downcast_ref::<ProbePruningExpr>().is_some() {
                if pruning_expr.is_some() {
                    return Err(ZarrQueryError::InvalidCompute(
                        "more than one chunk-pruning expression pushed to the zarr scan".into(),
                    ));
                }
                pruning_expr = Some(inner);
            }
        }

        if let Some(expr) = pruning_expr {
            self.pruning = Some(ChunkPruning::try_new(expr, &self.projected_schema_ref)?);
        }

        Ok(())
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

        // stop early once this partition has produced its row limit (if any).
        if self.limit == Some(0) {
            return Ok(None);
        }

        // wall-clock timer for the whole batch-producing body.
        let total_start = Instant::now();

        // resolve the pushed dynamic filters into a pruning expression once.
        self.resolve_pruning()?;

        let mut chunk_index = self.pop_chunk_idx();
        let mut filter_zarr_chunk: Option<ZarrInMemoryChunk> = None;

        // walk chunks until one passes all present gates (pruning, then static
        // filter).
        if self.pruning.is_some() || self.filter.is_some() {
            let filter = self.filter.take();
            let pruning = self.pruning.take();
            while let Some(idx) = chunk_index.clone() {
                filter_zarr_chunk = None;

                if let Some(pruning) = &pruning {
                    let cols = column_names(&pruning.schema);
                    let prune_chunk = self
                        .zarr_store
                        .get_chunk(cols, idx.clone(), false, None, &self.metrics)
                        .await?;
                    let min_max = prune_chunk.to_min_max_record_batch(&pruning.schema)?;
                    if !chunk_kept(&pruning.expr, &min_max)? {
                        chunk_index = self.pop_chunk_idx();
                        continue;
                    }
                }

                if filter.is_some() {
                    // Extract the columns without holding a filter reference across
                    // the read (the predicate isn't Sync, holding across await is a
                    // problem), re-borrow it afterwards.
                    #[allow(clippy::unnecessary_unwrap)]
                    let cols = column_names(filter.as_ref().unwrap().0.schema_ref());
                    let zarr_chunk = self
                        .zarr_store
                        .get_chunk(cols, idx.clone(), false, None, &self.metrics)
                        .await?;
                    #[allow(clippy::unnecessary_unwrap)]
                    let (chunk_filter, _) = filter.as_ref().unwrap();
                    if !zarr_chunk.check_filter(chunk_filter)? {
                        chunk_index = self.pop_chunk_idx();
                        continue;
                    }
                    filter_zarr_chunk = Some(zarr_chunk);
                }

                break;
            }
            self.filter = filter;
            self.pruning = pruning;
        }

        if let Some(chunk_index) = chunk_index {
            // Only interleave the next chunk when there's no filter (static or
            // dynamic), since a filter may skip it.
            let interleave = self.filter.is_none() && self.pruning.is_none();
            let next_chnk_idx = if interleave {
                self.see_chunk_idx()
            } else {
                None
            };
            let cols = match &self.filter {
                Some((_, schema_without_filter_cols)) => column_names(schema_without_filter_cols),
                None => column_names(&self.projected_schema_ref),
            };

            let mut zarr_chunk = self
                .zarr_store
                .get_chunk(cols, chunk_index, interleave, next_chnk_idx, &self.metrics)
                .await?;
            if let Some(filter_zarr_chunk) = filter_zarr_chunk {
                zarr_chunk.combine(filter_zarr_chunk);
            }

            let record_batch = zarr_chunk.into_record_batch(&self.projected_schema_ref)?;
            self.metrics.inc_chunks_read();
            self.metrics.add_rows(record_batch.num_rows());
            self.metrics.add_total_time(total_start.elapsed());

            if let Some(remaining) = self.limit.as_mut() {
                *remaining = remaining.saturating_sub(record_batch.num_rows());
            }
            Ok(Some(record_batch))
        } else {
            self.metrics.add_total_time(total_start.elapsed());
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
        let chunk_idx = self.chunk_indices.pop_front();
        if chunk_idx.is_some() {
            self.metrics.inc_chunks_looked_at();
        }
        chunk_idx
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
        let schema_without_filter_cols = Arc::new(Schema::new(fields));

        // set the filter (bundled with that schema) on the inner stream.
        self.filter = Some((filter, schema_without_filter_cols));

        Ok(self)
    }
}

/// An async stream of record batches read from the Zarr store.
///
/// This implementation is modeled to be used with the DataFusion
/// [`RecordBatchStream`] trait.
///
/// [`RecordBatchStream`]: https://docs.rs/datafusion/latest/datafusion/execution/trait.RecordBatchStream.html
pub struct ZarrRecordBatchStream {
    stream: BoxStream<'static, Result<RecordBatch, ArrowError>>,
    schema: SchemaRef,
}

impl ZarrRecordBatchStream {
    /// Create a new ZarrRecordBatchStream.
    #[allow(clippy::too_many_arguments)]
    pub async fn try_new<T: AsyncReadableListableStorageTraits + ?Sized + 'static>(
        store: Arc<T>,
        schema_ref: SchemaRef,
        prefix: Option<String>,
        projection: Option<Vec<usize>>,
        n_partitions: usize,
        partition: usize,
        filter: Option<ZarrChunkFilter>,
        dynamic_filters: Vec<Arc<dyn PhysicalExpr>>,
        metrics: ZarrMetrics,
        limit: Option<usize>,
    ) -> ZarrQueryResult<Self> {
        let mut inner = ZarrRecordBatchStreamInner::new(
            store,
            schema_ref,
            prefix,
            projection,
            n_partitions,
            partition,
            dynamic_filters,
            metrics,
            limit,
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
    use arrow::compute::concat_batches;
    use datafusion::config::ConfigOptions;
    use datafusion::logical_expr::ScalarUDF;
    use datafusion::physical_expr::expressions::{lit, Column, DynamicFilterPhysicalExpr};
    use datafusion::physical_expr::ScalarFunctionExpr;
    use futures_util::TryStreamExt;

    use super::*;
    use crate::geospatial::indexed_build_side::test_helpers::make_indexed_build_side;
    use crate::geospatial::udfs::StPointUdf;
    use crate::test_utils::{
        extract_col, get_local_zarr_store, get_local_zarr_store_3d, get_local_zarr_store_4d,
        get_local_zarr_store_mix_dims, get_local_zarr_store_no_coords, validate_names_and_types,
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

        let stream = ZarrRecordBatchStream::try_new(
            store,
            schema,
            None,
            None,
            1,
            0,
            None,
            Vec::new(),
            ZarrMetrics::disconnected(),
            None,
        )
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
    async fn spatial_pruning_test() {
        let (wrapper, schema) =
            get_local_zarr_store(true, 0.0, "lat_lon_data_spatial_pruning").await;
        let store = wrapper.get_store();

        // Build side: two squares, each landing inside one corner chunk of the
        // 3x3 grid (lat 35-42, lon -120 to -113). Their combined bbox spans the
        // whole store, but only chunks (0,0) and (2,2) intersect a square.
        let squares = [
            "POLYGON((-119.5 35.5, -118.5 35.5, -118.5 36.5, -119.5 36.5, -119.5 35.5))",
            "POLYGON((-114.5 40.5, -113.5 40.5, -113.5 41.5, -114.5 41.5, -114.5 40.5))",
        ];
        let index = make_indexed_build_side(&squares, 1, 1, false, "geometry", "val");

        // Probe geometry: st_point(lon, lat) over the store's coordinate columns.
        let st_point = Arc::new(ScalarUDF::from(StPointUdf::default()));
        let probe_expr: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            "st_point",
            st_point,
            vec![
                Arc::new(Column::new("lon", 2)),
                Arc::new(Column::new("lat", 1)),
            ],
            Arc::new(Field::new("geometry", DataType::Binary, true)),
            Arc::new(ConfigOptions::default()),
        ));

        // Wrap the pruning expression in a dynamic filter, as the spatial join does.
        let pruning = Arc::new(ProbePruningExpr::new(probe_expr, index)) as Arc<dyn PhysicalExpr>;
        let dynamic_filter = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![
                Arc::new(Column::new("lon", 2)),
                Arc::new(Column::new("lat", 1)),
            ],
            lit(true),
        ));
        dynamic_filter.update(pruning).unwrap();

        let stream = ZarrRecordBatchStream::try_new(
            store,
            schema,
            None,
            None,
            1,
            0,
            None,
            vec![dynamic_filter as Arc<dyn PhysicalExpr>],
            ZarrMetrics::disconnected(),
            None,
        )
        .await
        .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        assert_eq!(records.len(), 2);
        validate_primitive_column::<Float64Type, f64>(
            "data",
            &records[0],
            &[0.0, 1.0, 2.0, 8.0, 9.0, 10.0, 16.0, 17.0, 18.0],
        );
        validate_primitive_column::<Float64Type, f64>(
            "data",
            &records[1],
            &[54.0, 55.0, 62.0, 63.0],
        );
    }

    #[tokio::test]
    async fn read_data_no_coords_test() {
        let (wrapper, schema) = get_local_zarr_store_no_coords(0.0, "data_no_coords").await;
        let store = wrapper.get_store();

        let stream = ZarrRecordBatchStream::try_new(
            store,
            schema.clone(),
            None,
            None,
            1,
            0,
            None,
            Vec::new(),
            ZarrMetrics::disconnected(),
            None,
        )
        .await
        .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        let target_types = HashMap::from([
            ("data_1".to_string(), DataType::Float64),
            ("data_2".to_string(), DataType::Float64),
        ]);
        validate_names_and_types(&target_types, &records[0]);
        assert_eq!(records.len(), 3);

        let batch = concat_batches(&schema, &records).unwrap();

        validate_primitive_column::<Float64Type, f64>(
            "data_1",
            &batch,
            &[0., 1., 2., 3., 4., 5., 6., 7.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "data_2",
            &batch,
            &[100., 101., 102., 103., 104., 105., 106., 107.],
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

        let stream = ZarrRecordBatchStream::try_new(
            store,
            schema,
            None,
            None,
            1,
            0,
            filter,
            Vec::new(),
            ZarrMetrics::disconnected(),
            None,
        )
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

    // the store is an 8x8 grid split into 3x3 chunks, so a 3x3 = 9 chunk
    // grid enumerated row-major with per-chunk row counts
    // [9, 9, 6, 9, 9, 6, 6, 6, 4]. the chunk that reaches the limit is
    // emitted whole and the total can overshoot.
    #[tokio::test]
    async fn limit_test() {
        let (wrapper, schema) = get_local_zarr_store(true, 0.0, "lat_lon_data_with_limit").await;
        let store = wrapper.get_store();

        async fn count_rows<T: AsyncReadableListableStorageTraits + ?Sized + 'static>(
            store: Arc<T>,
            schema: SchemaRef,
            n_partitions: usize,
            partition: usize,
            limit: usize,
        ) -> usize {
            let inner = ZarrRecordBatchStreamInner::new(
                store,
                schema,
                None,
                None,
                n_partitions,
                partition,
                Vec::new(),
                ZarrMetrics::disconnected(),
                Some(limit),
            )
            .await
            .unwrap();

            let records: Vec<_> = inner.into_stream().try_collect().await.unwrap();
            records.iter().map(|b| b.num_rows()).sum()
        }

        // single partition, limit 8: the first chunk (9 rows) already reaches
        // the limit, so we stop after it -> 9 rows.
        assert_eq!(count_rows(store.clone(), schema.clone(), 1, 0, 8).await, 9);

        // single partition, limit 10: the first chunk (9) isn't enough, the
        // second chunk (9) crosses the limit -> 18 rows.
        assert_eq!(
            count_rows(store.clone(), schema.clone(), 1, 0, 10).await,
            18
        );

        // two partitions, limit 8 each (applied independently per partition):
        // partition 0 reads chunk [9] and stops -> 9. partition 1's chunks are
        // the edge chunks [6, 6, 6, 4], so [6] isn't enough and [6] crosses the
        // limit -> 12. total = 21.
        let p0 = count_rows(store.clone(), schema.clone(), 2, 0, 8).await;
        let p1 = count_rows(store.clone(), schema.clone(), 2, 1, 8).await;
        assert_eq!(p0 + p1, 21);
    }

    #[tokio::test]
    async fn dimension_tests() {
        // this store will have 2d lat coordinates and 1d lon coordinates.
        // that shoudl effecitvely given the same as 1d and 1d.
        let (wrapper, schema) = get_local_zarr_store_mix_dims(0.0, "lat_lon_mixed_dims_data").await;
        let store = wrapper.get_store();

        let stream = ZarrRecordBatchStream::try_new(
            store,
            schema,
            None,
            None,
            1,
            0,
            None,
            Vec::new(),
            ZarrMetrics::disconnected(),
            None,
        )
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
    async fn read_3d_data_test() {
        // 6 x 5 x 4 data with dims (lat, lon, height), 2 x 2 x 2 chunks, so a
        // 3 x 3 x 2 = 18 chunk grid. lat/lon are 1D coordinates broadcast up to
        // the full 3D chunk, height is the innermost dim.
        let (wrapper, schema) = get_local_zarr_store_3d(0.0, "lat_lon_height_data").await;
        let store = wrapper.get_store();

        let stream = ZarrRecordBatchStream::try_new(
            store,
            schema,
            None,
            None,
            1,
            0,
            None,
            Vec::new(),
            ZarrMetrics::disconnected(),
            None,
        )
        .await
        .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        let target_types = HashMap::from([
            ("lat".to_string(), DataType::Float64),
            ("lon".to_string(), DataType::Float64),
            ("height".to_string(), DataType::Float64),
            ("data".to_string(), DataType::Float64),
        ]);
        validate_names_and_types(&target_types, &records[0]);
        assert_eq!(records.len(), 18);

        // the first chunk, full 2 x 2 x 2.
        validate_primitive_column::<Float64Type, f64>(
            "lat",
            &records[0],
            &[35., 35., 35., 35., 36., 36., 36., 36.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            &records[0],
            &[-120., -120., -119., -119., -120., -120., -119., -119.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "height",
            &records[0],
            &[100., 200., 100., 200., 100., 200., 100., 200.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "data",
            &records[0],
            &[0., 1., 4., 5., 20., 21., 24., 25.],
        );

        // the 5th chunk [0, 2, 0] is the first edge chunk: lon has 5 values
        // with a chunk size of 2, so its last chunk is a partial one, only
        // spanning lon index 4.
        validate_primitive_column::<Float64Type, f64>("lat", &records[4], &[35., 35., 36., 36.]);
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            &records[4],
            &[-116., -116., -116., -116.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "height",
            &records[4],
            &[100., 200., 100., 200.],
        );
        validate_primitive_column::<Float64Type, f64>("data", &records[4], &[16., 17., 36., 37.]);
    }

    #[tokio::test]
    async fn read_4d_data_test() {
        // 6 x 5 x 4 x 3 data with dims (lat, lon, height, time), 2 x 2 x 2 x 2
        // chunks, so a 3 x 3 x 2 x 2 = 36 chunk grid. lat/lon/height/time are 1D
        // coordinates broadcast up to the full 4D chunk, time is a datetime64[s]
        // column decoded as an arrow Timestamp(Second, None).
        let (wrapper, schema) = get_local_zarr_store_4d(0.0, "lat_lon_height_time_data").await;
        let store = wrapper.get_store();

        let stream = ZarrRecordBatchStream::try_new(
            store,
            schema,
            None,
            None,
            1,
            0,
            None,
            Vec::new(),
            ZarrMetrics::disconnected(),
            None,
        )
        .await
        .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        let target_types = HashMap::from([
            ("lat".to_string(), DataType::Float64),
            ("lon".to_string(), DataType::Float64),
            ("height".to_string(), DataType::Float64),
            (
                "time".to_string(),
                DataType::Timestamp(TimeUnit::Second, None),
            ),
            ("data".to_string(), DataType::Float64),
        ]);
        validate_names_and_types(&target_types, &records[0]);
        assert_eq!(records.len(), 36);

        // the first chunk, full 2 x 2 x 2 x 2, spanning lat {0,1}, lon {0,1},
        // height {0,1}, time {0,1}. same broadcasting logic as the 3D case, just
        // with an extra (innermost) dimension changing the repeat/tile counts.
        validate_primitive_column::<Float64Type, f64>(
            "lat",
            &records[0],
            &[
                35., 35., 35., 35., 35., 35., 35., 35., 36., 36., 36., 36., 36., 36., 36., 36.,
            ],
        );
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            &records[0],
            &[
                -120., -120., -120., -120., -119., -119., -119., -119., -120., -120., -120., -120.,
                -119., -119., -119., -119.,
            ],
        );
        validate_primitive_column::<Float64Type, f64>(
            "height",
            &records[0],
            &[
                100., 100., 200., 200., 100., 100., 200., 200., 100., 100., 200., 200., 100., 100.,
                200., 200.,
            ],
        );
        validate_primitive_column::<TimestampSecondType, i64>(
            "time",
            &records[0],
            &[
                1_700_000_000,
                1_700_000_001,
                1_700_000_000,
                1_700_000_001,
                1_700_000_000,
                1_700_000_001,
                1_700_000_000,
                1_700_000_001,
                1_700_000_000,
                1_700_000_001,
                1_700_000_000,
                1_700_000_001,
                1_700_000_000,
                1_700_000_001,
                1_700_000_000,
                1_700_000_001,
            ],
        );
        validate_primitive_column::<Float64Type, f64>(
            "data",
            &records[0],
            &[
                0., 1., 3., 4., 12., 13., 15., 16., 60., 61., 63., 64., 72., 73., 75., 76.,
            ],
        );

        // the 9th chunk [0, 2, 0, 0] is an edge chunk along the 2nd dimension:
        // lon has 5 values with a chunk size of 2, so its last chunk is partial,
        // only spanning lon index 4. the full chunk shape is 2 x 1 x 2 x 2.
        validate_primitive_column::<Float64Type, f64>(
            "lat",
            &records[8],
            &[35., 35., 35., 35., 36., 36., 36., 36.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            &records[8],
            &[-116., -116., -116., -116., -116., -116., -116., -116.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "height",
            &records[8],
            &[100., 100., 200., 200., 100., 100., 200., 200.],
        );
        validate_primitive_column::<TimestampSecondType, i64>(
            "time",
            &records[8],
            &[
                1_700_000_000,
                1_700_000_001,
                1_700_000_000,
                1_700_000_001,
                1_700_000_000,
                1_700_000_001,
                1_700_000_000,
                1_700_000_001,
            ],
        );
        validate_primitive_column::<Float64Type, f64>(
            "data",
            &records[8],
            &[48., 49., 51., 52., 108., 109., 111., 112.],
        );

        // the last chunk [2, 2, 1, 1] is partial along two dimensions at once:
        // lon (size 5) and time (size 3) are both non-multiples of the chunk
        // size 2, so each collapses to width 1. lat and height stay full width.
        // the full chunk shape is 2 x 1 x 2 x 1.
        validate_primitive_column::<Float64Type, f64>("lat", &records[35], &[39., 39., 40., 40.]);
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            &records[35],
            &[-116., -116., -116., -116.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "height",
            &records[35],
            &[300., 400., 300., 400.],
        );
        validate_primitive_column::<TimestampSecondType, i64>(
            "time",
            &records[35],
            &[1_700_000_002, 1_700_000_002, 1_700_000_002, 1_700_000_002],
        );
        validate_primitive_column::<Float64Type, f64>(
            "data",
            &records[35],
            &[296., 299., 356., 359.],
        );
    }

    #[tokio::test]
    async fn read_missing_chunks_test() {
        let fillvalue = 1234.0;
        let (wrapper, schema) = get_local_zarr_store(false, fillvalue, "lat_lon_empty_data").await;
        let store = wrapper.get_store();

        let stream = ZarrRecordBatchStream::try_new(
            store,
            schema,
            None,
            None,
            1,
            0,
            None,
            Vec::new(),
            ZarrMetrics::disconnected(),
            None,
        )
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

        let stream = ZarrRecordBatchStream::try_new(
            store.clone(),
            schema.clone(),
            None,
            None,
            2,
            0,
            None,
            Vec::new(),
            ZarrMetrics::disconnected(),
            None,
        )
        .await
        .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();
        validate_names_and_types(&target_types, &records[0]);
        assert_eq!(records.len(), 5);

        let stream = ZarrRecordBatchStream::try_new(
            store,
            schema,
            None,
            None,
            2,
            1,
            None,
            Vec::new(),
            ZarrMetrics::disconnected(),
            None,
        )
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
        let stream = ZarrRecordBatchStream::try_new(
            store.clone(),
            schema.clone(),
            None,
            None,
            20,
            0,
            None,
            Vec::new(),
            ZarrMetrics::disconnected(),
            None,
        )
        .await
        .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();
        assert_eq!(records.len(), 1);

        let stream = ZarrRecordBatchStream::try_new(
            store.clone(),
            schema.clone(),
            None,
            None,
            20,
            8,
            None,
            Vec::new(),
            ZarrMetrics::disconnected(),
            None,
        )
        .await
        .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();
        assert_eq!(records.len(), 1);

        let stream = ZarrRecordBatchStream::try_new(
            store.clone(),
            schema.clone(),
            None,
            None,
            20,
            10,
            None,
            Vec::new(),
            ZarrMetrics::disconnected(),
            None,
        )
        .await
        .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();
        assert_eq!(records.len(), 0);

        let stream = ZarrRecordBatchStream::try_new(
            store,
            schema,
            None,
            None,
            20,
            19,
            None,
            Vec::new(),
            ZarrMetrics::disconnected(),
            None,
        )
        .await
        .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();
        assert_eq!(records.len(), 0);
    }

    #[tokio::test]
    async fn metrics_test() {
        use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;

        let (wrapper, schema) = get_local_zarr_store(true, 0.0, "lat_lon_data_with_metrics").await;
        let store = wrapper.get_store();

        let filter = Some(
            ZarrChunkFilter::new(
                vec![Box::new(DummyPredicate {})],
                Arc::new(schema.project(&[1, 2]).unwrap()),
            )
            .unwrap(),
        );

        // the metrics set is created here so we can read the registered
        // metrics back out after draining the stream.
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = ZarrMetrics::new(&metrics_set, 0);

        let stream = ZarrRecordBatchStream::try_new(
            store,
            schema,
            None,
            None,
            1,
            0,
            filter,
            Vec::new(),
            metrics,
            None,
        )
        .await
        .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();
        assert_eq!(records.len(), 4);

        let snapshot = metrics_set.clone_inner();

        for name in [
            "io_time",
            "decode_time",
            "total_time",
            "chunks_looked_at",
            "chunks_read",
            "rows_produced",
        ] {
            assert!(
                snapshot.sum_by_name(name).is_some(),
                "metric {name} missing from the metrics set"
            );
        }

        // we can't assert anything meaningful on the timings, but the counts are
        // deterministic: all 9 chunks are examined, 4 pass the filter, and those
        // 4 chunks (two of which are 3x2 edge chunks on the right of the grid)
        // hold 30 rows total.
        let value = |name: &str| snapshot.sum_by_name(name).unwrap().as_usize();
        assert_eq!(value("chunks_looked_at"), 9);
        assert_eq!(value("chunks_read"), 4);
        assert_eq!(value("rows_produced"), 30);
    }
}
