use super::{filter::ZarrChunkFilter, io_runtime::IoRuntime};
use crate::errors::zarr_errors::{ZarrQueryError, ZarrQueryResult};
use arrow::array::*;
use arrow::datatypes::*;
use arrow::record_batch::RecordBatch;
use arrow_schema::ArrowError;
use arrow_schema::{DataType, Field, Fields, Schema};
use bytes::Bytes;
use futures::stream::Stream;
use futures::{ready, FutureExt};
use futures_util::future::BoxFuture;
use itertools::iproduct;
use std::borrow::Cow;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::task::JoinSet;
use zarrs::array::codec::{ArrayToBytesCodecTraits, CodecOptions};
use zarrs::array::{
    Array, ArrayBytes, ArrayMetadata, ArrayMetadataV3, ArraySize, DataType as zDataType,
    ElementOwned,
};
use zarrs::array_subset::ArraySubset;
use zarrs_metadata::v3::array::data_type::DataTypeMetadataV3;
use zarrs_storage::{AsyncReadableListableStorageTraits, StorePrefix};

//********************************************
// various utils to handle metadata from the zarr array, data types,
// schemas, etc...
//********************************************

// extract the chunk size from the metadata.
fn extract_chunk_size(meta: &ArrayMetadataV3) -> ZarrQueryResult<Vec<u64>> {
    let chunks = meta
        .chunk_grid
        .configuration()
        .ok_or(ZarrQueryError::InvalidMetadata(
            "Could not find chunk grid configuration".into(),
        ))?
        .get("chunk_shape")
        .ok_or(ZarrQueryError::InvalidMetadata(
            "Could not find chunk_shape in configuration".into(),
        ))?
        .as_array()
        .ok_or(ZarrQueryError::InvalidMetadata(
            "Could not convert chunk shape to array".into(),
        ))?
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();

    Ok(chunks)
}

// extract the coordinate names from the array. it is possible to
// have an array with no coordinates.
fn get_coord_names<T: ?Sized>(arr: &Array<T>) -> ZarrQueryResult<Option<Vec<String>>> {
    if let Some(coords) = arr.dimension_names() {
        let coords: Vec<_> = coords
            .iter()
            .map(|d| d.as_str())
            .collect::<Option<Vec<_>>>()
            .ok_or(ZarrQueryError::InvalidMetadata(
                "Coodrinates without a name are not supported".into(),
            ))?
            .iter()
            .map(|s| s.to_string())
            .collect();

        return Ok(Some(coords));
    }

    Ok(None)
}

// extract the column (or array) names from the prefixes in the
// zarr store. a parent prefix can be provided, for example in the case
// where groups are used in the zarr store.
fn extract_columns(prefix: &str, keys: Vec<StorePrefix>) -> ZarrQueryResult<Vec<String>> {
    let cols: Vec<_> = keys
        .iter()
        .map(|k| {
            k.as_str()
                .replace(prefix, "")
                .split("/")
                .next()
                .map(|s| s.to_string())
        })
        .collect::<Option<Vec<_>>>()
        .ok_or(ZarrQueryError::InvalidMetadata(
            "Could not extract column name from store key".into(),
        ))?;

    Ok(cols.into_iter().collect())
}

// extract the metadata from array. only V3 metadata is supported.
fn extract_meta_from_array<T: ?Sized>(array: &Array<T>) -> ZarrQueryResult<&ArrayMetadataV3> {
    let meta = match array.metadata() {
        ArrayMetadata::V3(meta) => Ok(meta),
        _ => Err(ZarrQueryError::InvalidMetadata(
            "Only v3 metadata is supported".into(),
        )),
    }?;

    Ok(meta)
}

// convert the data type from the zarrs metadata to an arrow type.
fn get_schema_type(value: &DataTypeMetadataV3) -> ZarrQueryResult<DataType> {
    match value {
        DataTypeMetadataV3::Bool => Ok(DataType::Boolean),
        DataTypeMetadataV3::UInt8 => Ok(DataType::UInt8),
        DataTypeMetadataV3::UInt16 => Ok(DataType::UInt16),
        DataTypeMetadataV3::UInt32 => Ok(DataType::UInt32),
        DataTypeMetadataV3::UInt64 => Ok(DataType::UInt64),
        DataTypeMetadataV3::Int8 => Ok(DataType::Int8),
        DataTypeMetadataV3::Int16 => Ok(DataType::Int16),
        DataTypeMetadataV3::Int32 => Ok(DataType::Int32),
        DataTypeMetadataV3::Int64 => Ok(DataType::Int64),
        DataTypeMetadataV3::Float32 => Ok(DataType::Float32),
        DataTypeMetadataV3::Float64 => Ok(DataType::Float64),
        DataTypeMetadataV3::String => Ok(DataType::Utf8),
        _ => Err(ZarrQueryError::InvalidType(format!(
            "Unsupported type {value} from zarr metadata"
        ))),
    }
}

// produce an arrow schema given column names and arrays.
// the schema will be ordered following the names in the input
// vector of column names.
fn create_schema<T: ?Sized>(
    cols: Vec<String>,
    arrays: &HashMap<String, Arc<Array<T>>>,
) -> ZarrQueryResult<Schema> {
    let fields: Fields = cols
        .iter()
        .map(|c| arrays.get(c))
        .collect::<Option<Vec<_>>>()
        .ok_or(ZarrQueryError::InvalidMetadata(
            "Array missing from array map".into(),
        ))?
        .iter()
        .map(|a| extract_meta_from_array(a))
        .collect::<ZarrQueryResult<Vec<_>>>()?
        .iter()
        .map(|m| get_schema_type(&m.data_type))
        .collect::<ZarrQueryResult<Vec<_>>>()?
        .into_iter()
        .zip(cols)
        .map(|(d, c)| Arc::new(Field::new(c, d, false)))
        .collect();

    Ok(Schema::new(fields))
}

// this function handles having multiple values for a given vector,
// one per array, including some arrays that might be lower
// dimension coordinates.
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
                if final_vec[pos as usize] != vec[0] {
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
            final_vec[*p as usize] = vecs.get(k).ok_or(ZarrQueryError::InvalidMetadata(
                "Array is missing from array map".into(),
            ))?[0];
        }
        Ok(final_vec)
    }
}

//********************************************
// a struct to handle coordinate variables, and "broadcasting" them
// when reading multidimensional data.
//********************************************
#[derive(Debug)]
struct ZarrCoordinates {
    // the position of each coordinate in the overall chunk shape.
    // the coordinates are arrays that contain data that characterises
    // a dimension, such as time, or a longitude or latitude.
    coord_positions: HashMap<String, u64>,
}

impl ZarrCoordinates {
    fn new<T: ?Sized>(arrays: &HashMap<String, Array<T>>) -> ZarrQueryResult<Self> {
        let mut coord_positions: HashMap<String, u64> = HashMap::new();

        // first pass, determine what the coordinates are and what the
        // chunk dimensionaliy is.

        // this is a list of what columns are coordinates.
        let mut coords: Vec<String> = Vec::new();

        // this is an ordered list of the coordinates as they
        // show up in the array metadata.
        let mut chunk_coords: Option<Vec<String>> = None;

        for (k, arr) in arrays.iter() {
            let arr_coords = get_coord_names(arr)?;
            if let Some(arr_coords) = arr_coords {
                if arr_coords.contains(k) {
                    if arr_coords.len() != 1 {
                        return Err(ZarrQueryError::InvalidMetadata("Invalid coordinate".into()));
                    }
                    coords.push(k.to_string());
                } else {
                    // I don't like clippy's warning that I should collapse this.
                    #[allow(clippy::collapsible_else_if)]
                    if let Some(chk_coords) = &chunk_coords {
                        if chk_coords != &arr_coords {
                            return Err(ZarrQueryError::InvalidMetadata(
                                "Mismatch between variables' coordinates".into(),
                            ));
                        }
                    } else {
                        chunk_coords = Some(arr_coords);
                    }
                }
            }
        }

        // second pass, find the position of each coordinate in the chunk's
        // dimensionality.
        let chunk_coords = if let Some(chk_coords) = chunk_coords {
            chk_coords
        // the else branch here would happen if all the arrays are coordinates.
        } else {
            let mut coords_copy = coords.clone();
            coords_copy.sort();
            coords_copy
        };

        if coords.len() != chunk_coords.len() {
            return Err(ZarrQueryError::InvalidMetadata(
                "Mismatch with the number of coordinates".into(),
            ));
        }
        for d in coords {
            let pos = chunk_coords.iter().position(|s| &d == s).ok_or(
                ZarrQueryError::InvalidMetadata(
                    "Dimension is missing from variable's dimension list".into(),
                ),
            )?;
            coord_positions.insert(d.to_string(), pos as u64);
        }

        Ok(Self { coord_positions })
    }

    // checks if a column name corresponds to a coordinate.
    fn is_coordinate(&self, col: &str) -> bool {
        self.coord_positions.contains_key(col)
    }

    // returns the position of a coordinate within the chunk
    // dimensionality if the column is a coordinate, if not
    // returns None.
    fn get_coord_position(&self, col: &str) -> Option<u64> {
        self.coord_positions.get(col).cloned()
    }

    // return the vector element that corresponds to a coordinate's
    // position within the dimensionality (if the variable is a coordinate).
    fn reduce_if_coord(&self, vec: Vec<u64>, col: &str) -> Vec<u64> {
        if let Some(pos) = self.coord_positions.get(col) {
            return vec![vec[*pos as usize]];
        }

        vec
    }

    // broadacast a 1D array to a nD array if the variable is a coordinate.
    // note that we return a 1D vector, but this is just because we map all
    // the chunk to columnar data, so a m x n array gets mapped to a 1D
    // vector of length m x n.
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

//********************************************
// An interface to a zarr array that can be used to retrieve
// data and then decode it.
//********************************************

// the chunk index corresponds to the chunk that is being read, the
// coords to the coordinates for the full chunk (which can be made up
// of one or more arrays) and the full chunk shape is relevant when
// the chunk has some coordinate arrays, which are 1 dimensional, while
// the non coordinate arrays can be multi dimensional. the full chunk
// size is used to broadcast the coordinates to the full size.
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

// in most cases, we will read encoded bytes and decode them after,
// but in the case of a missing chunk the result of the read operation
// will be done.vin a few cases though we will read pre-decoded bytes,
// hence why we have this enum.
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

    // read the bytes from the chunk the interface was built for.
    async fn read_bytes(&self) -> ZarrQueryResult<BytesFromArray> {
        let chunk_grid = self.arr.chunk_grid_shape().ok_or_else(|| {
            ZarrQueryError::InvalidMetadata("Array is missing its chunk grid shape".into())
        })?;

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

    // decode the chunk that was read previously read from this interface.
    // the reason the 2 functionalities are separated is that we want to
    // interleave the async part (reading data) with the compute part
    // (decoding the data, creating the record batch) so that we can make
    // progress on the latter while the former is running.
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
            ($array_t: ty, $prim_type: ty) => {
                let arr_ref: $array_t = self
                    .coords
                    .broadcast_if_coord(
                        &self.name,
                        <$prim_type>::from_array_bytes(t, decoded_bytes)?,
                        &self.full_chunk_shape,
                    )?
                    .into();
                return Ok(Arc::new(arr_ref) as ArrayRef);
            };
        }

        match t {
            zDataType::Bool => {
                return_array_ref!(BooleanArray, bool);
            }
            zDataType::UInt8 => {
                return_array_ref!(PrimitiveArray<UInt8Type>, u8);
            }
            zDataType::UInt16 => {
                return_array_ref!(PrimitiveArray<UInt16Type>, u16);
            }
            zDataType::UInt32 => {
                return_array_ref!(PrimitiveArray<UInt32Type>, u32);
            }
            zDataType::UInt64 => {
                return_array_ref!(PrimitiveArray<UInt64Type>, u64);
            }
            zDataType::Int8 => {
                return_array_ref!(PrimitiveArray<Int8Type>, i8);
            }
            zDataType::Int16 => {
                return_array_ref!(PrimitiveArray<Int16Type>, i16);
            }
            zDataType::Int32 => {
                return_array_ref!(PrimitiveArray<Int32Type>, i32);
            }
            zDataType::Int64 => {
                return_array_ref!(PrimitiveArray<Int64Type>, i64);
            }
            zDataType::Float32 => {
                return_array_ref!(PrimitiveArray<Float32Type>, f32);
            }
            zDataType::Float64 => {
                return_array_ref!(PrimitiveArray<Float64Type>, f64);
            }
            zDataType::String => {
                return_array_ref!(StringArray, String);
            }
            _ => Err(ZarrQueryError::InvalidType(format!(
                "Unsupported type {t} from zarr metadata"
            ))),
        }
    }
}

//********************************************
// A structure to accumulate zarr array data until we can output
// the whole chunk as a record batch.
//********************************************
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

    // the columns in the record batch will be ordered following
    // the field names in the schema.
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

//********************************************
// A wrapper for a map of arrays, which will handle interleaving
// reading and decoding data from zarr storage.
//********************************************
struct ZarrStore<T: AsyncReadableListableStorageTraits + ?Sized> {
    arrays: HashMap<String, Arc<Array<T>>>,
    coordinates: Arc<ZarrCoordinates>,
    chunk_shape: Vec<u64>,
    chunk_grid_shape: Vec<u64>,
    array_shape: Vec<u64>,
    io_runtime: IoRuntime,
}

impl<T: AsyncReadableListableStorageTraits + ?Sized + 'static> ZarrStore<T> {
    fn new(arrays: HashMap<String, Array<T>>) -> ZarrQueryResult<Self> {
        let coordinates = ZarrCoordinates::new(&arrays)?;

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
            chk_grid_shapes.insert(
                k.to_owned(),
                arr.chunk_grid_shape()
                    .ok_or(ZarrQueryError::InvalidMetadata(
                        "Array is missing its chunk grid shape".into(),
                    ))?,
            );
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
        })
    }

    // return the chunk shape for a given index, taking into account
    // the array edges where the "real" chunk is smaller than the
    // chunk size in the metadata.
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

    // this is the main function that does the heavy lifting, getting
    // the data from the zarr store and decoding it.
    async fn get_chunk(
        self,
        cols: Vec<String>,
        chk_idx: Vec<u64>,
    ) -> ZarrQueryResult<(Self, ZarrInMemoryChunk, Vec<u64>)> {
        if cols.is_empty() {
            return Err(ZarrQueryError::InvalidProjection(
                "No columns when polling zarr store for chunks".into(),
            ));
        }
        let mut chk_data = ZarrInMemoryChunk::new();

        // this sets up the interfaces to each array chunk,
        // one for each column we are getting data for.
        let full_chunk_shape = self.get_chunk_shape(&chk_idx)?;
        let arr_interfaces: Vec<_> = cols
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

        // channel for the tasks on the io runtime to send results
        // back to this runtime.
        let (tx, mut rx) = tokio::sync::mpsc::channel(2);
        let mut join_set = JoinSet::new();

        // first column, we grab the data without decoding it, and we
        // clone the first array interface to use it later to decode
        // that first chunk of data.
        let tx_copy = tx.clone();
        let arr_for_future = arr_interfaces[0].clone();
        join_set.spawn_on(
            async move {
                let data = arr_for_future.read_bytes().await;
                let _ = tx_copy.send(data).await;
            },
            self.io_runtime.handle(),
        );
        let mut data;
        if let Some(Ok(d)) = rx.recv().await {
            data = d;
        } else {
            return Err(ZarrQueryError::InvalidCompute(
                "Unable to retrieve first data chunk".into(),
            ));
        }
        let mut last_arr = arr_interfaces[0].clone();

        // loop over all other column, intervealing decoding the
        // previously fetched data with reading from the next array.
        for arr in arr_interfaces.iter().skip(1) {
            // this is the name of a column we have already read
            // the data for.
            let last_col = last_arr.name.to_string();

            // this task, spawned on the io runtime, reads the next
            // data chunk.
            let tx_copy = tx.clone();
            let arr_for_future = arr.clone();
            let io_task = async move {
                let b = arr_for_future.read_bytes().await;
                let _ = tx_copy.send(b).await;
            };
            join_set.spawn_on(io_task, self.io_runtime.handle());

            // this decodes the last data chunk we read.
            let array_ref = last_arr.decode_data(data);

            // if everything went as expected, we have a decoded
            // chunk and a newly read (still encoded) chunk.
            if let (Some(Ok(d)), Ok(array_ref)) = (rx.recv().await, array_ref) {
                data = d;
                chk_data.add_data(last_col, array_ref);
            } else {
                return Err(ZarrQueryError::InvalidCompute(
                    "Unable to retrieve decoded chunk".into(),
                ));
            }

            // keep a copy of the array interface we just read
            // from, to use it to decode the data for the next
            // iteration (or the final task after the loop).
            last_arr = arr.clone();
        }

        // decode the last array chunk.
        let array_ref = last_arr.decode_data(data)?;
        chk_data.add_data(last_arr.name, array_ref);

        Ok((self, chk_data, chk_idx))
    }
}

//********************************************
// the stucture that will be used to stream data from the zarr
// store as it gets read.
//********************************************
type ChunkFuture<T> =
    BoxFuture<'static, ZarrQueryResult<(ZarrStore<T>, ZarrInMemoryChunk, Vec<u64>)>>;
enum ZarrStreamState<T: AsyncReadableListableStorageTraits + ?Sized> {
    Init,
    Reading(ChunkFuture<T>),
    ReadingForFilter(ChunkFuture<T>),
    Error,
    Done,
}

pub struct ZarrRecordBatchStream<T: AsyncReadableListableStorageTraits + ?Sized> {
    zarr_store: Option<ZarrStore<T>>,
    schema_ref: SchemaRef,
    projected_schema_ref: SchemaRef,
    filter: Option<ZarrChunkFilter>,
    state: ZarrStreamState<T>,
    chunk_indices: VecDeque<Vec<u64>>,
}

impl<T: AsyncReadableListableStorageTraits + ?Sized + 'static> ZarrRecordBatchStream<T> {
    pub(crate) async fn new(
        store: Arc<T>,
        schema_ref: SchemaRef,
        group: Option<String>,
        projection: Option<Vec<usize>>,
    ) -> ZarrQueryResult<Self> {
        // if there is a projection provided, modify the schema.
        let projected_schema_ref = match projection {
            Some(proj) => Arc::new(schema_ref.project(&proj)?),
            None => schema_ref.clone(),
        };

        // any groups the actual arrays fall under (e.g. /some_group/array1,
        // /some_group/array2, etc...)
        let grp = if let Some(group) = group {
            [group, "/".into()].join("")
        } else {
            "".to_string()
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
            let path = PathBuf::from(&grp)
                .join(["/", col].join(""))
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
        let zarr_store = ZarrStore::new(arrays)?;

        // this creates all the chunk indices we will be reading from.
        let chk_grid_shape = &zarr_store.chunk_grid_shape;
        let chunk_indices: Vec<_> = match chk_grid_shape.len() {
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
        let chunk_indices = VecDeque::from(chunk_indices);

        Ok(Self {
            zarr_store: Some(zarr_store),
            schema_ref,
            projected_schema_ref,
            filter: None,
            state: ZarrStreamState::Init,
            chunk_indices,
        })
    }

    pub(crate) fn get_projected_schema_ref(&self) -> Arc<Schema> {
        self.projected_schema_ref.clone()
    }

    fn pop_chunk_idx(&mut self) -> Option<Vec<u64>> {
        self.chunk_indices.pop_front()
    }

    // adds a filter to avoid reading whole chunks if no values
    // in the corresponding arrays pass the check. this not to filter
    // out values within a chunk, we rely on datafusion's default
    // filtering for that. basically this here is to handle filter
    // pushdowns.
    fn with_filter(mut self, filter: ZarrChunkFilter) -> ZarrQueryResult<Self> {
        self.filter = Some(filter);
        Ok(self)
    }
}

impl<T> Stream for ZarrRecordBatchStream<T>
where
    T: AsyncReadableListableStorageTraits + Unpin + Send + ?Sized + 'static,
{
    type Item = Result<RecordBatch, ArrowError>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            let cols: Vec<_> = self
                .projected_schema_ref
                .fields()
                .iter()
                .map(|f| f.name().to_owned())
                .collect();
            let mut filter_cols: Option<Vec<String>> = None;
            if let Some(filter) = &self.filter {
                filter_cols = Some(
                    filter
                        .get_schema_ref()
                        .fields()
                        .iter()
                        .map(|f| f.name().to_owned())
                        .collect(),
                );
            }
            match &mut self.state {
                ZarrStreamState::Init => {
                    let chk_idx = self.pop_chunk_idx();
                    if chk_idx.is_none() {
                        self.state = ZarrStreamState::Done;
                        return Poll::Ready(None);
                    }
                    let chk_idx = chk_idx.unwrap();

                    let store = self.zarr_store.take().expect("lost zarr store!");
                    if let Some(filter_cols) = filter_cols {
                        let fut = store.get_chunk(filter_cols, chk_idx).boxed();
                        self.state = ZarrStreamState::ReadingForFilter(fut);
                    } else {
                        let fut = store.get_chunk(cols, chk_idx).boxed();
                        self.state = ZarrStreamState::Reading(fut);
                    }
                }
                ZarrStreamState::Reading(f) => {
                    let res = ready!(f.poll_unpin(cx));
                    match res {
                        Err(e) => {
                            self.state = ZarrStreamState::Error;
                            let e = ArrowError::from_external_error(Box::new(e));
                            return Poll::Ready(Some(Err(e)));
                        }
                        Ok((store, chunk, _)) => {
                            self.zarr_store = Some(store);
                            self.state = ZarrStreamState::Init;
                            return Poll::Ready(Some(
                                chunk
                                    .into_record_batch(&self.projected_schema_ref)
                                    .map_err(|e| ArrowError::from_external_error(Box::new(e))),
                            ));
                        }
                    }
                }
                ZarrStreamState::ReadingForFilter(f) => {
                    let res = ready!(f.poll_unpin(cx));
                    match res {
                        Err(e) => {
                            self.state = ZarrStreamState::Error;
                            let e = ArrowError::from_external_error(Box::new(e));
                            return Poll::Ready(Some(Err(e)));
                        }
                        Ok((store, chunk, chk_idx)) => {
                            let chk_data;
                            if let Some(filter) = &self.filter {
                                chk_data = chunk
                                    .into_record_batch(&filter.get_schema_ref())
                                    .map_err(|e| ArrowError::from_external_error(Box::new(e)))?;
                            } else {
                                panic!("lost filter schema!");
                            }

                            if let Some(filter) = &mut self.filter {
                                let filter_res = filter.evaluate(&chk_data);
                                match filter_res {
                                    Ok(filter_passed) => {
                                        if filter_passed {
                                            let fut = store.get_chunk(cols, chk_idx).boxed();
                                            self.state = ZarrStreamState::Reading(fut);
                                        // this is effectively the filtering mechanism. the evaluate
                                        // method checks if any data in the chunk with just the filter
                                        // columns pass the check, if not, we go back to the init
                                        // state and simply don't read the full chunk for the current
                                        // chunk index.
                                        } else {
                                            self.zarr_store = Some(store);
                                            self.state = ZarrStreamState::Init;
                                        }
                                    }
                                    Err(e) => {
                                        self.state = ZarrStreamState::Error;
                                        return Poll::Ready(Some(Err(e)));
                                    }
                                };
                            } else {
                                panic!("lost filter!");
                            }
                        }
                    }
                }
                ZarrStreamState::Error => return Poll::Ready(None),
                ZarrStreamState::Done => return Poll::Ready(None),
            }
        }
    }
}

#[cfg(test)]
mod zarr_stream_tests {
    use super::*;
    use crate::test_utils::{
        get_lat_lon_data_store, validate_names_and_types, validate_primitive_column,
    };
    use futures_util::TryStreamExt;

    #[tokio::test]
    async fn read_data_test() {
        let (wrapper, schema) = get_lat_lon_data_store(true, 0.0, "lat_lon_data").await;
        let store = wrapper.get_store();

        let stream = ZarrRecordBatchStream::new(store, Arc::new(schema), None, None)
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
    async fn read_missing_chunks_test() {
        let fillvalue = 1234.0;
        let (wrapper, schema) =
            get_lat_lon_data_store(false, fillvalue, "lat_lon_empty_data").await;
        let store = wrapper.get_store();

        let stream = ZarrRecordBatchStream::new(store, Arc::new(schema), None, None)
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
}
