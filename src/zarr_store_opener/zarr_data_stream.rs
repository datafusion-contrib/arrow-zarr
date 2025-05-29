use super::{filter::ZarrChunkFilter, projection::ZarrQueryProjection};
use crate::errors::zarr_errors::{ZarrQueryError, ZarrQueryResult};
use arrow::array::*;
use arrow::datatypes::*;
use arrow::record_batch::RecordBatch;
use arrow_schema::{DataType, Field, Fields, Schema};
use bytes::Bytes;
use futures::stream::Stream;
use futures::{ready, FutureExt};
use futures_util::future::BoxFuture;
use itertools::iproduct;
use std::borrow::Cow;
use std::collections::VecDeque;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use zarrs::array::chunk_grid;
use zarrs::array::codec::{ArrayToBytesCodecTraits, CodecOptions};
use zarrs::array::{
    Array, ArrayBytes, ArrayMetadata, ArrayMetadataV3, ArraySize, DataType as zDataType,
    ElementOwned,
};
use zarrs::array_subset::ArraySubset;
use zarrs_metadata::v3::array::data_type::DataTypeMetadataV3;
use zarrs_storage::{AsyncReadableListableStorageTraits, StoreKey, StorePrefix};

//********************************************
// various utils to handle metadata from the zarr array, data types,
// schemas, etc...
//********************************************
fn extract_chunk_size(meta: &ArrayMetadataV3) -> ZarrQueryResult<Vec<u64>> {
    let err = "Failed to retrieve chunk shape for array";
    let chunks = meta
        .chunk_grid
        .configuration()
        .ok_or_else(|| ZarrQueryError::InvalidMetadata(err.into()))?
        .get("chunk_shape")
        .ok_or_else(|| ZarrQueryError::InvalidMetadata(err.into()))?
        .as_array()
        .ok_or_else(|| ZarrQueryError::InvalidMetadata(err.into()))?
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    Ok(chunks)
}

fn get_dims<T>(arr: &Array<T>) -> ZarrQueryResult<Option<Vec<String>>> {
    if let Some(dims) = arr.dimension_names() {
        let dims: Vec<_> = dims
            .iter()
            .map(|d| d.as_str())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                ZarrQueryError::InvalidMetadata("Dimensions without a name not supported".into())
            })?
            .iter()
            .map(|s| s.to_string())
            .collect();

        return Ok(Some(dims));
    }

    Ok(None)
}

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

fn extract_columns(prefix: &str, keys: Vec<StoreKey>) -> Vec<String> {
    let mut cols: HashSet<String> = HashSet::new();
    for key in keys {
        let key = key.as_str().to_string().replace(prefix, "");
        let key = key.split("/").next().unwrap();
        cols.insert(key.to_string());
    }

    cols.into_iter().collect()
}

fn extract_meta_from_array<T>(array: &Array<T>) -> ZarrQueryResult<&ArrayMetadataV3> {
    let meta = array.metadata();
    let meta = match meta {
        ArrayMetadata::V3(meta) => Ok(meta),
        _ => Err(ZarrQueryError::InvalidMetadata(
            "Only v3 metadata is suppoerted".into(),
        )),
    }?;

    Ok(meta)
}

fn create_schema<T>(
    cols: Vec<String>,
    arrays: &HashMap<String, Arc<Array<T>>>,
) -> ZarrQueryResult<Schema> {
    let fields: Fields = cols
        .iter()
        .map(|c| arrays.get(c))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| ZarrQueryError::InvalidMetadata("Array missing from array map".into()))?
        .into_iter()
        .map(|a| extract_meta_from_array(a))
        .collect::<ZarrQueryResult<Vec<_>>>()?
        .into_iter()
        .map(|m| get_schema_type(&m.data_type))
        .collect::<ZarrQueryResult<Vec<_>>>()?
        .into_iter()
        .zip(cols)
        .map(|(d, c)| Arc::new(Field::new(c, d, false)))
        .collect();

    Ok(Schema::new(fields))
}

fn resolve_vector(
    dims: &ZarrDimensions,
    vecs: HashMap<String, Vec<u64>>,
) -> ZarrQueryResult<Vec<u64>> {
    let mut final_vec: Option<Vec<u64>> = None;
    for (k, vec) in vecs.iter() {
        if let Some(final_vec) = &final_vec {
            if let Some(pos) = dims.get_dim_position(k) {
                if final_vec[pos as usize] != vec[0] {
                    return Err(ZarrQueryError::InvalidMetadata(
                        "Mismatch between vectors for different arrays".into(),
                    ));
                }
            } else if final_vec != vec {
                return Err(ZarrQueryError::InvalidMetadata(
                    "Mismatch between vectors for different arrays".into(),
                ));
            }
        } else if !dims.is_dimension(k) {
            final_vec = Some(vec.clone());
        }
    }

    if let Some(final_vec) = final_vec {
        Ok(final_vec)
    } else {
        let mut final_vec: Vec<u64> = vec![0; dims.dim_positions.len()];
        for (k, p) in dims.dim_positions.iter() {
            final_vec[*p as usize] = vecs.get(k).ok_or_else(|| {
                ZarrQueryError::InvalidProjection("Array is missing from array map".into())
            })?[0];
        }
        Ok(final_vec)
    }
}

//********************************************
// a struct to handle coordinate variables, and "broadcasting" them
// when reading multidimensional data.
//********************************************
#[derive(Debug)]
struct ZarrDimensions {
    dim_positions: HashMap<String, u64>,
}

impl ZarrDimensions {
    fn new<T>(arrays: &HashMap<String, Array<T>>) -> ZarrQueryResult<Self> {
        let mut dim_positions: HashMap<String, u64> = HashMap::new();

        // first pass, determine what the dimensions are and what the
        // chunk dimensionaliy is.
        let mut dims: Vec<String> = Vec::new();
        let mut chunk_dims: Option<Vec<String>> = None;
        for (k, arr) in arrays.iter() {
            let arr_dims = get_dims(arr)?;
            if let Some(arr_dims) = arr_dims {
                if arr_dims.contains(k) {
                    if arr_dims.len() != 1 {
                        return Err(ZarrQueryError::InvalidMetadata("Invalid dimension".into()));
                    }
                    dims.push(k.to_string());
                } else {
                    if let Some(chk_dims) = &chunk_dims {
                        if chk_dims != &arr_dims {
                            return Err(ZarrQueryError::InvalidMetadata(
                                "Mismatch between variables' dimensions".into(),
                            ));
                        }
                    }
                    chunk_dims = Some(arr_dims);
                }
            }
        }

        // second pass, determine what at each of the dims position in
        // the chunks' dimensionality.
        let chunk_dims = if let Some(chk_dims) = chunk_dims {
            chk_dims
        } else {
            let mut dims_copy = dims.clone();
            dims_copy.sort();
            dims_copy
        };

        if dims.len() != chunk_dims.len() {
            return Err(ZarrQueryError::InvalidMetadata(
                "Mismatch with the number of dimensions".into(),
            ));
        }
        for d in dims {
            let pos = chunk_dims.iter().position(|s| &d == s).ok_or_else(|| {
                ZarrQueryError::InvalidMetadata(
                    "Dimension is missing from variable's dimension list".into(),
                )
            })?;
            dim_positions.insert(d.to_string(), pos as u64);
        }

        Ok(Self { dim_positions })
    }

    fn is_dimension(&self, var: &str) -> bool {
        self.dim_positions.contains_key(var)
    }

    fn get_dim_position(&self, var: &str) -> Option<u64> {
        self.dim_positions.get(var).cloned()
    }

    fn reduce_if_dim(&self, v: Vec<u64>, var: &str) -> Vec<u64> {
        if let Some(pos) = self.dim_positions.get(var) {
            return vec![v[*pos as usize]];
        }

        v
    }

    fn broadcast_if_dim<T: Clone>(
        &self,
        dim_name: &str,
        data: Vec<T>,
        full_chunk_shape: &Vec<u64>,
    ) -> ZarrQueryResult<Vec<T>> {
        let dim_idx = self.get_dim_position(dim_name);
        if dim_idx.is_none() {
            return Ok(data);
        }
        let dim_idx = dim_idx.unwrap();

        if full_chunk_shape.len() == 1 {
            return Ok(data);
        }

        match (full_chunk_shape.len(), dim_idx) {
            (2, 0) => Ok(data
                .into_iter()
                .flat_map(|v| std::iter::repeat(v).take(full_chunk_shape[1] as usize))
                .collect()),
            (2, 1) => Ok(vec![&data[..]; full_chunk_shape[0] as usize].concat()),
            (3, 0) => Ok(data
                .into_iter()
                .flat_map(|v| {
                    std::iter::repeat(v).take((full_chunk_shape[1] * full_chunk_shape[2]) as usize)
                })
                .collect()),
            (3, 1) => {
                let v: Vec<_> = data
                    .into_iter()
                    .flat_map(|v| std::iter::repeat(v).take(full_chunk_shape[2] as usize))
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
struct ArrayInterface<T: AsyncReadableListableStorageTraits> {
    name: String,
    arr: Arc<Array<T>>,
    dims: Arc<ZarrDimensions>,
    full_chunk_shape: Vec<u64>,
    chk_index: Vec<u64>,
}

// T doesn't need to be Clone, but deriving apparently requires
// that, so I have implement manually.
impl<T: AsyncReadableListableStorageTraits> Clone for ArrayInterface<T> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.to_string(),
            arr: self.arr.clone(),
            dims: self.dims.clone(),
            full_chunk_shape: self.full_chunk_shape.clone(),
            chk_index: self.chk_index.clone(),
        }
    }
}

enum BytesFromArray {
    //Decoded(ArrayBytes<'a>),
    Decoded(Bytes),
    Encoded(Option<Bytes>),
}

impl<T: AsyncReadableListableStorageTraits + 'static> ArrayInterface<T> {
    fn new(
        name: String,
        arr: Arc<Array<T>>,
        dims: Arc<ZarrDimensions>,
        full_chunk_shape: Vec<u64>,
        mut chk_index: Vec<u64>,
    ) -> Self {
        chk_index = dims.reduce_if_dim(chk_index, &name);
        Self {
            name,
            arr,
            dims,
            full_chunk_shape,
            chk_index,
        }
    }

    async fn read_bytes(&self) -> ZarrQueryResult<BytesFromArray> {
        let chunk_grid = self.arr.chunk_grid_shape().ok_or_else(|| {
            ZarrQueryError::InvalidMetadata("Array is missing its chunk grid shape".into())
        })?;

        let is_edge_grid = self
            .chk_index
            .iter()
            .zip(chunk_grid.iter())
            .any(|(i, g)| i == &(g - 1));
        if is_edge_grid {
            let arr_shape = self.arr.shape();
            let chunk_shape = self.arr.chunk_shape(&self.chk_index)?.to_array_shape();
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
        } else {
            let data = self
                .arr
                .async_retrieve_encoded_chunk(&self.chk_index)
                .await?;
            Ok(BytesFromArray::Encoded(data))
        }
    }

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
                        .dims
                        .reduce_if_dim(self.full_chunk_shape.clone(), &self.name);
                    let num_elems = chk_shp.iter().fold(1, |mut acc, x| {
                        acc *= x;
                        acc
                    });
                    let array_size = ArraySize::new(self.arr.data_type().size(), num_elems);
                    ArrayBytes::new_fill_value(array_size, self.arr.fill_value())
                }
            }
            BytesFromArray::Decoded(bytes) => ArrayBytes::Fixed(bytes.to_vec().into()),
        };

        let t = self.arr.data_type();
        match t {
            zDataType::Bool => {
                let arr_ref: BooleanArray = self
                    .dims
                    .broadcast_if_dim(
                        &self.name,
                        bool::from_array_bytes(t, decoded_bytes)?,
                        &self.full_chunk_shape,
                    )?
                    .into();
                Ok(Arc::new(arr_ref) as ArrayRef)
            }
            zDataType::UInt8 => {
                let arr_ref: PrimitiveArray<UInt8Type> = self
                    .dims
                    .broadcast_if_dim(
                        &self.name,
                        u8::from_array_bytes(t, decoded_bytes)?,
                        &self.full_chunk_shape,
                    )?
                    .into();
                Ok(Arc::new(arr_ref) as ArrayRef)
            }
            zDataType::UInt16 => {
                let arr_ref: PrimitiveArray<UInt16Type> = self
                    .dims
                    .broadcast_if_dim(
                        &self.name,
                        u16::from_array_bytes(t, decoded_bytes)?,
                        &self.full_chunk_shape,
                    )?
                    .into();
                Ok(Arc::new(arr_ref) as ArrayRef)
            }
            zDataType::UInt32 => {
                let arr_ref: PrimitiveArray<UInt32Type> = self
                    .dims
                    .broadcast_if_dim(
                        &self.name,
                        u32::from_array_bytes(t, decoded_bytes)?,
                        &self.full_chunk_shape,
                    )?
                    .into();
                Ok(Arc::new(arr_ref) as ArrayRef)
            }
            zDataType::UInt64 => {
                let arr_ref: PrimitiveArray<UInt64Type> = self
                    .dims
                    .broadcast_if_dim(
                        &self.name,
                        u64::from_array_bytes(t, decoded_bytes)?,
                        &self.full_chunk_shape,
                    )?
                    .into();
                Ok(Arc::new(arr_ref) as ArrayRef)
            }
            zDataType::Int8 => {
                let arr_ref: PrimitiveArray<Int8Type> = self
                    .dims
                    .broadcast_if_dim(
                        &self.name,
                        i8::from_array_bytes(t, decoded_bytes)?,
                        &self.full_chunk_shape,
                    )?
                    .into();
                Ok(Arc::new(arr_ref) as ArrayRef)
            }
            zDataType::Int16 => {
                let arr_ref: PrimitiveArray<Int16Type> = self
                    .dims
                    .broadcast_if_dim(
                        &self.name,
                        i16::from_array_bytes(t, decoded_bytes)?,
                        &self.full_chunk_shape,
                    )?
                    .into();
                Ok(Arc::new(arr_ref) as ArrayRef)
            }
            zDataType::Int32 => {
                let arr_ref: PrimitiveArray<Int32Type> = self
                    .dims
                    .broadcast_if_dim(
                        &self.name,
                        i32::from_array_bytes(t, decoded_bytes)?,
                        &self.full_chunk_shape,
                    )?
                    .into();
                Ok(Arc::new(arr_ref) as ArrayRef)
            }
            zDataType::Int64 => {
                let arr_ref: PrimitiveArray<Int64Type> = self
                    .dims
                    .broadcast_if_dim(
                        &self.name,
                        i64::from_array_bytes(t, decoded_bytes)?,
                        &self.full_chunk_shape,
                    )?
                    .into();
                Ok(Arc::new(arr_ref) as ArrayRef)
            }
            zDataType::Float32 => {
                let arr_ref: PrimitiveArray<Float32Type> = self
                    .dims
                    .broadcast_if_dim(
                        &self.name,
                        f32::from_array_bytes(t, decoded_bytes)?,
                        &self.full_chunk_shape,
                    )?
                    .into();
                Ok(Arc::new(arr_ref) as ArrayRef)
            }
            zDataType::Float64 => {
                let arr_ref: PrimitiveArray<Float64Type> = self
                    .dims
                    .broadcast_if_dim(
                        &self.name,
                        f64::from_array_bytes(t, decoded_bytes)?,
                        &self.full_chunk_shape,
                    )?
                    .into();
                Ok(Arc::new(arr_ref) as ArrayRef)
            }
            zDataType::String => {
                let arr_ref: StringArray = self
                    .dims
                    .broadcast_if_dim(
                        &self.name,
                        String::from_array_bytes(t, decoded_bytes)?,
                        &self.full_chunk_shape,
                    )?
                    .into();
                Ok(Arc::new(arr_ref) as ArrayRef)
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

    fn into_record_batch(mut self, schema: &Schema) -> ZarrQueryResult<RecordBatch> {
        let array_refs: Vec<(String, ArrayRef)> = schema
            .fields()
            .iter()
            .map(|f| self.data.remove(f.name()))
            .collect::<Option<Vec<ArrayRef>>>()
            .ok_or_else(|| {
                ZarrQueryError::InvalidProjection("Array missing from array map".into())
            })?
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
struct ZarrStore<T: AsyncReadableListableStorageTraits> {
    arrays: HashMap<String, Arc<Array<T>>>,
    dimensions: Arc<ZarrDimensions>,
    chunk_grid_shape: Vec<u64>,
    array_shape: Vec<u64>,
}

impl<T: AsyncReadableListableStorageTraits + 'static> ZarrStore<T> {
    fn new(arrays: HashMap<String, Array<T>>) -> ZarrQueryResult<Self> {
        let dimensions = ZarrDimensions::new(&arrays)?;

        let mut shape: Option<Vec<u64>> = None;
        let mut chunk: Option<Vec<u64>> = None;
        for (k, arr) in arrays.iter() {
            if !dimensions.is_dimension(k) {
                if let Some(shape) = &shape {
                    if shape != arr.shape() {
                        return Err(ZarrQueryError::InvalidMetadata(
                            "Mismatch between variables' shapes".into(),
                        ));
                    }
                }
                shape = Some(arr.shape().to_vec());

                let arr_chk = extract_chunk_size(extract_meta_from_array(arr)?)?;
                if let Some(chunk) = chunk {
                    if chunk != arr_chk {
                        return Err(ZarrQueryError::InvalidMetadata(
                            "Mismatch between variables' chunks".into(),
                        ));
                    }
                }
                chunk = Some(arr_chk);
            }
        }

        if let (Some(chunk), Some(shape)) = (chunk, shape) {
            for (dim, pos) in dimensions.dim_positions.iter() {
                let dim_arr = arrays.get(dim).ok_or_else(|| {
                    ZarrQueryError::InvalidProjection("Array missing from array map".into())
                })?;
                let dim_shape = dim_arr.shape()[0];
                let dim_chunk = extract_chunk_size(extract_meta_from_array(dim_arr)?)?[0];

                if shape[*pos as usize] != dim_shape {
                    return Err(ZarrQueryError::InvalidMetadata(
                        "Mismatch between variables and dimensions shapes".into(),
                    ));
                }
                if chunk[*pos as usize] != dim_chunk {
                    return Err(ZarrQueryError::InvalidMetadata(
                        "Mismatch between variables and dimensions chunks".into(),
                    ));
                }
            }
        }

        let mut chk_grid_shapes: HashMap<String, Vec<u64>> = HashMap::new();
        for (k, arr) in arrays.iter() {
            chk_grid_shapes.insert(
                k.to_owned(),
                arr.chunk_grid_shape().ok_or_else(|| {
                    ZarrQueryError::InvalidProjection(
                        "Array is missing its chunk grid shape".into(),
                    )
                })?,
            );
        }
        let chunk_grid_shape = resolve_vector(&dimensions, chk_grid_shapes)?;

        let mut arr_shapes: HashMap<String, Vec<u64>> = HashMap::new();
        for (k, arr) in arrays.iter() {
            arr_shapes.insert(k.to_owned(), arr.shape().to_vec());
        }
        let array_shape = resolve_vector(&dimensions, arr_shapes)?;

        Ok(Self {
            arrays: arrays.into_iter().map(|(k, a)| (k, Arc::new(a))).collect(),
            dimensions: Arc::new(dimensions),
            chunk_grid_shape,
            array_shape,
        })
    }

    fn get_chunk_shape(&self, chk_idx: &[u64]) -> ZarrQueryResult<Vec<u64>> {
        let mut chk_shapes: HashMap<String, Vec<u64>> = HashMap::new();
        for (k, arr) in self.arrays.iter() {
            let chk_idx = self.dimensions.reduce_if_dim(chk_idx.to_vec(), k);
            chk_shapes.insert(k.to_owned(), arr.chunk_shape(&chk_idx)?.to_array_shape());
        }
        let mut chk_shape = resolve_vector(&self.dimensions, chk_shapes)?;

        let is_edge_grid = chk_idx
            .iter()
            .zip(self.chunk_grid_shape.iter())
            .any(|(i, g)| i == &(g - 1));

        if is_edge_grid {
            chk_shape = chk_idx
                .iter()
                .zip(self.array_shape.iter())
                .zip(chk_shape.iter())
                .map(|((i, a), c)| std::cmp::min(a - i * c, *c))
                .collect();
        }

        Ok(chk_shape)
    }

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
                    self.dimensions.clone(),
                    full_chunk_shape.clone(),
                    chk_idx.clone(),
                ))
            })
            .collect::<ZarrQueryResult<Vec<_>>>()
            .unwrap();

        let mut data = arr_interfaces[0].read_bytes().await?;
        let mut last_arr = arr_interfaces[0].clone();

        for arr in arr_interfaces.iter().skip(1) {
            let last_col = last_arr.name.to_string();
            let compute_fut =
                tokio::task::spawn_blocking(move || last_arr.decode_data(data)).boxed();
            let arr_for_future = arr.clone();
            let io_fut = tokio::spawn(async move { arr_for_future.read_bytes().await }).boxed();

            if let (Ok(d), Ok(array_ref)) = tokio::join!(io_fut, compute_fut) {
                data = d?;
                chk_data.add_data(last_col, array_ref?);
            } else {
                return Err(ZarrQueryError::InvalidCompute(
                    "Unable to retrieve decoded chunk".into(),
                ));
            }
            last_arr = arr.clone();
        }

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
enum ZarrStreamState<T: AsyncReadableListableStorageTraits> {
    Init,
    Reading(ChunkFuture<T>),
    ReadingForFilter(ChunkFuture<T>),
    Error,
    Done,
}

pub struct ZarrRecordBatchStream<T: AsyncReadableListableStorageTraits> {
    zarr_store: Option<ZarrStore<T>>,
    schema: Schema,
    filter: Option<ZarrChunkFilter>,
    filter_schema: Option<Schema>,
    all_cols: Vec<String>,
    state: ZarrStreamState<T>,
    chunk_indices: VecDeque<Vec<u64>>,
}

impl<T: AsyncReadableListableStorageTraits + 'static> ZarrRecordBatchStream<T> {
    async fn new(
        store: Arc<T>,
        group: Option<String>,
        projection: Option<ZarrQueryProjection>,
    ) -> ZarrQueryResult<Self> {
        let proj = if let Some(projection) = projection {
            projection
        } else {
            ZarrQueryProjection::all()
        };

        let grp = if let Some(group) = group {
            [group, "/".into()].join("")
        } else {
            "".to_string()
        };

        let store_keys = store.list_prefix(&StorePrefix::new(&grp)?).await?;
        let all_cols = extract_columns(&grp, store_keys);
        let cols = proj.apply_selection(&all_cols)?;

        let mut arrays: HashMap<String, Array<T>> = HashMap::new();
        for col in &cols {
            let path = PathBuf::from(&grp)
                .join(["/", col].join(""))
                .into_os_string()
                .to_str()
                .expect("could not form path from group and column name")
                .to_string();
            let arr = Array::async_open(store.clone(), &path).await?;
            arrays.insert(col.to_string(), arr);
        }

        let zarr_store = ZarrStore::new(arrays)?;
        let schema = create_schema(cols, &zarr_store.arrays)?;

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
            schema,
            filter: None,
            filter_schema: None,
            all_cols,
            state: ZarrStreamState::Init,
            chunk_indices,
        })
    }

    fn pop_chunk_idx(&mut self) -> Option<Vec<u64>> {
        self.chunk_indices.pop_front()
    }

    fn with_filter(mut self, filter: ZarrChunkFilter) -> ZarrQueryResult<Self> {
        let proj = filter.get_all_projections()?;
        let filter_cols = proj.apply_selection(&self.all_cols)?;

        let schema = if let Some(zarr_store) = &self.zarr_store {
            create_schema(filter_cols, &zarr_store.arrays)?
        } else {
            return Err(ZarrQueryError::InvalidMetadata(
                "Zarr store missing when creating filter".into(),
            ));
        };
        self.filter = Some(filter);
        self.filter_schema = Some(schema);
        Ok(self)
    }
}

impl<T> Stream for ZarrRecordBatchStream<T>
where
    T: AsyncReadableListableStorageTraits + Unpin + Send + 'static,
{
    type Item = ZarrQueryResult<RecordBatch>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            let cols: Vec<_> = self
                .schema
                .fields()
                .iter()
                .map(|f| f.name().to_owned())
                .collect();
            let mut filter_cols: Option<Vec<String>> = None;
            if let Some(filter_schema) = &self.filter_schema {
                filter_cols = Some(
                    filter_schema
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
                            return Poll::Ready(Some(Err(e)));
                        }
                        Ok((store, chunk, _)) => {
                            self.zarr_store = Some(store);
                            self.state = ZarrStreamState::Init;
                            return Poll::Ready(Some(chunk.into_record_batch(&self.schema)));
                        }
                    }
                }
                ZarrStreamState::ReadingForFilter(f) => {
                    let res = ready!(f.poll_unpin(cx));
                    match res {
                        Err(e) => {
                            self.state = ZarrStreamState::Error;
                            return Poll::Ready(Some(Err(e)));
                        }
                        Ok((store, chunk, chk_idx)) => {
                            let chk_data;
                            if let Some(filter_schema) = &self.filter_schema {
                                chk_data = chunk.into_record_batch(filter_schema)?;
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
                                        } else {
                                            self.zarr_store = Some(store);
                                            self.state = ZarrStreamState::Init;
                                        }
                                    }
                                    Err(e) => {
                                        self.state = ZarrStreamState::Error;
                                        return Poll::Ready(Some(Err(e.into())));
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
        validate_names_and_types, validate_primitive_column, write_1D_float_array,
        write_2D_float_array, StoreWrapper,
    };
    use crate::zarr_store_opener::filter::{
        ZarrArrowPredicate, ZarrArrowPredicateFn, ZarrChunkFilter,
    };
    use crate::zarr_store_opener::projection::ZarrQueryProjection;
    use arrow::compute::kernels::cmp::lt;
    use futures_util::TryStreamExt;

    async fn get_lat_lon_data_store(
        write_data: bool,
        fillvalue: f64,
        dir_name: &str,
    ) -> StoreWrapper {
        let wrapper = StoreWrapper::new(dir_name.into());
        let store = wrapper.get_store();

        let lats = vec![35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0];
        write_1D_float_array(
            lats,
            8,
            3,
            store.clone(),
            "/lat",
            Some(["lat".into()].to_vec()),
            true,
        )
        .await;

        let lons = vec![
            -120.0, -119.0, -118.0, -117.0, -116.0, -115.0, -114.0, -113.0,
        ];
        write_1D_float_array(
            lons,
            8,
            3,
            store.clone(),
            "/lon",
            Some(["lon".into()].to_vec()),
            true,
        )
        .await;

        let data: Option<Vec<_>> = if write_data {
            Some((0..64).map(|i| i as f64).collect())
        } else {
            None
        };
        write_2D_float_array(
            data,
            fillvalue,
            (8, 8),
            (3, 3),
            store.clone(),
            "/data",
            Some(["lat".into(), "lon".into()].to_vec()),
            true,
        )
        .await;

        wrapper
    }

    #[tokio::test]
    async fn read_data_test() {
        let wrapper = get_lat_lon_data_store(true, 0.0, "lat_lon_data").await;
        let store = wrapper.get_store();

        let stream = ZarrRecordBatchStream::new(store, None, None).await.unwrap();
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
        let wrapper = get_lat_lon_data_store(false, fillvalue, "lat_lon_empty_data").await;
        let store = wrapper.get_store();

        let stream = ZarrRecordBatchStream::new(store, None, None).await.unwrap();
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
    async fn read_data_with_filter_test() {
        let wrapper = get_lat_lon_data_store(true, 0.0, "lat_lon_data_w_filter").await;
        let store = wrapper.get_store();

        let mut filters: Vec<Box<dyn ZarrArrowPredicate>> = Vec::new();
        let f = ZarrArrowPredicateFn::new(
            ZarrQueryProjection::keep(vec!["lat".to_string()]),
            move |batch| {
                lt(
                    batch.column_by_name("lat").unwrap(),
                    &Scalar::new(&Float64Array::from(vec![39.0])),
                )
            },
        );
        filters.push(Box::new(f));
        let f = ZarrArrowPredicateFn::new(
            ZarrQueryProjection::keep(vec!["lon".to_string()]),
            move |batch| {
                lt(
                    batch.column_by_name("lon").unwrap(),
                    &Scalar::new(&Float64Array::from(vec![-116.0])),
                )
            },
        );
        filters.push(Box::new(f));
        let filter = ZarrChunkFilter::new(filters);

        let stream = ZarrRecordBatchStream::new(store, None, None)
            .await
            .unwrap()
            .with_filter(filter)
            .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        let target_types = HashMap::from([
            ("lat".to_string(), DataType::Float64),
            ("lon".to_string(), DataType::Float64),
            ("data".to_string(), DataType::Float64),
        ]);
        validate_names_and_types(&target_types, &records[0]);
        assert_eq!(records.len(), 4);

        // middle chunk, should still be 3x3 because filters only filter out whole
        // chunks if they don't contain anything that passes the predicate, if there's
        // even a single value that does pass, we keep the whole chunk.
        validate_primitive_column::<Float64Type, f64>(
            "lat",
            &records[3],
            &[38., 38., 38., 39., 39., 39., 40., 40., 40.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            &records[3],
            &[
                -117.0, -116.0, -115.0, -117.0, -116.0, -115.0, -117.0, -116.0, -115.0,
            ],
        );
        validate_primitive_column::<Float64Type, f64>(
            "data",
            &records[3],
            &[27.0, 28.0, 29.0, 35.0, 36.0, 37.0, 43.0, 44.0, 45.0],
        );
    }
}
