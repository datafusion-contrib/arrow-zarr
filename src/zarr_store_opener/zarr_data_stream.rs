use super::{filter::ZarrChunkFilter, projection::ZarrQueryProjection};
use crate::errors::zarr_errors::{ZarrQueryError, ZarrQueryResult};
use arrow_schema::{DataType, Field, Fields, Schema};
use hashbag::HashBag;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use zarrs::array::{Array, ArrayMetadata, ArrayMetadataV3};
use zarrs_metadata::v3::array::data_type::DataTypeMetadataV3;
use zarrs_storage::{AsyncReadableListableStorageTraits, StoreKey, StorePrefix};

//********************************************
// a struct to handle coordinate variables, and "broadcasting" them
// when reading multidimensional data.
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

fn get_dims(meta: &ArrayMetadataV3) -> ZarrQueryResult<Option<Vec<String>>> {
    if let Some(dims) = &meta.dimension_names {
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

struct ZarrDimensions {
    shapes: HashMap<String, Vec<u64>>,
    full_shape: Option<Vec<u64>>,
    full_chunk: Option<Vec<u64>>,
}

impl ZarrDimensions {
    fn new() -> Self {
        Self {
            shapes: HashMap::new(),
            full_shape: None,
            full_chunk: None,
        }
    }

    fn get_dim_inferred_shapes(
        &self,
        dim_names: &Vec<String>,
        idx: usize,
    ) -> ZarrQueryResult<Vec<u64>> {
        Ok(dim_names
            .into_iter()
            .map(|d| self.shapes.get(d))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| ZarrQueryError::InvalidMetadata("Unknown dimension".into()))?
            .into_iter()
            .map(|d| d[idx])
            .collect())
    }

    fn add_dim(&mut self, dim_name: String, meta: &ArrayMetadataV3) -> ZarrQueryResult<()> {
        if self.shapes.contains_key(&dim_name) {
            return Err(ZarrQueryError::InvalidMetadata(
                "Dimension already present in coordinates".into(),
            ));
        }

        let chunks = extract_chunk_size(meta)?;
        self.shapes
            .insert(dim_name.clone(), vec![meta.shape[0], chunks[0]]);
        Ok(())
    }

    fn validate_array(&self, arr_meta: &ArrayMetadataV3) -> ZarrQueryResult<()> {
        let dims = get_dims(arr_meta)?;
        if let Some(dims) = dims {
            if dims.iter().collect::<HashBag<&String>>()
                != self.shapes.keys().collect::<HashBag<&String>>()
            {
                return Err(ZarrQueryError::InvalidMetadata(
                    "Arrays must depend on all dimensions being queried".into(),
                ));
            }

            let target_shape = self.get_dim_inferred_shapes(&dims, 0)?;
            if arr_meta.shape != target_shape {
                return Err(ZarrQueryError::InvalidMetadata(
                    "Mismatch between array shape and dimensions".into(),
                ));
            }

            let target_chunks = self.get_dim_inferred_shapes(&dims, 1)?;
            let arr_chunks = extract_chunk_size(arr_meta)?;
            if arr_chunks != target_chunks {
                return Err(ZarrQueryError::InvalidMetadata(
                    "Mismatch between array chunks and dimensions".into(),
                ));
            }
        } else if !self.shapes.is_empty() {
            return Err(ZarrQueryError::InvalidMetadata(
                "Array has no dimensions but some dimensions were queried".into(),
            ));
        }

        Ok(())
    }
}

//********************************************
// a few helpers to handle metadata and types
//********************************************
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
        DataTypeMetadataV3::Float16 => Ok(DataType::Float16),
        DataTypeMetadataV3::Float32 => Ok(DataType::Float32),
        DataTypeMetadataV3::Float64 => Ok(DataType::Float64),
        DataTypeMetadataV3::String => Ok(DataType::Utf8),
        DataTypeMetadataV3::Bytes => Ok(DataType::Binary),
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

fn build_dimensions<T>(arrays: &HashMap<String, Array<T>>) -> ZarrQueryResult<ZarrDimensions> {
    // we first need to check if the shapes, chunks, and dimensions, if there are
    // any, all line up. All variables that are not dimensions must have the same
    // shape and chunks. If there are dimensions, all variables must depend on all
    // the dimensions, and the shapes and chunks must match the 1D shapes and chunks
    // of the dimensions. If everything lines up, we return the shape and chunk
    // size that every non dimension variable conforms to.

    // first pass, determine dimensions, if there are any.
    let mut zarr_dims = ZarrDimensions::new();
    for (k, arr) in arrays.iter() {
        let meta = extract_meta_from_array(arr)?;

        let dims = get_dims(meta)?;
        if let Some(dims) = dims {
            if meta.shape.len() == 1 && dims.len() == 1 && k == &dims[0] {
                zarr_dims.add_dim(k.to_string(), meta)?;
            }
        }
    }

    // second, go over non dimension variables and validate the shapes
    // and chunk sizes.
    let mut curr_shape: Option<Vec<u64>> = None;
    let mut curr_chunks: Option<Vec<u64>> = None;
    for (k, arr) in arrays.iter() {
        let meta = extract_meta_from_array(arr)?;

        // validate the shape and chunk sizes given the dimensions.
        zarr_dims.validate_array(meta)?;

        // all variables, if they are not a dimensions, must have the
        // same shape. We need this check here in case the data in the
        // just doesn't have any dimensions.
        let arr_chunks = extract_chunk_size(meta)?;
        if let Some(curr_chunks) = &curr_chunks {
            if curr_chunks != &arr_chunks {}
        } else {
            curr_chunks = Some(arr_chunks);
        }

        if let Some(curr_shape) = &curr_shape {
            if curr_shape != &meta.shape {}
        } else {
            curr_shape = Some(meta.shape.clone());
        }
    }

    Ok(zarr_dims)
}

fn create_schema<T>(
    cols: Vec<String>,
    arrays: &HashMap<String, Array<T>>,
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

//********************************************
// the stucture that will be used to stream data from the zarr
// store as it gets read.
//********************************************
pub struct ZarrDataStream<T: AsyncReadableListableStorageTraits> {
    arrays: HashMap<String, Array<T>>,
    schema: Schema,
    zarr_dims: ZarrDimensions,
    filter: Option<ZarrChunkFilter>,
}

impl<T: AsyncReadableListableStorageTraits + 'static> ZarrDataStream<T> {
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
            group
        } else {
            "".to_string()
        };

        let store_keys = store.list_prefix(&StorePrefix::new(&grp)?).await?;
        let mut cols = extract_columns(&grp, store_keys);
        cols = proj.apply_selection(&cols)?;

        let mut arrays: HashMap<String, Array<T>> = HashMap::new();
        for col in &cols {
            let path = PathBuf::from(&grp)
                .join(col)
                .into_os_string()
                .to_str()
                .expect("could not form path from group and column name")
                .to_string();
            let arr = Array::async_open(store.clone(), &path).await?;
            arrays.insert(col.to_string(), arr);
        }

        let schema = create_schema(cols, &arrays)?;
        let zarr_dims = build_dimensions(&arrays)?;
        Ok(Self {
            arrays,
            schema,
            zarr_dims,
            filter: None,
        })
    }

    fn with_filter(mut self, filter: ZarrChunkFilter) -> Self {
        self.filter = Some(filter);
        self
    }
}

#[cfg(test)]
mod zarr_stream_tests {
    use ndarray::{Array, Array2};
    use std::path::PathBuf;
    use std::sync::Arc;
    use zarrs::array::{ArrayBuilder, DataType, FillValue};
    use zarrs::array_subset::ArraySubset;
    use zarrs_filesystem::FilesystemStore;
    use zarrs_storage::{ListableStorageTraits, StorePrefix, WritableStorageTraits};

    use super::extract_columns;

    fn get_store_with_partition(store_name: String) -> Arc<FilesystemStore> {
        // create the store
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(store_name);
        let store = Arc::new(FilesystemStore::new(p).unwrap());

        // var=1, other_var=a
        // latitude
        let array = ArrayBuilder::new(
            vec![11, 11],
            DataType::Float64,
            vec![4, 4].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .build(store.clone(), "/var=1/other_var=a/lat")
        .unwrap();
        array.store_metadata().unwrap();

        let mut v = vec![
            38., 38.1, 38.2, 38.3, 38.4, 38.5, 38.6, 38.7, 38.8, 38.9, 39.,
        ];
        for _ in 0..10 {
            v.extend_from_within(..11);
        }

        let arr: Array2<f64> = Array::from_vec(v).into_shape_with_order((11, 11)).unwrap();
        array
            .store_array_subset_ndarray(ArraySubset::new_with_ranges(&[0..11, 0..11]).start(), arr)
            .unwrap();

        // longitude
        let array = ArrayBuilder::new(
            vec![11, 11],
            DataType::Float64,
            vec![4, 4].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .build(store.clone(), "/var=1/other_var=a/lon")
        .unwrap();
        array.store_metadata().unwrap();

        let mut v = vec![
            -110., -109.9, -109.8, -109.7, -109.6, -109.5, -109.4, -109.3, -109.2, -109.1, -109.,
        ];
        for _ in 0..10 {
            v.extend_from_within(..11);
        }

        let mut arr: Array2<f64> = Array::from_vec(v).into_shape_with_order((11, 11)).unwrap();
        arr.swap_axes(1, 0);
        array
            .store_array_subset_ndarray(ArraySubset::new_with_ranges(&[0..11, 0..11]).start(), arr)
            .unwrap();

        // var=2, other_var=a
        // latitude
        let array = ArrayBuilder::new(
            vec![11, 11],
            DataType::Float64,
            vec![4, 4].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .build(store.clone(), "/var=2/other_var=a/lat")
        .unwrap();
        array.store_metadata().unwrap();

        let mut v = vec![
            39., 39.1, 39.2, 39.3, 39.4, 39.5, 39.6, 39.7, 39.8, 39.9, 40.,
        ];
        for _ in 0..10 {
            v.extend_from_within(..11);
        }

        let arr: Array2<f64> = Array::from_vec(v).into_shape_with_order((11, 11)).unwrap();
        array
            .store_array_subset_ndarray(ArraySubset::new_with_ranges(&[0..11, 0..11]).start(), arr)
            .unwrap();

        // longitude
        let array = ArrayBuilder::new(
            vec![11, 11],
            DataType::Float64,
            vec![4, 4].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .build(store.clone(), "/var=2/other_var=a/lon")
        .unwrap();
        array.store_metadata().unwrap();

        let mut v = vec![
            -110., -109.9, -109.8, -109.7, -109.6, -109.5, -109.4, -109.3, -109.2, -109.1, -109.,
        ];
        for _ in 0..10 {
            v.extend_from_within(..11);
        }

        let mut arr: Array2<f64> = Array::from_vec(v).into_shape_with_order((11, 11)).unwrap();
        arr.swap_axes(1, 0);
        array
            .store_array_subset_ndarray(ArraySubset::new_with_ranges(&[0..11, 0..11]).start(), arr)
            .unwrap();

        // var=1, other_var=b
        // latitude
        let array = ArrayBuilder::new(
            vec![11, 11],
            DataType::Float64,
            vec![4, 4].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .build(store.clone(), "/var=1/other_var=b/lat")
        .unwrap();
        array.store_metadata().unwrap();

        let mut v = vec![
            38., 38.1, 38.2, 38.3, 38.4, 38.5, 38.6, 38.7, 38.8, 38.9, 39.,
        ];
        for _ in 0..10 {
            v.extend_from_within(..11);
        }

        let arr: Array2<f64> = Array::from_vec(v).into_shape_with_order((11, 11)).unwrap();
        array
            .store_array_subset_ndarray(ArraySubset::new_with_ranges(&[0..11, 0..11]).start(), arr)
            .unwrap();

        // longitude
        let array = ArrayBuilder::new(
            vec![11, 11],
            DataType::Float64,
            vec![4, 4].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .build(store.clone(), "/var=1/other_var=b/lon")
        .unwrap();
        array.store_metadata().unwrap();

        let mut v = vec![
            -108.9, -108.8, -108.7, -108.6, -108.5, -108.4, -108.3, -108.2, -108.1, -108.0, -107.9,
        ];
        for _ in 0..10 {
            v.extend_from_within(..11);
        }

        let mut arr: Array2<f64> = Array::from_vec(v).into_shape_with_order((11, 11)).unwrap();
        arr.swap_axes(1, 0);
        array
            .store_array_subset_ndarray(ArraySubset::new_with_ranges(&[0..11, 0..11]).start(), arr)
            .unwrap();

        // var=2, other_var=b
        // latitude
        let array = ArrayBuilder::new(
            vec![11, 11],
            DataType::Float64,
            vec![4, 4].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .build(store.clone(), "/var=2/other_var=b/lat")
        .unwrap();
        array.store_metadata().unwrap();

        let mut v = vec![
            39., 39.1, 39.2, 39.3, 39.4, 39.5, 39.6, 39.7, 39.8, 39.9, 40.,
        ];
        for _ in 0..10 {
            v.extend_from_within(..11);
        }

        let arr: Array2<f64> = Array::from_vec(v).into_shape_with_order((11, 11)).unwrap();
        array
            .store_array_subset_ndarray(ArraySubset::new_with_ranges(&[0..11, 0..11]).start(), arr)
            .unwrap();

        // longitude
        let array = ArrayBuilder::new(
            vec![11, 11],
            DataType::Float64,
            vec![4, 4].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .build(store.clone(), "/var=2/other_var=b/lon")
        .unwrap();
        array.store_metadata().unwrap();

        let mut v = vec![
            -108.9, -108.8, -108.7, -108.6, -108.5, -108.4, -108.3, -108.2, -108.1, -108.0, -107.9,
        ];
        for _ in 0..10 {
            v.extend_from_within(..11);
        }

        let mut arr: Array2<f64> = Array::from_vec(v).into_shape_with_order((11, 11)).unwrap();
        arr.swap_axes(1, 0);
        array
            .store_array_subset_ndarray(ArraySubset::new_with_ranges(&[0..11, 0..11]).start(), arr)
            .unwrap();

        store
    }

    #[test]
    fn store_keys_test() {
        let store = get_store_with_partition("test_store".to_string());

        let store_keys = store
            .list_prefix(&StorePrefix::new("var=1/other_var=a/").unwrap())
            .unwrap();
        for k in &store_keys {
            println!("{:?}", k.as_str());
        }

        println!("{:?}", extract_columns("var=1/other_var=a/", store_keys));

        let prefix = StorePrefix::new("").unwrap();
        store.erase_prefix(&prefix).unwrap();
    }
}
