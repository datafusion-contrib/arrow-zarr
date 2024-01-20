use std::collections::HashMap;
use serde::Deserialize;
use regex::Regex;
use itertools::Itertools;
use crate::reader::{ZarrError, ZarrResult};

// Various enums for the properties of the zarr arrays data.
#[derive(Debug, PartialEq, Clone)]
pub(crate) enum CompressorType {
    Blosc,
    Zlib,
    Bz2,
    Lzma,
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Endianness {
    Little,
    Big,
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum MatrixOrder {
    RowMajor,
    ColumnMajor,
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum ZarrDataType {
    Bool,
    UInt(usize),
    Int(usize),
    Float(usize),
    FixedLengthString(usize),
    FixedLengthPyUnicode(usize),
    TimeStamp(usize, String),
}

#[derive(Debug, PartialEq, Clone)]
enum Filter {
    _Delta,
    _FixedScaleOffser,
}


/// The metadata for a single zarr array, which holds various parameters
/// for the data stored in the array.
#[derive(Debug, PartialEq, Clone)]
pub struct ZarrArrayMetadata {
    zarr_format: u8,
    data_type: ZarrDataType,
    compressor: Option<CompressorType>,
    order: MatrixOrder,
    endianness: Endianness,
    _filter: Option<Filter>,
}

impl ZarrArrayMetadata {
    pub(crate) fn new(
        zarr_format: u8,
        data_type: ZarrDataType, 
        compressor: Option<CompressorType>,
        order: MatrixOrder,
        endianness: Endianness,
    ) -> Self {
        Self { zarr_format, data_type, compressor, order, endianness, _filter: None }
    }

    pub(crate) fn get_type(&self) -> &ZarrDataType {
        &self.data_type
    }

    pub(crate) fn get_compressor(&self) -> &Option<CompressorType> {
        &self.compressor
    }

    pub(crate) fn get_endianness(&self) -> &Endianness {
        &self.endianness
    }

    pub(crate) fn get_order(&self) -> &MatrixOrder {
        &self.order
    }
}

/// The metadata for a zarr store made up of one or more zarr arrays,
/// holding the metadata for all of the arrays and the parameters
/// that have to be consistent across all the arrays. Notably, all the
/// arrays must have the same number of chunks and the chunks must all
/// be of the same size.
#[derive(Debug, PartialEq, Clone)]
pub struct ZarrStoreMetadata {
    columns: Vec<String>,
    chunks: Option<Vec<usize>>,
    shape: Option<Vec<usize>>,
    last_chunk_idx: Option<Vec<usize>>,
    array_params: HashMap<String, ZarrArrayMetadata>
}


#[derive(Deserialize)]
struct RawCompParams {
    id: String
}

#[derive(Deserialize)]
struct RawOuterCompParams {
    compressor: RawCompParams
}


#[derive(Deserialize)]
struct RawArrayParams {
    zarr_format: u8,
    shape: Vec<usize>,
    chunks: Vec<usize>,
    dtype: String,
    order: String,
}


// This is the byte length of the Py Unicode characters that zarr writes
// when the output type is set to U<length>.
pub(crate) const PY_UNICODE_SIZE: usize = 4;

// TODO: this function isn't great, it will work on all valid types (that are supported
// by this library), but handling invalid types could be improved.
fn extract_type(dtype: &str) -> ZarrResult<ZarrDataType> {
    // regex expressions to extract types
    let bool_re = Regex::new(r"([\|><])(b1)").unwrap();
    let uint_re = Regex::new(r"([\|><])(u)([1248])").unwrap();
    let int_re = Regex::new(r"([\|><])(i)([1248])").unwrap();
    let float_re = Regex::new(r"([\|><])(f)([48])").unwrap();
    let ts_re = Regex::new(r"([\|><])(M8)(\[s\]|\[ms\]|\[us\]|\[ns\])").unwrap();
    let str_re = Regex::new(r"([\|><])(S)").unwrap();
    let uni_re = Regex::new(r"([\|><])(U)").unwrap();

    if str_re.is_match(dtype) {
        let str_len = dtype[2..dtype.len()].parse::<usize>().unwrap();
        return Ok(ZarrDataType::FixedLengthString(str_len))
    }

    if uni_re.is_match(dtype) {
        let str_len = dtype[2..dtype.len()].parse::<usize>().unwrap();
        return Ok(ZarrDataType::FixedLengthPyUnicode(PY_UNICODE_SIZE * str_len))
    }

    if let Some(capt) = ts_re.captures(&dtype) {
        return Ok(ZarrDataType::TimeStamp(
            8,
            capt.get(3).unwrap().as_str().strip_prefix("[").unwrap().strip_suffix("]").unwrap().to_string(),
        ))
    }

    // all other types should have a length of 3
    if dtype.len() != 3 {
        return Err(ZarrError::InvalidMetadata("could not match type in zarr metadata".to_string()))
    }

    if let Some(_capt) = bool_re.captures(&dtype) {
        return Ok(ZarrDataType::Bool)
    }

    if let Some(capt) = uint_re.captures(&dtype) {
        return Ok(ZarrDataType::UInt(capt.get(3).unwrap().as_str().parse::<usize>().unwrap()))
    }

    if let Some(capt) = int_re.captures(&dtype) {
        return Ok(ZarrDataType::Int(capt.get(3).unwrap().as_str().parse::<usize>().unwrap()))
    }

    if let Some(capt) = float_re.captures(&dtype) {
        return Ok(ZarrDataType::Float(capt.get(3).unwrap().as_str().parse::<usize>().unwrap()))
    }

    Err(ZarrError::InvalidMetadata("could not match type in zarr metadata".to_string()))
}

impl ZarrStoreMetadata {
    // creates an empty store metadata structure.
    pub(crate) fn new() -> Self {
        Self {
            columns: Vec::new(),
            chunks: None,
            shape: None,
            last_chunk_idx: None,
            array_params: HashMap::new(),
        }
    }

    // adds the metadata for one column (variable) to the store metadata.
    pub(crate) fn add_column(&mut self, col_name: String, metadata_str: &str) -> ZarrResult<()> {
        // extract compressor type
        let j: Result<RawOuterCompParams, serde_json::Error> =
            serde_json::from_str(&metadata_str);
        let compressor = if let Ok(comp) = j {
            match comp.compressor.id.as_str() {
                "blosc" => Some(CompressorType::Blosc),
                "zlib" => Some(CompressorType::Zlib),
                "lzma" => Some(CompressorType::Lzma),
                "bz2" => Some(CompressorType::Bz2),
                _ => return Err(
                    ZarrError::InvalidMetadata(
                        "Invalid compressor params in zarr metadata".to_string()
                    )
                )
            }
        } else {
            let j: serde_json::Value = serde_json::from_str(&metadata_str).unwrap();
            if j["compressor"].is_null() {
                None
            } else {
                return Err(
                    ZarrError::InvalidMetadata(
                        "Invalid compressor params in zarr metadata".to_string()
                    )
                )
            }
        };

        let j: Result<RawArrayParams, serde_json::Error> =
            serde_json::from_str(&metadata_str);
        if let Ok(raw_params) = j {
            // verify that all arrays have the same chunks
            if let Some(chnks) = &self.chunks {
                if chnks != &raw_params.chunks {
                    return Err(
                        ZarrError::InvalidMetadata(
                            "All arrays in the zarr store must have the same chunks".to_string()
                        )
                    )
                }
            }
            if raw_params.chunks.len() > 3 {
                return Err(ZarrError::InvalidMetadata("Chunk dimensionality must not exceed 3".to_string()))
            }

            // verify that all arrays have the same shape
            if let Some(shp) = &self.shape {
                if shp != &raw_params.shape {
                    return Err(
                        ZarrError::InvalidMetadata(
                            "All arrays in the zarr store must have the same shape".to_string()
                        )
                    )
                }
            }
            if raw_params.shape.len() > 3 {
                return Err(ZarrError::InvalidMetadata("Shape dimensionality must not exceed 3".to_string()))
            }


            // extract matrix order, endianness and data type
            let order = match raw_params.order.as_str() {
                "C" => MatrixOrder::RowMajor,
                "F" => MatrixOrder::ColumnMajor,
                _ => return Err(
                    ZarrError::InvalidMetadata(
                        "Invalid matrix order in zarr metadata".to_string()
                    )
                )
            };

            let endianness = match raw_params.dtype.chars().next().unwrap() {
                '<' | '|' => Endianness::Little,
                '>' => Endianness::Big,
                _ => return Err(
                    ZarrError::InvalidMetadata(
                        "Cannot extract endiannes from dtype in zarr metadata".to_string()
                    )
                ),
            };

            let data_type = extract_type(&raw_params.dtype)?;

            // if everything is valid, update chunks and shape (if not set yet) and create
            // the params for the zarr array.
            if self.last_chunk_idx.is_none() {
                self.last_chunk_idx = Some(raw_params.chunks
                    .iter()
                    .zip(&raw_params.shape)
                    .map(|(&chnk, &shp)| (shp as f64 / chnk as f64).ceil() as usize- 1)
                    .collect());
            }
            if self.chunks.is_none() {
                self.chunks = Some(raw_params.chunks);
            }
            if self.shape.is_none() {
                self.shape = Some(raw_params.shape);
            }

            self.columns.push(col_name.to_string());
            self.array_params.insert(
                col_name, 
                ZarrArrayMetadata::new ( 
                    raw_params.zarr_format,
                    data_type,
                    compressor,
                    order,
                    endianness,
                )
            );

            Ok(())
        }
        else {
            Err(ZarrError::InvalidMetadata("Cannot match zarr metadata".to_string()))
        }
    }

    pub(crate) fn get_num_columns(&self) -> usize {
        self.columns.len()
    }

    pub(crate) fn get_columns(&self) -> &Vec<String> {
        &self.columns
    }

    pub(crate) fn get_array_meta(&self, column: &str) -> ZarrResult<&ZarrArrayMetadata> {
        Ok(
            self.array_params
            .get(column)
            .ok_or(ZarrError::InvalidMetadata(format!("Cannot find variable {} in metadata", column)))?
        )
    }

    pub(crate) fn get_chunk_dims(&self) -> &Vec<usize> {
        self.chunks.as_ref().unwrap()
    }

    // get the indices of all the chunks, as a 1D, 2D or 3D vector, for each chunk.
    pub(crate) fn get_chunk_positions(&self) -> Vec<Vec<usize>> {
        let shape = self.shape.as_ref().unwrap();
        let chunks = self.chunks.as_ref().unwrap();

        let n_chunks: Vec<usize> = shape
            .iter()
            .zip(chunks)
            .map(|(&shp, &chnk)| (shp as f64 / chnk as f64).ceil() as usize)
            .collect();

        let grid_positions = match n_chunks.len() {
            1 => {(0..n_chunks[0]).map(|x| vec![x; 1]).collect()}
            2 => {
                (0..n_chunks[0])
                    .cartesian_product(0..n_chunks[1])
                    .map(|(a, b)| vec![a, b])
                    .collect()
            } 
            3 => {
                (0..n_chunks[0])
                    .cartesian_product(0..n_chunks[1])
                    .cartesian_product(0..n_chunks[2])
                    .map(|((a, b), c)| vec![a, b, c])
                    .collect()
            }
            _ => {panic!("Array has more than 3 domensions, 3 is the limit")}
        };

        grid_positions
    }

    // return the real dimensions of a chhunk, given its position, taking into
    // account that it can be at the "edge" of the array for one or more dimension.
    pub(crate) fn get_real_dims(&self, pos: &Vec<usize>) -> Vec<usize> {
        pos.iter()
            .zip(self.last_chunk_idx.as_ref().unwrap())
            .zip(self.chunks.as_ref().unwrap())
            .zip(self.shape.as_ref().unwrap())
            .map(
                |(((p, last), chnk), shp)| {
                    if p == last {
                        shp - last * chnk
                    } else {
                        *chnk
                    }
                }
            )
            .collect()
    }
}


#[cfg(test)]
mod zarr_metadata_tests {
    use super::*;

    // test various valid metadata strings.
    #[test]
    fn test_valid_metadata() {
        let mut meta = ZarrStoreMetadata::new();

        let metadata_str = r#"
        {
            "zarr_format": 2,
            "chunks": [10, 10],
            "shape": [100, 100],
            "dtype": "<i4",
            "order": "C",
            "compressor": {"id": "zlib"}
        }"#;
        meta.add_column("var1".to_string(), &metadata_str).unwrap();

        let metadata_str = r#"
        {
            "zarr_format": 2,
            "chunks": [10, 10],
            "shape": [100, 100],
            "dtype": "<M8[ms]",
            "order": "C",
            "compressor": {"id": "blosc"}
        }"#;
        meta.add_column("var2".to_string(), &metadata_str).unwrap();

        let metadata_str = r#"
        {
            "zarr_format": 2,
            "chunks": [10, 10],
            "shape": [100, 100],
            "dtype": "|b1",
            "order": "F",
            "compressor": null
        }"#;
        meta.add_column("var3".to_string(), &metadata_str).unwrap();

        let metadata_str = r#"
        {
            "zarr_format": 2,
            "chunks": [10, 10],
            "shape": [100, 100],
            "dtype": ">S112",
            "order": "C",
            "compressor": {"id": "lzma"}
        }"#;
        meta.add_column("var4".to_string(), &metadata_str).unwrap();

        assert_eq!(meta.chunks, Some(vec![10, 10]));
        assert_eq!(meta.shape, Some(vec![100, 100]));
        assert_eq!(meta.last_chunk_idx, Some(vec![9, 9]));
        assert_eq!(meta.columns, vec!["var1", "var2", "var3", "var4"]);
        assert_eq!(
            meta.array_params["var1"],
            ZarrArrayMetadata {
                zarr_format : 2,
                data_type: ZarrDataType::Int(4),
                compressor: Some(CompressorType::Zlib),
                order: MatrixOrder::RowMajor,
                endianness: Endianness::Little,
                _filter: None,
            }
        );
        assert_eq!(
            meta.array_params["var2"],
            ZarrArrayMetadata {
                zarr_format : 2,
                data_type: ZarrDataType::TimeStamp(8, "ms".to_string()),
                compressor: Some(CompressorType::Blosc),
                order: MatrixOrder::RowMajor,
                endianness: Endianness::Little,
                _filter: None,
            }
        );
        assert_eq!(
            meta.array_params["var3"],
            ZarrArrayMetadata {
                zarr_format : 2,
                data_type: ZarrDataType::Bool,
                compressor: None,
                order: MatrixOrder::ColumnMajor,
                endianness: Endianness::Little,
                _filter: None,
            }
        );
        assert_eq!(
            meta.array_params["var4"],
            ZarrArrayMetadata {
                zarr_format : 2,
                data_type: ZarrDataType::FixedLengthString(112),
                compressor: Some(CompressorType::Lzma),
                order: MatrixOrder::RowMajor,
                endianness: Endianness::Big,
                _filter: None,
            }
        );

        // check for an array with chunks that don't perfectly line up with
        // the shape.
        let mut meta = ZarrStoreMetadata::new();
        let metadata_str = r#"
        {
            "zarr_format": 2,
            "chunks": [12, 12],
            "shape": [100, 100],
            "dtype": "<f8",
            "order": "C",
            "compressor": {"id": "blosc"}
        }"#;
        meta.add_column("var".to_string(), &metadata_str).unwrap();
        assert_eq!(meta.last_chunk_idx, Some(vec![8, 8]))

    }

    // test various metadata strings that are invalid and should results in an
    // error being returned.
    #[test]
    fn test_invalid_metadata() {
        let mut meta = ZarrStoreMetadata::new();

        // invalid dtype
        let metadata_str = r#"
        {
            "zarr_format": 2,
            "chunks": [10, 10],
            "shape": [100, 100],
            "dtype": "<i44",
            "order": "C",
            "compressor": {"id": "zlib"}
        }"#;
        assert!(meta.add_column("var".to_string(), &metadata_str).is_err());

        // invalid compressor
        let metadata_str = r#"
        {
            "zarr_format": 2,
            "chunks": [10, 10],
            "shape": [100, 100],
            "dtype": "<f8",
            "order": "C",
            "compressor": {"id": "invalid_compressor"}
        }"#;
        assert!(meta.add_column("var".to_string(), &metadata_str).is_err());

        // mismatch between chunks
        // first let's create one valid array metadata
        let metadata_str = r#"
        {
            "zarr_format": 2,
            "chunks": [10, 10],
            "shape": [100, 100],
            "dtype": "<f8",
            "order": "C",
            "compressor": {"id": "blosc"}
        }"#;
        meta.add_column("var1".to_string(), &metadata_str).unwrap();

        let metadata_str = r#"
        {
            "zarr_format": 2,
            "chunks": [20, 20],
            "shape": [100, 100],
            "dtype": "<f8",
            "order": "C",
            "compressor": {"id": "blosc"}
        }"#;
        assert!(meta.add_column("var2".to_string(), &metadata_str).is_err());

        // mismatch between shapes
        let metadata_str = r#"
        {
            "zarr_format": 2,
            "chunks": [10, 10],
            "shape": [200, 200],
            "dtype": "<f8",
            "order": "C",
            "compressor": {"id": "blosc"}
        }"#;
        assert!(meta.add_column("var2".to_string(), &metadata_str).is_err());


    }
}