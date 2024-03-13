use std::collections::HashMap;
use serde_json::{Value, json};
use regex::Regex;
use std::str::FromStr;
use itertools::Itertools;
use crate::reader::{ZarrError, ZarrResult};
use crate::reader::codecs::{
    CodecType,
    ZarrCodec,
    BloscOptions,
    ShardingOptions,
    ShuffleOptions,
    CompressorName,
    Endianness,
    IndexLocation,
    ZarrDataType,
    PY_UNICODE_SIZE,
};

use super::errors::throw_invalid_meta;

// TODO: this function isn't great, it will work on all valid types (that are supported
// by this library), but handling invalid types could be improved.
fn extract_type_v2(dtype: &str) -> ZarrResult<ZarrDataType> {
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

// TODO: need to check for other types, not quite clear from the docs yet.
// this function also isn't great, it will work on all valid types (that are supported
// by this library), but handling invalid types could be improved.
fn extract_type_v3(dtype: &str) -> ZarrResult<ZarrDataType> {
    let bits_per_byte = 8;
    
    if dtype == "bool" {
        return Ok(ZarrDataType::Bool)
    }
    else if dtype.starts_with("float") {
        let size = dtype[5..dtype.len()].parse::<usize>().unwrap() / bits_per_byte;
        return Ok(ZarrDataType::Float(size))
    }
    else if dtype.starts_with("int") {
        let size = dtype[3..dtype.len()].parse::<usize>().unwrap() / bits_per_byte;
        return Ok(ZarrDataType::Int(size))
    }
    else if dtype.starts_with("uint") {
        let size = dtype[4..dtype.len()].parse::<usize>().unwrap() / bits_per_byte;
        return Ok(ZarrDataType::UInt(size))
    }

    Err(ZarrError::InvalidMetadata("could not match type in zarr metadata".to_string()))
}

// Enum for the chunk separator
#[derive(Debug, PartialEq, Clone)]
pub enum ChunkSeparator {
    Slash,
    Period
}

impl FromStr for ChunkSeparator {
    type Err = ZarrError;

    fn from_str(input: &str) -> ZarrResult<Self> {
        match input {
            "/" => Ok(ChunkSeparator::Slash),
            "." => Ok(ChunkSeparator::Period),
            _ => Err(ZarrError::InvalidMetadata("Invalid chunk separator".to_string()))
        }
    }
}

/// The metadata for a single zarr array, which holds various parameters
/// for the data stored in the array.
#[derive(Debug, PartialEq, Clone)]
pub struct ZarrArrayMetadata {
    zarr_format: u8,
    data_type: ZarrDataType,
    chunk_separator: ChunkSeparator,
    sharding_options: Option<ShardingOptions>,
    codecs: Vec<ZarrCodec>,
}

impl ZarrArrayMetadata {
    pub(crate) fn get_type(&self) -> &ZarrDataType {
        &self.data_type
    }

    pub(crate) fn new(
        zarr_format: u8,
        data_type: ZarrDataType,
        chunk_separator: ChunkSeparator,
        sharding_options: Option<ShardingOptions>,
        codecs: Vec<ZarrCodec>,
    ) -> Self {
        Self {
            zarr_format,
            data_type,
            chunk_separator,
            sharding_options,
            codecs
        }
    }

    pub(crate) fn get_codecs(&self) -> &Vec<ZarrCodec> {
        &self.codecs
    }

    pub(crate) fn get_sharding_params(&self) -> Option<ShardingOptions> {
        self.sharding_options.clone()
    }

    pub(crate) fn get_separator(&self) -> ChunkSeparator {
        self.chunk_separator.clone()
    }
}

/// The metadata for a zarr store made up of one or more zarr arrays,
/// holding the metadata for all of the arrays and the parameters
/// that have to be consistent across all the arrays. Notably, all the
/// arrays must have the same number of chunks and the chunks must all
/// be of the same size. For zarr v3, either all or none of the arrays can
/// be sharded (with the same sharding params), can't mix some shareded arrays
/// with arrays that are not sharded.
#[derive(Debug, PartialEq, Clone)]
pub struct ZarrStoreMetadata {
    columns: Vec<String>,
    chunks: Option<Vec<usize>>,
    shape: Option<Vec<usize>>,
    last_chunk_idx: Option<Vec<usize>>,
    array_params: HashMap<String, ZarrArrayMetadata>
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
}

// several helper functions to extract data from the matadata json object.
fn extract_string_from_json(map: &Value, key: &str, err_str: &str) -> ZarrResult<String> {
    let res = map.get(key)
                 .ok_or(ZarrError::InvalidMetadata(err_str.to_string()))?
                 .as_str()
                 .ok_or(ZarrError::InvalidMetadata(err_str.to_string()))?;
    Ok(res.to_string())
}

fn extract_u64_from_json(map: &Value, key: &str, err_str: &str) -> ZarrResult<u64> {
    let res = map.get(key)
                 .ok_or(ZarrError::InvalidMetadata(err_str.to_string()))?
                 .as_u64()
                 .ok_or(ZarrError::InvalidMetadata(err_str.to_string()))?;
    Ok(res)
}

fn extract_usize_array_from_json(map: &Value, key: &str, err_str: &str) -> ZarrResult<Vec<usize>> {
    let arr: Vec<usize> = map.get(key)
                             .ok_or(ZarrError::InvalidMetadata(err_str.to_string()))?
                             .as_array()
                             .ok_or(ZarrError::InvalidMetadata(err_str.to_string()))?
                             .iter()
                             .map(|v| v.as_u64().unwrap() as usize)
                             .collect();
    Ok(arr)
}

fn extract_arr_and_check(
    map: &Value, key: &str, err_str: &str, curr: &Option<Vec<usize>>) -> ZarrResult<Vec<usize>> {
    let arr = extract_usize_array_from_json(map, key, err_str)?;

    if let Some(curr) = curr {
        if curr != &arr {
            return Err(ZarrError::InvalidMetadata(err_str.to_string()))
        }
    }
    if arr.len() > 3 {
        return Err(ZarrError::InvalidMetadata(err_str.to_string()))
    }

    Ok(arr)
}

fn extract_compressor_params_v2(params_map: &serde_json::Value, dtype: &ZarrDataType) -> ZarrResult<BloscOptions> {
    let error_string = "error parsing metadata compressor";
    let id = params_map.get("id").ok_or(ZarrError::InvalidMetadata(error_string.to_string()))?;

    if id != "blosc" {
        return Err(ZarrError::InvalidMetadata("only blosc compressor is supported for v2".to_string()))
    }

    let cname = extract_string_from_json(params_map, "cname", error_string)?;
    let clevel = extract_u64_from_json(params_map, "clevel", error_string)?;
    let shuffle = extract_u64_from_json(params_map, "shuffle", error_string)?;

    let shuffle = match shuffle {
        0 => Ok(ShuffleOptions::Noshuffle),
        1 => Ok(ShuffleOptions::ByteShuffle(dtype.get_byte_size())),
        2 => Ok(ShuffleOptions::BitShuffle(dtype.get_byte_size())),
        _ => Err(throw_invalid_meta("invalid compressor shuffle option"))
    }?;

    let comp = BloscOptions::new(
        CompressorName::from_str(&cname)?,
        clevel as u8,
        shuffle,
        0
    );

    Ok(comp)
}

// Method to populate zarr metadata from zarr arrays metadata, using
// version 2 of the zarr format.
impl ZarrStoreMetadata {
    fn add_column_v2(&mut self, col_name: String, meta_map: Value) -> ZarrResult<()> {
        // parse chunks
        let error_string = "error parsing metadata chunks";
        let chunks = extract_arr_and_check(&meta_map, "chunks", error_string, &self.chunks)?;
        if self.chunks.is_none() {
            self.chunks = Some(chunks.clone());
        }

        // parse shape
        let error_string = "error parsing metadata shape";
        let shape = extract_arr_and_check(&meta_map, "shape", error_string, &self.shape)?;
        
        if chunks.len() != shape.len() {
            return Err(ZarrError::InvalidMetadata("chunk and shape dimensions must match".to_string()))
        }

        if self.shape.is_none() {
            self.shape = Some(shape.clone());
        } 

        // the index of the last chunk in each dimension
        if self.last_chunk_idx.is_none() {
            self.last_chunk_idx = Some(
                chunks
                .iter()
                .zip(&shape)
                .map(|(&chnk, &shp)| (shp as f64 / chnk as f64).ceil() as usize- 1)
                .collect()
            );
        }

        // parse data type
        let error_string = "error parsing metadata data type";
        let dtype = extract_string_from_json(&meta_map, "dtype", error_string)?;
        let data_type = extract_type_v2(&dtype)?;

        // parse endianness
        let endianness = match dtype.chars().next().unwrap() {
            '<' | '|' => Endianness::Little,
            '>' => Endianness::Big,
            _ => return Err(
                ZarrError::InvalidMetadata(
                    "error parsing endianness from metadata datatype".to_string()
                )
            ),
        };

        // parser order
        let error_string = "error parsing metadata order";
        let dim = self.shape.as_ref().unwrap().len();
        let order = extract_string_from_json(&meta_map, "order", error_string)?;
        let order = order.chars().next().unwrap();

        // compressor params
        let error_string = "error parsing metadata compressor";
        let mut comp: Option<BloscOptions> = None;
        let comp_params = meta_map.get("compressor")
                                  .ok_or(ZarrError::InvalidMetadata(error_string.to_string()))?;
        if !comp_params.is_null() {
            comp = Some(extract_compressor_params_v2(comp_params, &data_type)?);
        }

        // build the array params and push then into the store metadata
        let mut codecs = Vec::new();
        if order == 'F' {
            if dim == 2 {
                codecs.push(ZarrCodec::Transpose(vec![1, 0]));
            }
            else if dim == 3 {
                codecs.push(ZarrCodec::Transpose(vec![2, 1, 0]));
            }
        }
        codecs.push(ZarrCodec::Bytes(endianness));

        if let Some(comp) = comp {
            codecs.push(ZarrCodec::BloscCompressor(comp));
        }

        // finally create the zarr array metadata and insert it in params
        let array_meta = ZarrArrayMetadata{
            zarr_format: 2,
            data_type: data_type,
            chunk_separator: ChunkSeparator::Period,
            sharding_options: None,
            codecs: codecs,
        };

        self.columns.push(col_name.to_string());
        self.array_params.insert(col_name, array_meta);

        Ok(())
    }
}

// some more helpers for zarr format v3
fn extract_config (map: &Value) -> ZarrResult<(String, &Value)> {
    let name = extract_string_from_json(&map, "name", "can't retrieve name of configuration")?;
    let config = map.get("configuration");
    if config.is_none() {
        return Ok((name, &json!(null)))
    }

    Ok((name, config.unwrap()))
}

fn extract_codec(config: &Value, last_type: &CodecType) -> ZarrResult<ZarrCodec> {
    let error_string = "error parsing codec from metadata";
    let (name, config) = extract_config(config)?;

    let (codec_type, codec) = match name.as_str() {
        "blosc" => {
            let cname = extract_string_from_json(config, "cname", error_string)?;
            let shuffle = extract_string_from_json(config, "shuffle", error_string)?;
            let clevel = extract_u64_from_json(config, "clevel", error_string)? as u8;
            let blocksize = extract_u64_from_json(config, "blocksize", error_string)? as usize;
            let typesize = extract_u64_from_json(config, "typesize", error_string)? as usize;
            let c = BloscOptions::new(
                CompressorName::from_str(&cname)?,
                clevel,
                ShuffleOptions::new(&shuffle, typesize)?,
                blocksize,
            );
            Ok((CodecType::BytesToBytes, ZarrCodec::BloscCompressor(c)))
        },
        "bytes" => {
            let e = extract_string_from_json(config, "endian", error_string)?;
            Ok((CodecType::ArrayToBytes, ZarrCodec::Bytes(Endianness::from_str(&e)?)))
        },
        "crc32c" => {
            Ok((CodecType::BytesToBytes, ZarrCodec::Crc32c))
        },
        "gzip" => {
            let l = extract_u64_from_json(config, "level", error_string)? as u8;
            Ok((CodecType::BytesToBytes, ZarrCodec::Gzip(l)))
        },
        "transpose" => {
            let o = extract_usize_array_from_json(config, "order", error_string)?;
            Ok((CodecType::ArrayToArray, ZarrCodec::Transpose(o)))
        }
        _ => Err(ZarrError::InvalidMetadata(error_string.to_string()))
    }?;

    // verify that codecs are being read in the right order
    if last_type == &CodecType::ArrayToBytes && codec_type == CodecType::ArrayToBytes {
        return Err(ZarrError::InvalidMetadata(error_string.to_string()))
    }
    if last_type != &codec_type {
        if (last_type == &CodecType::ArrayToArray && codec_type != CodecType::ArrayToBytes) ||
            (last_type == &CodecType::ArrayToBytes && codec_type != CodecType::BytesToBytes)
        {
            return Err(ZarrError::InvalidMetadata(error_string.to_string()))
        }
    }

    Ok(codec)
}

fn extract_sharding_options(outer_chunk_shape: &Vec<usize>, config: &Value, pos: usize) -> ZarrResult<ShardingOptions> {
    let error_string = "error parsing sharding params from metadata";
    let chunk_shape = extract_usize_array_from_json(config, "chunk_shape", error_string)?;

    for (outer_n, inner_n) in outer_chunk_shape.iter().zip(chunk_shape.iter()) {
        if outer_n % inner_n != 0 {
            return Err(throw_invalid_meta(error_string))
        }
    }

    let codec_configs = config.get("codecs")
                              .ok_or(ZarrError::InvalidMetadata(error_string.to_string()))?
                              .as_array()
                              .ok_or(ZarrError::InvalidMetadata(error_string.to_string()))?;
    let mut codecs = Vec::new();
    let mut last_type = CodecType::ArrayToArray;
    let mut n_array_to_bytes = 0;
    for c in codec_configs {
        let c = extract_codec(c, &last_type)?;
        last_type = c.codec_type();
        n_array_to_bytes += (last_type == CodecType::ArrayToBytes) as u32;
        codecs.push(c);
    };

    if n_array_to_bytes != 1 {
        return Err(ZarrError::InvalidMetadata(error_string.to_string()))
    }

    let index_codec_configs = config.get("index_codecs")
                                    .ok_or(ZarrError::InvalidMetadata(error_string.to_string()))?
                                    .as_array()
                                    .ok_or(ZarrError::InvalidMetadata(error_string.to_string()))?;
    let mut index_codecs = Vec::new();
    last_type = CodecType::ArrayToArray;
    n_array_to_bytes = 0;
    for c in index_codec_configs {
        let c = extract_codec(c, &last_type)?;
        last_type = c.codec_type();
        n_array_to_bytes += (last_type == CodecType::ArrayToBytes) as u32;
        index_codecs.push(c);
    };

    if n_array_to_bytes != 1 {
        return Err(ZarrError::InvalidMetadata(error_string.to_string()))
    }

    let n_chunks = outer_chunk_shape.iter().zip(chunk_shape.iter()).map(|(o, i)| o / i).collect();

    let loc = extract_string_from_json(config, "index_location", error_string)?;
    let loc = IndexLocation::from_str(&loc)?;
    Ok(ShardingOptions::new(chunk_shape, n_chunks, codecs, index_codecs, loc, pos))
}

// Method to populate zarr metadata from zarr arrays metadata, using
// version 3 of the zarr format.
impl ZarrStoreMetadata {
    fn add_column_v3(&mut self, col_name: String, meta_map: Value) -> ZarrResult<()> {
        // verify the metadata is for an array.
        let error_string = "error parsing node type from metadata";
        let node_type = extract_string_from_json(&meta_map, "node_type", error_string)?;
        if node_type != "array" {
            return Err(ZarrError::InvalidMetadata("node type in metadata must be array".to_string()))
        }

        // parse shape
        let error_string = "error parsing metadata shape";
        let shape = extract_arr_and_check(&meta_map, "shape", error_string, &self.shape)?;
        if self.shape.is_none() {
            self.shape = Some(shape.clone());
        }

        // parse chunks
        let error_string = "error parsing metadata chunks";
        let chunk_grid = meta_map.get("chunk_grid").ok_or(
            ZarrError::InvalidMetadata("can't extract chunk_grid from metadata".to_string())
        )?;
        let (name, config) = extract_config(&chunk_grid)?;
        if name != "regular" {
            return Err(ZarrError::InvalidMetadata("only regular chunks are supported".to_string()));
        }
        let chunks = extract_arr_and_check(config, "chunk_shape", error_string, &self.chunks)?;
        if self.chunks.is_none() {
            self.chunks = Some(chunks.clone());
        }

        if chunks.len() != shape.len() {
            return Err(ZarrError::InvalidMetadata("chunk and shape dimensions must match".to_string()))
        }

        // the index of the last chunk in each dimension
        if self.last_chunk_idx.is_none() {
            self.last_chunk_idx = Some(
                chunks
                .iter()
                .zip(&shape)
                .map(|(&chnk, &shp)| (shp as f64 / chnk as f64).ceil() as usize- 1)
                .collect()
            );
        }

        // data type
        let error_string = "error parsing metadata data type";
        let dtype = extract_string_from_json(&meta_map, "data_type", error_string)?;
        let data_type = extract_type_v3(&dtype)?;

        // chunk separator
        let error_string = "error parsing metadata chunk key encoding";
        let chunk_key_encoding = meta_map.get("chunk_key_encoding").ok_or(
            ZarrError::InvalidMetadata("can't extract chunk_key_encoding from metadata".to_string())
        )?;
        let (_, config) = extract_config(&chunk_key_encoding)?;
        let chunk_key_encoding = extract_string_from_json(config, "separator", error_string)?;
        let chunk_key_encoding = ChunkSeparator::from_str(&chunk_key_encoding)?;

        // codecs
        let codec_configs = meta_map.get("codecs")
                                    .ok_or(ZarrError::InvalidMetadata(error_string.to_string()))?
                                    .as_array()
                                    .ok_or(ZarrError::InvalidMetadata(error_string.to_string()))?;
        let mut codecs = Vec::new();
        let mut sharding_options: Option<ShardingOptions> = None;
        let pos = 0;
        let mut last_type = CodecType::ArrayToArray;
        let error_string = "error parsing codecs";
        let mut n_array_to_bytes = 0;
        for c in codec_configs{
            let (name, config) = extract_config(c)?;
            if name == "sharding_indexed" {
                if last_type != CodecType::ArrayToArray {
                    return Err(ZarrError::InvalidMetadata(error_string.to_string()))
                }
                let s = extract_sharding_options(&chunks, config, pos)?;
                last_type = CodecType::ArrayToBytes;
                sharding_options = Some(s);
                n_array_to_bytes += 1;
                continue;
            }

            let c = extract_codec(c, &last_type)?;
            last_type = c.codec_type();
            n_array_to_bytes += (last_type == CodecType::ArrayToBytes) as u32;
            codecs.push(c);
        }

        if n_array_to_bytes != 1 {
            return Err(ZarrError::InvalidMetadata(error_string.to_string()))
        }

        // finally create the zarr array metadata and insert it in params
        let array_meta = ZarrArrayMetadata::new(
            3,
            data_type,
            chunk_key_encoding,
            sharding_options,
            codecs,
        );

        self.columns.push(col_name.to_string());
        self.array_params.insert(col_name, array_meta);

        Ok(())
    }
}

// Method to populate zarr metadata from zarr arrays metadata, works with either
// zarr_format version 2 or 3.
impl ZarrStoreMetadata {
    pub(crate) fn add_column(&mut self, col_name: String, metadata_str: &str) -> ZarrResult<()> {
        let meta_map: Value = serde_json::from_str(metadata_str).or(
            Err(ZarrError::InvalidMetadata("could not parse metadata string".to_string()))
        )?;
        let err_str = "could not extract valid zarr format from metadata";
        let version = extract_u64_from_json(&meta_map, "zarr_format", err_str)?;

        match version {
            2 => {self.add_column_v2(col_name, meta_map)?;},
            3 => {self.add_column_v3(col_name, meta_map)?;},
            _ => {return Err(throw_invalid_meta("invalid zarr format"));}
        }
    
        return Ok(())
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

    pub(crate) fn get_chunk_dims(&self) -> &Vec<usize> {
        self.chunks.as_ref().unwrap()
    }

    pub(crate) fn get_separators(&self) -> HashMap<String, ChunkSeparator> {
        let mut m = HashMap::new();
        for col in &self.columns {
            m.insert(col.to_string(), self.get_array_meta(&col).unwrap().get_separator());
        }

        return m;
    }
}

#[cfg(test)]
mod zarr_metadata_v3_tests {
    use super::*;

    // test various valid metadata strings.
    #[test]
    fn test_valid_metadata_v2() {
        let mut meta = ZarrStoreMetadata::new();

        let metadata_str = r#"
        {
            "zarr_format": 2,
            "chunks": [10, 10],
            "shape": [100, 100],
            "dtype": "<i4",
            "order": "C",
            "compressor": {
                "id": "blosc",
                "cname": "lz4",
                "clevel": 1,
                "shuffle": 1
            }
        }"#;
        meta.add_column("var1".to_string(), &metadata_str).unwrap();

        let metadata_str = r#"
        {
            "zarr_format": 2,
            "chunks": [10, 10],
            "shape": [100, 100],
            "dtype": "<M8[ms]",
            "order": "C",
            "compressor": {
                "id": "blosc",
                "cname": "zlib",
                "clevel": 1,
                "shuffle": 2
            }
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
            "compressor": {
                "id": "blosc",
                "cname": "lz4",
                "clevel": 1,
                "shuffle": 1
            }
        }"#;
        meta.add_column("var4".to_string(), &metadata_str).unwrap();

        assert_eq!(meta.chunks, Some(vec![10, 10]));
        assert_eq!(meta.shape, Some(vec![100, 100]));
        assert_eq!(meta.last_chunk_idx, Some(vec![9, 9]));
        assert_eq!(meta.columns, vec!["var1", "var2", "var3", "var4"]);

        assert_eq!(
            meta.array_params["var1"],
            ZarrArrayMetadata {
                zarr_format: 2,
                data_type: ZarrDataType::Int(4),
                chunk_separator: ChunkSeparator::Period,
                sharding_options: None,
                codecs: vec![
                    ZarrCodec::Bytes(Endianness::Little),
                    ZarrCodec::BloscCompressor(
                        BloscOptions::new(
                            CompressorName::Lz4,
                            1,
                            ShuffleOptions::ByteShuffle(4),
                            0,
                        )
                    )
                ]
            }
        );

        assert_eq!(
            meta.array_params["var2"],
            ZarrArrayMetadata {
                zarr_format: 2,
                data_type: ZarrDataType::TimeStamp(8, "ms".to_string()),
                chunk_separator: ChunkSeparator::Period,
                sharding_options: None,
                codecs: vec![
                    ZarrCodec::Bytes(Endianness::Little),
                    ZarrCodec::BloscCompressor(
                        BloscOptions::new(
                            CompressorName::Zlib,
                            1,
                            ShuffleOptions::BitShuffle(8),
                            0,
                        )
                    )
                ]
            }
        );

        assert_eq!(
            meta.array_params["var3"],
            ZarrArrayMetadata {
                zarr_format: 2,
                data_type: ZarrDataType::Bool,
                chunk_separator: ChunkSeparator::Period,
                sharding_options: None,
                codecs: vec![
                    ZarrCodec::Transpose(vec![1, 0]),
                    ZarrCodec::Bytes(Endianness::Little),
                ],
            }
        );

        assert_eq!(
            meta.array_params["var4"],
            ZarrArrayMetadata {
                zarr_format: 2,
                data_type: ZarrDataType::FixedLengthString(112),
                chunk_separator: ChunkSeparator::Period,
                sharding_options: None,
                codecs: vec![
                    ZarrCodec::Bytes(Endianness::Big),
                    ZarrCodec::BloscCompressor(
                        BloscOptions::new(
                            CompressorName::Lz4,
                            1,
                            ShuffleOptions::ByteShuffle(112),
                            0,
                        )
                    )
                ]
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
            "compressor": {
                "id": "blosc",
                "cname": "lz4",
                "clevel": 1,
                "shuffle": 1
            }
        }"#;
        meta.add_column("var".to_string(), &metadata_str).unwrap();
        assert_eq!(meta.last_chunk_idx, Some(vec![8, 8]))
    }

    // test various metadata strings that are invalid and should results in an
    // error being returned.
    #[test]
    fn test_invalid_metadata_v2() {
        let mut meta = ZarrStoreMetadata::new();

        // invalid dtype
        let metadata_str = r#"
        {
            "zarr_format": 2,
            "chunks": [10, 10],
            "shape": [100, 100],
            "dtype": "<i44",
            "order": "C",
            "compressor": {
                "id": "blosc",
                "cname": "zlib",
                "clevel": 1,
                "shuffle": 2
            }
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
            "compressor": {
                "id": "some invalid compressor",
                "cname": "zlib",
                "clevel": 1,
                "shuffle": 2
            }
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
            "compressor": {
                "id": "blosc",
                "cname": "lz4",
                "clevel": 1,
                "shuffle": 1
            }
        }"#;
        meta.add_column("var1".to_string(), &metadata_str).unwrap();

        let metadata_str = r#"
        {
            "zarr_format": 2,
            "chunks": [20, 20],
            "shape": [100, 100],
            "dtype": "<f8",
            "order": "C",
            "compressor": {
                "id": "blosc",
                "cname": "lz4",
                "clevel": 1,
                "shuffle": 1
            }
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
            "compressor": {
                "id": "blosc",
                "cname": "lz4",
                "clevel": 1,
                "shuffle": 1
            }
        }"#;
        assert!(meta.add_column("var2".to_string(), &metadata_str).is_err());
    }

    #[test]
    fn test_valid_metadata_v3() {
        let mut meta = ZarrStoreMetadata::new();
        let metadata_str = r#"
        {
            "shape": [16, 16],
            "data_type": "int32",
            "chunk_grid": {
                "configuration": {"chunk_shape": [4, 4]},
                "name": "regular"
            }, 
            "chunk_key_encoding": {
                "configuration": {"separator": "/"},
                "name": "default"
            },
            "codecs": [
                {"configuration": {"endian": "little"}, "name": "bytes"},
                {"configuration": {
                    "typesize": 4, "cname": "zstd", "clevel": 5, "shuffle": "noshuffle", "blocksize": 0}, "name": "blosc"
                }
            ],
            "zarr_format": 3,
            "node_type": "array"
        }"#;
        meta.add_column("var1".to_string(), &metadata_str).unwrap();

        assert_eq!(meta.chunks, Some(vec![4, 4]));
        assert_eq!(meta.shape, Some(vec![16, 16]));
        assert_eq!(meta.last_chunk_idx, Some(vec![3, 3]));
        assert_eq!(meta.columns, vec!["var1"]);

        assert_eq!(
            meta.array_params["var1"],
            ZarrArrayMetadata {
                zarr_format: 3,
                data_type: ZarrDataType::Int(4),
                chunk_separator: ChunkSeparator::Slash,
                sharding_options: None,
                codecs: vec![
                    ZarrCodec::Bytes(Endianness::Little),
                    ZarrCodec::BloscCompressor(
                        BloscOptions::new(
                            CompressorName::Zstd,
                            5,
                            ShuffleOptions::Noshuffle,
                            0,
                        )
                    )
                ]
            }
        );

        meta = ZarrStoreMetadata::new();
        let metadata_str = r#"
        {
            "shape": [16, 16],
            "data_type": "int32",
            "chunk_grid": {
                "configuration": {"chunk_shape": [8, 8]},
                "name": "regular"
            },
            "chunk_key_encoding": {
                "configuration": {"separator": "."},
                "name": "v2"
            },
            "codecs": [
                {
                    "configuration": {
                        "chunk_shape": [4, 4],
                        "codecs": [
                            {"configuration": {"endian": "little"}, "name": "bytes"},
                            {"configuration": {
                                "typesize": 4, "cname": "zstd", "clevel": 5, "shuffle": "noshuffle", "blocksize": 0}, "name": "blosc"
                            }
                        ],
                        "index_codecs": [
                            {"configuration": {"endian": "little"}, "name": "bytes"},
                            {"name": "crc32c"}
                        ],
                        "index_location": "end"
                    },
                    "name": "sharding_indexed"
                }
            ],
            "zarr_format": 3,
            "node_type": "array"
        }"#;
        meta.add_column("var2".to_string(), &metadata_str).unwrap();
        
        assert_eq!(meta.chunks, Some(vec![8, 8]));
        assert_eq!(meta.shape, Some(vec![16, 16]));
        assert_eq!(meta.last_chunk_idx, Some(vec![1, 1]));
        assert_eq!(meta.columns, vec!["var2"]);

        assert_eq!(
            meta.array_params["var2"],
            ZarrArrayMetadata {
                zarr_format: 3,
                data_type: ZarrDataType::Int(4),
                chunk_separator: ChunkSeparator::Period,
                sharding_options: Some(
                    ShardingOptions::new(
                        vec![4, 4],
                        vec![2, 2],
                        vec![
                            ZarrCodec::Bytes(Endianness::Little),
                            ZarrCodec::BloscCompressor(
                                BloscOptions::new(
                                    CompressorName::Zstd,
                                    5,
                                    ShuffleOptions::Noshuffle,
                                    0,
                                )
                            )
                        ],
                        vec![
                            ZarrCodec::Bytes(Endianness::Little),
                            ZarrCodec::Crc32c,
                        ],
                        IndexLocation::End,
                        0   
                    )
                ),
                codecs: vec![],
            }
        );
    }

    #[test]
    fn test_invalid_metadata_v3() {
        let mut meta = ZarrStoreMetadata::new();

        // no array to bytes codec
        let metadata_str = r#"
        {
            "shape": [16, 16],
            "data_type": "int32",
            "chunk_grid": {
                "configuration": {"chunk_shape": [4, 4]},
                "name": "regular"
            }, 
            "chunk_key_encoding": {
                "configuration": {"separator": "/"},
                "name": "default"
            },
            "codecs": [
                {"configuration": {"typesize": 4, "cname": "zstd", "clevel": 5, "shuffle": "noshuffle", "blocksize": 0}, "name": "blosc"}
            ],
            "zarr_format": 3,
            "node_type": "array"
        }"#;
        assert!(meta.add_column("var1".to_string(), &metadata_str).is_err());

        // mismatch between shape and chunks
        let metadata_str = r#"
        {
            "shape": [16, 16],
            "data_type": "int32",
            "chunk_grid": {
                "configuration": {"chunk_shape": [4, 4, 4]},
                "name": "regular"
            }, 
            "chunk_key_encoding": {
                "configuration": {"separator": "/"},
                "name": "default"
            },
            "codecs": [
                {"configuration": {"endian": "little"}, "name": "bytes"},
                {"configuration": {"typesize": 4, "cname": "zstd", "clevel": 5, "shuffle": "noshuffle", "blocksize": 0}, "name": "blosc"}
            ],
            "zarr_format": 3,
            "node_type": "array"
        }"#;
        assert!(meta.add_column("var2".to_string(), &metadata_str).is_err());
    }
}