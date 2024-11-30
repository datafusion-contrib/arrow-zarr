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

use crate::reader::errors::throw_invalid_meta;
use crate::reader::{ZarrError, ZarrResult};
use arrow_array::*;
use arrow_schema::{DataType, Field, FieldRef, TimeUnit};
use crc32c::crc32c;
use flate2::read::GzDecoder;
use itertools::Itertools;
use std::io::Read;
use std::str::FromStr;
use std::sync::Arc;
use std::vec;

// couple useful constant for empty shards
const NULL_OFFSET: usize = usize::pow(2, 63) + (usize::pow(2, 63) - 1);
const NULL_NBYTES: usize = usize::pow(2, 63) + (usize::pow(2, 63) - 1);

// Type enum and for the various support zarr types
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

impl ZarrDataType {
    pub(crate) fn get_byte_size(&self) -> usize {
        match self {
            Self::Bool => 1,
            Self::UInt(s)
            | Self::Int(s)
            | Self::Float(s)
            | Self::FixedLengthString(s)
            | Self::FixedLengthPyUnicode(s)
            | Self::TimeStamp(s, _) => *s,
        }
    }
}

impl TryFrom<&ZarrDataType> for DataType {
    type Error = ZarrError;

    fn try_from(value: &ZarrDataType) -> ZarrResult<Self> {
        match value {
            ZarrDataType::Bool => Ok(DataType::Boolean),
            ZarrDataType::UInt(s) => match s {
                1 => Ok(DataType::UInt8),
                2 => Ok(DataType::UInt16),
                4 => Ok(DataType::UInt32),
                8 => Ok(DataType::UInt64),
                _ => Err(throw_invalid_meta("Invalid uint size")),
            },
            ZarrDataType::Int(s) => match s {
                1 => Ok(DataType::Int8),
                2 => Ok(DataType::Int16),
                4 => Ok(DataType::Int32),
                8 => Ok(DataType::Int64),
                _ => Err(throw_invalid_meta("Invalid int size")),
            },
            ZarrDataType::Float(s) => match s {
                4 => Ok(DataType::Float32),
                8 => Ok(DataType::Float64),
                _ => Err(throw_invalid_meta("Invalid float size")),
            },
            ZarrDataType::FixedLengthString(_) => Ok(DataType::Utf8),
            ZarrDataType::FixedLengthPyUnicode(_) => Ok(DataType::Utf8),
            ZarrDataType::TimeStamp(_, s) => match s.as_str() {
                "s" => Ok(DataType::Timestamp(TimeUnit::Second, None)),
                "ms" => Ok(DataType::Timestamp(TimeUnit::Millisecond, None)),
                "us" => Ok(DataType::Timestamp(TimeUnit::Microsecond, None)),
                "ns" => Ok(DataType::Timestamp(TimeUnit::Nanosecond, None)),
                _ => Err(throw_invalid_meta("Invalid timestamp unit")),
            },
        }
    }
}

// This is the byte length of the Py Unicode characters that zarr writes
// when the output type is set to U<length>.
pub(crate) const PY_UNICODE_SIZE: usize = 4;

// Blosc compression options
#[derive(Debug, PartialEq, Clone)]
pub(crate) enum CompressorName {
    Lz4,
    Lz4hc,
    Blosclz,
    Zstd,
    Snappy,
    Zlib,
}

impl FromStr for CompressorName {
    type Err = ZarrError;

    fn from_str(input: &str) -> ZarrResult<Self> {
        match input {
            "lz4" => Ok(CompressorName::Lz4),
            "lz4hc" => Ok(CompressorName::Lz4hc),
            "blosclz" => Ok(CompressorName::Blosclz),
            "zstd" => Ok(CompressorName::Zstd),
            "snappy" => Ok(CompressorName::Snappy),
            "zlib" => Ok(CompressorName::Zlib),
            _ => Err(throw_invalid_meta("Invalid compressor name")),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum ShuffleOptions {
    Noshuffle,
    ByteShuffle(usize),
    BitShuffle(usize),
}

impl ShuffleOptions {
    pub(crate) fn new(opt: &str, typesize: usize) -> ZarrResult<Self> {
        match opt {
            "noshuffle" => Ok(ShuffleOptions::Noshuffle),
            "shuffle" => Ok(ShuffleOptions::ByteShuffle(typesize)),
            "bitshuffle" => Ok(ShuffleOptions::BitShuffle(typesize)),
            _ => Err(throw_invalid_meta("Invalid shuffle options")),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) struct BloscOptions {
    cname: CompressorName,
    clevel: u8,
    shuffle: ShuffleOptions,
    blocksize: usize,
}

impl BloscOptions {
    pub(crate) fn new(
        cname: CompressorName,
        clevel: u8,
        shuffle: ShuffleOptions,
        blocksize: usize,
    ) -> Self {
        Self {
            cname,
            clevel,
            shuffle,
            blocksize,
        }
    }
}

// Endianness options
#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Endianness {
    Big,
    Little,
}

impl FromStr for Endianness {
    type Err = ZarrError;

    fn from_str(input: &str) -> ZarrResult<Self> {
        match input {
            "big" => Ok(Endianness::Big),
            "little" => Ok(Endianness::Little),
            _ => Err(throw_invalid_meta("Invalid endianness")),
        }
    }
}

// Sharding options
#[derive(Debug, PartialEq, Clone)]
pub(crate) enum IndexLocation {
    Start,
    End,
}

impl FromStr for IndexLocation {
    type Err = ZarrError;

    fn from_str(input: &str) -> ZarrResult<Self> {
        match input {
            "start" => Ok(IndexLocation::Start),
            "end" => Ok(IndexLocation::End),
            _ => Err(throw_invalid_meta("Invalid sharding index location")),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) struct ShardingOptions {
    chunk_shape: Vec<usize>,
    n_chunks: Vec<usize>,
    codecs: Vec<ZarrCodec>,
    index_codecs: Vec<ZarrCodec>,
    index_location: IndexLocation,
    position_in_codecs: usize,
}

impl ShardingOptions {
    pub(crate) fn new(
        chunk_shape: Vec<usize>,
        n_chunks: Vec<usize>,
        codecs: Vec<ZarrCodec>,
        index_codecs: Vec<ZarrCodec>,
        index_location: IndexLocation,
        position_in_codecs: usize,
    ) -> Self {
        Self {
            chunk_shape,
            n_chunks,
            codecs,
            index_codecs,
            index_location,
            position_in_codecs,
        }
    }
}

// enum for all the supported codecs
#[derive(Debug, PartialEq, Clone)]
pub(crate) enum ZarrCodec {
    Transpose(Vec<usize>),
    Bytes(Endianness),
    BloscCompressor(BloscOptions),
    Crc32c,
    Gzip(u8),
}

#[derive(PartialEq)]
pub(crate) enum CodecType {
    ArrayToArray,
    ArrayToBytes,
    BytesToBytes,
}

impl ZarrCodec {
    pub(crate) fn codec_type(&self) -> CodecType {
        match self {
            Self::Transpose(_) => CodecType::ArrayToArray,
            Self::Bytes(_) => CodecType::ArrayToBytes,
            Self::BloscCompressor(_) | Self::Crc32c | Self::Gzip(_) => CodecType::BytesToBytes,
        }
    }
}

// function to decode data that was encoded using a transpose codec
fn decode_transpose<T: Clone>(
    input: Vec<T>,
    chunk_dims: &[usize],
    order: &[usize],
) -> ZarrResult<Vec<T>> {
    let new_indices: Vec<_> = match order.len() {
        2 => (0..chunk_dims[order[0]])
            .cartesian_product(0..chunk_dims[order[1]])
            .map(|t| t.0 * chunk_dims[1] + t.1)
            .collect(),
        3 => (0..chunk_dims[order[0]])
            .cartesian_product(0..chunk_dims[order[1]])
            .cartesian_product(0..chunk_dims[order[2]])
            .map(|t| t.0 .0 * chunk_dims[1] * chunk_dims[2] + t.0 .1 * chunk_dims[2] + t.1)
            .collect(),
        _ => {
            panic!("Invalid number of dims for transpose")
        }
    };

    Ok(new_indices
        .into_iter()
        .map(move |idx| input[idx].clone())
        .collect::<Vec<T>>())
}

// function to only keep the data at the specified indices from a vector.
// it function only works if the indices to keep are ordered.
fn keep_indices<T: Clone + Default>(v: &mut Vec<T>, indices: &[usize]) {
    for (move_to, i) in indices.iter().enumerate() {
        if i != &move_to {
            v[move_to] = v[*i].clone();
        }
    }
    v.resize(indices.len(), T::default());
}

// if a chunk doesn't exactly line up with the edge of an array, drop any
// "phantom" data that is just there for the chunk to be complete but is not
// actually part of the data stored in the array.
fn process_edge_chunk<T: Clone + Default>(
    buf: &mut Vec<T>,
    chunk_dims: &Vec<usize>,
    real_dims: &Vec<usize>,
) {
    if chunk_dims == real_dims {
        return;
    }

    let n_dims = chunk_dims.len();
    let indices_to_keep: Vec<_> = match n_dims {
        1 => (0..real_dims[0]).collect(),
        2 => (0..real_dims[0])
            .cartesian_product(0..real_dims[1])
            .map(|t| t.0 * chunk_dims[1] + t.1)
            .collect(),
        3 => (0..real_dims[0])
            .cartesian_product(0..real_dims[1])
            .cartesian_product(0..real_dims[2])
            .map(|t| t.0 .0 * chunk_dims[1] * chunk_dims[2] + t.0 .1 * chunk_dims[2] + t.1)
            .collect(),
        _ => {
            panic!("Zarr edge chunk with more than 3 dimensions, 3 is the limit")
        }
    };

    keep_indices(buf, &indices_to_keep);
}

// decode data that was encoded with a bytes to bytes codec.
fn apply_bytes_to_bytes_codec(codec: &ZarrCodec, bytes: &[u8]) -> ZarrResult<Vec<u8>> {
    let mut decompressed_bytes = Vec::new();
    match codec {
        ZarrCodec::Gzip(_) => {
            let mut decoder = GzDecoder::new(bytes);
            decoder.read_to_end(&mut decompressed_bytes)?;
        }
        ZarrCodec::BloscCompressor(_) => {
            decompressed_bytes = unsafe { blosc::decompress_bytes(bytes).unwrap() };
        }
        ZarrCodec::Crc32c => {
            let mut bytes = bytes.to_vec();
            let l = bytes.len();
            let checksum = bytes.split_off(l - 4);
            let checksum = [checksum[0], checksum[1], checksum[2], checksum[3]];
            if crc32c(&bytes[..]) != u32::from_le_bytes(checksum) {
                return Err(throw_invalid_meta("crc32c checksum failed"));
            }
            decompressed_bytes = bytes;
        }
        _ => return Err(throw_invalid_meta("invalid bytes to bytes codec")),
    }

    Ok(decompressed_bytes)
}

// decode data that was encoded with a sequence of bytes to bytes codedcs, and return
// the array to bytes codec and array to array codec that come after, if they are present.
fn decode_bytes_to_bytes(
    codecs: &[ZarrCodec],
    bytes: &[u8],
    sharding_params: &Option<ShardingOptions>,
) -> ZarrResult<(Vec<u8>, Option<ZarrCodec>, Option<ZarrCodec>)> {
    let mut array_to_bytes_codec: Option<ZarrCodec> = None;
    let mut array_to_array_codec: Option<ZarrCodec> = None;
    let mut decompressed_bytes: Option<Vec<u8>> = None;
    for codec in codecs.iter().rev() {
        match codec.codec_type() {
            CodecType::BytesToBytes => {
                if array_to_bytes_codec.is_some() {
                    return Err(throw_invalid_meta("incorrect codec order in zarr metadata"));
                }
                decompressed_bytes = Some(apply_bytes_to_bytes_codec(codec, bytes)?);
            }
            CodecType::ArrayToBytes => {
                if array_to_bytes_codec.is_some() {
                    return Err(throw_invalid_meta(
                        "only one array to bytes codec is allowed",
                    ));
                }
                array_to_bytes_codec = Some(codec.clone());
            }
            CodecType::ArrayToArray => {
                if sharding_params.is_none() && array_to_bytes_codec.is_none() {
                    return Err(throw_invalid_meta("incorrect codec order in zarr metadata"));
                }
                array_to_array_codec = Some(codec.clone());
            }
        }
    }

    if decompressed_bytes.is_none() {
        decompressed_bytes = Some(bytes.to_vec());
    }

    Ok((
        decompressed_bytes.unwrap(),
        array_to_bytes_codec,
        array_to_array_codec,
    ))
}

// macro to decode data by applying bytes to type conversions.
macro_rules! convert_bytes {
    ($bytes: expr, $e: expr, $type: ty, 1) => {
        if $e == &Endianness::Little {
            $bytes
                .into_iter()
                .map(|b| <$type>::from_le_bytes([b]))
                .collect()
        } else {
            $bytes
                .into_iter()
                .map(|b| <$type>::from_be_bytes([b]))
                .collect()
        }
    };
    ($bytes: expr, $e: expr, $type: ty, 2) => {
        if $e == &Endianness::Little {
            $bytes
                .chunks(2)
                .into_iter()
                .map(|arr| <$type>::from_le_bytes([arr[0], arr[1]]))
                .collect()
        } else {
            $bytes
                .chunks(2)
                .into_iter()
                .map(|arr| <$type>::from_be_bytes([arr[0], arr[1]]))
                .collect()
        }
    };
    ($bytes: expr, $e: expr, $type: ty, 4) => {
        if $e == &Endianness::Little {
            $bytes
                .chunks(4)
                .into_iter()
                .map(|arr| <$type>::from_le_bytes([arr[0], arr[1], arr[2], arr[3]]))
                .collect()
        } else {
            $bytes
                .chunks(4)
                .into_iter()
                .map(|arr| <$type>::from_be_bytes([arr[0], arr[1], arr[2], arr[3]]))
                .collect()
        }
    };
    ($bytes: expr, $e: expr, $type: ty, 8) => {
        if $e == &Endianness::Little {
            $bytes
                .chunks(8)
                .into_iter()
                .map(|arr| {
                    <$type>::from_le_bytes([
                        arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7],
                    ])
                })
                .collect()
        } else {
            $bytes
                .chunks(8)
                .into_iter()
                .map(|arr| {
                    <$type>::from_be_bytes([
                        arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7],
                    ])
                })
                .collect()
        }
    };
}

// extract the indices for the positions of the inner chunks within a shard.
fn extract_sharding_index(
    index_codecs: &[ZarrCodec],
    mut bytes: Vec<u8>,
) -> ZarrResult<(Vec<usize>, Vec<usize>)> {
    // here we are simplifying things, the codecs must include one endianness codec, and
    // optionally one checksum codec. while technically this could change in the future,
    // for now it's the recommended approach, and I think the only one that makes sense,
    // so nothing else will be allowed here, it makes things much easier.
    if index_codecs.len() > 2 {
        return Err(throw_invalid_meta("too many sharding index codecs"));
    }

    if index_codecs.len() == 2 {
        if index_codecs[1] != ZarrCodec::Crc32c {
            return Err(throw_invalid_meta(
                "second sharding index codec, if provided, must crc32c",
            ));
        }
        bytes = apply_bytes_to_bytes_codec(&ZarrCodec::Crc32c, &bytes[..])?;
    }

    let mut offsets = Vec::new();
    let mut nbytes = Vec::new();
    if let ZarrCodec::Bytes(e) = &index_codecs[0] {
        let mut indices: Vec<_> = convert_bytes!(&bytes[..], e, u64, 8);
        let mut flip = true;
        while !indices.is_empty() {
            if flip {
                offsets.push(indices.remove(0) as usize);
            } else {
                nbytes.push(indices.remove(0) as usize)
            }
            flip = !flip;
        }
    } else {
        return Err(throw_invalid_meta(
            "sharding index codecs must contain a bytes codec",
        ));
    }

    Ok((offsets, nbytes))
}

// determine the real dimensions (different from the chunk dimensions if the chunk doesn't exactly
// line up with the edge of the array) for an inner chunk within a shard.
fn get_inner_chunk_real_dims(
    params: &ShardingOptions,
    outer_real_dims: &Vec<usize>,
    pos: usize,
) -> Vec<usize> {
    if &params.chunk_shape == outer_real_dims {
        return params.chunk_shape.clone();
    }

    let vec_pos: Vec<usize>;
    match params.n_chunks.len() {
        1 => vec_pos = vec![pos],
        2 => {
            let i = pos / params.n_chunks[0];
            let j = pos - (i * params.n_chunks[0]);
            vec_pos = vec![i, j];
        }
        3 => {
            let s = params.n_chunks[0] * params.n_chunks[1];
            let i = pos / s;
            let j = (pos - i * s) / params.n_chunks[0];
            let k = pos - (i * s) - j * params.n_chunks[0];
            vec_pos = vec![i, j, k];
        }
        _ => {
            panic!("Zarr chunk with more than 3 dimensions, 3 is the limit")
        }
    }

    let real_dims = params
        .chunk_shape
        .iter()
        .zip(outer_real_dims.iter())
        .zip(vec_pos.iter())
        .map(|((cs, ord), vp)| std::cmp::min((vp + 1) * cs, *ord) - vp * cs)
        .collect();

    real_dims
}

fn broadcast_array<T: Clone>(
    input: Vec<T>,
    real_chunk_dims: &[usize],
    axis: usize,
) -> ZarrResult<Vec<T>> {
    let err = "invalid parameters while processing one dimenesional representation";
    let l = input.len();
    let n_dims = real_chunk_dims.len();
    if axis >= n_dims || l != real_chunk_dims[axis] {
        return Err(throw_invalid_meta(err));
    }

    match (n_dims, axis) {
        (2, 0) => Ok(input
            .into_iter()
            .flat_map(|v| std::iter::repeat(v).take(real_chunk_dims[1]))
            .collect()),
        (2, 1) => Ok(vec![&input[..]; real_chunk_dims[0]].concat()),
        (3, 0) => Ok(input
            .into_iter()
            .flat_map(|v| std::iter::repeat(v).take(real_chunk_dims[1] * real_chunk_dims[2]))
            .collect()),
        (3, 1) => {
            let v: Vec<_> = input
                .into_iter()
                .flat_map(|v| std::iter::repeat(v).take(real_chunk_dims[2]))
                .collect();
            Ok(vec![&v[..]; real_chunk_dims[0]].concat())
        }
        (3, 2) => Ok(vec![&input[..]; real_chunk_dims[0] * real_chunk_dims[1]].concat()),
        _ => Err(throw_invalid_meta(err)),
    }
}

// the logic here gets a little messy, but the goal is to fill in the
// the data for the outer chunks from the data within the inner shards
// in the outer chunk. the below function is called one inner shard at
// a time, and the "flat" data from the inner shard is read in small
// chunks and written to the correct position in the array for the entire
// outer chunk.
fn fill_data_from_shard<T: Clone>(
    data: &mut [T],
    shard_data: &[T],
    chunk_real_dims: &[usize],
    chunk_dims: &[usize],
    inner_real_dims: &[usize],
    inner_dims: &[usize],
    pos: usize,
) -> ZarrResult<()> {
    let l = inner_real_dims.len();
    let err = "mismatch between inner and outer chunks dimensions";
    if l != chunk_real_dims.len() || l != inner_dims.len() {
        return Err(throw_invalid_meta(err));
    }

    match l {
        // the 1D array case is easy, there's just one offset to compute,
        // from the start of the 1D outer chunk.
        1 => {
            let stride = inner_dims[0];
            data[pos * stride..(pos + 1) * stride].clone_from_slice(shard_data);
        }
        // The 2D case is trickier, we need to keep track of the inner shard
        // position within the outer chunk, as well the where we're reading
        // in the inner shard.
        2 => {
            // first, find the position of the shard within the chunk.
            let n_shards_1 = chunk_dims[1] / inner_dims[1];
            let i = pos / n_shards_1;
            let j = pos - i * n_shards_1;
            let shard_pos = [i, j];

            // then loop over each row in the shard and write to the
            // data array for the chunk.
            let stride = inner_real_dims[1];
            for row_idx in 0..inner_real_dims[0] {
                // read the row from the 1D shard array
                let shard_row = &shard_data[row_idx * stride..(row_idx + 1) * stride];

                // determine where in the chunk data array the shard array should go.
                // the first component of the offsets is the "vertrical" position of
                // the shard within the chunk times the rows per shard, plus the row
                // within the shard, all that times the number of rows per shards.
                // the second component is the "horizontal" position of the shard
                // within the chunk, times the number of columns per shard.
                let chunk_offset = (shard_pos[0] * inner_dims[0] + row_idx) * chunk_real_dims[1]
                    + shard_pos[1] * inner_dims[1];
                data[chunk_offset..chunk_offset + stride].clone_from_slice(shard_row);
            }
        }
        // similar to the 2D case, but a but more complicated, for 3D arrays.
        3 => {
            let n_shards_1 = chunk_dims[1] / inner_dims[1];
            let n_shards_2 = chunk_dims[2] / inner_dims[2];
            let i = pos / (n_shards_1 * n_shards_2);
            let j = (pos - i * (n_shards_1 * n_shards_2)) / n_shards_2;
            let k = pos - i * (n_shards_1 * n_shards_2) - j * n_shards_2;
            let shard_pos = [i, j, k];

            let stride = inner_real_dims[2];
            for depth_idx in 0..inner_real_dims[0] {
                for row_idx in 0..inner_real_dims[1] {
                    let shard_start = depth_idx * inner_real_dims[1] * stride + row_idx * stride;
                    let shard_row = &shard_data[shard_start..shard_start + stride];
                    let chunk_offset = (shard_pos[0] * inner_dims[0] + depth_idx)
                        * chunk_real_dims[1]
                        * chunk_real_dims[2]
                        + (shard_pos[1] * inner_dims[1] + row_idx) * chunk_real_dims[2]
                        + shard_pos[2] * inner_dims[2];

                    data[chunk_offset..chunk_offset + stride].clone_from_slice(shard_row);
                }
            }
        }
        _ => {
            return Err(throw_invalid_meta(
                "too many dimensions when processing shards",
            ));
        }
    }

    Ok(())
}

// a macro that instantiates functions to decode different data types
macro_rules! create_decode_function {
    ($func_name: tt, $type: ty, $byte_size: tt) => {
        fn $func_name(
            mut bytes: Vec<u8>,
            chunk_dims: &Vec<usize>,
            real_dims: &Vec<usize>,
            codecs: &Vec<ZarrCodec>,
            sharding_params: Option<ShardingOptions>,
        ) -> ZarrResult<Vec<$type>> {
            let array_to_bytes_codec: Option<ZarrCodec>;
            let array_to_array_codec: Option<ZarrCodec>;
            (bytes, array_to_bytes_codec, array_to_array_codec) =
                decode_bytes_to_bytes(&codecs, &bytes[..], &sharding_params)?;

            let mut data;
            if let Some(sharding_params) = sharding_params.as_ref() {
                let mut index_size: usize =
                    2 * 8 * sharding_params.n_chunks.iter().fold(1, |mult, x| mult * x);
                index_size += sharding_params
                    .index_codecs
                    .iter()
                    .any(|c| c == &ZarrCodec::Crc32c) as usize;
                let index_bytes = match sharding_params.index_location {
                    IndexLocation::Start => bytes[0..index_size].to_vec(),
                    IndexLocation::End => bytes[(bytes.len() - index_size)..].to_vec(),
                };
                let (offsets, nbytes) =
                    extract_sharding_index(&sharding_params.index_codecs, index_bytes)?;

                data = vec![<$type>::default(); bytes.len()];
                let mut total_length = 0;
                for (pos, (o, n)) in offsets.iter().zip(nbytes.iter()).enumerate() {
                    // the below condition indicates an empty shard
                    if o == &NULL_OFFSET && n == &NULL_NBYTES {
                        continue;
                    }
                    let inner_real_dims =
                        get_inner_chunk_real_dims(&sharding_params, &real_dims, pos);
                    let inner_data = $func_name(
                        bytes[*o..o + n].to_vec(),
                        &sharding_params.chunk_shape,
                        &inner_real_dims,
                        &sharding_params.codecs,
                        None,
                    )?;
                    total_length += inner_data.len();
                    fill_data_from_shard(
                        &mut data,
                        &inner_data,
                        &real_dims,
                        &chunk_dims,
                        &inner_real_dims,
                        &sharding_params.chunk_shape,
                        pos,
                    )?;
                }
                data = data[0..total_length].to_vec();
            } else {
                if let Some(ZarrCodec::Bytes(e)) = array_to_bytes_codec {
                    data = convert_bytes!(bytes, &e, $type, $byte_size);
                } else {
                    panic!("No bytes codec to apply on zarr chunk");
                }
            }

            if let Some(ZarrCodec::Transpose(o)) = array_to_array_codec {
                if sharding_params.is_some() {
                    return Err(throw_invalid_meta(
                        "no array to array codec allowed outside of sharding codec.",
                    ));
                }
                data = decode_transpose(data, chunk_dims, &o)?
            }
            if sharding_params.is_none() {
                process_edge_chunk(&mut data, chunk_dims, real_dims);
            }

            return Ok(data);
        }
    };
}

create_decode_function!(decode_u8_chunk, u8, 1);
create_decode_function!(decode_u16_chunk, u16, 2);
create_decode_function!(decode_u32_chunk, u32, 4);
create_decode_function!(decode_u64_chunk, u64, 8);
create_decode_function!(decode_i8_chunk, i8, 1);
create_decode_function!(decode_i16_chunk, i16, 2);
create_decode_function!(decode_i32_chunk, i32, 4);
create_decode_function!(decode_i64_chunk, i64, 8);
create_decode_function!(decode_f32_chunk, f32, 4);
create_decode_function!(decode_f64_chunk, f64, 8);

// a separate function to decode string data
fn decode_string_chunk(
    mut bytes: Vec<u8>,
    str_len: usize,
    chunk_dims: &Vec<usize>,
    real_dims: &Vec<usize>,
    codecs: &[ZarrCodec],
    sharding_params: Option<ShardingOptions>,
    pyunicode: bool,
) -> ZarrResult<Vec<String>> {
    let array_to_bytes_codec: Option<ZarrCodec>;
    let array_to_array_codec: Option<ZarrCodec>;
    (bytes, array_to_bytes_codec, array_to_array_codec) =
        decode_bytes_to_bytes(codecs, &bytes[..], &sharding_params)?;

    // special case of Py Unicode, with 4 byte characters. Here we simply
    // keep one byte, might need to be more robust, perhaps throw an error
    // if the other 3 bytes are not 0s. Might also not work with big endian?
    if pyunicode {
        bytes = bytes.iter().step_by(PY_UNICODE_SIZE).copied().collect();
    }

    let mut data;
    if let Some(sharding_params) = sharding_params.as_ref() {
        let mut index_size: usize = 2 * 8 * sharding_params.n_chunks.iter().product::<usize>();
        index_size += sharding_params
            .index_codecs
            .iter()
            .any(|c| c == &ZarrCodec::Crc32c) as usize;
        let index_bytes = match sharding_params.index_location {
            IndexLocation::Start => bytes[0..index_size].to_vec(),
            IndexLocation::End => bytes[(bytes.len() - index_size)..].to_vec(),
        };
        let (offsets, nbytes) = extract_sharding_index(&sharding_params.index_codecs, index_bytes)?;

        data = vec![String::default(); bytes.len()];
        let mut total_length = 0;
        for (pos, (o, n)) in offsets.iter().zip(nbytes.iter()).enumerate() {
            // the below condition indicates an empty shard
            if o == &NULL_OFFSET && n == &NULL_NBYTES {
                continue;
            }
            let inner_real_dims = get_inner_chunk_real_dims(sharding_params, real_dims, pos);
            let inner_data = decode_string_chunk(
                bytes[*o..o + n].to_vec(),
                str_len,
                &sharding_params.chunk_shape,
                &inner_real_dims,
                &sharding_params.codecs,
                None,
                pyunicode,
            )?;
            total_length += inner_data.len();
            fill_data_from_shard(
                &mut data,
                &inner_data,
                real_dims,
                chunk_dims,
                &inner_real_dims,
                &sharding_params.chunk_shape,
                pos,
            )?;
        }
        data = data[0..total_length].to_vec();
    } else if let Some(ZarrCodec::Bytes(_)) = array_to_bytes_codec {
        data = bytes
            .chunks(str_len)
            .map(|arr| std::str::from_utf8(arr).unwrap().to_string())
            .collect()
    } else {
        panic!("No bytes codec to apply on zarr chunk");
    }

    if let Some(ZarrCodec::Transpose(o)) = array_to_array_codec {
        if sharding_params.is_some() {
            return Err(throw_invalid_meta(
                "no array to array codec allowed outside of sharding codec.",
            ));
        }
        data = decode_transpose(data, chunk_dims, &o)?
    }
    if sharding_params.is_none() {
        process_edge_chunk(&mut data, chunk_dims, real_dims);
    }

    Ok(data)
}

// the entry point for this module, the only function that is meant to be called from other modules
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_codecs(
    col_name: String,
    raw_data: Vec<u8>,
    chunk_dims: &Vec<usize>,
    real_dims: &Vec<usize>,
    data_type: &ZarrDataType,
    codecs: &Vec<ZarrCodec>,
    sharding_params: Option<ShardingOptions>,
    one_d_array_params: Option<(usize, usize)>,
) -> ZarrResult<(ArrayRef, FieldRef)> {
    macro_rules! return_array {
        ($func_name: tt, $data_t: expr, $array_t: ty) => {
            let data = if let Some(one_d_array_params) = one_d_array_params {
                let one_d_data = $func_name(
                    raw_data,
                    &vec![one_d_array_params.0],
                    &vec![real_dims[one_d_array_params.1]],
                    &codecs,
                    sharding_params,
                )?;
                broadcast_array(one_d_data, &real_dims[..], one_d_array_params.1)?
            } else {
                $func_name(raw_data, &chunk_dims, &real_dims, &codecs, sharding_params)?
            };
            let field = Field::new(col_name, $data_t, false);
            let arr: $array_t = data.into();
            return Ok((Arc::new(arr), Arc::new(field)))
        };
    }

    match data_type {
        ZarrDataType::Bool => {
            let data = decode_u8_chunk(raw_data, chunk_dims, real_dims, codecs, sharding_params)?;
            let data: Vec<bool> = data.iter().map(|x| *x != 0).collect();
            let field = Field::new(col_name, DataType::Boolean, false);
            let arr: BooleanArray = data.into();
            Ok((Arc::new(arr), Arc::new(field)))
        }
        ZarrDataType::UInt(s) => match s {
            1 => {
                return_array!(decode_u8_chunk, DataType::UInt8, UInt8Array);
            }
            2 => {
                return_array!(decode_u16_chunk, DataType::UInt16, UInt16Array);
            }
            4 => {
                return_array!(decode_u32_chunk, DataType::UInt32, UInt32Array);
            }
            8 => {
                return_array!(decode_u64_chunk, DataType::UInt64, UInt64Array);
            }
            _ => Err(throw_invalid_meta("Invalid zarr data type")),
        },
        ZarrDataType::Int(s) => match s {
            1 => {
                return_array!(decode_i8_chunk, DataType::Int8, Int8Array);
            }
            2 => {
                return_array!(decode_i16_chunk, DataType::Int16, Int16Array);
            }
            4 => {
                return_array!(decode_i32_chunk, DataType::Int32, Int32Array);
            }
            8 => {
                return_array!(decode_i64_chunk, DataType::Int64, Int64Array);
            }
            _ => Err(throw_invalid_meta("Invalid zarr data type")),
        },
        ZarrDataType::Float(s) => match s {
            4 => {
                return_array!(decode_f32_chunk, DataType::Float32, Float32Array);
            }
            8 => {
                return_array!(decode_f64_chunk, DataType::Float64, Float64Array);
            }
            _ => Err(throw_invalid_meta("Invalid zarr data type")),
        },
        ZarrDataType::TimeStamp(8, u) => match u.as_str() {
            "s" => {
                return_array!(
                    decode_i64_chunk,
                    DataType::Timestamp(TimeUnit::Second, None),
                    TimestampSecondArray
                );
            }
            "ms" => {
                return_array!(
                    decode_i64_chunk,
                    DataType::Timestamp(TimeUnit::Millisecond, None),
                    TimestampMillisecondArray
                );
            }
            "us" => {
                return_array!(
                    decode_i64_chunk,
                    DataType::Timestamp(TimeUnit::Microsecond, None),
                    TimestampMicrosecondArray
                );
            }
            "ns" => {
                return_array!(
                    decode_i64_chunk,
                    DataType::Timestamp(TimeUnit::Nanosecond, None),
                    TimestampNanosecondArray
                );
            }
            _ => Err(throw_invalid_meta("Invalid zarr data type")),
        },
        ZarrDataType::FixedLengthString(s) | ZarrDataType::FixedLengthPyUnicode(s) => {
            let mut pyunicode = false;
            let mut str_len_adjustment = 1;
            if let ZarrDataType::FixedLengthPyUnicode(_) = data_type {
                pyunicode = true;
                str_len_adjustment = PY_UNICODE_SIZE;
            }
            let data = if let Some(one_d_array_params) = one_d_array_params {
                let one_d_data = decode_string_chunk(
                    raw_data,
                    *s / str_len_adjustment,
                    &vec![one_d_array_params.0],
                    &vec![real_dims[one_d_array_params.1]],
                    codecs,
                    sharding_params,
                    pyunicode,
                )?;
                broadcast_array(one_d_data, &real_dims[..], one_d_array_params.1)?
            } else {
                decode_string_chunk(
                    raw_data,
                    *s / str_len_adjustment,
                    chunk_dims,
                    real_dims,
                    codecs,
                    sharding_params,
                    pyunicode,
                )?
            };
            let field = Field::new(col_name, DataType::Utf8, false);
            let arr: StringArray = data.into();

            Ok((Arc::new(arr), Arc::new(field)))
        }
        _ => Err(throw_invalid_meta("Invalid zarr data type")),
    }
}

#[cfg(test)]
mod zarr_codecs_tests {
    use crate::tests::get_test_v3_data_path;

    use super::*;
    use ::std::fs::read;

    // reading a chunk and decoding it using hard coded, known options. this test
    // doesn't included any sharding.
    #[test]
    fn no_sharding_tests() {
        let path = get_test_v3_data_path("no_sharding.zarr/int_data/c/1/1".to_string());
        let raw_data = read(path).unwrap();

        let chunk_shape = vec![4, 4];
        let real_dims = vec![4, 4];
        let data_type = ZarrDataType::Int(4);
        let codecs = vec![
            ZarrCodec::Bytes(Endianness::Little),
            ZarrCodec::BloscCompressor(BloscOptions::new(
                CompressorName::Zstd,
                5,
                ShuffleOptions::Noshuffle,
                0,
            )),
        ];
        let sharding_params: Option<ShardingOptions> = None;

        let (arr, field) = apply_codecs(
            "int_data".to_string(),
            raw_data,
            &chunk_shape,
            &real_dims,
            &data_type,
            &codecs,
            sharding_params,
            None,
        )
        .unwrap();

        assert_eq!(
            field,
            Arc::new(Field::new("int_data", DataType::Int32, false))
        );
        let target_arr: Int32Array = vec![
            68, 69, 70, 71, 84, 85, 86, 87, 100, 101, 102, 103, 116, 117, 118, 119,
        ]
        .into();
        assert_eq!(*arr, target_arr);
    }

    // reading a chunk and decoding it using hard coded, known options. this test
    // includes sharding.
    #[test]
    fn with_sharding_tests() {
        let path = get_test_v3_data_path("with_sharding.zarr/float_data/1.1".to_string());
        let raw_data = read(path).unwrap();

        let chunk_shape = vec![4, 4];
        let real_dims = vec![4, 4];
        let data_type = ZarrDataType::Float(8);
        let codecs: Vec<ZarrCodec> = vec![];
        let sharding_params = Some(ShardingOptions::new(
            vec![2, 2],
            vec![2, 2],
            vec![
                ZarrCodec::Bytes(Endianness::Little),
                ZarrCodec::BloscCompressor(BloscOptions::new(
                    CompressorName::Zstd,
                    5,
                    ShuffleOptions::Noshuffle,
                    0,
                )),
            ],
            vec![ZarrCodec::Bytes(Endianness::Little)],
            IndexLocation::End,
            0,
        ));

        let (arr, field) = apply_codecs(
            "float_data".to_string(),
            raw_data,
            &chunk_shape,
            &real_dims,
            &data_type,
            &codecs,
            sharding_params,
            None,
        )
        .unwrap();

        assert_eq!(
            field,
            Arc::new(Field::new("float_data", DataType::Float64, false))
        );
        let target_arr: Float64Array = vec![
            36.0, 37.0, 38.0, 39.0, 44.0, 45.0, 46.0, 47.0, 52.0, 53.0, 54.0, 55.0, 60.0, 61.0,
            62.0, 63.0,
        ]
        .into();
        assert_eq!(*arr, target_arr);
    }

    // reading a chunk and decoding it using hard coded, known options. this test
    // includes sharding, and the shape doesn't exactly line up with the chunks.
    #[test]
    fn with_sharding_with_edge_tests() {
        let path = get_test_v3_data_path("with_sharding_with_edge.zarr/uint_data/1.1".to_string());
        let raw_data = read(path).unwrap();

        let chunk_shape = vec![4, 4];
        let real_dims = vec![3, 3];
        let data_type = ZarrDataType::UInt(2);
        let codecs: Vec<ZarrCodec> = vec![];
        let sharding_params = Some(ShardingOptions::new(
            vec![2, 2],
            vec![2, 2],
            vec![
                ZarrCodec::Bytes(Endianness::Little),
                ZarrCodec::BloscCompressor(BloscOptions::new(
                    CompressorName::Zstd,
                    5,
                    ShuffleOptions::Noshuffle,
                    0,
                )),
            ],
            vec![ZarrCodec::Bytes(Endianness::Little)],
            IndexLocation::End,
            0,
        ));

        let (arr, field) = apply_codecs(
            "uint_data".to_string(),
            raw_data,
            &chunk_shape,
            &real_dims,
            &data_type,
            &codecs,
            sharding_params,
            None,
        )
        .unwrap();

        assert_eq!(
            field,
            Arc::new(Field::new("uint_data", DataType::UInt16, false))
        );
        let target_arr: UInt16Array = vec![32, 33, 34, 39, 40, 41, 46, 47, 48].into();
        assert_eq!(*arr, target_arr);
    }

    #[test]
    fn test_zarr_data_type_to_arrow_datatype() -> ZarrResult<()> {
        let zarr_types = [
            ZarrDataType::Bool,
            ZarrDataType::UInt(1),
            ZarrDataType::UInt(2),
            ZarrDataType::UInt(4),
            ZarrDataType::UInt(8),
            ZarrDataType::Int(1),
            ZarrDataType::Int(2),
            ZarrDataType::Int(4),
            ZarrDataType::Int(8),
            ZarrDataType::Float(4),
            ZarrDataType::Float(8),
            ZarrDataType::TimeStamp(8, "s".to_string()),
            ZarrDataType::TimeStamp(8, "ms".to_string()),
            ZarrDataType::TimeStamp(8, "us".to_string()),
            ZarrDataType::TimeStamp(8, "ns".to_string()),
        ];

        let exepcted_types = [
            DataType::Boolean,
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::Float32,
            DataType::Float64,
            DataType::Timestamp(TimeUnit::Second, None),
            DataType::Timestamp(TimeUnit::Millisecond, None),
            DataType::Timestamp(TimeUnit::Microsecond, None),
            DataType::Timestamp(TimeUnit::Nanosecond, None),
        ];

        assert_eq!(zarr_types.len(), exepcted_types.len());

        for (zarr_type, expected_type) in zarr_types.iter().zip(exepcted_types.iter()) {
            let arrow_type = DataType::try_from(zarr_type)?;

            assert_eq!(arrow_type, *expected_type);
        }

        Ok(())
    }

    #[test]
    fn test_broadcast_array() {
        // (1, n) 1D projected to 2D
        let v = broadcast_array(vec![1, 2, 3], &[4, 3], 1).unwrap();
        assert_eq!(v, vec![1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]);

        //(n, 1) 1D projected to 2D
        let v = broadcast_array(vec![1, 2, 3, 4], &[4, 3], 0).unwrap();
        assert_eq!(v, vec![1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]);

        // (1, 1, n) 1D projected t0 3D
        let v = broadcast_array(vec![1, 2, 3], &[2, 4, 3], 2).unwrap();
        assert_eq!(
            v,
            vec![1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
        );

        // // (1, n, 1) 1D projected t0 3D
        let v = broadcast_array(vec![1, 2, 3, 4], &[2, 4, 3], 1).unwrap();
        assert_eq!(
            v,
            vec![1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        );

        // // (n, 1, 1) 1D projected t0 3D
        let v = broadcast_array(vec![1, 2], &[2, 4, 3], 0).unwrap();
        assert_eq!(
            v,
            vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        );
    }
}
