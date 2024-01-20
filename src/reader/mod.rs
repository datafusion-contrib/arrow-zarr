//! A module tha provides a sychronous reader for zarr store, to generate [`RecordBatch`]es.
//! 
//! ```
//! # use arrow_zarr::reader::{ZarrRecordBatchReaderBuilder, ZarrProjection};
//! # use arrow_cast::pretty::pretty_format_batches;
//! # use arrow_array::RecordBatch;
//! # use std::path::PathBuf;
//! #
//! # fn get_test_data_path(zarr_store: String) -> PathBuf {
//! #    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testing/data/zarr").join(zarr_store)
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
//! // The ZarrRead trait is implemented for PathBuf, so as long as it points
//! // to a directory with a valid zarr store, it can be used to initialize
//! // a zarr reader builder.
//! let p: PathBuf = get_test_data_path("lat_lon_example.zarr".to_string());
//! 
//! let proj = ZarrProjection::keep(vec!["lat".to_string(), "float_data".to_string()]);
//! let builder = ZarrRecordBatchReaderBuilder::new(p).with_projection(proj);
//! let mut reader = builder.build().unwrap();
//! let rec_batch = reader.next().unwrap().unwrap();
//! 
//! assert_batches_eq(
//!     &[rec_batch],
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
//! );
//! ```

use arrow_schema::{Field, FieldRef, Schema, DataType, TimeUnit};
use arrow_array::*;
use std::sync::Arc;
use arrow_data::ArrayData;
use arrow_buffer::Buffer;
use arrow_buffer::ToByteSlice;
use std::io::Read;
use itertools::Itertools;

use zarr_read::{ZarrRead, ZarrInMemoryArray};
use metadata::{ZarrDataType,  CompressorType, Endianness, MatrixOrder, PY_UNICODE_SIZE};
pub use zarr_read::{ZarrInMemoryChunk, ZarrProjection};
pub use filters::{ZarrChunkFilter, ZarrArrowPredicate, ZarrArrowPredicateFn};
pub use errors::{ZarrResult, ZarrError};
pub use metadata::ZarrStoreMetadata;


pub(crate) mod metadata;
mod zarr_read;
mod errors;
mod filters;

/// A zarr store that holds a reader for all the zarr data.
pub struct ZarrStore<T: ZarrRead> {
    meta: ZarrStoreMetadata,
    chunk_positions: Vec<Vec<usize>>,
    zarr_reader: T,
    projection: ZarrProjection,
    curr_chunk: usize,
}

impl<T: ZarrRead> ZarrStore<T> {
    pub(crate) fn new(
        zarr_reader: T,
        chunk_positions: Vec<Vec<usize>>,
        projection: ZarrProjection,
    ) -> ZarrResult<Self> {
        Ok(Self {
            meta: zarr_reader.get_zarr_metadata()?,
            chunk_positions,
            zarr_reader,
            projection,
            curr_chunk: 0,
        })
    }
}

/// A trait with a method to iterate on zarr chunks, but that also allows
/// skipping a chunk without reading it.
pub trait ZarrIterator {
    fn next_chunk(&mut self) -> Option<ZarrResult<ZarrInMemoryChunk>>;
    fn skip_chunk(&mut self);
}

macro_rules! unwrap_or_return {
    ( $e:expr ) => {
        match $e {
            Ok(x) => x,
            Err(e) => return Some(Err(ZarrError::from(e))),
        }
    }
}
pub(crate) use unwrap_or_return;

impl<T: ZarrRead> ZarrIterator for ZarrStore<T> {
    /// Get the data from the next zarr chunk.
    fn next_chunk(&mut self) -> Option<ZarrResult<ZarrInMemoryChunk>> {
        if self.curr_chunk == self.chunk_positions.len() {
            return None;
        }

        let pos = &self.chunk_positions[self.curr_chunk];
        let cols = self.projection.apply_selection(self.meta.get_columns());
        let cols = unwrap_or_return!(cols);

        let chnk = self.zarr_reader.get_zarr_chunk(
            pos, &cols, self.meta.get_real_dims(pos),
        )
        ;
        self.curr_chunk += 1;
        Some(chnk)
    }

    /// Skip the next zarr chunk without reading any data from the
    /// corresponding files.
    fn skip_chunk(&mut self) {
        if self.curr_chunk < self.chunk_positions.len() {
            self.curr_chunk += 1;
        }
    }
}

fn build_field(t: &ZarrDataType, col_name: String) -> ZarrResult<(usize, FieldRef)> {
    match t {
        ZarrDataType::Bool => {
            return Ok((1, Arc::new(Field::new(col_name, DataType::Boolean, false))))
        },
        ZarrDataType::UInt(s) => {
            match s {
                1 => {
                    return Ok((1, Arc::new(Field::new(col_name, DataType::UInt8, false))))
                },
                2 => {
                    return Ok((2, Arc::new(Field::new(col_name, DataType::UInt16, false))))
                },
                4 => {
                    return Ok((4, Arc::new(Field::new(col_name, DataType::UInt32, false))))
                },
                8 => {
                    return Ok((8, Arc::new(Field::new(col_name, DataType::UInt64, false))))
                },
                _ => {return Err(ZarrError::InvalidMetadata("Invalid data type".to_string()))}
            }
        },
        ZarrDataType::Int(s) => {
            match s {
                1 => {
                    return Ok((1, Arc::new(Field::new(col_name, DataType::Int8, false))))
                },
                2 => {
                    return Ok((2, Arc::new(Field::new(col_name, DataType::Int16, false))))
                },
                4 => {
                    return Ok((4, Arc::new(Field::new(col_name, DataType::Int32, false))))
                },
                8 => {
                    return Ok((8, Arc::new(Field::new(col_name, DataType::Int64, false))))
                },
                _ => {return Err(ZarrError::InvalidMetadata("Invalid data type".to_string()))}
            }
        },
        ZarrDataType::Float(s) => {
            match s {
                4 => {
                    return Ok((4, Arc::new(Field::new(col_name, DataType::Float32, false))))
                },
                8 => {
                    return Ok((8, Arc::new(Field::new(col_name, DataType::Float64, false))))
                },
                _ => {return Err(ZarrError::InvalidMetadata("Invalid data type".to_string()))}
            }
        },
        ZarrDataType::FixedLengthString(s) => {
            return Ok((*s, Arc::new(Field::new(col_name, DataType::Utf8, false))))
        },
        ZarrDataType::FixedLengthPyUnicode(s) => {
            return Ok((*s, Arc::new(Field::new(col_name, DataType::Utf8, false))))
        }
        ZarrDataType::TimeStamp(8, u) => {
            match u.as_str() {
                "s" => {
                    return Ok((
                        8, 
                        Arc::new(Field::new(
                            col_name,
                            DataType::Timestamp(TimeUnit::Second, None),
                            false,
                        )
                    )))
                },
                "ms" => {
                    return Ok((
                        8, 
                        Arc::new(Field::new(
                            col_name,
                            DataType::Timestamp(TimeUnit::Millisecond, None),
                            false,
                        )
                    )))
                },
                "us" => {
                    return Ok((
                        8, 
                        Arc::new(Field::new(
                            col_name,
                            DataType::Timestamp(TimeUnit::Microsecond, None),
                            false,
                        )
                    )))
                },
                "ns" => {
                    return Ok((
                        8, 
                        Arc::new(Field::new(
                            col_name,
                            DataType::Timestamp(TimeUnit::Nanosecond, None),
                            false,
                        )
                    )))
                }
                _ => {return Err(ZarrError::InvalidMetadata("Invalid data type".to_string()))}
            }
        },
        _ =>  {return Err(ZarrError::InvalidMetadata("Invalid data type".to_string()))}
    }
}

fn build_array(buf: Vec<u8>, t: &DataType, s: usize) -> ZarrResult<ArrayRef> {
    let data = match t {
        DataType::Utf8 => {
            ArrayData::builder(t.clone())
            .len(buf.len() / s)
            .add_buffer(
                Buffer::from(
                    (0..=buf.len()).step_by(s).map(|x| x as i32).collect::<Vec<i32>>().to_byte_slice()
                )
            )
            .add_buffer(Buffer::from(buf))
            .build()?
        },
        DataType::Boolean => {
            let bool_buf: Vec<bool> = buf.iter().map(|x| *x != 0).collect();
            return Ok(Arc::new(BooleanArray::from(bool_buf)))
        }
        _ => {
            ArrayData::builder(t.clone())
            .len(buf.len() / s)
            .add_buffer(Buffer::from(buf))
            .build()?
        }
    };

    match t {
        //DataType::Boolean => {return Ok(Arc::new(BooleanArray::from(data)))},
        DataType::UInt8 => return Ok(Arc::new(UInt8Array::from(data))),
        DataType::UInt16 => return Ok(Arc::new(UInt16Array::from(data))),
        DataType::UInt32 => return Ok(Arc::new(UInt32Array::from(data))),
        DataType::UInt64 => return Ok(Arc::new(UInt64Array::from(data))),
        DataType::Int8 => return Ok(Arc::new(Int8Array::from(data))),
        DataType::Int16 => return Ok(Arc::new(Int16Array::from(data))),
        DataType::Int32 => return Ok(Arc::new(Int32Array::from(data))),
        DataType::Int64 => return Ok(Arc::new(Int64Array::from(data))),
        DataType::Float32 => return Ok(Arc::new(Float32Array::from(data))),
        DataType::Float64 => return Ok(Arc::new(Float64Array::from(data))),
        DataType::Utf8 => return Ok(Arc::new(StringArray::from(data))),
        DataType::Timestamp(TimeUnit::Second, None) => {
            return Ok(Arc::new(TimestampSecondArray::from(data)))
        },
        DataType::Timestamp(TimeUnit::Millisecond, None) => {
            return Ok(Arc::new(TimestampMillisecondArray::from(data)))
        },
        DataType::Timestamp(TimeUnit::Microsecond, None) => {
            return Ok(Arc::new(TimestampMicrosecondArray::from(data)))
        },
        DataType::Timestamp(TimeUnit::Nanosecond, None) => {
            return Ok(Arc::new(TimestampNanosecondArray::from(data)))
        },
        _ => Err(ZarrError::InvalidMetadata("Invalid zarr datatype".to_string()))
    }
}

fn decompress_blosc(chunk_data: &[u8], output: &mut [u8]) -> Result<(), ZarrError> {
    output.copy_from_slice(unsafe { &blosc::decompress_bytes(chunk_data).unwrap() });
    Ok(())
}

fn decompress_zlib(chunk_data: &[u8], output: &mut [u8]) -> Result<(), ZarrError> {
    let mut z = flate2::read::ZlibDecoder::new(chunk_data);
    z.read(output).unwrap();
    Ok(())
}

fn decompress_bz2(chunk_data: &[u8], output: &mut [u8]) -> Result<(), ZarrError> {
    bzip2::Decompress::new(false)
        .decompress(chunk_data, output)
        .unwrap();
    Ok(())
}

fn decompress_lzma(chunk_data: &[u8], output: &mut [u8]) -> Result<(), ZarrError> {
    let decomp_data = lzma::decompress(chunk_data).unwrap();
    output.copy_from_slice(&decomp_data[..]);
    Ok(())
}
fn decompress_array(
    raw_data: Vec<u8>, uncompressed_size: usize, compressor_params: Option<&CompressorType>,
) -> Vec<u8> {
    if let Some(comp) = compressor_params {
        let mut output: Vec<u8> = vec![0; uncompressed_size];
        match comp {
            CompressorType::Zlib => {
                decompress_zlib(&raw_data, &mut output).unwrap();
            }
            CompressorType::Bz2 => {
                decompress_bz2(&raw_data, &mut output).unwrap();
            }
            CompressorType::Lzma => {
                decompress_lzma(&raw_data, &mut output).unwrap();
            }
            CompressorType::Blosc => {
                decompress_blosc(&raw_data, &mut output).unwrap();
            }
        }
        return output;
    }
    else {
        return raw_data;
    }
}

fn get_2d_dim_order(order: &MatrixOrder) -> [usize; 2] {
    match order {
        MatrixOrder::RowMajor => [0, 1],
        MatrixOrder::ColumnMajor => [1, 0],
    }
}

fn get_3d_dim_order(order: &MatrixOrder) -> [usize; 3] {
    match order {
        MatrixOrder::RowMajor => [0, 1, 2],
        MatrixOrder::ColumnMajor => [2, 1, 0],
    }
}

fn move_indices_to_front(buf: &mut [u8], indices: &Vec<usize>, data_size: usize) {
    let mut output_idx = 0;
    for data_idx in indices {
        buf.copy_within(
            data_idx * data_size..(data_idx + 1) * data_size,
            output_idx * data_size
        );
        output_idx += 1;
    }
}

fn process_edge_chunk(
    buf: &mut [u8],
    chunk_dims: &Vec<usize>,
    real_dims: &Vec<usize>,
    data_size: usize,
    order: &MatrixOrder,
) {
    let indices_to_keep: Vec<usize>;

    let n_dims = chunk_dims.len();
    indices_to_keep = match n_dims {
        1 => {(0..real_dims[0]).collect()},
        2 => {
            let [first_dim, second_dim] = get_2d_dim_order(order);
            (0..real_dims[first_dim])
                .cartesian_product(0..real_dims[second_dim])
                .map(|t| t.0 * chunk_dims[1] + t.1)
                .collect()
        },
        3 => {
            let [first_dim, second_dim, third_dim] = get_3d_dim_order(order);
            (0..real_dims[first_dim])
            .cartesian_product(0..real_dims[second_dim])
            .cartesian_product(0..real_dims[third_dim])
            .map(|t| {
                t.0 .0 * chunk_dims[1] * chunk_dims[2]
                + t.0 .1 * chunk_dims[2]
                + t.1
            })
            .collect()
        },
        _ => {panic!("Edge chunk with more than 3 domensions, 3 is the limit")}
    };

   move_indices_to_front(buf, &indices_to_keep, data_size);
}

/// A struct to read all the requested content from a zarr store, through the implementation
/// of the [`Iterator`] trait, with [`Item = ZarrResult<RecordBatch>`]. Can only be created
/// through a [`ZarrRecordBatchReaderBuilder`]. The data is read synchronously.
/// 
/// For an async API see [`crate::async_reader::ZarrRecordBatchStream`].
pub struct ZarrRecordBatchReader<T: ZarrIterator> 
{
    meta: ZarrStoreMetadata,
    zarr_store: Option<T>,
    filter: Option<ZarrChunkFilter>,
    predicate_projection_store: Option<T>,
    row_mask: Option<BooleanArray>,
}

impl<T: ZarrIterator> ZarrRecordBatchReader<T>
{
    pub(crate) fn new(
        meta: ZarrStoreMetadata,
        zarr_store: Option<T>,
        filter: Option<ZarrChunkFilter>,
        predicate_projection_store: Option<T>
    ) -> Self {
        Self {meta, zarr_store, filter, predicate_projection_store, row_mask: None}
    }

    pub(crate) fn with_row_mask(self, row_mask: BooleanArray) -> Self {
        Self {row_mask: Some(row_mask), ..self}
    }

    pub(crate) fn unpack_chunk(
        &self, mut chunk: ZarrInMemoryChunk, final_indices: Option<&Vec<usize>>
    ) -> ZarrResult<RecordBatch> {
        let mut arrs: Vec<ArrayRef> = Vec::with_capacity(self.meta.get_num_columns());
        let mut fields: Vec<FieldRef> = Vec::with_capacity(self.meta.get_num_columns());

        // the sort below is important, because within a zarr store, the different arrays are
        // not ordered, so there is no predefined order for the different columns. we effectively
        // define one here, my ordering the columns alphabetically.
        let mut cols = chunk.get_cols_in_chunk();
        cols.sort();

        for col in cols {
            let data = chunk.take_array(&col)?;
            let (arr, field) = self.unpack_array_chunk(
                col.to_string(), data, chunk.get_real_dims(), self.meta.get_chunk_dims(), final_indices
            )?;
            arrs.push(arr);
            fields.push(field);
        }

        Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), arrs)?)
    }

    fn unpack_array_chunk(
        &self,
        col_name: String,
        arr_chnk: ZarrInMemoryArray,
        real_dims: &Vec<usize>,
        chunk_dims: &Vec<usize>,
        final_indices: Option<&Vec<usize>>,
    ) -> ZarrResult<(ArrayRef, FieldRef)> {
            // get the metadata for the array
            let meta = self.meta.get_array_meta(&col_name)?;

            // get the field, data size and the data raw data from the array
            let (mut data_size, field) = build_field(meta.get_type(), col_name)?;
            let mut data = arr_chnk.take_data();

            // uncompress the data
            let chunk_size = chunk_dims.iter().fold(1, |mult, x| mult * x);
            data = decompress_array(data, chunk_size * data_size, meta.get_compressor().as_ref());

            // handle big endianness by converting to little endianness
            if meta.get_endianness() == &Endianness::Big {
                for idx in 0..chunk_size {
                    data[idx*data_size..(idx+1)*data_size].reverse();
                }
            }

            // handle edge chunks
            if chunk_dims != real_dims {
                process_edge_chunk(&mut data, chunk_dims, real_dims, data_size, meta.get_order());
            }

            // special case of Py Unicode, with 4 byte characters. Here we simply
            // keep one byte, might need to be more robust, perhaps throw an error
            // if the other 3 bytes are not 0s.
            if let ZarrDataType::FixedLengthPyUnicode(_) = meta.get_type() {
                data = data.iter().step_by(PY_UNICODE_SIZE).copied().collect();
                data_size = data_size / PY_UNICODE_SIZE;
            }

            // create the array
            let real_size = real_dims.iter().fold(1, |mult, x| mult * x) * data_size;
            data.resize(real_size, 0);

            // select the final indices in the data
            if let Some(final_indices) = final_indices {
                move_indices_to_front(&mut data, final_indices, data_size);
                data.resize(final_indices.len() * data_size, 0);
            }

            let arr = build_array(data, field.data_type(), data_size)?;

            Ok((arr, field))
    }
}

/// The [`Iterator`] trait implementation for a [`ZarrRecordBatchReader`]. Provides the interface
/// through which the record batches can be retrieved.
impl<T: ZarrIterator> Iterator for ZarrRecordBatchReader<T>
{
    type Item = ZarrResult<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        // handle filters first. 
        let mut bool_arr: Option<BooleanArray> = self.row_mask.clone();
        if let Some(store) = self.predicate_projection_store.as_mut() {
            let predicate_proj_chunk = store.next_chunk();
            if predicate_proj_chunk.is_none() {
                return None
            }

            let predicate_proj_chunk = unwrap_or_return!(predicate_proj_chunk.unwrap());
            let predicate_rec = self.unpack_chunk(predicate_proj_chunk, None);
            let predicate_rec = unwrap_or_return!(predicate_rec);
            
            if let Some(filter) = self.filter.as_mut() {
                for predicate in filter.predicates.iter_mut() {
                    let mask = predicate.evaluate(&predicate_rec);
                    let mask = unwrap_or_return!(mask);

                    if let Some(old_bool_arr) = bool_arr {
                        bool_arr = Some(BooleanArray::from(
                            old_bool_arr
                                .iter()
                                .zip(mask.iter())
                                .map(|(x, y)| x.unwrap() && y.unwrap())
                                .collect_vec()
                        ));
                    } else {
                        bool_arr = Some(mask);
                    }
                }

                // if there is no store provided, other than the one to evaluate the 
                // predicates, then this call to next is only meant to evaluate the predicate
                // by itself, and we return a record batch for the results.
                if self.zarr_store.is_none() {
                    let rec = RecordBatch::try_new(
                        Arc::new(Schema::new(vec![Field::new("mask", DataType::Boolean, false)])),
                        vec![Arc::new(bool_arr.unwrap())]
                    );
                    let rec = unwrap_or_return!(rec);
                    return Some(Ok(rec));
                }

                // this is the case where we've determined that no row in the chunk will
                // satisfy the filer condition(s), so we skip the chunk and move on to 
                // the next one by calling next().
                if bool_arr.as_ref().unwrap().true_count() == 0 {
                    // since we checked above that self.zarr_store is not None, we can simply
                    // unwrap it here.
                    self.zarr_store.as_mut().unwrap().skip_chunk();
                    return self.next();
                }
            }
            else {
                return Some(Err(ZarrError::InvalidPredicate("No filters found".to_string())))
            }
            
        }

        if self.zarr_store.is_none() {
            return Some(
                Err(
                    ZarrError::MissingArray("No zarr store provided in zarr record batch reader".to_string())
                )
            );
        }

         // main logic for the chunk
         let next_batch = self.zarr_store.as_mut().unwrap().next_chunk();
         if next_batch.is_none() {
             return None
         }
 
         let next_batch = unwrap_or_return!(next_batch.unwrap());
         let mut final_indices: Option<Vec<usize>> = None;

         // if we have a bool array to mask some values, we get the indices (the rows)
         // that we need to keep. those are then applied across all zarr array un the chunk.
         if let Some(mask) = bool_arr {
             let mask = mask.values();
             final_indices = Some(mask.iter().enumerate().filter(|x| x.1).map(|x| x.0).collect());
         }

         return Some(self.unpack_chunk(next_batch, final_indices.as_ref()));
    }
}

/// A builder used to construct a [`ZarrRecordBatchReader`] for a folder containing a
/// zarr store. 
/// 
/// To build the equivalent asynchronous reader see [`crate::async_reader::ZarrRecordBatchStreamBuilder`].
pub struct ZarrRecordBatchReaderBuilder<T: ZarrRead + Clone> 
{
    zarr_reader: T,
    projection: ZarrProjection,
    filter: Option<ZarrChunkFilter>,
}

impl<T: ZarrRead + Clone> ZarrRecordBatchReaderBuilder<T> {
    /// Create a [`ZarrRecordBatchReaderBuilder`] from a [`ZarrRead`] struct. 
    pub fn new(zarr_reader: T) -> Self {
        Self{zarr_reader, projection: ZarrProjection::all(), filter: None}
    }

    /// Adds a column projection to the builder, so that the resulting reader will only
    /// read some of the columns (zarr arrays) from the zarr store.
    pub fn with_projection(self, projection: ZarrProjection) -> Self {
        Self {projection: projection, ..self}
    }

    /// Adds a row filter to the builder, so that the resulting reader will only
    /// read rows that satisfy some conditions from the zarr store.
    pub fn with_filter(self, filter: ZarrChunkFilter) -> Self {
        Self {filter: Some(filter), ..self}
    }

    /// Build a [`ZarrRecordBatchReader`], consuming the builder. The option range
    /// argument controls the start and end chunk (following the way zarr chunks are
    /// named and numbered).
    pub fn build_partial_reader(self, chunk_range: Option<(usize, usize)>) -> ZarrResult<ZarrRecordBatchReader<ZarrStore<T>>> {
        let meta = self.zarr_reader.get_zarr_metadata()?;
        let mut chunk_pos: Vec<Vec<usize>> = meta.get_chunk_positions();
        if let Some(chunk_range) = chunk_range {
            if (chunk_range.0 > chunk_range.1) | (chunk_range.1 > chunk_pos.len()) {
                return Err(ZarrError::InvalidChunkRange(chunk_range.0, chunk_range.1, chunk_pos.len()))
            }
            chunk_pos = chunk_pos[chunk_range.0..chunk_range.1].to_vec();
        }

        let mut predicate_store: Option<ZarrStore<T>> = None;
        if let Some(filter) = &self.filter {
            let predicate_proj = filter.get_all_projections();
            predicate_store = Some(
                ZarrStore::new(self.zarr_reader.clone(), chunk_pos.clone(), predicate_proj.clone())?
            );
        }

        let zarr_store = ZarrStore::new(self.zarr_reader, chunk_pos, self.projection.clone())?;
        Ok(ZarrRecordBatchReader::new(meta, Some(zarr_store), self.filter, predicate_store))
    }

    /// Build a [`ZarrRecordBatchReader`], consuming the builder. The resulting reader
    /// will read all the chunks in the zarr store.
    pub fn build(self) -> ZarrResult<ZarrRecordBatchReader<ZarrStore<T>>> {
        self.build_partial_reader(None)
    }
}

#[cfg(test)]
mod zarr_reader_tests {
    use arrow_array::cast::AsArray;
    use arrow_array::types::*;
    use arrow_schema::{DataType, TimeUnit};
    use itertools::enumerate;
    use arrow::compute::kernels::cmp::{gt_eq, lt};
    use std::{path::PathBuf, collections::HashMap, fmt::Debug, boxed::Box};

    use super::*;
    use crate::reader::filters::{ZarrArrowPredicateFn, ZarrArrowPredicate};

    fn get_test_data_path(zarr_store: String) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testing/data/zarr").join(zarr_store)
    }

    fn validate_names_and_types(targets: &HashMap<String, DataType>, rec: &RecordBatch) {
        let mut target_cols: Vec<&String> = targets.keys().collect();
        let schema = rec.schema();
        let mut from_rec: Vec<&String> = schema.fields.iter().map(|f| f.name()).collect();

        from_rec.sort();
        target_cols.sort();
        assert_eq!(from_rec, target_cols);

        for field in schema.fields.iter() {
            assert_eq!(
                field.data_type(),
                targets.get(field.name()).unwrap()
            );
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
        U: Debug
    {
        let mut matched = false;
        for (idx, col) in enumerate(rec.schema().fields.iter()) {
            if col.name().as_str() == col_name {
                assert_eq!(
                    rec.column(idx).as_primitive::<T>().values(),
                    targets,
                );
                matched = true;
            }
        }
        assert!(matched);
    }

    fn validate_string_column(col_name: &str, rec: &RecordBatch, targets: &[&str]) {
        let mut matched = false;
        for (idx, col) in enumerate(rec.schema().fields.iter()) {
            if col.name().as_str() == col_name {
                assert_eq!(
                    rec.column(idx).as_string(),
                    &StringArray::from(targets.to_vec()),
                );
                matched = true;
            }
        }
        assert!(matched);
    }

    #[test]
    fn compression_tests() {
        let p = get_test_data_path("compression_example.zarr".to_string());
        let reader = ZarrRecordBatchReaderBuilder::new(p).build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([
            ("bool_data".to_string(), DataType::Boolean),
            ("uint_data".to_string(), DataType::UInt64),
            ("int_data".to_string(), DataType::Int64),
            ("float_data".to_string(), DataType::Float64),
            ("float_data_no_comp".to_string(), DataType::Float64),
        ]);


        // center chunk
        let rec = &records[4];
        validate_names_and_types(&target_types, rec);
        validate_bool_column(&"bool_data", rec, &[false, true, false, false, true, false, false, true, false]);
        validate_primitive_column::<Int64Type, i64>(&"int_data", rec, &[-4, -3, -2, 4, 5, 6, 12, 13, 14]);
        validate_primitive_column::<UInt64Type, u64>(&"uint_data", rec, &[27, 28, 29, 35, 36, 37, 43, 44, 45]);
        validate_primitive_column::<Float64Type, f64>(
            &"float_data", rec, &[127., 128., 129., 135., 136., 137., 143., 144., 145.]
        );
        validate_primitive_column::<Float64Type, f64>(
            &"float_data_no_comp", rec, &[227., 228., 229., 235., 236., 237., 243., 244., 245.]
        );

        // right edge chunk
        let rec = &records[5];
        validate_names_and_types(&target_types, rec);
        validate_bool_column(&"bool_data", rec, &[true, false, true, false, true, false]);
        validate_primitive_column::<Int64Type, i64>(&"int_data", rec, &[-1, 0, 7, 8, 15, 16]);
        validate_primitive_column::<UInt64Type, u64>(&"uint_data", rec, &[30, 31, 38, 39, 46, 47]);
        validate_primitive_column::<Float64Type, f64>(
            &"float_data", rec, &[130., 131., 138., 139., 146., 147.]
        );
        validate_primitive_column::<Float64Type, f64>(
            &"float_data_no_comp", rec, &[230., 231., 238., 239., 246., 247.]
        );

        // bottom right edge chunk
        let rec = &records[8];
        validate_names_and_types(&target_types, rec);
        validate_bool_column(&"bool_data", rec, &[true, false, true, false]);
        validate_primitive_column::<Int64Type, i64>(&"int_data", rec, &[23, 24, 31, 32]);
        validate_primitive_column::<UInt64Type, u64>(&"uint_data", rec, &[54, 55, 62, 63]);
        validate_primitive_column::<Float64Type, f64>(&"float_data", rec, &[154.0, 155.0, 162.0, 163.0]);
        validate_primitive_column::<Float64Type, f64>(&"float_data_no_comp", rec, &[254.0, 255.0, 262.0, 263.0]);
    }

    #[test]
    fn projection_tests() {
        let p = get_test_data_path("compression_example.zarr".to_string());
        let proj = ZarrProjection::keep(vec!["bool_data".to_string(), "int_data".to_string()]);
        let builder = ZarrRecordBatchReaderBuilder::new(p).with_projection(proj);
        let reader = builder.build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([
            ("bool_data".to_string(), DataType::Boolean),
            ("int_data".to_string(), DataType::Int64),
        ]);

        // center chunk
        let rec = &records[4];
        validate_names_and_types(&target_types, rec);
        validate_bool_column(&"bool_data", rec, &[false, true, false, false, true, false, false, true, false]);
        validate_primitive_column::<Int64Type, i64>(&"int_data", rec, &[-4, -3, -2, 4, 5, 6, 12, 13, 14]);
    }

    #[test]
    fn multiple_readers_tests() {
        let p = get_test_data_path("compression_example.zarr".to_string());
        let reader1 = ZarrRecordBatchReaderBuilder::new(p.clone()).build_partial_reader(Some((0, 5))).unwrap();
        let reader2 = ZarrRecordBatchReaderBuilder::new(p).build_partial_reader(Some((5, 9))).unwrap();

        let handle1 = std::thread::spawn(move || reader1.map(|x| x.unwrap()).collect());
        let handle2 = std::thread::spawn(move || reader2.map(|x| x.unwrap()).collect());

        let records1: Vec<RecordBatch> = handle1.join().unwrap();
        let records2: Vec<RecordBatch> = handle2.join().unwrap();

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
        validate_bool_column(&"bool_data", rec, &[false, true, false, false, true, false, false, true, false]);
        validate_primitive_column::<Int64Type, i64>(&"int_data", rec, &[-4, -3, -2, 4, 5, 6, 12, 13, 14]);
        validate_primitive_column::<UInt64Type, u64>(&"uint_data", rec, &[27, 28, 29, 35, 36, 37, 43, 44, 45]);
        validate_primitive_column::<Float64Type, f64>(
            &"float_data", rec, &[127., 128., 129., 135., 136., 137., 143., 144., 145.]
        );
        validate_primitive_column::<Float64Type, f64>(
            &"float_data_no_comp", rec, &[227., 228., 229., 235., 236., 237., 243., 244., 245.]
        );

        // bottom edge chunk
        let rec = &records2[2];
        validate_names_and_types(&target_types, rec);
        validate_bool_column(&"bool_data", rec, &[false, true, false, false, true, false]);
        validate_primitive_column::<Int64Type, i64>(&"int_data", rec, &[20, 21, 22, 28, 29, 30]);
        validate_primitive_column::<UInt64Type, u64>(&"uint_data", rec, &[51, 52, 53, 59, 60, 61]);
        validate_primitive_column::<Float64Type, f64>(
            &"float_data", rec, &[151.0, 152.0, 153.0, 159.0, 160.0, 161.0]
        );
        validate_primitive_column::<Float64Type, f64>(
            &"float_data_no_comp", rec, &[251.0, 252.0, 253.0, 259.0, 260.0, 261.0]
        );
    }

    #[test]
    fn endianness_and_order_tests() {
        let p = get_test_data_path("endianness_and_order_example.zarr".to_string());
        let reader = ZarrRecordBatchReaderBuilder::new(p).build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([
            ("var1".to_string(), DataType::Int32),
            ("var2".to_string(), DataType::Int32),
        ]);

        // bottom edge chunk
        let rec = &records[9];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Int32Type, i32>(&"var1", rec, &[69, 80, 91, 70, 81, 92, 71, 82, 93]);
        validate_primitive_column::<Int32Type, i32>(&"var2", rec, &[69, 80, 91, 70, 81, 92, 71, 82, 93]);
    }

    #[test]
    fn string_data_tests() {
        let p = get_test_data_path("string_example.zarr".to_string());
        let reader = ZarrRecordBatchReaderBuilder::new(p).build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([
            ("uint_data".to_string(), DataType::UInt8),
            ("string_data".to_string(), DataType::Utf8),
            ("utf8_data".to_string(), DataType::Utf8),
        ]);

        // bottom edge chunk
        let rec = &records[7];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<UInt8Type, u8>(&"uint_data", rec, &[51, 52, 53, 59, 60, 61]);
        validate_string_column("string_data", rec, &["abc61", "abc62", "abc63", "abc69", "abc70", "abc71"]);
        validate_string_column("utf8_data", rec, &["def61", "def62", "def63", "def69", "def70", "def71"]);
    }

    #[test]
    fn ts_data_tests() {
        let p = get_test_data_path("ts_example.zarr".to_string());
        let reader = ZarrRecordBatchReaderBuilder::new(p).build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([
            ("ts_s_data".to_string(), DataType::Timestamp(TimeUnit::Second, None)),
            ("ts_ms_data".to_string(), DataType::Timestamp(TimeUnit::Millisecond, None)),
            ("ts_us_data".to_string(), DataType::Timestamp(TimeUnit::Microsecond, None)),
            ("ts_ns_data".to_string(), DataType::Timestamp(TimeUnit::Nanosecond, None)),
        ]);

        // top center chunk
        let rec = &records[1];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<TimestampSecondType, i64>(
            &"ts_s_data", rec, &[1685750400, 1685836800, 1686182400, 1686268800]
        );
        validate_primitive_column::<TimestampMillisecondType, i64>(
            &"ts_ms_data", rec, &[1685750400000, 1685836800000, 1686182400000, 1686268800000]
        );
        validate_primitive_column::<TimestampMicrosecondType, i64>(
            &"ts_us_data", rec, &[1685750400000000, 1685836800000000, 1686182400000000, 1686268800000000]
        );
        validate_primitive_column::<TimestampNanosecondType, i64>(
            &"ts_ns_data", rec, &[1685750400000000000, 1685836800000000000, 1686182400000000000, 1686268800000000000]
        );

        // top right edge chunk
        let rec = &records[2];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<TimestampSecondType, i64>(
            &"ts_s_data", rec, &[1685923200, 1686355200]
        );
        validate_primitive_column::<TimestampMillisecondType, i64>(
            &"ts_ms_data", rec, &[1685923200000, 1686355200000]
        );
        validate_primitive_column::<TimestampMicrosecondType, i64>(
            &"ts_us_data", rec, &[1685923200000000, 1686355200000000]
        );
        validate_primitive_column::<TimestampNanosecondType, i64>(
            &"ts_ns_data", rec, &[1685923200000000000, 1686355200000000000]
        );
    }

    #[test]
    fn one_dim_tests() {
        let p = get_test_data_path("one_dim_example.zarr".to_string());
        let reader = ZarrRecordBatchReaderBuilder::new(p).build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([
            ("int_data".to_string(), DataType::Int64),
            ("float_data".to_string(), DataType::Float64),
        ]);

        // center chunk
        let rec = &records[1];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Int64Type, i64>(&"int_data", rec, &[-2, -1, 0]);
        validate_primitive_column::<Float64Type, f64>(&"float_data", rec, &[103.0, 104.0, 105.0]);

        // right edge chunk
        let rec = &records[3];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Int64Type, i64>(&"int_data", rec, &[4, 5]);
        validate_primitive_column::<Float64Type, f64>(&"float_data", rec, &[109.0, 110.0]);
    }

    #[test]
    fn three_dim_tests() {
        let p = get_test_data_path("three_dim_example.zarr".to_string());
        let reader = ZarrRecordBatchReaderBuilder::new(p).build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([
            ("int_data".to_string(), DataType::Int64),
            ("float_data".to_string(), DataType::Float64),
        ]);

        // center chunk
        let rec = &records[13];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Int64Type, i64>(&"int_data", rec, &[0, 1, 5, 6, 25, 26, 30, 31]);
        validate_primitive_column::<Float64Type, f64>(
            &"float_data", rec, &[162.0, 163.0, 167.0, 168.0, 187.0, 188.0, 192.0, 193.0]
        );

        // right edge chunk
        let rec = &records[14];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Int64Type, i64>(&"int_data", rec, &[2, 7, 27, 32]);
        validate_primitive_column::<Float64Type, f64>(&"float_data", rec, &[164.0, 169.0, 189.0, 194.0]);

        
        // right front edge chunk
        let rec = &records[23];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Int64Type, i64>(&"int_data", rec, &[52, 57]);
        validate_primitive_column::<Float64Type, f64>(&"float_data", rec, &[214.0, 219.0]);

        // bottom front edge chunk
        let rec = &records[24];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Int64Type, i64>(&"int_data", rec, &[58, 59]);
        validate_primitive_column::<Float64Type, f64>(&"float_data", rec, &[220.0, 221.0]);

        // right front bottom edge chunk
        let rec = &records[26];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Int64Type, i64>(&"int_data", rec, &[62]);
        validate_primitive_column::<Float64Type, f64>(&"float_data", rec, &[224.0]);
    }

    #[test]
    fn filters_tests() {
        let p = get_test_data_path("lat_lon_example.zarr".to_string());
        let mut builder = ZarrRecordBatchReaderBuilder::new(p);

        // set the filters to select part of the raster, based on lat and
        // lon coordinates.
        let mut filters: Vec<Box<dyn ZarrArrowPredicate>> = Vec::new();
        let f = ZarrArrowPredicateFn::new(
            ZarrProjection::keep(vec!["lat".to_string()]),
            move |batch| (
                gt_eq(batch.column_by_name("lat").unwrap(), &Scalar::new(&Float64Array::from(vec![38.6])))
            ),
        );
        filters.push(Box::new(f));
        let f = ZarrArrowPredicateFn::new(
            ZarrProjection::keep(vec!["lon".to_string()]),
            move |batch| (
                gt_eq(batch.column_by_name("lon").unwrap(), &Scalar::new(&Float64Array::from(vec![-109.7])))
            ),
        );
        filters.push(Box::new(f));
        let f = ZarrArrowPredicateFn::new(
            ZarrProjection::keep(vec!["lon".to_string()]),
            move |batch| (
               lt(batch.column_by_name("lon").unwrap(), &Scalar::new(&Float64Array::from(vec![-109.2])))
            ),
        );
        filters.push(Box::new(f));

        builder = builder.with_filter(ZarrChunkFilter::new(filters));
        let reader = builder.build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        // check the 4 chunks that have some in the specified lat/lon range
        // center chunk
        let target_types = HashMap::from([
            ("lat".to_string(), DataType::Float64),
            ("lon".to_string(), DataType::Float64),
            ("float_data".to_string(), DataType::Float64),
        ]);

        let rec = &records[0];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Float64Type, f64>(&"lat", rec, &[38.6, 38.7]);
        validate_primitive_column::<Float64Type, f64>(&"lon", rec, &[-109.7, -109.7]);
        validate_primitive_column::<Float64Type, f64>(&"float_data", rec, &[1040.0, 1041.0]);

        let rec = &records[1];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Float64Type, f64>(&"lat", rec, &[38.8, 38.9, 39.0]);
        validate_primitive_column::<Float64Type, f64>(&"lon", rec, &[-109.7, -109.7, -109.7]);
        validate_primitive_column::<Float64Type, f64>(&"float_data", rec, &[1042.0, 1043.0, 1044.0]);

        let rec = &records[2];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Float64Type, f64>(
            &"lat", rec, &[38.6, 38.7, 38.6, 38.7, 38.6, 38.7, 38.6, 38.7]
        );
        validate_primitive_column::<Float64Type, f64>(
            &"lon", rec, &[-109.6, -109.6, -109.5, -109.5, -109.4, -109.4, -109.3, -109.3]
        );
        validate_primitive_column::<Float64Type, f64>(
            &"float_data", rec, &[1051.0, 1052.0, 1062.0, 1063.0, 1073.0, 1074.0, 1084.0, 1085.0]
        );

        let rec = &records[3];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Float64Type, f64>(
            &"lat", rec, &[38.8, 38.9, 39.0, 38.8, 38.9, 39.0, 38.8, 38.9, 39.0, 38.8, 38.9, 39.0]
        );
        validate_primitive_column::<Float64Type, f64>(
            &"lon", rec, &[-109.6, -109.6, -109.6, -109.5, -109.5, -109.5, -109.4, -109.4, -109.4, -109.3, -109.3, -109.3]
        );
        validate_primitive_column::<Float64Type, f64>(
            &"float_data", rec, &[1053.0, 1054.0, 1055.0, 1064.0, 1065.0, 1066.0, 1075.0, 1076.0, 1077.0, 1086.0, 1087.0, 1088.0]
        );
    }

    #[test]
    fn empty_query_tests() {
        let p = get_test_data_path("lat_lon_example.zarr".to_string());
        let mut builder = ZarrRecordBatchReaderBuilder::new(p);

        // set a filter that will filter out all the data, there should be nothing left after
        // we apply it.
        let mut filters: Vec<Box<dyn ZarrArrowPredicate>> = Vec::new();
        let f = ZarrArrowPredicateFn::new(
            ZarrProjection::keep(vec!["lat".to_string()]),
            move |batch| (
                gt_eq(batch.column_by_name("lat").unwrap(), &Scalar::new(&Float64Array::from(vec![100.0])))
            ),
        );
        filters.push(Box::new(f));

        builder = builder.with_filter(ZarrChunkFilter::new(filters));
        let reader = builder.build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        // there should be no records, because of the filter.
        assert_eq!(records.len(), 0);
    }
}