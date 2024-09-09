//! A module tha provides a sychronous reader for zarr store, to generate [`RecordBatch`]es.
//!
//! ```
//! # use arrow_zarr::reader::{ZarrRecordBatchReaderBuilder, ZarrProjection};
//! # use arrow_cast::pretty::pretty_format_batches;
//! # use arrow_array::RecordBatch;
//! # use std::path::PathBuf;
//! #
//! # fn get_test_data_path(zarr_store: String) -> PathBuf {
//! #    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-data/data/zarr/v2_data").join(zarr_store)
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

use arrow_array::*;
use arrow_schema::{DataType, Field, FieldRef, Schema};
use itertools::Itertools;
use std::sync::Arc;

use codecs::apply_codecs;
pub use errors::{ZarrError, ZarrResult};
pub use filters::{ZarrArrowPredicate, ZarrArrowPredicateFn, ZarrChunkFilter};
pub use metadata::ZarrStoreMetadata;
use zarr_read::{ZarrInMemoryArray, ZarrRead};
pub use zarr_read::{ZarrInMemoryChunk, ZarrProjection};

mod errors;
mod filters;
mod zarr_read;

pub(crate) mod codecs;
pub(crate) mod metadata;

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
    };
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
            pos,
            &cols,
            self.meta.get_real_dims(pos),
            self.meta.get_chunk_patterns(),
            self.meta.get_one_dim_repr_meta(),
        );
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

/// A struct to read all the requested content from a zarr store, through the implementation
/// of the [`Iterator`] trait, with [`Item = ZarrResult<RecordBatch>`]. Can only be created
/// through a [`ZarrRecordBatchReaderBuilder`]. The data is read synchronously.
///
/// For an async API see [`crate::async_reader::ZarrRecordBatchStream`].
pub struct ZarrRecordBatchReader<T: ZarrIterator> {
    meta: ZarrStoreMetadata,
    zarr_store: Option<T>,
    filter: Option<ZarrChunkFilter>,
    predicate_projection_store: Option<T>,
}

impl<T: ZarrIterator> ZarrRecordBatchReader<T> {
    pub(crate) fn new(
        meta: ZarrStoreMetadata,
        zarr_store: Option<T>,
        filter: Option<ZarrChunkFilter>,
        predicate_projection_store: Option<T>,
    ) -> Self {
        Self {
            meta,
            zarr_store,
            filter,
            predicate_projection_store,
        }
    }

    pub(crate) fn unpack_chunk(&self, mut chunk: ZarrInMemoryChunk) -> ZarrResult<RecordBatch> {
        let mut arrs: Vec<ArrayRef> = Vec::with_capacity(self.meta.get_num_columns());
        let mut fields: Vec<FieldRef> = Vec::with_capacity(self.meta.get_num_columns());

        let cols = chunk.get_cols_in_chunk();
        for col in cols {
            let data = chunk.take_array(&col)?;
            let (arr, field) = self.unpack_array_chunk(
                col.to_string(),
                data,
                chunk.get_real_dims(),
                self.meta.get_chunk_dims(),
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
        chunk_dims: &[usize],
    ) -> ZarrResult<(ArrayRef, FieldRef)> {
        // check if this array has a one dimensional representation
        let one_dim_params = self.meta.get_one_dim_repr_meta().get(&col_name);

        // the logic here can be a bit confusing. the function arguments
        // correspond to the real chunk, which may or may not have a
        // 1D representation. so here we pick the meta and dimensions
        // depending on the situation. if there is a 1D representation,
        // we still need to pass the dimensions of the real chunk to the
        // [`apply_codecs`] function so that the 1D reprensation can be
        // projected to obrain the real chunk.
        let (meta, real_dims, chunk_dims, proj_params) =
            if let Some((pos, _, one_dim_meta)) = one_dim_params {
                (
                    one_dim_meta,
                    vec![real_dims[*pos]],
                    vec![chunk_dims[*pos]],
                    Some((*pos, real_dims)),
                )
            } else {
                (
                    self.meta.get_array_meta(&col_name)?,
                    real_dims.clone(),
                    chunk_dims.to_vec(),
                    None,
                )
            };

        // take the raw data from the chunk
        let data = arr_chnk.take_data();

        // apply codecs and decode the raw data
        let (arr, field) = apply_codecs(
            col_name,
            data,
            &chunk_dims,
            &real_dims,
            meta.get_type(),
            meta.get_codecs(),
            meta.get_sharding_params(),
            proj_params,
        )?;

        Ok((arr, field))
    }
}

/// The [`Iterator`] trait implementation for a [`ZarrRecordBatchReader`]. Provides the interface
/// through which the record batches can be retrieved.
impl<T: ZarrIterator> Iterator for ZarrRecordBatchReader<T> {
    type Item = ZarrResult<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        // handle filters first.
        if let Some(store) = self.predicate_projection_store.as_mut() {
            let predicate_proj_chunk = store.next_chunk();

            predicate_proj_chunk.as_ref()?;

            let predicate_proj_chunk = unwrap_or_return!(predicate_proj_chunk.unwrap());

            let predicate_rec = self.unpack_chunk(predicate_proj_chunk);
            let predicate_rec = unwrap_or_return!(predicate_rec);

            let mut bool_arr: Option<BooleanArray> = None;
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
                                .collect_vec(),
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
                        Arc::new(Schema::new(vec![Field::new(
                            "mask",
                            DataType::Boolean,
                            false,
                        )])),
                        vec![Arc::new(bool_arr.unwrap())],
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
            } else {
                return Some(Err(ZarrError::InvalidPredicate(
                    "No filters found".to_string(),
                )));
            }
        }

        if self.zarr_store.is_none() {
            return Some(Err(ZarrError::MissingArray(
                "No zarr store provided in zarr record batch reader".to_string(),
            )));
        }

        // main logic for the chunk
        let next_batch = self.zarr_store.as_mut().unwrap().next_chunk();
        next_batch.as_ref()?;
        let next_batch = unwrap_or_return!(next_batch.unwrap());

        Some(self.unpack_chunk(next_batch))
    }
}

/// A builder used to construct a [`ZarrRecordBatchReader`] for a folder containing a
/// zarr store.
///
/// To build the equivalent asynchronous reader see [`crate::async_reader::ZarrRecordBatchStreamBuilder`].
pub struct ZarrRecordBatchReaderBuilder<T: ZarrRead + Clone> {
    zarr_reader: T,
    projection: ZarrProjection,
    filter: Option<ZarrChunkFilter>,
}

impl<T: ZarrRead + Clone> ZarrRecordBatchReaderBuilder<T> {
    /// Create a [`ZarrRecordBatchReaderBuilder`] from a [`ZarrRead`] struct.
    pub fn new(zarr_reader: T) -> Self {
        Self {
            zarr_reader,
            projection: ZarrProjection::all(),
            filter: None,
        }
    }

    /// Adds a column projection to the builder, so that the resulting reader will only
    /// read some of the columns (zarr arrays) from the zarr store.
    pub fn with_projection(self, projection: ZarrProjection) -> Self {
        Self { projection, ..self }
    }

    /// Adds a row filter to the builder, so that the resulting reader will only
    /// read rows that satisfy some conditions from the zarr store.
    pub fn with_filter(self, filter: ZarrChunkFilter) -> Self {
        Self {
            filter: Some(filter),
            ..self
        }
    }

    /// Build a [`ZarrRecordBatchReader`], consuming the builder. The option range
    /// argument controls the start and end chunk (following the way zarr chunks are
    /// named and numbered).
    pub fn build_partial_reader(
        self,
        chunk_range: Option<(usize, usize)>,
    ) -> ZarrResult<ZarrRecordBatchReader<ZarrStore<T>>> {
        let meta = self.zarr_reader.get_zarr_metadata()?;
        let mut chunk_pos: Vec<Vec<usize>> = meta.get_chunk_positions();
        if let Some(chunk_range) = chunk_range {
            if (chunk_range.0 > chunk_range.1) | (chunk_range.1 > chunk_pos.len()) {
                return Err(ZarrError::InvalidChunkRange(
                    chunk_range.0,
                    chunk_range.1,
                    chunk_pos.len(),
                ));
            }
            chunk_pos = chunk_pos[chunk_range.0..chunk_range.1].to_vec();
        }

        let mut predicate_store: Option<ZarrStore<T>> = None;
        if let Some(filter) = &self.filter {
            let predicate_proj = filter.get_all_projections()?;
            predicate_store = Some(ZarrStore::new(
                self.zarr_reader.clone(),
                chunk_pos.clone(),
                predicate_proj.clone(),
            )?);
        }

        let zarr_store = ZarrStore::new(self.zarr_reader, chunk_pos, self.projection.clone())?;
        Ok(ZarrRecordBatchReader::new(
            meta,
            Some(zarr_store),
            self.filter,
            predicate_store,
        ))
    }

    /// Build a [`ZarrRecordBatchReader`], consuming the builder. The resulting reader
    /// will read all the chunks in the zarr store.
    pub fn build(self) -> ZarrResult<ZarrRecordBatchReader<ZarrStore<T>>> {
        self.build_partial_reader(None)
    }
}

#[cfg(test)]
mod zarr_reader_tests {
    use arrow::compute::kernels::cmp::{gt_eq, lt};
    use arrow_array::cast::AsArray;
    use arrow_array::types::*;
    use arrow_schema::{DataType, TimeUnit};
    use itertools::enumerate;
    use std::{boxed::Box, collections::HashMap, fmt::Debug};

    use super::*;
    use crate::reader::filters::{ZarrArrowPredicate, ZarrArrowPredicateFn};
    use crate::tests::{get_test_v2_data_path, get_test_v3_data_path};

    fn validate_names_and_types(targets: &HashMap<String, DataType>, rec: &RecordBatch) {
        let mut target_cols: Vec<&String> = targets.keys().collect();
        let schema = rec.schema();
        let from_rec: Vec<&String> = schema.fields.iter().map(|f| f.name()).collect();

        target_cols.sort();
        assert_eq!(from_rec, target_cols);

        for field in schema.fields.iter() {
            assert_eq!(field.data_type(), targets.get(field.name()).unwrap());
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
        U: Debug,
    {
        let mut matched = false;
        for (idx, col) in enumerate(rec.schema().fields.iter()) {
            if col.name().as_str() == col_name {
                assert_eq!(rec.column(idx).as_primitive::<T>().values(), targets);
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

    // create a test filter
    fn create_filter() -> ZarrChunkFilter {
        let mut filters: Vec<Box<dyn ZarrArrowPredicate>> = Vec::new();
        let f = ZarrArrowPredicateFn::new(
            ZarrProjection::keep(vec!["lat".to_string()]),
            move |batch| {
                gt_eq(
                    batch.column_by_name("lat").unwrap(),
                    &Scalar::new(&Float64Array::from(vec![38.6])),
                )
            },
        );
        filters.push(Box::new(f));
        let f = ZarrArrowPredicateFn::new(
            ZarrProjection::keep(vec!["lon".to_string()]),
            move |batch| {
                gt_eq(
                    batch.column_by_name("lon").unwrap(),
                    &Scalar::new(&Float64Array::from(vec![-109.7])),
                )
            },
        );
        filters.push(Box::new(f));
        let f = ZarrArrowPredicateFn::new(
            ZarrProjection::keep(vec!["lon".to_string()]),
            move |batch| {
                lt(
                    batch.column_by_name("lon").unwrap(),
                    &Scalar::new(&Float64Array::from(vec![-109.2])),
                )
            },
        );
        filters.push(Box::new(f));

        ZarrChunkFilter::new(filters)
    }

    //**************************
    // zarr format v2 tests
    //**************************

    #[test]
    fn compression_tests() {
        let p = get_test_v2_data_path("compression_example.zarr".to_string());
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
        validate_bool_column(
            "bool_data",
            rec,
            &[false, true, false, false, true, false, false, true, false],
        );
        validate_primitive_column::<Int64Type, i64>(
            "int_data",
            rec,
            &[-4, -3, -2, 4, 5, 6, 12, 13, 14],
        );
        validate_primitive_column::<UInt64Type, u64>(
            "uint_data",
            rec,
            &[27, 28, 29, 35, 36, 37, 43, 44, 45],
        );
        validate_primitive_column::<Float64Type, f64>(
            "float_data",
            rec,
            &[127., 128., 129., 135., 136., 137., 143., 144., 145.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "float_data_no_comp",
            rec,
            &[227., 228., 229., 235., 236., 237., 243., 244., 245.],
        );

        // right edge chunk
        let rec = &records[5];
        validate_names_and_types(&target_types, rec);
        validate_bool_column("bool_data", rec, &[true, false, true, false, true, false]);
        validate_primitive_column::<Int64Type, i64>("int_data", rec, &[-1, 0, 7, 8, 15, 16]);
        validate_primitive_column::<UInt64Type, u64>("uint_data", rec, &[30, 31, 38, 39, 46, 47]);
        validate_primitive_column::<Float64Type, f64>(
            "float_data",
            rec,
            &[130., 131., 138., 139., 146., 147.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "float_data_no_comp",
            rec,
            &[230., 231., 238., 239., 246., 247.],
        );

        // bottom right edge chunk
        let rec = &records[8];
        validate_names_and_types(&target_types, rec);
        validate_bool_column("bool_data", rec, &[true, false, true, false]);
        validate_primitive_column::<Int64Type, i64>("int_data", rec, &[23, 24, 31, 32]);
        validate_primitive_column::<UInt64Type, u64>("uint_data", rec, &[54, 55, 62, 63]);
        validate_primitive_column::<Float64Type, f64>(
            "float_data",
            rec,
            &[154.0, 155.0, 162.0, 163.0],
        );
        validate_primitive_column::<Float64Type, f64>(
            "float_data_no_comp",
            rec,
            &[254.0, 255.0, 262.0, 263.0],
        );
    }

    #[test]
    fn projection_tests() {
        let p = get_test_v2_data_path("compression_example.zarr".to_string());
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
        validate_bool_column(
            "bool_data",
            rec,
            &[false, true, false, false, true, false, false, true, false],
        );
        validate_primitive_column::<Int64Type, i64>(
            "int_data",
            rec,
            &[-4, -3, -2, 4, 5, 6, 12, 13, 14],
        );
    }

    #[test]
    fn multiple_readers_tests() {
        let p = get_test_v2_data_path("compression_example.zarr".to_string());
        let reader1 = ZarrRecordBatchReaderBuilder::new(p.clone())
            .build_partial_reader(Some((0, 5)))
            .unwrap();
        let reader2 = ZarrRecordBatchReaderBuilder::new(p)
            .build_partial_reader(Some((5, 9)))
            .unwrap();

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
        validate_bool_column(
            "bool_data",
            rec,
            &[false, true, false, false, true, false, false, true, false],
        );
        validate_primitive_column::<Int64Type, i64>(
            "int_data",
            rec,
            &[-4, -3, -2, 4, 5, 6, 12, 13, 14],
        );
        validate_primitive_column::<UInt64Type, u64>(
            "uint_data",
            rec,
            &[27, 28, 29, 35, 36, 37, 43, 44, 45],
        );
        validate_primitive_column::<Float64Type, f64>(
            "float_data",
            rec,
            &[127., 128., 129., 135., 136., 137., 143., 144., 145.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "float_data_no_comp",
            rec,
            &[227., 228., 229., 235., 236., 237., 243., 244., 245.],
        );

        // bottom edge chunk
        let rec = &records2[2];
        validate_names_and_types(&target_types, rec);
        validate_bool_column("bool_data", rec, &[false, true, false, false, true, false]);
        validate_primitive_column::<Int64Type, i64>("int_data", rec, &[20, 21, 22, 28, 29, 30]);
        validate_primitive_column::<UInt64Type, u64>("uint_data", rec, &[51, 52, 53, 59, 60, 61]);
        validate_primitive_column::<Float64Type, f64>(
            "float_data",
            rec,
            &[151.0, 152.0, 153.0, 159.0, 160.0, 161.0],
        );
        validate_primitive_column::<Float64Type, f64>(
            "float_data_no_comp",
            rec,
            &[251.0, 252.0, 253.0, 259.0, 260.0, 261.0],
        );
    }

    #[test]
    fn endianness_and_order_tests() {
        let p = get_test_v2_data_path("endianness_and_order_example.zarr".to_string());
        let reader = ZarrRecordBatchReaderBuilder::new(p).build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([
            ("var1".to_string(), DataType::Int32),
            ("var2".to_string(), DataType::Int32),
        ]);

        // bottom edge chunk
        let rec = &records[9];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Int32Type, i32>(
            "var1",
            rec,
            &[69, 80, 91, 70, 81, 92, 71, 82, 93],
        );
        validate_primitive_column::<Int32Type, i32>(
            "var2",
            rec,
            &[69, 80, 91, 70, 81, 92, 71, 82, 93],
        );
    }

    #[test]
    fn string_data_tests() {
        let p = get_test_v2_data_path("string_example.zarr".to_string());
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
        validate_primitive_column::<UInt8Type, u8>("uint_data", rec, &[51, 52, 53, 59, 60, 61]);
        validate_string_column(
            "string_data",
            rec,
            &["abc61", "abc62", "abc63", "abc69", "abc70", "abc71"],
        );
        validate_string_column(
            "utf8_data",
            rec,
            &["def61", "def62", "def63", "def69", "def70", "def71"],
        );
    }

    #[test]
    fn ts_data_tests() {
        let p = get_test_v2_data_path("ts_example.zarr".to_string());
        let reader = ZarrRecordBatchReaderBuilder::new(p).build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([
            (
                "ts_s_data".to_string(),
                DataType::Timestamp(TimeUnit::Second, None),
            ),
            (
                "ts_ms_data".to_string(),
                DataType::Timestamp(TimeUnit::Millisecond, None),
            ),
            (
                "ts_us_data".to_string(),
                DataType::Timestamp(TimeUnit::Microsecond, None),
            ),
            (
                "ts_ns_data".to_string(),
                DataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
        ]);

        // top center chunk
        let rec = &records[1];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<TimestampSecondType, i64>(
            "ts_s_data",
            rec,
            &[1685750400, 1685836800, 1686182400, 1686268800],
        );
        validate_primitive_column::<TimestampMillisecondType, i64>(
            "ts_ms_data",
            rec,
            &[1685750400000, 1685836800000, 1686182400000, 1686268800000],
        );
        validate_primitive_column::<TimestampMicrosecondType, i64>(
            "ts_us_data",
            rec,
            &[
                1685750400000000,
                1685836800000000,
                1686182400000000,
                1686268800000000,
            ],
        );
        validate_primitive_column::<TimestampNanosecondType, i64>(
            "ts_ns_data",
            rec,
            &[
                1685750400000000000,
                1685836800000000000,
                1686182400000000000,
                1686268800000000000,
            ],
        );

        // top right edge chunk
        let rec = &records[2];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<TimestampSecondType, i64>(
            "ts_s_data",
            rec,
            &[1685923200, 1686355200],
        );
        validate_primitive_column::<TimestampMillisecondType, i64>(
            "ts_ms_data",
            rec,
            &[1685923200000, 1686355200000],
        );
        validate_primitive_column::<TimestampMicrosecondType, i64>(
            "ts_us_data",
            rec,
            &[1685923200000000, 1686355200000000],
        );
        validate_primitive_column::<TimestampNanosecondType, i64>(
            "ts_ns_data",
            rec,
            &[1685923200000000000, 1686355200000000000],
        );
    }

    #[test]
    fn one_dim_tests() {
        let p = get_test_v2_data_path("one_dim_example.zarr".to_string());
        let reader = ZarrRecordBatchReaderBuilder::new(p).build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([
            ("int_data".to_string(), DataType::Int64),
            ("float_data".to_string(), DataType::Float64),
        ]);

        // center chunk
        let rec = &records[1];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Int64Type, i64>("int_data", rec, &[-2, -1, 0]);
        validate_primitive_column::<Float64Type, f64>("float_data", rec, &[103.0, 104.0, 105.0]);

        // right edge chunk
        let rec = &records[3];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Int64Type, i64>("int_data", rec, &[4, 5]);
        validate_primitive_column::<Float64Type, f64>("float_data", rec, &[109.0, 110.0]);
    }

    #[test]
    fn three_dim_tests() {
        let p = get_test_v2_data_path("three_dim_example.zarr".to_string());
        let reader = ZarrRecordBatchReaderBuilder::new(p).build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([
            ("int_data".to_string(), DataType::Int64),
            ("float_data".to_string(), DataType::Float64),
        ]);

        // center chunk
        let rec = &records[13];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Int64Type, i64>("int_data", rec, &[0, 1, 5, 6, 25, 26, 30, 31]);
        validate_primitive_column::<Float64Type, f64>(
            "float_data",
            rec,
            &[162.0, 163.0, 167.0, 168.0, 187.0, 188.0, 192.0, 193.0],
        );

        // right edge chunk
        let rec = &records[14];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Int64Type, i64>("int_data", rec, &[2, 7, 27, 32]);
        validate_primitive_column::<Float64Type, f64>(
            "float_data",
            rec,
            &[164.0, 169.0, 189.0, 194.0],
        );

        // right front edge chunk
        let rec = &records[23];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Int64Type, i64>("int_data", rec, &[52, 57]);
        validate_primitive_column::<Float64Type, f64>("float_data", rec, &[214.0, 219.0]);

        // bottom front edge chunk
        let rec = &records[24];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Int64Type, i64>("int_data", rec, &[58, 59]);
        validate_primitive_column::<Float64Type, f64>("float_data", rec, &[220.0, 221.0]);

        // right front bottom edge chunk
        let rec = &records[26];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Int64Type, i64>("int_data", rec, &[62]);
        validate_primitive_column::<Float64Type, f64>("float_data", rec, &[224.0]);
    }

    #[test]
    fn filters_tests() {
        let p = get_test_v2_data_path("lat_lon_example.zarr".to_string());
        let mut builder = ZarrRecordBatchReaderBuilder::new(p);

        builder = builder.with_filter(create_filter());
        let reader = builder.build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([
            ("lat".to_string(), DataType::Float64),
            ("lon".to_string(), DataType::Float64),
            ("float_data".to_string(), DataType::Float64),
        ]);

        // check the values in a chunk. the predicate pushdown only takes care of
        // skipping whole chunks, so there is no guarantee that the values in the
        // record batch fully satisfy the predicate, here we are only checking that
        // the first chunk that was read is the first one with some values that
        // satisfy the predicate.
        let rec = &records[0];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Float64Type, f64>(
            "lat",
            rec,
            &[
                38.4, 38.5, 38.6, 38.7, 38.4, 38.5, 38.6, 38.7, 38.4, 38.5, 38.6, 38.7, 38.4, 38.5,
                38.6, 38.7,
            ],
        );
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            rec,
            &[
                -110.0, -110.0, -110.0, -110.0, -109.9, -109.9, -109.9, -109.9, -109.8, -109.8,
                -109.8, -109.8, -109.7, -109.7, -109.7, -109.7,
            ],
        );
        validate_primitive_column::<Float64Type, f64>(
            "float_data",
            rec,
            &[
                1005.0, 1006.0, 1007.0, 1008.0, 1016.0, 1017.0, 1018.0, 1019.0, 1027.0, 1028.0,
                1029.0, 1030.0, 1038.0, 1039.0, 1040.0, 1041.0,
            ],
        );
    }

    #[test]
    fn empty_query_tests() {
        let p = get_test_v2_data_path("lat_lon_example.zarr".to_string());
        let mut builder = ZarrRecordBatchReaderBuilder::new(p);

        // set a filter that will filter out all the data, there should be nothing left after
        // we apply it.
        let mut filters: Vec<Box<dyn ZarrArrowPredicate>> = Vec::new();
        let f = ZarrArrowPredicateFn::new(
            ZarrProjection::keep(vec!["lat".to_string()]),
            move |batch| {
                gt_eq(
                    batch.column_by_name("lat").unwrap(),
                    &Scalar::new(&Float64Array::from(vec![100.0])),
                )
            },
        );
        filters.push(Box::new(f));

        builder = builder.with_filter(ZarrChunkFilter::new(filters));
        let reader = builder.build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        // there should be no records, because of the filter.
        assert_eq!(records.len(), 0);
    }

    #[test]
    fn one_dim_repr_tests() {
        let p = get_test_v2_data_path("lat_lon_example_w_1d_repr.zarr".to_string());
        let mut builder = ZarrRecordBatchReaderBuilder::new(p);

        builder = builder.with_filter(create_filter());
        let reader = builder.build().unwrap();
        let records_from_one_d_repr: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let p = get_test_v2_data_path("lat_lon_example.zarr".to_string());
        let mut builder = ZarrRecordBatchReaderBuilder::new(p);

        builder = builder.with_filter(create_filter());
        let reader = builder.build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        assert_eq!(records_from_one_d_repr.len(), records.len());
        for (rec, rec_from_one_d_repr) in records.iter().zip(records_from_one_d_repr.iter()) {
            assert_eq!(rec, rec_from_one_d_repr);
        }
    }

    #[test]
    fn no_sharding_tests() {
        let p = get_test_v3_data_path("no_sharding.zarr".to_string());
        let builder = ZarrRecordBatchReaderBuilder::new(p);

        let reader = builder.build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([("int_data".to_string(), DataType::Int32)]);

        let rec = &records[1];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Int32Type, i32>(
            "int_data",
            rec,
            &[4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 39, 52, 53, 54, 55],
        );
    }

    #[test]
    fn no_sharding_with_edge_tests() {
        let p = get_test_v3_data_path("no_sharding_with_edge.zarr".to_string());
        let builder = ZarrRecordBatchReaderBuilder::new(p);

        let reader = builder.build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([
            ("float_data".to_string(), DataType::Float32),
            ("uint_data".to_string(), DataType::UInt64),
        ]);

        let rec = &records[3];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Float32Type, f32>(
            "float_data",
            rec,
            &[
                12.0, 13.0, 14.0, 27.0, 28.0, 29.0, 42.0, 43.0, 44.0, 57.0, 58.0, 59.0,
            ],
        );
        validate_primitive_column::<UInt64Type, u64>(
            "uint_data",
            rec,
            &[12, 13, 14, 27, 28, 29, 42, 43, 44, 57, 58, 59],
        );
    }

    #[test]
    fn with_sharding_tests() {
        let p = get_test_v3_data_path("with_sharding.zarr".to_string());
        let builder = ZarrRecordBatchReaderBuilder::new(p);

        let reader = builder.build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([
            ("float_data".to_string(), DataType::Float64),
            ("int_data".to_string(), DataType::Int64),
        ]);

        let rec = &records[2];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Float64Type, f64>(
            "float_data",
            rec,
            &[
                32.0, 33.0, 40.0, 41.0, 34.0, 35.0, 42.0, 43.0, 48.0, 49.0, 56.0, 57.0, 50.0, 51.0,
                58.0, 59.0,
            ],
        );
        validate_primitive_column::<Int64Type, i64>(
            "int_data",
            rec,
            &[
                32, 33, 40, 41, 34, 35, 42, 43, 48, 49, 56, 57, 50, 51, 58, 59,
            ],
        );
    }

    #[test]
    fn with_sharding_with_edge_tests() {
        let p = get_test_v3_data_path("with_sharding_with_edge.zarr".to_string());
        let builder = ZarrRecordBatchReaderBuilder::new(p);

        let reader = builder.build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([("uint_data".to_string(), DataType::UInt16)]);

        let rec = &records[1];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<UInt16Type, u16>(
            "uint_data",
            rec,
            &[4, 5, 11, 12, 6, 13, 18, 19, 25, 26, 20, 27],
        );
    }

    #[test]
    fn three_dims_no_sharding_with_edge_tests() {
        let p = get_test_v3_data_path("no_sharding_with_edge_3d.zarr".to_string());
        let builder = ZarrRecordBatchReaderBuilder::new(p);

        let reader = builder.build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([("uint_data".to_string(), DataType::UInt64)]);

        let rec = &records[5];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<UInt64Type, u64>("uint_data", rec, &[14, 19, 39, 44]);
    }

    #[test]
    fn three_dims_with_sharding_with_edge_tests() {
        let p = get_test_v3_data_path("with_sharding_with_edge_3d.zarr".to_string());
        let builder = ZarrRecordBatchReaderBuilder::new(p);

        let reader = builder.build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([("float_data".to_string(), DataType::Float64)]);

        let rec = &records[23];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Float64Type, f64>(
            "float_data",
            rec,
            &[
                1020.0, 1021.0, 1031.0, 1032.0, 1141.0, 1142.0, 1152.0, 1153.0, 1022.0, 1033.0,
                1143.0, 1154.0, 1042.0, 1043.0, 1053.0, 1054.0, 1163.0, 1164.0, 1174.0, 1175.0,
                1044.0, 1055.0, 1165.0, 1176.0, 1262.0, 1263.0, 1273.0, 1274.0, 1264.0, 1275.0,
                1284.0, 1285.0, 1295.0, 1296.0, 1286.0, 1297.0,
            ],
        );
    }
}
