//! A module tha provides a sychronous reader for zarr store, to generate [`RecordBatch`]es.
//!

use arrow_array::*;
use arrow_schema::{DataType, Field, FieldRef, Schema};
use itertools::Itertools;
use std::{collections::HashMap, sync::Arc};

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
    broadcastable_array_axes: HashMap<String, Option<usize>>,
}

impl<T: ZarrRead> ZarrStore<T> {
    pub(crate) fn new(
        zarr_reader: T,
        chunk_positions: Vec<Vec<usize>>,
        projection: ZarrProjection,
    ) -> ZarrResult<Self> {
        let meta = zarr_reader.get_zarr_metadata()?;
        let mut bdc_axes: HashMap<String, Option<usize>> = HashMap::new();
        for col in meta.get_columns() {
            let mut axis = None;
            if let Some(params) = meta.get_array_meta(col)?.get_ond_d_array_params() {
                axis = Some(params.1);
            }
            bdc_axes.insert(col.to_string(), axis);
        }
        Ok(Self {
            meta,
            chunk_positions,
            zarr_reader,
            projection,
            curr_chunk: 0,
            broadcastable_array_axes: bdc_axes,
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
            &self.broadcastable_array_axes,
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
        chunk_dims: &Vec<usize>,
    ) -> ZarrResult<(ArrayRef, FieldRef)> {
        // get the metadata for the array
        let meta = self.meta.get_array_meta(&col_name)?;

        // take the raw data from the chunk
        let data = arr_chnk.take_data();

        // apply codecs and decode the raw data
        let (arr, field) = apply_codecs(
            col_name,
            data,
            chunk_dims,
            real_dims,
            meta.get_type(),
            meta.get_codecs(),
            meta.get_sharding_params(),
            meta.get_ond_d_array_params(),
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
    use crate::test_utils::{
        compare_values, create_filter, store_1d, store_3d, store_compression_codecs,
        store_endianness_and_order, store_endianness_and_order_3d, store_lat_lon,
        store_lat_lon_broadcastable, store_partial_sharding, store_partial_sharding_3d,
        validate_bool_column, validate_names_and_types, validate_primitive_column, StoreWrapper,
    };
    use arrow::compute::kernels::cmp::gt_eq;
    use arrow_array::types::*;
    use arrow_schema::DataType;
    use rstest::*;
    use std::{boxed::Box, collections::HashMap};

    use super::*;
    use crate::reader::filters::{ZarrArrowPredicate, ZarrArrowPredicateFn};

    #[rstest]
    fn compressed_data_tests(
        #[with("compressed_data_tests".to_string())] store_compression_codecs: StoreWrapper,
    ) {
        let p = store_compression_codecs.store_path();

        let reader = ZarrRecordBatchReaderBuilder::new(p).build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([
            ("bool_data".to_string(), DataType::Boolean),
            ("uint_data".to_string(), DataType::UInt64),
            ("int_data".to_string(), DataType::Int64),
            ("float_data".to_string(), DataType::Float32),
            ("float_data_no_comp".to_string(), DataType::Float64),
        ]);
        validate_names_and_types(&target_types, &records[0]);

        // center chunk
        let rec = &records[4];
        validate_bool_column(
            "bool_data",
            rec,
            &[false, true, false, false, true, false, false, true, false],
        );
        validate_primitive_column::<UInt64Type, u64>(
            "uint_data",
            rec,
            &[27, 28, 29, 35, 36, 37, 43, 44, 45],
        );
        validate_primitive_column::<Int64Type, i64>(
            "int_data",
            rec,
            &[-4, -3, -2, 4, 5, 6, 12, 13, 14],
        );
        validate_primitive_column::<Float32Type, f32>(
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
        validate_bool_column("bool_data", rec, &[true, false, true, false, true, false]);
        validate_primitive_column::<UInt64Type, u64>("uint_data", rec, &[30, 31, 38, 39, 46, 47]);
        validate_primitive_column::<Int64Type, i64>("int_data", rec, &[-1, 0, 7, 8, 15, 16]);
        validate_primitive_column::<Float32Type, f32>(
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
        validate_bool_column("bool_data", rec, &[true, false, true, false]);
        validate_primitive_column::<UInt64Type, u64>("uint_data", rec, &[54, 55, 62, 63]);
        validate_primitive_column::<Int64Type, i64>("int_data", rec, &[23, 24, 31, 32]);
        validate_primitive_column::<Float32Type, f32>(
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

    #[rstest]
    fn projection_tests(
        #[with("projection_tests".to_string())] store_compression_codecs: StoreWrapper,
    ) {
        let p = store_compression_codecs.store_path();
        let proj = ZarrProjection::keep(vec!["bool_data".to_string(), "int_data".to_string()]);
        let builder = ZarrRecordBatchReaderBuilder::new(p).with_projection(proj);
        let reader = builder.build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([
            ("bool_data".to_string(), DataType::Boolean),
            ("int_data".to_string(), DataType::Int64),
        ]);
        validate_names_and_types(&target_types, &records[0]);

        // center chunk
        let rec = &records[4];
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

    #[rstest]
    fn multiple_readers_tests(
        #[with("multiple_readers_tests".to_string())] store_compression_codecs: StoreWrapper,
    ) {
        let p = store_compression_codecs.store_path();
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
            ("float_data".to_string(), DataType::Float32),
            ("float_data_no_comp".to_string(), DataType::Float64),
        ]);
        validate_names_and_types(&target_types, &records1[0]);
        validate_names_and_types(&target_types, &records2[0]);

        // center chunk
        let rec = &records1[4];
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
        validate_primitive_column::<Float32Type, f32>(
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
        validate_bool_column("bool_data", rec, &[false, true, false, false, true, false]);
        validate_primitive_column::<Int64Type, i64>("int_data", rec, &[20, 21, 22, 28, 29, 30]);
        validate_primitive_column::<UInt64Type, u64>("uint_data", rec, &[51, 52, 53, 59, 60, 61]);
        validate_primitive_column::<Float32Type, f32>(
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

    #[rstest]
    fn endianness_and_order_tests(
        #[with("endianness_and_order_tests".to_string())] store_endianness_and_order: StoreWrapper,
    ) {
        let p = store_endianness_and_order.store_path();
        let reader = ZarrRecordBatchReaderBuilder::new(p).build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([
            ("int_data_big_endian_f_order".to_string(), DataType::Int32),
            ("int_data".to_string(), DataType::Int32),
        ]);
        validate_names_and_types(&target_types, &records[0]);

        for rec in &records {
            compare_values::<Int32Type>("int_data_big_endian_f_order", "int_data", rec);
        }
    }

    #[rstest]
    fn endianness_and_order_3d_tests(
        #[with("endianness_and_order_3d_tests".to_string())]
        store_endianness_and_order_3d: StoreWrapper,
    ) {
        let p = store_endianness_and_order_3d.store_path();
        let reader = ZarrRecordBatchReaderBuilder::new(p).build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([
            ("int_data_big_endian_f_order".to_string(), DataType::Int32),
            ("int_data".to_string(), DataType::Int32),
        ]);
        validate_names_and_types(&target_types, &records[0]);

        for rec in &records {
            compare_values::<Int32Type>("int_data_big_endian_f_order", "int_data", rec);
        }
    }

    // apparently in zarr v3 the type is just "string", so I'm not too sure
    // how this is supposed to be handled, how should I get the length for
    // example... I will revisit later and uncomment this test.
    // #[rstest]
    // fn string_data_tests(
    //     #[with("string_data_tests".to_string())] store_strings: StoreWrapper
    // ) {
    //     let p = store_strings.prefix_to_fs_path(&StorePrefix::new("").unwrap());
    //     let reader = ZarrRecordBatchReaderBuilder::new(p).build().unwrap();
    //     let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

    //     let target_types = HashMap::from([
    //         ("int_data".to_string(), DataType::Int8),
    //         ("string_data".to_string(), DataType::Utf8),
    //     ]);
    //     validate_names_and_types(&target_types, &records[0]);

    //     // top left corner
    //     let rec = &records[0];
    //     validate_primitive_column::<UInt8Type, u8>("uint_data", rec, &[1, 2, 3, 9, 10, 11, 17, 18, 19]);
    //     validate_string_column(
    //         "string_data",
    //         rec,
    //         &["abc01", "abc02", "abc03", "abc09", "abc10", "abc11", "abc17", "abc18", "abc19"],
    //     );

    //     // bottom edge chunk
    //     let rec = &records[7];
    //     validate_primitive_column::<UInt8Type, u8>("uint_data", rec, &[61, 62, 63, 69, 70, 71]);
    //     validate_string_column(
    //         "string_data",
    //         rec,
    //         &["abc61", "abc62", "abc63", "abc69", "abc70", "abc71"],
    //     );
    // }

    #[rstest]
    fn one_dim_tests(#[with("one_dim_tests".to_string())] store_1d: StoreWrapper) {
        let p = store_1d.store_path();
        let reader = ZarrRecordBatchReaderBuilder::new(p).build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([
            ("int_data".to_string(), DataType::Int32),
            ("float_data".to_string(), DataType::Float32),
        ]);
        validate_names_and_types(&target_types, &records[0]);

        // center chunk
        let rec = &records[1];
        validate_primitive_column::<Int32Type, i32>("int_data", rec, &[-2, -1, 0]);
        validate_primitive_column::<Float32Type, f32>("float_data", rec, &[103.0, 104.0, 105.0]);

        // right edge chunk
        let rec = &records[3];
        validate_primitive_column::<Int32Type, i32>("int_data", rec, &[4, 5]);
        validate_primitive_column::<Float32Type, f32>("float_data", rec, &[109.0, 110.0]);
    }

    #[rstest]
    fn three_dim_tests(#[with("three_dim_tests".to_string())] store_3d: StoreWrapper) {
        let p = store_3d.store_path();
        let reader = ZarrRecordBatchReaderBuilder::new(p).build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([
            ("int_data".to_string(), DataType::Int32),
            ("float_data".to_string(), DataType::Float32),
        ]);
        validate_names_and_types(&target_types, &records[0]);

        // center chunk
        let rec = &records[13];
        validate_primitive_column::<Int32Type, i32>("int_data", rec, &[0, 1, 5, 6, 25, 26, 30, 31]);
        validate_primitive_column::<Float32Type, f32>(
            "float_data",
            rec,
            &[162.0, 163.0, 167.0, 168.0, 187.0, 188.0, 192.0, 193.0],
        );

        // right edge chunk
        let rec = &records[14];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Int32Type, i32>("int_data", rec, &[2, 7, 27, 32]);
        validate_primitive_column::<Float32Type, f32>(
            "float_data",
            rec,
            &[164.0, 169.0, 189.0, 194.0],
        );

        // right front edge chunk
        let rec = &records[23];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Int32Type, i32>("int_data", rec, &[52, 57]);
        validate_primitive_column::<Float32Type, f32>("float_data", rec, &[214.0, 219.0]);

        // bottom front edge chunk
        let rec = &records[24];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Int32Type, i32>("int_data", rec, &[58, 59]);
        validate_primitive_column::<Float32Type, f32>("float_data", rec, &[220.0, 221.0]);

        // right front bottom edge chunk
        let rec = &records[26];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Int32Type, i32>("int_data", rec, &[62]);
        validate_primitive_column::<Float32Type, f32>("float_data", rec, &[224.0]);
    }

    #[rstest]
    fn filters_tests(#[with("filters_tests".to_string())] store_lat_lon: StoreWrapper) {
        let p = store_lat_lon.store_path();
        let mut builder = ZarrRecordBatchReaderBuilder::new(p);

        // set the filters to select part of the raster, based on lat and
        // lon coordinates.
        let filter = create_filter();

        builder = builder.with_filter(filter);
        let reader = builder.build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([
            ("lat".to_string(), DataType::Float64),
            ("lon".to_string(), DataType::Float64),
            ("float_data".to_string(), DataType::Float64),
        ]);
        validate_names_and_types(&target_types, &records[0]);

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
                4.0, 5.0, 6.0, 7.0, 15.0, 16.0, 17.0, 18.0, 26.0, 27.0, 28.0, 29.0, 37.0, 38.0,
                39.0, 40.0,
            ],
        );
    }

    #[rstest]
    fn empty_query_tests(#[with("empty_query_tests".to_string())] store_lat_lon: StoreWrapper) {
        let p = store_lat_lon.store_path();
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

    #[rstest]
    fn array_broadcast_tests(
        #[with("array_broadcast_tests_part1".to_string())] store_lat_lon: StoreWrapper,
        #[with("array_broadcast_tests_part2".to_string())]
        store_lat_lon_broadcastable: StoreWrapper,
    ) {
        // reference that doesn't broadcast a 1D array
        let p = store_lat_lon.store_path();
        let mut builder = ZarrRecordBatchReaderBuilder::new(p);

        builder = builder.with_filter(create_filter());
        let reader = builder.build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        // v3 format with array broadcast
        let p = store_lat_lon_broadcastable.store_path();
        let mut builder = ZarrRecordBatchReaderBuilder::new(p);

        builder = builder.with_filter(create_filter());
        let reader = builder.build().unwrap();
        let records_from_one_d_repr: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        assert_eq!(records_from_one_d_repr.len(), records.len());
        for (rec, rec_from_one_d_repr) in records.iter().zip(records_from_one_d_repr.iter()) {
            assert_eq!(rec, rec_from_one_d_repr);
        }
    }

    #[rstest]
    fn partial_sharding_tests(
        #[with("partial_sharding_tests".to_string())] store_partial_sharding: StoreWrapper,
    ) {
        let p = store_partial_sharding.store_path();
        let builder = ZarrRecordBatchReaderBuilder::new(p);

        let reader = builder.build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        for rec in records {
            compare_values::<Float64Type>("float_data_not_sharded", "float_data_sharded", &rec);
        }
    }

    #[rstest]
    fn partial_sharding_3d_tests(
        #[with("partial_sharding_3d_tests".to_string())] store_partial_sharding_3d: StoreWrapper,
    ) {
        let p = store_partial_sharding_3d.store_path();
        let builder = ZarrRecordBatchReaderBuilder::new(p);

        let reader = builder.build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        for rec in records {
            compare_values::<Float64Type>("float_data_not_sharded", "float_data_sharded", &rec);
        }
    }
}
