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

pub mod errors;
pub mod table;
pub(crate) mod zarr_store_opener;

//pub mod async_reader;
//pub mod reader;

//#[cfg(feature = "datafusion")]
//pub mod datafusion;

#[cfg(test)]
mod test_utils {
    use arrow_array::cast::AsArray;
    use arrow_array::types::*;
    use arrow_array::RecordBatch;
    use futures::executor::block_on;
    use itertools::enumerate;
    use ndarray::{Array, Array1, Array2, Array3};
    use object_store::local::LocalFileSystem;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::{collections::HashMap, fmt::Debug};
    use zarrs::array::{codec, ArrayBuilder, DataType, FillValue};
    use zarrs::array_subset::ArraySubset;
    use zarrs_object_store::AsyncObjectStore;
    use zarrs_storage::{AsyncWritableStorageTraits, StorePrefix};

    use walkdir::WalkDir;

    // convenience class to make sure the stores get cleanup
    // after we're done running a test.
    pub(crate) struct StoreWrapper {
        store: Arc<AsyncObjectStore<LocalFileSystem>>,
        path: PathBuf,
    }

    impl StoreWrapper {
        pub(crate) fn new(store_name: String) -> Self {
            let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(store_name);
            fs::create_dir(p.clone()).unwrap();
            let store = AsyncObjectStore::new(LocalFileSystem::new_with_prefix(p.clone()).unwrap());
            StoreWrapper {
                store: Arc::new(store),
                path: p,
            }
        }

        pub(crate) fn get_store(&self) -> Arc<AsyncObjectStore<LocalFileSystem>> {
            self.store.clone()
        }
    }

    impl Drop for StoreWrapper {
        fn drop(&mut self) {
            let prefix = StorePrefix::new("").unwrap();
            block_on(self.store.erase_prefix(&prefix)).unwrap();

            while fs::exists(self.path.clone()).unwrap() {
                for d in WalkDir::new(self.path.clone()) {
                    let r = fs::remove_dir(d.unwrap().path());
                }
            }
        }
    }

    fn get_lz4_compressor() -> codec::BloscCodec {
        codec::BloscCodec::new(
            codec::bytes_to_bytes::blosc::BloscCompressor::LZ4,
            5.try_into().unwrap(),
            Some(0),
            codec::bytes_to_bytes::blosc::BloscShuffleMode::NoShuffle,
            Some(1),
        )
        .unwrap()
    }

    pub(crate) async fn write_1D_float_array(
        data: Vec<f64>,
        shape: u64,
        chunk: u64,
        store: Arc<AsyncObjectStore<LocalFileSystem>>,
        path: &str,
        dimensions: Option<Vec<String>>,
        compress_data: bool,
    ) {
        let mut array_builder = ArrayBuilder::new(
            vec![shape],
            DataType::Float64,
            vec![chunk].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        );
        let mut builder_ref = &mut array_builder;
        if compress_data {
            let codec = get_lz4_compressor();
            builder_ref = builder_ref.bytes_to_bytes_codecs(vec![Arc::new(codec)]);
        }
        if let Some(dimensions) = dimensions {
            builder_ref = builder_ref.dimension_names(dimensions.into());
        }

        let arr = builder_ref.build(store, path).unwrap();
        arr.async_store_metadata().await.unwrap();

        let arr_data: Array1<f64> = Array::from_vec(data)
            .into_shape_with_order(shape as usize)
            .unwrap();
        arr.async_store_array_subset_ndarray(&[0], arr_data)
            .await
            .unwrap();
    }

    pub(crate) async fn write_2D_float_array(
        data: Option<Vec<f64>>,
        fillvalue: f64,
        shape: (u64, u64),
        chunk: (u64, u64),
        store: Arc<AsyncObjectStore<LocalFileSystem>>,
        path: &str,
        dimensions: Option<Vec<String>>,
        compress_data: bool,
    ) {
        let mut array_builder = ArrayBuilder::new(
            vec![shape.0, shape.1],
            DataType::Float64,
            vec![chunk.0, chunk.1].try_into().unwrap(),
            FillValue::from(fillvalue),
        );

        let mut builder_ref = &mut array_builder;
        if compress_data {
            let codec = get_lz4_compressor();
            builder_ref = builder_ref.bytes_to_bytes_codecs(vec![Arc::new(codec)]);
        }
        if let Some(dimensions) = dimensions {
            builder_ref = builder_ref.dimension_names(dimensions.into());
        }

        let arr = builder_ref.build(store, path).unwrap();
        arr.async_store_metadata().await.unwrap();

        if let Some(data) = data {
            let arr_data: Array2<f64> = Array::from_vec(data)
                .into_shape_with_order((shape.0 as usize, shape.1 as usize))
                .unwrap();
            arr.async_store_array_subset_ndarray(
                ArraySubset::new_with_ranges(&[0..shape.0, 0..shape.1]).start(),
                arr_data,
            )
            .await
            .unwrap();
        }
    }

    pub(crate) fn validate_primitive_column<T, U>(col_name: &str, rec: &RecordBatch, targets: &[U])
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

    pub(crate) fn validate_names_and_types(
        targets: &HashMap<String, arrow_schema::DataType>,
        rec: &RecordBatch,
    ) {
        let mut target_cols: Vec<&String> = targets.keys().collect();
        let schema = rec.schema();
        let from_rec: Vec<&String> = schema.fields.iter().map(|f| f.name()).collect();

        target_cols.sort();
        assert_eq!(from_rec, target_cols);

        for field in schema.fields.iter() {
            assert_eq!(field.data_type(), targets.get(field.name()).unwrap());
        }
    }
}
