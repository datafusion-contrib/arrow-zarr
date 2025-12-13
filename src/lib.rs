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
pub mod zarr_store_opener;

pub use zarr_store_opener::ZarrRecordBatchStream;

#[cfg(test)]
mod test_utils {
    use std::collections::HashMap;
    use std::fmt::Debug;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;

    use arrow::buffer::ScalarBuffer;
    use arrow_array::cast::AsArray;
    use arrow_array::types::*;
    use arrow_array::RecordBatch;
    use arrow_schema::{DataType as ArrowDataType, Field, Schema, SchemaRef};
    use futures::executor::block_on;
    #[cfg(feature = "icechunk")]
    use icechunk::{ObjectStorage, Repository};
    use itertools::enumerate;
    use ndarray::{Array, Array1, Array2};
    use object_store::local::LocalFileSystem;
    use walkdir::WalkDir;
    use zarrs::array::{codec, ArrayBuilder, DataType, FillValue};
    use zarrs::array_subset::ArraySubset;
    #[cfg(feature = "icechunk")]
    use zarrs_icechunk::AsyncIcechunkStore;
    use zarrs_object_store::AsyncObjectStore;
    use zarrs_storage::{
        AsyncReadableWritableListableStorageTraits, AsyncWritableStorageTraits, StorePrefix,
    };

    // convenience class to make sure the local zarr stores get cleanup
    // after we're done running a test.
    pub(crate) struct LocalZarrStoreWrapper {
        store: Arc<AsyncObjectStore<LocalFileSystem>>,
        path: PathBuf,
    }

    impl LocalZarrStoreWrapper {
        pub(crate) fn new(store_name: String) -> Self {
            if store_name.is_empty() {
                panic!("name for test zarr store cannot be empty!")
            }

            let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(store_name);
            fs::create_dir(p.clone()).unwrap();
            let store = AsyncObjectStore::new(LocalFileSystem::new_with_prefix(p.clone()).unwrap());
            Self {
                store: Arc::new(store),
                path: p,
            }
        }

        pub(crate) fn get_store(&self) -> Arc<AsyncObjectStore<LocalFileSystem>> {
            self.store.clone()
        }

        pub(crate) fn get_store_path(&self) -> String {
            self.path.as_os_str().to_str().unwrap().into()
        }
    }

    impl Drop for LocalZarrStoreWrapper {
        fn drop(&mut self) {
            let prefix = StorePrefix::new("").unwrap();
            block_on(self.store.erase_prefix(&prefix)).unwrap();

            while fs::exists(self.path.clone()).unwrap() {
                for d in WalkDir::new(self.path.clone()) {
                    let _ = fs::remove_dir(d.unwrap().path());
                }
            }
        }
    }

    // convenience class to make sure the local icechunk repos get cleanup
    // after we're done running a test.
    #[cfg(feature = "icechunk")]
    pub(crate) struct LocalIcechunkRepoWrapper {
        store: Arc<AsyncIcechunkStore>,
        path: PathBuf,
    }

    #[cfg(feature = "icechunk")]
    impl LocalIcechunkRepoWrapper {
        pub(crate) async fn new(store_name: String) -> Self {
            if store_name.is_empty() {
                panic!("name for test icechunk repo cannot be empty!")
            }
            let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(store_name);
            fs::create_dir(p.clone()).unwrap();
            let repo = Repository::create(
                None,
                Arc::new(ObjectStorage::new_local_filesystem(&p).await.unwrap()),
                HashMap::new(),
            )
            .await
            .unwrap();
            let session = repo.writable_session("main").await.unwrap();
            Self {
                store: Arc::new(AsyncIcechunkStore::new(session)),
                path: p,
            }
        }

        pub(crate) fn get_store(&self) -> Arc<AsyncIcechunkStore> {
            self.store.clone()
        }

        pub(crate) fn get_store_path(&self) -> String {
            self.path.to_str().unwrap().into()
        }
    }

    // TODO: Implement Drop. Just not sure how to do this cleanly yet.
    #[cfg(feature = "icechunk")]
    impl Drop for LocalIcechunkRepoWrapper {
        fn drop(&mut self) {
            if !self
                .path
                .to_str()
                .unwrap()
                .contains(env!("CARGO_MANIFEST_DIR"))
            {
                panic!("should not be deleting this icechunk repo!")
            }

            //delete the different icechunk repo components one at a time.
            fs::remove_dir_all(self.path.join("manifests")).unwrap();
            fs::remove_dir_all(self.path.join("refs")).unwrap();
            fs::remove_dir_all(self.path.join("snapshots")).unwrap();
            fs::remove_dir_all(self.path.join("transactions")).unwrap();
            fs::remove_dir(self.path.clone()).unwrap();
        }
    }

    // helpers to create some test data on the fly.
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

    pub(crate) async fn write_1d_float_array(
        data: Vec<f64>,
        fillvalue: f64,
        shape: u64,
        chunk: u64,
        store: Arc<dyn AsyncReadableWritableListableStorageTraits>,
        path: &str,
        dimensions: Option<Vec<String>>,
    ) {
        let mut array_builder = ArrayBuilder::new(
            vec![shape],
            [chunk],
            DataType::Float64,
            FillValue::from(fillvalue),
        );
        let mut builder_ref = &mut array_builder;
        let codec = get_lz4_compressor();
        builder_ref = builder_ref.bytes_to_bytes_codecs(vec![Arc::new(codec)]);
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

    pub(crate) async fn write_2d_float_array(
        data: Option<Vec<f64>>,
        fillvalue: f64,
        shape: (u64, u64),
        chunk: (u64, u64),
        store: Arc<dyn AsyncReadableWritableListableStorageTraits>,
        path: &str,
        dimensions: Option<Vec<String>>,
    ) {
        let mut array_builder = ArrayBuilder::new(
            vec![shape.0, shape.1],
            [chunk.0, chunk.1],
            DataType::Float64,
            FillValue::from(fillvalue),
        );

        let mut builder_ref = &mut array_builder;
        let codec = get_lz4_compressor();
        builder_ref = builder_ref.bytes_to_bytes_codecs(vec![Arc::new(codec)]);
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

    // helpers to validate test data.
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

    pub(crate) fn extract_col<T>(col_name: &str, rec_batch: &RecordBatch) -> ScalarBuffer<T::Native>
    where
        T: ArrowPrimitiveType,
    {
        rec_batch
            .column_by_name(col_name)
            .unwrap()
            .as_primitive::<T>()
            .values()
            .clone()
    }

    async fn write_lat_lon_data_to_store(
        store: Arc<dyn AsyncReadableWritableListableStorageTraits>,
        write_data: bool,
        fillvalue: f64,
    ) {
        let lats = vec![35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0];
        write_1d_float_array(
            lats,
            0.0,
            8,
            3,
            store.clone(),
            "/lat",
            Some(["lat".into()].to_vec()),
        )
        .await;

        let lons = vec![
            -120.0, -119.0, -118.0, -117.0, -116.0, -115.0, -114.0, -113.0,
        ];
        write_1d_float_array(
            lons,
            0.0,
            8,
            3,
            store.clone(),
            "/lon",
            Some(["lon".into()].to_vec()),
        )
        .await;

        let data: Option<Vec<_>> = if write_data {
            Some((0..64).map(|i| i as f64).collect())
        } else {
            None
        };
        write_2d_float_array(
            data,
            fillvalue,
            (8, 8),
            (3, 3),
            store.clone(),
            "/data",
            Some(["lat".into(), "lon".into()].to_vec()),
        )
        .await;
    }

    async fn write_mixed_dims_lat_lon_data_to_store(
        store: Arc<dyn AsyncReadableWritableListableStorageTraits>,
        fillvalue: f64,
    ) {
        let lats = [
            vec![35.0; 8],
            vec![36.0; 8],
            vec![37.0; 8],
            vec![38.0; 8],
            vec![39.0; 8],
            vec![40.0; 8],
            vec![41.0; 8],
            vec![42.0; 8],
        ]
        .concat();
        write_2d_float_array(
            Some(lats),
            0.0,
            (8, 8),
            (3, 3),
            store.clone(),
            "/lat",
            Some(["lat".into(), "lon".into()].to_vec()),
        )
        .await;

        let lons = vec![
            -120.0, -119.0, -118.0, -117.0, -116.0, -115.0, -114.0, -113.0,
        ];
        write_1d_float_array(
            lons,
            0.0,
            8,
            3,
            store.clone(),
            "/lon",
            Some(["lon".into()].to_vec()),
        )
        .await;

        let data = (0..64).map(|i| i as f64).collect();
        write_2d_float_array(
            Some(data),
            fillvalue,
            (8, 8),
            (3, 3),
            store.clone(),
            "/data",
            Some(["lat".into(), "lon".into()].to_vec()),
        )
        .await;
    }

    pub(crate) async fn get_local_zarr_store(
        write_data: bool,
        fillvalue: f64,
        dir_name: &str,
    ) -> (LocalZarrStoreWrapper, SchemaRef) {
        let wrapper = LocalZarrStoreWrapper::new(dir_name.into());
        let store = wrapper.get_store();

        write_lat_lon_data_to_store(store, write_data, fillvalue).await;
        let schema = Arc::new(Schema::new(vec![
            Field::new("data", ArrowDataType::Float64, true),
            Field::new("lat", ArrowDataType::Float64, true),
            Field::new("lon", ArrowDataType::Float64, true),
        ]));

        (wrapper, schema)
    }

    pub(crate) async fn get_local_zarr_store_mix_dims(
        fillvalue: f64,
        dir_name: &str,
    ) -> (LocalZarrStoreWrapper, SchemaRef) {
        let wrapper = LocalZarrStoreWrapper::new(dir_name.into());
        let store = wrapper.get_store();

        write_mixed_dims_lat_lon_data_to_store(store, fillvalue).await;
        let schema = Arc::new(Schema::new(vec![
            Field::new("data", ArrowDataType::Float64, true),
            Field::new("lat", ArrowDataType::Float64, true),
            Field::new("lon", ArrowDataType::Float64, true),
        ]));

        (wrapper, schema)
    }

    #[cfg(feature = "icechunk")]
    pub(crate) async fn get_local_icechunk_repo(
        write_data: bool,
        fillvalue: f64,
        dir_name: &str,
    ) -> (LocalIcechunkRepoWrapper, SchemaRef) {
        let wrapper = LocalIcechunkRepoWrapper::new(dir_name.into()).await;
        let store = wrapper.get_store();

        write_lat_lon_data_to_store(store.clone(), write_data, fillvalue).await;
        let _ = store
            .session()
            .write()
            .await
            .commit("some test data", None)
            .await
            .unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("data", ArrowDataType::Float64, true),
            Field::new("lat", ArrowDataType::Float64, true),
            Field::new("lon", ArrowDataType::Float64, true),
        ]));

        (wrapper, schema)
    }
}
