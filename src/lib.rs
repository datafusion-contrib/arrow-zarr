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

pub mod async_reader;
pub mod reader;

#[cfg(feature = "datafusion")]
pub mod datafusion;

#[cfg(test)]
mod test_utils {
    use std::path::PathBuf;
    use zarrs::array::codec::array_to_bytes::sharding::ShardingCodecBuilder;
    use zarrs_filesystem::FilesystemStore;
    use rstest::*;
    use zarrs_storage::{WritableStorageTraits, StorePrefix};
    use zarrs::array::{codec, ArrayBuilder, DataType, FillValue};
    use zarrs::array_subset::ArraySubset;
    use std::sync::Arc;
    use ndarray::{Array, Array1, Array2, Array3};
    use itertools::enumerate;
    use arrow_array::types::*;
    use arrow_array::*;
    use std::{collections::HashMap, fmt::Debug};
    use arrow_array::RecordBatch;
    use arrow_array::cast::AsArray;
    use crate::reader::{ZarrArrowPredicate, ZarrArrowPredicateFn, ZarrChunkFilter, ZarrProjection};
    use arrow::compute::kernels::cmp::{gt_eq, lt};

    fn create_zarr_store(store_name: String) -> FilesystemStore {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(store_name);
        FilesystemStore::new(p).unwrap()
    }

    fn clear_store(store: &Arc<FilesystemStore>) {
         let prefix = StorePrefix::new("").unwrap();
         store.erase_prefix(&prefix).unwrap();
    }

    fn get_lz4_compressor() -> codec::BloscCodec {
        codec::BloscCodec::new(
            codec::bytes_to_bytes::blosc::BloscCompressor::LZ4,
            5.try_into().unwrap(),
            Some(0),
            codec::bytes_to_bytes::blosc::BloscShuffleMode::NoShuffle,
            Some(1),
        ).unwrap()
    }

    // we won't actually use this, it's a place holder
    #[fixture]
    fn dummy_name() -> String {
        "test_store".to_string()
    }

    // convenience class to make sure the stores get cleanup
    // after we're done running a test.
    pub(crate) struct StoreWrapper {
        store: Arc<FilesystemStore>
    }

    impl StoreWrapper {
        fn new(store_name: String) -> Self {
            StoreWrapper{
                store: Arc::new(create_zarr_store(store_name))
            }
        }

        pub(crate) fn get_store(&self) -> Arc<FilesystemStore> {
            self.store.clone()
        }

        pub(crate) fn store_path(&self) -> PathBuf {
            self.store.prefix_to_fs_path(&StorePrefix::new("").unwrap())
        }
    }

    impl Drop for StoreWrapper {
        fn drop(&mut self) {
            clear_store(&self.store);
        }
    }

    // various fixtures to create some test data on the fly.
    #[fixture]
    pub(crate) fn store_raw_bytes(dummy_name: String) -> StoreWrapper {
        // create the store
        let store_wrapper = StoreWrapper::new(dummy_name);
        let store = store_wrapper.get_store();
    
        // uint array with no compression
        let array = ArrayBuilder::new(
            vec![9, 9],
            DataType::UInt8,
            vec![3, 3].try_into().unwrap(),
            FillValue::new(vec![0]),
        )
        .build(store.clone(), "/byte_data")
        .unwrap();
        array.store_metadata().unwrap();

        let arr: Array2<u8> =  Array::from_vec((0..81).collect()).into_shape_with_order((9, 9)).unwrap();
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..9, 0..9]).start(),
            arr,
        ).unwrap();

        // float data with no compression
        let array = ArrayBuilder::new(
            vec![9, 9],
            DataType::Float64,
            vec![3, 3].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .build(store.clone(), "/float_data")
        .unwrap();
        array.store_metadata().unwrap();

        let arr: Array2<f64> = Array::range(0.0, 81.0, 1.0).into_shape_with_order((9, 9)).unwrap();
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..9, 0..9]).start(),
            arr,
        ).unwrap();

        store_wrapper
    }

    #[fixture]
    pub(crate) fn store_compression_codecs(dummy_name: String) -> StoreWrapper {
        // create the store
        let store_wrapper = StoreWrapper::new(dummy_name);
        let store = store_wrapper.get_store();

        // bool array with blosc lz4 compression
        let array = ArrayBuilder::new(
            vec![8, 8],
            DataType::Bool,
            vec![3, 3].try_into().unwrap(),
            FillValue::new(vec![0]),
        )
        .bytes_to_bytes_codecs(vec![
            Arc::new(get_lz4_compressor()),
        ])
        .build(store.clone(), "/bool_data")
        .unwrap();
        array.store_metadata().unwrap();

        let mut v: Vec<bool> = Vec::with_capacity(64);
        for i in 0..64 {
            v.push(i % 2 == 0);
        }
        let arr: Array2<bool> = Array::from_vec(v).into_shape_with_order((8, 8)).unwrap();
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..8, 0..8]).start(),
            arr,
        ).unwrap();

        // uint array with blosc zlib compression
        let codec = codec::BloscCodec::new(
            codec::bytes_to_bytes::blosc::BloscCompressor::Zlib,
            5.try_into().unwrap(),
            Some(0),
            codec::bytes_to_bytes::blosc::BloscShuffleMode::Shuffle,
            Some(1),
        ).unwrap();

        let array = ArrayBuilder::new(
            vec![8, 8],
            DataType::UInt64,
            vec![3, 3].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .bytes_to_bytes_codecs(vec![
            Arc::new(codec),
        ])
        .build(store.clone(), "/uint_data")
        .unwrap();
        array.store_metadata().unwrap();

        let arr: Array2<u64> = Array::from_vec((0..64).collect()).into_shape_with_order((8, 8)).unwrap();
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..8, 0..8]).start(),
            arr,
        ).unwrap();

        // int array with zstd compression
        let codec = codec::BloscCodec::new(
            codec::bytes_to_bytes::blosc::BloscCompressor::Zlib,
            3.try_into().unwrap(),
            Some(0),
            codec::bytes_to_bytes::blosc::BloscShuffleMode::BitShuffle,
            Some(1),
        ).unwrap();

        let array = ArrayBuilder::new(
            vec![8, 8],
            DataType::Int64,
            vec![3, 3].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .bytes_to_bytes_codecs(vec![
            Arc::new(codec),
        ])
        .build(store.clone(), "/int_data")
        .unwrap();
        array.store_metadata().unwrap();

        let arr: Array2<i64> = Array::from_vec((-31..33).collect()).into_shape_with_order((8, 8)).unwrap();
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..8, 0..8]).start(),
            arr,
        ).unwrap();

        // float32 array with blosclz compression
        let codec = codec::BloscCodec::new(
            codec::bytes_to_bytes::blosc::BloscCompressor::BloscLZ,
            7.try_into().unwrap(),
            Some(0),
            codec::bytes_to_bytes::blosc::BloscShuffleMode::NoShuffle,
            Some(1),
        ).unwrap();

        let array = ArrayBuilder::new(
            vec![8, 8],
            DataType::Float32,
            vec![3, 3].try_into().unwrap(),
            FillValue::new(vec![0; 4]),
        )
        .bytes_to_bytes_codecs(vec![
            Arc::new(codec),
        ])
        .build(store.clone(), "/float_data")
        .unwrap();
        array.store_metadata().unwrap();

        let arr: Array2<f32> = Array::range(100.0, 164.0, 1.0).into_shape_with_order((8, 8)).unwrap();
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..8, 0..8]).start(),
            arr,
        ).unwrap();

        // float64 array with no compression
        let array = ArrayBuilder::new(
            vec![8, 8],
            DataType::Float64,
            vec![3, 3].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .build(store.clone(), "/float_data_no_comp")
        .unwrap();
        array.store_metadata().unwrap();

        let arr: Array2<f64> = Array::range(200.0, 264.0, 1.0).into_shape_with_order((8, 8)).unwrap();
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..8, 0..8]).start(),
            arr,
        ).unwrap();

        store_wrapper
    }

    #[fixture]
    pub(crate) fn store_endianness_and_order(dummy_name: String) -> StoreWrapper {
        // create the store
        let store_wrapper = StoreWrapper::new(dummy_name);
        let store = store_wrapper.get_store();

        // big endian and F order 
        let array = ArrayBuilder::new(
            vec![10, 11],
            DataType::Int32,
            vec![3, 3].try_into().unwrap(),
            FillValue::new(vec![0; 4]),
        )
        .bytes_to_bytes_codecs(vec![
            Arc::new(get_lz4_compressor())
        ])
        .array_to_bytes_codec(
            Arc::new(codec::BytesCodec::new(Some(zarrs::array::Endianness::Big))),
        )
        .array_to_array_codecs(vec![
            Arc::new(codec::TransposeCodec::new(codec::array_to_array::transpose::TransposeOrder::new(&[1, 0]).unwrap()))
        ])
        .build(store.clone(), "/int_data_big_endian_f_order")
        .unwrap();
        array.store_metadata().unwrap();

        let arr: Array2<i32> = Array::from_vec((0..110).collect()).into_shape_with_order((10, 11)).unwrap();
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..10, 0..11]).start(),
            arr,
        ).unwrap();

        // little endian and C order 
        let array = ArrayBuilder::new(
            vec![10, 11],
            DataType::Int32,
            vec![3, 3].try_into().unwrap(),
            FillValue::new(vec![0; 4]),
        )
        .bytes_to_bytes_codecs(vec![
            Arc::new(get_lz4_compressor())
        ])
        .build(store.clone(), "/int_data")
        .unwrap();
        array.store_metadata().unwrap();

        let arr: Array2<i32> = Array::from_vec((0..110).collect()).into_shape_with_order((10, 11)).unwrap();
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..10, 0..11]).start(),
            arr,
        ).unwrap();

        store_wrapper
    }

    #[fixture]
    pub(crate) fn store_endianness_and_order_3d(dummy_name: String) -> StoreWrapper {
        // create the store
        let store_wrapper = StoreWrapper::new(dummy_name);
        let store = store_wrapper.get_store();

        // big endian and F order 
        let array = ArrayBuilder::new(
            vec![10, 11, 12],
            DataType::Int32,
            vec![3, 4, 5].try_into().unwrap(),
            FillValue::new(vec![0; 4]),
        )
        .bytes_to_bytes_codecs(vec![
            Arc::new(get_lz4_compressor())
        ])
        .array_to_bytes_codec(
            Arc::new(codec::BytesCodec::new(Some(zarrs::array::Endianness::Big))),
        )
        .array_to_array_codecs(vec![
            Arc::new(codec::TransposeCodec::new(codec::array_to_array::transpose::TransposeOrder::new(&[2, 1, 0]).unwrap()))
        ])
        .build(store.clone(), "/int_data_big_endian_f_order")
        .unwrap();
        array.store_metadata().unwrap();

        let arr: Array3<i32> = Array::from_vec((0..(10*11*12)).collect()).into_shape_with_order((10, 11, 12)).unwrap();
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..10, 0..11, 0..12]).start(),
            arr,
        ).unwrap();

        // little endian and C order 
        let array = ArrayBuilder::new(
            vec![10, 11, 12],
            DataType::Int32,
            vec![3, 4, 5].try_into().unwrap(),
            FillValue::new(vec![0; 4]),
        )
        .bytes_to_bytes_codecs(vec![
            Arc::new(get_lz4_compressor())
        ])
        .build(store.clone(), "/int_data")
        .unwrap();
        array.store_metadata().unwrap();

        let arr: Array3<i32> = Array::from_vec((0..(10*11*12)).collect()).into_shape_with_order((10, 11, 12)).unwrap();
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..10, 0..11, 0..12]).start(),
            arr,
        ).unwrap();

        store_wrapper
    }

    // don't need that for now, I commented out tests with string data.
    // #[fixture]
    // pub(crate) fn store_strings(dummy_name: String) -> StoreWrapper {
    //     // create the store
    //     let store_wrapper = StoreWrapper::new(dummy_name);
    //     let store = store_wrapper.get_store();

    //     // integer data, for validation
    //     let array = ArrayBuilder::new(
    //         vec![8, 8],
    //         DataType::Int32,
    //         vec![3, 3].try_into().unwrap(),
    //         FillValue::new(vec![0; 4]),
    //     )
    //     .bytes_to_bytes_codecs(vec![
    //         Arc::new(get_lz4_compressor()),
    //     ])
    //     .build(store.clone(), "/int_data")
    //     .unwrap();
    //     array.store_metadata().unwrap();

    //     let arr: Array2<i32> = Array::from_vec((0..64).collect()).into_shape_with_order((8, 8)).unwrap();
    //     array.store_array_subset_ndarray(
    //         ArraySubset::new_with_ranges(&[0..8, 0..8]).start(),
    //         arr,
    //     ).unwrap();

    //     // some string data
    //     let array = ArrayBuilder::new(
    //         vec![8, 8],
    //         DataType::String,
    //         vec![3, 3].try_into().unwrap(),
    //         FillValue::from("     "),
    //     )
    //     .bytes_to_bytes_codecs(vec![
    //         Arc::new(get_lz4_compressor()),
    //     ])
    //     .build(store.clone(), "/string_data")
    //     .unwrap();
    //     array.store_metadata().unwrap();

    //     let arr: Array2<String> = Array::from_vec(
    //         (0..64).map(|i| format!("abc{:0>2}", i)).collect()
    //     )
    //     .into_shape_with_order((8, 8))
    //     .unwrap();
    //     array.store_array_subset_ndarray(
    //         ArraySubset::new_with_ranges(&[0..8, 0..8]).start(),
    //         arr,
    //     ).unwrap();

    //     store_wrapper
    // }

    #[fixture]
    pub(crate) fn store_1d(dummy_name: String) -> StoreWrapper {
        // create the store
        let store_wrapper = StoreWrapper::new(dummy_name);
        let store = store_wrapper.get_store();

        // integer data
        let array = ArrayBuilder::new(
            vec![11],
            DataType::Int32,
            vec![3].try_into().unwrap(),
            FillValue::new(vec![0; 4]),
        )
        .bytes_to_bytes_codecs(vec![
            Arc::new(get_lz4_compressor()),
        ])
        .build(store.clone(), "/int_data")
        .unwrap();
        array.store_metadata().unwrap();

        let arr: Array1<i32> = Array::from_vec((-5..6).collect());
        array.store_array_subset_ndarray(
            &[0],
            arr,
        ).unwrap();

        // float data
        let array = ArrayBuilder::new(
            vec![11],
            DataType::Float32,
            vec![3].try_into().unwrap(),
            FillValue::new(vec![0; 4]),
        )
        .bytes_to_bytes_codecs(vec![
            Arc::new(get_lz4_compressor()),
        ])
        .build(store.clone(), "/float_data")
        .unwrap();
        array.store_metadata().unwrap();

        let arr: Array1<f32> = Array::range(100.0, 111.0, 1.0);
        array.store_array_subset_ndarray(
            &[0],
            arr,
        ).unwrap();

        store_wrapper
    }

    #[fixture]
    pub(crate) fn store_3d(dummy_name: String) -> StoreWrapper {
        // create the store
        let store_wrapper = StoreWrapper::new(dummy_name);
        let store = store_wrapper.get_store();

        // integer data
        let array = ArrayBuilder::new(
            vec![5, 5, 5],
            DataType::Int32,
            vec![2, 2, 2].try_into().unwrap(),
            FillValue::new(vec![0; 4]),
        )
        .bytes_to_bytes_codecs(vec![
            Arc::new(get_lz4_compressor()),
        ])
        .build(store.clone(), "/int_data")
        .unwrap();
        array.store_metadata().unwrap();

        let arr: Array3<i32> = Array::from_vec((-62..63).collect()).into_shape_with_order((5, 5, 5)).unwrap();
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..5, 0..5, 0..5]).start(),
            arr,
        ).unwrap();

        // float data
        let array = ArrayBuilder::new(
            vec![5, 5, 5],
            DataType::Float32,
            vec![2, 2, 2].try_into().unwrap(),
            FillValue::new(vec![0; 4]),
        )
        .bytes_to_bytes_codecs(vec![
            Arc::new(get_lz4_compressor()),
        ])
        .build(store.clone(), "/float_data")
        .unwrap();
        array.store_metadata().unwrap();

        let arr: Array3<f32> = Array::range(100.0, 225.0, 1.0).into_shape_with_order((5, 5, 5)).unwrap();
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..5, 0..5, 0..5]).start(),
            arr,
        ).unwrap();

        store_wrapper
    }

    #[fixture]
    pub(crate) fn store_lat_lon(dummy_name: String) -> StoreWrapper {
        // create the store
        let store_wrapper = StoreWrapper::new(dummy_name);
        let store = store_wrapper.get_store();

        // latitude
        let array = ArrayBuilder::new(
            vec![11, 11],
            DataType::Float64,
            vec![4, 4].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .bytes_to_bytes_codecs(vec![
            Arc::new(get_lz4_compressor()),
        ])
        .build(store.clone(), "/lat")
        .unwrap();
        array.store_metadata().unwrap();

        let mut v = vec![38. , 38.1, 38.2, 38.3, 38.4, 38.5, 38.6, 38.7, 38.8, 38.9, 39.];
        for _ in 0..10 {
            v.extend_from_within(..11);
        }
        
        let arr: Array2<f64> = Array::from_vec(v).into_shape_with_order((11, 11)).unwrap();
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..11, 0..11]).start(),
            arr,
        ).unwrap();

        // longitude
        let array = ArrayBuilder::new(
            vec![11, 11],
            DataType::Float64,
            vec![4, 4].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .bytes_to_bytes_codecs(vec![
            Arc::new(get_lz4_compressor()),
        ])
        .build(store.clone(), "/lon")
        .unwrap();
        array.store_metadata().unwrap();

        let mut v = vec![-110. , -109.9, -109.8, -109.7, -109.6, -109.5, -109.4, -109.3, -109.2, -109.1, -109.];
        for _ in 0..10 {
            v.extend_from_within(..11);
        }
        
        let mut arr: Array2<f64> = Array::from_vec(v).into_shape_with_order((11, 11)).unwrap();
        arr.swap_axes(1, 0);
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..11, 0..11]).start(),
            arr,
        ).unwrap();

        // float data
        let array = ArrayBuilder::new(
            vec![11, 11],
            DataType::Float64,
            vec![4, 4].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .build(store.clone(), "/float_data")
        .unwrap();
        array.store_metadata().unwrap();

        let arr: Array2<f64> = Array::range(0.0, 121.0, 1.0).into_shape_with_order((11, 11)).unwrap();
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..11, 0..11]).start(),
            arr,
        ).unwrap();

        store_wrapper
    }

    #[fixture]
    pub(crate) fn store_lat_lon_broadcastable(dummy_name: String) -> StoreWrapper {
        // create the store
        let store_wrapper = StoreWrapper::new(dummy_name);
        let store = store_wrapper.get_store();

        // latitude
        let array = ArrayBuilder::new(
            vec![11],
            DataType::Float64,
            vec![4].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .bytes_to_bytes_codecs(vec![
            Arc::new(get_lz4_compressor()),
        ])
        .attributes(serde_json::from_str(
            r#"{
                "broadcast_params": {
                    "target_shape": [11, 11],
                    "target_chunks": [4, 4],
                    "axis": 1
                }
            }"#
        ).unwrap())
        .build(store.clone(), "/lat")
        .unwrap();
        array.store_metadata().unwrap();

        let v = vec![38. , 38.1, 38.2, 38.3, 38.4, 38.5, 38.6, 38.7, 38.8, 38.9, 39.];
        let arr: Array1<f64> = Array::from_vec(v);
        array.store_array_subset_ndarray(
            &[0],
            arr,
        ).unwrap();

        // longitude
        let array = ArrayBuilder::new(
            vec![11],
            DataType::Float64,
            vec![4].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .bytes_to_bytes_codecs(vec![
            Arc::new(get_lz4_compressor()),
        ])
        .attributes(serde_json::from_str(
            r#"{
                "broadcast_params": {
                    "target_shape": [11, 11],
                    "target_chunks": [4, 4],
                    "axis": 0
                }
            }"#
        ).unwrap())
        .build(store.clone(), "/lon")
        .unwrap();
        array.store_metadata().unwrap();

        let v = vec![-110. , -109.9, -109.8, -109.7, -109.6, -109.5, -109.4, -109.3, -109.2, -109.1, -109.];
        let arr: Array1<f64> = Array::from_vec(v);
        array.store_array_subset_ndarray(
            &[0],
            arr,
        ).unwrap();

        // float data
        let array = ArrayBuilder::new(
            vec![11, 11],
            DataType::Float64,
            vec![4, 4].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .build(store.clone(), "/float_data")
        .unwrap();
        array.store_metadata().unwrap();

        let arr: Array2<f64> = Array::range(0.0, 121.0, 1.0).into_shape_with_order((11, 11)).unwrap();
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..11, 0..11]).start(),
            arr,
        ).unwrap();

        store_wrapper
    }

    #[fixture]
    pub(crate) fn store_partial_sharding(dummy_name: String) -> StoreWrapper {
        // create the store
        let store_wrapper = StoreWrapper::new(dummy_name);
        let store = store_wrapper.get_store();

        // float data with sharding
        let sharding_chunk = vec![3, 2];
        let mut codec_builder = ShardingCodecBuilder::new(sharding_chunk.as_slice().try_into().unwrap());
        codec_builder.bytes_to_bytes_codecs(vec![Arc::new(get_lz4_compressor())]);
        let array = ArrayBuilder::new(
            vec![11, 10],
            DataType::Float64,
            vec![6, 4].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .array_to_bytes_codec(Arc::new(codec_builder.build()))
        .build(store.clone(), "/float_data_sharded")
        .unwrap();
        array.store_metadata().unwrap();

        let arr: Array2<f64> = Array::range(0.0, 110.0, 1.0).into_shape_with_order((11, 10)).unwrap();
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..11, 0..10]).start(),
            arr,
        ).unwrap();

        // float data without sharding
        let array = ArrayBuilder::new(
            vec![11, 10],
            DataType::Float64,
            vec![6, 4].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .build(store.clone(), "/float_data_not_sharded")
        .unwrap();
        array.store_metadata().unwrap();

        let arr: Array2<f64> = Array::range(0.0, 110.0, 1.0).into_shape_with_order((11, 10)).unwrap();
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..11, 0..10]).start(),
            arr,
        ).unwrap();

        store_wrapper
    }

    #[fixture]
    pub(crate) fn store_partial_sharding_3d(dummy_name: String) -> StoreWrapper {
        // create the store
        let store_wrapper = StoreWrapper::new(dummy_name);
        let store = store_wrapper.get_store();

        // float data with sharding
        let sharding_chunk = vec![3, 2, 4];
        let mut codec_builder = ShardingCodecBuilder::new(sharding_chunk.as_slice().try_into().unwrap());
        codec_builder.bytes_to_bytes_codecs(vec![Arc::new(get_lz4_compressor())]);
        let array = ArrayBuilder::new(
            vec![11, 10, 9],
            DataType::Float64,
            vec![6, 4, 8].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .array_to_bytes_codec(Arc::new(codec_builder.build()))
        .build(store.clone(), "/float_data_sharded")
        .unwrap();
        array.store_metadata().unwrap();

        let arr: Array3<f64> = Array::range(0.0, 990.0, 1.0).into_shape_with_order((11, 10, 9)).unwrap();
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..11, 0..10, 0..9]).start(),
            arr,
        ).unwrap();

        // float data without sharding
        let array = ArrayBuilder::new(
            vec![11, 10, 9],
            DataType::Float64,
            vec![6, 4, 8].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .build(store.clone(), "/float_data_not_sharded")
        .unwrap();
        array.store_metadata().unwrap();

        let arr: Array3<f64> = Array::range(0.0, 990.0, 1.0).into_shape_with_order((11, 10, 9)).unwrap();
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..11, 0..10, 0..9]).start(),
            arr,
        ).unwrap();

        store_wrapper
    }

    #[fixture]
    pub(crate) fn store_lat_lon_with_partition(dummy_name: String) -> StoreWrapper {
        // create the store
        let store_wrapper = StoreWrapper::new(dummy_name);
        let store = store_wrapper.get_store();

        //var=1, other_var=a
        // latitude
        let array = ArrayBuilder::new(
            vec![11, 11],
            DataType::Float64,
            vec![4, 4].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .bytes_to_bytes_codecs(vec![
            Arc::new(get_lz4_compressor()),
        ])
        .build(store.clone(), "/var=1/other_var=a/lat")
        .unwrap();
        array.store_metadata().unwrap();

        let mut v = vec![38. , 38.1, 38.2, 38.3, 38.4, 38.5, 38.6, 38.7, 38.8, 38.9, 39.];
        for _ in 0..10 {
            v.extend_from_within(..11);
        }
        
        let arr: Array2<f64> = Array::from_vec(v).into_shape_with_order((11, 11)).unwrap();
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..11, 0..11]).start(),
            arr,
        ).unwrap();

        // longitude
        let array = ArrayBuilder::new(
            vec![11, 11],
            DataType::Float64,
            vec![4, 4].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .bytes_to_bytes_codecs(vec![
            Arc::new(get_lz4_compressor()),
        ])
        .build(store.clone(), "/var=1/other_var=a/lon")
        .unwrap();
        array.store_metadata().unwrap();

        let mut v = vec![-110. , -109.9, -109.8, -109.7, -109.6, -109.5, -109.4, -109.3, -109.2, -109.1, -109.];
        for _ in 0..10 {
            v.extend_from_within(..11);
        }
        
        let mut arr: Array2<f64> = Array::from_vec(v).into_shape_with_order((11, 11)).unwrap();
        arr.swap_axes(1, 0);
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..11, 0..11]).start(),
            arr,
        ).unwrap();

        //var=2, other_var=a
        // latitude
        let array = ArrayBuilder::new(
            vec![11, 11],
            DataType::Float64,
            vec![4, 4].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .bytes_to_bytes_codecs(vec![
            Arc::new(get_lz4_compressor()),
        ])
        .build(store.clone(), "/var=2/other_var=a/lat")
        .unwrap();
        array.store_metadata().unwrap();

        let mut v = vec![39. , 39.1, 39.2, 39.3, 39.4, 39.5, 39.6, 39.7, 39.8, 39.9, 40.];
        for _ in 0..10 {
            v.extend_from_within(..11);
        }
        
        let arr: Array2<f64> = Array::from_vec(v).into_shape_with_order((11, 11)).unwrap();
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..11, 0..11]).start(),
            arr,
        ).unwrap();

        // longitude
        let array = ArrayBuilder::new(
            vec![11, 11],
            DataType::Float64,
            vec![4, 4].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .bytes_to_bytes_codecs(vec![
            Arc::new(get_lz4_compressor()),
        ])
        .build(store.clone(), "/var=2/other_var=a/lon")
        .unwrap();
        array.store_metadata().unwrap();

        let mut v = vec![-110. , -109.9, -109.8, -109.7, -109.6, -109.5, -109.4, -109.3, -109.2, -109.1, -109.];
        for _ in 0..10 {
            v.extend_from_within(..11);
        }
        
        let mut arr: Array2<f64> = Array::from_vec(v).into_shape_with_order((11, 11)).unwrap();
        arr.swap_axes(1, 0);
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..11, 0..11]).start(),
            arr,
        ).unwrap();

        //var=1, other_var=b
        // latitude
        let array = ArrayBuilder::new(
            vec![11, 11],
            DataType::Float64,
            vec![4, 4].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .bytes_to_bytes_codecs(vec![
            Arc::new(get_lz4_compressor()),
        ])
        .build(store.clone(), "/var=1/other_var=b/lat")
        .unwrap();
        array.store_metadata().unwrap();

        let mut v = vec![38. , 38.1, 38.2, 38.3, 38.4, 38.5, 38.6, 38.7, 38.8, 38.9, 39.];
        for _ in 0..10 {
            v.extend_from_within(..11);
        }
        
        let arr: Array2<f64> = Array::from_vec(v).into_shape_with_order((11, 11)).unwrap();
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..11, 0..11]).start(),
            arr,
        ).unwrap();

        // longitude
        let array = ArrayBuilder::new(
            vec![11, 11],
            DataType::Float64,
            vec![4, 4].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .bytes_to_bytes_codecs(vec![
            Arc::new(get_lz4_compressor()),
        ])
        .build(store.clone(), "/var=1/other_var=b/lon")
        .unwrap();
        array.store_metadata().unwrap();

        let mut v = vec![-108.9, -108.8, -108.7, -108.6, -108.5, -108.4, -108.3, -108.2, -108.1, -108.0, -107.9];
        for _ in 0..10 {
            v.extend_from_within(..11);
        }
        
        let mut arr: Array2<f64> = Array::from_vec(v).into_shape_with_order((11, 11)).unwrap();
        arr.swap_axes(1, 0);
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..11, 0..11]).start(),
            arr,
        ).unwrap();

        //var=2, other_var=b
        // latitude
        let array = ArrayBuilder::new(
            vec![11, 11],
            DataType::Float64,
            vec![4, 4].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .bytes_to_bytes_codecs(vec![
            Arc::new(get_lz4_compressor()),
        ])
        .build(store.clone(), "/var=2/other_var=b/lat")
        .unwrap();
        array.store_metadata().unwrap();

        let mut v = vec![39. , 39.1, 39.2, 39.3, 39.4, 39.5, 39.6, 39.7, 39.8, 39.9, 40.];
        for _ in 0..10 {
            v.extend_from_within(..11);
        }
        
        let arr: Array2<f64> = Array::from_vec(v).into_shape_with_order((11, 11)).unwrap();
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..11, 0..11]).start(),
            arr,
        ).unwrap();

        // longitude
        let array = ArrayBuilder::new(
            vec![11, 11],
            DataType::Float64,
            vec![4, 4].try_into().unwrap(),
            FillValue::new(vec![0; 8]),
        )
        .bytes_to_bytes_codecs(vec![
            Arc::new(get_lz4_compressor()),
        ])
        .build(store.clone(), "/var=2/other_var=b/lon")
        .unwrap();
        array.store_metadata().unwrap();

        let mut v = vec![-108.9, -108.8, -108.7, -108.6, -108.5, -108.4, -108.3, -108.2, -108.1, -108.0, -107.9];
        for _ in 0..10 {
            v.extend_from_within(..11);
        }
        
        let mut arr: Array2<f64> = Array::from_vec(v).into_shape_with_order((11, 11)).unwrap();
        arr.swap_axes(1, 0);
        array.store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..11, 0..11]).start(),
            arr,
        ).unwrap();

        store_wrapper
    }

    pub(crate) fn validate_names_and_types(targets: &HashMap<String, arrow_schema::DataType>, rec: &RecordBatch) {
        let mut target_cols: Vec<&String> = targets.keys().collect();
        let schema = rec.schema();
        let from_rec: Vec<&String> = schema.fields.iter().map(|f| f.name()).collect();

        target_cols.sort();
        assert_eq!(from_rec, target_cols);

        for field in schema.fields.iter() {
            assert_eq!(field.data_type(), targets.get(field.name()).unwrap());
        }
    }

    pub(crate) fn validate_bool_column(col_name: &str, rec: &RecordBatch, targets: &[bool]) {
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

    pub(crate) fn compare_values<T>(col_name1: &str, col_name2: &str, rec: &RecordBatch)
    where
        T: ArrowPrimitiveType,
    {
        let mut vals1 = None;
        let mut vals2 = None;
        for (idx, col) in enumerate(rec.schema().fields.iter()) {
            if col.name().as_str() == col_name1 {
                vals1 = Some(rec.column(idx).as_primitive::<T>().values())
            } else if col.name().as_str() == col_name2 {
                vals2 = Some(rec.column(idx).as_primitive::<T>().values())
            }
        }

        if let (Some(vals1), Some(vals2)) = (vals1, vals2) {
            assert_eq!(vals1, vals2);
            return;
        }

        panic!("columns not found");
    }

    // create a test filter
    pub(crate) fn create_filter() -> ZarrChunkFilter {
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

    // don't need that for now, I commented out tests with string data.
    // pub(crate) fn validate_string_column(col_name: &str, rec: &RecordBatch, targets: &[&str]) {
    //     let mut matched = false;
    //     for (idx, col) in enumerate(rec.schema().fields.iter()) {
    //         if col.name().as_str() == col_name {
    //             assert_eq!(
    //                 rec.column(idx).as_string(),
    //                 &StringArray::from(targets.to_vec()),
    //             );
    //             matched = true;
    //         }
    //     }
    //     assert!(matched);
    // }
}
