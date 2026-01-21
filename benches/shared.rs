use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_zarr::table::ZarrTableFactory;
use criterion::Criterion;
use datafusion::dataframe::DataFrame;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use ndarray::{Array, Array2};
use zarrs::array::{codec, ArrayBuilder, DataType, FillValue};
use zarrs::array_subset::ArraySubset;
use zarrs_icechunk::AsyncIcechunkStore;
use zarrs_storage::AsyncReadableWritableListableStorageTraits;

pub fn get_lz4_compressor() -> codec::BloscCodec {
    codec::BloscCodec::new(
        codec::bytes_to_bytes::blosc::BloscCompressor::LZ4,
        5.try_into().unwrap(),
        Some(0),
        codec::bytes_to_bytes::blosc::BloscShuffleMode::NoShuffle,
        Some(1),
    )
    .unwrap()
}

pub async fn write_data_to_store(
    store: Arc<dyn AsyncReadableWritableListableStorageTraits>,
    start_var_idx: usize,
    prefix: &str,
) {
    let n = 512;
    let fill_value: i64 = 0;
    let mut array_builder = ArrayBuilder::new(
        vec![n, n],
        [8, 8],
        DataType::Int64,
        FillValue::from(fill_value),
    );

    let mut builder_ref = &mut array_builder;
    let codec = get_lz4_compressor();
    builder_ref = builder_ref.bytes_to_bytes_codecs(vec![Arc::new(codec)]);

    let prefix = if prefix.is_empty() {
        prefix
    } else {
        &format!("/{}", prefix)
    };
    
    for var_idx in start_var_idx..(start_var_idx + 8) {
        let arr = builder_ref
            .build(store.clone(), &format!("{}/var{}", prefix, var_idx))
            .unwrap();
        arr.async_store_metadata().await.unwrap();

        let arr_data: Array2<i64> = Array::from_vec((0..(n * n) as i64).step_by(1).collect())
            .into_shape_with_order((n as usize, n as usize))
            .unwrap();
        arr.async_store_array_subset_ndarray(
            ArraySubset::new_with_ranges(&[0..n, 0..n]).start(),
            arr_data,
        )
        .await
        .unwrap();
    }
}

async fn run_query(query: &str, session: SessionContext) {
    let df: DataFrame = session.sql(query).await.unwrap();
    let _: Vec<RecordBatch> = df.collect().await.unwrap();
}

pub fn run_benchmark_group(session: &SessionContext, c: &mut Criterion, group_name: &str) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group(group_name);
    group.sample_size(20);

    let session = session.clone();
    
    let query1 = "
        SELECT t1.*, t2.*
        FROM zarr_table as t1
        JOIN zarr_table as t2
            ON t1.var1 % 12 = 0
            AND t1.var1 < t2.var1 + 1
            AND t1.var1 >= t2.var1 - 1
    ";

    group.bench_function("join_benchmark", |b| {
        b.to_async(&rt)
            .iter(|| async { run_query(black_box(query1), black_box(session.clone())).await })
    });

    let query2 = "
        SELECT *
        FROM zarr_table

        UNION ALL

        SELECT *
        FROM zarr_table
    ";
    
    group.bench_function("union_benchmark", |b| {
        b.to_async(&rt)
            .iter(|| async { run_query(black_box(query2), black_box(session.clone())).await })
    });
    
    group.finish();
}


#[async_trait::async_trait]
pub trait CloudStorageBenchBackend: Send + Sync {
    // must be implemented for different cloud storage providers to enable benchmarking
    async fn create_icechunk_store(url: &str) -> Arc<AsyncIcechunkStore>;
    async fn cleanup(&self);
    fn bucket(&self) -> &str;
    fn prefix(&self) -> &str;
}

pub struct TestFixture<B: CloudStorageBenchBackend> {
    backend: B,
    session: SessionContext,
}

impl<B: CloudStorageBenchBackend> TestFixture<B> {
    pub async fn new(backend: B, url: &str) -> Self {
        let store = B::create_icechunk_store(url).await;
        write_data_to_store(store.clone(), 1, "").await;
        let _ = store
            .session()
            .write()
            .await
            .commit("Test data for benchmarking", None)
            .await
            .unwrap();

        let mut state = SessionStateBuilder::new().build();
        state
            .table_factories_mut()
            .insert("ICECHUNK_REPO".into(), Arc::new(ZarrTableFactory {}));
        let session = SessionContext::new_with_state(state.clone());

        let query = format!(
            "
            CREATE EXTERNAL TABLE zarr_table
            STORED AS ICECHUNK_REPO LOCATION '{}'
            ",
            url
        );
        session.sql(&query).await.unwrap();

        Self { backend, session }
    }

    pub fn get_session(&self) -> &SessionContext {
        &self.session
    }
}

impl<B: CloudStorageBenchBackend> Drop for TestFixture<B> {
    fn drop(&mut self) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            self.backend.cleanup().await;
        });
    }
}
