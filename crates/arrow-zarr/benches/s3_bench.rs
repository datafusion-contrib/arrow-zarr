use std::collections::HashMap;
use std::env;
use std::hint::black_box;
use std::sync::Arc;

use arrow_zarr::table::ZarrTableFactory;
use aws_config::{self, BehaviorVersion};
use aws_sdk_s3::types::{Delete, ObjectIdentifier};
use aws_sdk_s3::Client;
use criterion::{criterion_group, criterion_main, Criterion};
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use icechunk::config::{S3Credentials, S3Options};
use icechunk::{ObjectStorage, Repository};
use ndarray::{Array, Array2};
use zarrs::array::{codec, ArrayBuilder, DataType, FillValue};
use zarrs::array_subset::ArraySubset;
use zarrs_icechunk::AsyncIcechunkStore;
use zarrs_storage::AsyncReadableWritableListableStorageTraits;

async fn create_s3_icechunk(url: &str) -> Arc<AsyncIcechunkStore> {
    let listing_url = ListingTableUrl::parse(url).unwrap();
    let bucket = listing_url
        .object_store()
        .as_str()
        .replace("s3://", "")
        .trim_end_matches("/")
        .to_string();

    let credentials = S3Credentials::FromEnv;
    let config = S3Options {
        region: env::var("AWS_DEFAULT_REGION").ok(),
        endpoint_url: None,
        anonymous: false,
        allow_http: false,
        force_path_style: false,
        network_stream_timeout_seconds: None,
        requester_pays: false,
    };

    let store = ObjectStorage::new_s3(
        bucket,
        Some(listing_url.prefix().as_ref().to_string()),
        Some(credentials),
        Some(config),
    )
    .await
    .unwrap();

    let repo = Repository::create(None, Arc::new(store), HashMap::new())
        .await
        .unwrap();
    let session = repo.writable_session("main").await.unwrap();

    Arc::new(AsyncIcechunkStore::new(session))
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

async fn write_data_to_store(
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

struct S3TestFixture {
    bucket: String,
    prefix: String,
    client: Client,
    session: SessionContext,
}

impl S3TestFixture {
    fn new() -> Self {
        let url = "s3://zarr-unit-tests/test_data_1";
        let rt = tokio::runtime::Runtime::new().unwrap();

        let (client, session) = rt.block_on(async {
            let store = create_s3_icechunk(url).await;
            write_data_to_store(store.clone(), 1, "").await;
            let _ = store
                .session()
                .write()
                .await
                .commit("some test data", None)
                .await
                .unwrap();

            let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
            let client = Client::new(&config);

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

            (client, session)
        });

        Self {
            bucket: "zarr-unit-tests".into(),
            prefix: "test_data_1".into(),
            client,
            session,
        }
    }

    fn get_session(&self) -> &SessionContext {
        &self.session
    }
}

impl Drop for S3TestFixture {
    fn drop(&mut self) {
        let rt = tokio::runtime::Runtime::new().unwrap();

        rt.block_on(async {
            let objects = self
                .client
                .list_objects_v2()
                .bucket(self.bucket.clone())
                .prefix(self.prefix.clone())
                .send()
                .await
                .unwrap();

            let to_delete: Vec<_> = objects
                .contents()
                .iter()
                .filter_map(|obj| {
                    obj.key()
                        .map(|k| ObjectIdentifier::builder().key(k).build().unwrap())
                })
                .collect();

            if !to_delete.is_empty() {
                let delete = Delete::builder()
                    .set_objects(Some(to_delete))
                    .build()
                    .unwrap();
                self.client
                    .delete_objects()
                    .bucket(self.bucket.clone())
                    .delete(delete)
                    .send()
                    .await
                    .unwrap();
            }
        })
    }
}

async fn run_query(query: &str, session: SessionContext) {
    let df = session.sql(query).await.unwrap();
    let _ = df.collect().await.unwrap();
}

fn benchmark_query(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let s3_fixture = S3TestFixture::new();

    let mut group = c.benchmark_group("my_group");
    group.sample_size(20);

    let session = s3_fixture.get_session().clone();
    let query = "
                SELECT t1.*, t2.*
                FROM zarr_table as t1
                JOIN zarr_table as t2
                    ON t1.var1 % 12 = 0
                    AND t1.var1 < t2.var1 + 1
                    AND t1.var1 >= t2.var1 - 1
                ";

    group.bench_function("benchmark 1", |b| {
        b.to_async(&rt)
            .iter(|| async { run_query(black_box(query), black_box(session.clone())).await })
    });

    let query = "
                SELECT *
                FROM zarr_table

                UNION ALL

                SELECT *
                FROM zarr_table
                ";
    group.bench_function("benchmark 2", |b| {
        b.to_async(&rt)
            .iter(|| async { run_query(black_box(query), black_box(session.clone())).await })
    });
}

criterion_group!(benches, benchmark_query);
criterion_main!(benches);
