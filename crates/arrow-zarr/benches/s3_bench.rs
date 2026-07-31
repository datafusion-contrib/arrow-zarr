mod shared;

use std::collections::HashMap;
use std::env;
use std::sync::Arc;

use criterion::{criterion_group, criterion_main, Criterion};
use datafusion::datasource::listing::ListingTableUrl;
use futures::{StreamExt, TryStreamExt};
use icechunk::config::{S3Credentials, S3Options};
use icechunk::{ObjectStorage, Repository};
use object_store::aws::{AmazonS3, AmazonS3Builder};
use object_store::path::Path;
use object_store::ObjectStore;
use shared::{run_benchmark_group, CloudStorageBenchBackend, TestFixture};
use zarrs_icechunk::AsyncIcechunkStore;

struct S3BenchBackend {
    prefix: String,
    store: AmazonS3,
}

impl S3BenchBackend {
    async fn new(bucket: String, prefix: String) -> Self {
        let store = AmazonS3Builder::from_env()
            .with_bucket_name(bucket)
            .build()
            .unwrap();
        Self { prefix, store }
    }
}

#[async_trait::async_trait]
impl CloudStorageBenchBackend for S3BenchBackend {
    async fn create_icechunk_store(url: &str) -> Arc<AsyncIcechunkStore> {
        let listing_url = ListingTableUrl::parse(url).unwrap();
        let bucket = listing_url
            .object_store()
            .as_str()
            .replace("s3://", "")
            .trim_end_matches("/")
            .to_string();

        let credentials = S3Credentials::FromEnv;
        // S3Options is #[non_exhaustive]; only override the region.
        let mut config = S3Options::default();
        if let Ok(region) = env::var("AWS_DEFAULT_REGION") {
            config = config.with_region(region);
        }

        let store = ObjectStorage::new_s3(
            bucket,
            Some(listing_url.prefix().as_ref().to_string()),
            Some(credentials),
            Some(config),
            Vec::new(),
            Vec::new(),
        )
        .await
        .unwrap();

        let repo = Repository::create(None, Arc::new(store), HashMap::new(), None, true)
            .await
            .unwrap();
        let session = repo.writable_session("main").await.unwrap();

        Arc::new(AsyncIcechunkStore::new(session))
    }

    async fn cleanup(&self) {
        let prefix = Path::from(self.prefix.clone());
        let locations = self
            .store
            .list(Some(&prefix))
            .map_ok(|meta| meta.location)
            .boxed();

        self.store
            .delete_stream(locations)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
    }
}

fn s3_benchmark_group(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let url = "s3://zarr-unit-tests/test_data_s3";

    let fixture = rt.block_on(async {
        let backend = S3BenchBackend::new("zarr-unit-tests".into(), "test_data_s3".into()).await;
        TestFixture::new(backend, url).await
    });

    run_benchmark_group(fixture.get_session(), c, "s3_benchmarks");
}

criterion_group!(s3_benches, s3_benchmark_group);
criterion_main!(s3_benches);
