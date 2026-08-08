mod shared;

use std::collections::HashMap;
use std::sync::Arc;

use criterion::{criterion_group, criterion_main, Criterion};
use datafusion::datasource::listing::ListingTableUrl;
use futures::{StreamExt, TryStreamExt};
use shared::{run_benchmark_group, CloudStorageBenchBackend, TestFixture};
use zarrs_icechunk::icechunk::config::GcsCredentials;
use zarrs_icechunk::icechunk::{ObjectStorage, Repository};
use zarrs_icechunk::AsyncIcechunkStore;
use zarrs_object_store::object_store::gcp::{GoogleCloudStorage, GoogleCloudStorageBuilder};
use zarrs_object_store::object_store::path::Path;
use zarrs_object_store::object_store::ObjectStore;

struct GCSBenchBackend {
    prefix: String,
    store: GoogleCloudStorage,
}

impl GCSBenchBackend {
    async fn new(bucket: String, prefix: String) -> Self {
        let store = GoogleCloudStorageBuilder::from_env()
            .with_bucket_name(bucket)
            .build()
            .unwrap();
        Self { prefix, store }
    }
}

#[async_trait::async_trait]
impl CloudStorageBenchBackend for GCSBenchBackend {
    async fn create_icechunk_store(url: &str) -> Arc<AsyncIcechunkStore> {
        let listing_url = ListingTableUrl::parse(url).unwrap();
        let bucket = listing_url
            .object_store()
            .as_str()
            .replace("gs://", "")
            .trim_end_matches("/")
            .to_string();

        let credentials = GcsCredentials::FromEnv;

        let store = Arc::new(
            ObjectStorage::new_gcs(
                bucket,
                Some(listing_url.prefix().as_ref().to_string()),
                Some(credentials),
                None,
                Vec::new(),
                Vec::new(),
            )
            .unwrap(),
        );

        let repo = Repository::create(None, store, HashMap::new(), None, true)
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

fn gcs_benchmark_group(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let url = "gs://zarr-unit-tests/test_data_gcs";

    let fixture = rt.block_on(async {
        let backend = GCSBenchBackend::new("zarr-unit-tests".into(), "test_data_gcs".into()).await;
        TestFixture::new(backend, url).await
    });

    run_benchmark_group(fixture.get_session(), c, "gcs_benchmarks");
}

criterion_group!(gcs_benches, gcs_benchmark_group);
criterion_main!(gcs_benches);
