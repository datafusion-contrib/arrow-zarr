mod shared;

use std::collections::HashMap;
use std::sync::Arc;

use criterion::{criterion_group, criterion_main, Criterion};
use datafusion::datasource::listing::ListingTableUrl;
use icechunk::config::GcsCredentials;
use icechunk::{ObjectStorage, Repository};
use zarrs_icechunk::AsyncIcechunkStore;

use shared::{CloudStorageBenchBackend, TestFixture, run_benchmark_group};

// ============================================================================
// GCS Backend Implementation
// ============================================================================

struct GCSBenchBackend {
    _bucket: String,
    _prefix: String,
}

impl GCSBenchBackend {
    async fn new(bucket: String, prefix: String) -> Self {
        Self {
            _bucket: bucket,
            _prefix: prefix,
        }
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
            )
            .await
            .unwrap()
        );

        let repo = match Repository::open(None, store.clone(), HashMap::new()).await {
            Ok(repo) => repo,
            Err(_) => {
                Repository::create(None, store, HashMap::new())
                    .await
                    .unwrap()
            }
        };
        let session = repo.writable_session("main").await.unwrap();

        Arc::new(AsyncIcechunkStore::new(session))
    }

    async fn cleanup(&self) {
        // Cleanup is handled by the TestFixture Drop implementation
        // which uses the icechunk store to clean up resources
    }

    fn bucket(&self) -> &str {
        &self._bucket
    }

    fn prefix(&self) -> &str {
        &self._prefix
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
