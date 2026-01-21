mod shared;

use std::collections::HashMap;
use std::sync::Arc;

use criterion::{criterion_group, criterion_main, Criterion};
use datafusion::datasource::listing::ListingTableUrl;
use google_cloud_storage::http::objects::delete::DeleteObjectRequest;
use google_cloud_storage::http::objects::list::ListObjectsRequest;
use icechunk::config::{GcsCredentials, GcsOptions};
use icechunk::{ObjectStorage, Repository};
use zarrs_icechunk::AsyncIcechunkStore;

use shared::{CloudStorageBackend, TestFixture, run_benchmark_group};

// ============================================================================
// GCS Backend Implementation
// ============================================================================

struct GCSBenchBackend {
    bucket: String,
    prefix: String,
    client: google_cloud_storage::client::Client,
}

impl GCSBenchBackend {
    async fn new(bucket: String, prefix: String) -> Self {
        let config = google_cloud_storage::client::ClientConfig::default()
            .with_auth()
            .await
            .unwrap();
        let client = google_cloud_storage::client::Client::new(config);
        
        Self {
            bucket,
            prefix,
            client,
        }
    }
}

#[async_trait::async_trait]
impl CloudStorageBackend for GCSBenchBackend {
    async fn create_store(url: &str) -> Arc<AsyncIcechunkStore> {
        let listing_url = ListingTableUrl::parse(url).unwrap();
        let bucket = listing_url
            .object_store()
            .as_str()
            .replace("gs://", "")
            .trim_end_matches("/")
            .to_string();

        let credentials = GcsCredentials::FromEnv;
        let config = GcsOptions {
            endpoint_url: None,
            anonymous: false,
            allow_http: false,
        };

        let store = ObjectStorage::new_gcs(
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

    async fn cleanup(&self) {

        let list_request = ListObjectsRequest {
            bucket: self.bucket.clone(),
            prefix: Some(self.prefix.clone()),
            ..Default::default()
        };

        let objects = self.client.list_objects(&list_request).await.unwrap();

        for obj in objects.items.unwrap_or_default() {
            let delete_request = DeleteObjectRequest {
                bucket: self.bucket.clone(),
                object: obj.name,
                ..Default::default()
            };
            let _ = self.client.delete_object(&delete_request).await;
        }
    }

    fn bucket(&self) -> &str {
        &self.bucket
    }

    fn prefix(&self) -> &str {
        &self.prefix
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
