mod shared;

use std::collections::HashMap;
use std::env;
use std::sync::Arc;

use aws_config::{self, BehaviorVersion};
use aws_sdk_s3::types::{Delete, ObjectIdentifier};
use aws_sdk_s3::Client as S3Client;
use criterion::{criterion_group, criterion_main, Criterion};
use datafusion::datasource::listing::ListingTableUrl;
use icechunk::config::{S3Credentials, S3Options};
use icechunk::{ObjectStorage, Repository};
use zarrs_icechunk::AsyncIcechunkStore;

use shared::{CloudStorageBenchBackend, TestFixture, run_benchmark_group};

struct S3BenchBackend {
    bucket: String,
    prefix: String,
    client: S3Client,
}

impl S3BenchBackend {
    async fn new(bucket: String, prefix: String) -> Self {
        let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
        let client = S3Client::new(&config);
        Self {
            bucket,
            prefix,
            client,
        }
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

    async fn cleanup(&self) {
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
    }

    fn bucket(&self) -> &str {
        &self.bucket
    }

    fn prefix(&self) -> &str {
        &self.prefix
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
