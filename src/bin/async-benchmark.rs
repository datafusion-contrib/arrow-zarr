use arrow_zarr::async_reader::{ZarrPath, ZarrRecordBatchStreamBuilderNonBlocking};
use futures::TryStreamExt;
use object_store::{local::LocalFileSystem, path::Path};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

fn get_v2_test_data_path(zarr_store: String) -> ZarrPath {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test-data/data/zarr/v2_data")
        .join(zarr_store);
    ZarrPath::new(
        Arc::new(LocalFileSystem::new()),
        Path::from_absolute_path(p).unwrap(),
    )
}

#[tokio::main]
async fn main() {
    let zp = get_v2_test_data_path("lat_lon_example.zarr".to_string());
    let stream_builder = ZarrRecordBatchStreamBuilderNonBlocking::new(zp);

    let stream = stream_builder.build().await.unwrap();
    let now = Instant::now();
    let _: Vec<_> = stream.try_collect().await.unwrap();

    println!("{:?}", now.elapsed());
}
