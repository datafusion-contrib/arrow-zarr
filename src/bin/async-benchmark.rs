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
