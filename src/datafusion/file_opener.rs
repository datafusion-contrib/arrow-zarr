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

use arrow_schema::ArrowError;
use datafusion::{datasource::physical_plan::FileOpener, error::DataFusionError};
use futures::{StreamExt, TryStreamExt};

use crate::async_reader::{ZarrPath, ZarrRecordBatchStreamBuilder};

use super::config::ZarrConfig;

pub struct ZarrFileOpener {
    config: ZarrConfig,
}

impl ZarrFileOpener {
    pub fn new(config: ZarrConfig) -> Self {
        Self { config }
    }
}

impl FileOpener for ZarrFileOpener {
    fn open(
        &self,
        file_meta: datafusion::datasource::physical_plan::FileMeta,
    ) -> datafusion::error::Result<datafusion::datasource::physical_plan::FileOpenFuture> {
        let config = self.config.clone();

        Ok(Box::pin(async move {
            let zarr_path = ZarrPath::new(config.object_store, file_meta.object_meta.location);

            let rng = file_meta.range.map(|r| (r.start as usize, r.end as usize));
            let batch_reader = ZarrRecordBatchStreamBuilder::new(zarr_path)
                .build_partial_reader(rng)
                .await
                .map_err(|_| {
                    DataFusionError::Execution("Error creating zarr reader".to_string())
                })?;

            let stream = batch_reader
                .map_err(|_| ArrowError::ComputeError("Error reading zarr".to_string()));

            Ok(stream.boxed())
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::datasource::physical_plan::FileMeta;
    use object_store::{local::LocalFileSystem, path::Path, ObjectMeta};

    use crate::tests::get_test_v2_data_path;

    use super::*;

    #[tokio::test]
    async fn test_open() {
        let local_fs = LocalFileSystem::new();

        let test_data = get_test_v2_data_path("lat_lon_example.zarr".to_string());

        let config = ZarrConfig {
            object_store: Arc::new(local_fs),
        };

        let opener = ZarrFileOpener::new(config);

        let file_meta = FileMeta {
            object_meta: ObjectMeta {
                location: Path::from_filesystem_path(test_data).unwrap(),
                last_modified: chrono::Utc::now(),
                size: 0,
                e_tag: None,
                version: None,
            },
            range: None,
            extensions: None,
        };

        let open_future = opener.open(file_meta).unwrap();

        let stream = open_future.await.unwrap();

        let batches: Vec<_> = stream.try_collect().await.unwrap();

        assert_eq!(batches.len(), 9);
    }
}
