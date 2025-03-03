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
use datafusion_physical_expr::PhysicalExpr;
use futures::{StreamExt, TryStreamExt};
use std::sync::Arc;

use crate::{
    async_reader::{ZarrPath, ZarrReadAsync, ZarrRecordBatchStreamBuilder},
    reader::ZarrProjection,
};

use super::config::ZarrConfig;
use super::helpers::build_row_filter;

pub struct ZarrFileOpener {
    config: ZarrConfig,
    filters: Option<Arc<dyn PhysicalExpr>>,
}

impl ZarrFileOpener {
    pub fn new(config: ZarrConfig, filters: Option<Arc<dyn PhysicalExpr>>) -> Self {
        Self { config, filters }
    }
}

impl FileOpener for ZarrFileOpener {
    fn open(
        &self,
        file_meta: datafusion::datasource::physical_plan::FileMeta,
    ) -> datafusion::error::Result<datafusion::datasource::physical_plan::FileOpenFuture> {
        let config = self.config.clone();

        let filters_to_pushdown = self.filters.clone();
        Ok(Box::pin(async move {
            let zarr_path = ZarrPath::new(config.object_store, file_meta.object_meta.location);
            let rng = file_meta.range.map(|r| (r.start as usize, r.end as usize));
            let projection = ZarrProjection::from(config.projection.as_ref());
            let mut batch_reader_builder =
                ZarrRecordBatchStreamBuilder::new(zarr_path.clone()).with_projection(projection);
            if let Some(filters) = filters_to_pushdown {
                let file_schema = zarr_path
                    .get_zarr_metadata()
                    .await
                    .map_err(|e| DataFusionError::External(Box::new(e)))?
                    .arrow_schema()
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                let filters = build_row_filter(&filters, &file_schema)?;

                if let Some(filters) = filters {
                    batch_reader_builder = batch_reader_builder.with_filter(filters);
                }
            }
            let batch_reader = batch_reader_builder
                .build_partial_reader(rng)
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            let stream = batch_reader.map_err(|e| ArrowError::from_external_error(Box::new(e)));
            Ok(stream.boxed())
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::{error::Error, sync::Arc};

    use datafusion::datasource::physical_plan::FileMeta;
    use object_store::{local::LocalFileSystem, path::Path, ObjectMeta};

    use crate::test_utils::{store_lat_lon, StoreWrapper};
    use rstest::*;

    use super::*;

    #[rstest]
    #[tokio::test]
    async fn test_open(
        #[with("ftest_open".to_string())] store_lat_lon: StoreWrapper,
    ) -> Result<(), Box<dyn Error>> {
        let local_fs = LocalFileSystem::new();

        let test_data = store_lat_lon.store_path();

        let config = ZarrConfig::new(Arc::new(local_fs));
        let opener = ZarrFileOpener::new(config, None);

        let file_meta = FileMeta {
            object_meta: ObjectMeta {
                location: Path::from_filesystem_path(test_data)?,
                last_modified: chrono::Utc::now(),
                size: 0,
                e_tag: None,
                version: None,
            },
            range: None,
            extensions: None,
        };

        let open_future = opener.open(file_meta)?;
        let stream = open_future.await?;
        let batches: Vec<_> = stream.try_collect().await?;

        assert_eq!(batches.len(), 9);

        Ok(())
    }
}
