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

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::error::Result as DfResult;
use datafusion::prelude::SessionContext;

use super::config::ZarrTableUrl;
use super::table_provider::ZarrTable;

// Extension trait adding a convenience method to register a zarr- or
// icechunk-backed table directly on a [`SessionContext`], mirroring
// datafusion's built-in `register_csv` / `register_parquet` helpers.
//
// The caller builds a [`ZarrTableUrl`] with [`ZarrUrlBuilder`] /
// [`IcechunkUrlBuilder`] so a single method covers both plain zarr
// stores and icechunk repos.
#[async_trait]
pub trait RegisterZarr {
    async fn register_zarr(&self, name: &str, table_url: ZarrTableUrl) -> DfResult<()>;
}

#[async_trait]
impl RegisterZarr for SessionContext {
    async fn register_zarr(&self, name: &str, table_url: ZarrTableUrl) -> DfResult<()> {
        let table = ZarrTable::try_new(table_url).await?;
        self.register_table(name, Arc::new(table))?;
        Ok(())
    }
}
