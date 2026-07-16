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

use super::table_provider::ZarrTable;

// Extension trait adding convenience methods to register zarr tables directly
// on a [`SessionContext`], mirroring datafusion's built-in `register_csv` /
// `register_parquet` helpers.
#[async_trait]
pub trait RegisterZarr {
    // Register a table backed by a zarr store at url, under the given
    // table name.
    async fn register_zarr(&self, name: &str, url: String) -> DfResult<()>;

    // Register a table backed by an icechunk repo at url, under the given
    // table name. Reads the tip of the main branch.
    #[cfg(feature = "icechunk")]
    async fn register_icechunk(&self, name: &str, url: String) -> DfResult<()>;

    // Register an icechunk repo at url, reading the tip of branch.
    #[cfg(feature = "icechunk")]
    async fn register_icechunk_branchtip(
        &self,
        name: &str,
        url: String,
        branch: String,
    ) -> DfResult<()>;

    // Register an icechunk repo at url, reading the given tag.
    #[cfg(feature = "icechunk")]
    async fn register_icechunk_reftag(&self, name: &str, url: String, tag: String) -> DfResult<()>;

    // Register an icechunk repo at url, reading the given snapshot_id.
    #[cfg(feature = "icechunk")]
    async fn register_icechunk_snapshot_id(
        &self,
        name: &str,
        url: String,
        snapshot_id: String,
    ) -> DfResult<()>;
}

#[async_trait]
impl RegisterZarr for SessionContext {
    async fn register_zarr(&self, name: &str, url: String) -> DfResult<()> {
        let table = ZarrTable::from_path(url).await?;
        self.register_table(name, Arc::new(table))?;
        Ok(())
    }

    #[cfg(feature = "icechunk")]
    async fn register_icechunk(&self, name: &str, url: String) -> DfResult<()> {
        let table = ZarrTable::from_path_to_icechunk(url).await?;
        self.register_table(name, Arc::new(table))?;
        Ok(())
    }

    #[cfg(feature = "icechunk")]
    async fn register_icechunk_branchtip(
        &self,
        name: &str,
        url: String,
        branch: String,
    ) -> DfResult<()> {
        let table = ZarrTable::from_path_to_icechunk(url)
            .await?
            .with_icechunk_branchtip(branch)?;
        self.register_table(name, Arc::new(table))?;
        Ok(())
    }

    #[cfg(feature = "icechunk")]
    async fn register_icechunk_reftag(&self, name: &str, url: String, tag: String) -> DfResult<()> {
        let table = ZarrTable::from_path_to_icechunk(url)
            .await?
            .with_icechunk_reftag(tag)?;
        self.register_table(name, Arc::new(table))?;
        Ok(())
    }

    #[cfg(feature = "icechunk")]
    async fn register_icechunk_snapshot_id(
        &self,
        name: &str,
        url: String,
        snapshot_id: String,
    ) -> DfResult<()> {
        let table = ZarrTable::from_path_to_icechunk(url)
            .await?
            .with_icechunk_snapshot_id(snapshot_id)?;
        self.register_table(name, Arc::new(table))?;
        Ok(())
    }
}
