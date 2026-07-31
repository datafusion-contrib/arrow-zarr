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

use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::{Schema, SchemaRef};
use datafusion::catalog::{Session, TableProviderFactory};
use datafusion::common::ToDFSchema;
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::logical_expr::utils::conjunction;
use datafusion::logical_expr::{CreateExternalTable, Expr, TableProviderFilterPushDown};
use datafusion::physical_expr::create_physical_expr;
use datafusion::physical_plan::ExecutionPlan;

#[cfg(feature = "icechunk")]
use super::config::IcechunkVersion;
use super::config::ZarrTableConfig;
use super::scanner::ZarrScan;
use crate::table::config::ZarrTableUrl;

/// The table provider for zarr stores.
pub struct ZarrTable {
    table_config: ZarrTableConfig,
}

impl Debug for ZarrTable {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl ZarrTable {
    pub fn new(table_config: ZarrTableConfig) -> Self {
        Self { table_config }
    }

    pub async fn from_path(path: String) -> DfResult<Self> {
        let table_url = ListingTableUrl::parse(path)?;
        let zarr_url = ZarrTableUrl::ZarrStore(table_url);
        let (schema, stats_base) = zarr_url.infer_schema().await?;
        let table_config = ZarrTableConfig::new(zarr_url, schema).with_stats_base(stats_base);
        Ok(Self { table_config })
    }

    #[cfg(feature = "icechunk")]
    pub async fn from_path_to_icechunk(path: String) -> DfResult<Self> {
        let table_url = ListingTableUrl::parse(path)?;
        let zarr_url = ZarrTableUrl::IcechunkRepo(table_url, IcechunkVersion::default());
        let (schema, stats_base) = zarr_url.infer_schema().await?;
        let table_config = ZarrTableConfig::new(zarr_url, schema).with_stats_base(stats_base);
        Ok(Self { table_config })
    }

    // Read the icechunk repo at the tip of the given branch.
    #[cfg(feature = "icechunk")]
    pub fn with_icechunk_branchtip(self, branch: String) -> DfResult<Self> {
        self.with_icechunk_version(IcechunkVersion::BranchTip(branch))
    }

    // Read the icechunk repo at the given tag.
    #[cfg(feature = "icechunk")]
    pub fn with_icechunk_reftag(self, tag: String) -> DfResult<Self> {
        self.with_icechunk_version(IcechunkVersion::RefTag(tag))
    }

    // Read the icechunk repo at the given snapshot id.
    #[cfg(feature = "icechunk")]
    pub fn with_icechunk_snapshot_id(self, snapshot_id: String) -> DfResult<Self> {
        self.with_icechunk_version(IcechunkVersion::SnapshotId(snapshot_id))
    }

    #[cfg(feature = "icechunk")]
    fn with_icechunk_version(mut self, version: IcechunkVersion) -> DfResult<Self> {
        self.table_config = self.table_config.with_icechunk_version(version)?;
        Ok(self)
    }
}

#[async_trait]
impl TableProvider for ZarrTable {
    fn schema(&self) -> SchemaRef {
        self.table_config.get_schema_ref()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    // there's no projected columns or partitions with the zarr data,
    // so really all we have are arrays that are present in all the data
    // chunks. there's not much to check here, we do use the filter
    // pushdown to avoid reading entire chunk, so pretty much all the
    // available arrays can be used as Inexact filters.
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion::error::Result<Vec<TableProviderFilterPushDown>> {
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        _limit: Option<usize>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        let mut filters_physical_expr = None;
        if let Some(filters) = conjunction(filters.to_vec()) {
            filters_physical_expr = Some(create_physical_expr(
                &filters,
                &self.table_config.get_schema_ref().to_dfschema()?,
                state.execution_props(),
            )?);
        }

        let mut config = self.table_config.clone();
        if let Some(proj) = projection {
            config = config.with_projection(proj.to_vec());
        }
        let scanner = ZarrScan::new(config, filters_physical_expr);

        Ok(Arc::new(scanner))
    }
}

/// The factory for the zarr table.
#[derive(Debug)]
pub struct ZarrTableFactory {}

#[async_trait]
impl TableProviderFactory for ZarrTableFactory {
    async fn create(
        &self,
        _state: &dyn Session,
        cmd: &CreateExternalTable,
    ) -> DfResult<Arc<dyn TableProvider>> {
        let table_url = match cmd.file_type.as_str() {
            "ZARR_STORE" => ZarrTableUrl::ZarrStore(ListingTableUrl::parse(&cmd.location)?),
            #[cfg(feature = "icechunk")]
            "ICECHUNK_REPO" => {
                let version = icechunk_version_from_options(&cmd.options)?;
                ZarrTableUrl::IcechunkRepo(ListingTableUrl::parse(&cmd.location)?, version)
            }
            _ => {
                return Err(DataFusionError::Execution(format!(
                    "Unsupported file type {}",
                    cmd.file_type
                )))
            }
        };

        let (inferred_schema, stats_base) = table_url.infer_schema().await?;
        let schema = if cmd.schema.fields().is_empty() {
            inferred_schema
        } else {
            let provided_schema: Schema = cmd.schema.as_arrow().to_owned();
            for field in provided_schema.fields() {
                let target_type = inferred_schema.field_with_name(field.name())?.data_type();
                if field.data_type() != target_type {
                    return Err(DataFusionError::Execution(format!(
                        "Requested column {}'s type does not match data from store",
                        field.name()
                    )));
                }
            }

            Arc::new(provided_schema)
        };

        let zarr_config = ZarrTableConfig::new(table_url, schema).with_stats_base(stats_base);
        let table_provider = ZarrTable::new(zarr_config);
        Ok(Arc::new(table_provider))
    }
}

// Parse the icechunk version selector out of a `CREATE EXTERNAL TABLE`'s
// `OPTIONS`. At most one selector key may be set; recognized keys are
// `branch`, `tag`, and `snapshot_id`. An empty options map defaults to the
// tip of the `main` branch.
#[cfg(feature = "icechunk")]
fn icechunk_version_from_options(
    options: &std::collections::HashMap<String, String>,
) -> DfResult<IcechunkVersion> {
    if options.is_empty() {
        return Ok(IcechunkVersion::default());
    }
    if options.len() > 1 {
        return Err(DataFusionError::Execution(
            "at most one of the icechunk version options (branch, tag, snapshot_id) may be set"
                .into(),
        ));
    }

    let (key, value) = options.iter().next().expect("options is non-empty");
    // datafusion's sql parser prefixes any OPTIONS key that doesn't already
    // contain a '.' with "format.", so e.g. 'snapshot_id' arrives here as
    // 'format.snapshot_id'. strip that prefix before matching.
    let key = key.strip_prefix("format.").unwrap_or(key);
    match key {
        "branch" => Ok(IcechunkVersion::BranchTip(value.clone())),
        "tag" => Ok(IcechunkVersion::RefTag(value.clone())),
        "snapshot_id" => Ok(IcechunkVersion::SnapshotId(value.clone())),
        other => Err(DataFusionError::Execution(format!(
            "unknown icechunk version option '{other}', expected one of: branch, tag, snapshot_id"
        ))),
    }
}

#[cfg(test)]
mod table_provider_tests {
    use std::collections::HashMap;

    use arrow::array::AsArray;
    use arrow::compute::concat_batches;
    use arrow::datatypes::Float64Type;
    use arrow_schema::DataType;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionContext;
    use futures_util::TryStreamExt;

    use super::*;
    use crate::table::table_provider::ZarrTable;
    use crate::test_utils::{
        extract_col, get_local_zarr_store, validate_names_and_types, validate_primitive_column,
    };
    #[cfg(feature = "icechunk")]
    use crate::test_utils::{get_local_icechunk_repo, get_local_icechunk_repo_multiple_commits};

    async fn read_and_validate(table_provider: ZarrTable, shift: f64) {
        let state = SessionStateBuilder::new().build();
        let session = SessionContext::new();

        let scan = table_provider
            .scan(&state, None, &Vec::new(), None)
            .await
            .unwrap();
        let records: Vec<_> = scan
            .execute(0, session.task_ctx())
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        let target_types = HashMap::from([
            ("lat".to_string(), DataType::Float64),
            ("lon".to_string(), DataType::Float64),
            ("data".to_string(), DataType::Float64),
        ]);
        validate_names_and_types(&target_types, &records[0]);
        assert_eq!(records.len(), 9);

        // the top left chunk, full 3x3
        validate_primitive_column::<Float64Type, f64>(
            "lat",
            &records[0],
            &[35., 35., 35., 36., 36., 36., 37., 37., 37.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            &records[0],
            &[
                -120.0, -119.0, -118.0, -120.0, -119.0, -118.0, -120.0, -119.0, -118.0,
            ],
        );
        let data_targets: Vec<f64> = [0.0, 1.0, 2.0, 8.0, 9.0, 10.0, 16.0, 17.0, 18.0]
            .iter()
            .map(|v| v + shift)
            .collect();
        validate_primitive_column::<Float64Type, f64>("data", &records[0], &data_targets);
    }

    #[tokio::test]
    async fn read_data_test() {
        // a zarr store in a local directory.
        let (wrapper, schema) = get_local_zarr_store(true, 0.0, "lat_lon_data_for_provider").await;
        let path = wrapper.get_store_path();
        let table_url = ZarrTableUrl::ZarrStore(ListingTableUrl::parse(path).unwrap());
        let table = ZarrTable::new(ZarrTableConfig::new(table_url, schema));

        read_and_validate(table, 0.0).await;

        // a local icechunk repo.
        #[cfg(feature = "icechunk")]
        {
            let (wrapper, schema) =
                get_local_icechunk_repo(true, 0.0, "lat_lon_repo_for_provider").await;
            let path = wrapper.get_store_path();
            let table_url = ZarrTableUrl::IcechunkRepo(
                ListingTableUrl::parse(path).unwrap(),
                IcechunkVersion::default(),
            );
            let table = ZarrTable::new(ZarrTableConfig::new(table_url, schema));

            read_and_validate(table, 0.0).await;
        }
    }

    // reads the same icechunk repo at three different versions (a snapshot id, a
    // tag, and the main branch tip), each of which was committed with a different
    // shift applied to the data (0, 1 and 2 respectively).
    #[cfg(feature = "icechunk")]
    #[tokio::test]
    async fn read_data_with_icechunk_commits_test() {
        let (wrapper, _schema, snapshot_id, tag) =
            get_local_icechunk_repo_multiple_commits("lat_lon_repo_multiple_commits").await;
        let path = wrapper.get_store_path();

        // the snapshot id points at the first commit, with a shift of 0.
        let table = ZarrTable::from_path_to_icechunk(path.clone())
            .await
            .unwrap()
            .with_icechunk_snapshot_id(snapshot_id)
            .unwrap();
        read_and_validate(table, 0.0).await;

        // the tag points at the second commit, with a shift of 1.
        let table = ZarrTable::from_path_to_icechunk(path.clone())
            .await
            .unwrap()
            .with_icechunk_reftag(tag)
            .unwrap();
        read_and_validate(table, 1.0).await;

        // the main branch tip is the third commit, with a shift of 2.
        let table = ZarrTable::from_path_to_icechunk(path)
            .await
            .unwrap()
            .with_icechunk_branchtip("main".to_string())
            .unwrap();
        read_and_validate(table, 2.0).await;
    }

    #[tokio::test]
    async fn create_table_provider_test() {
        let (wrapper, _) = get_local_zarr_store(true, 0.0, "lat_lon_data_for_factory").await;
        let mut state = SessionStateBuilder::new().build();
        let table_path = wrapper.get_store_path();

        // here we create the table via a sql command so that we can explicitly
        // create it on some of the columns, to test the case where all the
        // selected columns are coordinates that need to be broadcasted.
        state
            .table_factories_mut()
            .insert("ZARR_STORE".into(), Arc::new(ZarrTableFactory {}));
        let query = format!(
            "CREATE EXTERNAL TABLE zarr_table_partial(lat double, lon double) STORED AS ZARR_STORE LOCATION '{}'",
            table_path,
        );

        let session = SessionContext::new_with_state(state.clone());
        session.sql(&query).await.unwrap();

        // both columns are 1d coordinates. This should get resolved to
        // all combinations of lat with lon (8 lats, 8 lons -> 64 rows).
        let query = "SELECT lat, lon FROM zarr_table_partial";
        let df = session.sql(query).await.unwrap();
        let batches = df.collect().await.unwrap();

        let schema = batches[0].schema();
        let batch = concat_batches(&schema, &batches).unwrap();
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 64);

        // now we want the full table so we can just register the table
        // directly on the session context.
        let session = SessionContext::new_with_state(state.clone());
        session
            .register_table(
                "zarr_table",
                Arc::new(ZarrTable::from_path(table_path.clone()).await.unwrap()),
            )
            .unwrap();

        // a simple select statement with a limit.
        let query = "SELECT lat, lon FROM zarr_table LIMIT 10";
        let df = session.sql(query).await.unwrap();
        let batches = df.collect().await.unwrap();

        let schema = batches[0].schema();
        let batch = concat_batches(&schema, &batches).unwrap();
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 10);

        // a slightly more complex query involving a join.
        let query = "
                    WITH d1 AS (
                        SELECT lat, lon, data
                        FROM zarr_table
                    ),

                    d2 AS (
                        SELECT lat, lon, data*2 as data2
                        FROM zarr_table
                    )

                    SELECT data, data2
                    FROM d1
                    JOIN d2
                        ON d1.lat = d2.lat
                        AND d1.lon = d2.lon
                    ";
        let df = session.sql(query).await.unwrap();
        let batches = df.collect().await.unwrap();

        let schema = batches[0].schema();
        let batch = concat_batches(&schema, &batches).unwrap();

        let data1: Vec<_> = batch
            .column_by_name("data")
            .unwrap()
            .as_primitive::<Float64Type>()
            .values()
            .iter()
            .map(|f| f * 2.0)
            .collect();
        let data2 = batch
            .column_by_name("data2")
            .unwrap()
            .as_primitive::<Float64Type>()
            .values()
            .to_vec();
        assert_eq!(data1, data2);

        // create a table from an icechunk repo.
        #[cfg(feature = "icechunk")]
        {
            let (wrapper, _) = get_local_icechunk_repo(true, 0.0, "lat_lon_repo_for_factory").await;
            let table_path = wrapper.get_store_path();

            let session = SessionContext::new_with_state(state.clone());
            session
                .register_table(
                    "zarr_table_icechunk",
                    Arc::new(ZarrTable::from_path_to_icechunk(table_path).await.unwrap()),
                )
                .unwrap();

            let query = "SELECT lat, lon FROM zarr_table_icechunk LIMIT 10";
            let df = session.sql(query).await.unwrap();
            let batches = df.collect().await.unwrap();

            let schema = batches[0].schema();
            let batch = concat_batches(&schema, &batches).unwrap();
            assert_eq!(batch.num_columns(), 2);
            assert_eq!(batch.num_rows(), 10);
        }
    }

    // creates a table via a CREATE EXTERNAL TABLE statement for each of the three
    // committed versions of the icechunk repo (a snapshot id, a tag, and the main
    // branch tip), exercising the OPTIONS-based version selection. we only check
    // the shape of the result (all 3 columns, all 64 rows).
    #[cfg(feature = "icechunk")]
    #[tokio::test]
    async fn create_table_provider_icechunk_test() {
        let (wrapper, _schema, snapshot_id, tag) =
            get_local_icechunk_repo_multiple_commits("lat_lon_repo_for_factory_commits").await;
        let table_path = wrapper.get_store_path();

        let mut state = SessionStateBuilder::new().build();
        state
            .table_factories_mut()
            .insert("ICECHUNK_REPO".into(), Arc::new(ZarrTableFactory {}));

        // one CREATE EXTERNAL TABLE per commit, selecting the version via OPTIONS.
        let cases = [
            (
                "zarr_table_snapshot",
                format!("OPTIONS ('snapshot_id' '{}')", snapshot_id),
            ),
            ("zarr_table_tag", format!("OPTIONS ('tag' '{}')", tag)),
            ("zarr_table_branch", "OPTIONS ('branch' 'main')".to_string()),
        ];

        for (table_name, options) in cases {
            let query = format!(
                "CREATE EXTERNAL TABLE {} STORED AS ICECHUNK_REPO LOCATION '{}' {}",
                table_name, table_path, options,
            );

            let session = SessionContext::new_with_state(state.clone());
            session.sql(&query).await.unwrap();

            let query = format!("SELECT lat, lon, data FROM {}", table_name);
            let df = session.sql(&query).await.unwrap();
            let batches = df.collect().await.unwrap();

            let schema = batches[0].schema();
            let batch = concat_batches(&schema, &batches).unwrap();
            assert_eq!(batch.num_columns(), 3);
            assert_eq!(batch.num_rows(), 64);
        }
    }

    #[tokio::test]
    async fn partial_coordinates_query() {
        let (wrapper, _) =
            get_local_zarr_store(true, 0.0, "lat_lon_data_partial_coord_query").await;
        let state = SessionStateBuilder::new().build();
        let table_path = wrapper.get_store_path();

        let session = SessionContext::new_with_state(state.clone());
        session
            .register_table(
                "zarr_table",
                Arc::new(ZarrTable::from_path(table_path).await.unwrap()),
            )
            .unwrap();

        // select the 2D data and only one of the 1D coordinates. This should get
        // resolved to the lon being brodacasted to match the 2D data.
        let query = "SELECT data, lon FROM zarr_table";
        let df = session.sql(query).await.unwrap();
        let batches = df.collect().await.unwrap();

        let schema = batches[0].schema();
        let batch = concat_batches(&schema, &batches).unwrap();
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 64);
    }

    #[tokio::test]
    async fn query_with_filter() {
        let (wrapper, _) = get_local_zarr_store(true, 0.0, "lat_lon_data_filter_query").await;
        let state = SessionStateBuilder::new().build();
        let table_path = wrapper.get_store_path();

        let session = SessionContext::new_with_state(state.clone());
        session
            .register_table(
                "zarr_table",
                Arc::new(ZarrTable::from_path(table_path).await.unwrap()),
            )
            .unwrap();

        // select the 2D data and only one of the 1D coordinates. This should get
        // resolved to the lon being brodacasted to match the 2D data.
        let query = "
                    SELECT lat, lon, data
                    FROM zarr_table
                    WHERE lat < 38.1
                    AND lon > -116.9
                    ";
        let df = session.sql(query).await.unwrap();
        let batches = df.collect().await.unwrap();

        // this tests for the actual WHERE clause, which is a combination
        // of the filter pushdown and some filtering provided by datafusion,
        // out-of-the-box, so the condition in the test matches the WHERE
        // clause exactly.
        for batch in batches {
            let lat_values = extract_col::<Float64Type>("lat", &batch);
            let lon_values = extract_col::<Float64Type>("lon", &batch);
            assert!(lat_values
                .iter()
                .zip(lon_values.iter())
                .all(|(lat, lon)| *lat < 38.1 && *lon > -116.9));
        }
    }

    #[tokio::test]
    async fn table_factory_error_test() {
        let (wrapper, _) = get_local_zarr_store(true, 0.0, "lat_lon_data_for_factory_error").await;
        let mut state = SessionStateBuilder::new().build();
        let table_path = wrapper.get_store_path();
        state
            .table_factories_mut()
            .insert("ZARR_STORE".into(), Arc::new(ZarrTableFactory {}));

        // create a table with 2 explicitly selected columns, but the names
        // are wrong so it should error out.
        let query = format!(
            "CREATE EXTERNAL TABLE zarr_table(latitude double, longitude double) STORED AS ZARR_STORE LOCATION '{}'",
            table_path,
        );

        let session = SessionContext::new_with_state(state.clone());
        let res = session.sql(&query).await;
        match res {
            Ok(_) => panic!(),
            Err(e) => {
                assert_eq!(
                    e.to_string(),
                    "Arrow error: Schema error: Unable to get field named \"latitude\". Valid fields: [\"data\", \"lat\", \"lon\"]"
                );
            }
        }

        // create a table with 2 explicitly selected columns, but the type for the
        // columns are wrong so it should error out.
        let query = format!(
            "CREATE EXTERNAL TABLE zarr_table(lat int, lon int) STORED AS ZARR_STORE LOCATION '{}'",
            table_path,
        );

        let session = SessionContext::new_with_state(state.clone());
        let res = session.sql(&query).await;
        match res {
            Ok(_) => panic!(),
            Err(e) => {
                assert_eq!(
                    e.to_string(),
                    "Execution error: Requested column lat's type does not match data from store"
                );
            }
        }
    }

    #[tokio::test]
    async fn stats_and_metrics_test() {
        use datafusion::common::stats::Precision;

        let (wrapper, _schema) =
            get_local_zarr_store(true, 0.0, "lat_lon_data_provider_stats_metrics").await;
        let path = wrapper.get_store_path();
        // build via `from_path` so the config carries the inferred stats base.
        let table = ZarrTable::from_path(path).await.unwrap();

        let state = SessionStateBuilder::new().build();
        let session = SessionContext::new();
        let scan = table.scan(&state, None, &Vec::new(), None).await.unwrap();

        let stats = scan.partition_statistics(None).unwrap();
        assert_eq!(stats.num_rows, Precision::Exact(64));

        let records: Vec<_> = scan
            .execute(0, session.task_ctx())
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert_eq!(records.len(), 9);

        let snapshot = scan.metrics().unwrap();
        for name in [
            "io_time",
            "decode_time",
            "total_time",
            "chunks_looked_at",
            "chunks_read",
            "rows_produced",
        ] {
            assert!(
                snapshot.sum_by_name(name).is_some(),
                "metric {name} missing from the metrics set"
            );
        }

        let value = |name: &str| snapshot.sum_by_name(name).unwrap().as_usize();
        assert_eq!(value("chunks_looked_at"), 9);
        assert_eq!(value("chunks_read"), 9);
        assert_eq!(value("rows_produced"), 64);
    }
}
