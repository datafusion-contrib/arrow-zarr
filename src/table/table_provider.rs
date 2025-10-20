use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::{Schema, SchemaRef};
use datafusion::catalog::{Session, TableProviderFactory};
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::logical_expr::{CreateExternalTable, Expr};
use datafusion::physical_plan::ExecutionPlan;

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

    pub async fn from_path(path: String) -> Self {
        let table_url = ListingTableUrl::parse(path).unwrap();
        // TODO(alxmrs): Figure out how to optionally support icechunk
        let zarr_url = ZarrTableUrl::ZarrStore(table_url);
        let schema = zarr_url.infer_schema().await.unwrap();
        let table_config = ZarrTableConfig::new(zarr_url, schema);
        Self { table_config }
    }
}

#[async_trait]
impl TableProvider for ZarrTable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.table_config.get_schema_ref()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        let mut config = self.table_config.clone();
        if let Some(proj) = projection {
            config = config.with_projection(proj.to_vec());
        }
        let scanner = ZarrScan::new(config, None);

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
            "ICECHUNK_REPO" => ZarrTableUrl::IcechunkRepo(ListingTableUrl::parse(&cmd.location)?),
            _ => {
                return Err(DataFusionError::Execution(format!(
                    "Unsupported file type {}",
                    cmd.file_type
                )))
            }
        };

        let inferred_schema = table_url.infer_schema().await?;
        let schema = if cmd.schema.fields().is_empty() {
            inferred_schema
        } else {
            let provided_schema: Schema = cmd.schema.as_ref().into();
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

        let zarr_config = ZarrTableConfig::new(table_url, schema);
        let table_provider = ZarrTable::new(zarr_config);
        Ok(Arc::new(table_provider))
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
    #[cfg(feature = "icechunk")]
    use crate::test_utils::get_local_icechunk_repo;
    use crate::test_utils::{
        get_local_zarr_store, validate_names_and_types, validate_primitive_column,
    };

    async fn read_and_validate(table_url: ZarrTableUrl, schema: SchemaRef) {
        let config = ZarrTableConfig::new(table_url, schema);

        let table_provider = ZarrTable::new(config);
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
        validate_primitive_column::<Float64Type, f64>(
            "data",
            &records[0],
            &[0.0, 1.0, 2.0, 8.0, 9.0, 10.0, 16.0, 17.0, 18.0],
        );
    }

    #[tokio::test]
    async fn read_data_test() {
        // a zarr store in a local directory.
        let (wrapper, schema) = get_local_zarr_store(true, 0.0, "lat_lon_data_for_provider").await;
        let path = wrapper.get_store_path();
        let table_url = ZarrTableUrl::ZarrStore(ListingTableUrl::parse(path).unwrap());

        read_and_validate(table_url, schema).await;

        // a local icechunk repo.
        #[cfg(feature = "icechunk")]
        {
            let (wrapper, schema) =
                get_local_icechunk_repo(true, 0.0, "lat_lon_repo_for_provider").await;
            let path = wrapper.get_store_path();
            let table_url = ZarrTableUrl::IcechunkRepo(ListingTableUrl::parse(path).unwrap());

            read_and_validate(table_url, schema).await;
        }
    }

    #[tokio::test]
    async fn create_table_provider_test() {
        let (wrapper, _) = get_local_zarr_store(true, 0.0, "lat_lon_data_for_factory").await;
        let mut state = SessionStateBuilder::new().build();
        let table_path = wrapper.get_store_path();
        state
            .table_factories_mut()
            .insert("ZARR_STORE".into(), Arc::new(ZarrTableFactory {}));

        // create a table with 2 explicitly selected columns
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

        // create a table, with 3 columns, lat, lon and data.
        let query = format!(
            "CREATE EXTERNAL TABLE zarr_table STORED AS ZARR_STORE LOCATION '{}'",
            table_path,
        );

        let session = SessionContext::new_with_state(state.clone());
        session.sql(&query).await.unwrap();

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
            state
                .table_factories_mut()
                .insert("ICECHUNK_REPO".into(), Arc::new(ZarrTableFactory {}));

            let query = format!(
                "CREATE EXTERNAL TABLE zarr_table_icechunk STORED AS ICECHUNK_REPO LOCATION '{}'",
                table_path,
            );

            let session = SessionContext::new_with_state(state.clone());
            session.sql(&query).await.unwrap();

            let query = "SELECT lat, lon FROM zarr_table LIMIT 10";
            let df = session.sql(query).await.unwrap();
            let batches = df.collect().await.unwrap();

            let schema = batches[0].schema();
            let batch = concat_batches(&schema, &batches).unwrap();
            assert_eq!(batch.num_columns(), 2);
            assert_eq!(batch.num_rows(), 10);
        }
    }

    #[tokio::test]
    async fn partial_coordinates_query() {
        let (wrapper, _) =
            get_local_zarr_store(true, 0.0, "lat_lon_data_partial_coord_query").await;
        let mut state = SessionStateBuilder::new().build();
        let table_path = wrapper.get_store_path();
        state
            .table_factories_mut()
            .insert("ZARR_STORE".into(), Arc::new(ZarrTableFactory {}));

        let query = format!(
            "CREATE EXTERNAL TABLE zarr_table STORED AS ZARR_STORE LOCATION '{}'",
            table_path,
        );

        let session = SessionContext::new_with_state(state.clone());
        session.sql(&query).await.unwrap();

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
}
