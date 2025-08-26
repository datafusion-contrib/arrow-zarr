use super::config::ZarrConfig;
use super::scanner::ZarrScan;
use arrow_schema::{DataType, Field, Fields};
use async_trait::async_trait;
use datafusion::arrow::datatypes::{Schema, SchemaRef};
use datafusion::catalog::{Session, TableProviderFactory};
use datafusion::common::not_impl_err;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::logical_expr::CreateExternalTable;
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::ExecutionPlan;
use object_store::local::LocalFileSystem;
use std::fmt::Debug;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use zarrs::array::Array;
use zarrs_metadata::v3::array::data_type::DataTypeMetadataV3;
use zarrs_metadata::ArrayMetadata;
use zarrs_object_store::AsyncObjectStore;
use zarrs_storage::{AsyncReadableListableStorageTraits, StorePrefix};

/// The table provider for zarr stores.
pub struct ZarrTable {
    table_schema: Schema,
    zarr_storage: Arc<dyn AsyncReadableListableStorageTraits + Unpin + Send>,
}

impl Debug for ZarrTable {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl ZarrTable {
    pub fn new(
        table_schema: Schema,
        zarr_storage: Arc<dyn AsyncReadableListableStorageTraits + Unpin + Send>,
    ) -> Self {
        Self {
            table_schema,
            zarr_storage,
        }
    }
}

#[async_trait]
impl TableProvider for ZarrTable {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::new(self.table_schema.clone())
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
        let mut config = ZarrConfig::new(self.zarr_storage.clone(), self.table_schema.clone());
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

enum ZarrStoreType {
    LocalFolder,
    IcechunkRepo,
}

impl FromStr for ZarrStoreType {
    type Err = DataFusionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let e = match s {
            "ZARR_LOCAL_FOLDER" => Self::LocalFolder,
            "ICECHUNK_REPO" => Self::IcechunkRepo,
            _ => {
                return Err(DataFusionError::Execution(format!(
                    "Invalid file type {}",
                    s
                )))
            }
        };

        Ok(e)
    }
}

#[async_trait]
impl TableProviderFactory for ZarrTableFactory {
    async fn create(
        &self,
        _state: &dyn Session,
        cmd: &CreateExternalTable,
    ) -> DfResult<Arc<dyn TableProvider>> {
        let store_type = ZarrStoreType::from_str(&cmd.file_type)?;
        let store = match store_type {
            ZarrStoreType::LocalFolder => {
                let p = PathBuf::from(cmd.location.clone());
                let f = LocalFileSystem::new_with_prefix(p.clone())?;
                Arc::new(AsyncObjectStore::new(f))
            }
            ZarrStoreType::IcechunkRepo => not_impl_err!("Icechunk repos are not yet supported")?,
        };

        let inferred_schema = infer_schema(store.clone()).await?;
        let schema = if cmd.schema.fields().is_empty() {
            inferred_schema
        } else {
            let provided_schema: Schema = cmd.schema.as_ref().into();
            for field in provided_schema.fields() {
                let target_type = inferred_schema
                    .fields()
                    .find(field.name())
                    .ok_or(DataFusionError::Execution(format!(
                        "Requested column {} is missing from store",
                        field.name()
                    )))?
                    .1
                    .data_type();
                if field.data_type() != target_type {
                    return Err(DataFusionError::Execution(format!(
                        "Requested column {}'s type does not match data from store",
                        field.name()
                    )));
                }
            }

            provided_schema
        };

        let table_provider = ZarrTable::new(schema, store);
        Ok(Arc::new(table_provider))
    }
}

// helpers to infer the schema from a zarr store, which involves reading
// directory names and reading some metadata, so it's a bit trickier than
// e.g. get a schema from a parquet file.
fn get_schema_type(value: &DataTypeMetadataV3) -> DfResult<DataType> {
    match value {
        DataTypeMetadataV3::Bool => Ok(DataType::Boolean),
        DataTypeMetadataV3::UInt8 => Ok(DataType::UInt8),
        DataTypeMetadataV3::UInt16 => Ok(DataType::UInt16),
        DataTypeMetadataV3::UInt32 => Ok(DataType::UInt32),
        DataTypeMetadataV3::UInt64 => Ok(DataType::UInt64),
        DataTypeMetadataV3::Int8 => Ok(DataType::Int8),
        DataTypeMetadataV3::Int16 => Ok(DataType::Int16),
        DataTypeMetadataV3::Int32 => Ok(DataType::Int32),
        DataTypeMetadataV3::Int64 => Ok(DataType::Int64),
        DataTypeMetadataV3::Float32 => Ok(DataType::Float32),
        DataTypeMetadataV3::Float64 => Ok(DataType::Float64),
        DataTypeMetadataV3::String => Ok(DataType::Utf8),
        _ => Err(DataFusionError::Execution(format!(
            "Unsupported type {value} from zarr metadata"
        ))),
    }
}

async fn infer_schema(store: Arc<dyn AsyncReadableListableStorageTraits>) -> DfResult<Schema> {
    let dirs = store
        .list_dir(&StorePrefix::new("").map_err(|e| DataFusionError::External(Box::new(e)))?)
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    let prefixes = dirs.prefixes();
    let mut fields = Vec::with_capacity(prefixes.len());

    for prefix in prefixes {
        let field_name = prefix
            .as_str()
            .strip_suffix("/")
            .ok_or(DataFusionError::Execution(
                "Invalid directory name in zarr store".into(),
            ))?;

        let arr = Array::async_open(store.clone(), &("/".to_owned() + field_name))
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let meta = match arr.metadata() {
            ArrayMetadata::V3(meta) => Ok(meta),
            _ => Err(DataFusionError::Execution(
                "Only Zarr v3 metadata is supported".into(),
            )),
        }?;

        fields.push(Field::new(
            field_name,
            get_schema_type(&meta.data_type)?,
            true,
        ));
    }

    Ok(Schema::new(Fields::from(fields)))
}

#[cfg(test)]
mod table_provider_tests {
    use arrow::array::AsArray;
    use std::collections::HashMap;

    use super::*;
    use crate::table::table_provider::ZarrTable;
    use crate::test_utils::{
        get_lat_lon_data_store, validate_names_and_types, validate_primitive_column,
    };
    use arrow::compute::concat_batches;
    use arrow::datatypes::Float64Type;
    use arrow_schema::DataType;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionContext;
    use futures_util::TryStreamExt;

    #[tokio::test]
    async fn read_data_test() {
        let (wrapper, schema) =
            get_lat_lon_data_store(true, 0.0, "lat_lon_data_for_provider").await;
        let store = wrapper.get_store();

        let table_provider = ZarrTable::new(schema, store);
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
    async fn create_table_provider_test() {
        let (wrapper, _) = get_lat_lon_data_store(true, 0.0, "lat_lon_data_for_factory").await;
        let mut state = SessionStateBuilder::new().build();
        let table_path = wrapper.get_store_path();
        state
            .table_factories_mut()
            .insert("ZARR_LOCAL_FOLDER".into(), Arc::new(ZarrTableFactory {}));

        // create a table with 2 explicitly selected columns
        let query = format!(
            "CREATE EXTERNAL TABLE zarr_table_partial(lat double, lon double) STORED AS ZARR_LOCAL_FOLDER LOCATION '{}'",
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
            "CREATE EXTERNAL TABLE zarr_table STORED AS ZARR_LOCAL_FOLDER LOCATION '{}'",
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
    }

    #[tokio::test]
    async fn table_factory_error_test() {
        let (wrapper, _) =
            get_lat_lon_data_store(true, 0.0, "lat_lon_data_for_factory_error").await;
        let mut state = SessionStateBuilder::new().build();
        let table_path = wrapper.get_store_path();
        state
            .table_factories_mut()
            .insert("ZARR_LOCAL_FOLDER".into(), Arc::new(ZarrTableFactory {}));

        // create a table with 2 explicitly selected columns, but the names
        // are wrong so it should error out.
        let query = format!(
            "CREATE EXTERNAL TABLE zarr_table(latitude double, longitude double) STORED AS ZARR_LOCAL_FOLDER LOCATION '{}'",
            table_path,
        );

        let session = SessionContext::new_with_state(state.clone());
        let res = session.sql(&query).await;
        match res {
            Ok(_) => panic!(),
            Err(e) => {
                assert_eq!(
                    e.to_string(),
                    "Execution error: Requested column latitude is missing from store"
                );
            }
        }

        // create a table with 2 explicitly selected columns, but the type for the
        // columns are wrong so it should error out.
        let query = format!(
            "CREATE EXTERNAL TABLE zarr_table(lat int, lon int) STORED AS ZARR_LOCAL_FOLDER LOCATION '{}'",
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
