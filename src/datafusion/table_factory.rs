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

use arrow::datatypes::{DataType, Schema};
use arrow_schema::Field;
use async_trait::async_trait;
use datafusion::{
    common::arrow_datafusion_err,
    datasource::{listing::ListingTableUrl, provider::TableProviderFactory, TableProvider},
    error::DataFusionError,
    execution::context::SessionState,
    logical_expr::CreateExternalTable,
};

use super::table_provider::{ListingZarrTableConfig, ListingZarrTableOptions, ZarrTableProvider};

pub struct ZarrListingTableFactory {}

#[async_trait]
impl TableProviderFactory for ZarrListingTableFactory {
    async fn create(
        &self,
        state: &SessionState,
        cmd: &CreateExternalTable,
    ) -> datafusion::common::Result<Arc<dyn TableProvider>> {
        if cmd.file_type != "ZARR" {
            return Err(datafusion::error::DataFusionError::Execution(
                "Invalid file type".to_string(),
            ));
        }

        // mostly copied over from datafusion.
        let (provided_schema, table_partition_cols) = if cmd.schema.fields().is_empty() {
            (
                None,
                cmd.table_partition_cols
                    .iter()
                    .map(|x| {
                        (
                            x.clone(),
                            DataType::Dictionary(
                                Box::new(DataType::UInt16),
                                Box::new(DataType::Utf8),
                            ),
                        )
                    })
                    .collect::<Vec<_>>(),
            )
        } else {
            // this bit here is to ensure that the fields in the schema are alphabetically
            // ordered. because a zarr store doesn't provide any ordering, we need some
            // convention, and the schema needs to follow that convention here.
            let mut schema: Schema = cmd.schema.as_ref().into();
            let mut fields: Vec<Field> = Vec::new();
            for f in schema.fields() {
                let test = Field::new(f.name(), f.data_type().clone(), false);
                fields.push(test);
            }
            fields.sort_by(|f1, f2| f1.name().cmp(f2.name()));
            schema = Schema::new(fields);

            let table_partition_cols = cmd
                .table_partition_cols
                .iter()
                .map(|col| {
                    schema
                        .field_with_name(col)
                        .map_err(|e| arrow_datafusion_err!(e))
                })
                .collect::<datafusion_common::Result<Vec<_>>>()?
                .into_iter()
                .map(|f| (f.name().to_owned(), f.data_type().to_owned()))
                .collect();
            // exclude partition columns to support creating partitioned external table
            // with a specified column definition like `create external table a(c0 int, c1 int)
            // stored as csv partitioned by (c1)...`
            let mut project_idx = Vec::new();
            for i in 0..schema.fields().len() {
                if !cmd.table_partition_cols.contains(schema.field(i).name()) {
                    project_idx.push(i);
                }
            }
            let schema = schema.project(&project_idx)?;
            (Some(schema), table_partition_cols)
        };

        let table_path = ListingTableUrl::parse(&cmd.location)?;
        let options = ListingZarrTableOptions::new().with_partition_cols(table_partition_cols);
        let schema = match provided_schema {
            None => options
                .infer_schema(state, &table_path)
                .await
                .map_err(|e| DataFusionError::Execution(format!("infer error: {:?}", e)))?,
            Some(s) => s,
        };

        let config = ListingZarrTableConfig::new(table_path, schema, Some(options));
        let table_provider = ZarrTableProvider::try_new(config)?;

        Ok(Arc::new(table_provider))
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::get_test_v2_data_path;
    use arrow::record_batch::RecordBatch;
    use arrow_array::{cast::AsArray, StringArray};
    use arrow_array::types::*;
    use arrow_buffer::ScalarBuffer;

    use datafusion::execution::{
        config::SessionConfig,
        context::{SessionContext, SessionState},
        runtime_env::RuntimeEnv,
    };
    use std::sync::Arc;

    fn extract_col<T>(col_name: &str, rec_batch: &RecordBatch) -> ScalarBuffer<T::Native>
    where
        T: ArrowPrimitiveType,
    {
        rec_batch
            .column_by_name(col_name)
            .unwrap()
            .as_primitive::<T>()
            .values()
            .clone()
    }

    fn extract_str_col(col_name: &str, rec_batch: &RecordBatch) -> StringArray
    {
        rec_batch
            .column_by_name(col_name)
            .unwrap()
            .as_string()
            .to_owned()
    }

    #[tokio::test]
    async fn test_create() -> Result<(), Box<dyn std::error::Error>> {
        let mut state = SessionState::new_with_config_rt(
            SessionConfig::default(),
            Arc::new(RuntimeEnv::default()),
        );

        state
            .table_factories_mut()
            .insert("ZARR".into(), Arc::new(super::ZarrListingTableFactory {}));

        let test_data = get_test_v2_data_path("lat_lon_example.zarr".to_string());

        let sql = format!(
            "CREATE EXTERNAL TABLE zarr_table STORED AS ZARR LOCATION '{}'",
            test_data.display(),
        );

        let session = SessionContext::new_with_state(state);
        session.sql(&sql).await?;

        let sql = "SELECT lat, lon FROM zarr_table LIMIT 10";
        let df = session.sql(sql).await?;

        let batches = df.collect().await?;
        assert_eq!(batches.len(), 1);

        let batch = &batches[0];

        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 10);

        Ok(())
    }

    #[tokio::test]
    async fn test_predicates() -> Result<(), Box<dyn std::error::Error>> {
        let mut state = SessionState::new_with_config_rt(
            SessionConfig::default(),
            Arc::new(RuntimeEnv::default()),
        );

        state
            .table_factories_mut()
            .insert("ZARR".into(), Arc::new(super::ZarrListingTableFactory {}));

        let test_data = get_test_v2_data_path("lat_lon_example.zarr".to_string());

        let sql = format!(
            "CREATE EXTERNAL TABLE zarr_table STORED AS ZARR LOCATION '{}'",
            test_data.display(),
        );

        let session = SessionContext::new_with_state(state);
        session.sql(&sql).await?;

        // apply one predicate on one column.
        let sql = "SELECT lat, lon FROM zarr_table WHERE lat > 38.21";
        let df = session.sql(sql).await?;

        let batches = df.collect().await?;
        for batch in batches {
            let values = extract_col::<Float64Type>("lat", &batch);
            assert!(values.iter().all(|v| *v > 38.21));
        }

        // apply 2 predicates, each on one column.
        let sql = "SELECT lat, lon FROM zarr_table WHERE lat > 38.21 AND lon > -109.59";
        let df = session.sql(sql).await?;

        let batches = df.collect().await?;
        for batch in batches {
            let lat_values = extract_col::<Float64Type>("lat", &batch);
            let lon_values = extract_col::<Float64Type>("lon", &batch);
            assert!(lat_values
                .iter()
                .zip(lon_values.iter())
                .all(|(lat, lon)| *lat > 38.21 && *lon > -109.59));
        }

        // same as above, but flip the column order in the predicates.
        let sql = "SELECT lat, lon FROM zarr_table WHERE lon > -109.59 AND lat > 38.21";
        let df = session.sql(sql).await?;

        let batches = df.collect().await?;
        for batch in batches {
            let lat_values = extract_col::<Float64Type>("lat", &batch);
            let lon_values = extract_col::<Float64Type>("lon", &batch);
            assert!(lat_values
                .iter()
                .zip(lon_values.iter())
                .all(|(lat, lon)| *lat > 38.21 && *lon > -109.59));
        }

        // apply one predicate that includes 2 columns
        let sql = "SELECT lat, lon FROM zarr_table WHERE lat + lon > -71.39";
        let df = session.sql(sql).await?;

        let batches = df.collect().await?;
        for batch in batches {
            let lat_values = extract_col::<Float64Type>("lat", &batch);
            let lon_values = extract_col::<Float64Type>("lon", &batch);
            assert!(lat_values
                .iter()
                .zip(lon_values.iter())
                .all(|(lat, lon)| *lat + *lon > -71.39));
        }

        // same as above, but flip the column order in the predicates.
        let sql = "SELECT lat, lon FROM zarr_table WHERE lon + lat > -71.39";
        let df = session.sql(sql).await?;

        let batches = df.collect().await?;
        for batch in batches {
            let lat_values = extract_col::<Float64Type>("lat", &batch);
            let lon_values = extract_col::<Float64Type>("lon", &batch);
            assert!(lat_values
                .iter()
                .zip(lon_values.iter())
                .all(|(lat, lon)| *lat + *lon > -71.39));
        }

        // apply 3 predicates, 2 on one column and one on 2 columns.
        let sql = "SELECT lat, lon FROM zarr_table WHERE lat > 38.21 AND lon > -109.59 AND lat + lon > -71.09";
        let df = session.sql(sql).await?;

        let batches = df.collect().await?;
        for batch in batches {
            let lat_values = extract_col::<Float64Type>("lat", &batch);
            let lon_values = extract_col::<Float64Type>("lon", &batch);
            assert!(lat_values
                .iter()
                .zip(lon_values.iter())
                .all(|(lat, lon)| *lat > 38.21 && *lon > -109.59 && *lat + *lon > -71.09));
        }

        // check a query that doesn't include the column needed in the predicate. the first query
        // below is used to produce the reference values, and the second one is the one we're testing
        // for, since it has a predicate on lon, but doesn't select lon.
        let sql = "SELECT lat, lon FROM zarr_table WHERE lon > -109.59";
        let df = session.sql(sql).await?;
        let lat_lon_batches = df.collect().await?;

        let sql = "SELECT lat FROM zarr_table WHERE lon > -109.59";
        let df = session.sql(sql).await?;
        let lat_batches = df.collect().await?;

        for (lat_batch, lat_lon_batch) in lat_batches.iter().zip(lat_lon_batches.iter()) {
            let lat_values_1 = extract_col::<Float64Type>("lat", lat_batch);
            let lat_values_2 = extract_col::<Float64Type>("lat", lat_lon_batch);
            assert_eq!(lat_values_1, lat_values_2);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_partitions() -> Result<(), Box<dyn std::error::Error>> {
        let mut state = SessionState::new_with_config_rt(
            SessionConfig::default(),
            Arc::new(RuntimeEnv::default()),
        );

        state
            .table_factories_mut()
            .insert("ZARR".into(), Arc::new(super::ZarrListingTableFactory {}));

        let test_data = get_test_v2_data_path("lat_lon_w_groups_example.zarr".to_string());

        let sql = format!(
            "CREATE EXTERNAL TABLE zarr_table (
               lat double,
               lon double,
               float_data double,
               var int,
               other_var string
            )
            STORED AS ZARR LOCATION '{}'
            PARTITIONED BY (var, other_var)",
            test_data.display(),
        );

        let session = SessionContext::new_with_state(state);
        session.sql(&sql).await?;

        // select a particular partition for each partitioned variable
        let sql = "SELECT lat, lon, var, other_var FROM zarr_table
                   WHERE var=1
                   AND other_var='b'";
        let df = session.sql(sql).await?;

        let batches = df.collect().await?;
        for batch in batches {
            let lat_values = extract_col::<Float64Type>("lat", &batch);
            let lon_values = extract_col::<Float64Type>("lon", &batch);
            let var_values = extract_col::<Int32Type>("var", &batch);
            let other_var_values = extract_str_col("other_var", &batch);
            assert!(lat_values
                .iter()
                .zip(lon_values.iter())
                .all(|(lat, lon)| *lat >= 38.0 && *lat <= 39.0 && *lon >= -108.9 && *lon <= -107.9));
            assert!(var_values.iter().all(|var| var == &1));
            assert!(other_var_values.iter().all(|other_var| other_var.unwrap() == "b"));
        }

        // select a different partition for each partitioned variable
        let sql = "SELECT lat, lon, var, other_var FROM zarr_table
                   WHERE var=2
                   AND other_var='a'";
        let df = session.sql(sql).await?;

        let batches = df.collect().await?;
        for batch in batches {
            let lat_values = extract_col::<Float64Type>("lat", &batch);
            let lon_values = extract_col::<Float64Type>("lon", &batch);
            let var_values = extract_col::<Int32Type>("var", &batch);
            let other_var_values = extract_str_col("other_var", &batch);
            assert!(lat_values
                .iter()
                .zip(lon_values.iter())
                .all(|(lat, lon)| *lat >= 39.0 && *lat <= 40.0 && *lon >= -110.0 && *lon <= -108.9));
            assert!(var_values.iter().all(|var| var == &2));
            assert!(other_var_values.iter().all(|other_var| other_var.unwrap() == "a"));
        }

        // select the same partition but without selection the partitioned variables
        let sql = "SELECT lat, lon FROM zarr_table
                   WHERE var=2
                   AND other_var='a'";
        let df = session.sql(sql).await?;

        let batches = df.collect().await?;
        for batch in batches {
            let lat_values = extract_col::<Float64Type>("lat", &batch);
            let lon_values = extract_col::<Float64Type>("lon", &batch);
            assert!(lat_values
                .iter()
                .zip(lon_values.iter())
                .all(|(lat, lon)| *lat >= 39.0 && *lat <= 40.0 && *lon >= -110.0 && *lon <= -108.9));
        }

        // select a partition for only one of the partitioned variables
        let sql = "SELECT lat, lon, var, other_var FROM zarr_table
                   WHERE var=1";
        let df = session.sql(sql).await?;

        let batches = df.collect().await?;
        for batch in batches {
            let lat_values = extract_col::<Float64Type>("lat", &batch);
            let var_values = extract_col::<Int32Type>("var", &batch);
            let other_var_values = extract_str_col("other_var", &batch);
            assert!(lat_values
                .iter()
                .all(|lat| *lat >= 38.0 && *lat <= 39.0));
            assert!(var_values.iter().all(|var| var == &1));
            assert!(other_var_values.iter().all(
                |other_var| other_var.unwrap() == "a" || other_var.unwrap() == "b"
            ));
        }

        Ok(())
    }
}
