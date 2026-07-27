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

use std::collections::HashMap;
#[cfg(all(feature = "icechunk", feature = "s3"))]
use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef, TimeUnit};
use datafusion::common::stats::Precision;
use datafusion::common::Statistics;
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::error::{DataFusionError, Result as DfResult};
#[cfg(feature = "icechunk")]
use icechunk::{ObjectStorage, Repository};
#[cfg(feature = "s3")]
use object_store::aws::AmazonS3Builder;
#[cfg(feature = "gcs")]
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::local::LocalFileSystem;
use zarrs::array::data_type::DataType as zarr_dtype;
use zarrs::array::Array;
use zarrs::metadata_ext::data_type::NumpyTimeUnit;
use zarrs::registry::ExtensionAliases;
#[cfg(feature = "icechunk")]
use zarrs_icechunk::AsyncIcechunkStore;
use zarrs_metadata::v3::MetadataV3;
use zarrs_metadata::ArrayMetadata;
use zarrs_object_store::AsyncObjectStore;
use zarrs_storage::{AsyncReadableListableStorageTraits, StorePrefix};

// Cached, projection-independent ingredients for computing scan statistics.
//
// The two maps mirror the reader's two regimes (see `ZarrCoordinates::new`
// / `resolve_vector` in `zarr_data_stream.rs`): dimension names only trigger
// when every relevant array is named; with partial or absent names the reader
// does no broadcasting and simply requires the shapes to match.
#[derive(Clone, Debug, Default)]
pub(crate) struct ZarrStatsBase {
    column_dim_names: HashMap<String, Vec<String>>,
    column_dims: HashMap<String, Vec<u64>>,
}

impl ZarrStatsBase {
    // Record one array's contribution. The raw shape is always stored,
    // dimension names are stored only when the array has them. Dimension
    // lengths aren't stored separately, they are recovered from `column_dims`
    // (a dim's length is the shape entry at its position).
    fn add_array(&mut self, column: &str, shape: &[u64], dim_names: Option<Vec<String>>) {
        self.column_dims.insert(column.to_string(), shape.to_vec());
        if let Some(names) = dim_names {
            self.column_dim_names.insert(column.to_string(), names);
        }
    }

    // Build DataFusion statistics for a projected schema. `num_rows` is computed
    // to match the reader's two regimes; everything else (byte size, per-column stats)
    // is left `Absent`.
    pub(crate) fn to_datafusion_statistics(
        &self,
        projected_schema: &SchemaRef,
    ) -> DfResult<Statistics> {
        let columns: Vec<&str> = projected_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();

        let stats = Statistics::new_unknown(projected_schema);
        if columns.is_empty() {
            return Ok(stats);
        }

        let num_rows = if columns
            .iter()
            .all(|c| self.column_dim_names.contains_key(*c))
        {
            // first case, union the named dimensions and multiply their lengths. a dim's length
            // is read from the shape of any column that spans it (the shape entry at the dim's
            // position); same-dim entries agree, so overwriting is harmless.
            let mut dim_lengths: HashMap<&str, u64> = HashMap::new();
            for c in &columns {
                for (name, len) in self.column_dim_names[*c]
                    .iter()
                    .zip(self.column_dims[*c].iter())
                {
                    dim_lengths.insert(name.as_str(), *len);
                }
            }
            dim_lengths.values().product::<u64>()
        } else {
            // second case, no broadcasting, so every projected column must share
            // one shape.
            let mut shape: Option<&Vec<u64>> = None;
            for c in &columns {
                let Some(col_shape) = self.column_dims.get(*c) else {
                    // every array is recorded in `column_dims` at schema inference, and the
                    // projection is always a subset of the inferred schema, so a missing entry
                    // means the stats base and schema are out of sync.
                    return Err(DataFusionError::Internal(format!(
                        "zarr stats base is missing shape info for projected column '{c}'"
                    )));
                };
                match shape {
                    Some(s) if s != col_shape => {
                        return Err(DataFusionError::Execution(
                            "Cannot compute zarr statistics: selected arrays without consistent \
                             dimension names have mismatched shapes"
                                .into(),
                        ));
                    }
                    _ => shape = Some(col_shape),
                }
            }
            shape
                .map(|s| s.iter().product::<u64>())
                .expect("non-empty projection always resolves a shape")
        };

        Ok(stats.with_num_rows(Precision::Exact(num_rows as usize)))
    }
}

/// A zarr table configuration.
#[derive(Clone, Debug)]
pub struct ZarrTableConfig {
    schema_ref: SchemaRef,
    table_url: ZarrTableUrl,
    projection: Option<Vec<usize>>,
    stats_base: Option<ZarrStatsBase>,
}

impl ZarrTableConfig {
    pub(crate) fn new(table_url: ZarrTableUrl, schema_ref: SchemaRef) -> Self {
        Self {
            schema_ref,
            table_url,
            projection: None,
            stats_base: None,
        }
    }

    pub(crate) fn with_stats_base(mut self, stats_base: ZarrStatsBase) -> Self {
        self.stats_base = Some(stats_base);
        self
    }

    // Statistics for the (projected) scan output. Falls back to all-unknown when
    // no stats base was captured (e.g. configs built directly in tests).
    pub(crate) fn statistics(&self) -> DfResult<Statistics> {
        let projected_schema = self.get_projected_schema_ref();
        match &self.stats_base {
            Some(base) => base.to_datafusion_statistics(&projected_schema),
            None => Ok(Statistics::new_unknown(&projected_schema)),
        }
    }

    pub(crate) async fn get_store_pointer_and_prefix(
        &self,
    ) -> DfResult<(
        Arc<dyn AsyncReadableListableStorageTraits + Unpin + Send>,
        Option<String>,
    )> {
        self.table_url.get_store_pointer_and_prefix().await
    }

    pub(crate) fn with_projection(mut self, projection: Vec<usize>) -> Self {
        self.projection = Some(projection);
        self
    }

    #[cfg(feature = "icechunk")]
    pub(crate) fn with_icechunk_version(mut self, version: IcechunkVersion) -> DfResult<Self> {
        self.table_url = self.table_url.with_icechunk_version(version)?;
        Ok(self)
    }

    pub(crate) fn get_projection(&self) -> Option<Vec<usize>> {
        self.projection.clone()
    }

    pub(crate) fn get_schema_ref(&self) -> SchemaRef {
        self.schema_ref.clone()
    }

    pub(crate) fn get_projected_schema_ref(&self) -> SchemaRef {
        if let Some(projection) = &self.projection {
            let projected_fields: Fields = projection
                .iter()
                .map(|&i| self.schema_ref.field(i).clone())
                .collect();
            Arc::new(Schema::new(projected_fields))
        } else {
            self.schema_ref.clone()
        }
    }
}

// Selects which version of an icechunk repo to read: a branch tip,
// a tag, or an exact snapshot id.
#[cfg(feature = "icechunk")]
#[derive(Clone, Debug)]
pub(crate) enum IcechunkVersion {
    BranchTip(String),
    RefTag(String),
    SnapshotId(String),
}

#[cfg(feature = "icechunk")]
impl Default for IcechunkVersion {
    fn default() -> Self {
        Self::BranchTip("main".to_string())
    }
}

#[cfg(feature = "icechunk")]
impl IcechunkVersion {
    // Resolve into an icechunk `VersionInfo`, parsing the snapshot id if needed.
    fn to_version_info(&self) -> DfResult<icechunk::repository::VersionInfo> {
        use icechunk::format::SnapshotId;
        use icechunk::repository::VersionInfo;

        Ok(match self {
            Self::BranchTip(branch) => VersionInfo::BranchTipRef(branch.clone()),
            Self::RefTag(tag) => VersionInfo::TagRef(tag.clone()),
            Self::SnapshotId(id) => {
                VersionInfo::SnapshotId(SnapshotId::try_from(id.as_str()).map_err(|e| {
                    DataFusionError::Execution(format!("invalid icechunk snapshot id '{id}': {e}"))
                })?)
            }
        })
    }
}

// We can create a table based on a directory with a supported zarr
// file/folder structure, or from an icechunk repo.
#[derive(Clone, Debug)]
pub(crate) enum ZarrTableUrl {
    ZarrStore(ListingTableUrl),
    #[cfg(feature = "icechunk")]
    IcechunkRepo(ListingTableUrl, IcechunkVersion),
}

#[cfg(feature = "icechunk")]
impl ZarrTableUrl {
    // Set the icechunk version selector. Errors if this is not an
    // icechunk repo.
    pub(crate) fn with_icechunk_version(self, version: IcechunkVersion) -> DfResult<Self> {
        match self {
            Self::IcechunkRepo(url, _) => Ok(Self::IcechunkRepo(url, version)),
            Self::ZarrStore(_) => Err(DataFusionError::Execution(
                "cannot set an icechunk version selector on a plain zarr store".into(),
            )),
        }
    }
}

impl ZarrTableUrl {
    async fn get_store_pointer_and_prefix(
        &self,
    ) -> DfResult<(
        Arc<dyn AsyncReadableListableStorageTraits + Unpin + Send>,
        Option<String>,
    )> {
        // the Option<String> that is returned here requires some explanation.
        // for some remote stores, the full url is not used as a prefix when
        // writing and reading from the store. for example for aws s3, it
        // seems the bucket is extracted from the url, but other than that
        // the object path is not kept, so when reading from the store, you
        // always need to provide a prefix. but for local object stores, it
        // actually can store the prefix. so we need to sometimes return no
        // prefix (None) and sometimes return one (Some(prefix)).
        match self {
            // this is for the case of a directory with a zarr file structure inside.
            Self::ZarrStore(table_url) => match table_url.scheme() {
                "file" => {
                    let path = PathBuf::from("/".to_owned() + table_url.prefix().as_ref());
                    let store = AsyncObjectStore::new(LocalFileSystem::new_with_prefix(path)?);
                    Ok((Arc::new(store), None))
                }
                #[cfg(feature = "s3")]
                "s3" => {
                    let store = AmazonS3Builder::from_env()
                        .with_url(table_url.get_url().as_str())
                        .build()?;
                    let store = AsyncObjectStore::new(store);
                    Ok((Arc::new(store), Some(table_url.prefix().to_string())))
                }
                #[cfg(feature = "gcs")]
                "gs" => {
                    let store = GoogleCloudStorageBuilder::from_env()
                        .with_url(table_url.get_url().as_str())
                        .build()?;
                    let store = AsyncObjectStore::new(store);
                    Ok((Arc::new(store), Some(table_url.prefix().to_string())))
                }
                _ => Err(DataFusionError::Execution(format!(
                    "Unsupported table url scheme {} for zarr stores",
                    table_url.scheme()
                ))),
            },

            // this is for the case of an icechunk repo.
            #[cfg(feature = "icechunk")]
            Self::IcechunkRepo(table_url, version) => {
                let object_storage = match table_url.scheme() {
                    "file" => {
                        let path = PathBuf::from("/".to_owned() + table_url.prefix().as_ref());
                        ObjectStorage::new_local_filesystem(&path)
                            .await
                            .map_err(|e| DataFusionError::External(Box::new(e)))?
                    }
                    #[cfg(feature = "s3")]
                    "s3" => {
                        use icechunk::config::{S3Credentials, S3Options};

                        let bucket = table_url
                            .object_store()
                            .as_str()
                            .replace("s3://", "")
                            .trim_end_matches("/")
                            .to_string();
                        let credentials = S3Credentials::FromEnv;
                        let config = S3Options {
                            region: env::var("AWS_DEFAULT_REGION").ok(),
                            endpoint_url: None,
                            anonymous: false,
                            allow_http: false,
                            force_path_style: false,
                            network_stream_timeout_seconds: None,
                            requester_pays: false,
                        };

                        ObjectStorage::new_s3(
                            bucket,
                            Some(table_url.prefix().as_ref().to_string()),
                            Some(credentials),
                            Some(config),
                        )
                        .await
                        .map_err(|e| DataFusionError::External(Box::new(e)))?
                    }
                    #[cfg(feature = "gcs")]
                    "gs" => {
                        use icechunk::config::GcsCredentials;

                        let bucket = table_url
                            .object_store()
                            .as_str()
                            .replace("gs://", "")
                            .trim_end_matches("/")
                            .to_string();
                        let credentials = GcsCredentials::FromEnv;

                        ObjectStorage::new_gcs(
                            bucket,
                            Some(table_url.prefix().as_ref().to_string()),
                            Some(credentials),
                            None,
                        )
                        .await
                        .map_err(|e| DataFusionError::External(Box::new(e)))?
                    }
                    _ => {
                        return Err(DataFusionError::Execution(format!(
                            "Unsupported table url scheme {} for icechunk repos",
                            table_url.scheme()
                        )))
                    }
                };
                let repo = Repository::open(None, Arc::new(object_storage), HashMap::new())
                    .await
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                let version_info = version.to_version_info()?;
                let session = repo
                    .readonly_session(&version_info)
                    .await
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                Ok((Arc::new(AsyncIcechunkStore::new(session)), None))
            }
        }
    }

    pub(crate) async fn infer_schema(&self) -> DfResult<(SchemaRef, ZarrStatsBase)> {
        let (store, store_prefix) = self.get_store_pointer_and_prefix().await?;
        let store_prefix = store_prefix
            .as_ref()
            .map_or("".into(), |p| p.to_owned() + "/");

        // zarr has no inherent column order, so we need to impose a deterministic
        // one. `list_prefix` returns the array prefixes in lexicographic order, so
        // the fields below end up sorted by array name without an explicit sort.
        // downstream code (and the test helpers) rely on this alphabetical ordering.
        let prefixes = store
            .list_prefix(
                &StorePrefix::new(store_prefix.to_owned())
                    .map_err(|e| DataFusionError::External(Box::new(e)))?,
            )
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let mut fields = Vec::with_capacity(prefixes.len());
        let mut stats_base = ZarrStatsBase::default();

        for prefix in prefixes {
            if prefix.as_str().contains("zarr.json") {
                let field_name = prefix.parent();
                if field_name.as_str() == "" {
                    continue;
                }

                // this is ugly, but I'm not sure there's a better way
                // to extract the array name...
                let field_name_prefix = field_name.parent();
                let mut field_name = field_name
                    .as_str()
                    .strip_suffix("/")
                    .ok_or(DataFusionError::Execution(
                        "Invalid directory name in zarr store".into(),
                    ))?
                    .to_string();
                let read_prefix = field_name.clone();
                if let Some(field_name_prefix) = field_name_prefix {
                    let to_remove = field_name_prefix.as_str();
                    field_name = field_name.replace(to_remove, "");
                }

                let arr = Array::async_open(store.clone(), &("/".to_owned() + &read_prefix))
                    .await
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                let meta = match arr.metadata() {
                    ArrayMetadata::V3(meta) => Ok(meta),
                    _ => Err(DataFusionError::Execution(
                        "Only Zarr v3 metadata is supported".into(),
                    )),
                }?;

                // capture the array's shape and dimension names for statistics.
                let dim_names: Option<Vec<String>> = arr
                    .dimension_names()
                    .clone()
                    .and_then(|names| names.into_iter().collect::<Option<Vec<String>>>());
                stats_base.add_array(&field_name, arr.shape(), dim_names);

                fields.push(Field::new(
                    field_name,
                    get_schema_type(&meta.data_type)?,
                    true,
                ));
            }
        }

        Ok((Arc::new(Schema::new(Fields::from(fields))), stats_base))
    }
}

fn get_schema_type(value: &MetadataV3) -> DfResult<DataType> {
    let data_type = zarr_dtype::from_metadata(value, &ExtensionAliases::default())
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

    match data_type {
        zarr_dtype::Bool => Ok(DataType::Boolean),
        zarr_dtype::UInt8 => Ok(DataType::UInt8),
        zarr_dtype::UInt16 => Ok(DataType::UInt16),
        zarr_dtype::UInt32 => Ok(DataType::UInt32),
        zarr_dtype::UInt64 => Ok(DataType::UInt64),
        zarr_dtype::Int8 => Ok(DataType::Int8),
        zarr_dtype::Int16 => Ok(DataType::Int16),
        zarr_dtype::Int32 => Ok(DataType::Int32),
        zarr_dtype::Int64 => Ok(DataType::Int64),
        zarr_dtype::Float32 => Ok(DataType::Float32),
        zarr_dtype::Float64 => Ok(DataType::Float64),
        zarr_dtype::String => Ok(DataType::Utf8),
        // datetime64 -> Timestamp (no timezone; zarr datetime64 carries none),
        // timedelta64 -> Duration. these must agree with the arrays produced by the
        // reader's decode_data, or the produced batch won't match the declared schema.
        zarr_dtype::NumpyDateTime64 { unit, scale_factor } => Ok(DataType::Timestamp(
            map_time_unit(unit, scale_factor.get())?,
            None,
        )),
        zarr_dtype::NumpyTimeDelta64 { unit, scale_factor } => {
            Ok(DataType::Duration(map_time_unit(unit, scale_factor.get())?))
        }
        _ => Err(DataFusionError::Execution(format!(
            "Unsupported type {value} from zarr metadata"
        ))),
    }
}

// maps a numpy temporal unit to an arrow [`TimeUnit`], requiring an exact match and a
// scale factor of 1. arrow can't represent the coarser/finer numpy units or a scale
// multiplier, so anything else is an error.
fn map_time_unit(unit: NumpyTimeUnit, scale_factor: u32) -> DfResult<TimeUnit> {
    if scale_factor != 1 {
        return Err(DataFusionError::Execution(format!(
            "Unsupported scale factor {scale_factor} for temporal type from zarr metadata"
        )));
    }
    match unit {
        NumpyTimeUnit::Second => Ok(TimeUnit::Second),
        NumpyTimeUnit::Millisecond => Ok(TimeUnit::Millisecond),
        NumpyTimeUnit::Microsecond => Ok(TimeUnit::Microsecond),
        NumpyTimeUnit::Nanosecond => Ok(TimeUnit::Nanosecond),
        _ => Err(DataFusionError::Execution(format!(
            "Unsupported temporal unit {unit} from zarr metadata"
        ))),
    }
}

#[cfg(test)]
mod zarr_config_tests {
    use super::*;
    #[cfg(feature = "icechunk")]
    use crate::test_utils::get_local_icechunk_repo;
    use crate::test_utils::{get_local_zarr_store, get_local_zarr_store_mix_dims};

    #[tokio::test]
    async fn schema_inference_tests() {
        // local zarr directory.
        let (wrapper, schema) = get_local_zarr_store(true, 0.0, "data_for_config_dir").await;
        let path = wrapper.get_store_path();

        let table_url = ListingTableUrl::parse(path).unwrap();
        let zarr_table_url = ZarrTableUrl::ZarrStore(table_url);
        let (inferred_schema, _stats) = zarr_table_url.infer_schema().await.unwrap();
        assert_eq!(inferred_schema, schema);

        // local icechunk repo.
        #[cfg(feature = "icechunk")]
        {
            let (wrapper, schema) =
                get_local_icechunk_repo(true, 0.0, "data_for_config_repo").await;
            let path = wrapper.get_store_path();

            let table_url = ListingTableUrl::parse(path).unwrap();
            let zarr_table_url = ZarrTableUrl::IcechunkRepo(table_url, IcechunkVersion::default());
            let (inferred_schema, _stats) = zarr_table_url.infer_schema().await.unwrap();
            assert_eq!(inferred_schema, schema);
        }
    }

    #[tokio::test]
    async fn statistics_tests() {
        // local zarr directory with mixed-dimension coordinates: `lat` is a 2D (8x8)
        // broadcasted coordinate, `lon` is 1D (8), and `data` is 2D (8x8). so `lat` and
        // `data` span both dimensions (64 rows), while `lon` spans a single one (8 rows).
        let (wrapper, schema) = get_local_zarr_store_mix_dims(0.0, "data_for_config_stats").await;
        let path = wrapper.get_store_path();
        let table_url = ZarrTableUrl::ZarrStore(ListingTableUrl::parse(path).unwrap());
        let (inferred_schema, stats_base) = table_url.infer_schema().await.unwrap();
        assert_eq!(inferred_schema, schema);

        let base =
            ZarrTableConfig::new(table_url, inferred_schema.clone()).with_stats_base(stats_base);
        let num_rows = |cols: &[&str]| {
            let projection: Vec<usize> = cols
                .iter()
                .map(|c| inferred_schema.index_of(c).unwrap())
                .collect();
            base.clone()
                .with_projection(projection)
                .statistics()
                .unwrap()
                .num_rows
        };

        assert_eq!(num_rows(&["lat"]), Precision::Exact(64));
        assert_eq!(num_rows(&["lon"]), Precision::Exact(8));
        assert_eq!(num_rows(&["data"]), Precision::Exact(64));
        assert_eq!(num_rows(&["data", "lat", "lon"]), Precision::Exact(64));

        // local icechunk repo with 1D `lat`/`lon` coordinates and 2D `data`. selecting
        // both coordinates broadcasts them to the full 2D grid -> 64 rows.
        #[cfg(feature = "icechunk")]
        {
            let (wrapper, _schema) =
                get_local_icechunk_repo(true, 0.0, "data_for_config_stats_repo").await;
            let path = wrapper.get_store_path();
            let table_url = ZarrTableUrl::IcechunkRepo(
                ListingTableUrl::parse(path).unwrap(),
                IcechunkVersion::default(),
            );
            let (inferred_schema, stats_base) = table_url.infer_schema().await.unwrap();

            let projection = vec![
                inferred_schema.index_of("lat").unwrap(),
                inferred_schema.index_of("lon").unwrap(),
            ];
            let config = ZarrTableConfig::new(table_url, inferred_schema)
                .with_stats_base(stats_base)
                .with_projection(projection);
            assert_eq!(config.statistics().unwrap().num_rows, Precision::Exact(64));
        }
    }
}
