#[cfg(feature = "icechunk")]
use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::error::{DataFusionError, Result as DfResult};
#[cfg(feature = "icechunk")]
use icechunk::{ObjectStorage, Repository};
use object_store::aws::AmazonS3Builder;
use object_store::local::LocalFileSystem;
use zarrs::array::data_type::DataType as zarr_dtype;
use zarrs::array::Array;
use zarrs::registry::ExtensionAliases;
#[cfg(feature = "icechunk")]
use zarrs_icechunk::AsyncIcechunkStore;
use zarrs_metadata::v3::MetadataV3;
use zarrs_metadata::ArrayMetadata;
use zarrs_object_store::AsyncObjectStore;
use zarrs_storage::{AsyncReadableListableStorageTraits, StorePrefix};
/// A zarr table configuration.
#[derive(Clone, Debug)]
pub struct ZarrTableConfig {
    schema_ref: SchemaRef,
    table_url: ZarrTableUrl,
    projection: Option<Vec<usize>>,
}

impl ZarrTableConfig {
    pub(crate) fn new(table_url: ZarrTableUrl, schema_ref: SchemaRef) -> Self {
        Self {
            schema_ref,
            table_url,
            projection: None,
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

/// We can create a table based on a directory with a supported zarr
/// file/folder structure, or from an icechunk repo.
#[derive(Clone, Debug)]
pub(crate) enum ZarrTableUrl {
    ZarrStore(ListingTableUrl),
    #[cfg(feature = "icechunk")]
    IcechunkRepo(ListingTableUrl),
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
        // seems the bucket is extracted from the url, but not the rest, so
        // when reading from the store, you always need to provide a prefix
        // to get to the actual zarr store. but for local object stores, it
        // actually can store the prefix, to be applied when you read from
        // the store. so we need to sometimes return no prefix (None) and
        // sometimes return one (Some(prefix)).
        match self {
            // this is for the case of a directory with a zarr file structure inside.
            Self::ZarrStore(table_url) => match table_url.scheme() {
                "file" => {
                    let path = PathBuf::from("/".to_owned() + table_url.prefix().as_ref());
                    let store = AsyncObjectStore::new(LocalFileSystem::new_with_prefix(path)?);
                    Ok((Arc::new(store), None))
                }
                "s3" => {
                    let store = AmazonS3Builder::from_env()
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

            // this is for the case of an icechunk repo. note that here we hard code
            // reading from the main branch, and "as of" now.
            #[cfg(feature = "icechunk")]
            Self::IcechunkRepo(table_url) => {
                let object_storage = match table_url.scheme() {
                    "file" => {
                        let path = PathBuf::from("/".to_owned() + table_url.prefix().as_ref());
                        ObjectStorage::new_local_filesystem(&path)
                            .await
                            .map_err(|e| DataFusionError::External(Box::new(e)))?
                    }
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
                let session = repo
                    .readonly_session(&icechunk::repository::VersionInfo::AsOf {
                        branch: "main".into(),
                        at: chrono::Utc::now(),
                    })
                    .await
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                Ok((Arc::new(AsyncIcechunkStore::new(session)), None))
            }
        }
    }

    pub(crate) async fn infer_schema(&self) -> DfResult<SchemaRef> {
        let (store, store_prefix) = self.get_store_pointer_and_prefix().await?;
        let store_prefix = store_prefix
            .as_ref()
            .map_or("".into(), |p| p.to_owned() + "/");

        let prefixes = store
            .list_prefix(
                &StorePrefix::new(store_prefix.to_owned())
                    .map_err(|e| DataFusionError::External(Box::new(e)))?,
            )
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let mut fields = Vec::with_capacity(prefixes.len());

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

                fields.push(Field::new(
                    field_name,
                    get_schema_type(&meta.data_type)?,
                    true,
                ));
            }
        }

        Ok(Arc::new(Schema::new(Fields::from(fields))))
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
        _ => Err(DataFusionError::Execution(format!(
            "Unsupported type {value} from zarr metadata"
        ))),
    }
}

#[cfg(test)]
mod zarr_config_tests {
    use super::*;
    #[cfg(feature = "icechunk")]
    use crate::test_utils::get_local_icechunk_repo;
    use crate::test_utils::get_local_zarr_store;

    #[tokio::test]
    async fn schema_inference_tests() {
        // local zarr directory.
        let (wrapper, schema) = get_local_zarr_store(true, 0.0, "data_for_config_dir").await;
        let path = wrapper.get_store_path();

        let table_url = ListingTableUrl::parse(path).unwrap();
        let zarr_table_url = ZarrTableUrl::ZarrStore(table_url);
        let inferred_schema = zarr_table_url.infer_schema().await.unwrap();
        assert_eq!(inferred_schema, schema);

        // local icechunk repo.
        #[cfg(feature = "icechunk")]
        {
            let (wrapper, schema) =
                get_local_icechunk_repo(true, 0.0, "data_for_config_repo").await;
            let path = wrapper.get_store_path();

            let table_url = ListingTableUrl::parse(path).unwrap();
            let zarr_table_url = ZarrTableUrl::IcechunkRepo(table_url);
            let inferred_schema = zarr_table_url.infer_schema().await.unwrap();
            assert_eq!(inferred_schema, schema);
        }
    }
}
