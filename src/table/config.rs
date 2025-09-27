#[cfg(feature = "icechunk")]
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::error::{DataFusionError, Result as DfResult};
#[cfg(feature = "icechunk")]
use icechunk::{ObjectStorage, Repository};
use object_store::local::LocalFileSystem;
use zarrs::array::Array;
#[cfg(feature = "icechunk")]
use zarrs_icechunk::AsyncIcechunkStore;
use zarrs_metadata::v3::array::data_type::DataTypeMetadataV3;
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

    pub(crate) async fn get_store_pointer(
        &self,
    ) -> DfResult<Arc<dyn AsyncReadableListableStorageTraits + Unpin + Send>> {
        self.table_url.get_store_pointer().await
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
    async fn get_store_pointer(
        &self,
    ) -> DfResult<Arc<dyn AsyncReadableListableStorageTraits + Unpin + Send>> {
        // currently only local storage is supported.
        match self {
            // this is for the case of a directory with a zarr file structure inside.
            Self::ZarrStore(table_url) => match table_url.scheme() {
                "file" => {
                    let path = PathBuf::from("/".to_owned() + table_url.prefix().as_ref());
                    let store = AsyncObjectStore::new(LocalFileSystem::new_with_prefix(path)?);
                    Ok(Arc::new(store))
                }
                _ => Err(DataFusionError::Execution(format!(
                    "Invalid table url scheme {}",
                    table_url.scheme()
                ))),
            },

            // this is for the case of an icechunk repo. note that here we hard code
            // reading from the main branch, and "as of" now.
            #[cfg(feature = "icechunk")]
            Self::IcechunkRepo(table_url) => match table_url.scheme() {
                "file" => {
                    let path = PathBuf::from("/".to_owned() + table_url.prefix().as_ref());
                    let object_storage = ObjectStorage::new_local_filesystem(&path)
                        .await
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
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
                    Ok(Arc::new(AsyncIcechunkStore::new(session)))
                }
                _ => Err(DataFusionError::Execution(format!(
                    "Invalid table url scheme {}",
                    table_url.scheme()
                ))),
            },
        }
    }

    pub(crate) async fn infer_schema(&self) -> DfResult<SchemaRef> {
        let store = self.get_store_pointer().await?;
        let prefixes = store
            .list_prefix(&StorePrefix::new("").map_err(|e| DataFusionError::External(Box::new(e)))?)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        let mut fields = Vec::with_capacity(prefixes.len());

        for prefix in prefixes {
            if prefix.as_str().contains("zarr.json") {
                let parent = prefix.parent();
                // Skip the root zarr.json (group metadata file)
                if parent.as_str().is_empty() {
                    continue;
                }
                let field_name =
                    parent
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
        }

        Ok(Arc::new(Schema::new(Fields::from(fields))))
    }
}

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
