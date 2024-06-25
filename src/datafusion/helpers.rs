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

use crate::reader::{ZarrArrowPredicate, ZarrChunkFilter, ZarrProjection};
use arrow::array::{ArrayRef, BooleanArray, StringBuilder};
use arrow::compute::{and, cast, prep_null_mask_filter};
use arrow::datatypes::{DataType, Field};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::Array;
use arrow_schema::{Fields, Schema};
use datafusion::datasource::listing::{ListingTableUrl, PartitionedFile};
use datafusion_common::cast::as_boolean_array;
use datafusion_common::scalar::ScalarValue;
use datafusion_common::tree_node::{RewriteRecursion, TreeNode, TreeNodeRewriter, VisitRecursion};
use datafusion_common::Result as DataFusionResult;
use datafusion_common::{internal_err, DFField, DFSchema, DataFusionError};
use datafusion_expr::{Expr, ScalarFunctionDefinition, Volatility};
use datafusion_physical_expr::create_physical_expr;
use datafusion_physical_expr::execution_props::ExecutionProps;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::utils::reassign_predicate_columns;
use datafusion_physical_expr::{split_conjunction, PhysicalExpr};
use futures::stream::FuturesUnordered;
use futures::stream::{self, BoxStream, StreamExt};
use object_store::path::DELIMITER;
use object_store::{path::Path, ObjectStore};
use std::collections::BTreeSet;
use std::sync::Arc;

// Checks whether the given expression can be resolved using only the columns `col_names`.
// Copied from datafusion, because it's not accessible from the outside.
pub fn expr_applicable_for_cols(col_names: &[String], expr: &Expr) -> bool {
    let mut is_applicable = true;
    expr.apply(&mut |expr| match expr {
        Expr::Column(datafusion_common::Column { ref name, .. }) => {
            is_applicable &= col_names.contains(name);
            if is_applicable {
                Ok(VisitRecursion::Skip)
            } else {
                Ok(VisitRecursion::Stop)
            }
        }
        Expr::Literal(_)
        | Expr::Alias(_)
        | Expr::OuterReferenceColumn(_, _)
        | Expr::ScalarVariable(_, _)
        | Expr::Not(_)
        | Expr::IsNotNull(_)
        | Expr::IsNull(_)
        | Expr::IsTrue(_)
        | Expr::IsFalse(_)
        | Expr::IsUnknown(_)
        | Expr::IsNotTrue(_)
        | Expr::IsNotFalse(_)
        | Expr::IsNotUnknown(_)
        | Expr::Negative(_)
        | Expr::Cast { .. }
        | Expr::TryCast { .. }
        | Expr::BinaryExpr { .. }
        | Expr::Between { .. }
        | Expr::Like { .. }
        | Expr::SimilarTo { .. }
        | Expr::InList { .. }
        | Expr::Exists { .. }
        | Expr::InSubquery(_)
        | Expr::ScalarSubquery(_)
        | Expr::GetIndexedField { .. }
        | Expr::GroupingSet(_)
        | Expr::Case { .. } => Ok(VisitRecursion::Continue),

        Expr::ScalarFunction(scalar_function) => match &scalar_function.func_def {
            ScalarFunctionDefinition::BuiltIn(fun) => match fun.volatility() {
                Volatility::Immutable => Ok(VisitRecursion::Continue),
                Volatility::Stable | Volatility::Volatile => {
                    is_applicable = false;
                    Ok(VisitRecursion::Stop)
                }
            },
            ScalarFunctionDefinition::UDF(fun) => match fun.signature().volatility {
                Volatility::Immutable => Ok(VisitRecursion::Continue),
                Volatility::Stable | Volatility::Volatile => {
                    is_applicable = false;
                    Ok(VisitRecursion::Stop)
                }
            },
            ScalarFunctionDefinition::Name(_) => {
                internal_err!("Function `Expr` with name should be resolved.")
            }
        },

        Expr::AggregateFunction { .. }
        | Expr::Sort { .. }
        | Expr::WindowFunction { .. }
        | Expr::Wildcard { .. }
        | Expr::Unnest { .. }
        | Expr::Placeholder(_) => {
            is_applicable = false;
            Ok(VisitRecursion::Stop)
        }
    })
    .unwrap();
    is_applicable
}

// Below is all the logic necessary (I think) to convert a PhysicalExpr into a ZarrChunkFilter.
// The logic is mostly copied from datafusion, and is simplified here for the zarr use case.
pub struct ZarrFilterCandidate {
    expr: Arc<dyn PhysicalExpr>,
    projection: Vec<usize>,
}

struct ZarrFilterCandidateBuilder<'a> {
    expr: Arc<dyn PhysicalExpr>,
    file_schema: &'a Schema,
    required_column_indices: BTreeSet<usize>,
    projected_columns: bool,
}

impl<'a> ZarrFilterCandidateBuilder<'a> {
    pub fn new(expr: Arc<dyn PhysicalExpr>, file_schema: &'a Schema) -> Self {
        Self {
            expr,
            file_schema,
            required_column_indices: BTreeSet::default(),
            projected_columns: false,
        }
    }

    pub fn build(mut self) -> DataFusionResult<Option<ZarrFilterCandidate>> {
        let expr = self.expr.clone().rewrite(&mut self)?;

        // if we are dealing with a projected column, which here means it's
        // a partitioned column, we don't produce a filter for it.
        if self.projected_columns {
            return Ok(None);
        }

        Ok(Some(ZarrFilterCandidate {
            expr,
            projection: self.required_column_indices.into_iter().collect(),
        }))
    }
}

impl<'a> TreeNodeRewriter for ZarrFilterCandidateBuilder<'a> {
    type N = Arc<dyn PhysicalExpr>;

    fn pre_visit(&mut self, node: &Arc<dyn PhysicalExpr>) -> DataFusionResult<RewriteRecursion> {
        if let Some(column) = node.as_any().downcast_ref::<Column>() {
            if let Ok(idx) = self.file_schema.index_of(column.name()) {
                self.required_column_indices.insert(idx);
            } else {
                // set the flag is we detect that the column is not in the file schema. for the
                // zarr implementation, this would mean that the column is actually a partitioned
                // column, and we shouldn't be pushing down a filter for it.
                // TODO handle cases where a filter contains a column that doesn't exist (not even
                // as a partition).
                self.projected_columns = true;
                return Ok(RewriteRecursion::Stop);
            }
        }

        Ok(RewriteRecursion::Continue)
    }

    fn mutate(&mut self, expr: Arc<dyn PhysicalExpr>) -> DataFusionResult<Arc<dyn PhysicalExpr>> {
        Ok(expr)
    }
}

#[derive(Clone)]
pub struct ZarrDatafusionArrowPredicate {
    physical_expr: Arc<dyn PhysicalExpr>,
    projection_mask: ZarrProjection,
    projection: Vec<String>,
}

impl ZarrDatafusionArrowPredicate {
    pub fn new(candidate: ZarrFilterCandidate, schema: &Schema) -> DataFusionResult<Self> {
        let cols: Vec<_> = candidate
            .projection
            .iter()
            .map(|idx| schema.field(*idx).name().to_string())
            .collect();

        let schema = Arc::new(schema.project(&candidate.projection)?);
        let physical_expr = reassign_predicate_columns(candidate.expr, &schema, true)?;

        Ok(Self {
            physical_expr,
            projection_mask: ZarrProjection::keep(cols.clone()),
            projection: cols,
        })
    }
}

impl ZarrArrowPredicate for ZarrDatafusionArrowPredicate {
    fn projection(&self) -> &ZarrProjection {
        &self.projection_mask
    }

    fn evaluate(&mut self, batch: &RecordBatch) -> Result<BooleanArray, ArrowError> {
        let index_projection = self
            .projection
            .iter()
            .map(|col| batch.schema().index_of(col))
            .collect::<Result<Vec<_>, _>>()?;
        let batch = batch.project(&index_projection[..])?;

        match self
            .physical_expr
            .evaluate(&batch)
            .and_then(|v| v.into_array(batch.num_rows()))
        {
            Ok(array) => {
                let bool_arr = as_boolean_array(&array)?.clone();
                Ok(bool_arr)
            }
            Err(e) => Err(ArrowError::ComputeError(format!(
                "Error evaluating filter predicate: {e:?}"
            ))),
        }
    }
}

pub(crate) fn build_row_filter(
    expr: &Arc<dyn PhysicalExpr>,
    file_schema: &Schema,
) -> DataFusionResult<Option<ZarrChunkFilter>> {
    let predicates = split_conjunction(expr);
    let candidates: Vec<ZarrFilterCandidate> = predicates
        .into_iter()
        .flat_map(|expr| {
            if let Ok(candidate) =
                ZarrFilterCandidateBuilder::new(expr.clone(), file_schema).build()
            {
                candidate
            } else {
                None
            }
        })
        .collect();

    if candidates.is_empty() {
        Ok(None)
    } else {
        let mut filters: Vec<Box<dyn ZarrArrowPredicate>> = vec![];
        for candidate in candidates {
            let filter = ZarrDatafusionArrowPredicate::new(candidate, file_schema)?;
            filters.push(Box::new(filter));
        }

        let chunk_filter = ZarrChunkFilter::new(filters);

        Ok(Some(chunk_filter))
    }
}

// Below is all the logic related to hive still partitioning, mostly copied and
// slightly modified from datafusion.
const CONCURRENCY_LIMIT: usize = 100;
const MAX_PARTITION_DEPTH: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Partition {
    /// The path to the partition, including the table prefix
    path: Path,
    /// How many path segments below the table prefix `path` contains
    /// or equivalently the number of partition values in `path`
    depth: usize,
}

impl Partition {
    /// List the direct children of this partition updating `self.files` with
    /// any child files, and returning a list of child "directories"
    async fn list(self, store: &dyn ObjectStore) -> DataFusionResult<(Self, Vec<Path>)> {
        let prefix = Some(&self.path).filter(|p| !p.as_ref().is_empty());
        let result = store.list_with_delimiter(prefix).await?;
        Ok((self, result.common_prefixes))
    }
}

async fn list_partitions(
    store: &dyn ObjectStore,
    table_path: &ListingTableUrl,
    max_depth: usize,
) -> DataFusionResult<Vec<Partition>> {
    let partition = Partition {
        path: table_path.prefix().clone(),
        depth: 0,
    };

    let mut final_partitions = Vec::with_capacity(MAX_PARTITION_DEPTH);
    let mut pending = vec![];
    let mut futures = FuturesUnordered::new();
    futures.push(partition.list(store));

    while let Some((partition, paths)) = futures.next().await.transpose()? {
        // If pending contains a future it implies prior to this iteration
        // `futures.len == CONCURRENCY_LIMIT`. We can therefore add a single
        // future from `pending` to the working set
        if let Some(next) = pending.pop() {
            futures.push(next)
        }

        let depth = partition.depth;
        if depth == max_depth {
            final_partitions.push(partition);
        }
        for path in paths {
            let child = Partition {
                path,
                depth: depth + 1,
            };
            // if we have reached the max depth, we don't need to list all the
            // directories under the last partition, those will all be zarr arrays,
            // and the last partition itself will be a store to be read atomically.
            if depth < max_depth {
                match futures.len() < CONCURRENCY_LIMIT {
                    true => futures.push(child.list(store)),
                    false => pending.push(child.list(store)),
                }
            }
        }
    }
    Ok(final_partitions)
}

fn parse_partitions_for_path(
    table_path: &ListingTableUrl,
    file_path: &Path,
    table_partition_cols: Vec<&str>,
) -> Option<Vec<String>> {
    let mut stripped = file_path
        .as_ref()
        .strip_prefix(table_path.prefix().as_ref())?;
    if !stripped.is_empty() && !table_path.prefix().as_ref().is_empty() {
        stripped = stripped.strip_prefix(DELIMITER)?;
    }
    let subpath = stripped.split_terminator(DELIMITER).map(|s| s.to_string());

    let mut part_values = vec![];
    for (part, pn) in subpath.zip(table_partition_cols) {
        match part.split_once('=') {
            Some((name, val)) if name == pn => part_values.push(val.to_string()),
            _ => {
                return None;
            }
        }
    }
    Some(part_values)
}

async fn prune_partitions(
    table_path: &ListingTableUrl,
    partitions: Vec<Partition>,
    filters: &[Expr],
    partition_cols: &[(String, DataType)],
) -> DataFusionResult<Vec<Partition>> {
    if filters.is_empty() {
        return Ok(partitions);
    }

    let mut builders: Vec<_> = (0..partition_cols.len())
        .map(|_| StringBuilder::with_capacity(partitions.len(), partitions.len() * 10))
        .collect();

    for partition in &partitions {
        let cols = partition_cols.iter().map(|x| x.0.as_str()).collect();
        let parsed =
            parse_partitions_for_path(table_path, &partition.path, cols).unwrap_or_default();

        let mut builders = builders.iter_mut();
        for (p, b) in parsed.iter().zip(&mut builders) {
            b.append_value(p);
        }
        builders.for_each(|b| b.append_null());
    }

    let arrays = partition_cols
        .iter()
        .zip(builders)
        .map(|((_, d), mut builder)| {
            let array = builder.finish();
            cast(&array, d)
        })
        .collect::<Result<_, _>>()?;

    let fields: Fields = partition_cols
        .iter()
        .map(|(n, d)| Field::new(n, d.clone(), true))
        .collect();
    let schema = Arc::new(Schema::new(fields));

    let df_schema = DFSchema::new_with_metadata(
        partition_cols
            .iter()
            .map(|(n, d)| DFField::new_unqualified(n, d.clone(), true))
            .collect(),
        Default::default(),
    )?;

    let batch = RecordBatch::try_new(schema.clone(), arrays)?;

    // TODO: Plumb this down
    let props = ExecutionProps::new();

    // Applies `filter` to `batch` returning `None` on error
    let do_filter = |filter| -> Option<ArrayRef> {
        let expr = create_physical_expr(filter, &df_schema, &props).ok()?;
        expr.evaluate(&batch)
            .ok()?
            .into_array(partitions.len())
            .ok()
    };

    // Compute the conjunction of the filters, ignoring errors
    let mask = filters
        .iter()
        .fold(None, |acc, filter| match (acc, do_filter(filter)) {
            (Some(a), Some(b)) => Some(and(&a, b.as_boolean()).unwrap_or(a)),
            (None, Some(r)) => Some(r.as_boolean().clone()),
            (r, None) => r,
        });

    let mask = match mask {
        Some(mask) => mask,
        None => return Ok(partitions),
    };

    // Don't retain partitions that evaluated to null
    let prepared = match mask.null_count() {
        0 => mask,
        _ => prep_null_mask_filter(&mask),
    };

    let filtered = partitions
        .into_iter()
        .zip(prepared.values())
        .filter_map(|(p, f)| f.then_some(p))
        .collect();

    Ok(filtered)
}

pub async fn pruned_partition_list<'a>(
    store: &'a dyn ObjectStore,
    table_path: &'a ListingTableUrl,
    filters: &'a [Expr],
    partition_cols: &'a [(String, DataType)],
) -> DataFusionResult<BoxStream<'a, DataFusionResult<PartitionedFile>>> {
    // if no partition col => simply return the table path
    if partition_cols.is_empty() {
        let pf = PartitionedFile::new(table_path.prefix().clone(), 0);
        return Ok(Box::pin(stream::iter(vec![Ok(pf)])));
    }

    let partitions = list_partitions(store, table_path, partition_cols.len()).await?;
    let pruned = prune_partitions(table_path, partitions, filters, partition_cols).await?;

    let stream = futures::stream::iter(pruned)
        .map(move |partition: Partition| async move {
            let cols = partition_cols.iter().map(|x| x.0.as_str()).collect();
            let parsed = parse_partitions_for_path(table_path, &partition.path, cols);

            let partition_values = parsed
                .into_iter()
                .flatten()
                .zip(partition_cols)
                .map(|(parsed, (_, datatype))| {
                    ScalarValue::try_from_string(parsed.to_string(), datatype)
                })
                .collect::<DataFusionResult<Vec<_>>>()?;

            let mut pf = PartitionedFile::new(partition.path, 0);
            pf.partition_values.clone_from(&partition_values);

            Ok(pf)
        })
        .buffer_unordered(CONCURRENCY_LIMIT)
        .boxed();

    Ok(stream)
}

// copied from datafusion
pub fn split_files(
    mut partitioned_files: Vec<PartitionedFile>,
    n: usize,
) -> Vec<Vec<PartitionedFile>> {
    if partitioned_files.is_empty() {
        return vec![];
    }

    // ObjectStore::list does not guarantee any consistent order and for some
    // implementations such as LocalFileSystem, it may be inconsistent. Thus
    // Sort files by path to ensure consistent plans when run more than once.
    partitioned_files.sort_by(|a, b| a.path().cmp(b.path()));

    // effectively this is div with rounding up instead of truncating
    let chunk_size = (partitioned_files.len() + n - 1) / n;
    partitioned_files
        .chunks(chunk_size)
        .map(|c| c.to_vec())
        .collect()
}

#[cfg(test)]
mod helpers_tests {
    use super::*;
    use crate::tests::get_test_v2_data_path;
    use datafusion_expr::{and, col, lit};
    use itertools::Itertools;
    use object_store::local::LocalFileSystem;

    #[tokio::test]
    async fn test_listing_and_pruning_partitions() {
        let table_path = get_test_v2_data_path("lat_lon_w_groups_example.zarr".to_string())
            .to_str()
            .unwrap()
            .to_string();

        let store = LocalFileSystem::new();
        let url = ListingTableUrl::parse(table_path).unwrap();
        let partitions = list_partitions(&store, &url, 2).await.unwrap();

        let expr1 = col("var").eq(lit(1_i32));
        let expr2 = col("other_var").eq(lit::<String>("b".to_string()));
        let partition_cols = [
            ("var".to_string(), DataType::Int32),
            ("other_var".to_string(), DataType::Utf8),
        ];

        let prefix = "home/max/Documents/repos/arrow-zarr/test-data/data/zarr/v2_data/lat_lon_w_groups_example.zarr";
        let part_1a = Partition {
            path: Path::parse(prefix)
                .unwrap()
                .child("var=1")
                .child("other_var=a"),
            depth: 2,
        };
        let part_1b = Partition {
            path: Path::parse(prefix)
                .unwrap()
                .child("var=1")
                .child("other_var=b"),
            depth: 2,
        };
        let part_2b = Partition {
            path: Path::parse(prefix)
                .unwrap()
                .child("var=2")
                .child("other_var=b"),
            depth: 2,
        };

        let filters = [expr1.clone()];
        let pruned = prune_partitions(&url, partitions.clone(), &filters, &partition_cols)
            .await
            .unwrap();
        assert_eq!(
            pruned.into_iter().sorted().collect::<Vec<_>>(),
            vec![part_1a.clone(), part_1b.clone()]
                .into_iter()
                .sorted()
                .collect::<Vec<_>>(),
        );

        let filters = [expr2.clone()];
        let pruned = prune_partitions(&url, partitions.clone(), &filters, &partition_cols)
            .await
            .unwrap();
        assert_eq!(
            pruned.into_iter().sorted().collect::<Vec<_>>(),
            vec![part_1b.clone(), part_2b.clone()]
                .into_iter()
                .sorted()
                .collect::<Vec<_>>(),
        );

        let expr = and(expr1, expr2);
        let filters = [expr];
        let pruned = prune_partitions(&url, partitions.clone(), &filters, &partition_cols)
            .await
            .unwrap();
        assert_eq!(
            pruned.into_iter().sorted().collect::<Vec<_>>(),
            vec![part_1b].into_iter().sorted().collect::<Vec<_>>(),
        );
    }
}
