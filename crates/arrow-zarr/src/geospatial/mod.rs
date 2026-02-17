pub(crate) mod build_side;
pub(crate) mod exec;
pub(crate) mod extension_traits;
pub(crate) mod operations;
pub(crate) mod optimizer;
pub(crate) mod probe_side;
pub(crate) mod spatial_predicate;
pub(crate) mod stream;
pub(crate) mod udfs;

pub use exec::BulkSpatialJoinExec;
pub use optimizer::SpatialJoinPhysicalOptimizer;
pub use udfs::{StContainsUdf, StWithinUdf};

#[cfg(test)]
pub(crate) mod test_utils;
#[cfg(test)]
mod tests;
