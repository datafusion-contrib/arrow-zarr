// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

pub(crate) mod boxed_geo_batch;
pub(crate) mod indexed_build_side;
pub(crate) mod joinable_geo;
pub(crate) mod probe_pruning_expr;
pub(crate) mod spatial_join_exec;
pub(crate) mod spatial_join_optimizer;
pub(crate) mod spatial_join_stream;
pub mod spatial_predicate;
pub(crate) mod st_within;
pub mod udfs;

pub use spatial_join_exec::SpatialJoinExec;
pub use spatial_join_optimizer::SpatialJoinPhysicalOptimizer;
pub use udfs::{StContainsUdf, StWithinUdf};

// #[cfg(test)]
pub(crate) mod test_utils;
