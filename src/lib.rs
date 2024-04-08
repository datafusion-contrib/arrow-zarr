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

pub mod async_reader;
pub mod reader;

#[cfg(feature = "datafusion")]
pub mod datafusion;

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    #[cfg(feature = "datafusion")]
    pub(crate) fn get_test_v2_data_path(zarr_store: String) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test-data/data/zarr/v2_data")
            .join(zarr_store)
    }

    pub(crate) fn get_test_v3_data_path(zarr_array: String) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test-data/data/zarr/v3_data")
            .join(zarr_array)
    }
}
