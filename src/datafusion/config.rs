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

use object_store::ObjectStore;

/// Configuration for Zarr DataFusion processing.
#[derive(Clone)]
pub struct ZarrConfig {
    /// The object store to use.
    pub object_store: Arc<dyn ObjectStore>,

    /// The projection for the scan.
    pub projection: Option<Vec<usize>>,
}

impl ZarrConfig {
    /// Create a new ZarrConfig.
    pub fn new(object_store: Arc<dyn ObjectStore>) -> Self {
        Self {
            object_store,
            projection: None,
        }
    }

    /// Set the projection for the scan.
    pub fn with_projection(mut self, projection: Option<Vec<usize>>) -> Self {
        self.projection = projection;
        self
    }
}
