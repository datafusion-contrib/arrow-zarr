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

use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, Fields};

/// Projects onto the named columns, in the order of the names.
pub(crate) trait ProjectByNames: Sized {
    fn project_by_names<S: AsRef<str>>(
        &self,
        names: impl IntoIterator<Item = S>,
    ) -> Result<Self, ArrowError>;
}

impl ProjectByNames for RecordBatch {
    fn project_by_names<S: AsRef<str>>(
        &self,
        names: impl IntoIterator<Item = S>,
    ) -> Result<Self, ArrowError> {
        let schema = self.schema();
        let indices = names
            .into_iter()
            .map(|name| schema.index_of(name.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        self.project(&indices)
    }
}

/// The names of the columns, in order.
pub(crate) trait ColumnNames {
    fn iter_column_names(&self) -> impl Iterator<Item = String> + '_;

    fn column_names(&self) -> Vec<String> {
        self.iter_column_names().collect()
    }
}

impl ColumnNames for Fields {
    fn iter_column_names(&self) -> impl Iterator<Item = String> + '_ {
        self.iter().map(|f| f.name().clone())
    }
}

/// The number of elements in an array (or chunk) of a given shape.
pub(crate) trait NumElements {
    fn num_elements(&self) -> u64;
}

impl NumElements for [u64] {
    fn num_elements(&self) -> u64 {
        self.iter().product()
    }
}
