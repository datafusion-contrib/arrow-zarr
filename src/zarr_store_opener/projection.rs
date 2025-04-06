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

use crate::errors::zarr_errors::{ZarrQueryError, ZarrQueryResult};

use itertools::Itertools;
// The projection type, we either skip or select columns, and if
// it's the latter, we can select by column name or index.
// The projection can also be null, in which case it does nothing.
// We could use it as an option<ProjectionType> without the
// null type, but hving the null variant allows the projection
// type to be instantiated with nothing and be updated later on.
#[derive(Clone, PartialEq, Debug)]
pub(crate) enum ProjectionType {
    Select,
    SelectByIndex,
    Null,
}

/// A structure to handle skipping or selecting specific columns (zarr
/// arrays) from a zarr store.
#[derive(Clone, Debug)]
pub struct ZarrQueryProjection {
    projection_type: ProjectionType,
    col_names: Option<Vec<String>>,
    col_indices: Option<Vec<usize>>,
}

impl From<Vec<usize>> for ZarrQueryProjection {
    fn from(indices: Vec<usize>) -> Self {
        Self::keep_by_index(indices)
    }
}

impl From<Option<&Vec<usize>>> for ZarrQueryProjection {
    fn from(indices: Option<&Vec<usize>>) -> Self {
        match indices {
            Some(i) => Self::keep_by_index(i.to_vec()),
            None => Self::all(),
        }
    }
}

impl ZarrQueryProjection {
    /// Create a projection that keeps all columns.
    pub fn all() -> Self {
        Self {
            projection_type: ProjectionType::Null,
            col_names: None,
            col_indices: None,
        }
    }

    /// Create a projection that keeps certain columns (and skips all the other columns).
    pub fn keep(col_names: Vec<String>) -> Self {
        Self {
            projection_type: ProjectionType::Select,
            col_names: Some(col_names),
            col_indices: None,
        }
    }

    /// Create a projection that keeps certain columns by index (and skips all the other columns).
    pub fn keep_by_index(col_indices: Vec<usize>) -> Self {
        Self {
            projection_type: ProjectionType::SelectByIndex,
            col_names: None,
            col_indices: Some(col_indices),
        }
    }

    pub(crate) fn apply_selection(&self, all_cols: &[String]) -> ZarrQueryResult<Vec<String>> {
        match self.projection_type {
            // for the Null case, we sort the columns, because zarr stores
            // don't have a built-in ordering, since arrays are not stored
            // in the same file. but we do need to have some deterministic
            // ordering, so we relying on alphabetical ordering.
            ProjectionType::Null => Ok(all_cols.iter().cloned().sorted().collect()),
            ProjectionType::Select => {
                let col_names = self.col_names.as_ref().unwrap();
                for col in col_names {
                    if !all_cols.contains(col) {
                        return Err(ZarrQueryError::InvalidProjection(
                            "Projection column not in columns to select from".to_string(),
                        ));
                    }
                }
                Ok(col_names.clone())
            }
            ProjectionType::SelectByIndex => {
                let col_indices = self.col_indices.as_ref().unwrap();
                let col_names: Vec<String> = col_indices
                    .iter()
                    .map(|i| all_cols[*i].to_string())
                    .collect();
                Ok(col_names)
            }
        }
    }

    // update a projection with another projection, effectively combining
    // the 2 projection. Only certain combinations are alloed.
    pub(crate) fn update(&mut self, other_proj: ZarrQueryProjection) -> ZarrQueryResult<()> {
        match (&self.projection_type, &other_proj.projection_type) {
            (_, ProjectionType::Null) => (),
            (ProjectionType::Null, _) => {
                self.projection_type = other_proj.projection_type;
                self.col_names = other_proj.col_names;
                self.col_indices = other_proj.col_indices;
            }
            (ProjectionType::SelectByIndex, ProjectionType::SelectByIndex) => {
                let mut indices =
                    self.col_indices
                        .take()
                        .ok_or(ZarrQueryError::InvalidProjection(
                            "Projection has no indices".to_string(),
                        ))?;
                for i in other_proj
                    .col_indices
                    .ok_or(ZarrQueryError::InvalidProjection(
                        "Projection update has no indices".to_string(),
                    ))?
                {
                    if !indices.contains(&i) {
                        indices.push(i);
                    }
                }
                self.col_indices = Some(indices);
            }
            (ProjectionType::Select, ProjectionType::Select) => {
                let mut col_names =
                    self.col_names
                        .take()
                        .ok_or(ZarrQueryError::InvalidProjection(
                            "Projection has no column names".to_string(),
                        ))?;
                for col in other_proj
                    .col_names
                    .ok_or(ZarrQueryError::InvalidProjection(
                        "Projection update has column names".to_string(),
                    ))?
                {
                    if !col_names.contains(&col) {
                        col_names.push(col);
                    }
                }
                self.col_names = Some(col_names);
            }
            _ => {
                return Err(ZarrQueryError::InvalidProjection(
                    "Invalid combination in projection update".to_string(),
                ))
            }
        };

        Ok(())
    }
}

#[cfg(test)]
mod projection_tests {
    use super::*;
    use crate::errors::zarr_errors::ZarrQueryResult;

    #[test]
    fn projection_keep_test() -> ZarrQueryResult<()> {
        let proj = ZarrQueryProjection::keep(vec!["var1".into(), "var2".into()]);
        let mut proj_fields =
            proj.apply_selection(&["var1".into(), "var2".into(), "var3".into()])?;
        proj_fields.sort();
        assert_eq!(vec!["var1", "var2"], proj_fields);

        Ok(())
    }
}
