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

use arrow_array::{BooleanArray, RecordBatch};
use arrow_schema::{ArrowError, SchemaRef};
use itertools::Itertools;

/// A predicate operating on [`RecordBatch`].
pub trait ZarrArrowPredicate: Send + 'static {
    /// Evaluate this predicate for the given [`RecordBatch`]. Rows that are `true`
    /// in the returned [`BooleanArray`] satisfy the predicate condition, whereas those
    /// that are `false` do not. The method should not return any `Null` values.
    /// Note that the [`RecordBatch`] is passed by reference and not consumed by
    /// the method.
    fn evaluate(&self, batch: &RecordBatch) -> Result<BooleanArray, ArrowError>;
}

/// A collection of one or more objects that implement [`ZarrArrowPredicate`]. The way
/// filters are used for zarr store is by determining whether or not the a chunk needs to be
/// read based on the predicate. First, only the columns needed in the predicate are read,
/// then the predicate is evaluated, and if there is a least one row that satistifes the
/// condition, the other variables that we requested are read.
pub struct ZarrChunkFilter {
    /// A list of [`ZarrArrowPredicate`]
    predicates: Vec<Box<dyn ZarrArrowPredicate>>,
    schema_ref: SchemaRef,
}

impl ZarrChunkFilter {
    pub fn new(
        predicates: Vec<Box<dyn ZarrArrowPredicate>>,
        schema_ref: SchemaRef,
    ) -> Result<Self, ArrowError> {
        Ok(Self {
            predicates,
            schema_ref,
        })
    }

    pub fn schema_ref(&self) -> &SchemaRef {
        &self.schema_ref
    }

    /// Applies all the filters in the chunk filter and returns true only
    /// if all the filters return true for at least one row in the record
    /// batch.
    pub fn evaluate(&self, rec_batch: &RecordBatch) -> Result<bool, ArrowError> {
        let mut bool_arr: Option<BooleanArray> = None;
        for predicate in self.predicates.iter() {
            let mask = predicate.evaluate(rec_batch)?;
            if let Some(old_bool_arr) = bool_arr {
                bool_arr = Some(BooleanArray::from(
                    old_bool_arr
                        .iter()
                        .zip(mask.iter())
                        .map(|(x, y)| x.unwrap() && y.unwrap())
                        .collect_vec(),
                ));
            } else {
                bool_arr = Some(mask);
            }
        }

        if let Some(bool_arr) = bool_arr {
            Ok(bool_arr.true_count() > 0)
        } else {
            Ok(true)
        }
    }
}
