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
use arrow_schema::ArrowError;
use dyn_clone::{clone_trait_object, DynClone};

use crate::reader::ZarrProjection;
use crate::reader::ZarrResult;

/// A predicate operating on [`RecordBatch`]. Here we have the [`DynClone`] trait
/// bound because when dealing with the async zarr reader, it's useful to be able
/// to clone filters.
pub trait ZarrArrowPredicate: Send + DynClone + 'static {
    /// Returns the [`ZarrProjecction`] that describes the columns required
    /// to evaluate this predicate. Those must be present in record batches
    /// that are passed into the [`evaluate`] method.
    fn projection(&self) -> &ZarrProjection;

    /// Evaluate this predicate for the given [`RecordBatch`] containing the columns
    /// identified by [`projection`]. Rows that are `true` in the returned [`BooleanArray`]
    /// satisfy the predicate condition, whereas those that are `false` or do not.
    /// The method should not return any `Null` values. Note that the [`RecordBatch`] is
    /// passed by reference and not consumed by the method.
    fn evaluate(&mut self, batch: &RecordBatch) -> Result<BooleanArray, ArrowError>;
}
clone_trait_object!(ZarrArrowPredicate);

/// A [`ZarrArrowPredicate`] created from an [`FnMut`]. The predicate function has
/// to be [`Clone`] because of the trait bound on [`ZarrArrowPredicate`].
#[derive(Clone)]
pub struct ZarrArrowPredicateFn<F> {
    f: F,
    projection: ZarrProjection,
}

impl<F> ZarrArrowPredicateFn<F>
where
    F: FnMut(&RecordBatch) -> Result<BooleanArray, ArrowError> + Send + 'static,
{
    pub fn new(projection: ZarrProjection, f: F) -> Self {
        Self { f, projection }
    }
}

impl<F> ZarrArrowPredicate for ZarrArrowPredicateFn<F>
where
    F: FnMut(&RecordBatch) -> Result<BooleanArray, ArrowError> + Send + Clone + 'static,
{
    fn projection(&self) -> &ZarrProjection {
        &self.projection
    }

    fn evaluate(&mut self, batch: &RecordBatch) -> Result<BooleanArray, ArrowError> {
        (self.f)(batch)
    }
}

/// A collection of one or more objects that implement [`ZarrArrowPredicate`]. The way
/// filters are used for zarr store is by determining whether or not the a chunk needs to be
/// read based on the predicate. First, only the columns needed in the predicate are read,
/// then the predicate is evaluated, and if there is a least one row that satistifes the
/// condition, the other variables that we requested are read.
///
/// When the predicate is evaluated, the row indices that satisfy the predicate are carried
/// over to when the reuqested data is fetched. However, because zarr data is typically
/// compressed, here even when we have a subset of the rows to read, the whole chunk is read,
/// and the rows mask is applied after.
///
/// Additionally, if the data needed for the predicate is also requested, it is read twice,
/// once for the predicate and once when the actual data is fetched. In the end, if a lot of
/// columns are being requested, and the predicate only needs a few columns, AND if the
/// predicate only evaluates to true for the data in a small fraction of the chunks, then
/// a lot of data reads can be avoided. Under other circumstances though, applying filters
/// that way could be somewhat inefficient, but there's not much else that can be done,
/// considering the typical structure of zarr data (and that it's typically compressed).
#[derive(Clone)]
pub struct ZarrChunkFilter {
    /// A list of [`ZarrArrowPredicate`]
    pub(crate) predicates: Vec<Box<dyn ZarrArrowPredicate>>,
}

impl ZarrChunkFilter {
    /// Create a new [`ZarrChunkFilter`] from an array of [`ZarrArrowPredicate`]
    pub fn new(predicates: Vec<Box<dyn ZarrArrowPredicate>>) -> Self {
        Self { predicates }
    }

    /// Get the combined projections for all the predicates in the filter.
    pub fn get_all_projections(&self) -> ZarrResult<ZarrProjection> {
        let mut proj = ZarrProjection::all();
        for pred in self.predicates.iter() {
            proj.update(pred.projection().clone())?;
        }

        Ok(proj)
    }
}

#[cfg(test)]
mod zarr_predicate_tests {
    use arrow::compute::kernels::cmp::eq;
    use arrow_array::RecordBatch;
    use arrow_array::{ArrayRef, BooleanArray, Float64Array};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    use super::*;
    use crate::reader::ZarrProjection;

    fn generate_rec_batch() -> RecordBatch {
        let fields = vec![
            Arc::new(Field::new("var1".to_string(), DataType::Float64, false)),
            Arc::new(Field::new("var2".to_string(), DataType::Float64, false)),
            Arc::new(Field::new("var3".to_string(), DataType::Float64, false)),
        ];
        let arrs = vec![
            Arc::new(Float64Array::from(vec![38.0, 39.0, 40.0, 41.0, 42.0, 43.0])) as ArrayRef,
            Arc::new(Float64Array::from(vec![39.0, 38.0, 40.0, 41.0, 52.0, 53.0])) as ArrayRef,
            Arc::new(Float64Array::from(vec![38.0, 1.0, 2.0, 3.0, 4.0, 5.0])) as ArrayRef,
        ];

        RecordBatch::try_new(Arc::new(Schema::new(fields)), arrs).unwrap()
    }

    #[test]
    fn apply_predicate_tests() {
        let rec = generate_rec_batch();
        let mut filter = ZarrArrowPredicateFn::new(
            ZarrProjection::keep(vec!["var1".to_string(), "var2".to_string()]),
            move |batch| {
                eq(
                    batch.column_by_name("var1").unwrap(),
                    batch.column_by_name("var2").unwrap(),
                )
            },
        );
        let mask = filter.evaluate(&rec).unwrap();
        assert_eq!(
            mask,
            BooleanArray::from(vec![false, false, true, true, false, false]),
        );

        let rec = generate_rec_batch();
        let mut filter = ZarrArrowPredicateFn::new(
            ZarrProjection::keep(vec!["var1".to_string(), "var3".to_string()]),
            move |batch| {
                eq(
                    batch.column_by_name("var1").unwrap(),
                    batch.column_by_name("var3").unwrap(),
                )
            },
        );
        let mask = filter.evaluate(&rec).unwrap();
        assert_eq!(
            mask,
            BooleanArray::from(vec![true, false, false, false, false, false]),
        );
    }
}
