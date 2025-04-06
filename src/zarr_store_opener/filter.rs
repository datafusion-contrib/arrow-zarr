use crate::errors::zarr_errors::ZarrQueryResult;
use crate::zarr_store_opener::projection::ZarrQueryProjection;
use arrow_array::{BooleanArray, RecordBatch};
use arrow_schema::ArrowError;

/// A predicate operating on [`RecordBatch`].
pub trait ZarrArrowPredicate: Send + 'static {
    /// Returns the [`ZarrProjecction`] that describes the columns required
    /// to evaluate this predicate. Those must be present in record batches
    /// that are passed into the [`evaluate`] method.
    fn projection(&self) -> &ZarrQueryProjection;

    /// Evaluate this predicate for the given [`RecordBatch`] containing the columns
    /// identified by [`projection`]. Rows that are `true` in the returned [`BooleanArray`]
    /// satisfy the predicate condition, whereas those that are `false` or do not.
    /// The method should not return any `Null` values. Note that the [`RecordBatch`] is
    /// passed by reference and not consumed by the method.
    fn evaluate(&mut self, batch: &RecordBatch) -> Result<BooleanArray, ArrowError>;
}

/// A [`ZarrArrowPredicate`] created from an [`FnMut`]. The predicate function has
/// to be [`Clone`] because of the trait bound on [`ZarrArrowPredicate`].
#[derive(Clone)]
pub struct ZarrArrowPredicateFn<F> {
    f: F,
    projection: ZarrQueryProjection,
}

impl<F> ZarrArrowPredicateFn<F>
where
    F: FnMut(&RecordBatch) -> Result<BooleanArray, ArrowError> + Send + 'static,
{
    pub fn new(projection: ZarrQueryProjection, f: F) -> Self {
        Self { f, projection }
    }
}

impl<F> ZarrArrowPredicate for ZarrArrowPredicateFn<F>
where
    F: FnMut(&RecordBatch) -> Result<BooleanArray, ArrowError> + Send + Clone + 'static,
{
    fn projection(&self) -> &ZarrQueryProjection {
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
    pub fn get_all_projections(&self) -> ZarrQueryResult<ZarrQueryProjection> {
        let mut proj = ZarrQueryProjection::all();
        for pred in self.predicates.iter() {
            proj.update(pred.projection().clone())?;
        }

        Ok(proj)
    }
}

#[cfg(test)]
mod filter_tests {
    use arrow::compute::kernels::cmp::eq;
    use arrow_array::RecordBatch;
    use arrow_array::{ArrayRef, BooleanArray, Float64Array};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    use super::*;
    use crate::zarr_store_opener::projection::{self, ZarrQueryProjection};

    // generate a record batch to test filters on.
    fn generate_rec_batch() -> RecordBatch {
        let fields = vec![
            Arc::new(Field::new("var1".to_string(), DataType::Float64, false)),
            Arc::new(Field::new("var2".to_string(), DataType::Float64, false)),
        ];
        let arrs = vec![
            Arc::new(Float64Array::from(vec![38.0, 39.0, 40.0, 41.0, 42.0, 43.0])) as ArrayRef,
            Arc::new(Float64Array::from(vec![39.0, 38.0, 40.0, 41.0, 52.0, 53.0])) as ArrayRef,
        ];

        RecordBatch::try_new(Arc::new(Schema::new(fields)), arrs).unwrap()
    }

    #[test]
    fn filter_predicate_test() {
        let rec = generate_rec_batch();
        let mut filter1 = ZarrArrowPredicateFn::new(
            ZarrQueryProjection::keep(vec!["var1".to_string(), "var2".to_string()]),
            move |batch| {
                eq(
                    batch.column_by_name("var1").unwrap(),
                    batch.column_by_name("var2").unwrap(),
                )
            },
        );
        let mask = filter1.evaluate(&rec).unwrap();
        assert_eq!(
            mask,
            BooleanArray::from(vec![false, false, true, true, false, false]),
        );

        let mut filter2 = ZarrArrowPredicateFn::new(
            ZarrQueryProjection::keep(vec!["var1".to_string(), "var3".to_string()]),
            move |batch| {
                eq(
                    batch.column_by_name("var1").unwrap(),
                    batch.column_by_name("var3").unwrap(),
                )
            },
        );
    }
}
