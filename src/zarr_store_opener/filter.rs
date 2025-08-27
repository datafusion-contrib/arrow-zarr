use arrow_array::{BooleanArray, RecordBatch};
use arrow_schema::ArrowError;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use datafusion::physical_expr::PhysicalExpr;
use itertools::Itertools;
use std::sync::Arc;

/// A predicate operating on [`RecordBatch`].
pub trait ZarrArrowPredicate: Send + 'static {
    /// Evaluate this predicate for the given [`RecordBatch`]. Rows that are `true`
    /// in the returned [`BooleanArray`] satisfy the predicate condition, whereas those
    /// that are `false` do not. The method should not return any `Null` values.
    /// Note that the [`RecordBatch`] is passed by reference and not consumed by
    /// the method.
    fn evaluate(&mut self, batch: &RecordBatch) -> Result<BooleanArray, ArrowError>;
}

/// A [`ZarrArrowPredicate`] created from an [`FnMut`]. The predicate function has
/// to be [`Clone`] because of the trait bound on [`ZarrArrowPredicate`].
#[derive(Clone)]
pub struct ZarrArrowPredicateFn<F> {
    f: F,
}

impl<F> ZarrArrowPredicateFn<F>
where
    F: FnMut(&RecordBatch) -> Result<BooleanArray, ArrowError> + Send + 'static,
{
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

impl<F> ZarrArrowPredicate for ZarrArrowPredicateFn<F>
where
    F: FnMut(&RecordBatch) -> Result<BooleanArray, ArrowError> + Send + Clone + 'static,
{
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
    schema_ref: SchemaRef,
}

impl ZarrChunkFilter {
    /// Create a new [`ZarrChunkFilter`] from an a ['PhysicalExpr']
    pub fn new(_physical_expr: Arc<dyn PhysicalExpr>) -> Self {
        Self {
            predicates: Vec::new(),
            schema_ref: Arc::new(Schema::empty()),
        }
    }

    /// A reference to the schema for the columns needed to evaluate the filter.
    pub fn schema_ref(&self) -> &SchemaRef {
        &self.schema_ref
    }

    pub fn evaluate(&mut self, rec_batch: &RecordBatch) -> Result<bool, ArrowError> {
        let mut bool_arr: Option<BooleanArray> = None;
        for predicate in self.predicates.iter_mut() {
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

#[cfg(test)]
#[allow(unused_imports)]
mod filter_tests {
    use arrow::compute::kernels::cmp::eq;
    use arrow_array::RecordBatch;
    use arrow_array::{ArrayRef, BooleanArray, Float64Array};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    use super::*;

    // generate a record batch to test filters on.
    fn _generate_rec_batch() -> RecordBatch {
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
}
