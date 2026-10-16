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

use std::future::Future;
use std::time::{Duration, Instant};

use datafusion::physical_plan::metrics::{Count, ExecutionPlanMetricsSet, MetricBuilder, Time};

/// A bundle of metric handles pushed down into the zarr reader.
#[derive(Debug, Clone)]
pub struct ZarrMetrics {
    /// Wall-clock time spent reading bytes on the I/O runtime (includes edge-chunk decode).
    io_time: Time,
    /// Wall-clock time spent decoding/decompressing chunk data on the compute runtime.
    decode_time: Time,
    /// Wall-clock time spent producing record batches (around the `next_chunk` body).
    total_time: Time,
    /// Number of chunks examined, including those dropped by the chunk-level filter.
    chunks_looked_at: Count,
    /// Number of chunks materialized into a record batch (passed the filter, if any).
    chunks_read: Count,
    /// Total number of rows produced.
    rows_produced: Count,
}

impl ZarrMetrics {
    /// Build a bundle of handles registered into `metrics` for the given `partition`.
    pub fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            io_time: MetricBuilder::new(metrics).subset_time("io_time", partition),
            decode_time: MetricBuilder::new(metrics).subset_time("decode_time", partition),
            total_time: MetricBuilder::new(metrics).subset_time("total_time", partition),
            chunks_looked_at: MetricBuilder::new(metrics).counter("chunks_looked_at", partition),
            chunks_read: MetricBuilder::new(metrics).counter("chunks_read", partition),
            rows_produced: MetricBuilder::new(metrics).counter("rows_produced", partition),
        }
    }

    // Build a bundle backed by a throwaway metrics set, for streams that
    // aren't driven by an execution plan that reports metrics.
    pub(crate) fn disconnected() -> Self {
        Self::new(&ExecutionPlanMetricsSet::default(), 0)
    }

    pub(crate) fn add_io_time(&self, elapsed: Duration) {
        self.io_time.add_duration(elapsed);
    }

    pub(crate) fn add_decode_time(&self, elapsed: Duration) {
        self.decode_time.add_duration(elapsed);
    }

    // awaits the future, adding the time it took to the i/o time.
    pub(crate) async fn time_io<F: Future>(&self, fut: F) -> F::Output {
        let start = Instant::now();
        let output = fut.await;
        self.add_io_time(start.elapsed());
        output
    }

    // runs the function, adding the time it took to the decode time.
    pub(crate) fn time_decode<R>(&self, f: impl FnOnce() -> R) -> R {
        let start = Instant::now();
        let output = f();
        self.add_decode_time(start.elapsed());
        output
    }

    pub(crate) fn add_total_time(&self, elapsed: Duration) {
        self.total_time.add_duration(elapsed);
    }

    pub(crate) fn inc_chunks_looked_at(&self) {
        self.chunks_looked_at.add(1);
    }

    pub(crate) fn inc_chunks_read(&self) {
        self.chunks_read.add(1);
    }

    pub(crate) fn add_rows(&self, rows: usize) {
        self.rows_produced.add(rows);
    }
}
