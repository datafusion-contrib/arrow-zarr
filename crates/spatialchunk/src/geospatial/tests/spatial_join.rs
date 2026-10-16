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

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::RecordBatch;
use datafusion::assert_batches_sorted_eq;
use datafusion::execution::SessionStateBuilder;
use datafusion::logical_expr::ScalarUDF;
use datafusion::prelude::SessionContext;
use parquet::arrow::ArrowWriter;

use crate::geospatial::{SpatialJoinPhysicalOptimizer, StWithinUdf};

struct TestFixture {
    base_dir: PathBuf,
}

impl TestFixture {
    fn new() -> Self {
        let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src/geospatial/tests/spatial_join_data");
        std::fs::create_dir_all(base_dir.join("build")).unwrap();
        std::fs::create_dir_all(base_dir.join("probe")).unwrap();

        let build = make_build_batch();
        write_parquet(&base_dir.join("build/part0.parquet"), &build.slice(0, 4));
        write_parquet(&base_dir.join("build/part1.parquet"), &build.slice(4, 4));

        for (i, batch) in make_probe_batches().iter().enumerate() {
            write_parquet(&base_dir.join(format!("probe/part{i}.parquet")), batch);
        }

        Self { base_dir }
    }
}

impl Drop for TestFixture {
    fn drop(&mut self) {
        for subdir in ["build", "probe"] {
            let dir = self.base_dir.join(subdir);
            for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
                std::fs::remove_file(entry.path()).ok();
            }
            std::fs::remove_dir(&dir).expect("dir should be empty");
        }
        std::fs::remove_dir(&self.base_dir).expect("base dir should be empty");
    }
}

fn write_parquet(path: &Path, batch: &RecordBatch) {
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
}

fn make_ctx() -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_physical_optimizer_rule(Arc::new(SpatialJoinPhysicalOptimizer))
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_udf(ScalarUDF::from(StWithinUdf::default()));
    ctx
}

async fn make_registered_ctx(fixture: &TestFixture) -> SessionContext {
    let ctx = make_ctx();
    ctx.register_parquet(
        "build",
        fixture.base_dir.join("build").to_str().unwrap(),
        Default::default(),
    )
    .await
    .unwrap();
    ctx.register_parquet(
        "probe",
        fixture.base_dir.join("probe").to_str().unwrap(),
        Default::default(),
    )
    .await
    .unwrap();
    ctx
}

#[tokio::test]
async fn test_spatial_join_from_parquet() {
    let fixture = TestFixture::new();
    let ctx = make_registered_ctx(&fixture).await;

    // Inner join: only matched rows
    let inner_batches = ctx
        .sql(
            "
            SELECT b.col1, p.col4
            FROM build b
            JOIN probe p
                ON st_within(b.geo, p.geo)
            ",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    assert_batches_sorted_eq!(
        [
            "+------+------+",
            "| col1 | col4 |",
            "+------+------+",
            "| 1    | 0.0  |",
            "| 2    | 0.1  |",
            "| 3    | 0.0  |",
            "| 4    | 0.1  |",
            "+------+------+",
        ],
        &inner_batches
    );

    // Left join: matched rows + unmatched build rows (B4, B5, null-geo, empty-geo)
    let left_batches = ctx
        .sql(
            "
            SELECT b.col1, p.col4
            FROM build b
            LEFT JOIN probe p
                ON st_within(b.geo, p.geo)
            ",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    assert_batches_sorted_eq!(
        [
            "+------+------+",
            "| col1 | col4 |",
            "+------+------+",
            "| 1    | 0.0  |",
            "| 2    | 0.1  |",
            "| 3    | 0.0  |",
            "| 4    | 0.1  |",
            "| 5    |      |",
            "| 6    |      |",
            "| 7    |      |",
            "| 8    |      |",
            "+------+------+",
        ],
        &left_batches
    );

    // Inner join with col4 > 0 filter: removes P0→B0 (col4=0.0) and P4→B2 (col4=0.0)
    let inner_filtered_batches = ctx
        .sql(
            "
            SELECT b.col1, p.col4
            FROM build b
            JOIN probe p
                ON st_within(b.geo, p.geo) AND p.col4 > 0
            ",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    assert_batches_sorted_eq!(
        [
            "+------+------+",
            "| col1 | col4 |",
            "+------+------+",
            "| 2    | 0.1  |",
            "| 4    | 0.1  |",
            "+------+------+",
        ],
        &inner_filtered_batches
    );

    // Left join with col4 > 0 filter: B0 and B2 not marked visited → appear as unmatched
    let left_filtered_batches = ctx
        .sql(
            "
            SELECT b.col1, p.col4
            FROM build b
            LEFT JOIN probe p
                ON p.col4 > 0.0 AND st_within(b.geo, p.geo)
            ",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    assert_batches_sorted_eq!(
        [
            "+------+------+",
            "| col1 | col4 |",
            "+------+------+",
            "| 1    |      |",
            "| 2    | 0.1  |",
            "| 3    |      |",
            "| 4    | 0.1  |",
            "| 5    |      |",
            "| 6    |      |",
            "| 7    |      |",
            "| 8    |      |",
            "+------+------+",
        ],
        &left_filtered_batches
    );
}
