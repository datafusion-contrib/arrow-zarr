#![cfg(test)]

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::{BinaryArray, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::common::JoinSide;
use datafusion::error::Result;
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::expressions::{BinaryExpr, Column};
use datafusion::physical_plan::joins::utils::{ColumnIndex, JoinFilter};
use datafusion::physical_plan::{PhysicalExpr, RecordBatchStream, SendableRecordBatchStream};
use futures::Stream;
use geo_types::Geometry;
use wkb::writer::{
    write_line_string, write_multi_line_string, write_multi_point, write_multi_polygon,
    write_point, write_polygon, WriteOptions,
};
use wkt::TryFromWkt;

pub(crate) fn wkt_to_wkb(wkt_str: &str) -> Vec<u8> {
    let geom = Geometry::<f64>::try_from_wkt_str(wkt_str).unwrap();
    let opts = WriteOptions::default();
    let mut buf = Vec::new();
    match geom {
        Geometry::Point(g) => write_point(&mut buf, &g, &opts).unwrap(),
        Geometry::LineString(g) => write_line_string(&mut buf, &g, &opts).unwrap(),
        Geometry::Polygon(g) => write_polygon(&mut buf, &g, &opts).unwrap(),
        Geometry::MultiPoint(g) => write_multi_point(&mut buf, &g, &opts).unwrap(),
        Geometry::MultiLineString(g) => write_multi_line_string(&mut buf, &g, &opts).unwrap(),
        Geometry::MultiPolygon(g) => write_multi_polygon(&mut buf, &g, &opts).unwrap(),
        _ => panic!("unsupported geometry: {wkt_str}"),
    }
    buf
}

/// Builds unit squares from bottom-left corners. `None` produces an empty polygon.
pub(crate) fn make_squares(corners: &[Option<(f64, f64)>]) -> Vec<Vec<u8>> {
    corners
        .iter()
        .map(|corner| match corner {
            &Some((x, y)) => wkt_to_wkb(&format!(
                "POLYGON(({x} {y}, {x1} {y}, {x1} {y1}, {x} {y1}, {x} {y}))",
                x1 = x + 1.0,
                y1 = y + 1.0
            )),
            None => wkt_to_wkb("POLYGON EMPTY"),
        })
        .collect()
}

pub(crate) struct MockGeoStream {
    schema: SchemaRef,
    batches: std::vec::IntoIter<RecordBatch>,
}

impl Stream for MockGeoStream {
    type Item = Result<RecordBatch>;
    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(self.get_mut().batches.next().map(Ok))
    }
}

impl RecordBatchStream for MockGeoStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

pub(crate) fn make_geo_stream(
    wkts: &[&str],
    num_batches: usize,
    geo_column_name: &str,
    value_column_name: &str,
) -> SendableRecordBatchStream {
    let schema = Arc::new(Schema::new(vec![
        Field::new(geo_column_name, DataType::Binary, false),
        Field::new(value_column_name, DataType::Int32, false),
    ]));

    let chunk_size = wkts.len().div_ceil(num_batches);
    let mut batches = Vec::new();
    let mut row_offset: i32 = 0;

    for chunk in wkts.chunks(chunk_size) {
        let wkbs: Vec<Vec<u8>> = chunk.iter().map(|wkt| wkt_to_wkb(wkt)).collect();
        let values: Vec<i32> = (row_offset..row_offset + chunk.len() as i32).collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(BinaryArray::from_iter_values(
                    wkbs.iter().map(|b| b.as_slice()),
                )),
                Arc::new(Int32Array::from(values)),
            ],
        )
        .unwrap();
        batches.push(batch);
        row_offset += chunk.len() as i32;
    }

    Box::pin(MockGeoStream {
        schema,
        batches: batches.into_iter(),
    })
}

pub(crate) fn col1_gte_col2_filter() -> JoinFilter {
    let schema = Arc::new(Schema::new(vec![
        Field::new("col_1", DataType::Int32, true),
        Field::new("col_2", DataType::Int32, true),
    ]));
    let expr = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("col_1", 0)) as Arc<dyn PhysicalExpr>,
        Operator::GtEq,
        Arc::new(Column::new("col_2", 1)) as Arc<dyn PhysicalExpr>,
    ));
    JoinFilter::new(
        expr,
        vec![
            ColumnIndex {
                index: 1,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 1,
                side: JoinSide::Right,
            },
        ],
        schema,
    )
}
