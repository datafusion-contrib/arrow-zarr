use std::sync::Arc;

use arrow_array::{BinaryArray, BinaryViewArray, Float64Array, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use datafusion::common::{JoinSide, ScalarValue};
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::expressions::col;
use datafusion::physical_plan::expressions::{binary, lit};
use datafusion::physical_plan::joins::utils::{ColumnIndex, JoinFilter};
use geo::CoordsIter;
use geo_types::{line_string, point, polygon, MultiLineString, MultiPoint, MultiPolygon};
use wkb::writer::{
    write_line_string, write_multi_line_string, write_multi_point, write_multi_polygon,
    write_point, write_polygon, WriteOptions,
};

fn geo_rect<G: CoordsIter<Scalar = f64>>(g: &G) -> (f32, f32, f32, f32) {
    let mut x_min = f64::INFINITY;
    let mut y_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for c in g.coords_iter() {
        x_min = x_min.min(c.x);
        y_min = y_min.min(c.y);
        x_max = x_max.max(c.x);
        y_max = y_max.max(c.y);
    }
    (x_min as f32, y_min as f32, x_max as f32, y_max as f32)
}

#[allow(clippy::type_complexity)]
pub(crate) fn make_wkb_bufs(
    x_shift: f64,
    y_shift: f64,
) -> ([Vec<u8>; 6], [(f32, f32, f32, f32); 6]) {
    let opts = WriteOptions::default();

    let pt = point!(x: 1.0 + x_shift, y: 2.0 + y_shift);
    let mp = MultiPoint::new(vec![
        point!(x: 3.0 + x_shift, y: 4.0 + y_shift),
        point!(x: 5.0 + x_shift, y: 6.0 + y_shift),
    ]);
    let ls = line_string![
        (x: 0.0 + x_shift, y: 0.0 + y_shift),
        (x: 1.0 + x_shift, y: 1.0 + y_shift),
        (x: 2.0 + x_shift, y: 0.0 + y_shift)
    ];
    let mls = MultiLineString::new(vec![
        line_string![(x: 0.0 + x_shift, y: 0.0 + y_shift), (x: 1.0 + x_shift, y: 1.0 + y_shift)],
        line_string![(x: 2.0 + x_shift, y: 2.0 + y_shift), (x: 3.0 + x_shift, y: 3.0 + y_shift)],
    ]);
    let poly = polygon![
        (x: 0.0 + x_shift, y: 0.0 + y_shift),
        (x: 4.0 + x_shift, y: 0.0 + y_shift),
        (x: 4.0 + x_shift, y: 4.0 + y_shift),
        (x: 0.0 + x_shift, y: 4.0 + y_shift),
        (x: 0.0 + x_shift, y: 0.0 + y_shift),
    ];
    let mpoly = MultiPolygon::new(vec![
        polygon![
            (x: 0.0 + x_shift, y: 0.0 + y_shift),
            (x: 1.0 + x_shift, y: 0.0 + y_shift),
            (x: 1.0 + x_shift, y: 1.0 + y_shift),
            (x: 0.0 + x_shift, y: 1.0 + y_shift),
            (x: 0.0 + x_shift, y: 0.0 + y_shift),
        ],
        polygon![
            (x: 5.0 + x_shift, y: 5.0 + y_shift),
            (x: 6.0 + x_shift, y: 5.0 + y_shift),
            (x: 6.0 + x_shift, y: 6.0 + y_shift),
            (x: 5.0 + x_shift, y: 6.0 + y_shift),
            (x: 5.0 + x_shift, y: 5.0 + y_shift),
        ],
    ]);

    let rects = [
        geo_rect(&pt),
        geo_rect(&mp),
        geo_rect(&ls),
        geo_rect(&mls),
        geo_rect(&poly),
        geo_rect(&mpoly),
    ];

    let mut pt_buf = Vec::new();
    write_point(&mut pt_buf, &pt, &opts).unwrap();

    let mut mp_buf = Vec::new();
    write_multi_point(&mut mp_buf, &mp, &opts).unwrap();

    let mut ls_buf = Vec::new();
    write_line_string(&mut ls_buf, &ls, &opts).unwrap();

    let mut mls_buf = Vec::new();
    write_multi_line_string(&mut mls_buf, &mls, &opts).unwrap();

    let mut poly_buf = Vec::new();
    write_polygon(&mut poly_buf, &poly, &opts).unwrap();

    let mut mpoly_buf = Vec::new();
    write_multi_polygon(&mut mpoly_buf, &mpoly, &opts).unwrap();

    (
        [pt_buf, mp_buf, ls_buf, mls_buf, poly_buf, mpoly_buf],
        rects,
    )
}

pub(crate) fn rect_wkb(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<u8> {
    let poly = polygon![
        (x: x0, y: y0), (x: x1, y: y0), (x: x1, y: y1),
        (x: x0, y: y1), (x: x0, y: y0),
    ];
    let mut buf = Vec::new();
    write_polygon(&mut buf, &poly, &WriteOptions::default()).unwrap();
    buf
}

pub(crate) fn empty_polygon_wkb() -> Vec<u8> {
    let poly = polygon![];
    let mut buf = Vec::new();
    write_polygon(&mut buf, &poly, &WriteOptions::default()).unwrap();
    buf
}

pub(crate) fn build_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("geo", DataType::Binary, true),
        Field::new("col1", DataType::Int32, false),
        Field::new("col2", DataType::Float64, false),
    ]))
}

pub(crate) fn make_build_batch() -> RecordBatch {
    let build_rects = [
        (0.0f64, 0.0, 10.0, 10.0),
        (20.0, 0.0, 30.0, 10.0),
        (40.0, 0.0, 50.0, 10.0),
        (0.0, 20.0, 10.0, 30.0),
        (20.0, 20.0, 30.0, 30.0),
        (40.0, 20.0, 50.0, 30.0),
    ];
    let valid_wkbs: Vec<Vec<u8>> = build_rects
        .iter()
        .map(|&(x0, y0, x1, y1)| rect_wkb(x0, y0, x1, y1))
        .collect();
    let empty = empty_polygon_wkb();
    let geo_values: Vec<Option<&[u8]>> = valid_wkbs
        .iter()
        .map(|b| Some(b.as_slice()))
        .chain([None, Some(empty.as_slice())])
        .collect();
    let geo_col = Arc::new(BinaryArray::from(geo_values));
    let col1 = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8]));
    let col2 = Arc::new(Float64Array::from(vec![
        1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8,
    ]));
    RecordBatch::try_new(build_schema(), vec![geo_col, col1, col2]).unwrap()
}

pub(crate) fn probe_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("geo", DataType::BinaryView, false),
        Field::new("col3", DataType::Int32, false),
        Field::new("col4", DataType::Float64, false),
    ]))
}

pub(crate) fn make_probe_batch(geo_rects: &[(f64, f64, f64, f64)], col3_start: i32) -> RecordBatch {
    let n = geo_rects.len();
    let wkbs: Vec<Vec<u8>> = geo_rects
        .iter()
        .map(|&(x0, y0, x1, y1)| rect_wkb(x0, y0, x1, y1))
        .collect();
    let geo_col = Arc::new(BinaryViewArray::from_iter_values(
        wkbs.iter().map(|b| b.as_slice()),
    ));
    let col3 = Arc::new(Int32Array::from(
        (col3_start..col3_start + n as i32).collect::<Vec<_>>(),
    ));
    let col4 = Arc::new(Float64Array::from(
        (0..n).map(|i| i as f64 * 0.1).collect::<Vec<f64>>(),
    ));
    RecordBatch::try_new(probe_schema(), vec![geo_col, col3, col4]).unwrap()
}

pub(crate) fn col4_gt_zero_filter() -> JoinFilter {
    let filter_schema = Arc::new(Schema::new(vec![Field::new(
        "col4",
        DataType::Float64,
        true,
    )]));
    let expr = binary(
        col("col4", &filter_schema).unwrap(),
        Operator::Gt,
        lit(ScalarValue::Float64(Some(0.0))),
        &filter_schema,
    )
    .unwrap();
    JoinFilter::new(
        expr,
        vec![ColumnIndex {
            index: 2,
            side: JoinSide::Right,
        }],
        filter_schema,
    )
}

pub(crate) fn make_probe_batches() -> Vec<RecordBatch> {
    vec![
        // Batch 1: P0 inside B0, P1 inside B1, P2/P3 outside
        make_probe_batch(
            &[
                (1.0, 1.0, 2.0, 2.0),         // inside B0 (0,0,10,10)
                (21.0, 1.0, 22.0, 2.0),       // inside B1 (20,0,30,10)
                (100.0, 100.0, 101.0, 101.0), // outside
                (200.0, 200.0, 201.0, 201.0), // outside
            ],
            100,
        ),
        // Batch 2: P4 inside B2, P5 inside B3, P6/P7 outside
        make_probe_batch(
            &[
                (41.0, 1.0, 42.0, 2.0),       // inside B2 (40,0,50,10)
                (1.0, 21.0, 2.0, 22.0),       // inside B3 (0,20,10,30)
                (300.0, 300.0, 301.0, 301.0), // outside
                (400.0, 400.0, 401.0, 401.0), // outside
            ],
            104,
        ),
    ]
}
