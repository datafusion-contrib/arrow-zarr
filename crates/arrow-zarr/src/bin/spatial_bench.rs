use std::sync::Arc;

use arrow_zarr::geospatial::{SpatialJoinPhysicalOptimizer, StWithinUdf};
use datafusion::execution::SessionStateBuilder;
use datafusion::logical_expr::ScalarUDF;
use datafusion::prelude::SessionContext;

//fn main() {}

#[tokio::main]
async fn main() {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_physical_optimizer_rule(Arc::new(SpatialJoinPhysicalOptimizer))
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_udf(ScalarUDF::from(StWithinUdf::default()));

    ctx.register_parquet(
        "trip",
        "/home/max/Documents/notebooks/test_data/trip/",
        Default::default(),
    )
    .await
    .unwrap();

    ctx.register_parquet(
        "zone",
        "/home/max/Documents/notebooks/test_data/zone/",
        Default::default(),
    )
    .await
    .unwrap();

    // let df = ctx
    //     .sql(
    //         "
    //         SELECT z.z_zonekey,
    //             z.z_name AS pickup_zone,
    //             AVG(t.t_dropofftime - t.t_pickuptime) AS avg_duration,
    //             AVG(t.t_distance) AS avg_distance, COUNT(t.t_tripkey) AS num_trips
    //         FROM trip t
    //         JOIN zone z
    //             ON ST_Within(t.t_pickuploc, z.z_boundary)
    //         GROUP BY z.z_zonekey, z.z_name
    //         ORDER BY avg_duration DESC NULLS LAST, z.z_zonekey ASC
    //         ",
    //     )
    //     .await
    //     .unwrap();
    // let plan = df.create_physical_plan().await.unwrap();
    // println!(
    //     "{}",
    //     datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
    // );

    let t = std::time::Instant::now();
    let batches = ctx
        .sql(
            "
            SELECT z.z_zonekey,
                   z.z_name AS pickup_zone
                   --AVG(t.t_dropofftime - t.t_pickuptime) AS avg_duration,
                   --AVG(t.t_distance) AS avg_distance, COUNT(t.t_tripkey) AS num_trips
            FROM trip t
            RIGHT JOIN zone z
                ON ST_Within(t.t_pickuploc, z.z_boundary)
            --GROUP BY z.z_zonekey, z.z_name
            --ORDER BY avg_duration DESC NULLS LAST, z.z_zonekey ASC
        ",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    println!("query took {:?}, {} batches", t.elapsed(), batches.len());
    let mut sum = 0;
    for batch in batches {
        sum += batch.num_rows();
    }
    println!("total: {}", sum);
}
