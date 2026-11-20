//! Criterion harness over the pipeline scene set (`pipeline_scenes`):
//! one workload per rasteriser stage on a 1920 × 1080 canvas.
//!
//! Run with: `cargo bench -p oxideav-raster --bench pipeline`
//! (append a case name to run one).

use criterion::{Criterion, criterion_group, criterion_main};

#[path = "pipeline_scenes/mod.rs"]
mod pipeline_scenes;

fn bench_pipeline(c: &mut Criterion) {
    let mut g = c.benchmark_group("pipeline");
    g.sample_size(10);
    g.measurement_time(std::time::Duration::from_secs(2));
    g.warm_up_time(std::time::Duration::from_millis(300));
    for mut case in pipeline_scenes::cases() {
        g.bench_function(case.name, |b| b.iter(|| (case.run)()));
    }
    g.finish();
}

criterion_group!(benches, bench_pipeline);
criterion_main!(benches);
