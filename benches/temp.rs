#[macro_use]
extern crate criterion;
extern crate ganymede;

use criterion::Criterion;
use ganymede::core::geometry::vector::*;

fn bench_vec_div(c: &mut Criterion) {
    let a = Vec2f::new(3.0, 10.0);
    c.bench_function("vec div", move |b| b.iter(|| a / 2.0));
}

criterion_group!(benches, bench_vec_div);
criterion_main!(benches);
