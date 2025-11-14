use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use rust_tvtf::funcs::trans::bench_transaction_process_owned;
use std::hint::black_box;
use std::time::Duration;

fn generate_events(
    num_events: usize,
) -> Vec<hashbrown::HashMap<String, smallvec::SmallVec<[String; 4]>>> {
    use chrono::{Duration as ChronoDuration, Utc};
    use hashbrown::HashMap;
    use smallvec::{SmallVec, smallvec};
    const FIELD_TIME: &str = "_time";
    const FIELD_MESSAGE: &str = "_message";
    let base = Utc::now();
    let mut out = Vec::with_capacity(num_events);
    for i in 0..num_events {
        let msg = match i & 3 {
            0 => "end",
            1 => "middle",
            2 => "start",
            _ => "other",
        };
        let ts = (base - ChronoDuration::seconds((i as i64) * 60)).timestamp_micros();
        let mut event: HashMap<String, SmallVec<[String; 4]>> = HashMap::new();
        event.insert(FIELD_TIME.to_string(), smallvec![ts.to_string()]);
        event.insert(FIELD_MESSAGE.to_string(), smallvec![msg.to_string()]);
        event.insert("host".to_string(), smallvec!["host1".to_string()]);
        out.push(event);
    }
    out
}

fn bench_transaction_10k(c: &mut Criterion) {
    let mut group = c.benchmark_group("transaction");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(40);
    group.bench_function("process_10k", |b| {
        b.iter_batched(
            || generate_events(10_000),
            |events| black_box(bench_transaction_process_owned(events)),
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn bench_transaction_100k(c: &mut Criterion) {
    let mut group = c.benchmark_group("transaction");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(30);
    group.bench_function("process_100k", |b| {
        b.iter_batched(
            || generate_events(100_000),
            |events| black_box(bench_transaction_process_owned(events)),
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group!(benches, bench_transaction_10k, bench_transaction_100k);
criterion_main!(benches);
