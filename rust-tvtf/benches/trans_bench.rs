use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hashbrown::HashMap;
use rust_tvtf::funcs::trans::{EventMap, TransParams, TransactionPool};
use rust_tvtf_api::arg::Arg;
use smallvec::smallvec;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

const FIELD_TIME: &str = "_time";
const FIELD_MESSAGE: &str = "_message";
const FIELD_HOST: &str = "host";
const BASE_TS_MICROS: i64 = 1_700_000_000_000_000;

fn arc_str(value: impl Into<String>) -> Arc<str> {
    Arc::<str>::from(value.into().into_boxed_str())
}

fn build_event(timestamp_micros: i64, host: &str, message: &'static str) -> EventMap {
    let mut event: HashMap<String, _> = HashMap::new();
    event.insert(
        FIELD_TIME.to_string(),
        smallvec![arc_str(timestamp_micros.to_string())],
    );
    event.insert(
        FIELD_MESSAGE.to_string(),
        smallvec![Arc::<str>::from(message)],
    );
    event.insert(FIELD_HOST.to_string(), smallvec![arc_str(host)]);
    event
}

fn make_bracket_params(with_max_span: bool) -> TransParams {
    let mut named_arguments = vec![
        ("fields".to_string(), Arg::String(FIELD_HOST.to_string())),
        ("starts_with".to_string(), Arg::String("start".to_string())),
        ("ends_with".to_string(), Arg::String("end".to_string())),
        ("max_events".to_string(), Arg::Int(-1)),
    ];
    if with_max_span {
        named_arguments.push(("max_span".to_string(), Arg::String("7d".to_string())));
    }
    TransParams::new(None, named_arguments).unwrap()
}

fn generate_regular_events(num_events: usize, key_count: usize) -> Vec<EventMap> {
    assert!(key_count > 0);

    let mut events = Vec::with_capacity(num_events);
    for i in 0..num_events {
        let host = format!("host-{}", i % key_count);
        let timestamp_micros = BASE_TS_MICROS - (i as i64 * 1_000_000);
        events.push(build_event(timestamp_micros, &host, "middle"));
    }
    events
}

fn generate_pending_end_events(num_events: usize) -> Vec<EventMap> {
    let mut events = Vec::with_capacity(num_events);
    for i in 0..num_events {
        let host = format!("host-{}", i);
        let timestamp_micros = BASE_TS_MICROS - (i as i64 * 1_000_000);
        events.push(build_event(timestamp_micros, &host, "end"));
    }
    events
}

fn run_full(events: &[EventMap], params: &TransParams) -> usize {
    let mut pool = TransactionPool::new(params.clone());
    for event in events.iter().cloned() {
        pool.add_event(event).unwrap();
    }
    let mut groups = pool.get_frozen_trans();
    groups.extend(pool.get_live_trans());
    groups.len()
}

fn seed_pool(
    params: &TransParams,
    seed_events: &[EventMap],
    probe_event: EventMap,
) -> (TransactionPool, EventMap) {
    let mut pool = TransactionPool::new(params.clone());
    for event in seed_events.iter().cloned() {
        pool.add_event(event).unwrap();
    }
    (pool, probe_event)
}

fn bench_full_run(c: &mut Criterion) {
    let params_no_span = make_bracket_params(false);
    let params_with_span = make_bracket_params(true);
    let same_key_events = generate_regular_events(5_000, 1);
    let wide_key_events = generate_regular_events(5_000, 5_000);

    let mut group = c.benchmark_group("transaction/full_run");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for (label, events) in [
        ("same_key_5k", &same_key_events),
        ("wide_keys_5k", &wide_key_events),
    ] {
        group.throughput(Throughput::Elements(events.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("no_max_span", label),
            events,
            |b, events| {
                b.iter(|| black_box(run_full(events, &params_no_span)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("with_max_span", label),
            events,
            |b, events| {
                b.iter(|| black_box(run_full(events, &params_with_span)));
            },
        );
    }

    group.finish();
}

fn bench_single_add_after_live_buffer(c: &mut Criterion) {
    let params_no_span = make_bracket_params(false);
    let params_with_span = make_bracket_params(true);
    let mut group = c.benchmark_group("transaction/single_add_after_live_buffer");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for pending_count in [500usize, 2_000, 5_000] {
        let seed_events = generate_regular_events(pending_count, pending_count);
        let probe_event = build_event(
            BASE_TS_MICROS - (pending_count as i64 * 1_000_000) - 1_000_000,
            "probe-host",
            "middle",
        );

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("no_max_span", pending_count),
            &pending_count,
            |b, _| {
                b.iter_batched_ref(
                    || seed_pool(&params_no_span, &seed_events, probe_event.clone()),
                    |(pool, probe)| {
                        pool.add_event(probe.clone()).unwrap();
                        black_box(pool);
                    },
                    BatchSize::PerIteration,
                );
            },
        );
        group.bench_with_input(
            BenchmarkId::new("with_max_span", pending_count),
            &pending_count,
            |b, _| {
                b.iter_batched_ref(
                    || seed_pool(&params_with_span, &seed_events, probe_event.clone()),
                    |(pool, probe)| {
                        pool.add_event(probe.clone()).unwrap();
                        black_box(pool);
                    },
                    BatchSize::PerIteration,
                );
            },
        );
    }

    group.finish();
}

fn bench_single_add_after_pending_end(c: &mut Criterion) {
    let params_no_span = make_bracket_params(false);
    let params_with_span = make_bracket_params(true);
    let mut group = c.benchmark_group("transaction/single_add_after_pending_end");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for pending_count in [500usize, 2_000, 5_000] {
        let seed_events = generate_pending_end_events(pending_count);
        let probe_event = build_event(
            BASE_TS_MICROS - (pending_count as i64 * 1_000_000) - 1_000_000,
            "probe-host",
            "middle",
        );

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("no_max_span", pending_count),
            &pending_count,
            |b, _| {
                b.iter_batched_ref(
                    || seed_pool(&params_no_span, &seed_events, probe_event.clone()),
                    |(pool, probe)| {
                        pool.add_event(probe.clone()).unwrap();
                        black_box(pool);
                    },
                    BatchSize::PerIteration,
                );
            },
        );
        group.bench_with_input(
            BenchmarkId::new("with_max_span", pending_count),
            &pending_count,
            |b, _| {
                b.iter_batched_ref(
                    || seed_pool(&params_with_span, &seed_events, probe_event.clone()),
                    |(pool, probe)| {
                        pool.add_event(probe.clone()).unwrap();
                        black_box(pool);
                    },
                    BatchSize::PerIteration,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_full_run,
    bench_single_add_after_live_buffer,
    bench_single_add_after_pending_end
);
criterion_main!(benches);
