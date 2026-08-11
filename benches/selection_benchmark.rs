use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use digit_bin_index::DigitBinIndex;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::hint::black_box;
use wyrand::WyRand;

// --- Core Fenwick Tree (Unchanged) ---
#[derive(Clone)]
struct FenwickTree {
    tree: Vec<f64>,
    capacity: usize,
    total_weight: f64,
}

impl FenwickTree {
    fn new(capacity: usize) -> Self {
        Self {
            tree: vec![0.0; capacity + 1],
            capacity,
            total_weight: 0.0,
        }
    }

    fn modify_weight(&mut self, mut index: usize, delta: f64) {
        self.total_weight += delta;
        index += 1;
        while index <= self.capacity {
            self.tree[index] += delta;
            index += index & index.wrapping_neg();
        }
    }

    fn find(&self, target: f64) -> usize {
        let mut target = target;
        let mut current_index = 0;
        let mut bit_mask = 1
            << (self
                .capacity
                .next_power_of_two()
                .trailing_zeros()
                .saturating_sub(1));
        while bit_mask != 0 {
            let test_index = current_index + bit_mask;
            if test_index <= self.capacity && target >= self.tree[test_index] {
                target -= self.tree[test_index];
                current_index = test_index;
            }
            bit_mask >>= 1;
        }
        current_index
    }
}

// --- User-Facing Wrapper with Corrected Function Signatures ---
pub struct WeightedSelector {
    ft: FenwickTree,
    id_to_index: HashMap<u32, usize>,
    index_to_id: Vec<u32>,
    index_to_weight: Vec<f64>,
    free_indices: Vec<usize>,
}

impl WeightedSelector {
    pub fn new(capacity: usize) -> Self {
        Self {
            ft: FenwickTree::new(capacity),
            id_to_index: HashMap::with_capacity(capacity),
            index_to_id: vec![0; capacity],
            index_to_weight: vec![0.0; capacity],
            free_indices: (0..capacity).rev().collect(),
        }
    }

    pub fn add(&mut self, id: u32, weight: f64) -> Result<(), &'static str> {
        if self.id_to_index.contains_key(&id) {
            return Err("ID already exists");
        }
        if let Some(index) = self.free_indices.pop() {
            self.id_to_index.insert(id, index);
            self.index_to_id[index] = id;
            self.index_to_weight[index] = weight;
            self.ft.modify_weight(index, weight);
            Ok(())
        } else {
            Err("At maximum capacity")
        }
    }

    // CHANGED: Signature now matches DigitBinIndex
    pub fn select_and_remove<R: Rng>(&mut self, rng: &mut R) -> Option<(u32, f64)> {
        if self.ft.total_weight == 0.0 {
            return None;
        }
        let target = rng.random_range(0.0..self.ft.total_weight);
        let index = self.ft.find(target);
        let id = self.index_to_id[index];
        let weight = self.index_to_weight[index];

        self.id_to_index.remove(&id);
        self.ft.modify_weight(index, -weight);
        self.index_to_weight[index] = 0.0;
        self.free_indices.push(index);

        // CHANGED: Return tuple with weight
        Some((id, weight))
    }

    // --- Comparison baseline: systematic PPS sampling (not Fisher's law) ---
    pub fn select_many_and_remove<R: Rng>(
        &mut self,
        num_to_draw: u32,
        rng: &mut R,
    ) -> Option<Vec<(u32, f64)>> {
        let num_to_draw = num_to_draw as usize;
        let current_pop_size = self.id_to_index.len();
        if num_to_draw > current_pop_size {
            return None;
        }
        if self.ft.total_weight == 0.0 || num_to_draw == 0 {
            return Some(Vec::new());
        }

        // Collect active items and shuffle
        let mut active: Vec<(u32, f64, f64, usize)> = self
            .id_to_index
            .iter()
            .map(|(&id, &index)| {
                let w = self.index_to_weight[index];
                (id, w, w, index)
            })
            .collect();
        active.shuffle(rng);

        // Compute step and start
        let total: f64 = active.iter().map(|&(_, _, w_f, _)| w_f).sum();
        let step = total / num_to_draw as f64;
        let start = rng.random_range(0.0..step);

        // Select using systematic sampling
        let mut cum: f64 = 0.0;
        let mut current = start;
        let mut result_vec: Vec<(u32, f64)> = Vec::with_capacity(num_to_draw);
        let mut pos = 0;
        for _ in 0..num_to_draw {
            while pos < active.len() && cum < current {
                cum += active[pos].2;
                pos += 1;
            }
            if pos > 0 && pos <= active.len() {
                let (id, w, _, _) = active[pos - 1];
                result_vec.push((id, w));
            }
            current += step;
        }

        // Perform removals
        for (id, weight) in result_vec.iter() {
            if let Some(index) = self.id_to_index.remove(id) {
                self.ft.modify_weight(index, -*weight);
                self.index_to_weight[index] = 0.0;
                self.free_indices.push(index);
            }
        }

        Some(result_vec)
    }
}

// --- Common benchmark parameters ---
const INITIAL_POP: u64 = 1_000_000;
const CHURN_COUNT: u64 = 100_000;
const ACQUISITION_COUNT: u64 = 110_000;
const MAX_CAPACITY: u64 = INITIAL_POP + ACQUISITION_COUNT;

const VERY_LARGE_POP: u64 = 10_000_000;
const VERY_LARGE_CHURN: u64 = 1_000_000;
const VERY_LARGE_ACQ: u64 = 1_000_000;
const VERY_LARGE_MAX: u64 = VERY_LARGE_POP + VERY_LARGE_ACQ;

// --- Benchmarks (No changes needed here, the logic is identical) ---

fn benchmark_wallenius_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Wallenius Simulation (Iterative Churn)");
    group.throughput(Throughput::Elements(CHURN_COUNT + ACQUISITION_COUNT));

    group.bench_function(
        BenchmarkId::new("DigitBinIndex (precision 3)", INITIAL_POP),
        |b| {
            b.iter_batched(
                || {
                    let mut dbi = DigitBinIndex::with_precision_and_capacity(3, MAX_CAPACITY);
                    let mut rng = WyRand::from_os_rng();
                    for i in 0..INITIAL_POP {
                        dbi.add(i, rng.random_range(0.001..0.999)).unwrap();
                    }
                    (dbi, INITIAL_POP, rng)
                },
                |(mut dbi, mut next_id, mut rng)| {
                    for _ in 0..CHURN_COUNT {
                        black_box(dbi.select_and_remove_with_rng(&mut rng));
                    }
                    for _ in 0..ACQUISITION_COUNT {
                        dbi.add(next_id, rng.random_range(0.001..0.999)).unwrap();
                        next_id += 1;
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.bench_function(
        BenchmarkId::new("WeightedSelector (precision 3)", INITIAL_POP),
        |b| {
            b.iter_batched(
                || {
                    let mut selector = WeightedSelector::new(MAX_CAPACITY as usize);
                    let mut rng = WyRand::from_os_rng();
                    for i in 0..INITIAL_POP {
                        selector
                            .add(i as u32, rng.random_range(0.001..0.999))
                            .unwrap();
                    }
                    (selector, INITIAL_POP as u32, rng)
                },
                |(mut selector, mut next_id, mut rng)| {
                    for _ in 0..CHURN_COUNT {
                        black_box(selector.select_and_remove(&mut rng));
                    }
                    for _ in 0..ACQUISITION_COUNT {
                        selector
                            .add(next_id, rng.random_range(0.001..0.999))
                            .unwrap();
                        next_id += 1;
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.bench_function(
        BenchmarkId::new("DigitBinIndex (precision 5)", INITIAL_POP),
        |b| {
            b.iter_batched(
                || {
                    let mut dbi = DigitBinIndex::with_precision_and_capacity(5, MAX_CAPACITY);
                    let mut rng = WyRand::from_os_rng();
                    for i in 0..INITIAL_POP {
                        dbi.add(i, rng.random_range(0.00001..0.99999)).unwrap();
                    }
                    (dbi, INITIAL_POP, rng)
                },
                |(mut dbi, mut next_id, mut rng)| {
                    for _ in 0..CHURN_COUNT {
                        black_box(dbi.select_and_remove_with_rng(&mut rng));
                    }
                    for _ in 0..ACQUISITION_COUNT {
                        dbi.add(next_id, rng.random_range(0.00001..0.99999))
                            .unwrap();
                        next_id += 1;
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.bench_function(
        BenchmarkId::new("WeightedSelector (precision 5)", INITIAL_POP),
        |b| {
            b.iter_batched(
                || {
                    let mut selector = WeightedSelector::new(MAX_CAPACITY as usize);
                    let mut rng = WyRand::from_os_rng();
                    for i in 0..INITIAL_POP {
                        selector
                            .add(i as u32, rng.random_range(0.00001..0.99999))
                            .unwrap();
                    }
                    (selector, INITIAL_POP as u32, rng)
                },
                |(mut selector, mut next_id, mut rng)| {
                    for _ in 0..CHURN_COUNT {
                        black_box(selector.select_and_remove(&mut rng));
                    }
                    for _ in 0..ACQUISITION_COUNT {
                        selector
                            .add(next_id, rng.random_range(0.00001..0.99999))
                            .unwrap();
                        next_id += 1;
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.sample_size(10);

    group.bench_function(
        BenchmarkId::new("DigitBinIndex (very large, Roaring)", VERY_LARGE_POP),
        |b| {
            b.iter_batched(
                || {
                    let mut dbi = DigitBinIndex::with_precision_and_capacity(3, VERY_LARGE_MAX);
                    let mut rng = WyRand::from_os_rng();
                    for i in 0..VERY_LARGE_POP {
                        dbi.add(i, rng.random_range(0.001..0.999)).unwrap();
                    }
                    (dbi, VERY_LARGE_POP, rng)
                },
                |(mut dbi, mut next_id, mut rng)| {
                    for _ in 0..VERY_LARGE_CHURN {
                        black_box(dbi.select_and_remove_with_rng(&mut rng));
                    }
                    for _ in 0..VERY_LARGE_ACQ {
                        dbi.add(next_id, rng.random_range(0.001..0.999)).unwrap();
                        next_id += 1;
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.bench_function(
        BenchmarkId::new("WeightedSelector (very large, FenwickTree)", VERY_LARGE_POP),
        |b| {
            b.iter_batched(
                || {
                    let mut selector = WeightedSelector::new(VERY_LARGE_MAX as usize);
                    let mut rng = WyRand::from_os_rng();
                    for i in 0..VERY_LARGE_POP {
                        selector
                            .add(i as u32, rng.random_range(0.001..0.999))
                            .unwrap();
                    }
                    (selector, VERY_LARGE_POP as u32, rng)
                },
                |(mut selector, mut next_id, mut rng)| {
                    for _ in 0..VERY_LARGE_CHURN {
                        black_box(selector.select_and_remove(&mut rng));
                    }
                    for _ in 0..VERY_LARGE_ACQ {
                        selector
                            .add(next_id, rng.random_range(0.001..0.999))
                            .unwrap();
                        next_id += 1;
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.finish();
}

fn benchmark_skewed_wallenius_simulation(c: &mut Criterion) {
    const HIGH_WEIGHT_ITEMS: u64 = INITIAL_POP / 1_000;
    const LOW_WEIGHT_ITEMS: u64 = INITIAL_POP - HIGH_WEIGHT_ITEMS;

    let mut group = c.benchmark_group("Wallenius Simulation (Skewed Low-Digit Mass)");
    group.sample_size(10);
    group.throughput(Throughput::Elements(CHURN_COUNT));

    group.bench_function(
        BenchmarkId::new("DigitBinIndex (mass ordered)", INITIAL_POP),
        |b| {
            b.iter_batched(
                || {
                    let mut index = DigitBinIndex::with_precision_and_capacity(3, INITIAL_POP);
                    for id in 0..LOW_WEIGHT_ITEMS {
                        index.add(id, 0.099).unwrap();
                    }
                    for id in LOW_WEIGHT_ITEMS..INITIAL_POP {
                        index.add(id, 0.999).unwrap();
                    }
                    (index, WyRand::from_os_rng())
                },
                |(mut index, mut rng)| {
                    for _ in 0..CHURN_COUNT {
                        black_box(index.select_and_remove_with_rng(&mut rng));
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.bench_function(
        BenchmarkId::new("WeightedSelector (FenwickTree)", INITIAL_POP),
        |b| {
            b.iter_batched(
                || {
                    let mut selector = WeightedSelector::new(INITIAL_POP as usize);
                    for id in 0..LOW_WEIGHT_ITEMS {
                        selector.add(id as u32, 0.099).unwrap();
                    }
                    for id in LOW_WEIGHT_ITEMS..INITIAL_POP {
                        selector.add(id as u32, 0.999).unwrap();
                    }
                    (selector, WyRand::from_os_rng())
                },
                |(mut selector, mut rng)| {
                    for _ in 0..CHURN_COUNT {
                        black_box(selector.select_and_remove(&mut rng));
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.finish();
}

fn benchmark_fisher_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Exact Fisher Simulation (Batch Churn)");
    group.throughput(Throughput::Elements(CHURN_COUNT + ACQUISITION_COUNT));

    group.bench_function(
        BenchmarkId::new("DigitBinIndex (precision 3)", INITIAL_POP),
        |b| {
            b.iter_batched(
                || {
                    let mut dbi = DigitBinIndex::with_precision_and_capacity(3, MAX_CAPACITY);
                    let mut rng = WyRand::from_os_rng();
                    for i in 0..INITIAL_POP {
                        dbi.add(i, rng.random_range(0.001..0.999)).unwrap();
                    }
                    (dbi, INITIAL_POP, rng)
                },
                |(mut dbi, mut next_id, mut rng)| {
                    black_box(dbi.select_many_and_remove(CHURN_COUNT));
                    for _ in 0..ACQUISITION_COUNT {
                        dbi.add(next_id, rng.random_range(0.001..0.999)).unwrap();
                        next_id += 1;
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.bench_function(
        BenchmarkId::new(
            "Systematic PPS comparison (not Fisher, precision 3)",
            INITIAL_POP,
        ),
        |b| {
            b.iter_batched(
                || {
                    let mut selector = WeightedSelector::new(MAX_CAPACITY as usize);
                    let mut rng = WyRand::from_os_rng();
                    for i in 0..INITIAL_POP {
                        selector
                            .add(i as u32, rng.random_range(0.001..0.999))
                            .unwrap();
                    }
                    (selector, INITIAL_POP as u32, rng)
                },
                |(mut selector, mut next_id, mut rng)| {
                    black_box(selector.select_many_and_remove(CHURN_COUNT as u32, &mut rng));
                    for _ in 0..ACQUISITION_COUNT {
                        selector
                            .add(next_id, rng.random_range(0.001..0.999))
                            .unwrap();
                        next_id += 1;
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.bench_function(
        BenchmarkId::new("DigitBinIndex (precision 5)", INITIAL_POP),
        |b| {
            b.iter_batched(
                || {
                    let mut dbi = DigitBinIndex::with_precision_and_capacity(5, MAX_CAPACITY);
                    let mut rng = WyRand::from_os_rng();
                    for i in 0..INITIAL_POP {
                        dbi.add(i, rng.random_range(0.00001..0.99999)).unwrap();
                    }
                    (dbi, INITIAL_POP, rng)
                },
                |(mut dbi, mut next_id, mut rng)| {
                    black_box(dbi.select_many_and_remove(CHURN_COUNT));
                    for _ in 0..ACQUISITION_COUNT {
                        dbi.add(next_id, rng.random_range(0.00001..0.99999))
                            .unwrap();
                        next_id += 1;
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.bench_function(
        BenchmarkId::new(
            "Systematic PPS comparison (not Fisher, precision 5)",
            INITIAL_POP,
        ),
        |b| {
            b.iter_batched(
                || {
                    let mut selector = WeightedSelector::new(MAX_CAPACITY as usize);
                    let mut rng = WyRand::from_os_rng();
                    for i in 0..INITIAL_POP {
                        selector
                            .add(i as u32, rng.random_range(0.00001..0.99999))
                            .unwrap();
                    }
                    (selector, INITIAL_POP as u32, rng)
                },
                |(mut selector, mut next_id, mut rng)| {
                    black_box(selector.select_many_and_remove(CHURN_COUNT as u32, &mut rng));
                    for _ in 0..ACQUISITION_COUNT {
                        selector
                            .add(next_id, rng.random_range(0.00001..0.99999))
                            .unwrap();
                        next_id += 1;
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.sample_size(10);

    group.bench_function(
        BenchmarkId::new("DigitBinIndex (very large, Roaring)", VERY_LARGE_POP),
        |b| {
            b.iter_batched(
                || {
                    let mut dbi = DigitBinIndex::with_precision_and_capacity(3, VERY_LARGE_MAX);
                    let mut rng = WyRand::from_os_rng();
                    for i in 0..VERY_LARGE_POP {
                        dbi.add(i, rng.random_range(0.001..0.999)).unwrap();
                    }
                    (dbi, VERY_LARGE_POP, rng)
                },
                |(mut dbi, mut next_id, mut rng)| {
                    black_box(dbi.select_many_and_remove(VERY_LARGE_CHURN));
                    for _ in 0..VERY_LARGE_ACQ {
                        dbi.add(next_id, rng.random_range(0.001..0.999)).unwrap();
                        next_id += 1;
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.bench_function(
        BenchmarkId::new(
            "Systematic PPS comparison (not Fisher, very large FenwickTree)",
            VERY_LARGE_POP,
        ),
        |b| {
            b.iter_batched(
                || {
                    let mut selector = WeightedSelector::new(VERY_LARGE_MAX as usize);
                    let mut rng = WyRand::from_os_rng();
                    for i in 0..VERY_LARGE_POP {
                        selector
                            .add(i as u32, rng.random_range(0.001..0.999))
                            .unwrap();
                    }
                    (selector, VERY_LARGE_POP as u32, rng)
                },
                |(mut selector, mut next_id, mut rng)| {
                    black_box(selector.select_many_and_remove(VERY_LARGE_CHURN as u32, &mut rng));
                    for _ in 0..VERY_LARGE_ACQ {
                        selector
                            .add(next_id, rng.random_range(0.001..0.999))
                            .unwrap();
                        next_id += 1;
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.finish();
}

fn benchmark_pps_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Fixed-Size PPS Simulation (Batch Churn)");
    group.throughput(Throughput::Elements(CHURN_COUNT + ACQUISITION_COUNT));

    for precision in [3, 5] {
        group.bench_function(
            BenchmarkId::new(
                format!("DigitBinIndex (precision {precision})"),
                INITIAL_POP,
            ),
            |b| {
                b.iter_batched(
                    || {
                        let mut index =
                            DigitBinIndex::with_precision_and_capacity(precision, MAX_CAPACITY);
                        let mut rng = WyRand::from_os_rng();
                        let minimum_weight = 10f64.powi(-(precision as i32));
                        for id in 0..INITIAL_POP {
                            index
                                .add(id, rng.random_range(minimum_weight..(1.0 - minimum_weight)))
                                .unwrap();
                        }
                        (index, INITIAL_POP, rng)
                    },
                    |(mut index, mut next_id, mut rng)| {
                        black_box(index.select_pps_many_and_remove(CHURN_COUNT));
                        let minimum_weight = 10f64.powi(-(precision as i32));
                        for _ in 0..ACQUISITION_COUNT {
                            index
                                .add(
                                    next_id,
                                    rng.random_range(minimum_weight..(1.0 - minimum_weight)),
                                )
                                .unwrap();
                            next_id += 1;
                        }
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn benchmark_bernoulli_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Independent Bernoulli Simulation");
    group.throughput(Throughput::Elements(INITIAL_POP));

    for precision in [3, 5] {
        let scale = 10u64.pow(precision as u32);
        let maximum_scaled_probability = scale / 20;

        group.bench_function(
            BenchmarkId::new(
                format!("DigitBinIndex (precision {precision})"),
                INITIAL_POP,
            ),
            |b| {
                b.iter_batched(
                    || {
                        let mut index =
                            DigitBinIndex::with_precision_and_capacity(precision, INITIAL_POP);
                        let mut rng = WyRand::from_os_rng();
                        for id in 0..INITIAL_POP {
                            let scaled_probability =
                                rng.random_range(1..=maximum_scaled_probability);
                            index
                                .add(id, scaled_probability as f64 / scale as f64)
                                .unwrap();
                        }
                        (index, rng)
                    },
                    |(mut index, mut rng)| {
                        black_box(index.select_bernoulli_many_and_remove_with_rng(&mut rng))
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        group.bench_function(
            BenchmarkId::new(format!("Item scan (precision {precision})"), INITIAL_POP),
            |b| {
                b.iter_batched(
                    || {
                        let mut rng = WyRand::from_os_rng();
                        let items: Vec<_> = (0..INITIAL_POP)
                            .map(|id| {
                                let scaled_probability =
                                    rng.random_range(1..=maximum_scaled_probability);
                                (id, scaled_probability as f64 / scale as f64)
                            })
                            .collect();
                        (items, rng)
                    },
                    |(mut items, mut rng)| {
                        let mut selected = Vec::new();
                        items.retain(|&(id, probability)| {
                            if rng.random::<f64>() < probability {
                                selected.push((id, probability));
                                false
                            } else {
                                true
                            }
                        });
                        black_box(selected)
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn insertion_benchmark(c: &mut Criterion) {
    let mut rng = WyRand::from_os_rng();

    // Shared data generator
    let mut generate_items = |n: usize| -> Vec<(u64, f64)> {
        (0..n)
            .map(|i| ((i as u64).wrapping_add(1), rng.random::<f64>()))
            .collect()
    };

    // 1M items test
    let items_1m: Vec<_> = generate_items(1_000_000);

    // Test single add (looped)
    c.bench_function(
        "Insertion (Single Add Loop)/DigitBinIndex (p=3)/1000000",
        |b| {
            b.iter(|| {
                let mut index = DigitBinIndex::with_precision(3);
                for &(id, weight) in &items_1m {
                    index.add(id, weight).unwrap();
                }
                black_box(&index);
            })
        },
    );

    c.bench_function(
        "Insertion (Single Add Loop)/DigitBinIndex (p=5)/1000000",
        |b| {
            b.iter(|| {
                let mut index = DigitBinIndex::with_precision(5);
                for &(id, weight) in &items_1m {
                    index.add(id, weight).unwrap();
                }
                black_box(&index);
            })
        },
    );

    // Test add_many with batch sizes
    c.bench_function(
        "Insertion (Add Many At Once)/DigitBinIndex (p=3)/1000000",
        |b| {
            b.iter(|| {
                let mut index = DigitBinIndex::with_precision(3);
                index.add_many(&items_1m).unwrap();
                black_box(&index);
            })
        },
    );

    c.bench_function(
        "Insertion (Add Many At Once)/DigitBinIndex (p=5)/1000000",
        |b| {
            b.iter(|| {
                let mut index = DigitBinIndex::with_precision(5);
                index.add_many(&items_1m).unwrap();
                black_box(&index);
            })
        },
    );

    // 10M items test (use capacity hint for Roaring)
    let items_10m: Vec<_> = generate_items(10_000_000);
    c.bench_function(
        "Insertion (Single Add Loop)/DigitBinIndex (very large, Roaring)/10000000",
        |b| {
            b.iter(|| {
                let mut index = DigitBinIndex::with_precision_and_capacity(3, 10_000_000);
                for &(id, weight) in &items_10m {
                    index.add(id, weight).unwrap();
                }
                black_box(&index);
            })
        },
    );

    c.bench_function(
        "Insertion (Add Many At Once)/DigitBinIndex (very large, Roaring)/10000000",
        |b| {
            b.iter(|| {
                let mut index = DigitBinIndex::with_precision_and_capacity(3, 10_000_000);
                index.add_many(&items_10m).unwrap();
                black_box(&index);
            })
        },
    );
}

fn skewed_backend_benchmark(c: &mut Criterion) {
    const ITEMS: u64 = 100_000;
    const DRAWS: u64 = 10_000;
    let items: Vec<_> = (0..ITEMS).map(|id| (id, 0.5)).collect();
    let mut group = c.benchmark_group("Skewed Single-Bin Backend");

    group.bench_function("Small Vec/add_many/100000", |b| {
        b.iter(|| {
            let mut index = DigitBinIndex::small(3);
            index.add_many(&items).unwrap();
            black_box(index);
        });
    });
    group.bench_function("Medium Roaring/add_many/100000", |b| {
        b.iter(|| {
            let mut index = DigitBinIndex::medium(3);
            index.add_many(&items).unwrap();
            black_box(index);
        });
    });
    group.bench_function("Small Vec/Wallenius remove/10000", |b| {
        b.iter_batched(
            || {
                let mut index = DigitBinIndex::small(3);
                index.add_many(&items).unwrap();
                index
            },
            |mut index| black_box(index.select_wallenius_many_and_remove(DRAWS)),
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("Medium Roaring/Wallenius remove/10000", |b| {
        b.iter_batched(
            || {
                let mut index = DigitBinIndex::medium(3);
                index.add_many(&items).unwrap();
                index
            },
            |mut index| black_box(index.select_wallenius_many_and_remove(DRAWS)),
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(
    benches,
    benchmark_wallenius_simulation,
    benchmark_skewed_wallenius_simulation,
    benchmark_fisher_simulation,
    benchmark_pps_simulation,
    benchmark_bernoulli_simulation,
    insertion_benchmark,
    skewed_backend_benchmark
);
criterion_main!(benches);
