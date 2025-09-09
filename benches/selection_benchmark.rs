use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration, Throughput, AxisScale,
};
use digit_bin_index::DigitBinIndex;
use rand::Rng;
use rust_decimal::Decimal;
use std::collections::HashSet;
use std::convert::TryFrom;
use std::hint::black_box;

// --- Competitor Implementation: Fenwick Tree ---
#[derive(Clone)]
struct FenwickTree {
    tree: Vec<Decimal>,
    original_weights: Vec<Decimal>,
}

impl FenwickTree {
    fn new(size: usize) -> Self {
        Self { 
            tree: vec![Decimal::ZERO; size + 1], 
            original_weights: vec![Decimal::ZERO; size] 
        }
    }

    fn add(&mut self, mut index: usize, delta: Decimal) {
        if !delta.is_zero() && self.original_weights[index].is_zero() { 
            self.original_weights[index] = delta; 
        }
        index += 1;
        while index < self.tree.len() {
            self.tree[index] += delta;
            index += index & index.wrapping_neg();
        }
    }

    fn find(&self, target: Decimal) -> usize {
        let mut target = target;
        let mut current_index = 0;
        let mut bit_mask = 1 << (self.tree.len().next_power_of_two().trailing_zeros() as u32 - 1);
        while bit_mask != 0 {
            let test_index = current_index + bit_mask;
            if test_index < self.tree.len() && target >= self.tree[test_index] {
                target -= self.tree[test_index];
                current_index = test_index;
            }
            bit_mask >>= 1;
        }
        current_index
    }

    fn total_weight(&self) -> Decimal { 
        self.original_weights.iter().sum() 
    }

    // Wallenius' draw helper
    fn wallenius_select_and_remove(&mut self, current_total: Decimal) -> Option<usize> {
        if current_total.is_zero() { return None; }
        let mut rng = rand::thread_rng();
        let random_target = rng.gen_range(Decimal::ZERO..current_total);
        let index = self.find(random_target);
        if index < self.original_weights.len() { 
            self.add(index, -self.original_weights[index]); 
        }
        Some(index)
    }

    fn fisher_select_many_and_remove(&mut self, num_to_draw: u32) -> Option<HashSet<usize>> {
        if num_to_draw as usize > self.original_weights.len() { return None; }
        let total_weight = self.total_weight();
        if total_weight.is_zero() { return Some(HashSet::new()); }
        
        let mut selected_ids = HashSet::with_capacity(num_to_draw as usize);
        let mut rng = rand::thread_rng();

        // Keep sampling until we have exactly k unique items
        while selected_ids.len() < num_to_draw as usize {
            let random_target = rng.gen_range(Decimal::ZERO..total_weight);
            let candidate_id = self.find(random_target);
            selected_ids.insert(candidate_id); // HashSet automatically handles uniqueness
        }

        // Remove all selected items
        for &id in &selected_ids {
            if id < self.original_weights.len() { 
                self.add(id, -self.original_weights[id]); 
            }
        }
        
        Some(selected_ids)
    }
}

// --- Benchmark Suite 1: Wallenius Draw Simulation Loop ---
fn benchmark_wallenius_draw(c: &mut Criterion) {
    let mut group = c.benchmark_group("Wallenius Draw (1000 Selections)");
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);
    let num_draws = 1000;

    for &n in &[100_000, 1_000_000] {
        group.throughput(Throughput::Elements(num_draws as u64));
        let mut rng = rand::thread_rng();
        let weights: Vec<Decimal> = (0..n)
            .map(|_| Decimal::try_from(rng.gen_range(0.001..1.0) / n as f64).unwrap_or_default())
            .collect();

        group.bench_with_input(BenchmarkId::new("DigitBinIndex", n), &n, |b, _| {
            b.iter_batched(|| {
                let mut dbi = DigitBinIndex::with_precision(5);
                for (i, &weight) in weights.iter().enumerate() { 
                    dbi.add(i as u32, weight); 
                }
                dbi
            }, |mut dbi| { 
                for _ in 0..num_draws { 
                    black_box(dbi.select_and_remove()); 
                } 
            }, criterion::BatchSize::SmallInput);
        });

        group.bench_with_input(BenchmarkId::new("FenwickTree", n), &n, |b, _| {
            b.iter_batched(|| {
                let mut ft = FenwickTree::new(n);
                for (i, &weight) in weights.iter().enumerate() { 
                    ft.add(i, weight); 
                }
                ft
            }, |mut ft| {
                let mut total_weight = ft.total_weight();
                for _ in 0..num_draws {
                    if let Some(index_removed) = ft.wallenius_select_and_remove(total_weight) {
                        if index_removed < ft.original_weights.len() { 
                            total_weight -= ft.original_weights[index_removed]; 
                        }
                    } else { 
                        break; 
                    }
                }
            }, criterion::BatchSize::SmallInput);
        });
    }
    group.finish();
}

// --- Benchmark Suite 2: Fisher's Draw (Single Batch) ---
fn benchmark_fisher_draw(c: &mut Criterion) {
    let mut group = c.benchmark_group("Fisher's Draw (Simultaneous Selection)");
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    for &n in &[100_000, 1_000_000] {
        // Test drawing 1% of the population
        let k = n / 100;
        group.throughput(Throughput::Elements(k as u64));
        let mut rng = rand::thread_rng();
        let weights: Vec<Decimal> = (0..n)
            .map(|_| Decimal::try_from(rng.gen_range(0.001..1.0) / n as f64).unwrap_or_default())
            .collect();
        
        let bench_id = format!("N={}, k={}", n, k);

        group.bench_with_input(BenchmarkId::new("DigitBinIndex", &bench_id), &k, |b, &k| {
            b.iter_batched(|| {
                let mut dbi = DigitBinIndex::with_precision(5);
                for (i, &weight) in weights.iter().enumerate() { 
                    dbi.add(i as u32, weight); 
                }
                dbi
            }, |mut dbi| { 
                black_box(dbi.select_many_and_remove(k as u32)); 
            }, criterion::BatchSize::SmallInput);
        });

        group.bench_with_input(BenchmarkId::new("FenwickTree", &bench_id), &k, |b, &k| {
            b.iter_batched(|| {
                let mut ft = FenwickTree::new(n);
                for (i, &weight) in weights.iter().enumerate() { 
                    ft.add(i, weight); 
                }
                ft
            }, |mut ft| { 
                black_box(ft.fisher_select_many_and_remove(k as u32)); 
            }, criterion::BatchSize::SmallInput);
        });
    }
    group.finish();
}

// --- NEW: Pure Operation Benchmarks (Most Fair Comparison) ---
fn benchmark_single_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Single Operation Performance");
    
    for &n in &[100_000, 1_000_000] {
        let mut rng = rand::thread_rng();
        let weights: Vec<Decimal> = (0..n)
            .map(|_| Decimal::try_from(rng.gen_range(0.001..1.0) / n as f64).unwrap_or_default())
            .collect();

        // Single selection operation
        group.bench_with_input(BenchmarkId::new("DigitBinIndex-SingleSelect", n), &n, |b, _| {
            b.iter_batched(|| {
                let mut dbi = DigitBinIndex::with_precision(5);
                for (i, &weight) in weights.iter().enumerate() { 
                    dbi.add(i as u32, weight); 
                }
                dbi
            }, |mut dbi| { 
                black_box(dbi.select_and_remove()); 
            }, criterion::BatchSize::SmallInput);
        });

        group.bench_with_input(BenchmarkId::new("FenwickTree-SingleSelect", n), &n, |b, _| {
            b.iter_batched(|| {
                let mut ft = FenwickTree::new(n);
                for (i, &weight) in weights.iter().enumerate() { 
                    ft.add(i, weight); 
                }
                (ft, weights.iter().sum::<Decimal>())
            }, |(mut ft, total_weight)| { 
                black_box(ft.wallenius_select_and_remove(total_weight)); 
            }, criterion::BatchSize::SmallInput);
        });
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_wallenius_draw, benchmark_fisher_draw, benchmark_single_operations);
criterion_main!(benches);