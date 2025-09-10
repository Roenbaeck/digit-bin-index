use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration, Throughput, AxisScale,
};
use digit_bin_index::DigitBinIndex; // Assuming your crate is named digit_bin_index
use rand::Rng;
use rust_decimal::Decimal;
use std::collections::HashSet;
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
        // Only store original weight on first add
        if !delta.is_zero() && self.original_weights[index].is_zero() { 
            self.original_weights[index] = delta; 
        }
        index += 1; // 1-based indexing for Fenwick tree
        while index < self.tree.len() {
            self.tree[index] += delta;
            index += index & index.wrapping_neg(); // Add LSB
        }
    }
    
    fn find(&self, target: Decimal) -> usize {
        let mut target = target;
        let mut current_index = 0;
        // Start from the largest power of 2 less than tree size
        let mut bit_mask = 1 << (self.tree.len().next_power_of_two().trailing_zeros().saturating_sub(1));
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
        // This is now an O(N) operation, so we calculate it once per batch.
        self.original_weights.iter().sum() 
    }

    // Wallenius' draw helper
    fn wallenius_select_and_remove(&mut self, current_total: Decimal) -> Option<usize> {
        if current_total.is_zero() { return None; }
        let mut rng = rand::thread_rng();
        let random_target = rng.gen_range(Decimal::ZERO..current_total);
        let index = self.find(random_target);
        if index < self.original_weights.len() { 
            // Remove the weight from the tree
            self.add(index, -self.original_weights[index]); 
        }
        Some(index)
    }

    // Fisher's draw helper
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
            selected_ids.insert(candidate_id);
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

        group.bench_with_input(BenchmarkId::new("DigitBinIndex", n), &n, |b, &n_items| {
            b.iter_batched(|| {
                let mut dbi = DigitBinIndex::with_precision(5);
                let mut rng = rand::thread_rng();
                let smallest = Decimal::new(1, 4);
                let largest = Decimal::new(9999, 4);
                let mut i = 0u32;
                while dbi.count() < n_items as u32 {
                    let weight = rng.gen_range(smallest..largest);
                    if dbi.add(i, weight) {
                        i += 1;
                    }
                }
                dbi
            }, |mut dbi| { 
                for _ in 0..num_draws { 
                    black_box(dbi.select_and_remove()); 
                } 
            }, criterion::BatchSize::SmallInput);
        });

        group.bench_with_input(BenchmarkId::new("FenwickTree", n), &n, |b, &n_items| {
            b.iter_batched(|| {
                let mut ft = FenwickTree::new(n_items);
                let mut rng = rand::thread_rng();
                let smallest = Decimal::new(1, 4);
                let largest = Decimal::new(9999, 4);
                for i in 0..n_items {
                    let weight = rng.gen_range(smallest..largest);
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
        let k = n / 100; // Test drawing 1% of the population
        group.throughput(Throughput::Elements(k as u64));
        let bench_id = format!("N={}, k={}", n, k);

        group.bench_with_input(BenchmarkId::new("DigitBinIndex", &bench_id), &k, |b, &k_items| {
            b.iter_batched(|| {
                let mut dbi = DigitBinIndex::with_precision(5);
                let mut rng = rand::thread_rng();
                let smallest = Decimal::new(1, 4);
                let largest = Decimal::new(9999, 4);
                let mut i = 0u32;
                while dbi.count() < n as u32 {
                    let weight = rng.gen_range(smallest..largest);
                    if dbi.add(i, weight) {
                        i += 1;
                    }
                }
                dbi
            }, |mut dbi| { 
                black_box(dbi.select_many_and_remove(k_items as u32)); 
            }, criterion::BatchSize::SmallInput);
        });

        group.bench_with_input(BenchmarkId::new("FenwickTree", &bench_id), &k, |b, &k_items| {
            b.iter_batched(|| {
                let mut ft = FenwickTree::new(n);
                let mut rng = rand::thread_rng();
                let smallest = Decimal::new(1, 4);
                let largest = Decimal::new(9999, 4);
                for i in 0..n {
                    let weight = rng.gen_range(smallest..largest);
                    ft.add(i, weight); 
                }
                ft
            }, |mut ft| { 
                black_box(ft.fisher_select_many_and_remove(k_items as u32)); 
            }, criterion::BatchSize::SmallInput);
        });
    }
    group.finish();
}

// --- NEW: Pure Operation Benchmarks (Most Fair Comparison) ---
fn benchmark_single_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Single Operation Performance");
    
    for &n in &[100_000, 1_000_000] {
        group.bench_with_input(BenchmarkId::new("DigitBinIndex-SingleSelect", n), &n, |b, &n_items| {
            b.iter_batched(|| {
                let mut dbi = DigitBinIndex::with_precision(5);
                let mut rng = rand::thread_rng();
                let smallest = Decimal::new(1, 4);
                let largest = Decimal::new(9999, 4);
                let mut i = 0u32;
                while dbi.count() < n_items as u32 {
                    let weight = rng.gen_range(smallest..largest);
                    if dbi.add(i, weight) {
                        i += 1;
                    }
                }
                dbi
            }, |mut dbi| { 
                black_box(dbi.select_and_remove()); 
            }, criterion::BatchSize::SmallInput);
        });

        group.bench_with_input(BenchmarkId::new("FenwickTree-SingleSelect", n), &n, |b, &n_items| {
            b.iter_batched(|| {
                // SETUP: Runs before the timer starts
                let mut ft = FenwickTree::new(n_items);
                let mut rng = rand::thread_rng();
                let smallest = Decimal::new(1, 4);
                let largest = Decimal::new(9999, 4);
                for i in 0..n_items {
                    let weight = rng.gen_range(smallest..largest);
                    ft.add(i, weight); 
                }
                // Pre-calculate the expensive O(N) total weight here
                let total_weight = ft.total_weight();
                // Pass both the tree and the pre-calculated weight to the timed section
                (ft, total_weight)
            }, |(mut ft, total_weight)| { 
                // TIMED SECTION: Only this code is measured
                // We use the pre-calculated total_weight, making this a pure O(log N) measurement
                black_box(ft.wallenius_select_and_remove(total_weight)); 
            }, criterion::BatchSize::SmallInput);
        });
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_wallenius_draw, benchmark_fisher_draw, benchmark_single_operations);
criterion_main!(benches);
