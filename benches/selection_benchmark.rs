use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration, Throughput, AxisScale,
};
// --- Switch to rust_decimal ---
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
// --- And bring in the necessary traits ---
use num_traits::{One, Zero};
use rand::Rng;
use std::collections::HashSet;

// --- A FenwickTree using rust_decimal ---
// (This is now much faster due to the fast arithmetic)
struct FenwickTree {
    tree: Vec<Decimal>,
    original_weights: Vec<Decimal>,
}
impl FenwickTree {
    fn new(size: usize) -> Self {
        Self { tree: vec![Decimal::zero(); size + 1], original_weights: vec![Decimal::zero(); size] }
    }
    fn add(&mut self, mut index: usize, delta: Decimal) {
        if !delta.is_zero() && self.original_weights[index].is_zero() { self.original_weights[index] = delta; }
        index += 1;
        while index < self.tree.len() {
            self.tree[index] += delta;
            index += index & index.wrapping_neg();
        }
    }
    fn find(&self, target: Decimal) -> usize {
        let mut target = target;
        let mut current_index = 0;
        let mut bit_mask = 1 << (self.tree.len().next_power_of_two().trailing_zeros() - 1);
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
    fn total_weight(&self) -> Decimal { self.original_weights.iter().sum() }
}


// --- THE NEW, CORRECT BENCHMARK ---
fn benchmark_simulation_loop(c: &mut Criterion) {
    let mut group = c.benchmark_group("Simulation Loop (1000 draws)");
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);
    
    let num_draws = 1000;

    for &n in &[100_000, 1_000_000] {
        // Throughput is now the number of draws we are performing.
        group.throughput(Throughput::Elements(num_draws as u64));

        // Setup the initial weights (this part is not timed)
        let mut rng = rand::thread_rng();
        let weights: Vec<Decimal> = (0..n)
            .map(|_| Decimal::from_f64(rng.gen_range(0.0..1.0) / n as f64).unwrap_or_default())
            .collect();

        // --- DigitBinIndex Benchmark ---
        group.bench_with_input(BenchmarkId::new("DigitBinIndex", n), &n, |b, &n| {
            // Setup: Build the index ONCE.
            let mut dbi = your_crate_name::DigitBinIndex::with_precision(5); // Replace with your crate name
            for (i, &weight) in weights.iter().enumerate() { dbi.add(i as u32, weight); }

            // Action: The timed portion is the ENTIRE loop.
            b.iter(|| {
                // We must clone the index to not exhaust it during the benchmark.
                let mut dbi_clone = dbi.clone();
                for _ in 0..num_draws {
                    // Use criterion::black_box to prevent the compiler from optimizing away the loop.
                    criterion::black_box(dbi_clone.select_and_remove());
                }
            });
        });

        // --- FenwickTree Benchmark ---
        group.bench_with_input(BenchmarkId::new("FenwickTree", n), &n, |b, &n| {
            // Setup: Build the tree ONCE.
            let mut ft = FenwickTree::new(n);
            for (i, &weight) in weights.iter().enumerate() { ft.add(i, weight); }
            
            // Action: The timed portion is the ENTIRE loop.
            b.iter(|| {
                let mut ft_clone = ft.clone(); // Clone to not exhaust it.
                let mut total_weight = ft_clone.total_weight();
                for _ in 0..num_draws {
                    if let Some(index_removed) = criterion::black_box(ft_clone.select_and_remove(total_weight)) {
                        total_weight -= ft_clone.original_weights[index_removed];
                    }
                }
            });
        });
    }
    group.finish();
}

// You will need to make your structs cloneable: `#[derive(Debug, Clone)]`
// And update the crate name `your_crate_name` to your actual crate name.
// Also, I've noticed you are using `fraction::Decimal` in the provided benchmark, 
// if you switched to `rust_decimal`, ensure all types are consistent.

criterion_group!(benches, benchmark_simulation_loop);
criterion_main!(benches);
