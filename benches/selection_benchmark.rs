use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration, Throughput, AxisScale
};
use digit_bin_index::DigitBinIndex;
use fraction::{Decimal, Zero};
use rand::Rng;
use std::collections::HashSet; // Import HashSet for the new method

// --- Competitor Implementation: Fenwick Tree (Binary Indexed Tree) ---
struct FenwickTree {
    tree: Vec<Decimal>,
    // We need to store original weights for the Fisher's draw benchmark
    original_weights: Vec<Decimal>,
}

impl FenwickTree {
    fn new(size: usize) -> Self {
        Self {
            tree: vec![Decimal::from(0); size + 1],
            original_weights: vec![Decimal::from(0); size],
        }
    }

    fn add(&mut self, mut index: usize, delta: Decimal) {
        // Store original weight for Fisher's draw if adding for the first time
        if !delta.is_zero() && self.original_weights[index].is_zero() {
            self.original_weights[index] = delta;
        }
        index += 1; // 1-based indexing
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

    fn total_weight(&self) -> Decimal {
        // The total weight is the sum of all elements, which is the prefix sum up to the end.
        // For a Fenwick tree, this is not a simple O(1) operation. We calculate it once.
        self.original_weights.iter().sum()
    }

    // Wallenius' draw: select and immediately remove
    fn select_and_remove(&mut self, current_total: Decimal) -> Option<usize> {
        if current_total.is_zero() { return None; }
        let mut rng = rand::rng();
        let random_target = Decimal::from(rng.random_range(0.0..current_total.try_into().unwrap()));
        let index = self.find(random_target);
        self.add(index, -self.original_weights[index]); // Remove the weight
        Some(index)
    }

    // Fisher's draw: pure rejection sampling
    fn select_many_and_remove(&mut self, num_to_draw: usize) -> Option<HashSet<usize>> {
        if num_to_draw > self.original_weights.len() { return None; }
        let total_weight = self.total_weight();
        if total_weight.is_zero() { return Some(HashSet::new()); }

        let mut selected_ids = HashSet::with_capacity(num_to_draw);
        let mut rng = rand::rng();

        while selected_ids.len() < num_to_draw {
            let random_target = Decimal::from(rng.random_range(0.0..total_weight.try_into().unwrap()));
            let candidate_id = self.find(random_target);
            selected_ids.insert(candidate_id);
        }
        Some(selected_ids)
    }
}

// --- Benchmark Suite 1: Wallenius Draw ---
fn benchmark_wallenius_draw(c: &mut Criterion) {
    let mut group = c.benchmark_group("Wallenius Draw (select_and_remove)");
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    for &n in &[100_000, 1_000_000] { // Reduced sizes for faster runs
        group.throughput(Throughput::Elements(1));
        let mut rng = rand::rng();
        let precision_val = 5;
        let denominator = Decimal::from(u64::pow(10, precision_val));
        let weights: Vec<Decimal> = (0..n)
            .map(|_| Decimal::from(rng.random_range(1..=100_000)) / denominator)
            .collect();
        
        group.bench_with_input(BenchmarkId::new("DigitBinIndex", n), &n, |b, _| {
            b.iter_batched(|| {
                let mut dbi = DigitBinIndex::with_precision(precision_val as u8);
                for (i, &weight) in weights.iter().enumerate() { dbi.add(i as u32, weight); }
                dbi
            }, |mut dbi| { dbi.select_and_remove(); }, criterion::BatchSize::SmallInput);
        });

        group.bench_with_input(BenchmarkId::new("FenwickTree", n), &n, |b, _| {
            b.iter_batched(|| {
                let mut ft = FenwickTree::new(n);
                let mut total_weight = Decimal::from(0);
                for (i, &weight) in weights.iter().enumerate() {
                    ft.add(i, weight);
                    total_weight += weight;
                }
                (ft, total_weight)
            }, |(mut ft, total_weight)| { ft.select_and_remove(total_weight); }, criterion::BatchSize::SmallInput);
        });
    }
    group.finish();
}

// --- Benchmark Suite 2: Fisher's Draw ---
fn benchmark_fisher_draw(c: &mut Criterion) {
    let mut group = c.benchmark_group("Fisher's Draw (select_many_and_remove)");
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    for &n in &[10_000, 100_000] { // Using smaller N as rejection sampling is slow
        let k = n / 100; // Let's draw 1% of the population
        group.throughput(Throughput::Elements(k as u64));

        let mut rng = rand::rng();
        let precision_val = 5;
        let denominator = Decimal::from(u64::pow(10, precision_val));
        let weights: Vec<Decimal> = (0..n)
            .map(|_| Decimal::from(rng.random_range(1..=100_000)) / denominator)
            .collect();
        
        group.bench_with_input(BenchmarkId::new("DigitBinIndex", n), &k, |b, &k| {
            b.iter_batched(|| {
                let mut dbi = DigitBinIndex::with_precision(precision_val as u8);
                for (i, &weight) in weights.iter().enumerate() { dbi.add(i as u32, weight); }
                dbi
            }, |mut dbi| { dbi.select_many_and_remove(k as u32); }, criterion::BatchSize::SmallInput);
        });

        group.bench_with_input(BenchmarkId::new("FenwickTree (Rejection)", n), &k, |b, &k| {
            b.iter_batched(|| {
                let mut ft = FenwickTree::new(n);
                for (i, &weight) in weights.iter().enumerate() { ft.add(i, weight); }
                ft
            }, |mut ft| { ft.select_many_and_remove(k); }, criterion::BatchSize::SmallInput);
        });
    }
    group.finish();
}

// Register both benchmark suites to be run with `cargo bench`
criterion_group!(benches, benchmark_wallenius_draw, benchmark_fisher_draw);
criterion_main!(benches);