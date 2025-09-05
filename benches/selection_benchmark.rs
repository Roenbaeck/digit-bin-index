use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use digit_bin_index::DigitBinIndex;
use fraction::Decimal;
use rand::prelude::*;

// --- Competitor Implementation: Fenwick Tree (Binary Indexed Tree) ---
// This is the standard, general-purpose data structure for this problem.
// It has O(log N) complexity for both updates and lookups.

struct FenwickTree {
    tree: Vec<Decimal>,
}

impl FenwickTree {
    fn new(size: usize) -> Self {
        Self {
            tree: vec![Decimal::from(0); size + 1],
        }
    }

    // Adds a delta to a specific index. O(log N)
    fn add(&mut self, mut index: usize, delta: Decimal) {
        index += 1; // 1-based indexing
        while index < self.tree.len() {
            self.tree[index] += delta;
            index += index & index.wrapping_neg(); // Add last set bit
        }
    }

    // Finds the first index with a cumulative sum >= target. O(log N)
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
        self.tree[1..].iter().sum()
    }

    // A combined operation to simulate Wallenius' draw.
    fn select_and_remove(&mut self) -> Option<(usize, Decimal)> {
        let total_weight = self.total_weight();
        if total_weight.is_zero() {
            return None;
        }

        let mut rng = thread_rng();
        let random_target = rng.gen_range(Decimal::from(0)..total_weight);

        let index = self.find(random_target);
        
        // To get the original weight, we'd need another structure.
        // For this benchmark, we just remove a fixed value as an approximation.
        // This is a known limitation when benchmarking against a simple Fenwick Tree.
        let approx_weight = total_weight / Decimal::from(self.tree.len() as u32);
        self.add(index, -approx_weight);
        
        Some((index, approx_weight))
    }
}

// --- Benchmark Setup ---

fn benchmark_select_and_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("Select and Remove Performance");

    // We will test for different numbers of individuals (N)
    for &n in &[10_000, 100_000, 1_000_000] {
        group.throughput(Throughput::Elements(1)); // We measure time per single operation

        // --- Setup ---
        let mut rng = StdRng::seed_from_u64(42);
        let precision_val = 5;
        let denominator = Decimal::from(u64::pow(10, precision_val));
        let weights: Vec<Decimal> = (0..n)
            .map(|_| Decimal::from(rng.gen_range(1..=100_000)) / denominator)
            .collect();
        
        // --- DigitBinIndex Benchmark ---
        group.bench_with_input(BenchmarkId::new("DigitBinIndex", n), &n, |b, _| {
            // iter_batched separates setup from the code being measured.
            b.iter_batched(
                || {
                    // Setup: Create and populate a fresh index for each batch.
                    let mut dbi = DigitBinIndex::with_precision(precision_val as u8);
                    for (i, &weight) in weights.iter().enumerate() {
                        dbi.add(i as u32, weight);
                    }
                    dbi
                },
                |mut dbi| {
                    // Action: The code being benchmarked.
                    let _ = dbi.select_and_remove();
                },
                criterion::BatchSize::SmallInput,
            );
        });

        // --- FenwickTree Benchmark ---
        group.bench_with_input(BenchmarkId::new("FenwickTree", n), &n, |b, _| {
            b.iter_batched(
                || {
                    // Setup: Create and populate the Fenwick tree.
                    let mut ft = FenwickTree::new(n);
                    for (i, &weight) in weights.iter().enumerate() {
                        ft.add(i, weight);
                    }
                    ft
                },
                |mut ft| {
                    // Action: The code being benchmarked.
                    let _ = ft.select_and_remove();
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_select_and_remove);
criterion_main!(benches);