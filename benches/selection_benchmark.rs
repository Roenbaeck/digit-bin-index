use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use digit_bin_index::DigitBinIndex;
use wyrand::WyRand;
use rand::{Rng, SeedableRng}; 
use std::collections::HashMap; 
use std::hint::black_box;
use rand::seq::SliceRandom;

// --- Core Fenwick Tree (Unchanged) ---
#[derive(Clone)]
struct FenwickTree {
    tree: Vec<f64>,
    capacity: usize,
    total_weight: f64,
}

impl FenwickTree {
    fn new(capacity: usize) -> Self {
        Self { tree: vec![0.0; capacity + 1], capacity, total_weight: 0.0 }
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
        let mut bit_mask = 1 << (self.capacity.next_power_of_two().trailing_zeros().saturating_sub(1));
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
        if self.id_to_index.contains_key(&id) { return Err("ID already exists"); }
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
        if self.ft.total_weight == 0.0 { return None; }
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

    // --- CORRECTED FISHER'S DRAW (Systematic PPS Sampling) ---
    pub fn select_many_and_remove<R: Rng>(&mut self, num_to_draw: u32, rng: &mut R) -> Option<Vec<(u32, f64)>> {
        let num_to_draw = num_to_draw as usize;
        let current_pop_size = self.id_to_index.len();
        if num_to_draw > current_pop_size { return None; }
        if self.ft.total_weight == 0.0 || num_to_draw == 0 { return Some(Vec::new()); }

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
const INITIAL_POP: usize = 1_000_000;
const CHURN_COUNT: usize = 100_000;
const ACQUISITION_COUNT: usize = 110_000;
const MAX_CAPACITY: usize = INITIAL_POP + ACQUISITION_COUNT;

const VERY_LARGE_POP: usize = 10_000_000;
const VERY_LARGE_CHURN: usize = 1_000_000;
const VERY_LARGE_ACQ: usize = 1_000_000;
const VERY_LARGE_MAX: usize = VERY_LARGE_POP + VERY_LARGE_ACQ;

// --- Benchmarks (No changes needed here, the logic is identical) ---

fn benchmark_wallenius_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Wallenius Simulation (Iterative Churn)");
    group.throughput(Throughput::Elements((CHURN_COUNT + ACQUISITION_COUNT) as u64));

    group.bench_function(BenchmarkId::new("DigitBinIndex (precision 3)", INITIAL_POP), |b| {
        b.iter_batched(|| {
            let mut dbi = DigitBinIndex::with_precision_and_capacity(3, MAX_CAPACITY);
            let mut rng = WyRand::from_os_rng();
            for i in 0..INITIAL_POP { dbi.add(i as u32, rng.random_range(0.001..0.999)); }
            (dbi, INITIAL_POP as u32, rng)
        }, |(mut dbi, mut next_id, mut rng)| {
            for _ in 0..CHURN_COUNT { black_box(dbi.select_and_remove()); }
            for _ in 0..ACQUISITION_COUNT {
                dbi.add(next_id, rng.random_range(0.001..0.999));
                next_id += 1;
            }
        }, criterion::BatchSize::SmallInput);
    });

    group.bench_function(BenchmarkId::new("WeightedSelector (precision 3)", INITIAL_POP), |b| {
        b.iter_batched(|| {
            let mut selector = WeightedSelector::new(MAX_CAPACITY);
            let mut rng = WyRand::from_os_rng();
            for i in 0..INITIAL_POP { selector.add(i as u32, rng.random_range(0.001..0.999)).unwrap(); }
            (selector, INITIAL_POP as u32, rng)
        }, |(mut selector, mut next_id, mut rng)| {
            for _ in 0..CHURN_COUNT { black_box(selector.select_and_remove(&mut rng)); }
            for _ in 0..ACQUISITION_COUNT {
                selector.add(next_id, rng.random_range(0.001..0.999)).unwrap();
                next_id += 1;
            }
        }, criterion::BatchSize::SmallInput);
    });

    group.bench_function(BenchmarkId::new("DigitBinIndex (precision 5)", INITIAL_POP), |b| {
        b.iter_batched(|| {
            let mut dbi = DigitBinIndex::with_precision_and_capacity(5, MAX_CAPACITY);
            let mut rng = WyRand::from_os_rng();
            for i in 0..INITIAL_POP { dbi.add(i as u32, rng.random_range(0.00001..0.99999)); }
            (dbi, INITIAL_POP as u32, rng)
        }, |(mut dbi, mut next_id, mut rng)| {
            for _ in 0..CHURN_COUNT { black_box(dbi.select_and_remove()); }
            for _ in 0..ACQUISITION_COUNT {
                dbi.add(next_id, rng.random_range(0.00001..0.99999));
                next_id += 1;
            }
        }, criterion::BatchSize::SmallInput);
    });

    group.bench_function(BenchmarkId::new("WeightedSelector (precision 5)", INITIAL_POP), |b| {
        b.iter_batched(|| {
            let mut selector = WeightedSelector::new(MAX_CAPACITY);
            let mut rng = WyRand::from_os_rng();
            for i in 0..INITIAL_POP { selector.add(i as u32, rng.random_range(0.00001..0.99999)).unwrap(); }
            (selector, INITIAL_POP as u32, rng)
        }, |(mut selector, mut next_id, mut rng)| {
            for _ in 0..CHURN_COUNT { black_box(selector.select_and_remove(&mut rng)); }
            for _ in 0..ACQUISITION_COUNT {
                selector.add(next_id, rng.random_range(0.00001..0.99999)).unwrap();
                next_id += 1;
            }
        }, criterion::BatchSize::SmallInput);
    });

    group.sample_size(10); 

    group.bench_function(BenchmarkId::new("DigitBinIndex (very large, Roaring)", VERY_LARGE_POP), |b| {
        b.iter_batched(|| {
            let mut dbi = DigitBinIndex::with_precision_and_capacity(3, VERY_LARGE_MAX);
            let mut rng = WyRand::from_os_rng();
            for i in 0..VERY_LARGE_POP {
                dbi.add(i as u32, rng.random_range(0.001..0.999));
            }
            (dbi, VERY_LARGE_POP as u32, rng)
        }, |(mut dbi, mut next_id, mut rng)| {
            for _ in 0..VERY_LARGE_CHURN { black_box(dbi.select_and_remove()); }
            for _ in 0..VERY_LARGE_ACQ {
                dbi.add(next_id, rng.random_range(0.001..0.999));
                next_id += 1;
            }
        }, criterion::BatchSize::SmallInput);
    });

    group.bench_function(BenchmarkId::new("WeightedSelector (very large, FenwickTree)", VERY_LARGE_POP), |b| {
        b.iter_batched(|| {
            let mut selector = WeightedSelector::new(VERY_LARGE_MAX);
            let mut rng = WyRand::from_os_rng();
            for i in 0..VERY_LARGE_POP {
                selector.add(i as u32, rng.random_range(0.001..0.999)).unwrap();
            }
            (selector, VERY_LARGE_POP as u32, rng)
        }, |(mut selector, mut next_id, mut rng)| {
            for _ in 0..VERY_LARGE_CHURN { black_box(selector.select_and_remove(&mut rng)); }
            for _ in 0..VERY_LARGE_ACQ {
                selector.add(next_id, rng.random_range(0.001..0.999)).unwrap();
                next_id += 1;
            }
        }, criterion::BatchSize::SmallInput);
    });    

    group.finish();
}

fn benchmark_fisher_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Fisher Simulation (Batch Churn)");
    group.throughput(Throughput::Elements((CHURN_COUNT + ACQUISITION_COUNT) as u64));

    group.bench_function(BenchmarkId::new("DigitBinIndex (precision 3)", INITIAL_POP), |b| {
        b.iter_batched(|| {
            let mut dbi = DigitBinIndex::with_precision_and_capacity(3, MAX_CAPACITY);
            let mut rng = WyRand::from_os_rng();
            for i in 0..INITIAL_POP { dbi.add(i as u32, rng.random_range(0.001..0.999)); }
            (dbi, INITIAL_POP as u32, rng)
        }, |(mut dbi, mut next_id, mut rng)| {
            black_box(dbi.select_many_and_remove(CHURN_COUNT as u32));
            for _ in 0..ACQUISITION_COUNT {
                dbi.add(next_id, rng.random_range(0.001..0.999));
                next_id += 1;
            }
        }, criterion::BatchSize::SmallInput);
    });

    group.bench_function(BenchmarkId::new("WeightedSelector (precision 3)", INITIAL_POP), |b| {
        b.iter_batched(|| {
            let mut selector = WeightedSelector::new(MAX_CAPACITY);
            let mut rng = WyRand::from_os_rng();
            for i in 0..INITIAL_POP { selector.add(i as u32, rng.random_range(0.001..0.999)).unwrap(); }
            (selector, INITIAL_POP as u32, rng)
        }, |(mut selector, mut next_id, mut rng)| {
            black_box(selector.select_many_and_remove(CHURN_COUNT as u32, &mut rng));
            for _ in 0..ACQUISITION_COUNT {
                selector.add(next_id, rng.random_range(0.001..0.999)).unwrap();
                next_id += 1;
            }
        }, criterion::BatchSize::SmallInput);
    });

    group.bench_function(BenchmarkId::new("DigitBinIndex (precision 5)", INITIAL_POP), |b| {
        b.iter_batched(|| {
            let mut dbi = DigitBinIndex::with_precision_and_capacity(5, MAX_CAPACITY);
            let mut rng = WyRand::from_os_rng();
            for i in 0..INITIAL_POP { dbi.add(i as u32, rng.random_range(0.00001..0.99999)); }
            (dbi, INITIAL_POP as u32, rng)
        }, |(mut dbi, mut next_id, mut rng)| {
            black_box(dbi.select_many_and_remove(CHURN_COUNT as u32));
            for _ in 0..ACQUISITION_COUNT {
                dbi.add(next_id, rng.random_range(0.00001..0.99999));
                next_id += 1;
            }
        }, criterion::BatchSize::SmallInput);
    });

    group.bench_function(BenchmarkId::new("WeightedSelector (precision 5)", INITIAL_POP), |b| {
        b.iter_batched(|| {
            let mut selector = WeightedSelector::new(MAX_CAPACITY);
            let mut rng = WyRand::from_os_rng();
            for i in 0..INITIAL_POP { selector.add(i as u32, rng.random_range(0.00001..0.99999)).unwrap(); }
            (selector, INITIAL_POP as u32, rng)
        }, |(mut selector, mut next_id, mut rng)| {
            black_box(selector.select_many_and_remove(CHURN_COUNT as u32, &mut rng));
            for _ in 0..ACQUISITION_COUNT {
                selector.add(next_id, rng.random_range(0.00001..0.99999)).unwrap();
                next_id += 1;
            }
        }, criterion::BatchSize::SmallInput);
    });

    group.sample_size(10); 

    group.bench_function(BenchmarkId::new("DigitBinIndex (very large, Roaring)", VERY_LARGE_POP), |b| {
        b.iter_batched(|| {
            let mut dbi = DigitBinIndex::with_precision_and_capacity(3, VERY_LARGE_MAX);
            let mut rng = WyRand::from_os_rng();
            for i in 0..VERY_LARGE_POP {
                dbi.add(i as u32, rng.random_range(0.001..0.999));
            }
            (dbi, VERY_LARGE_POP as u32, rng)
        }, |(mut dbi, mut next_id, mut rng)| {
            black_box(dbi.select_many_and_remove(VERY_LARGE_CHURN as u32));
            for _ in 0..VERY_LARGE_ACQ {
                dbi.add(next_id, rng.random_range(0.001..0.999));
                next_id += 1;
            }
        }, criterion::BatchSize::SmallInput);
    });

    group.bench_function(BenchmarkId::new("WeightedSelector (very large, FenwickTree)", VERY_LARGE_POP), |b| {
        b.iter_batched(|| {
            let mut selector = WeightedSelector::new(VERY_LARGE_MAX);
            let mut rng = WyRand::from_os_rng();
            for i in 0..VERY_LARGE_POP {
                selector.add(i as u32, rng.random_range(0.001..0.999)).unwrap();
            }
            (selector, VERY_LARGE_POP as u32, rng)
        }, |(mut selector, mut next_id, mut rng)| {
            black_box(selector.select_many_and_remove(VERY_LARGE_CHURN as u32, &mut rng));
            for _ in 0..VERY_LARGE_ACQ {
                selector.add(next_id, rng.random_range(0.001..0.999)).unwrap();
                next_id += 1;
            }
        }, criterion::BatchSize::SmallInput);
    });    

    group.finish();
}

criterion_group!(benches, benchmark_wallenius_simulation, benchmark_fisher_simulation);
criterion_main!(benches);

/*
// --- Statistical Validation Tests ---
// Commented since they confised cargo, and not really necessary now that 
// desired functionality is confirmed. Can be uncommented for future reference.
#[cfg(test)]
mod tests {
    use super::*; // Import parent module's items like WeightedSelector

    #[test]
    fn test_weighted_selector_wallenius_distribution() {
        // --- Setup: Create a controlled population ---
        const ITEMS_PER_GROUP: u32 = 1000;
        const TOTAL_ITEMS: u32 = ITEMS_PER_GROUP * 2;
        const NUM_DRAWS: u32 = TOTAL_ITEMS / 2;

        let low_risk_weight = 0.1f64;
        let high_risk_weight = 0.2f64;

        // --- Execution: Run many simulations to average out randomness ---
        const NUM_SIMULATIONS: u32 = 100;
        let mut total_high_risk_selected = 0;

        for _ in 0..NUM_SIMULATIONS {
            let mut selector = WeightedSelector::new(TOTAL_ITEMS as usize);
            for i in 0..ITEMS_PER_GROUP { selector.add(i, low_risk_weight).unwrap(); }
            for i in ITEMS_PER_GROUP..TOTAL_ITEMS { selector.add(i, high_risk_weight).unwrap(); }

            let mut high_risk_in_this_run = 0;
            let mut rng = WyRand::from_os_rng();
            for _ in 0..NUM_DRAWS {
                if let Some((selected_id, _)) = selector.select_and_remove(&mut rng) {
                    if selected_id >= ITEMS_PER_GROUP {
                        high_risk_in_this_run += 1;
                    }
                }
            }
            total_high_risk_selected += high_risk_in_this_run;
        }

        // --- Validation: Check against theoretical means ---
        let avg_high_risk = total_high_risk_selected as f64 / NUM_SIMULATIONS as f64;
        let uniform_mean = NUM_DRAWS as f64 * 0.5;
        let fishers_mean = NUM_DRAWS as f64 * (2.0 / 3.0); // Based on initial weight ratio 0.2 / (0.1 + 0.2)

        assert!(
            avg_high_risk > uniform_mean,
            "Wallenius test failed: Result {:.2} was not biased towards higher weights (uniform mean is {:.2})",
            avg_high_risk, uniform_mean
        );

        assert!(
            avg_high_risk < fishers_mean,
            "Wallenius test failed: Result {:.2} showed too much bias. It should be less than the Fisher's mean of {:.2}.",
            avg_high_risk, fishers_mean
        );
        
        println!(
            "WeightedSelector Wallenius test passed: Avg high-risk selections {:.2} is between uniform mean ({:.2}) and Fisher's mean ({:.2}).",
            avg_high_risk, uniform_mean, fishers_mean
        );
    }

    #[test]
    fn test_weighted_selector_fisher_distribution() {
        // --- Setup: Create a controlled population ---
        const ITEMS_PER_GROUP: u32 = 1000;
        const TOTAL_ITEMS: u32 = ITEMS_PER_GROUP * 2;
        const NUM_DRAWS: u32 = TOTAL_ITEMS / 2;

        let low_risk_weight = 0.1f64;
        let high_risk_weight = 0.2f64;

        // --- Execution: Run many simulations ---
        const NUM_SIMULATIONS: u32 = 100;
        let mut total_high_risk_selected = 0;

        for _ in 0..NUM_SIMULATIONS {
            let mut selector = WeightedSelector::new(TOTAL_ITEMS as usize);
            for i in 0..ITEMS_PER_GROUP { selector.add(i, low_risk_weight).unwrap(); }
            for i in ITEMS_PER_GROUP..TOTAL_ITEMS { selector.add(i, high_risk_weight).unwrap(); }
            
            let mut rng = WyRand::from_os_rng();
            if let Some(selected_items) = selector.select_many_and_remove(NUM_DRAWS, &mut rng) {
                let high_risk_in_this_run = selected_items.iter().filter(|&&(id, _)| id >= ITEMS_PER_GROUP).count();
                total_high_risk_selected += high_risk_in_this_run as u32;
            }
        }
        
        // --- Validation: Check against theoretical mean ---
        let avg_high_risk = total_high_risk_selected as f64 / NUM_SIMULATIONS as f64;
        let fishers_mean = NUM_DRAWS as f64 * (2.0 / 3.0);
        let tolerance = fishers_mean * 0.02; // Allow 5% tolerance for statistical noise

        assert!(
            (avg_high_risk - fishers_mean).abs() < tolerance,
            "Fisher's test failed: Result {:.2} was not close enough to the expected mean of {:.2} (tolerance {:.2})",
            avg_high_risk, fishers_mean, tolerance
        );
        
        println!(
            "WeightedSelector Fisher's test passed: Got avg {:.2} high-risk selections (expected ~{:.2}).",
            avg_high_risk, fishers_mean
        );
    }
}
*/