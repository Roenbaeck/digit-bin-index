use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use digit_bin_index::DigitBinIndex;
use rand::Rng;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use std::collections::HashMap; 
use std::hint::black_box;
use rand::seq::SliceRandom;

// --- Core Fenwick Tree (Unchanged) ---
#[derive(Clone)]
struct FenwickTree {
    tree: Vec<Decimal>,
    capacity: usize,
    total_weight: Decimal,
}

impl FenwickTree {
    fn new(capacity: usize) -> Self {
        Self { tree: vec![Decimal::ZERO; capacity + 1], capacity, total_weight: Decimal::ZERO }
    }

    fn modify_weight(&mut self, mut index: usize, delta: Decimal) {
        self.total_weight += delta;
        index += 1;
        while index <= self.capacity {
            self.tree[index] += delta;
            index += index & index.wrapping_neg();
        }
    }

    fn find(&self, target: Decimal) -> usize {
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
    index_to_weight: Vec<Decimal>,
    free_indices: Vec<usize>,
}

impl WeightedSelector {
    pub fn new(capacity: usize) -> Self {
        Self {
            ft: FenwickTree::new(capacity),
            id_to_index: HashMap::with_capacity(capacity),
            index_to_id: vec![0; capacity],
            index_to_weight: vec![Decimal::ZERO; capacity],
            free_indices: (0..capacity).rev().collect(),
        }
    }

    pub fn add(&mut self, id: u32, weight: Decimal) -> Result<(), &'static str> {
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
    pub fn select_and_remove(&mut self) -> Option<(u32, Decimal)> {
        if self.ft.total_weight.is_zero() { return None; }
        let mut rng = rand::thread_rng();
        let target = rng.gen_range(Decimal::ZERO..self.ft.total_weight);
        let index = self.ft.find(target);
        let id = self.index_to_id[index];
        let weight = self.index_to_weight[index];
        
        self.id_to_index.remove(&id);
        self.ft.modify_weight(index, -weight);
        self.index_to_weight[index] = Decimal::ZERO;
        self.free_indices.push(index);

        // CHANGED: Return tuple with weight
        Some((id, weight))
    }

    // --- CORRECTED FISHER'S DRAW (Systematic PPS Sampling) ---
    pub fn select_many_and_remove(&mut self, num_to_draw: u32) -> Option<Vec<(u32, Decimal)>> {
        let num_to_draw = num_to_draw as usize;
        let current_pop_size = self.id_to_index.len();
        if num_to_draw > current_pop_size { return None; }
        if self.ft.total_weight.is_zero() || num_to_draw == 0 { return Some(Vec::new()); }

        let mut rng = rand::thread_rng();

        // Collect active items and shuffle
        let mut active: Vec<(u32, Decimal, f64, usize)> = self
            .id_to_index
            .iter()
            .map(|(&id, &index)| {
                let dec_w = self.index_to_weight[index];
                (id, dec_w, dec_w.to_f64().unwrap(), index)
            })
            .collect();
        active.shuffle(&mut rng);

        // Compute step and start
        let total: f64 = active.iter().map(|&(_, _, w_f, _)| w_f).sum();
        let step = total / num_to_draw as f64;
        let start = rng.gen_range(0.0..step);

        // Select using systematic sampling
        let mut cum: f64 = 0.0;
        let mut current = start;
        let mut result_vec: Vec<(u32, Decimal)> = Vec::with_capacity(num_to_draw);
        let mut pos = 0;
        for _ in 0..num_to_draw {
            while pos < active.len() && cum < current {
                cum += active[pos].2;
                pos += 1;
            }
            if pos > 0 && pos <= active.len() {
                let (id, dec_w, _, _) = active[pos - 1];
                result_vec.push((id, dec_w));
            }
            current += step;
        }

        // Perform removals
        for (id, weight) in result_vec.iter() {
            if let Some(index) = self.id_to_index.remove(id) {
                self.ft.modify_weight(index, -*weight);
                self.index_to_weight[index] = Decimal::ZERO;
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

// --- Benchmarks (No changes needed here, the logic is identical) ---

fn benchmark_wallenius_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Wallenius Simulation (Iterative Churn)");
    group.throughput(Throughput::Elements((CHURN_COUNT + ACQUISITION_COUNT) as u64));

    group.bench_function(BenchmarkId::new("DigitBinIndex", INITIAL_POP), |b| {
        b.iter_batched(|| {
            let mut dbi = DigitBinIndex::with_precision(5);
            let mut rng = rand::thread_rng();
            for i in 0..INITIAL_POP { dbi.add(i as u32, rng.gen_range(Decimal::new(1,4)..Decimal::new(9999,4))); }
            (dbi, INITIAL_POP as u32)
        }, |(mut dbi, mut next_id)| {
            for _ in 0..CHURN_COUNT { black_box(dbi.select_and_remove()); }
            let mut rng = rand::thread_rng();
            for _ in 0..ACQUISITION_COUNT {
                dbi.add(next_id, rng.gen_range(Decimal::new(1,4)..Decimal::new(9999,4)));
                next_id += 1;
            }
        }, criterion::BatchSize::SmallInput);
    });

    group.bench_function(BenchmarkId::new("WeightedSelector", INITIAL_POP), |b| {
        b.iter_batched(|| {
            let mut selector = WeightedSelector::new(MAX_CAPACITY);
            let mut rng = rand::thread_rng();
            for i in 0..INITIAL_POP { selector.add(i as u32, rng.gen_range(Decimal::new(1,4)..Decimal::new(9999,4))).unwrap(); }
            (selector, INITIAL_POP as u32)
        }, |(mut selector, mut next_id)| {
            for _ in 0..CHURN_COUNT { black_box(selector.select_and_remove()); }
            let mut rng = rand::thread_rng();
            for _ in 0..ACQUISITION_COUNT {
                selector.add(next_id, rng.gen_range(Decimal::new(1,4)..Decimal::new(9999,4))).unwrap();
                next_id += 1;
            }
        }, criterion::BatchSize::SmallInput);
    });
    group.finish();
}

fn benchmark_fisher_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Fisher Simulation (Batch Churn)");
    group.throughput(Throughput::Elements((CHURN_COUNT + ACQUISITION_COUNT) as u64));

    group.bench_function(BenchmarkId::new("DigitBinIndex", INITIAL_POP), |b| {
        b.iter_batched(|| {
            let mut dbi = DigitBinIndex::with_precision(5);
            let mut rng = rand::thread_rng();
            for i in 0..INITIAL_POP { dbi.add(i as u32, rng.gen_range(Decimal::new(1,4)..Decimal::new(9999,4))); }
            (dbi, INITIAL_POP as u32)
        }, |(mut dbi, mut next_id)| {
            black_box(dbi.select_many_and_remove(CHURN_COUNT as u32));
            let mut rng = rand::thread_rng();
            for _ in 0..ACQUISITION_COUNT {
                dbi.add(next_id, rng.gen_range(Decimal::new(1,4)..Decimal::new(9999,4)));
                next_id += 1;
            }
        }, criterion::BatchSize::SmallInput);
    });

    group.bench_function(BenchmarkId::new("WeightedSelector", INITIAL_POP), |b| {
        b.iter_batched(|| {
            let mut selector = WeightedSelector::new(MAX_CAPACITY);
            let mut rng = rand::thread_rng();
            for i in 0..INITIAL_POP { selector.add(i as u32, rng.gen_range(Decimal::new(1,4)..Decimal::new(9999,4))).unwrap(); }
            (selector, INITIAL_POP as u32)
        }, |(mut selector, mut next_id)| {
            black_box(selector.select_many_and_remove(CHURN_COUNT as u32));
            let mut rng = rand::thread_rng();
            for _ in 0..ACQUISITION_COUNT {
                selector.add(next_id, rng.gen_range(Decimal::new(1,4)..Decimal::new(9999,4))).unwrap();
                next_id += 1;
            }
        }, criterion::BatchSize::SmallInput);
    });
    group.finish();
}

criterion_group!(benches, benchmark_wallenius_simulation, benchmark_fisher_simulation);
criterion_main!(benches);

#[cfg(test)]
mod tests {
    use super::*; // Import parent module's items like WeightedSelector
    use rust_decimal_macros::dec;

    #[test]
    fn test_weighted_selector_wallenius_distribution() {
        // --- Setup: Create a controlled population ---
        const ITEMS_PER_GROUP: u32 = 1000;
        const TOTAL_ITEMS: u32 = ITEMS_PER_GROUP * 2;
        const NUM_DRAWS: u32 = TOTAL_ITEMS / 2;

        let low_risk_weight = dec!(0.1);
        let high_risk_weight = dec!(0.2);

        // --- Execution: Run many simulations to average out randomness ---
        const NUM_SIMULATIONS: u32 = 100;
        let mut total_high_risk_selected = 0;

        for _ in 0..NUM_SIMULATIONS {
            let mut selector = WeightedSelector::new(TOTAL_ITEMS as usize);
            for i in 0..ITEMS_PER_GROUP { selector.add(i, low_risk_weight).unwrap(); }
            for i in ITEMS_PER_GROUP..TOTAL_ITEMS { selector.add(i, high_risk_weight).unwrap(); }

            let mut high_risk_in_this_run = 0;
            for _ in 0..NUM_DRAWS {
                if let Some((selected_id, _)) = selector.select_and_remove() {
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

        let low_risk_weight = dec!(0.1);
        let high_risk_weight = dec!(0.2);

        // --- Execution: Run many simulations ---
        const NUM_SIMULATIONS: u32 = 100;
        let mut total_high_risk_selected = 0;

        for _ in 0..NUM_SIMULATIONS {
            let mut selector = WeightedSelector::new(TOTAL_ITEMS as usize);
            for i in 0..ITEMS_PER_GROUP { selector.add(i, low_risk_weight).unwrap(); }
            for i in ITEMS_PER_GROUP..TOTAL_ITEMS { selector.add(i, high_risk_weight).unwrap(); }
            
            if let Some(selected_items) = selector.select_many_and_remove(NUM_DRAWS) {
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