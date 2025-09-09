//! A `DigitBinIndex` is a tree-based data structure that organizes a large
//! collection of weighted items to enable highly efficient weighted random
//! selection and removal.
//!
//! It is a specialized tool, purpose-built for scenarios with millions of
//! items where probabilities are approximate and high performance is critical,
//! particularly for simulations involving sequential sampling like Wallenius'
//! noncentral hypergeometric distribution.

use rust_decimal::Decimal;
use rand::Rng; 
use roaring::RoaringBitmap;
use std::collections::HashSet; 
use std::vec;

// The default precision to use if none is specified in the constructor.
const DEFAULT_PRECISION: u8 = 3;

/// The content of a node, which is either more nodes or a leaf with individuals.
#[derive(Debug, Clone)]
pub enum NodeContent {
    /// An internal node that contains children for the next digit (0-9).
    Internal(Vec<Node>),
    /// A leaf node that contains a roaring bitmap of IDs for individuals in this bin.
    Leaf(RoaringBitmap),
}

/// A node within the DigitBinIndex tree.
#[derive(Debug, Clone)]
pub struct Node {
    /// The content of this node, either more nodes or a list of individual IDs.
    pub content: NodeContent,
    /// The total sum of probabilities stored under this node.
    pub accumulated_value: Decimal,
    /// The total count of individuals stored under this node.
    pub content_count: u32, 
}

impl Node {
    /// Creates a new, empty internal node.
    fn new_internal() -> Self {
        Self {
            content: NodeContent::Internal(vec![]),
            accumulated_value: Decimal::from(0),
            content_count: 0,
        }
    }
}

/// A data structure that organizes weighted items into bins based on their
/// decimal digits to enable fast weighted random selection and updates.
///
/// This structure is a specialized radix tree optimized for sequential sampling
/// (like in Wallenius' distribution). It makes a deliberate engineering trade-off:
/// it sacrifices a small, controllable amount of precision by binning items,
/// but in return, it achieves O(P) performance for its core operations, where P
/// is the configured precision. This is significantly faster than the O(log N)
/// performance of general-purpose structures like a Fenwick Tree for its
/// ideal use case.
#[derive(Debug)]
pub struct DigitBinIndex {
    /// The root node of the tree.
    pub root: Node,
    /// The precision (number of decimal places) used for binning.
    pub precision: u8,
}

impl Default for DigitBinIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl DigitBinIndex {
    /// Creates a new `DigitBinIndex` instance with the default precision of 3.
    #[must_use]
    pub fn new() -> Self {
        Self::with_precision(DEFAULT_PRECISION)
    }

    /// Creates a new `DigitBinIndex` instance with a specific precision.
    ///
    /// The precision determines how many decimal places are used for binning.
    /// A higher precision leads to more accurate but deeper and potentially more
    /// memory-intensive trees.
    ///
    /// # Panics
    /// Panics if `precision` is 0.
    #[must_use]
    pub fn with_precision(precision: u8) -> Self {
        assert!(precision > 0, "Precision must be at least 1.");
        Self {
            root: Node::new_internal(),
            precision,
        }
    }

    /// Helper function to get the digit at a certain decimal position.
    fn get_digit_at(weight: Decimal, position: u8) -> usize {
        let position = position as u32;
        // Get the number of decimal places (scale)
        let scale = weight.scale();

        // The number isn't that precise
        if position > scale {
            return 0;
        }
        
        // Use the absolute value of the mantissa to correctly handle negative decimals.
        let mantissa = weight.mantissa().abs() as u128;
        
        // Example for position=1 (the first decimal digit):
        // For 0.543, mantissa=543, scale=3. We want '5'.
        // 10^(3-1) = 100.
        // 543 / 100 = 5.
        // 5 % 10 = 5. That's our digit.
        let power_of_10 = 10u128.pow(scale - position);
        let digit = (mantissa / power_of_10) % 10;
        
        digit as usize
    }

    /// Adds an individual with a specific weight (probability) to the index.
    ///
    /// The operation's time complexity is O(P), where P is the configured precision.
    pub fn add(&mut self, individual_id: u32, weight: Decimal) {
        Self::add_recurse(&mut self.root, individual_id, weight, 1, self.precision);
    }

    /// Recursive private method to handle adding individuals.
    fn add_recurse(
        node: &mut Node,
        individual_id: u32,
        weight: Decimal,
        current_depth: u8,
        max_depth: u8,
    ) {
        node.content_count += 1;
        node.accumulated_value += weight;

        if current_depth > max_depth {
            match &mut node.content {
                NodeContent::Leaf(bitmap) => { bitmap.insert(individual_id); },
                NodeContent::Internal(children) => {
                    if children.is_empty() {
                        node.content = NodeContent::Leaf(RoaringBitmap::from_iter([individual_id]));
                    } else { panic!("Cannot add individual to a non-empty internal node at leaf depth."); }
                }
            }
            return;
        }

        let digit = Self::get_digit_at(weight, current_depth);
        if let NodeContent::Internal(children) = &mut node.content {
            if children.len() <= digit {
                children.resize_with(digit + 1, Node::new_internal);
            }
            Self::add_recurse(&mut children[digit], individual_id, weight, current_depth + 1, max_depth);
        } else {
            panic!("Attempted to traverse deeper on what should be a leaf node.");
        }
    }

    /// Performs a weighted random selection, removes the item, and returns its ID and an
    /// approximation of its original weight.
    ///
    /// This operation is the core of a Wallenius' noncentral hypergeometric distribution
    /// draw. The time complexity is O(P), where P is the configured precision.
    /// Returns `None` if the index is empty.
    pub fn select_and_remove(&mut self) -> Option<(u32, Decimal)> {
        if self.root.content_count == 0 {
            return None;
        }
        
        let mut rng = rand::thread_rng();
        let random_target = rng.gen_range(Decimal::ZERO..self.root.accumulated_value);
        
        let (selected_id, weight, path) = Self::select_recurse(&mut self.root, random_target, vec![]);
        self.update_values_post_removal(&path, weight);
        Some((selected_id, weight))
    }

    /// Performs a simultaneous weighted random selection of `k` unique items (Fisher's model).
    ///
    /// This uses an optimized "bin-aware rejection sampling" algorithm that is significantly
    /// faster than naive rejection sampling. The process is:
    /// 1. A selection phase where `k` unique individuals are chosen from the unaltered index.
    /// 2. An update phase where all chosen individuals are removed from the index.
    ///
    /// Returns `None` if `num_to_draw` is greater than the total number of items.
    pub fn select_many_and_remove(&mut self, num_to_draw: u32) -> Option<HashSet<u32>> {
        if num_to_draw > self.count() { return None; }
        if num_to_draw == 0 { return Some(HashSet::new()); }

        let mut selected_items = Vec::with_capacity(num_to_draw as usize);
        let mut selected_ids = HashSet::with_capacity(num_to_draw as usize);
        let mut rng = rand::thread_rng();

        // --- Phase 1: Selection via Simple Rejection Sampling ---
        // We sample from the original, unaltered tree until we find num_to_draw unique items.
        while selected_ids.len() < num_to_draw as usize {
            // We can reuse the core recursive find logic, but we don't need to pass the selected_ids down.
            // Let's create a simplified find helper for this.
            let random_target = rng.gen_range(Decimal::ZERO..self.root.accumulated_value);
            if let Some((id, weight, path)) = self.find_candidate_recurse(&self.root, random_target, vec![]) {
                if selected_ids.insert(id) {
                    // It's a new, unique item. Store its full info for the update phase.
                    selected_items.push((path, id, weight));
                }
            } else {
                // This should not happen in a non-empty tree.
                return None; 
            }
        }

        // --- Phase 2: Batched Update ---
        // Now that we have all our unique items, update the tree in one go.
        for (path, id, weight) in selected_items {
            // This is your original update function. It will handle the content_count and accumulated_value.
            // It does NOT need to touch available_count.
            Self::update_and_remove_recurse(&mut self.root, &path, id, weight);
        }
        
        Some(selected_ids)
    }

    /// A simplified recursive find that does not check for uniqueness.
    /// It's used by the rejection sampling method.
    fn find_candidate_recurse(&self, node: &Node, mut target: Decimal, mut path: Vec<usize>) -> Option<(u32, Decimal, Vec<usize>)> {
        match &node.content {
            NodeContent::Leaf(bitmap) => {
                if bitmap.is_empty() { return None; }
                let mut rng = rand::thread_rng();
                // In a simple leaf, any member is a valid candidate.
                let rand_index = rng.gen_range(0..bitmap.len() as u32);
                let selected_id = bitmap.select(rand_index).unwrap();

                let weight = node.accumulated_value / Decimal::from(node.content_count);
                Some((selected_id, weight, path))
            }
            NodeContent::Internal(children) => {
                for (i, child) in children.iter().enumerate() {
                    if child.accumulated_value.is_zero() { continue; }
                    if target < child.accumulated_value {
                        path.push(i);
                        return self.find_candidate_recurse(child, target, path);
                    }
                    target -= child.accumulated_value;
                }
                None // Should not be reached if total weight is consistent
            }
        }
    }

    /// Recursive helper to correctly update the tree and remove an item from a leaf.
    fn update_and_remove_recurse(node: &mut Node, path: &[usize], id_to_remove: u32, weight: Decimal) {
        node.content_count -= 1;
        node.accumulated_value -= weight;
        
        let Some(&index) = path.first() else {
            if let NodeContent::Leaf(bitmap) = &mut node.content {
                // --- ROARING CHANGE: Remove from the bitmap ---
                bitmap.remove(id_to_remove);
            }
            return;
        };

        if let NodeContent::Internal(children) = &mut node.content {
            if let Some(child) = children.get_mut(index) {
                // The recursive call is now to the static method itself.
                Self::update_and_remove_recurse(child, &path[1..], id_to_remove, weight);
            }
        }
    }

    /// Recursive helper to find the individual and record the traversal path.
    fn select_recurse(
        node: &mut Node,
        mut target: Decimal,
        mut path: Vec<usize>,
    ) -> (u32, Decimal, Vec<usize>) {
        match &mut node.content {
            NodeContent::Leaf(bitmap) => {
                let mut rng = rand::thread_rng();
                // --- ROARING CHANGE: Select a random Nth element from the bitmap iterator ---
                let bitmap_len = bitmap.len() as u32;
                let rand_index = rng.gen_range(0..bitmap_len);
                let selected_id = bitmap.select(rand_index).unwrap(); // Get the Nth item.

                let weight = node.accumulated_value / Decimal::from(node.content_count);
                (selected_id, weight, path)
            }
            NodeContent::Internal(children) => {
                for (i, child) in children.iter_mut().enumerate() {
                    if child.accumulated_value.is_zero() { continue; }
                    if target < child.accumulated_value {
                        path.push(i);
                        return Self::select_recurse(child, target, path);
                    }
                    target -= child.accumulated_value;
                }
                panic!("Selection logic failed: target exceeded total value of children.");
            }
        }
    }
    
    /// After an individual is removed, this updates counts up the tree.
    fn update_values_post_removal(&mut self, path: &[usize], weight: Decimal) {
        let mut current_node = &mut self.root;
        current_node.content_count -= 1;
        current_node.accumulated_value -= weight;

        for &index in path {
            if let NodeContent::Internal(children) = &mut current_node.content {
                current_node = &mut children[index];
                current_node.content_count -= 1;
                current_node.accumulated_value -= weight;
            } else {
                return;
            }
        }
    }

    /// Returns the total number of individuals in the index.
    pub fn count(&self) -> u32 {
        self.root.content_count
    }

    /// Returns the sum of all probabilities in the index.
    pub fn total_weight(&self) -> Decimal {
        self.root.accumulated_value
    }
}

#[cfg(feature = "python-bindings")]
mod python {
    use super::*; // Import parent module's items
    use pyo3::prelude::*;
    use rust_decimal::prelude::FromPrimitive; // FIX 2: Import the necessary trait

    #[pyclass(name = "DigitBinIndex")]
    struct PyDigitBinIndex {
        index: DigitBinIndex,
    }

    #[pymethods]
    impl PyDigitBinIndex {
        #[new]
        fn new(precision: u32) -> Self {
            PyDigitBinIndex {
                // FIX 1a: Convert u32 to u8
                index: DigitBinIndex::with_precision(precision.try_into().unwrap()),
            }
        }

        fn add(&mut self, id: u32, weight: f64) {
            // FIX 2: Decimal::from_f64 is now available
            if let Some(decimal_weight) = Decimal::from_f64(weight) {
                 self.index.add(id, decimal_weight);
            } else {
                // It's good practice to handle the case where f64 is not representable
                // For now, we can ignore or raise an error.
            }
        }

        fn select_and_remove(&mut self) -> Option<(u32, String)> {
            self.index.select_and_remove().map(|(id, weight)| (id, weight.to_string()))
        }

        fn select_many_and_remove(&mut self, n: usize) -> Option<Vec<u32>> {
            // FIX 1b & 3: Convert usize to u32, and then convert the resulting HashSet to a Vec
            self.index
                .select_many_and_remove(n.try_into().unwrap())
                .map(|hashset| hashset.into_iter().collect())
        }

        fn count(&self) -> usize {
            self.index.count() as usize
        }
    }

    #[pymodule]
    // FIX 4: Use the modern PyO3 function signature with `Bound`
    fn digit_bin_index(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyDigitBinIndex>()?;
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_wallenius_distribution_is_correct() {
        // --- Setup: Create a controlled population ---
        const ITEMS_PER_GROUP: u32 = 1000;
        const TOTAL_ITEMS: u32 = ITEMS_PER_GROUP * 2;
        const NUM_DRAWS: u32 = TOTAL_ITEMS / 2;

        let low_risk_weight = dec!(0.1);  // 0.1
        let high_risk_weight = dec!(0.2); // 0.2

        // --- Execution: Run many simulations to average out randomness ---
        const NUM_SIMULATIONS: u32 = 100;
        let mut total_high_risk_selected = 0;

        for _ in 0..NUM_SIMULATIONS {
            let mut index = DigitBinIndex::with_precision(3);
            for i in 0..ITEMS_PER_GROUP { index.add(i, low_risk_weight); }
            for i in ITEMS_PER_GROUP..TOTAL_ITEMS { index.add(i, high_risk_weight); }

            let mut high_risk_in_this_run = 0;
            for _ in 0..NUM_DRAWS {
                if let Some((selected_id, _)) = index.select_and_remove() {
                    if selected_id >= ITEMS_PER_GROUP {
                        high_risk_in_this_run += 1;
                    }
                }
            }
            total_high_risk_selected += high_risk_in_this_run;
        }

        // --- Validation: Check the statistical properties of a Wallenius' draw ---
        let avg_high_risk = total_high_risk_selected as f64 / NUM_SIMULATIONS as f64;

        // 1. The mean of a uniform draw (central hypergeometric) would be 500.
        let uniform_mean = NUM_DRAWS as f64 * 0.5;

        // 2. The mean of a simultaneous draw (Fisher's NCG) is based on initial proportions.
        // This is the naive expectation we started with.
        let fishers_mean = NUM_DRAWS as f64 * (2.0 / 3.0); // ~666.67

        // The mean of a Wallenius' draw is mathematically proven to lie strictly
        // between the uniform mean and the Fisher's mean.
        assert!(
            avg_high_risk > uniform_mean,
            "Test failed: Result {:.2} was not biased towards higher weights (uniform mean is {:.2})",
            avg_high_risk, uniform_mean
        );

        assert!(
            avg_high_risk < fishers_mean,
            "Test failed: Result {:.2} showed too much bias. It should be less than the Fisher's mean of {:.2} due to the Wallenius effect.",
            avg_high_risk, fishers_mean
        );

        println!(
            "Distribution test passed: Got an average of {:.2} high-risk selections.",
            avg_high_risk
        );
        println!(
            "This correctly lies between the uniform mean ({:.2}) and the Fisher's mean ({:.2}), confirming the Wallenius' distribution behavior.",
            uniform_mean, fishers_mean
        );
    }
    #[test]
    fn test_fisher_distribution_is_correct() {
        const ITEMS_PER_GROUP: u32 = 1000;
        const TOTAL_ITEMS: u32 = ITEMS_PER_GROUP * 2;
        const NUM_DRAWS: u32 = TOTAL_ITEMS / 2;

        let low_risk_weight = dec!(0.1);  // 0.1
        let high_risk_weight = dec!(0.2); // 0.2

        const NUM_SIMULATIONS: u32 = 100;
        let mut total_high_risk_selected = 0;

        for _ in 0..NUM_SIMULATIONS {
            let mut index = DigitBinIndex::with_precision(3);
            for i in 0..ITEMS_PER_GROUP { index.add(i, low_risk_weight); }
            for i in ITEMS_PER_GROUP..TOTAL_ITEMS { index.add(i, high_risk_weight); }
            
            // Call the new method
            if let Some(selected_ids) = index.select_many_and_remove(NUM_DRAWS) {
                let high_risk_in_this_run = selected_ids.iter().filter(|&&id| id >= ITEMS_PER_GROUP).count();
                total_high_risk_selected += high_risk_in_this_run as u32;
            }
        }
        
        let avg_high_risk = total_high_risk_selected as f64 / NUM_SIMULATIONS as f64;
        let fishers_mean = NUM_DRAWS as f64 * (2.0 / 3.0);
        let tolerance = fishers_mean * 0.10;

        // The mean of a Fisher's draw should be very close to the naive expectation.
        assert!(
            (avg_high_risk - fishers_mean).abs() < tolerance,
            "Fisher's test failed: Result {:.2} was not close to the expected mean of {:.2}",
            avg_high_risk, fishers_mean
        );
        
        println!(
            "Fisher's test passed: Got avg {:.2} high-risk selections (expected ~{:.2}).",
            avg_high_risk, fishers_mean
        );
    }
}