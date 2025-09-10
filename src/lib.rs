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
#[derive(Debug, Clone)]
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

    // --- Standard Functions ---

    /// Adds an individual with a specific weight (probability) to the index.
    pub fn add(&mut self, individual_id: u32, mut weight: Decimal) -> bool {
        // Guard against adding non-positive weights, which are invalid for this structure.
        if weight <= Decimal::ZERO {
            return false;
        }

        weight.rescale(self.precision as u32);

        // After rescaling, a very small positive weight might become zero.
        if weight.is_zero() {
            return false;
        }

        Self::add_recurse(&mut self.root, individual_id, weight, 1, self.precision);
        true
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
            if let NodeContent::Internal(_) = &node.content {
                 // First time adding to this path, convert to Leaf
                node.content = NodeContent::Leaf(RoaringBitmap::new());
            }
            if let NodeContent::Leaf(bitmap) = &mut node.content {
                bitmap.insert(individual_id);
            }
            return;
        }

        let digit = Self::get_digit_at(weight, current_depth);
        if let NodeContent::Internal(children) = &mut node.content {
            if children.len() <= digit {
                children.resize_with(digit + 1, Node::new_internal);
            }
            Self::add_recurse(&mut children[digit], individual_id, weight, current_depth + 1, max_depth);
        }
    }

    /// Removes an individual with a specific weight (probability) from the index.
    pub fn remove(&mut self, individual_id: u32, mut weight: Decimal) {
        weight.rescale(self.precision as u32);
        Self::remove_recurse(&mut self.root, individual_id, weight, 1, self.precision);
    }

    /// Recursive private method to handle removing individuals.
    fn remove_recurse(
        node: &mut Node,
        individual_id: u32,
        weight: Decimal,
        current_depth: u8,
        max_depth: u8,
    ) -> bool {
        if current_depth > max_depth {
            if let NodeContent::Leaf(bitmap) = &mut node.content {
                if bitmap.remove(individual_id) {
                    node.content_count -= 1;
                    node.accumulated_value -= weight;
                    return true;
                }
            }
            return false;
        }

        let digit = Self::get_digit_at(weight, current_depth);
        if let NodeContent::Internal(children) = &mut node.content {
            if children.len() > digit && Self::remove_recurse(&mut children[digit], individual_id, weight, current_depth + 1, max_depth) {
                node.content_count -= 1;
                node.accumulated_value -= weight;
                return true;
            }
        }
        false
    }


    // --- Selection Functions ---

    /// Performs random selection of one individual.
    pub fn select(&self) -> Option<(u32, Decimal)> {
        if self.root.content_count == 0 {
            return None;
        }

        let mut rng = rand::thread_rng();
        let random_target = rng.gen_range(Decimal::ZERO..self.root.accumulated_value);

        self.select_recurse(&self.root, random_target, Decimal::ZERO, 1)
    }

    /// Recursive helper for the select function.
    fn select_recurse(&self, node: &Node, mut target: Decimal, weight: Decimal, current_depth: u8) -> Option<(u32, Decimal)> {
        if current_depth > self.precision {
             if let NodeContent::Leaf(bitmap) = &node.content {
                if bitmap.is_empty() { return None; }
                let mut rng = rand::thread_rng();
                // Select a random Nth element from the bitmap iterator
                let rand_index = rng.gen_range(0..bitmap.len() as u32);
                let selected_id = bitmap.select(rand_index).unwrap();
                // The accumulated weight is the correct binned weight for this leaf
                return Some((selected_id, weight));
            }
        }

        if let NodeContent::Internal(children) = &node.content {
            for (i, child) in children.iter().enumerate() {
                if child.accumulated_value.is_zero() { continue; }
                if target < child.accumulated_value {
                    // CORRECTED LOGIC: Add the digit value at the current decimal place.
                    let new_weight = weight + Decimal::new(i as i64, current_depth as u32);
                    return self.select_recurse(child, target, new_weight, current_depth + 1);
                }
                target -= child.accumulated_value;
            }
        }
        None // Should not be reached in a consistent tree
    }
    

    /// Private helper for finding a unique item using bin-aware rejection sampling.
    /// It performs one weighted traversal and returns a unique item, or None if the
    /// chosen bin is already exhausted.
    fn select_unique(&self, selected_ids: &RoaringBitmap) -> Option<(u32, Decimal)> {
        if self.root.content_count == 0 {
            return None;
        }
        let mut rng = rand::thread_rng();
        let random_target = rng.gen_range(Decimal::ZERO..self.root.accumulated_value);

        // Call the new recursive helper that is aware of already selected IDs
        self.select_unique_recurse(&self.root, random_target, Decimal::ZERO, 1, selected_ids)
    }

    /// NEW recursive helper for the unique selection process.
    fn select_unique_recurse(
        &self,
        node: &Node,
        mut target: Decimal,
        weight: Decimal,
        current_depth: u8,
        selected_ids: &RoaringBitmap,
    ) -> Option<(u32, Decimal)> {
        // Base Case: We've reached a leaf bin.
        if current_depth > self.precision {
            if let NodeContent::Leaf(bitmap) = &node.content {
                // Find all individuals in this bin who have NOT already been selected.
                let available_ids = bitmap - selected_ids;
                if available_ids.is_empty() {
                    // This bin is exhausted for this batch. Trigger a rejection by returning None.
                    return None;
                }

                // Select any individual from the available set.
                let selected_id = available_ids.min().unwrap();
                
                // The weight was constructed on the way down.
                return Some((selected_id, weight));
            }
        }

        // Recursive Step: Traverse internal nodes.
        if let NodeContent::Internal(children) = &node.content {
            for (i, child) in children.iter().enumerate() {
                if child.accumulated_value.is_zero() { continue; }
                if target < child.accumulated_value {
                    let new_weight = weight + Decimal::new(i as i64, current_depth as u32);
                    // Propagate the result (or the rejection) upwards.
                    return self.select_unique_recurse(child, target, new_weight, current_depth + 1, selected_ids);
                }
                target -= child.accumulated_value;
            }
        }
        None // Should not be reached in a consistent tree
    }    

    /// Selects multiple unique individuals.
    pub fn select_many(&self, num_to_draw: u32) -> Option<HashSet<(u32, Decimal)>> {
        if num_to_draw > self.count() {
            return None;
        }
        if num_to_draw == 0 {
            return Some(HashSet::new());
        }

        let mut selected = HashSet::with_capacity(num_to_draw as usize);
        let mut selected_ids = RoaringBitmap::new();
        while selected.len() < num_to_draw as usize {
            if let Some((id, weight)) = self.select_unique(&selected_ids) {
                if selected_ids.insert(id) {
                    selected.insert((id, weight));
                }
            } 
        }
        Some(selected)
    }

    /// Selects and removes a single individual.
    pub fn select_and_remove(&mut self) -> Option<(u32, Decimal)> {
        if let Some((individual_id, weight)) = self.select() {
            self.remove(individual_id, weight);
            Some((individual_id, weight))
        } else {
            None
        }
    }

    /// Selects and removes multiple unique individuals.
    pub fn select_many_and_remove(&mut self, num_to_draw: u32) -> Option<HashSet<(u32, Decimal)>> {
        if let Some(selected) = self.select_many(num_to_draw) {
            for &(individual_id, weight) in &selected {
                self.remove(individual_id, weight);
            }
            Some(selected)
        } else {
            None
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
    use rust_decimal::prelude::FromPrimitive;

    #[pyclass(name = "DigitBinIndex")]
    struct PyDigitBinIndex {
        index: DigitBinIndex,
    }

    #[pymethods]
    impl PyDigitBinIndex {
        #[new]
        fn new(precision: u32) -> Self {
            PyDigitBinIndex {
                index: DigitBinIndex::with_precision(precision.try_into().unwrap()),
            }
        }

        fn add(&mut self, id: u32, weight: f64) -> bool {
            if let Some(decimal_weight) = Decimal::from_f64(weight) {
                 self.index.add(id, decimal_weight)
            } else {
                false
            }
        }

        fn remove(&mut self, id: u32, weight: f64) {
            if let Some(decimal_weight) = Decimal::from_f64(weight) {
                self.index.remove(id, decimal_weight);
            }
        }

        fn select(&self) -> Option<(u32, String)> {
            self.index.select().map(|(id, weight)| (id, weight.to_string()))
        }

        fn select_many(&self, n: u32) -> Option<Vec<(u32, String)>> {
            self.index.select_many(n).map(|set| {
                set.into_iter().map(|(id, w)| (id, w.to_string())).collect()
            })
        }

        fn select_and_remove(&mut self) -> Option<(u32, String)> {
            self.index.select_and_remove().map(|(id, weight)| (id, weight.to_string()))
        }

        fn select_many_and_remove(&mut self, n: u32) -> Option<Vec<(u32, String)>> {
            self.index.select_many_and_remove(n).map(|set| {
                set.into_iter().map(|(id, w)| (id, w.to_string())).collect()
            })
        }

        fn count(&self) -> u32 {
            self.index.count()
        }

        fn total_weight(&self) -> String {
            self.index.total_weight().to_string()
        }
    }

    #[pymodule]
    fn digit_bin_index(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
        m.add_class::<PyDigitBinIndex>()?;
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_select_and_remove() {
        let mut index = DigitBinIndex::with_precision(3);
        index.add(1, dec!(0.085));
        index.add(2, dec!(0.205));
        index.add(3, dec!(0.346));
        index.add(4, dec!(0.364));
        println!("Initial state: {} individuals, total weight = {}", index.count(), index.total_weight());    
        if let Some((id, weight)) = index.select_and_remove() {
            println!("Selected ID: {} with weight: {}", id, weight);
        }
        println!("Intermediate state: {} individuals, total weight = {}", index.count(), index.total_weight()); 
        if let Some((id, weight)) = index.select_and_remove() {
            println!("Selected ID: {} with weight: {}", id, weight);
        }
        println!("Intermediate state: {} individuals, total weight = {}", index.count(), index.total_weight()); 
        if let Some((id, weight)) = index.select_and_remove() {
            println!("Selected ID: {} with weight: {}", id, weight);
        }
        println!("Final state: {} individuals, total weight = {}", index.count(), index.total_weight()); 
    }

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
                let high_risk_in_this_run = selected_ids.iter().filter(|&&(id, _)| id >= ITEMS_PER_GROUP).count();
                total_high_risk_selected += high_risk_in_this_run as u32;
            }
        }
        
        let avg_high_risk = total_high_risk_selected as f64 / NUM_SIMULATIONS as f64;
        let fishers_mean = NUM_DRAWS as f64 * (2.0 / 3.0);
        let tolerance = fishers_mean * 0.02;

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