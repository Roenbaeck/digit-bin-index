//! A `DigitBinIndex` is a tree-based data structure that organizes a large
//! collection of weighted items to enable highly efficient weighted random
//! selection and removal.
//!
//! It is a specialized tool, purpose-built for scenarios with millions of
//! items where probabilities are approximate and high performance is critical,
//! particularly for simulations involving sequential sampling like Wallenius'
//! noncentral hypergeometric distribution.

use rust_decimal::Decimal;
use rand::{rngs::ThreadRng, Rng};
use roaring::RoaringBitmap;
use std::vec;

// The default precision to use if none is specified in the constructor.
const DEFAULT_PRECISION: u8 = 3;
const MAX_PRECISION: usize = 10;

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
    // For weight_to_digits
    powers: [u128; MAX_PRECISION], 
}

impl Default for DigitBinIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl DigitBinIndex {
    /// Creates a new `DigitBinIndex` instance with the default precision.
    ///
    /// The default precision is set to 3 decimal places, which provides a good balance
    /// between accuracy and performance for most use cases. For custom precision, use
    /// [`with_precision`](Self::with_precision).
    ///
    /// # Returns
    ///
    /// A new `DigitBinIndex` instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use digit_bin_index::DigitBinIndex;
    ///
    /// let index = DigitBinIndex::new();
    /// assert_eq!(index.precision, 3);
    /// ```    
    #[must_use]
    pub fn new() -> Self {
        Self::with_precision(DEFAULT_PRECISION)
    }

    /// Creates a new `DigitBinIndex` instance with the specified precision.
    ///
    /// The precision determines the number of decimal places used for binning weights.
    /// Higher precision improves sampling accuracy but increases memory usage and tree depth.
    /// Precision must be between 1 and 10 (inclusive).
    ///
    /// # Arguments
    ///
    /// * `precision` - The number of decimal places for binning (1 to 10).
    ///
    /// # Returns
    ///
    /// A new `DigitBinIndex` instance with the given precision.
    ///
    /// # Panics
    ///
    /// Panics if `precision` is 0 or greater than 10.
    ///
    /// # Examples
    ///
    /// ```
    /// use digit_bin_index::DigitBinIndex;
    ///
    /// let index = DigitBinIndex::with_precision(4);
    /// assert_eq!(index.precision, 4);
    /// ```
    #[must_use]
    pub fn with_precision(precision: u8) -> Self {
        assert!(precision > 0, "Precision must be at least 1.");
        assert!(precision <= MAX_PRECISION as u8, "Precision cannot be larger than {}.", MAX_PRECISION);
        let mut powers = [0u128; MAX_PRECISION];
        for i in 0..MAX_PRECISION {
            powers[i] = 10u128.pow(i as u32)
        }
        Self {
            root: Node::new_internal(),
            precision,
            powers,
        }        
    }

    /// Converts a Decimal weight to an array of digits [0-9] for the given precision.
    /// Returns None if the weight is invalid (non-positive or zero after scaling).
    fn weight_to_digits(&self, weight: Decimal) -> Option<[u8; MAX_PRECISION]> {
        if weight <= Decimal::ZERO {
            return None;
        }

        // Rescale to desired precision
        let mut scaled = weight;
        scaled.rescale(self.precision as u32);
        if scaled.is_zero() {
            return None;
        }

        let mut digits = [0u8; MAX_PRECISION];
        let scale = scaled.scale() as usize;
        let mantissa = scaled.mantissa().abs() as u128;

        // Extract digits from mantissa
        for i in 0..self.precision as usize {
            if i >= scale {
                digits[i] = 0; // Pad with zeros for less precise numbers
            } else {
                digits[i] = ((mantissa / self.powers[scale - i]) % 10) as u8;
            }
        }
        Some(digits)
    }

    // --- Standard Functions ---

    /// Adds an item with the given ID and weight to the index.
    ///
    /// The weight is rescaled to the index's precision and binned accordingly.
    /// If the weight is non-positive or becomes zero after scaling, the item is not added.
    ///
    /// # Arguments
    ///
    /// * `individual_id` - The unique ID of the item to add (u32).
    /// * `weight` - The positive weight (probability) of the item.
    ///
    /// # Returns
    ///
    /// `true` if the item was successfully added, `false` otherwise (e.g., invalid weight).
    ///
    /// # Examples
    ///
    /// ```
    /// use digit_bin_index::DigitBinIndex;
    /// use rust_decimal_macros::dec;
    ///
    /// let mut index = DigitBinIndex::new();
    /// let added = index.add(1, dec!(0.5));
    /// assert!(added);
    /// assert_eq!(index.count(), 1);
    /// ```    
    pub fn add(&mut self, individual_id: u32, mut weight: Decimal) -> bool {
        if let Some(digits) = self.weight_to_digits(weight) {
            weight.rescale(self.precision as u32);
            Self::add_recurse(&mut self.root, individual_id, weight, &digits, 1, self.precision);
            true
        } else {
            false
        }
    }

    /// Recursive private method to handle adding individuals.
    fn add_recurse(
        node: &mut Node,
        individual_id: u32,
        weight: Decimal, // Still needed for accumulated_value
        digits: &[u8; MAX_PRECISION],
        current_depth: u8,
        max_depth: u8,
    ) {
        node.content_count += 1;
        node.accumulated_value += weight;

        if current_depth > max_depth {
            if let NodeContent::Internal(_) = &node.content {
                node.content = NodeContent::Leaf(RoaringBitmap::new());
            }
            if let NodeContent::Leaf(bitmap) = &mut node.content {
                bitmap.insert(individual_id);
            }
            return;
        }

        let digit = digits[current_depth as usize - 1] as usize;
        if let NodeContent::Internal(children) = &mut node.content {
            if children.len() <= digit {
                children.resize_with(digit + 1, Node::new_internal);
            }
            Self::add_recurse(&mut children[digit], individual_id, weight, digits, current_depth + 1, max_depth);
        }
    }

    /// Removes an item with the given ID and weight from the index.
    ///
    /// The weight must match the one used during addition (after rescaling).
    /// If the item is not found in the corresponding bin, no removal occurs.
    ///
    /// # Arguments
    ///
    /// * `individual_id` - The ID of the item to remove.
    /// * `weight` - The weight of the item (must match the added weight).
    ///
    /// # Examples
    ///
    /// ```
    /// use digit_bin_index::DigitBinIndex;
    /// use rust_decimal_macros::dec;
    ///
    /// let mut index = DigitBinIndex::new();
    /// index.add(1, dec!(0.5));
    /// index.remove(1, dec!(0.5));
    /// assert_eq!(index.count(), 0);
    /// ```
    pub fn remove(&mut self, individual_id: u32, mut weight: Decimal) {
        if let Some(digits) = self.weight_to_digits(weight) {
            weight.rescale(self.precision as u32);
            self.remove_with_digits(individual_id, weight, digits);
        }
    }

    // Helper function
    fn remove_with_digits(&mut self, individual_id: u32, weight: Decimal, digits: [u8; MAX_PRECISION]) {
        Self::remove_recurse(&mut self.root, individual_id, weight, &digits, 1, self.precision);
    }

    /// Recursive private method to handle removing individuals.
    fn remove_recurse(
        node: &mut Node,
        individual_id: u32,
        weight: Decimal,
        digits: &[u8; MAX_PRECISION],
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

        let digit = digits[current_depth as usize - 1] as usize;
        if let NodeContent::Internal(children) = &mut node.content {
            if children.len() > digit && Self::remove_recurse(&mut children[digit], individual_id, weight, digits, current_depth + 1, max_depth) {
                node.content_count -= 1;
                node.accumulated_value -= weight;
                return true;
            }
        }
        false
    }


    // --- Selection Functions ---

    /// Selects a single item randomly based on weights without removal.
    ///
    /// Performs weighted random selection. Returns `None` if the index is empty.
    ///
    /// # Returns
    ///
    /// An `Option` containing the selected item's ID and its (rescaled) weight.
    ///
    /// # Examples
    ///
    /// ```
    /// use digit_bin_index::DigitBinIndex;
    /// use rust_decimal_macros::dec;
    ///
    /// let mut index = DigitBinIndex::new();
    /// index.add(1, dec!(0.5));
    /// if let Some((id, weight)) = index.select() {
    ///     assert_eq!(id, 1);
    ///     assert_eq!(weight, dec!(0.500));
    /// }
    /// ```
    pub fn select(&self) -> Option<(u32, Decimal)> {
        if let Some((id, weight, _)) = self.select_with_digits() {
            return Some((id, weight))
        }
        None
    }

    // Helper function
    fn select_with_digits(&self) -> Option<(u32, Decimal, [u8; MAX_PRECISION])> {
        if self.root.content_count == 0 {
            return None;
        }
        let mut rng = rand::thread_rng();
        let random_target = rng.gen_range(Decimal::ZERO..self.root.accumulated_value);
        let mut digits = [0u8; MAX_PRECISION];
        self.select_recurse(&self.root, random_target, &mut digits, 1)
    }

    /// Recursive helper for the select function.
    fn select_recurse(
        &self,
        node: &Node,
        mut target: Decimal,
        digits: &mut [u8; MAX_PRECISION], // Accumulate digits during traversal
        current_depth: u8,
    ) -> Option<(u32, Decimal, [u8; MAX_PRECISION])> {
        if current_depth > self.precision {
            if let NodeContent::Leaf(bitmap) = &node.content {
                if bitmap.is_empty() {
                    return None;
                }
                let mut rng = rand::thread_rng();
                let rand_index = rng.gen_range(0..bitmap.len() as u32);
                let selected_id = bitmap.select(rand_index).unwrap();
                let weight = node.accumulated_value / Decimal::from(node.content_count);
                return Some((selected_id, weight, digits.clone()));
            }
        }

        if let NodeContent::Internal(children) = &node.content {
            for (i, child) in children.iter().enumerate() {
                if child.accumulated_value.is_zero() {
                    continue;
                }
                if target < child.accumulated_value {
                    digits[current_depth as usize - 1] = i as u8;
                    return self.select_recurse(child, target, digits, current_depth + 1);
                }
                target -= child.accumulated_value;
            }
        }
        None
    }
    

    /// Private helper for finding a unique item using bin-aware rejection sampling.
    /// It performs one weighted traversal and returns a unique item, or None if the
    /// chosen bin is already exhausted.
    fn select_unique(&self, selected_ids: &RoaringBitmap) -> Option<(u32, Decimal, [u8; MAX_PRECISION])> {
        if self.root.content_count == 0 {
            return None;
        }
        let mut rng = rand::thread_rng();
        let random_target = rng.gen_range(Decimal::ZERO..self.root.accumulated_value);
        let mut digits = [0u8; MAX_PRECISION];
        self.select_unique_recurse(&self.root, random_target, &mut digits, 1, selected_ids, &mut rng)
    }

    /// NEW recursive helper for the unique selection process.
    fn select_unique_recurse(
        &self,
        node: &Node,
        mut target: Decimal,
        digits: &mut [u8; MAX_PRECISION],
        current_depth: u8,
        selected_ids: &RoaringBitmap,
        rng: &mut ThreadRng,
    ) -> Option<(u32, Decimal, [u8; MAX_PRECISION])> {
        // Base Case: We've reached a leaf bin.
        if current_depth > self.precision {
            if let NodeContent::Leaf(bitmap) = &node.content {
                let total_in_bin = bitmap.len() as u32;
                if total_in_bin == 0 { return None; }
                let mut attempts = 0;
                while attempts < total_in_bin {  // Bound loops to avoid infinite
                    let rand_index = rng.gen_range(0..total_in_bin);
                    if let Some(candidate_id) = bitmap.select(rand_index) {
                        if !selected_ids.contains(candidate_id) {
                            let weight = node.accumulated_value / Decimal::from(node.content_count);
                            return Some((candidate_id, weight, digits.clone()));
                        }
                    }
                    attempts += 1;
                }
                return None;  // Bin exhausted
            }
        }

        // Recursive Step: Traverse internal nodes.
        if let NodeContent::Internal(children) = &node.content {
            for (i, child) in children.iter().enumerate() {
                if child.accumulated_value.is_zero() { continue; }
                if target < child.accumulated_value {
                    digits[current_depth as usize - 1] = i as u8;                    
                    return self.select_unique_recurse(child, target, digits, current_depth + 1, selected_ids, rng);
                }
                target -= child.accumulated_value;
            }
        }
        None // Should not be reached in a consistent tree
    }    

    // Internal helper
    fn select_many_with_digits(&self, num_to_draw: u32) -> Option<Vec<(u32, Decimal, [u8; MAX_PRECISION])>> {
        if num_to_draw > self.count() {
            return None;
        }
        if num_to_draw == 0 {
            return Some(Vec::new());
        }

        // CHANGED: Use a Vec for storage
        let mut selected = Vec::with_capacity(num_to_draw as usize);
        let mut selected_ids = RoaringBitmap::new();
        
        while selected.len() < num_to_draw as usize {
            // Rejection sampling loop
            if let Some((id, weight, digits)) = self.select_unique(&selected_ids) {
                if selected_ids.insert(id) {
                    // CHANGED: Use .push() which is more efficient
                    selected.push((id, weight, digits));
                }
            }
        }
        Some(selected)
    }

    /// Selects multiple unique items randomly based on weights without removal.
    ///
    /// Uses rejection sampling to ensure uniqueness. Returns `None` if `num_to_draw`
    /// exceeds the number of items in the index.
    ///
    /// # Arguments
    ///
    /// * `num_to_draw` - The number of unique items to select.
    ///
    /// # Returns
    ///
    /// An `Option` containing a vector of selected (ID, weight) pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use digit_bin_index::DigitBinIndex;
    /// use rust_decimal_macros::dec;
    ///
    /// let mut index = DigitBinIndex::new();
    /// index.add(1, dec!(0.3));
    /// index.add(2, dec!(0.7));
    /// if let Some(selected) = index.select_many(2) {
    ///     assert_eq!(selected.len(), 2);
    /// }
    /// ```
    pub fn select_many(&self, num_to_draw: u32) -> Option<Vec<(u32, Decimal)>> {
        let mut selected_without_digits = Vec::with_capacity(num_to_draw as usize);
        if let Some(selected) = self.select_many_with_digits(num_to_draw) {
            selected_without_digits = selected.into_iter().map(|(id, weight, _)| (id, weight)).collect();            
        }
        Some(selected_without_digits)
    }

    /// Selects a single item randomly and removes it from the index.
    ///
    /// Combines selection and removal in one operation. Returns `None` if empty.
    ///
    /// # Returns
    ///
    /// An `Option` containing the selected item's ID and weight.
    ///
    /// # Examples
    ///
    /// ```
    /// use digit_bin_index::DigitBinIndex;
    /// use rust_decimal_macros::dec;
    ///
    /// let mut index = DigitBinIndex::new();
    /// index.add(1, dec!(0.5));
    /// if let Some((id, _)) = index.select_and_remove() {
    ///     assert_eq!(id, 1);
    /// }
    /// assert_eq!(index.count(), 0);
    /// ```
    pub fn select_and_remove(&mut self) -> Option<(u32, Decimal)> {
        if let Some((individual_id, weight, digits)) = self.select_with_digits() {
            self.remove_with_digits(individual_id, weight, digits);
            Some((individual_id, weight))
        } else {
            None
        }
    }

    /// Selects multiple unique items randomly and removes them from the index.
    ///
    /// Selects and removes in batch. Returns `None` if `num_to_draw` exceeds item count.
    ///
    /// # Arguments
    ///
    /// * `num_to_draw` - The number of unique items to select and remove.
    ///
    /// # Returns
    ///
    /// An `Option` containing a vector of selected (ID, weight) pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use digit_bin_index::DigitBinIndex;
    /// use rust_decimal_macros::dec;
    ///
    /// let mut index = DigitBinIndex::new();
    /// index.add(1, dec!(0.3));
    /// index.add(2, dec!(0.7));
    /// if let Some(selected) = index.select_many_and_remove(2) {
    ///     assert_eq!(selected.len(), 2);
    /// }
    /// assert_eq!(index.count(), 0);
    /// ```
    pub fn select_many_and_remove(&mut self, num_to_draw: u32) -> Option<Vec<(u32, Decimal)>> {
        if let Some(selected) = self.select_many_with_digits(num_to_draw) {
            // Iteration works the same for Vec as for HashSet
            for &(individual_id, weight, digits) in &selected {
                self.remove_with_digits(individual_id, weight, digits);
            }
            Some(selected.into_iter().map(|(id, weight, _)| (id, weight)).collect())
        } else {
            None
        }
    }

    /// Returns the total number of items currently in the index.
    ///
    /// # Returns
    ///
    /// The count of items as a `u32`.
    ///
    /// # Examples
    ///
    /// ```
    /// use digit_bin_index::DigitBinIndex;
    ///
    /// let mut index = DigitBinIndex::new();
    /// assert_eq!(index.count(), 0);
    /// ```
    pub fn count(&self) -> u32 {
        self.root.content_count
    }

    /// Returns the sum of all weights in the index.
    ///
    /// This represents the total accumulated probability mass.
    ///
    /// # Returns
    ///
    /// The total weight as a `Decimal`.
    ///
    /// # Examples
    ///
    /// ```
    /// use digit_bin_index::DigitBinIndex;
    /// use rust_decimal_macros::dec;
    ///
    /// let mut index = DigitBinIndex::new();
    /// index.add(1, dec!(0.5));
    /// assert_eq!(index.total_weight(), dec!(0.500));
    /// ```
    pub fn total_weight(&self) -> Decimal {
        self.root.accumulated_value
    }
}

#[cfg(feature = "python-bindings")]
mod python {
    use super::*;
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
            self.index.select_many(n).map(|items| {
                items.into_iter().map(|(id, w)| (id, w.to_string())).collect()
            })
        }

        fn select_and_remove(&mut self) -> Option<(u32, String)> {
            self.index.select_and_remove().map(|(id, weight)| (id, weight.to_string()))
        }

        fn select_many_and_remove(&mut self, n: u32) -> Option<Vec<(u32, String)>> {
            self.index.select_many_and_remove(n).map(|items| {
                items.into_iter().map(|(id, w)| (id, w.to_string())).collect()
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
    fn digit_bin_index(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
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
        assert!(
            index.count() == 3,
            "The count is now {} and not 3 as expected",
            index.count()
        );
        println!("Intermediate state: {} individuals, total weight = {}", index.count(), index.total_weight()); 
        if let Some(selection) = index.select_many_and_remove(2) {
            println!("Selection: {:?}", selection);
        }
        assert!(
            index.count() == 1,
            "The count is now {} and not 1 as expected",
            index.count()
        );
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