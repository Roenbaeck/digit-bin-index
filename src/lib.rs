//! A `DigitBinIndex` is a tree-based data structure that organizes a large
//! collection of weighted items to enable highly efficient weighted random
//! selection and removal.
//!
//! It is a specialized tool, purpose-built for scenarios with millions of
//! items where probabilities are approximate and high performance is critical,
//! particularly for simulations involving sequential sampling like Wallenius'
//! noncentral hypergeometric distribution.

use fraction::{Decimal, Zero}; 
use rand::Rng; 
use std::vec;

// The default precision to use if none is specified in the constructor.
const DEFAULT_PRECISION: u8 = 3;

/// The content of a node, which is either more nodes or a leaf with individuals.
#[derive(Debug, Clone)]
pub enum NodeContent {
    /// An internal node that contains children for the next digit (0-9).
    Internal(Vec<Node>),
    /// A leaf node that contains a list of IDs for individuals in this bin.
    Leaf(Vec<u32>),
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
        let s = weight.to_string();
        if let Some(dot_pos) = s.find('.') {
            let digit_pos = dot_pos + (position as usize);
            if digit_pos < s.len() {
                return s.chars().nth(digit_pos).unwrap().to_digit(10).unwrap() as usize;
            }
        }
        0 // Return 0 if precision is higher than number of decimals.
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
                NodeContent::Leaf(individuals) => individuals.push(individual_id),
                NodeContent::Internal(children) => {
                    if children.is_empty() {
                        node.content = NodeContent::Leaf(vec![individual_id]);
                    } else {
                        panic!("Cannot add individual to a non-empty internal node at leaf depth.");
                    }
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
        
        // --- FIX: Use the modern, fully-qualified call ---
        let mut rng = rand::rng();
        let random_target = Decimal::from(rng.random_range(0.0..self.root.accumulated_value.try_into().unwrap()));
        
        let (selected_id, weight, path) = Self::select_recurse(&mut self.root, random_target, vec![]);
        self.update_values_post_removal(&path, weight);
        Some((selected_id, weight))
    }

    /// Recursive helper to find the individual and record the traversal path.
    fn select_recurse(
        node: &mut Node,
        mut target: Decimal,
        mut path: Vec<usize>,
    ) -> (u32, Decimal, Vec<usize>) {
        match &mut node.content {
            NodeContent::Leaf(individuals) => {
        let mut rng = rand::rng();
                let rand_index = rng.random_range(0..individuals.len());
                let selected_id = individuals.swap_remove(rand_index);
                // The average weight of an item in this bin is the total accumulated value
                // divided by the number of items *before* removal.
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


#[cfg(test)]
mod tests {
    use super::*;
    use fraction::{Decimal};

    #[test]
    fn test_selection_distribution_is_biased_correctly() {
        // --- Setup: Create a controlled population ---
        const ITEMS_PER_GROUP: u32 = 1000;
        const TOTAL_ITEMS: u32 = ITEMS_PER_GROUP * 2;
        const NUM_DRAWS: u32 = TOTAL_ITEMS / 2;

        let low_risk_weight = Decimal::from(0.1); // 0.1
        let high_risk_weight = Decimal::from(0.2); // 0.2

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
}