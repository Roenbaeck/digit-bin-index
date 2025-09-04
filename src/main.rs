use fraction::Decimal;
use rand::{thread_rng, Rng};
use std::vec;

// The default precision to use if none is specified in the constructor.
const DEFAULT_PRECISION: u8 = 3;

// --- Data Structures ---

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
            // Start with no children to save space. We will grow the vector as needed.
            content: NodeContent::Internal(vec![]),
            accumulated_value: Decimal::from(0),
            content_count: 0,
        }
    }
}

/// A data structure that organizes weighted items into bins based on their
/// decimal digits to enable fast weighted random selection and updates.
#[derive(Debug)]
pub struct DigitBinIndex {
    pub root: Node,
    pub precision: u8,
}

impl DigitBinIndex {
    /// Creates a new DigitBinIndex instance with the default precision.
    pub fn new() -> Self {
        Self::with_precision(DEFAULT_PRECISION)
    }

    /// Creates a new DigitBinIndex instance with a specific precision.
    /// The precision determines how many decimal places are used for binning.
    pub fn with_precision(precision: u8) -> Self {
        assert!(precision > 0, "Precision must be at least 1.");
        Self {
            root: Node::new_internal(),
            precision,
        }
    }

    /// Helper function to get the digit at a certain decimal position.
    fn get_digit_at(weight: Decimal, position: u8) -> usize {
        // Using string conversion is robust for decimals.
        let s = weight.to_string();
        if let Some(dot_pos) = s.find('.') {
            let digit_pos = dot_pos + (position as usize);
            if digit_pos < s.len() {
                // Safely parse the character to a digit.
                return s.chars().nth(digit_pos).unwrap().to_digit(10).unwrap() as usize;
            }
        }
        0 // Return 0 if precision is higher than number of decimals.
    }

    /// Adds an individual with a specific weight (probability) to the index.
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

        // If we've reached the desired precision, this path terminates in a leaf.
        if current_depth > max_depth {
            match &mut node.content {
                NodeContent::Leaf(individuals) => individuals.push(individual_id),
                NodeContent::Internal(children) => {
                    // This node was previously internal but is now becoming a leaf.
                    if children.is_empty() {
                        node.content = NodeContent::Leaf(vec![individual_id]);
                    } else {
                        panic!("Cannot add individual to a non-empty internal node at leaf depth.");
                    }
                }
            }
            return;
        }

        // Otherwise, traverse deeper.
        let digit = Self::get_digit_at(weight, current_depth);

        if let NodeContent::Internal(children) = &mut node.content {
            // Ensure the children vector is large enough.
            if children.len() <= digit {
                children.resize_with(digit + 1, Node::new_internal);
            }
            Self::add_recurse(&mut children[digit], individual_id, weight, current_depth + 1, max_depth);
        } else {
            // This case should not be hit if the logic is correct.
             panic!("Attempted to traverse deeper on what should be a leaf node.");
        }
    }

    /// Performs a weighted random selection, removes the item, and returns its ID and weight.
    pub fn select_and_remove(&mut self) -> Option<(u32, Decimal)> {
        if self.root.content_count == 0 {
            return None;
        }

        let mut rng = thread_rng();
        let random_target = rng.gen_range(Decimal::from(0)..self.root.accumulated_value);

        // Select finds the individual and tells us the path taken to get there.
        let (selected_id, weight, path) = Self::select_recurse(&mut self.root, random_target, vec![]);

        // Update the values back up the tree using the recorded path.
        self.update_values_post_removal(&path, weight);

        Some((selected_id, weight))
    }

    /// Recursive helper to find the individual and record the traversal path.
    fn select_recurse(
        node: &mut Node,
        mut target: Decimal,
        mut path: Vec<usize>
    ) -> (u32, Decimal, Vec<usize>) {
        match &mut node.content {
            NodeContent::Leaf(individuals) => {
                let mut rng = thread_rng();
                let rand_index = rng.gen_range(0..individuals.len());
                // Efficiently remove the item by swapping it with the last element.
                let selected_id = individuals.swap_remove(rand_index);
                // Calculate the average weight of items in this bin.
                let weight = node.accumulated_value / Decimal::from(node.content_count + 1);

                return (selected_id, weight, path);
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
                // This panic should ideally not be reached if the accumulated values are correct.
                panic!("Selection logic failed: target value exceeded total accumulated value of children.");
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
                // Stop if we reach a leaf node in the path (should not happen with this logic).
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


fn main() {
    println!("--- Simulation 1: Default Precision (3) ---");
    let mut dbi1 = DigitBinIndex::new();
    dbi1.add(101, Decimal::from_parts(543, 0, 3, false)); // 0.543
    dbi1.add(102, Decimal::from_parts(12, 0, 2, false));  // 0.120
    // Note: 0.12345 will be binned as 0.123 due to precision=3
    dbi1.add(103, Decimal::from_parts(12345, 0, 5, false));
    println!("Initial state: {} individuals, total weight = {}", dbi1.count(), dbi1.total_weight());
    if let Some((id, _)) = dbi1.select_and_remove() {
        println!("Selected ID: {}", id);
    }
    println!("Final state: {} individuals, total weight = {}\n", dbi1.count(), dbi1.total_weight());


    println!("--- Simulation 2: High Precision (5) ---");
    let mut dbi2 = DigitBinIndex::with_precision(5);
    dbi2.add(201, Decimal::from_parts(543, 0, 3, false));    // 0.54300
    dbi2.add(202, Decimal::from_parts(12, 0, 2, false));     // 0.12000
    dbi2.add(203, Decimal::from_parts(12345, 0, 5, false));  // 0.12345
    println!("Initial state: {} individuals, total weight = {}", dbi2.count(), dbi2.total_weight());
     if let Some((id, _)) = dbi2.select_and_remove() {
        println!("Selected ID: {}", id);
    }
    println!("Final state: {} individuals, total weight = {}", dbi2.count(), dbi2.total_weight());
}