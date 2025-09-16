//! A `DigitBinIndex` is a tree-based data structure that organizes a large
//! collection of weighted items to enable highly efficient weighted random
//! selection and removal.
//!
//! It is a specialized tool, purpose-built for scenarios with millions of
//! items where probabilities are approximate and high performance is critical,
//! particularly for simulations involving sequential sampling like Wallenius'
//! noncentral hypergeometric distribution.

use wyrand::WyRand;
use rand::{distr::{Distribution, Uniform}, Rng, SeedableRng}; 
use roaring::{RoaringBitmap, RoaringTreemap};

// The default precision to use if none is specified in the constructor.
const DEFAULT_PRECISION: u8 = 3;
const MAX_PRECISION: usize = 9;

/// Trait for types that can be used as leaf bins in a `DigitBinIndex`.
///
/// Implement this trait for any container you want to use for storing IDs in the leaf nodes.
/// Provided implementations: [`Vec<u32>`], [`RoaringBitmap`].
pub trait DigitBin: Clone + Default {
    fn insert(&mut self, id: u64);
    fn remove(&mut self, id: u64) -> bool;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn get_random(&self, rng: &mut impl rand::Rng) -> Option<u64>;
    fn get_random_and_remove(&mut self, rng: &mut impl rand::Rng) -> Option<u64>;
}

impl DigitBin for Vec<u32> {
    fn insert(&mut self, id: u64) { self.push(id as u32); }
    fn remove(&mut self, id: u64) -> bool {
        if let Some(pos) = self.iter().position(|&x| x == id as u32) {
            self.swap_remove(pos);
            true
        } else {
            false
        }
    }
    fn len(&self) -> usize { self.len() }
    fn is_empty(&self) -> bool { self.is_empty() }
    fn get_random(&self, rng: &mut impl rand::Rng) -> Option<u64> {
        if self.is_empty() { None } else { Some(self[rng.random_range(0..self.len())] as u64) }
    }
    fn get_random_and_remove(&mut self, rng: &mut impl rand::Rng) -> Option<u64> {
        if self.is_empty() { None } else {
            let pos = rng.random_range(0..self.len());
            Some(self.swap_remove(pos) as u64)
        }
    }
}

impl DigitBin for RoaringBitmap {
    fn insert(&mut self, id: u64) { self.insert(id as u32); }
    fn remove(&mut self, id: u64) -> bool { self.remove(id as u32) }
    fn len(&self) -> usize { self.len() as usize }
    fn is_empty(&self) -> bool { self.is_empty() }
    fn get_random(&self, rng: &mut impl rand::Rng) -> Option<u64> {
        if self.is_empty() { None } else {
            let idx = rng.random_range(0..self.len() as u32);
            self.select(idx).map(|v| v as u64)
        }
    }
    fn get_random_and_remove(&mut self, rng: &mut impl rand::Rng) -> Option<u64> {
        if self.is_empty() { None } else {
            let idx = rng.random_range(0..self.len() as u32);
            let selected = self.select(idx);
            self.remove(selected.unwrap());
            selected.map(|v| v as u64)
        }
    }
}

impl DigitBin for RoaringTreemap {
    fn insert(&mut self, id: u64) { self.insert(id); }
    fn remove(&mut self, id: u64) -> bool { self.remove(id) }
    fn len(&self) -> usize { self.len() as usize }
    fn is_empty(&self) -> bool { self.is_empty() }
    fn get_random(&self, rng: &mut impl rand::Rng) -> Option<u64> {
        if self.is_empty() { None } else {
            let idx = rng.random_range(0..self.len() as u64);
            self.select(idx)
        }
    }
    fn get_random_and_remove(&mut self, rng: &mut impl rand::Rng) -> Option<u64> {
        if self.is_empty() { None } else {
            let idx = rng.random_range(0..self.len());
            let selected = self.select(idx);
            self.remove(selected.unwrap());
            selected
        }
    }
}

// Helper to create an array of Option<T>
fn new_children_array<B: DigitBin>() -> Box<[Option<Node<B>>; 10]> {
    // This is a standard way to initialize an array of non-Copy types.
    let data: [Option<Node<B>>; 10] = Default::default();
    Box::new(data)
}

/// The content of a node, which is either more nodes or a leaf with individuals.
#[derive(Debug, Clone)]
pub enum NodeContent<B: DigitBin> {
    /// An internal node that contains children for the next digit (0-9).
    DigitIndex(Box<[Option<Node<B>>; 10]>),
    /// A leaf node that contains a bin of IDs for individuals in this bin.
    Bin(B),
}

/// A node within the DigitBinIndex tree.
#[derive(Debug, Clone)]
pub struct Node<B: DigitBin> {
    /// The content of this node, either more nodes or a list of individual IDs.
    pub content: NodeContent<B>,
    /// The total sum of scaled values stored under this node.
    pub accumulated_value: u64,
    /// The total count of individuals stored under this node.
    pub content_count: u64,
}

impl<B: DigitBin> Node<B> {
    /// Creates a new, empty internal node.
    fn new_internal() -> Self {
        Self {
            content: NodeContent::DigitIndex(new_children_array()), 
            accumulated_value: 0u64,
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
///
/// # Examples
///
/// ```
/// use digit_bin_index::DigitBinIndex;
/// let mut index = DigitBinIndex::with_precision_and_capacity(3, 100);
/// ```
#[derive(Debug, Clone)]
pub enum DigitBinIndex {
    Small(DigitBinIndexGeneric<Vec<u32>>),
    Medium(DigitBinIndexGeneric<RoaringBitmap>),
    Large(DigitBinIndexGeneric<RoaringTreemap>),
}

impl DigitBinIndex {
    /// Creates a new DigitBinIndex with the given precision and expected capacity.
    /// Uses Vec<u32> for small bins, RoaringBitmap for large bins.
    ///
    /// # Arguments
    ///
    /// * `precision` - The number of decimal places for binning (1 to 9).
    /// * `capacity` - The expected number of items to be stored in the index.
    ///
    /// # Returns
    ///
    /// A new `DigitBinIndex` instance with the appropriate bin type.
    ///
    /// # Panics
    ///
    /// Panics if `precision` is 0 or greater than 9.
    ///
    /// # Examples
    ///
    /// ```
    /// use digit_bin_index::DigitBinIndex;
    ///
    /// let index = DigitBinIndex::with_precision_and_capacity(3, 100);
    /// // Uses Vec<u32> because capacity is small
    /// ```
    pub fn with_precision_and_capacity(precision: u8, capacity: u64) -> Self {
        let max_bins = 10u64.pow(precision as u32);
        if capacity / max_bins > 1_000_000_000 {
            // Heuristic: Use RoaringTreemap if average bin size (capacity / 10^precision) exceeds threshold
            DigitBinIndex::Large(DigitBinIndexGeneric::<RoaringTreemap>::with_precision(precision))
        }
        else if capacity / max_bins > 1_000 {
            // Heuristic: Use RoaringBitmap if average bin size (capacity / 10^precision) exceeds threshold
            DigitBinIndex::Medium(DigitBinIndexGeneric::<RoaringBitmap>::with_precision(precision))
        } else {
            // Heuristic: Use Vec<u32> for small average bin sizes
            DigitBinIndex::Small(DigitBinIndexGeneric::<Vec<u32>>::with_precision(precision))
        }
    }

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
    /// assert_eq!(index.precision(), 3);
    /// ```    
    pub fn new() -> Self {
        DigitBinIndex::Small(DigitBinIndexGeneric::<Vec<u32>>::new())
    }

    /// Creates a new `DigitBinIndex` instance with the specified precision.
    ///
    /// The precision determines the number of decimal places used for binning weights.
    /// Higher precision improves sampling accuracy but increases memory usage and tree depth.
    /// Precision must be between 1 and 9 (inclusive).
    ///
    /// # Arguments
    ///
    /// * `precision` - The number of decimal places for binning (1 to 9).
    ///
    /// # Returns
    ///
    /// A new `DigitBinIndex` instance with the given precision.
    ///
    /// # Panics
    ///
    /// Panics if `precision` is 0 or greater than 9.
    ///
    /// # Examples
    ///
    /// ```
    /// use digit_bin_index::DigitBinIndex;
    ///
    /// let index = DigitBinIndex::with_precision(4);
    /// assert_eq!(index.precision(), 4);
    /// ```
    pub fn with_precision(precision: u8) -> Self {
        DigitBinIndex::Small(DigitBinIndexGeneric::<Vec<u32>>::with_precision(precision))
    }

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
    ///
    /// let mut index = DigitBinIndex::new();
    /// let added = index.add(1, 0.5);
    /// assert_eq!(index.count(), 1);
    /// ```    
    pub fn add(&mut self, id: u64, weight: f64) {
        match self {
            DigitBinIndex::Small(index) => index.add(id, weight),
            DigitBinIndex::Medium(index) => index.add(id, weight),
            DigitBinIndex::Large(index) => index.add(id, weight),
        }
    }

    /// Adds multiple items to the index in a highly optimized batch operation.
    ///
    /// This method is significantly faster than calling `add` in a loop for large
    /// collections of items. It works by pre-processing the input, grouping items
    /// by their shared weight, and then propagating each group through the tree in
    /// a single pass. This minimizes cache misses and reduces function call overhead.
    ///
    /// Weights are rescaled to the index's precision and binned accordingly.
    /// Items with non-positive weights or weights that become zero after scaling
    /// will be ignored.
    ///
    /// # Arguments
    ///
    /// * `items` - A slice of `(id, weight)` tuples to add to the index.
    ///
    /// # Examples
    ///
    /// ```
    /// use digit_bin_index::DigitBinIndex;
    ///
    /// let mut index = DigitBinIndex::new();
    /// let items_to_add = vec![(1, 0.1), (2, 0.2), (3, 0.1)];
    /// index.add_many(&items_to_add);
    ///
    /// assert_eq!(index.count(), 3);
    /// // The total weight should be 0.1 + 0.2 + 0.1 = 0.4
    /// assert!((index.total_weight() - 0.4).abs() < f64::EPSILON);
    /// ```
    pub fn add_many(&mut self, items: &[(u64, f64)]) {
        match self {
            DigitBinIndex::Small(index) => index.add_many(items),
            DigitBinIndex::Medium(index) => index.add_many(items),
            DigitBinIndex::Large(index) => index.add_many(items),
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
    ///
    /// let mut index = DigitBinIndex::new();
    /// index.add(1, 0.5);
    /// index.remove(1, 0.5);
    /// assert_eq!(index.count(), 0);
    /// ```
    pub fn remove(&mut self, id: u64, weight: f64) -> bool {
        match self {
            DigitBinIndex::Small(index) => index.remove(id, weight),
            DigitBinIndex::Medium(index) => index.remove(id, weight),
            DigitBinIndex::Large(index) => index.remove(id, weight),
        }
    }

    /// Removes multiple items from the index in a highly optimized batch operation.
    ///
    /// This method is significantly faster than calling `remove` in a loop. It
    /// groups the items to be removed by their weight path and traverses the tree
    /// only once per group, performing aggregated updates on the way up.
    ///
    /// The `(id, weight)` pairs must match items that are currently in the index.
    /// If a given pair is not found, it is silently ignored.
    ///
    /// # Arguments
    ///
    /// * `items` - A slice of `(id, weight)` tuples to remove from the index.
    ///
    /// # Examples
    ///
    /// ```
    /// use digit_bin_index::DigitBinIndex;
    ///
    /// let mut index = DigitBinIndex::new();
    /// let items_to_add = vec![(1, 0.1), (2, 0.2), (3, 0.1), (4, 0.3)];
    /// index.add_many(&items_to_add);
    /// assert_eq!(index.count(), 4);
    ///
    /// let items_to_remove = vec![(2, 0.2), (3, 0.1)];
    /// index.remove_many(&items_to_remove);
    ///
    /// assert_eq!(index.count(), 2); // Items 1 and 4 should remain
    /// // The total weight should be 0.1 + 0.3 = 0.4
    /// assert!((index.total_weight() - 0.4).abs() < f64::EPSILON);
    /// ```
    pub fn remove_many(&mut self, items: &[(u64, f64)]) -> bool {
        match self {
            DigitBinIndex::Small(index) => index.remove_many(items),
            DigitBinIndex::Medium(index) => index.remove_many(items),
            DigitBinIndex::Large(index) => index.remove_many(items),
        }
    }    

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
    ///
    /// let mut index = DigitBinIndex::new();
    /// index.add(1, 0.5);
    /// if let Some((id, weight)) = index.select() {
    ///     assert_eq!(id, 1);
    ///     assert_eq!(weight, 0.5);
    /// }
    /// ```
    pub fn select(&mut self) -> Option<(u64, f64)> {
        match self {
            DigitBinIndex::Small(index) => index.select(),
            DigitBinIndex::Medium(index) => index.select(),
            DigitBinIndex::Large(index) => index.select(),
        }
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
    ///
    /// let mut index = DigitBinIndex::new();
    /// index.add(1, 0.5);
    /// if let Some((id, _)) = index.select_and_remove() {
    ///     assert_eq!(id, 1);
    /// }
    /// assert_eq!(index.count(), 0);
    /// ```
    pub fn select_and_remove(&mut self) -> Option<(u64, f64)> {
        match self {
            DigitBinIndex::Small(index) => index.select_and_remove(),
            DigitBinIndex::Medium(index) => index.select_and_remove(),
            DigitBinIndex::Large(index) => index.select_and_remove(),
        }
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
    ///
    /// let mut index = DigitBinIndex::new();
    /// index.add(1, 0.3);
    /// index.add(2, 0.7);
    /// if let Some(selected) = index.select_many(2) {
    ///     assert_eq!(selected.len(), 2);
    /// }
    /// ```
    pub fn select_many(&mut self, num_to_draw: u64) -> Option<Vec<(u64, f64)>> {
        match self {
            DigitBinIndex::Small(index) => index.select_many(num_to_draw),
            DigitBinIndex::Medium(index) => index.select_many(num_to_draw),
            DigitBinIndex::Large(index) => index.select_many(num_to_draw),
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
    ///
    /// let mut index = DigitBinIndex::new();
    /// index.add(1, 0.3);
    /// index.add(2, 0.7);
    /// if let Some(selected) = index.select_many_and_remove(2) {
    ///     assert_eq!(selected.len(), 2);
    /// }
    /// assert_eq!(index.count(), 0);
    /// ```
    pub fn select_many_and_remove(&mut self, num_to_draw: u64) -> Option<Vec<(u64, f64)>> {
        match self {
            DigitBinIndex::Small(index) => index.select_many_and_remove(num_to_draw),
            DigitBinIndex::Medium(index) => index.select_many_and_remove(num_to_draw),
            DigitBinIndex::Large(index) => index.select_many_and_remove(num_to_draw),
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
    pub fn count(&self) -> u64 {
        match self {
            DigitBinIndex::Small(index) => index.count(),
            DigitBinIndex::Medium(index) => index.count(),
            DigitBinIndex::Large(index) => index.count(),
        }
    }

    /// Returns the sum of all weights in the index.
    ///
    /// This represents the total accumulated probability mass.
    ///
    /// # Returns
    ///
    /// The total weight as a `f64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use digit_bin_index::DigitBinIndex;
    ///
    /// let mut index = DigitBinIndex::new();
    /// index.add(1, 0.5);
    /// assert_eq!(index.total_weight(), 0.5);
    /// ```
    pub fn total_weight(&self) -> f64 {
        match self {
            DigitBinIndex::Small(index) => index.total_weight(),
            DigitBinIndex::Medium(index) => index.total_weight(),
            DigitBinIndex::Large(index) => index.total_weight(),
        }
    }

    /// Prints detailed statistics about the index's structure, memory usage,
    /// and data distribution.
    pub fn print_stats(&self) {
        println!("DigitBinIndex Statistics:");
        println!("=========================");
        match self {
            DigitBinIndex::Small(idx) => {
                println!("- Index Type: Small (Vec<u32>)");
                idx.print_stats_generic();
            },
            DigitBinIndex::Medium(idx) => {
                println!("- Index Type: Medium (RoaringBitmap)");
                idx.print_stats_generic();
            },
            DigitBinIndex::Large(idx) => {
                println!("- Index Type: Large (RoaringTreemap)");
                idx.print_stats_generic();
            },
        }
    }

    /// Returns the precision (number of decimal places) used for binning.
    pub fn precision(&self) -> u8 {
        match self {
            DigitBinIndex::Small(idx) => idx.precision,
            DigitBinIndex::Medium(idx) => idx.precision,
            DigitBinIndex::Large(idx) => idx.precision,
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
///
/// # Type Parameters
///
/// * `B` - The bin container type for leaf nodes. Must implement the [`DigitBin`] trait.
///   Common choices are [`Vec<u32>`] for maximum speed with small bins, or [`RoaringBitmap`]
///   for memory efficiency with large, sparse bins.
///
/// # Examples
///
/// ```
/// use digit_bin_index::DigitBinIndexGeneric;
/// // Use Vec<u32> for leaf bins
/// let mut index = DigitBinIndexGeneric::<Vec<u32>>::new();
/// // Or use RoaringBitmap for leaf bins
/// // let mut index = DigitBinIndexGeneric::<roaring::RoaringBitmap>::new();
/// ```
#[derive(Debug, Clone)]
pub struct DigitBinIndexGeneric<B: DigitBin> {
    /// The root node of the tree.
    pub root: Node<B>,
    /// The precision (number of decimal places) used for binning.
    pub precision: u8,
    /// The scaling factor (10^precision) as f64 for conversions.
    scale: f64,
}

impl<B: DigitBin> Default for DigitBinIndexGeneric<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: DigitBin> DigitBinIndexGeneric<B> {
    #[must_use]
    pub fn new() -> Self {
        Self::with_precision(DEFAULT_PRECISION)
    }

    #[must_use]
    pub fn with_precision(precision: u8) -> Self {
        assert!(precision > 0, "Precision must be at least 1.");
        assert!(precision <= MAX_PRECISION as u8, "Precision cannot be larger than {}.", MAX_PRECISION);
        Self {
            root: Node::new_internal(),
            precision,
            scale: 10f64.powi(precision as i32),
        }        
    }

    /// Converts a f64 weight to an array of digits [0-9] for the given precision and the scaled u64 value.
    /// Returns None if the weight is invalid (non-positive or zero after scaling).
    fn weight_to_digits(&self, weight: f64, digits: &mut [u8; MAX_PRECISION]) -> Option<u64> {
        if weight <= 0.0 {
            return None;
        }

        let scaled_f = weight * self.scale;
        let scaled = scaled_f.round() as u64;
        if scaled == 0 {
            return None;
        }

        let mut temp = scaled;
        for i in (0..self.precision as usize).rev() {
            digits[i] = (temp % 10) as u8;
            temp /= 10;
        }
        if temp != 0 {
            // Overflow in scaling, shouldn't happen for weight < 1
            return None;
        }
        Some(scaled)
    }

    // --- Standard Functions ---

    pub fn add(&mut self, individual_id: u64, weight: f64) {
        let mut digits = [0u8; MAX_PRECISION];
        if let Some(scaled) = self.weight_to_digits(weight, &mut digits) {
            Self::add_recurse(&mut self.root, individual_id, scaled, &digits, 1, self.precision)
        }
    }

    /// Recursive private method to handle adding individuals.
    fn add_recurse(
        node: &mut Node<B>,
        individual_id: u64,
        scaled: u64, // Scaled weight as u64
        digits: &[u8; MAX_PRECISION],
        current_depth: u8,
        max_depth: u8,
    ) {
        node.content_count += 1;
        node.accumulated_value += scaled;

        if current_depth > max_depth {
            if let NodeContent::DigitIndex(_) = &node.content {
                node.content = NodeContent::Bin(B::default());
            }
            if let NodeContent::Bin(bin) = &mut node.content {
                bin.insert(individual_id);
            }
            return;
        }

        let digit = digits[current_depth as usize - 1] as usize;
        if let NodeContent::DigitIndex(children) = &mut node.content {
            // Get the child, creating it if it doesn't exist.
            let child_node = children[digit].get_or_insert_with(Node::new_internal);
            Self::add_recurse(child_node, individual_id, scaled, digits, current_depth + 1, max_depth);
        }
    }

    /// Adds multiple items to the index in a highly optimized batch operation.
    ///
    /// This method is significantly faster than calling `add` in a loop for large
    /// collections of items. It works by first pre-processing the input items,
    /// grouping them by their shared weight path (e.g., all items with weight 0.123...).
    /// It then traverses the tree once per group, rather than once per item,
    /// drastically reducing function call overhead and improving CPU cache performance
    /// by performing aggregated updates at each node.
    ///
    /// # Arguments
    ///
    /// * `items` - A slice of `(individual_id, weight)` tuples to add to the index.
    ///
    pub fn add_many(&mut self, items: &[(u64, f64)]) {
        if items.is_empty() {
            return;
        }

        let mut digits = [0u8; MAX_PRECISION];
        for &(id, weight) in items {
            if let Some(scaled) = self.weight_to_digits(weight, &mut digits) {
                Self::add_recurse(&mut self.root, id, scaled, &digits, 1, self.precision)
            } 
        }
    }

    pub fn remove(&mut self, individual_id: u64, weight: f64) -> bool{
        let mut digits = [0u8; MAX_PRECISION];
        if let Some(scaled) = self.weight_to_digits(weight, &mut digits) {
            return Self::remove_recurse(&mut self.root, individual_id, scaled, &digits, 1, self.precision);
        }
        false
    }

    /// Recursive private method to handle removing individuals.
    fn remove_recurse(
        node: &mut Node<B>,
        individual_id: u64,
        scaled: u64,
        digits: &[u8; MAX_PRECISION],
        current_depth: u8,
        max_depth: u8,
    ) -> bool {
        if current_depth > max_depth {
            if let NodeContent::Bin(bin) = &mut node.content {
                let orig_len = bin.len();
                bin.remove(individual_id);
                if bin.len() < orig_len {
                    node.content_count -= 1;
                    node.accumulated_value -= scaled;
                    return true;
                }
            }
            return false;
        }

        let digit = digits[current_depth as usize - 1] as usize;
        if let NodeContent::DigitIndex(children) = &mut node.content {
            // Check if the child at 'digit' exists and get a mutable reference to it.
            if let Some(child_node) = children[digit].as_mut() {
                // If it exists, recurse. If the recursion returns true (success)...
                if Self::remove_recurse(child_node, individual_id, scaled, digits, current_depth + 1, max_depth) {
                    // ...then update this node's stats and propagate the success upwards.
                    node.content_count -= 1;
                    node.accumulated_value -= scaled;
                    return true;
                }
            }
        }
        false
    }

    /// Removes multiple items from the index in a highly optimized batch operation.
    ///
    /// This method is significantly faster than calling `remove` in a loop. It
    /// groups the items to be removed by their weight path and traverses the tree
    /// only once per group, performing aggregated updates on the way up.
    ///
    /// The `(id, weight)` pairs must match items that are currently in the index.
    /// If a given pair is not found, it is silently ignored.
    ///
    /// # Arguments
    ///
    /// * `items` - A slice of `(id, weight)` tuples to remove from the index.
    ///
    pub fn remove_many(&mut self, items: &[(u64, f64)]) -> bool {
        if items.is_empty() {
            return false;
        }

        let mut digits = [0u8; MAX_PRECISION];
        let mut success = true;
        for &(id, weight) in items {
            if let Some(scaled) = self.weight_to_digits(weight, &mut digits) {
                success &= Self::remove_recurse(&mut self.root, id, scaled, &digits, 1, self.precision)
            } else {
                success &= false;                
            }
        }
        success
    }

    // --- Selection Functions ---

    pub fn select(&mut self) -> Option<(u64, f64)> {
        self.select_and_optionally_remove(false)
    }

    pub fn select_many(&mut self, num_to_draw: u64) -> Option<Vec<(u64, f64)>> {
        self.select_many_and_optionally_remove(num_to_draw, false)
    }

    pub fn select_and_remove(&mut self) -> Option<(u64, f64)> {
        self.select_and_optionally_remove(true)
    }

    // Wrapper function to handle both select and select_and_remove
    pub fn select_and_optionally_remove(&mut self, with_removal: bool) -> Option<(u64, f64)> {
        if self.root.content_count == 0 {
            return None;
        }
        let mut rng = WyRand::from_os_rng();
        let random_target = rng.random_range(0u64..self.root.accumulated_value);
        Self::select_and_optionally_remove_recurse(&mut self.root, random_target, 1, self.precision, &mut rng, with_removal, self.scale)
    }

    // Helper function
    fn select_and_optionally_remove_recurse(
        node: &mut Node<B>,
        target: u64,
        current_depth: u8,
        max_depth: u8,
        rng: &mut WyRand,
        with_removal: bool,
        scale: f64,
    ) -> Option<(u64, f64)> {
        // Base case: Bin node
        if current_depth > max_depth {
            if let NodeContent::Bin(bin) = &mut node.content {
                if bin.is_empty() {
                    return None;
                }
                let scaled_weight = node.accumulated_value / node.content_count as u64;
                let weight = scaled_weight as f64 / scale;
                let selected_id = if with_removal {
                    bin.get_random_and_remove(rng)?
                } else {
                    bin.get_random(rng)?
                };
                if with_removal {
                    node.content_count -= 1;
                    node.accumulated_value -= scaled_weight;
                }
                return Some((selected_id, weight));
            }
            return None;
        }

        // Recursive case: DigitIndex node
        if let NodeContent::DigitIndex(children) = &mut node.content {
            let mut cum: u64 = 0;
            // The iterator now gives us a mutable reference to the Option.
            for child_option in children.iter_mut() {
                // We pattern match to see if a child Node exists.
                if let Some(child) = child_option.as_mut() {
                    // Now, 'child' is a &mut Node<B>, and we can proceed with the original logic.
                    if child.accumulated_value == 0 {
                        continue;
                    }
                    if target < cum + child.accumulated_value {
                        if let Some((selected_id, weight)) = Self::select_and_optionally_remove_recurse(
                            child,
                            target - cum,
                            current_depth + 1,
                            max_depth,
                            rng,
                            with_removal,
                            scale,
                        ) {
                            if with_removal {
                                node.content_count -= 1;
                                node.accumulated_value -= (weight * scale).round() as u64;
                            }
                            return Some((selected_id, weight));
                        }
                        // This path is taken if recursion fails, which implies an empty bin was selected.
                        return None; 
                    }
                    cum += child.accumulated_value;
                }
            }
        }
        None
    } 

    pub fn select_many_and_remove(&mut self, num_to_draw: u64) -> Option<Vec<(u64, f64)>> {
        self.select_many_and_optionally_remove(num_to_draw, true)
    }

    // Wrapper function to handle both select_many and select_many_and_remove
    pub fn select_many_and_optionally_remove(&mut self, num_to_draw: u64, with_removal: bool) -> Option<Vec<(u64, f64)>> {
        if num_to_draw > self.count() || num_to_draw == 0 {
            return if num_to_draw == 0 { Some(Vec::new()) } else { None };
        }
        let mut rng = WyRand::from_os_rng();
        let mut selected: Vec<(u64, f64)> = Vec::with_capacity(num_to_draw as usize);
        let total_accum = self.root.accumulated_value;
        // Create a Uniform distribution for the range [0, total_accum)
        let uniform = Uniform::new(0u64, total_accum).expect("Valid range for Uniform");  
        // Generate num_to_draw random numbers using sample_iter
        let passed_targets: Vec<u64> = uniform
            .sample_iter(&mut rng)
            .take(num_to_draw as usize)
            .collect();
        Self::select_many_and_optionally_remove_recurse(
            &mut self.root,
            total_accum,
            &mut selected,
            &mut rng,
            1,
            self.precision,
            with_removal,
            passed_targets,
            self.scale,
        );
        if selected.len() == num_to_draw as usize {
            Some(selected)
        } else {
            None // Should not happen if logic is correct
        }
    }

    /// Recursive helper for batch selection and removal.
    /// - node: Current subtree root.
    /// - subtree_total: Accumulated value of this node (passed to avoid borrowing issues).
    /// - selected: Mutable vec to collect (id, weight) from leaves.
    /// - rng: Mutable RNG.
    /// - current_depth: Current digit level.
    /// - precision: The precision of the DigitBinIndex (passed explicitly).
    /// - with_removal: Whether to remove selected items.
    /// - passed_targets: Pre-computed relative targets from parent (in [0, subtree_total)).
    /// - scale: The scaling factor for weight conversions.
    fn select_many_and_optionally_remove_recurse(
        node: &mut Node<B>,
        subtree_total: u64,
        selected: &mut Vec<(u64, f64)>,
        rng: &mut WyRand,
        current_depth: u8,
        precision: u8,
        with_removal: bool,
        passed_targets: Vec<u64>,
        scale: f64,
    ) {
        let original_target_count = passed_targets.len() as u64;
        if original_target_count == 0 {
            return;
        }

        // This base case (leaf node) logic does not change, as it doesn't interact
        // with the DigitIndex.
        if current_depth > precision {
            if let NodeContent::Bin(bin) = &mut node.content {
                let bin_scaled = if node.content_count > 0 {
                    node.accumulated_value / node.content_count as u64
                } else {
                    0u64
                };
                let bin_weight = bin_scaled as f64 / scale;
                let to_select = original_target_count.min(node.content_count);
                let mut picked = 0u64;
                while picked < to_select && !bin.is_empty() {
                    let id = if with_removal {
                        bin.get_random_and_remove(rng).unwrap()
                    } else {
                        bin.get_random(rng).unwrap()
                    };
                    selected.push((id, bin_weight));
                    picked += 1;
                }
                if with_removal {
                    node.content_count -= picked;
                    node.accumulated_value -= bin_scaled * picked as u64;
                }
            }
            return;
        }

        // --- START OF MODIFIED LOGIC ---
        if let NodeContent::DigitIndex(children) = &mut node.content {
            // CHANGE: Use fixed-size arrays of length 10 instead of dynamically sized Vecs.
            let mut child_assigned = [0u64; 10];
            // Note: `Default::default()` works for arrays where the element type is `Default`.
            let mut child_rel_targets: [Vec<u64>; 10] = Default::default();
            let mut assigned = 0u64;

            // --- Main assignment loop ---
            for &target in &passed_targets {
                let mut cum: u64 = 0;
                let mut chosen_idx = None;
                // CHANGE: Iterate over the array of Options.
                for (i, child_option) in children.iter().enumerate() {
                    // CHANGE: Only process existing children.
                    if let Some(child) = child_option {
                        if child.accumulated_value == 0 {
                            continue;
                        }
                        if target < cum + child.accumulated_value {
                            if child_assigned[i] + 1 <= child.content_count {
                                chosen_idx = Some(i);
                            }
                            break;
                        }
                        cum += child.accumulated_value;
                    }
                }
                if let Some(idx) = chosen_idx {
                    child_assigned[idx] += 1;
                    // We need to re-calculate `cum` up to the chosen index to get the relative target.
                    let start_of_child_range: u64 = children[..idx].iter().filter_map(|c| c.as_ref()).map(|c| c.accumulated_value).sum();
                    let rel_target = target - start_of_child_range;
                    child_rel_targets[idx].push(rel_target);
                    assigned += 1;
                }
            }

            // --- Rejection sampling for any remaining targets ---
            let remaining = original_target_count - assigned;
            let mut additional_assigned = 0u64;
            while additional_assigned < remaining {
                let target = rng.random_range(0u64..subtree_total);
                let mut cum: u64 = 0;
                let mut chosen_idx = None;
                // CHANGE: Same iteration pattern as the loop above.
                for (i, child_option) in children.iter().enumerate() {
                    if let Some(child) = child_option {
                        if child.accumulated_value == 0 {
                            continue;
                        }
                        if target < cum + child.accumulated_value {
                            if child_assigned[i] + 1 <= child.content_count {
                                chosen_idx = Some(i);
                            }
                            break;
                        }
                        cum += child.accumulated_value;
                    }
                }
                if let Some(idx) = chosen_idx {
                    child_assigned[idx] += 1;
                    let start_of_child_range: u64 = children[..idx].iter().filter_map(|c| c.as_ref()).map(|c| c.accumulated_value).sum();
                    let rel_target = target - start_of_child_range;
                    child_rel_targets[idx].push(rel_target);
                    additional_assigned += 1;
                }
            }
            
            // CHANGE: Store accumulated values in a fixed-size array for the recursive calls.
            let child_accums: [u64; 10] = std::array::from_fn(|i| {
                children[i].as_ref().map_or(0, |c| c.accumulated_value)
            });

            // --- Recurse into children ---
            // CHANGE: Iterate through mutable options.
            for (i, child_option) in children.iter_mut().enumerate() {
                let assign_count = child_assigned[i];
                if assign_count > 0 {
                    // We must have a child here if it was assigned targets.
                    if let Some(child) = child_option {
                        let rel_targets = std::mem::take(&mut child_rel_targets[i]);
                        Self::select_many_and_optionally_remove_recurse(
                            child,
                            child_accums[i],
                            selected,
                            rng,
                            current_depth + 1,
                            precision,
                            with_removal,
                            rel_targets,
                            scale,
                        );
                    }
                }
            }

            if with_removal {
                // --- Unwind: Update this node's stats ---
                // CHANGE: Sum up counts and values from the existing children in the array.
                node.content_count = children.iter().filter_map(|c| c.as_ref()).map(|c| c.content_count).sum();
                node.accumulated_value = children.iter().filter_map(|c| c.as_ref()).map(|c| c.accumulated_value).sum();
            }
        }
    }

    pub fn count(&self) -> u64 {
        self.root.content_count
    }

    pub fn total_weight(&self) -> f64 {
        self.root.accumulated_value as f64 / self.scale
    }

    /// Prints detailed statistics about the tree: node count, bin stats, and weight stats.
    pub fn print_stats_generic(&self) {
        // This struct holds all the metrics we want to collect.
        struct Stats {
            node_count: usize,
            non_empty_node_count: usize,
            internal_node_count: usize, // NEW: For branching factor
            child_slots_used: usize,    // NEW: For branching factor
            bin_count: usize,
            empty_bin_count: usize,
            total_bin_items: u64,
            min_weight: Option<f64>,
            max_weight: Option<f64>,
            // We collect all bin sizes to calculate standard deviation later.
            bin_sizes: Vec<usize>, 
            // Memory estimates
            mem_nodes: usize,
            mem_bins: usize,
        }

        fn traverse<B: DigitBin>(
            node: &Node<B>,
            stats: &mut Stats,
            scale: f64,
        ) {
            stats.node_count += 1;
            stats.mem_nodes += std::mem::size_of::<Node<B>>();

            if node.content_count > 0 {
                stats.non_empty_node_count += 1;
            }
            
            match &node.content {
                NodeContent::DigitIndex(children) => {
                    // --- NEW: Calculate branching factor stats ---
                    stats.internal_node_count += 1;
                    let used_children = children.iter().filter(|c| c.is_some()).count();
                    stats.child_slots_used += used_children;
                    // --- END NEW ---

                    // Add memory for the heap-allocated array of 10 optional nodes.
                    stats.mem_nodes += std::mem::size_of::<[Option<Node<B>>; 10]>();
                    
                    // Iterate over the options in the array
                    for child_option in children.iter() {
                        // Only recurse into the children that actually exist (are Some)
                        if let Some(child) = child_option {
                            traverse(child, stats, scale);
                        }
                    }
                }
                NodeContent::Bin(bin) => {
                    stats.bin_count += 1;
                    let bin_size = bin.len();
                    stats.bin_sizes.push(bin_size);
                    stats.total_bin_items += bin_size as u64;

                    // Estimate memory for the bin's contents.
                    // This is an approximation. For RoaringBitmap, `serialized_size()` would be more accurate.
                    stats.mem_bins += bin_size * std::mem::size_of::<u32>();

                    if bin_size == 0 {
                        stats.empty_bin_count += 1;
                    } else {
                        // All items in a bin share the same weight.
                        let scaled_weight = node.accumulated_value / node.content_count;
                        let weight = scaled_weight as f64 / scale;
                        stats.min_weight = Some(stats.min_weight.map_or(weight, |min| min.min(weight)));
                        stats.max_weight = Some(stats.max_weight.map_or(weight, |max| max.max(weight)));
                    }
                }
            }
        }

        let mut stats = Stats {
            node_count: 0,
            non_empty_node_count: 0,
            internal_node_count: 0, // NEW
            child_slots_used: 0,    // NEW
            bin_count: 0,
            empty_bin_count: 0,
            total_bin_items: 0,
            min_weight: None,
            max_weight: None,
            bin_sizes: Vec::new(),
            mem_nodes: 0,
            mem_bins: 0,
        };

        traverse(&self.root, &mut stats, self.scale);
        
        // --- Calculations ---
        let fill_ratio = if stats.node_count > 0 {
            stats.non_empty_node_count as f64 / stats.node_count as f64 * 100.0
        } else { 0.0 };

        // NEW: Calculate average branching factor
        let avg_branching_factor = if stats.internal_node_count > 0 {
            stats.child_slots_used as f64 / stats.internal_node_count as f64
        } else { 0.0 };
        
        let avg_bin_size = if stats.bin_count > 0 {
            stats.total_bin_items as f64 / stats.bin_count as f64
        } else { 0.0 };

        let std_dev_bin_size = if stats.bin_count > 1 {
            let variance = stats.bin_sizes.iter()
                .map(|&size| (size as f64 - avg_bin_size).powi(2)) // Corrected: removed the `*`
                .sum::<f64>() / (stats.bin_count - 1) as f64;
            variance.sqrt()
        } else { 0.0 };

        // NEW: Calculate bin size quartiles
        let (q1_bin_size, median_bin_size, q3_bin_size) = if !stats.bin_sizes.is_empty() {
            let mut sorted_sizes = stats.bin_sizes.clone();
            sorted_sizes.sort_unstable();
            let q1 = sorted_sizes.get(sorted_sizes.len() / 4).cloned().unwrap_or(0);
            let median = sorted_sizes.get(sorted_sizes.len() / 2).cloned().unwrap_or(0);
            let q3 = sorted_sizes.get(sorted_sizes.len() * 3 / 4).cloned().unwrap_or(0);
            (q1, median, q3)
        } else {
            (0, 0, 0)
        };
        
        let total_mem_mb = (stats.mem_nodes + stats.mem_bins) as f64 / (1024.0 * 1024.0);
        let nodes_mem_mb = stats.mem_nodes as f64 / (1024.0 * 1024.0);
        let bins_mem_mb = stats.mem_bins as f64 / (1024.0 * 1024.0);
        
        // NEW: Calculate average weight
        let avg_weight = if self.count() > 0 {
            self.total_weight() / self.count() as f64
        } else { 0.0 };
        

        // --- Printing ---
        println!("\n[Tree Structure]");
        println!("- Total Nodes Created:  {}", stats.node_count);
        println!("- Internal Nodes:       {}", stats.internal_node_count); // NEW
        println!("- Avg Branching Factor: {:.2} / 10", avg_branching_factor); // NEW
        println!("- Tree Fill Ratio:      {:.2}%", fill_ratio);
        println!("- Max Depth:            {}", self.precision);

        println!("\n[Memory (Estimated)]");
        println!("- Tree Structure:       {:.2} MB", nodes_mem_mb);
        println!("- Leaf Bins:            {:.2} MB", bins_mem_mb);
        println!("- Total Estimated:      {:.2} MB", total_mem_mb);
 
        println!("\n[Items & Bins]");
        println!("- Total Items:          {}", stats.total_bin_items);
        println!("- Total Bins (Leaves):  {}", stats.bin_count);
        println!("- Empty Bins:           {}", stats.empty_bin_count);
        println!("- Avg Items per Bin:    {:.2}", avg_bin_size);
        println!("- Std Dev of Bin Size:  {:.2}", std_dev_bin_size);
        println!("- Bin Size (min/max):   {} / {}", stats.bin_sizes.iter().min().map_or(0, |v| *v), stats.bin_sizes.iter().max().map_or(0, |v| *v));
        println!("- Bin Size (Q1/Med/Q3): {} / {} / {}", q1_bin_size, median_bin_size, q3_bin_size); // NEW
        
        println!("\n[Weights]");
        println!("- Smallest Weight:      {}", stats.min_weight.map_or("-".to_string(), |v| format!("{:.prec$}", v, prec = self.precision as usize)));
        println!("- Largest Weight:       {}", stats.max_weight.map_or("-".to_string(), |v| format!("{:.prec$}", v, prec = self.precision as usize)));
        println!("- Average Weight:       {:.prec$}", avg_weight, prec = self.precision as usize); // NEW
    }
}

#[cfg(feature = "python-bindings")]
mod python {
    use super::*;
    use pyo3::prelude::*;

    #[pyclass(name = "DigitBinIndex")]
    struct PyDigitBinIndex {
        index: DigitBinIndex,
    }

    #[pymethods]
    impl PyDigitBinIndex {
        #[new]
        fn new() -> Self {
            PyDigitBinIndex {
                index: DigitBinIndex::new(),
            }
        }

        /// Create a DigitBinIndex with a specific precision.
        #[staticmethod]
        fn with_precision(precision: u64) -> Self {
            PyDigitBinIndex {
                index: DigitBinIndex::with_precision(precision.try_into().unwrap()),
            }
        }

        /// Create a DigitBinIndex with a specific precision and expected capacity.
        #[staticmethod]
        fn with_precision_and_capacity(precision: u8, capacity: u64) -> Self {
            PyDigitBinIndex {
                index: DigitBinIndex::with_precision_and_capacity(precision, capacity),
            }
        }        

        fn add(&mut self, id: u64, weight: f64) {
            self.index.add(id, weight)
        }

        fn add_many(&mut self, items: Vec<(u64, f64)>) {
            self.index.add_many(&items);
        }

        fn remove(&mut self, id: u64, weight: f64) -> bool {
            self.index.remove(id, weight)
        }

        fn remove_many(&mut self, items: Vec<(u64, f64)>) -> bool {
            self.index.remove_many(&items)
        }        

        fn select(&mut self) -> Option<(u64, f64)> {
            self.index.select()
        }

        fn select_many(&mut self, n: u64) -> Option<Vec<(u64, f64)>> {
            self.index.select_many(n)
        }

        fn select_and_remove(&mut self) -> Option<(u64, f64)> {
            self.index.select_and_remove()
        }

        fn select_many_and_remove(&mut self, n: u64) -> Option<Vec<(u64, f64)>> {
            self.index.select_many_and_remove(n)
        }

        fn total_weight(&self) -> f64 {
            self.index.total_weight()
        }

        fn count(&self) -> u64 {
            self.index.count()
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

    #[test]
    fn test_select_and_remove() {
        let mut index = DigitBinIndex::with_precision(3);
        index.add(1, 0.085);
        index.add(2, 0.205);
        index.add(3, 0.346);
        index.add(4, 0.364);
        index.print_stats();
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
        const ITEMS_PER_GROUP: u64 = 1000;
        const TOTAL_ITEMS: u64 = ITEMS_PER_GROUP * 2;
        const NUM_DRAWS: u64 = TOTAL_ITEMS / 2;

        let low_risk_weight = 0.1f64;  // 0.1
        let high_risk_weight = 0.2f64; // 0.2

        // --- Execution: Run many simulations to average out randomness ---
        const NUM_SIMULATIONS: u32 = 100;
        let mut total_high_risk_selected = 0;

        for _ in 0..NUM_SIMULATIONS {
            let mut index = DigitBinIndex::with_precision_and_capacity(3, TOTAL_ITEMS);
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
        const ITEMS_PER_GROUP: u64 = 1000;
        const TOTAL_ITEMS: u64 = ITEMS_PER_GROUP * 2;
        const NUM_DRAWS: u64 = TOTAL_ITEMS / 2;

        let low_risk_weight = 0.1f64;  // 0.1
        let high_risk_weight = 0.2f64; // 0.2

        const NUM_SIMULATIONS: u32 = 100;
        let mut total_high_risk_selected = 0;

        for _ in 0..NUM_SIMULATIONS {
            let mut index = DigitBinIndex::with_precision_and_capacity(3, TOTAL_ITEMS);
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

#[cfg(test)]
#[test]
fn test_weight_to_digits() {
    // Create an instance (using Vec<u32> as the bin type for simplicity)
    let index = DigitBinIndexGeneric::<Vec<u32>>::with_precision(3);

    // Test valid weight
    let mut digits = [0u8; MAX_PRECISION];
    if let Some(scaled) = index.weight_to_digits(0.123, &mut digits) {    
        assert_eq!(scaled, 123);
        assert_eq!(digits[0..3], [1, 2, 3]);
        assert_eq!(digits[3..], [0; 6]); // Remaining digits should be zero-padded
    } else {
        panic!("Expected Some for valid weight");
    }

    // Test invalid weights
    assert!(index.weight_to_digits(0.0, &mut digits).is_none());
    assert!(index.weight_to_digits(-0.1, &mut digits).is_none());
    assert!(index.weight_to_digits(0.0000001, &mut digits).is_none()); // Rounds to zero after scaling

    // Test overflow (though unlikely for weights <1, but for completeness)
    assert!(index.weight_to_digits(2.0, &mut digits).is_none()); // Should trigger temp != 0 check
}

#[cfg(test)]
#[test]
fn test_add_many() {
    const CAPACITY: u64 = 1_000_000u64;
    let mut index_one_at_a_time = DigitBinIndex::with_precision_and_capacity(3, CAPACITY);
    let mut index_all_at_once = DigitBinIndex::with_precision_and_capacity(3, CAPACITY);
    let mut population = Vec::with_capacity(CAPACITY as usize);
    let mut rng = WyRand::from_os_rng();
    for i in 0..CAPACITY {
        let weight: f64 = rng.random_range(0.001..=0.999);
        population.push((i, weight));
        index_one_at_a_time.add(i, weight);
    }
    index_all_at_once.add_many(&population);
    index_one_at_a_time.print_stats();
    index_all_at_once.print_stats();
}
#[test]
fn test_add() {
    let mut index = DigitBinIndex::new();
    index.add(1, 0.5);
    index.print_stats();
}

