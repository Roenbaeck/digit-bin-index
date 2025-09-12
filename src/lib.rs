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
const MAX_PRECISION: usize = 9;

/// Trait for types that can be used as leaf bins in a `DigitBinIndex`.
///
/// Implement this trait for any container you want to use for storing IDs in the leaf nodes.
/// Provided implementations: [`Vec<u32>`], [`RoaringBitmap`].
pub trait DigitBin: Clone + Default {
    fn insert(&mut self, id: u32);
    fn remove(&mut self, id: u32) -> bool;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn get_random(&self, rng: &mut impl rand::Rng) -> Option<u32>;
    fn get_random_and_remove(&mut self, rng: &mut impl rand::Rng) -> Option<u32>;
}

impl DigitBin for Vec<u32> {
    fn insert(&mut self, id: u32) { self.push(id); }
    fn remove(&mut self, id: u32) -> bool {
        if let Some(pos) = self.iter().position(|&x| x == id) {
            self.swap_remove(pos);
            true
        } else {
            false
        }
    }
    fn len(&self) -> usize { self.len() }
    fn is_empty(&self) -> bool { self.is_empty() }
    fn get_random(&self, rng: &mut impl rand::Rng) -> Option<u32> {
        if self.is_empty() { None } else { Some(self[rng.gen_range(0..self.len())]) }
    }
    fn get_random_and_remove(&mut self, rng: &mut impl rand::Rng) -> Option<u32> {
        if self.is_empty() { None } else {
            let pos = rng.gen_range(0..self.len());
            Some(self.swap_remove(pos))
        }
    }
}

impl DigitBin for RoaringBitmap {
    fn insert(&mut self, id: u32) { self.insert(id); }
    fn remove(&mut self, id: u32) -> bool { self.remove(id) }
    fn len(&self) -> usize { self.len() as usize }
    fn is_empty(&self) -> bool { self.is_empty() }
    fn get_random(&self, rng: &mut impl rand::Rng) -> Option<u32> {
        if self.is_empty() { None } else {
            let idx = rng.gen_range(0..self.len() as u32);
            self.select(idx)
        }
    }
    fn get_random_and_remove(&mut self, rng: &mut impl rand::Rng) -> Option<u32> {
        if self.is_empty() { None } else {
            let idx = rng.gen_range(0..self.len() as u32);
            let selected = self.select(idx);
            self.remove(idx);
            selected
        }
    }
}

/// The content of a node, which is either more nodes or a leaf with individuals.
#[derive(Debug, Clone)]
pub enum NodeContent<B: DigitBin> {
    /// An internal node that contains children for the next digit (0-9).
    DigitIndex(Vec<Node<B>>),
    /// A leaf node that contains a bin of IDs for individuals in this bin.
    Bin(B),
}

/// A node within the DigitBinIndex tree.
#[derive(Debug, Clone)]
pub struct Node<B: DigitBin> {
    /// The content of this node, either more nodes or a list of individual IDs.
    pub content: NodeContent<B>,
    /// The total sum of probabilities stored under this node.
    pub accumulated_value: Decimal,
    /// The total count of individuals stored under this node.
    pub content_count: u32,
}

impl<B: DigitBin> Node<B> {
    /// Creates a new, empty internal node.
    fn new_internal() -> Self {
        Self {
            content: NodeContent::DigitIndex(vec![]), 
            accumulated_value: Decimal::ZERO,
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
    Vec(DigitBinIndexGeneric<Vec<u32>>),
    Roaring(DigitBinIndexGeneric<RoaringBitmap>),
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
    pub fn with_precision_and_capacity(precision: u8, capacity: usize) -> Self {
        // Heuristic: Use RoaringBitmap if average bin size (capacity / 10^precision) exceeds threshold
        let threshold = 1000;
        let max_bins = 10usize.pow(precision as u32);
        if capacity / max_bins > threshold {
            DigitBinIndex::Roaring(DigitBinIndexGeneric::<RoaringBitmap>::with_precision(precision))
        } else {
            DigitBinIndex::Vec(DigitBinIndexGeneric::<Vec<u32>>::with_precision(precision))
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
        DigitBinIndex::Vec(DigitBinIndexGeneric::<Vec<u32>>::new())
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
        DigitBinIndex::Vec(DigitBinIndexGeneric::<Vec<u32>>::with_precision(precision))
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
    /// use rust_decimal_macros::dec;
    ///
    /// let mut index = DigitBinIndex::new();
    /// let added = index.add(1, dec!(0.5));
    /// assert!(added);
    /// assert_eq!(index.count(), 1);
    /// ```    
    pub fn add(&mut self, id: u32, weight: Decimal) -> bool {
        match self {
            DigitBinIndex::Vec(index) => index.add(id, weight),
            DigitBinIndex::Roaring(index) => index.add(id, weight),
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
    pub fn remove(&mut self, id: u32, weight: Decimal) {
        match self {
            DigitBinIndex::Vec(index) => index.remove(id, weight),
            DigitBinIndex::Roaring(index) => index.remove(id, weight),
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
    /// use rust_decimal_macros::dec;
    ///
    /// let mut index = DigitBinIndex::new();
    /// index.add(1, dec!(0.5));
    /// if let Some((id, weight)) = index.select() {
    ///     assert_eq!(id, 1);
    ///     assert_eq!(weight, dec!(0.500));
    /// }
    /// ```
    pub fn select(&mut self) -> Option<(u32, Decimal)> {
        match self {
            DigitBinIndex::Vec(index) => index.select(),
            DigitBinIndex::Roaring(index) => index.select(),
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
        match self {
            DigitBinIndex::Vec(index) => index.select_and_remove(),
            DigitBinIndex::Roaring(index) => index.select_and_remove(),
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
    /// use rust_decimal_macros::dec;
    ///
    /// let mut index = DigitBinIndex::new();
    /// index.add(1, dec!(0.3));
    /// index.add(2, dec!(0.7));
    /// if let Some(selected) = index.select_many(2) {
    ///     assert_eq!(selected.len(), 2);
    /// }
    /// ```
    pub fn select_many(&mut self, num_to_draw: u32) -> Option<Vec<(u32, Decimal)>> {
        match self {
            DigitBinIndex::Vec(index) => index.select_many(num_to_draw),
            DigitBinIndex::Roaring(index) => index.select_many(num_to_draw),
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
        match self {
            DigitBinIndex::Vec(index) => index.select_many_and_remove(num_to_draw),
            DigitBinIndex::Roaring(index) => index.select_many_and_remove(num_to_draw),
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
        match self {
            DigitBinIndex::Vec(index) => index.count(),
            DigitBinIndex::Roaring(index) => index.count(),
        }
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
        match self {
            DigitBinIndex::Vec(index) => index.total_weight(),
            DigitBinIndex::Roaring(index) => index.total_weight(),
        }
    }

    /// Prints statistics about the underlying tree.
    pub fn print_stats(&self) {
        match self {
            DigitBinIndex::Vec(idx) => idx.print_stats(),
            DigitBinIndex::Roaring(idx) => idx.print_stats(),
        }
    }    

    /// Returns the precision (number of decimal places) used for binning.
    pub fn precision(&self) -> u8 {
        match self {
            DigitBinIndex::Vec(idx) => idx.precision,
            DigitBinIndex::Roaring(idx) => idx.precision,
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
    // For weight_to_digits
    powers: [u32; MAX_PRECISION], 
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
        let mut powers = [0u32; MAX_PRECISION];
        for i in 0..MAX_PRECISION {
            powers[i] = 10u32.pow(i as u32)
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

        let mut scaled = weight;
        scaled.rescale(self.precision as u32);
        if scaled.is_zero() {
            return None;
        }

        let mut digits = [0u8; MAX_PRECISION];
        let scale = scaled.scale() as usize;
        let mantissa = scaled.mantissa().abs() as u32;

        for i in 0..self.precision as usize {
            if i >= scale {
                digits[i] = 0;
            } else {
                digits[i] = ((mantissa / self.powers[self.precision as usize - 1 - i]) % 10) as u8;
            }
        }
        Some(digits)
    }

    // --- Standard Functions ---

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
        node: &mut Node<B>,
        individual_id: u32,
        weight: Decimal, // Still needed for accumulated_value
        digits: &[u8; MAX_PRECISION],
        current_depth: u8,
        max_depth: u8,
    ) {
        node.content_count += 1;
        node.accumulated_value += weight;

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
            if children.len() <= digit {
                children.resize_with(digit + 1, Node::new_internal);
            }
            Self::add_recurse(&mut children[digit], individual_id, weight, digits, current_depth + 1, max_depth);
        }
    }

    pub fn remove(&mut self, individual_id: u32, mut weight: Decimal) {
        if let Some(digits) = self.weight_to_digits(weight) {
            weight.rescale(self.precision as u32);
            Self::remove_recurse(&mut self.root, individual_id, weight, &digits, 1, self.precision);
        }
    }

    /// Recursive private method to handle removing individuals.
    fn remove_recurse(
        node: &mut Node<B>,
        individual_id: u32,
        weight: Decimal,
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
                    node.accumulated_value -= weight;
                    return true;
                }
            }
            return false;
        }

        let digit = digits[current_depth as usize - 1] as usize;
        if let NodeContent::DigitIndex(children) = &mut node.content {
            if children.len() > digit && Self::remove_recurse(&mut children[digit], individual_id, weight, digits, current_depth + 1, max_depth) {
                node.content_count -= 1;
                node.accumulated_value -= weight;
                return true;
            }
        }
        false
    }


    // --- Selection Functions ---

    pub fn select(&mut self) -> Option<(u32, Decimal)> {
        self.select_and_optionally_remove(false)
    }

    pub fn select_many(&mut self, num_to_draw: u32) -> Option<Vec<(u32, Decimal)>> {
        self.select_many_and_optionally_remove(num_to_draw, false)
    }

    pub fn select_and_remove(&mut self) -> Option<(u32, Decimal)> {
        self.select_and_optionally_remove(true)
    }

    // Wrapper function to handle both select and select_and_remove
    pub fn select_and_optionally_remove(&mut self, with_removal: bool) -> Option<(u32, Decimal)> {
        if self.root.content_count == 0 {
            return None;
        }
        let mut rng = rand::thread_rng();
        let random_target = rng.gen_range(Decimal::ZERO..self.root.accumulated_value);
        Self::select_and_optionally_remove_recurse(&mut self.root, random_target, 1, self.precision, &mut rng, with_removal)
    }

    // Helper function
    fn select_and_optionally_remove_recurse(
        node: &mut Node<B>,
        mut target: Decimal,
        current_depth: u8,
        max_depth: u8,
        rng: &mut ThreadRng,
        with_removal: bool,
    ) -> Option<(u32, Decimal)> {
        // Base case: Bin node
        if current_depth > max_depth {
            if let NodeContent::Bin(bin) = &mut node.content {
                if bin.is_empty() {
                    return None;
                }
                let weight = node.accumulated_value / Decimal::from(node.content_count);
                if with_removal {
                    let selected_id = bin.get_random_and_remove(rng).unwrap();
                    node.content_count -= 1;
                    node.accumulated_value -= weight;
                    return Some((selected_id, weight));
                }
                else {
                    let selected_id = bin.get_random(rng).unwrap();
                    return Some((selected_id, weight));
                }
            }
            return None;
        }

        // Recursive case: DigitIndex node
        if let NodeContent::DigitIndex(children) = &mut node.content {
            for child in children.iter_mut() {
                if child.accumulated_value.is_zero() {
                    continue;
                }
                if target < child.accumulated_value {
                    if let Some((selected_id, weight)) = Self::select_and_optionally_remove_recurse(
                        child,
                        target,
                        current_depth + 1,
                        max_depth,
                        rng,
                        with_removal,
                    ) {
                        if with_removal {
                            node.content_count -= 1;
                            node.accumulated_value -= weight;
                        }
                        return Some((selected_id, weight));
                    }
                    return None; // If recurse failed (e.g., empty after skip), but shouldn't happen
                }
                target -= child.accumulated_value;
            }
        }
        None
    } 

    pub fn select_many_and_remove(&mut self, num_to_draw: u32) -> Option<Vec<(u32, Decimal)>> {
        self.select_many_and_optionally_remove(num_to_draw, true)
    }

    // Wrapper function to handle both select_many and select_many_and_remove
    pub fn select_many_and_optionally_remove(&mut self, num_to_draw: u32, with_removal: bool) -> Option<Vec<(u32, Decimal)>> {
        if num_to_draw > self.count() || num_to_draw == 0 {
            return if num_to_draw == 0 { Some(Vec::new()) } else { None };
        }
        let mut rng = rand::thread_rng();
        let mut selected = Vec::with_capacity(num_to_draw as usize);
        let total_weight = self.root.accumulated_value;
        // Generate initial random targets for the root
        let passed_targets: Vec<Decimal> = (0..num_to_draw)
            .map(|_| rng.gen_range(Decimal::ZERO..total_weight))
            .collect();
        // Call with pre-generated targets
        Self::select_many_and_optionally_remove_recurse(
            &mut self.root,
            total_weight,
            &mut selected,
            &mut rng,
            1,
            self.precision,
            with_removal,
            passed_targets,
        );
        if selected.len() == num_to_draw as usize {
            Some(selected)
        } else {
            None // Should not happen if logic is correct
        }
    }

    /// Recursive helper for batch selection and removal.
    /// - node: Current subtree root.
    /// - subtree_total: Accumulated weight of this node (passed to avoid borrowing issues).
    /// - selected: Mutable vec to collect (id, weight) from leaves.
    /// - rng: Mutable RNG.
    /// - current_depth: Current digit level.
    /// - precision: The precision of the DigitBinIndex (passed explicitly).
    /// - with_removal: Whether to remove selected items.
    /// - passed_targets: Pre-computed relative targets from parent (in [0, subtree_total)).
    fn select_many_and_optionally_remove_recurse(
        node: &mut Node<B>,
        subtree_total: Decimal,
        selected: &mut Vec<(u32, Decimal)>,
        rng: &mut ThreadRng,
        current_depth: u8,
        precision: u8,
        with_removal: bool,
        passed_targets: Vec<Decimal>,
    ) {
        let original_target_count = passed_targets.len() as u32;
        if original_target_count == 0 {
            return;
        }

        if current_depth > precision {
            if let NodeContent::Bin(bin) = &mut node.content {
                let bin_weight = if node.content_count > 0 {
                    node.accumulated_value / Decimal::from(node.content_count)
                } else {
                    Decimal::ZERO
                };
                let to_select = original_target_count.min(node.content_count);
                let mut picked = 0u32;
                while picked < to_select && !bin.is_empty() {
                    if with_removal {
                        let id = bin.get_random_and_remove(rng).unwrap();
                        selected.push((id, bin_weight));
                    } else {
                        let id = bin.get_random(rng).unwrap();
                        selected.push((id, bin_weight));
                    }  
                    picked += 1;
                }
                if with_removal {
                    node.content_count -= picked;
                    node.accumulated_value -= bin_weight * Decimal::from(picked);
                }
            }
            return;
        }

        // DigitIndex node: Assign to children using passed_targets first, with rejection.
        if let NodeContent::DigitIndex(children) = &mut node.content {
            let mut child_assigned = vec![0u32; children.len()];
            let mut child_rel_targets: Vec<Vec<Decimal>> = vec![Vec::new(); children.len()];
            let mut assigned = 0u32;

            for target in passed_targets {
                let mut cum = Decimal::ZERO;
                let mut chosen_idx = None;
                for (i, child) in children.iter().enumerate() {
                    if child.accumulated_value.is_zero() {
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
                if let Some(idx) = chosen_idx {
                    child_assigned[idx] += 1;
                    let rel_target = target - cum;
                    child_rel_targets[idx].push(rel_target);
                    assigned += 1;
                } else {
                    // println!("Rejected target {}", target); // Debug
                }
            }

            // Generate additional targets for any rejected ones
            let remaining = original_target_count - assigned;
            let mut additional_assigned = 0u32;
            while additional_assigned < remaining {
                let target = rng.gen_range(Decimal::ZERO..subtree_total);
                let mut cum = Decimal::ZERO;
                let mut chosen_idx = None;
                for (i, child) in children.iter().enumerate() {
                    if child.accumulated_value.is_zero() {
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
                if let Some(idx) = chosen_idx {
                    child_assigned[idx] += 1;
                    let rel_target = target - cum;
                    child_rel_targets[idx].push(rel_target);
                    additional_assigned += 1;
                } else {
                    // println!("Rejected additional target {}", target); // Debug
                }
            }

            // Store accumulated values to avoid immutable borrow during iteration
            let child_weights: Vec<Decimal> = children.iter().map(|c| c.accumulated_value).collect();

            // Recurse into each child with assigned > 0, passing rel_targets
            for (i, child) in children.iter_mut().enumerate() {
                let assign = child_assigned[i];
                if assign > 0 {
                    let rel_targets = std::mem::take(&mut child_rel_targets[i]);
                    Self::select_many_and_optionally_remove_recurse(
                        child,
                        child_weights[i],
                        selected,
                        rng,
                        current_depth + 1,
                        precision,
                        with_removal,
                        rel_targets,
                    );
                }
            }

            if with_removal {
                // On unwind: Update this node's counts and values based on what was removed below
                node.content_count = children.iter().map(|c| c.content_count).sum();
                node.accumulated_value = children.iter().map(|c| c.accumulated_value).sum();
            }
        }
    }

    pub fn count(&self) -> u32 {
        self.root.content_count
    }

    pub fn total_weight(&self) -> Decimal {
        self.root.accumulated_value
    }

    /// Prints statistics about the tree: node count, bin stats, and weight stats.
    pub fn print_stats(&self) {
        struct Stats {
            node_count: usize,
            non_empty_node_count: usize,
            bin_count: usize,
            min_bin_size: Option<usize>,
            max_bin_size: Option<usize>,
            total_bin_items: usize,
            min_weight: Option<Decimal>,
            max_weight: Option<Decimal>,
            total_weight: Decimal,
        }

        fn traverse<B: DigitBin>(
            node: &Node<B>,
            stats: &mut Stats,
            precision: u8,
            current_depth: u8,
            path: &str,
        ) {
            stats.node_count += 1;
            let is_non_empty = node.content_count > 0;
            if is_non_empty {
                stats.non_empty_node_count += 1;
            }
            /* 
            println!(
                "Node at depth {} (path: {}): content_count = {}, non_empty = {}",
                current_depth, path, node.content_count, is_non_empty
            );
            */
            match &node.content {
                NodeContent::DigitIndex(children) => {
                    for (i, child) in children.iter().enumerate() {
                        let new_path = format!("{} -> {}", path, i);
                        traverse(child, stats, precision, current_depth + 1, &new_path);
                    }
                }
                NodeContent::Bin(bin) => {
                    let bin_size = bin.len();
                    stats.bin_count += 1;
                    stats.total_bin_items += bin_size;
                    stats.min_bin_size = Some(stats.min_bin_size.map_or(bin_size, |min| min.min(bin_size)));
                    stats.max_bin_size = Some(stats.max_bin_size.map_or(bin_size, |max| max.max(bin_size)));

                    if bin_size > 0 {
                        let weight = node.accumulated_value / Decimal::from(node.content_count);
                        stats.min_weight = Some(stats.min_weight.map_or(weight, |min| min.min(weight)));
                        stats.max_weight = Some(stats.max_weight.map_or(weight, |max| max.max(weight)));
                        stats.total_weight += node.accumulated_value;
                    }
                }
            }
        }

        let mut stats = Stats {
            node_count: 0,
            non_empty_node_count: 0,
            bin_count: 0,
            min_bin_size: None,
            max_bin_size: None,
            total_bin_items: 0,
            min_weight: None,
            max_weight: None,
            total_weight: Decimal::ZERO,
        };

        traverse(&self.root, &mut stats, self.precision, 1, "Root");

        let avg_bin_size = if stats.bin_count > 0 {
            stats.total_bin_items as f64 / stats.bin_count as f64
        } else {
            0.0
        };
        let avg_weight = if stats.bin_count > 0 {
            stats.total_weight / Decimal::from(stats.bin_count as u32)
        } else {
            Decimal::ZERO
        };

        println!("DigitBinIndex statistics:");
        println!("  Total nodes: {}", stats.node_count);
        println!("  Non-empty nodes: {}", stats.non_empty_node_count);
        println!("  Total bins (leaves): {}", stats.bin_count);
        println!("  Total items: {}", stats.total_bin_items);
        println!("  Average items per bin: {:.2}", avg_bin_size);
        println!(
            "  Smallest bin size: {}",
            stats.min_bin_size.map_or_else(|| "-".to_string(), |v| v.to_string())
        );
        println!(
            "  Largest bin size: {}",
            stats.max_bin_size.map_or_else(|| "-".to_string(), |v| v.to_string())
        );
        println!(
            "  Smallest represented weight: {}",
            stats.min_weight.map_or_else(|| "-".to_string(), |v| v.to_string())
        );
        println!(
            "  Largest represented weight: {}",
            stats.max_weight.map_or_else(|| "-".to_string(), |v| v.to_string())
        );
        println!("  Average weight: {}", avg_weight);
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
        fn new() -> Self {
            PyDigitBinIndex {
                index: DigitBinIndex::new(),
            }
        }

        /// Create a DigitBinIndex with a specific precision.
        #[staticmethod]
        fn with_precision(precision: u32) -> Self {
            PyDigitBinIndex {
                index: DigitBinIndex::with_precision(precision.try_into().unwrap()),
            }
        }

        /// Create a DigitBinIndex with a specific precision and expected capacity.
        #[staticmethod]
        fn with_precision_and_capacity(precision: u32, capacity: usize) -> Self {
            PyDigitBinIndex {
                index: DigitBinIndex::with_precision_and_capacity(precision.try_into().unwrap(), capacity),
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

        fn select(&mut self) -> Option<(u32, String)> {
            self.index.select().map(|(id, weight)| (id, weight.to_string()))
        }

        fn select_many(&mut self, n: u32) -> Option<Vec<(u32, String)>> {
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
        const ITEMS_PER_GROUP: u32 = 1000;
        const TOTAL_ITEMS: u32 = ITEMS_PER_GROUP * 2;
        const NUM_DRAWS: u32 = TOTAL_ITEMS / 2;

        let low_risk_weight = dec!(0.1);  // 0.1
        let high_risk_weight = dec!(0.2); // 0.2

        // --- Execution: Run many simulations to average out randomness ---
        const NUM_SIMULATIONS: u32 = 100;
        let mut total_high_risk_selected = 0;

        for _ in 0..NUM_SIMULATIONS {
            let mut index = DigitBinIndex::with_precision_and_capacity(3, TOTAL_ITEMS as usize);
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
            let mut index = DigitBinIndex::with_precision_and_capacity(3, TOTAL_ITEMS as usize);
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