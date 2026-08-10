//! A `DigitBinIndex` is a tree-based data structure that organizes a large
//! collection of weighted items to enable highly efficient weighted random
//! selection and removal.
//!
//! It is a specialized tool, purpose-built for scenarios with millions of
//! items where probabilities are approximate and high performance is critical,
//! particularly for simulations involving sequential sampling like Wallenius'
//! noncentral hypergeometric distribution.

use wyrand::WyRand;
use rand::{distr::Distribution, Rng, SeedableRng};
use rand_distr::Binomial;
use roaring::{RoaringBitmap, RoaringTreemap};
use std::collections::HashSet;

// The default precision to use if none is specified in the constructor.
const DEFAULT_PRECISION: u8 = 3;
const MAX_PRECISION: usize = 9;

#[derive(Clone)]
struct IndexRng(WyRand);

impl std::fmt::Debug for IndexRng {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("IndexRng(..)")
    }
}

#[derive(Debug, Clone, Copy)]
struct FisherBin {
    count: u64,
    scaled_weight: u64,
}

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
    fn get_random_many(
        &mut self,
        rng: &mut impl rand::Rng,
        count: usize,
        with_removal: bool,
    ) -> Option<Vec<u64>>;
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
    fn get_random_many(
        &mut self,
        rng: &mut impl rand::Rng,
        count: usize,
        with_removal: bool,
    ) -> Option<Vec<u64>> {
        if count > self.len() {
            return None;
        }
        if with_removal && count == self.len() {
            return Some(std::mem::take(self).into_iter().map(u64::from).collect());
        }
        if with_removal {
            let mut selected = Vec::with_capacity(count);
            for _ in 0..count {
                let pos = rng.random_range(0..self.len());
                selected.push(self.swap_remove(pos) as u64);
            }
            Some(selected)
        } else {
            Some(
                rand::seq::index::sample(rng, self.len(), count)
                    .into_iter()
                    .map(|index| self[index] as u64)
                    .collect(),
            )
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
    fn get_random_many(
        &mut self,
        rng: &mut impl rand::Rng,
        count: usize,
        with_removal: bool,
    ) -> Option<Vec<u64>> {
        if count > self.len() as usize {
            return None;
        }
        if with_removal && count == self.len() as usize {
            let selected = self.iter().map(u64::from).collect();
            self.clear();
            return Some(selected);
        }
        if with_removal {
            let selected: Option<Vec<u64>> = rand::seq::index::sample(
                rng,
                self.len() as usize,
                count,
            )
            .into_iter()
            .map(|index| self.select(index as u32).map(u64::from))
            .collect();
            let selected = selected?;
            for &id in &selected {
                self.remove(id as u32);
            }
            Some(selected)
        } else {
            rand::seq::index::sample(rng, self.len() as usize, count)
                .into_iter()
                .map(|index| self.select(index as u32).map(u64::from))
                .collect()
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
    fn get_random_many(
        &mut self,
        rng: &mut impl rand::Rng,
        count: usize,
        with_removal: bool,
    ) -> Option<Vec<u64>> {
        if count > self.len() as usize {
            return None;
        }
        if with_removal && count == self.len() as usize {
            let selected = self.iter().collect();
            self.clear();
            return Some(selected);
        }
        if with_removal {
            let selected: Option<Vec<u64>> = rand::seq::index::sample(
                rng,
                self.len() as usize,
                count,
            )
            .into_iter()
            .map(|index| self.select(index as u64))
            .collect();
            let selected = selected?;
            for &id in &selected {
                self.remove(id);
            }
            Some(selected)
        } else {
            rand::seq::index::sample(rng, self.len() as usize, count)
                .into_iter()
                .map(|index| self.select(index as u64))
                .collect()
        }
    }
}

/// Direct digit lookup plus the optimal descending-mass order for selection.
#[derive(Debug, Clone)]
pub struct DigitIndex<B: DigitBin> {
    children: Box<[Option<Node<B>>; 10]>,
    order: [u8; 10],
    active_len: u8,
}

impl<B: DigitBin> DigitIndex<B> {
    fn new() -> Self {
        Self {
            children: Box::new(Default::default()),
            order: [0; 10],
            active_len: 0,
        }
    }

    fn get_or_insert_with(
        &mut self,
        digit: usize,
        create: impl FnOnce() -> Node<B>,
    ) -> &mut Node<B> {
        self.children[digit].get_or_insert_with(create)
    }

    fn get_mut(&mut self, digit: usize) -> Option<&mut Node<B>> {
        self.children[digit].as_mut()
    }

    fn iter(&self) -> impl DoubleEndedIterator<Item = &Option<Node<B>>> {
        self.children.iter()
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Option<Node<B>>> {
        self.children.iter_mut()
    }

    fn child_increased(&mut self, digit: usize) {
        let digit = digit as u8;
        let active_len = self.active_len as usize;
        let mut position = self.order[..active_len]
            .iter()
            .position(|&ordered_digit| ordered_digit == digit)
            .unwrap_or_else(|| {
                self.order[active_len] = digit;
                self.active_len += 1;
                active_len
            });

        while position > 0 && self.precedes(digit, self.order[position - 1]) {
            self.order.swap(position, position - 1);
            position -= 1;
        }
    }

    fn child_decreased(&mut self, digit: usize) {
        if let Some(position) = self.order[..self.active_len as usize]
            .iter()
            .position(|&ordered_digit| ordered_digit as usize == digit)
        {
            self.child_decreased_at(position);
        }
    }

    fn child_decreased_at(&mut self, mut position: usize) {
        let digit = self.order[position];
        let mass = self.children[digit as usize]
            .as_ref()
            .map_or(0, |child| child.accumulated_value);
        if mass == 0 {
            let active_len = self.active_len as usize;
            self.order.copy_within(position + 1..active_len, position);
            self.active_len -= 1;
            return;
        }

        while position + 1 < self.active_len as usize
            && self.precedes(self.order[position + 1], digit)
        {
            self.order.swap(position, position + 1);
            position += 1;
        }
    }

    fn precedes(&self, left: u8, right: u8) -> bool {
        let left_mass = self.children[left as usize]
            .as_ref()
            .map_or(0, |child| child.accumulated_value);
        let right_mass = self.children[right as usize]
            .as_ref()
            .map_or(0, |child| child.accumulated_value);
        left_mass > right_mass || (left_mass == right_mass && left > right)
    }

    fn ordered_digits(&self) -> &[u8] {
        &self.order[..self.active_len as usize]
    }

    fn rebuild_order(&mut self) {
        self.active_len = 0;
        for digit in 0..10 {
            if self.children[digit]
                .as_ref()
                .is_some_and(|child| child.accumulated_value > 0)
            {
                self.child_increased(digit);
            }
        }
    }
}

/// The content of a node, which is either more nodes or a leaf with individuals.
#[derive(Debug, Clone)]
pub enum NodeContent<B: DigitBin> {
    /// An internal node that contains children for the next digit (0-9).
    DigitIndex(DigitIndex<B>),
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
            content: NodeContent::DigitIndex(DigitIndex::new()),
            accumulated_value: 0u64,
            content_count: 0,
        }
    }

    /// Creates a new, empty leaf node.
    fn new_bin() -> Self {
        Self {
            content: NodeContent::Bin(B::default()),
            accumulated_value: 0,
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

    /// Creates a new DigitBinIndex with Vec<u32> bins and the specified precision.
    ///
    /// Optimized for small to medium-sized problems (average <= 1,000 items per bin).
    /// Provides the fastest O(1) select_and_remove performance but truncates u64 IDs to u32.
    ///
    /// # Arguments
    ///
    /// * `precision` - The number of decimal places for binning (1 to 9).
    ///
    /// # Returns
    ///
    /// A new `DigitBinIndex` instance with Vec<u32> bins.
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
    /// let index = DigitBinIndex::small(3);
    /// assert_eq!(index.precision(), 3);
    /// ```
    pub fn small(precision: u8) -> Self {
        DigitBinIndex::Small(DigitBinIndexGeneric::<Vec<u32>>::with_precision(precision))
    }

    /// Creates a new DigitBinIndex with RoaringBitmap bins and the specified precision.
    ///
    /// Optimized for large-scale problems (average > 1,000 items per bin) where IDs fit within u32.
    /// Provides excellent memory compression and fast set operations but truncates u64 IDs to u32.
    ///
    /// # Arguments
    ///
    /// * `precision` - The number of decimal places for binning (1 to 9).
    ///
    /// # Returns
    ///
    /// A new `DigitBinIndex` instance with RoaringBitmap bins.
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
    /// let index = DigitBinIndex::medium(3);
    /// assert_eq!(index.precision(), 3);
    /// ```
    pub fn medium(precision: u8) -> Self {
        DigitBinIndex::Medium(DigitBinIndexGeneric::<RoaringBitmap>::with_precision(precision))
    }

    /// Creates a new DigitBinIndex with RoaringTreemap bins and the specified precision.
    ///
    /// Optimized for massive-scale problems requiring full u64 ID support (average > 1,000,000,000 items per bin).
    /// Supports the full 64-bit ID space, ideal for extremely large datasets.
    ///
    /// # Arguments
    ///
    /// * `precision` - The number of decimal places for binning (1 to 9).
    ///
    /// # Returns
    ///
    /// A new `DigitBinIndex` instance with RoaringTreemap bins.
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
    /// let index = DigitBinIndex::large(3);
    /// assert_eq!(index.precision(), 3);
    /// ```
    pub fn large(precision: u8) -> Self {
        DigitBinIndex::Large(DigitBinIndexGeneric::<RoaringTreemap>::with_precision(precision))
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

    /// Selects a single item using the caller-provided random number generator.
    pub fn select_with_rng(&mut self, rng: &mut impl Rng) -> Option<(u64, f64)> {
        match self {
            DigitBinIndex::Small(index) => index.select_with_rng(rng),
            DigitBinIndex::Medium(index) => index.select_with_rng(rng),
            DigitBinIndex::Large(index) => index.select_with_rng(rng),
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

    /// Selects and removes a single item using the caller-provided random
    /// number generator.
    pub fn select_and_remove_with_rng(
        &mut self,
        rng: &mut impl Rng,
    ) -> Option<(u64, f64)> {
        match self {
            DigitBinIndex::Small(index) => index.select_and_remove_with_rng(rng),
            DigitBinIndex::Medium(index) => index.select_and_remove_with_rng(rng),
            DigitBinIndex::Large(index) => index.select_and_remove_with_rng(rng),
        }
    }

    /// Performs repeated weighted draws with removal, producing Wallenius'
    /// noncentral hypergeometric law for the index's quantized weights.
    pub fn select_wallenius_many_and_remove(
        &mut self,
        num_to_draw: u64,
    ) -> Option<Vec<(u64, f64)>> {
        match self {
            DigitBinIndex::Small(index) => index.select_wallenius_many_and_remove(num_to_draw),
            DigitBinIndex::Medium(index) => index.select_wallenius_many_and_remove(num_to_draw),
            DigitBinIndex::Large(index) => index.select_wallenius_many_and_remove(num_to_draw),
        }
    }

    /// Performs repeated weighted draws with removal using the caller-provided
    /// random number generator.
    pub fn select_wallenius_many_and_remove_with_rng(
        &mut self,
        num_to_draw: u64,
        rng: &mut impl Rng,
    ) -> Option<Vec<(u64, f64)>> {
        match self {
            DigitBinIndex::Small(index) => {
                index.select_wallenius_many_and_remove_with_rng(num_to_draw, rng)
            }
            DigitBinIndex::Medium(index) => {
                index.select_wallenius_many_and_remove_with_rng(num_to_draw, rng)
            }
            DigitBinIndex::Large(index) => {
                index.select_wallenius_many_and_remove_with_rng(num_to_draw, rng)
            }
        }
    }

    /// Selects multiple unique items randomly based on weights without removal.
    ///
    /// Samples the exact multivariate Fisher noncentral hypergeometric law for
    /// the index's quantized weights. Small draws use weighted proposals
    /// conditioned on unique IDs; larger draws use exponentially tilted
    /// binomial counts conditioned on the requested total. Returns `None` if
    /// `num_to_draw` exceeds the number of items in the index.
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
    /// Samples the exact multivariate Fisher noncentral hypergeometric law for
    /// the index's quantized weights and removes the selected IDs in batches.
    /// Returns `None` if `num_to_draw` exceeds item count.
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
    /// Persistent random state used by convenience selection methods.
    rng: IndexRng,
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
            rng: IndexRng(WyRand::from_os_rng()),
        }        
    }

    /// Converts a f64 weight to an array of digits [0-9] for the given precision and the scaled u64 value.
    /// Returns None if the weight is invalid (non-positive or zero after scaling).
    fn weight_to_digits(&self, weight: f64, digits: &mut [u8; MAX_PRECISION]) -> Option<u64> {
        if weight <= 0.0 || weight >= 1.0 {
            return None;
        }

        let scaled = (weight * self.scale) as u64;
        if scaled == 0 {
            return None;
        }

        let mut temp = scaled;
        for i in (0..self.precision as usize).rev() {
            digits[i] = (temp % 10) as u8;
            temp /= 10;
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
            {
                let child_node = children.get_or_insert_with(digit, || {
                    if current_depth == max_depth {
                        Node::new_bin()
                    } else {
                        Node::new_internal()
                    }
                });
                Self::add_recurse(child_node, individual_id, scaled, digits, current_depth + 1, max_depth);
            }
            children.child_increased(digit);
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
            if let Some(child_node) = children.get_mut(digit) {
                // If it exists, recurse. If the recursion returns true (success)...
                if Self::remove_recurse(child_node, individual_id, scaled, digits, current_depth + 1, max_depth) {
                    // ...then update this node's stats and propagate the success upwards.
                    node.content_count -= 1;
                    node.accumulated_value -= scaled;
                    children.child_decreased(digit);
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
        let Self { root, precision, scale, rng } = self;
        Self::select_with_rng_inner::<false, _>(root, *precision, *scale, &mut rng.0)
    }

    pub fn select_with_rng(&mut self, rng: &mut impl Rng) -> Option<(u64, f64)> {
        Self::select_with_rng_inner::<false, _>(&mut self.root, self.precision, self.scale, rng)
    }

    pub fn select_many(&mut self, num_to_draw: u64) -> Option<Vec<(u64, f64)>> {
        self.select_many_and_optionally_remove(num_to_draw, false)
    }

    pub fn select_and_remove(&mut self) -> Option<(u64, f64)> {
        let Self { root, precision, scale, rng } = self;
        Self::select_with_rng_inner::<true, _>(root, *precision, *scale, &mut rng.0)
    }

    pub fn select_and_remove_with_rng(&mut self, rng: &mut impl Rng) -> Option<(u64, f64)> {
        Self::select_with_rng_inner::<true, _>(&mut self.root, self.precision, self.scale, rng)
    }

    pub fn select_wallenius_many_and_remove(
        &mut self,
        num_to_draw: u64,
    ) -> Option<Vec<(u64, f64)>> {
        if num_to_draw > self.count() {
            return None;
        }
        let Self { root, precision, scale, rng } = self;
        let mut selected = Vec::with_capacity(num_to_draw as usize);
        for _ in 0..num_to_draw {
            selected.push(Self::select_with_rng_inner::<true, _>(
                root,
                *precision,
                *scale,
                &mut rng.0,
            )?);
        }
        Some(selected)
    }

    pub fn select_wallenius_many_and_remove_with_rng(
        &mut self,
        num_to_draw: u64,
        rng: &mut impl Rng,
    ) -> Option<Vec<(u64, f64)>> {
        if num_to_draw > self.count() {
            return None;
        }
        let precision = self.precision;
        let scale = self.scale;
        let mut selected = Vec::with_capacity(num_to_draw as usize);
        for _ in 0..num_to_draw {
            selected.push(Self::select_with_rng_inner::<true, _>(
                &mut self.root,
                precision,
                scale,
                rng,
            )?);
        }
        Some(selected)
    }

    // Wrapper function to handle both select and select_and_remove
    pub fn select_and_optionally_remove(&mut self, with_removal: bool) -> Option<(u64, f64)> {
        if with_removal { self.select_and_remove() } else { self.select() }
    }

    fn select_with_rng_inner<const REMOVE: bool, R: Rng>(
        root: &mut Node<B>,
        precision: u8,
        scale: f64,
        rng: &mut R,
    ) -> Option<(u64, f64)> {
        if root.content_count == 0 {
            return None;
        }
        let random_target = rng.random_range(0..root.accumulated_value);
        let (selected_id, scaled_weight) = Self::select_recurse::<REMOVE, R>(
            root,
            random_target,
            1,
            precision,
            0,
            rng,
        )?;
        Some((selected_id, scaled_weight as f64 / scale))
    }

    fn select_recurse<const REMOVE: bool, R: Rng>(
        node: &mut Node<B>,
        target: u64,
        current_depth: u8,
        max_depth: u8,
        scaled_prefix: u64,
        rng: &mut R,
    ) -> Option<(u64, u64)> {
        if current_depth > max_depth {
            if let NodeContent::Bin(bin) = &mut node.content {
                if bin.is_empty() {
                    return None;
                }
                let selected_id = if REMOVE {
                    bin.get_random_and_remove(rng)?
                } else {
                    bin.get_random(rng)?
                };
                if REMOVE {
                    node.content_count -= 1;
                    node.accumulated_value -= scaled_prefix;
                }
                return Some((selected_id, scaled_prefix));
            }
            return None;
        }

        if let NodeContent::DigitIndex(children) = &mut node.content {
            let mut cumulative = 0;
            for position in 0..children.active_len as usize {
                let digit = children.order[position] as usize;
                if let Some(child) = children.children[digit].as_mut() {
                    if child.accumulated_value == 0 {
                        continue;
                    }
                    if target < cumulative + child.accumulated_value {
                        if let Some((selected_id, scaled_weight)) = Self::select_recurse::<REMOVE, R>(
                            child,
                            target - cumulative,
                            current_depth + 1,
                            max_depth,
                            scaled_prefix * 10 + digit as u64,
                            rng,
                        ) {
                            if REMOVE {
                                node.content_count -= 1;
                                node.accumulated_value -= scaled_weight;
                                children.child_decreased_at(position);
                            }
                            return Some((selected_id, scaled_weight));
                        }
                        return None;
                    }
                    cumulative += child.accumulated_value;
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
        self.select_many_with_rng(num_to_draw, with_removal, &mut rng)
    }

    fn select_many_with_rng(
        &mut self,
        num_to_draw: u64,
        with_removal: bool,
        rng: &mut impl Rng,
    ) -> Option<Vec<(u64, f64)>> {
        let max_scaled_weight = Self::max_scaled_weight(&self.root)?;
        if Self::use_unique_fisher_path(
            num_to_draw,
            self.root.accumulated_value,
            max_scaled_weight,
        ) {
            return self.select_fisher_by_unique_rejection(num_to_draw, with_removal, rng);
        }
        let mut bins = Vec::new();
        Self::collect_fisher_bins(&self.root, &mut bins);
        let counts = Self::sample_fisher_counts(&bins, num_to_draw, self.count(), rng);
        let mut selected: Vec<(u64, f64)> = Vec::with_capacity(num_to_draw as usize);
        let mut leaf_index = 0;
        Self::extract_fisher_sample(
            &mut self.root,
            &counts,
            &mut leaf_index,
            &mut selected,
            rng,
            with_removal,
            self.scale,
        )?;
        if selected.len() == num_to_draw as usize && leaf_index == counts.len() {
            Some(selected)
        } else {
            None
        }
    }

    fn use_unique_fisher_path(
        num_to_draw: u64,
        total_weight: u64,
        max_scaled_weight: u64,
    ) -> bool {
        if num_to_draw < 2 {
            return true;
        }
        let pair_count = num_to_draw as f64 * (num_to_draw - 1) as f64 * 0.5;
        let max_item_probability = max_scaled_weight as f64 / total_weight as f64;
        pair_count * max_item_probability <= 0.25
    }

    fn max_scaled_weight(node: &Node<B>) -> Option<u64> {
        match &node.content {
            NodeContent::DigitIndex(children) => children
                .iter()
                .rev()
                .flatten()
                .find(|child| child.content_count > 0)
                .and_then(Self::max_scaled_weight),
            NodeContent::Bin(bin) if !bin.is_empty() => {
                Some(node.accumulated_value / node.content_count)
            }
            NodeContent::Bin(_) => None,
        }
    }

    fn select_fisher_by_unique_rejection(
        &mut self,
        num_to_draw: u64,
        with_removal: bool,
        rng: &mut impl Rng,
    ) -> Option<Vec<(u64, f64)>> {
        let mut seen = HashSet::with_capacity(num_to_draw as usize);
        let mut sample = Vec::with_capacity(num_to_draw as usize);

        loop {
            seen.clear();
            sample.clear();
            let mut duplicate = false;
            for _ in 0..num_to_draw {
                let target = rng.random_range(0..self.root.accumulated_value);
                let (id, scaled_weight) = Self::select_scaled_recurse(
                    &self.root,
                    target,
                    1,
                    self.precision,
                    0,
                    rng,
                )?;
                if !seen.insert(id) {
                    duplicate = true;
                    break;
                }
                sample.push((id, scaled_weight));
            }
            if !duplicate {
                break;
            }
        }

        if with_removal {
            let mut digits = [0; MAX_PRECISION];
            for &(id, scaled_weight) in &sample {
                Self::scaled_to_digits(scaled_weight, self.precision, &mut digits);
                if !Self::remove_recurse(
                    &mut self.root,
                    id,
                    scaled_weight,
                    &digits,
                    1,
                    self.precision,
                ) {
                    return None;
                }
            }
        }

        Some(
            sample
                .into_iter()
                .map(|(id, scaled_weight)| (id, scaled_weight as f64 / self.scale))
                .collect(),
        )
    }

    fn select_scaled_recurse(
        node: &Node<B>,
        target: u64,
        current_depth: u8,
        max_depth: u8,
        scaled_prefix: u64,
        rng: &mut impl Rng,
    ) -> Option<(u64, u64)> {
        if current_depth > max_depth {
            if let NodeContent::Bin(bin) = &node.content {
                return bin.get_random(rng).map(|id| (id, scaled_prefix));
            }
            return None;
        }

        if let NodeContent::DigitIndex(children) = &node.content {
            let mut cumulative = 0;
            for &digit in children.ordered_digits() {
                let child = children.children[digit as usize].as_ref()?;
                if target < cumulative + child.accumulated_value {
                    return Self::select_scaled_recurse(
                        child,
                        target - cumulative,
                        current_depth + 1,
                        max_depth,
                        scaled_prefix * 10 + digit as u64,
                        rng,
                    );
                }
                cumulative += child.accumulated_value;
            }
        }
        None
    }

    fn scaled_to_digits(
        scaled_weight: u64,
        precision: u8,
        digits: &mut [u8; MAX_PRECISION],
    ) {
        let mut remaining = scaled_weight;
        for index in (0..precision as usize).rev() {
            digits[index] = (remaining % 10) as u8;
            remaining /= 10;
        }
    }

    fn collect_fisher_bins(node: &Node<B>, bins: &mut Vec<FisherBin>) {
        match &node.content {
            NodeContent::DigitIndex(children) => {
                for child in children.iter().flatten() {
                    Self::collect_fisher_bins(child, bins);
                }
            }
            NodeContent::Bin(bin) if !bin.is_empty() => bins.push(FisherBin {
                count: node.content_count,
                scaled_weight: node.accumulated_value / node.content_count,
            }),
            NodeContent::Bin(_) => {}
        }
    }

    fn sample_fisher_counts(
        bins: &[FisherBin],
        num_to_draw: u64,
        total_count: u64,
        rng: &mut impl Rng,
    ) -> Vec<u64> {
        if bins.len() == 1 {
            return vec![num_to_draw];
        }

        let use_complement = num_to_draw > total_count / 2;
        let target = if use_complement {
            total_count - num_to_draw
        } else {
            num_to_draw
        };
        if target == 0 {
            return bins.iter().map(|bin| bin.count).collect();
        }

        let log_weights: Vec<f64> = bins
            .iter()
            .map(|bin| {
                let log_weight = (bin.scaled_weight as f64).ln();
                if use_complement { -log_weight } else { log_weight }
            })
            .collect();
        let log_tilt = Self::find_fisher_log_tilt(bins, &log_weights, target);
        let probabilities: Vec<f64> = log_weights
            .iter()
            .map(|&log_weight| Self::logistic(log_tilt + log_weight))
            .collect();
        let distributions: Vec<Binomial> = bins
            .iter()
            .zip(&probabilities)
            .map(|(bin, &probability)| {
                Binomial::new(bin.count, probability)
                    .expect("valid Fisher binomial parameters")
            })
            .collect();

        let mut proposal = vec![0; bins.len()];
        loop {
            let mut proposal_total = 0;
            for (sample, distribution) in proposal.iter_mut().zip(&distributions) {
                *sample = distribution.sample(rng);
                proposal_total += *sample;
            }
            if proposal_total == target {
                if use_complement {
                    for (sample, bin) in proposal.iter_mut().zip(bins) {
                        *sample = bin.count - *sample;
                    }
                }
                return proposal;
            }
        }
    }

    fn find_fisher_log_tilt(
        bins: &[FisherBin],
        log_weights: &[f64],
        target: u64,
    ) -> f64 {
        let target = target as f64;
        let mut lower = -128.0;
        let mut upper = 128.0;
        let mut current = 0.0;

        for _ in 0..32 {
            let mut mean = 0.0;
            let mut variance = 0.0;
            for (bin, &log_weight) in bins.iter().zip(log_weights) {
                let probability = Self::logistic(current + log_weight);
                mean += bin.count as f64 * probability;
                variance += bin.count as f64 * probability * (1.0 - probability);
            }

            if mean < target {
                lower = current;
            } else {
                upper = current;
            }
            if (mean - target).abs() <= 1e-10 * target.max(1.0) {
                break;
            }

            let candidate = current - (mean - target) / variance;
            current = if candidate.is_finite() && candidate > lower && candidate < upper {
                candidate
            } else {
                (lower + upper) * 0.5
            };
        }
        current
    }

    fn logistic(value: f64) -> f64 {
        if value >= 0.0 {
            1.0 / (1.0 + (-value).exp())
        } else {
            let exp_value = value.exp();
            exp_value / (1.0 + exp_value)
        }
    }

    fn extract_fisher_sample(
        node: &mut Node<B>,
        counts: &[u64],
        leaf_index: &mut usize,
        selected: &mut Vec<(u64, f64)>,
        rng: &mut impl Rng,
        with_removal: bool,
        scale: f64,
    ) -> Option<()> {
        match &mut node.content {
            NodeContent::DigitIndex(children) => {
                for child in children.iter_mut().flatten() {
                    Self::extract_fisher_sample(
                        child,
                        counts,
                        leaf_index,
                        selected,
                        rng,
                        with_removal,
                        scale,
                    )?;
                }
            }
            NodeContent::Bin(bin) if !bin.is_empty() => {
                let count = *counts.get(*leaf_index)?;
                *leaf_index += 1;
                let scaled_weight = node.accumulated_value / node.content_count;
                let weight = scaled_weight as f64 / scale;
                let ids = bin.get_random_many(
                    rng,
                    usize::try_from(count).ok()?,
                    with_removal,
                )?;
                selected.extend(ids.into_iter().map(|id| (id, weight)));
                if with_removal {
                    node.content_count -= count;
                    node.accumulated_value -= scaled_weight * count;
                }
            }
            NodeContent::Bin(_) => {}
        }
        if with_removal {
            if let NodeContent::DigitIndex(children) = &mut node.content {
                node.content_count = children.iter().flatten().map(|child| child.content_count).sum();
                node.accumulated_value = children.iter().flatten().map(|child| child.accumulated_value).sum();
                children.rebuild_order();
            }
        }
        Some(())
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

        /// Create a DigitBinIndex with Vec<u32> bins and the specified precision.
        #[staticmethod]
        fn small(precision: u8) -> Self {
            PyDigitBinIndex {
                index: DigitBinIndex::small(precision),
            }
        }

        /// Create a DigitBinIndex with RoaringBitmap bins and the specified precision.
        #[staticmethod]
        fn medium(precision: u8) -> Self {
            PyDigitBinIndex {
                index: DigitBinIndex::medium(precision),
            }
        }

        /// Create a DigitBinIndex with RoaringTreemap bins and the specified precision.
        #[staticmethod]
        fn large(precision: u8) -> Self {
            PyDigitBinIndex {
                index: DigitBinIndex::large(precision),
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

        fn select_wallenius_many_and_remove(&mut self, n: u64) -> Option<Vec<(u64, f64)>> {
            self.index.select_wallenius_many_and_remove(n)
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

        fn print_stats(&self) {
            self.index.print_stats();
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
    use rand::rngs::StdRng;

    fn assert_mass_ordered<B: DigitBin>(node: &Node<B>) {
        if let NodeContent::DigitIndex(children) = &node.content {
            let ordered = children.ordered_digits();
            let active_digits: HashSet<_> = ordered.iter().copied().collect();
            assert_eq!(active_digits.len(), ordered.len());

            let expected_digits: HashSet<_> = children
                .iter()
                .enumerate()
                .filter(|(_, child)| child.as_ref().is_some_and(|child| child.accumulated_value > 0))
                .map(|(digit, _)| digit as u8)
                .collect();
            assert_eq!(active_digits, expected_digits);

            for pair in ordered.windows(2) {
                let left_digit = pair[0];
                let right_digit = pair[1];
                let left_mass = children.children[left_digit as usize]
                    .as_ref()
                    .unwrap()
                    .accumulated_value;
                let right_mass = children.children[right_digit as usize]
                    .as_ref()
                    .unwrap()
                    .accumulated_value;
                assert!(
                    left_mass > right_mass
                        || (left_mass == right_mass && left_digit > right_digit),
                    "digits {left_digit} and {right_digit} are out of mass order"
                );
            }

            for child in children.iter().flatten() {
                assert_mass_ordered(child);
            }
        }
    }

    #[test]
    fn test_digit_scan_order_tracks_mass() {
        let mut index = DigitBinIndexGeneric::<Vec<u32>>::with_precision(1);
        for id in 0..100 {
            index.add(id, 0.1);
        }
        index.add(100, 0.9);

        let NodeContent::DigitIndex(children) = &index.root.content else {
            panic!("root must be an internal node");
        };
        assert_eq!(children.ordered_digits(), &[1, 9]);
        assert_mass_ordered(&index.root);

        for id in 0..99 {
            assert!(index.remove(id, 0.1));
        }
        let NodeContent::DigitIndex(children) = &index.root.content else {
            panic!("root must be an internal node");
        };
        assert_eq!(children.ordered_digits(), &[9, 1]);
        assert_mass_ordered(&index.root);
    }

    #[test]
    fn test_mass_ordered_scan_preserves_weight_intervals() {
        let mut index = DigitBinIndexGeneric::<Vec<u32>>::with_precision(2);
        for (id, weight) in [(1, 0.01), (2, 0.01), (3, 0.17), (4, 0.52), (5, 0.99)] {
            index.add(id, weight);
        }
        let mut observed = std::collections::BTreeMap::new();
        let mut rng = StdRng::seed_from_u64(1);
        for target in 0..index.root.accumulated_value {
            let (_, scaled_weight) = DigitBinIndexGeneric::<Vec<u32>>::select_recurse::<false, _>(
                &mut index.root,
                target,
                1,
                index.precision,
                0,
                &mut rng,
            )
            .unwrap();
            *observed.entry(scaled_weight).or_insert(0u64) += 1;
        }

        assert_eq!(observed.get(&1), Some(&2));
        assert_eq!(observed.get(&17), Some(&17));
        assert_eq!(observed.get(&52), Some(&52));
        assert_eq!(observed.get(&99), Some(&99));
        assert_mass_ordered(&index.root);
    }

    #[test]
    fn test_fisher_small_exact_distribution() {
        const SIMULATIONS: usize = 13_000;
        let mut frequencies = [0usize; 3];
        let mut rng = StdRng::seed_from_u64(0xF15E_2026);

        for _ in 0..SIMULATIONS {
            let mut index = DigitBinIndexGeneric::<Vec<u32>>::with_precision(1);
            index.add(0, 0.1);
            index.add(1, 0.1);
            index.add(2, 0.2);
            index.add(3, 0.2);

            let selected = index.select_many_with_rng(2, false, &mut rng).unwrap();
            let high_weight_count = selected.iter().filter(|(_, weight)| *weight == 0.2).count();
            frequencies[high_weight_count] += 1;
        }

        let expected = [1.0 / 13.0, 8.0 / 13.0, 4.0 / 13.0];
        for (observed, expected) in frequencies.into_iter().zip(expected) {
            let observed = observed as f64 / SIMULATIONS as f64;
            assert!(
                (observed - expected).abs() < 0.015,
                "Observed Fisher probability {observed:.4} differs from {expected:.4}"
            );
        }
    }

    #[test]
    fn test_fisher_selection_is_unique_without_removal() {
        let mut index = DigitBinIndexGeneric::<Vec<u32>>::with_precision(3);
        for id in 0..100 {
            index.add(id, 0.5);
        }
        let mut rng = StdRng::seed_from_u64(42);
        let small_selection = index.select_many_with_rng(2, false, &mut rng).unwrap();
        let small_unique: HashSet<_> = small_selection.iter().map(|(id, _)| id).collect();
        assert_eq!(small_unique.len(), 2);
        assert_eq!(index.count(), 100);

        let selected = index.select_many_with_rng(100, false, &mut rng).unwrap();
        let unique: HashSet<_> = selected.iter().map(|(id, _)| id).collect();

        assert_eq!(selected.len(), 100);
        assert_eq!(unique.len(), 100);
        assert_eq!(index.count(), 100);
        assert_eq!(index.total_weight(), 50.0);
    }

    #[test]
    fn test_fisher_complement_removal_preserves_aggregates() {
        let mut index = DigitBinIndexGeneric::<Vec<u32>>::with_precision(1);
        for id in 0..5 {
            index.add(id, 0.1);
        }
        for id in 5..10 {
            index.add(id, 0.2);
        }
        let mut rng = StdRng::seed_from_u64(7);
        let selected = index.select_many_with_rng(9, true, &mut rng).unwrap();
        let removed_weight: f64 = selected.iter().map(|(_, weight)| weight).sum();

        assert_eq!(selected.len(), 9);
        assert_eq!(index.count(), 1);
        assert!((index.total_weight() - (1.5 - removed_weight)).abs() < f64::EPSILON);
        assert_mass_ordered(&index.root);
    }

    #[test]
    fn test_wallenius_batch_matches_repeated_draws() {
        let mut original = DigitBinIndexGeneric::<Vec<u32>>::with_precision(3);
        for id in 0..100 {
            original.add(id, (id % 9 + 1) as f64 / 10.0);
        }
        let initial_weight = original.total_weight();
        let mut batch_index = original.clone();
        let mut repeated_index = original;
        let mut batch_rng = StdRng::seed_from_u64(0xA11E_2026);
        let mut repeated_rng = StdRng::seed_from_u64(0xA11E_2026);

        let batch = batch_index
            .select_wallenius_many_and_remove_with_rng(50, &mut batch_rng)
            .unwrap();
        let repeated: Vec<_> = (0..50)
            .map(|_| repeated_index.select_and_remove_with_rng(&mut repeated_rng).unwrap())
            .collect();
        let removed_weight: f64 = batch.iter().map(|(_, weight)| weight).sum();

        assert_eq!(batch, repeated);
        assert_eq!(batch_index.count(), 50);
        assert_eq!(batch_index.count(), repeated_index.count());
        assert_eq!(batch_index.total_weight(), repeated_index.total_weight());
        assert!((batch_index.total_weight() - (initial_weight - removed_weight)).abs() < 1e-12);
        assert_mass_ordered(&batch_index.root);
        assert_mass_ordered(&repeated_index.root);
    }

    #[test]
    fn test_cloned_indexes_reproduce_persistent_rng_stream() {
        let mut original = DigitBinIndexGeneric::<Vec<u32>>::with_precision(3);
        for id in 0..20 {
            original.add(id, (id + 1) as f64 / 21.0);
        }
        let mut first = original.clone();
        let mut second = original;

        let first_sequence = first.select_wallenius_many_and_remove(10).unwrap();
        let second_sequence = second.select_wallenius_many_and_remove(10).unwrap();

        assert_eq!(first_sequence, second_sequence);
    }

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

        // --- Validation: Compare with the exact sequential Wallenius mean ---
        let avg_high_risk = total_high_risk_selected as f64 / NUM_SIMULATIONS as f64;
        let wallenius_mean = 618.050_138_498_636;
        let tolerance = wallenius_mean * 0.02;
        assert!((avg_high_risk - wallenius_mean).abs() < tolerance,
            "Wallenius test failed: Result {avg_high_risk:.2} was not close to {wallenius_mean:.2}");

        println!(
            "Wallenius test passed: Got avg {:.2} high-risk selections (expected ~{:.2}).",
            avg_high_risk, wallenius_mean
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
        let fishers_mean = 585.829_352_955_203;
        let tolerance = fishers_mean * 0.02;

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

