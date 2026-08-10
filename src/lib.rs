//! A `DigitBinIndex` is a tree-based data structure that organizes a large
//! collection of weighted items to enable highly efficient weighted random
//! selection and removal.
//!
//! It is a specialized tool, purpose-built for scenarios with millions of
//! items where probabilities are approximate and high performance is critical,
//! particularly for simulations involving sequential sampling like Wallenius'
//! noncentral hypergeometric distribution.

use rand::{distr::Distribution, Rng, SeedableRng};
use rand_distr::Binomial;
use roaring::{RoaringBitmap, RoaringTreemap};
use std::collections::HashSet;
use std::fmt;
use wyrand::WyRand;

// The default precision to use if none is specified in the constructor.
const DEFAULT_PRECISION: u8 = 3;
const MAX_PRECISION: usize = 9;
const FISHER_CONVOLUTION_BLOCK_SIZE: usize = 32;
const FISHER_CONVOLUTION_MIN_BINS: usize = 4_096;
const FISHER_CONVOLUTION_MAX_BIN_COUNT: u64 = 64;
const FISHER_CONVOLUTION_TAIL_BLOCKS: usize = 16;

/// Errors produced while constructing or mutating an index.
#[derive(Debug, Clone, PartialEq)]
pub enum DigitBinError {
    /// Precision must be in the inclusive range `1..=9`.
    InvalidPrecision(u64),
    /// Weights must be finite and strictly between zero and one.
    InvalidWeight(f64),
    /// The selected backend cannot represent this ID without truncation.
    IdOutOfRange { id: u64, maximum: u64 },
    /// IDs are unique across the complete index, including different weights.
    DuplicateId(u64),
    /// An aggregate count or scaled-weight sum would exceed `u64::MAX`.
    CapacityOverflow,
    /// Internal aggregate state was inconsistent during subtraction.
    InvariantViolation,
}

impl fmt::Display for DigitBinError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPrecision(precision) => {
                write!(
                    formatter,
                    "precision must be between 1 and {MAX_PRECISION}, got {precision}"
                )
            }
            Self::InvalidWeight(weight) => {
                write!(
                    formatter,
                    "weight must be finite and strictly between 0 and 1, got {weight}"
                )
            }
            Self::IdOutOfRange { id, maximum } => {
                write!(
                    formatter,
                    "ID {id} exceeds this backend's maximum ID {maximum}"
                )
            }
            Self::DuplicateId(id) => write!(formatter, "ID {id} is already present in the index"),
            Self::CapacityOverflow => {
                formatter.write_str("index aggregate capacity would overflow u64")
            }
            Self::InvariantViolation => {
                formatter.write_str("index aggregate invariants are inconsistent")
            }
        }
    }
}

impl std::error::Error for DigitBinError {}

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

#[derive(Debug)]
struct FisherConvolutionNode {
    start: usize,
    end: usize,
    pmf: Vec<f64>,
    left: Option<Box<FisherConvolutionNode>>,
    right: Option<Box<FisherConvolutionNode>>,
}

impl FisherConvolutionNode {
    fn build(start: usize, bins: &[FisherBin], probabilities: &[f64]) -> Option<Self> {
        if bins.is_empty() || bins.len() != probabilities.len() {
            return None;
        }
        if bins.len() == 1 {
            return Some(Self {
                start,
                end: start + 1,
                pmf: Self::binomial_pmf(bins[0].count, probabilities[0])?,
                left: None,
                right: None,
            });
        }

        let midpoint = bins.len() / 2;
        let left = Box::new(Self::build(
            start,
            &bins[..midpoint],
            &probabilities[..midpoint],
        )?);
        let right = Box::new(Self::build(
            start + midpoint,
            &bins[midpoint..],
            &probabilities[midpoint..],
        )?);
        let pmf = Self::convolve(&left.pmf, &right.pmf)?;
        Some(Self {
            start,
            end: start + bins.len(),
            pmf,
            left: Some(left),
            right: Some(right),
        })
    }

    fn binomial_pmf(count: u64, probability: f64) -> Option<Vec<f64>> {
        let count = usize::try_from(count).ok()?;
        if !(0.0..=1.0).contains(&probability) || !probability.is_finite() {
            return None;
        }
        if probability == 0.0 {
            let mut pmf = vec![0.0; count + 1];
            pmf[0] = 1.0;
            return Some(pmf);
        }
        if probability == 1.0 {
            let mut pmf = vec![0.0; count + 1];
            pmf[count] = 1.0;
            return Some(pmf);
        }

        let failure_probability = 1.0 - probability;
        let mode = (((count + 1) as f64 * probability).floor() as usize).min(count);
        let mut pmf = vec![0.0; count + 1];
        pmf[mode] = 1.0;

        for successes in (1..=mode).rev() {
            pmf[successes - 1] = pmf[successes] * successes as f64 / (count - successes + 1) as f64
                * failure_probability
                / probability;
        }
        for successes in mode..count {
            pmf[successes + 1] =
                pmf[successes] * (count - successes) as f64 / (successes + 1) as f64 * probability
                    / failure_probability;
        }

        Self::normalize(&mut pmf)?;
        Some(pmf)
    }

    fn convolve(left: &[f64], right: &[f64]) -> Option<Vec<f64>> {
        let mut result = vec![0.0; left.len() + right.len() - 1];
        for (left_count, &left_probability) in left.iter().enumerate() {
            if left_probability == 0.0 {
                continue;
            }
            for (right_count, &right_probability) in right.iter().enumerate() {
                result[left_count + right_count] += left_probability * right_probability;
            }
        }
        Self::normalize(&mut result)?;
        Some(result)
    }

    fn normalize(probabilities: &mut [f64]) -> Option<()> {
        let total: f64 = probabilities.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return None;
        }
        for probability in probabilities {
            *probability /= total;
        }
        Some(())
    }

    fn sample_conditioned(
        &self,
        total: usize,
        counts: &mut [u64],
        rng: &mut impl Rng,
    ) -> Option<()> {
        if total >= self.pmf.len() {
            return None;
        }
        let (Some(left), Some(right)) = (&self.left, &self.right) else {
            if self.end != self.start + 1 {
                return None;
            }
            counts[self.start] = total as u64;
            return Some(());
        };

        let left_max = left.pmf.len() - 1;
        let right_max = right.pmf.len() - 1;
        let minimum_left = total.saturating_sub(right_max);
        let maximum_left = total.min(left_max);
        let conditional_total: f64 = (minimum_left..=maximum_left)
            .map(|left_count| left.pmf[left_count] * right.pmf[total - left_count])
            .sum();
        if !conditional_total.is_finite() || conditional_total <= 0.0 {
            return None;
        }

        let mut threshold = rng.random::<f64>() * conditional_total;
        let mut selected_left = maximum_left;
        for left_count in minimum_left..=maximum_left {
            threshold -= left.pmf[left_count] * right.pmf[total - left_count];
            if threshold <= 0.0 {
                selected_left = left_count;
                break;
            }
        }
        left.sample_conditioned(selected_left, counts, rng)?;
        right.sample_conditioned(total - selected_left, counts, rng)
    }
}

#[derive(Debug)]
struct FisherConvolutionBlock {
    root: FisherConvolutionNode,
    cdf: Vec<f64>,
}

impl FisherConvolutionBlock {
    fn build(start: usize, bins: &[FisherBin], probabilities: &[f64]) -> Option<Self> {
        let root = FisherConvolutionNode::build(start, bins, probabilities)?;
        let mut cumulative = 0.0;
        let mut cdf = Vec::with_capacity(root.pmf.len());
        for &probability in &root.pmf {
            cumulative += probability;
            cdf.push(cumulative.min(1.0));
        }
        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }
        Some(Self { root, cdf })
    }

    fn sample_total(&self, rng: &mut impl Rng) -> usize {
        let threshold = rng.random::<f64>();
        self.cdf
            .partition_point(|&cumulative| cumulative < threshold)
            .min(self.cdf.len() - 1)
    }

    fn sample_conditioned(
        &self,
        total: usize,
        counts: &mut [u64],
        rng: &mut impl Rng,
    ) -> Option<()> {
        self.root.sample_conditioned(total, counts, rng)
    }

    fn variance(&self) -> f64 {
        let mean = self
            .root
            .pmf
            .iter()
            .enumerate()
            .map(|(count, &probability)| count as f64 * probability)
            .sum::<f64>();
        let second_moment = self
            .root
            .pmf
            .iter()
            .enumerate()
            .map(|(count, &probability)| (count * count) as f64 * probability)
            .sum::<f64>();
        (second_moment - mean * mean).max(0.0)
    }
}

#[derive(Debug)]
struct FisherBlockConvolutionNode {
    block_index: Option<usize>,
    pmf: Vec<f64>,
    left: Option<Box<FisherBlockConvolutionNode>>,
    right: Option<Box<FisherBlockConvolutionNode>>,
}

impl FisherBlockConvolutionNode {
    fn build(blocks: &[(usize, &FisherConvolutionBlock)]) -> Option<Self> {
        if blocks.is_empty() {
            return None;
        }
        if blocks.len() == 1 {
            let (block_index, block) = blocks[0];
            return Some(Self {
                block_index: Some(block_index),
                pmf: block.root.pmf.clone(),
                left: None,
                right: None,
            });
        }

        let midpoint = blocks.len() / 2;
        let left = Box::new(Self::build(&blocks[..midpoint])?);
        let right = Box::new(Self::build(&blocks[midpoint..])?);
        let pmf = FisherConvolutionNode::convolve(&left.pmf, &right.pmf)?;
        Some(Self {
            block_index: None,
            pmf,
            left: Some(left),
            right: Some(right),
        })
    }

    fn sample_conditioned(
        &self,
        total: usize,
        block_totals: &mut [usize],
        rng: &mut impl Rng,
    ) -> Option<()> {
        if total >= self.pmf.len() {
            return None;
        }
        let (Some(left), Some(right)) = (&self.left, &self.right) else {
            block_totals[*self.block_index.as_ref()?] = total;
            return Some(());
        };

        let left_max = left.pmf.len() - 1;
        let right_max = right.pmf.len() - 1;
        let minimum_left = total.saturating_sub(right_max);
        let maximum_left = total.min(left_max);
        let conditional_total: f64 = (minimum_left..=maximum_left)
            .map(|left_count| left.pmf[left_count] * right.pmf[total - left_count])
            .sum();
        if !conditional_total.is_finite() || conditional_total <= 0.0 {
            return None;
        }

        let mut threshold = rng.random::<f64>() * conditional_total;
        let mut selected_left = maximum_left;
        for left_count in minimum_left..=maximum_left {
            threshold -= left.pmf[left_count] * right.pmf[total - left_count];
            if threshold <= 0.0 {
                selected_left = left_count;
                break;
            }
        }
        left.sample_conditioned(selected_left, block_totals, rng)?;
        right.sample_conditioned(total - selected_left, block_totals, rng)
    }

    fn sample_block_totals(
        blocks: &[FisherConvolutionBlock],
        target: usize,
        tail_block_count: usize,
        rng: &mut impl Rng,
    ) -> Option<Vec<usize>> {
        if blocks.is_empty() || tail_block_count == 0 {
            return None;
        }

        let tail_block_count = tail_block_count.min(blocks.len());
        // A wider tail PMF has a lower peak, which increases the acceptance
        // probability below. Variance is therefore the useful budget metric.
        let mut candidates: Vec<(usize, f64)> = blocks
            .iter()
            .enumerate()
            .map(|(index, block)| (index, block.variance()))
            .collect();
        candidates.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
        let mut tail_indices: Vec<usize> = candidates
            .into_iter()
            .take(tail_block_count)
            .map(|(index, _)| index)
            .collect();
        tail_indices.sort_unstable();

        let mut is_tail = vec![false; blocks.len()];
        for &index in &tail_indices {
            is_tail[index] = true;
        }
        let tail_blocks: Vec<_> = tail_indices
            .iter()
            .map(|&index| (index, &blocks[index]))
            .collect();
        let tail = Self::build(&tail_blocks)?;
        let maximum_tail_probability = tail.pmf.iter().copied().fold(0.0, f64::max);
        if !maximum_tail_probability.is_finite() || maximum_tail_probability <= 0.0 {
            return None;
        }

        let mut block_totals = vec![0usize; blocks.len()];
        if tail_block_count == blocks.len() {
            let target_probability = *tail.pmf.get(target)?;
            if !target_probability.is_finite() || target_probability <= 0.0 {
                return None;
            }
            tail.sample_conditioned(target, &mut block_totals, rng)?;
            return Some(block_totals);
        }

        // For prefix total s, accepting with P(T = target - s) / max P(T)
        // changes the proposal mass from P(S = s) to a value proportional to
        // P(S = s) P(T = target - s), exactly the desired conditional law.
        loop {
            let mut prefix_total = 0usize;
            for (index, block) in blocks.iter().enumerate() {
                if !is_tail[index] {
                    let block_total = block.sample_total(rng);
                    block_totals[index] = block_total;
                    prefix_total = prefix_total.checked_add(block_total)?;
                }
            }

            let Some(residual) = target.checked_sub(prefix_total) else {
                continue;
            };
            let Some(&residual_probability) = tail.pmf.get(residual) else {
                continue;
            };
            if rng.random::<f64>() * maximum_tail_probability < residual_probability {
                tail.sample_conditioned(residual, &mut block_totals, rng)?;
                return Some(block_totals);
            }
        }
    }
}

#[derive(Clone, Copy)]
struct FisherExtractionSettings {
    with_removal: bool,
    scale: f64,
    order_generation: u64,
}

/// Trait for types that can be used as leaf bins in a `DigitBinIndex`.
///
/// Implement this trait for any container you want to use for storing IDs in the leaf nodes.
/// Provided implementations: [`Vec<u32>`], [`RoaringBitmap`].
pub trait DigitBin: Clone + Default {
    /// Largest ID representable by this backend without truncation.
    const MAX_ID: u64;

    /// Inserts an ID and reports whether it was newly stored.
    fn insert(&mut self, id: u64) -> bool;
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
    /// Approximate heap storage owned by the bin, excluding its inline value.
    fn estimated_heap_bytes(&self) -> usize;
}

impl DigitBin for Vec<u32> {
    const MAX_ID: u64 = u32::MAX as u64;

    fn insert(&mut self, id: u64) -> bool {
        let Ok(id) = u32::try_from(id) else {
            return false;
        };
        self.push(id);
        true
    }
    fn remove(&mut self, id: u64) -> bool {
        let Ok(id) = u32::try_from(id) else {
            return false;
        };
        if let Some(pos) = self.iter().position(|&x| x == id) {
            self.swap_remove(pos);
            true
        } else {
            false
        }
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn is_empty(&self) -> bool {
        self.is_empty()
    }
    fn get_random(&self, rng: &mut impl rand::Rng) -> Option<u64> {
        if self.is_empty() {
            None
        } else {
            Some(self[rng.random_range(0..self.len())] as u64)
        }
    }
    fn get_random_and_remove(&mut self, rng: &mut impl rand::Rng) -> Option<u64> {
        if self.is_empty() {
            None
        } else {
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

    fn estimated_heap_bytes(&self) -> usize {
        self.capacity() * std::mem::size_of::<u32>()
    }
}

impl DigitBin for RoaringBitmap {
    const MAX_ID: u64 = u32::MAX as u64;

    fn insert(&mut self, id: u64) -> bool {
        u32::try_from(id).is_ok_and(|id| self.insert(id))
    }
    fn remove(&mut self, id: u64) -> bool {
        u32::try_from(id).is_ok_and(|id| self.remove(id))
    }
    fn len(&self) -> usize {
        self.len() as usize
    }
    fn is_empty(&self) -> bool {
        self.is_empty()
    }
    fn get_random(&self, rng: &mut impl rand::Rng) -> Option<u64> {
        if self.is_empty() {
            None
        } else {
            let idx = rng.random_range(0..self.len() as u32);
            self.select(idx).map(|v| v as u64)
        }
    }
    fn get_random_and_remove(&mut self, rng: &mut impl rand::Rng) -> Option<u64> {
        if self.is_empty() {
            None
        } else {
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
            let selected: Option<Vec<u64>> =
                rand::seq::index::sample(rng, self.len() as usize, count)
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

    fn estimated_heap_bytes(&self) -> usize {
        self.serialized_size()
    }
}

impl DigitBin for RoaringTreemap {
    const MAX_ID: u64 = u64::MAX;

    fn insert(&mut self, id: u64) -> bool {
        self.insert(id)
    }
    fn remove(&mut self, id: u64) -> bool {
        self.remove(id)
    }
    fn len(&self) -> usize {
        self.len() as usize
    }
    fn is_empty(&self) -> bool {
        self.is_empty()
    }
    fn get_random(&self, rng: &mut impl rand::Rng) -> Option<u64> {
        if self.is_empty() {
            None
        } else {
            let idx = rng.random_range(0..self.len());
            self.select(idx)
        }
    }
    fn get_random_and_remove(&mut self, rng: &mut impl rand::Rng) -> Option<u64> {
        if self.is_empty() {
            None
        } else {
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
            let selected: Option<Vec<u64>> =
                rand::seq::index::sample(rng, self.len() as usize, count)
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

    fn estimated_heap_bytes(&self) -> usize {
        self.serialized_size()
    }
}

/// Direct digit lookup plus the optimal descending-mass order for selection.
#[derive(Debug, Clone)]
struct DigitIndex<B: DigitBin> {
    children: Box<[Option<Node<B>>; 10]>,
    order: [u8; 10],
    active_len: u8,
    active_mask: u16,
    order_generation: u64,
}

impl<B: DigitBin> DigitIndex<B> {
    fn new() -> Self {
        Self {
            children: Box::default(),
            order: [0; 10],
            active_len: 0,
            active_mask: 0,
            order_generation: 0,
        }
    }

    fn get_or_insert_with(
        &mut self,
        digit: usize,
        create: impl FnOnce() -> Node<B>,
    ) -> &mut Node<B> {
        let digit_mask = 1u16 << digit;
        if self.children[digit].is_none() {
            self.order[self.active_len as usize] = digit as u8;
            self.active_len += 1;
            self.active_mask |= digit_mask;
            self.children[digit] = Some(create());
        } else if self.active_mask & digit_mask == 0 {
            self.order[self.active_len as usize] = digit as u8;
            self.active_len += 1;
            self.active_mask |= digit_mask;
        }
        self.children[digit].as_mut().unwrap()
    }

    fn get_mut(&mut self, digit: usize) -> Option<&mut Node<B>> {
        self.children[digit].as_mut()
    }

    fn iter(&self) -> impl DoubleEndedIterator<Item = &Option<Node<B>>> {
        self.children.iter()
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
            self.active_mask &= !(1u16 << digit);
            self.children[digit as usize] = None;
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

    #[cfg(test)]
    fn ordered_digits(&self) -> &[u8] {
        &self.order[..self.active_len as usize]
    }

    fn ensure_order(&mut self, current_generation: u64) {
        if self.order_generation == current_generation {
            return;
        }

        let children = &self.children;
        self.order[..self.active_len as usize].sort_unstable_by(|&left, &right| {
            let left_mass = children[left as usize]
                .as_ref()
                .map_or(0, |child| child.accumulated_value);
            let right_mass = children[right as usize]
                .as_ref()
                .map_or(0, |child| child.accumulated_value);
            right_mass.cmp(&left_mass).then_with(|| right.cmp(&left))
        });
        self.order_generation = current_generation;
    }

    fn rebuild_order(&mut self, current_generation: u64) {
        self.active_len = 0;
        self.active_mask = 0;
        for digit in 0..10 {
            if self.children[digit]
                .as_ref()
                .is_some_and(|child| child.accumulated_value > 0)
            {
                self.order[self.active_len as usize] = digit as u8;
                self.active_len += 1;
                self.active_mask |= 1u16 << digit;
            }
        }
        self.order_generation = current_generation.wrapping_sub(1);
        self.ensure_order(current_generation);
    }

    fn remove_digit(&mut self, digit: usize) {
        self.children[digit] = None;
        let digit_mask = 1u16 << digit;
        if self.active_mask & digit_mask == 0 {
            return;
        }
        if let Some(position) = self.order[..self.active_len as usize]
            .iter()
            .position(|&active_digit| usize::from(active_digit) == digit)
        {
            let active_len = self.active_len as usize;
            self.order.copy_within(position + 1..active_len, position);
            self.active_len -= 1;
        }
        self.active_mask &= !digit_mask;
    }
}

/// The content of a node, which is either more nodes or a leaf with individuals.
#[derive(Debug, Clone)]
enum NodeContent<B: DigitBin> {
    /// An internal node that contains children for the next digit (0-9).
    DigitIndex(DigitIndex<B>),
    /// A leaf node that contains a bin of IDs for individuals in this bin.
    Bin(B),
}

/// A node within the DigitBinIndex tree.
#[derive(Debug, Clone)]
struct Node<B: DigitBin> {
    content: NodeContent<B>,
    accumulated_value: u64,
    content_count: u64,
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
/// but in return, tree traversal is O(P), where P is the configured precision.
/// Explicit removal from a `Vec` leaf is linear in that leaf's size, while exact
/// Fisher batching also depends on the number of occupied weight bins.
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

impl Default for DigitBinIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl DigitBinIndex {
    /// Creates a new DigitBinIndex with the given precision and expected capacity.
    /// Chooses a backend from expected average bin occupancy. Highly skewed
    /// workloads should prefer [`small`](Self::small), [`medium`](Self::medium),
    /// or [`large`](Self::large) explicitly after benchmarking representative data.
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
        Self::try_with_precision_and_capacity(precision, capacity)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// Validated variant of [`with_precision_and_capacity`](Self::with_precision_and_capacity).
    pub fn try_with_precision_and_capacity(
        precision: u8,
        capacity: u64,
    ) -> Result<Self, DigitBinError> {
        if precision == 0 || usize::from(precision) > MAX_PRECISION {
            return Err(DigitBinError::InvalidPrecision(u64::from(precision)));
        }
        let max_bins = 10u64.pow(precision as u32);
        if capacity / max_bins > 1_000_000_000 {
            Ok(DigitBinIndex::Large(
                DigitBinIndexGeneric::<RoaringTreemap>::try_with_precision(precision)?,
            ))
        } else if capacity / max_bins > 1_000 {
            Ok(DigitBinIndex::Medium(
                DigitBinIndexGeneric::<RoaringBitmap>::try_with_precision(precision)?,
            ))
        } else {
            Ok(DigitBinIndex::Small(
                DigitBinIndexGeneric::<Vec<u32>>::try_with_precision(precision)?,
            ))
        }
    }

    /// Creates a new DigitBinIndex with Vec<u32> bins and the specified precision.
    ///
    /// Optimized for small to medium-sized problems (average <= 1,000 items per bin).
    /// IDs above `u32::MAX` are rejected by mutation methods.
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
        Self::try_small(precision).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Validated variant of [`small`](Self::small).
    pub fn try_small(precision: u8) -> Result<Self, DigitBinError> {
        Ok(DigitBinIndex::Small(
            DigitBinIndexGeneric::<Vec<u32>>::try_with_precision(precision)?,
        ))
    }

    /// Creates a new DigitBinIndex with RoaringBitmap bins and the specified precision.
    ///
    /// Optimized for large-scale problems (average > 1,000 items per bin) where IDs fit within u32.
    /// Provides memory compression and fast set operations. IDs above
    /// `u32::MAX` are rejected by mutation methods.
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
        Self::try_medium(precision).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Validated variant of [`medium`](Self::medium).
    pub fn try_medium(precision: u8) -> Result<Self, DigitBinError> {
        Ok(DigitBinIndex::Medium(
            DigitBinIndexGeneric::<RoaringBitmap>::try_with_precision(precision)?,
        ))
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
        Self::try_large(precision).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Validated variant of [`large`](Self::large).
    pub fn try_large(precision: u8) -> Result<Self, DigitBinError> {
        Ok(DigitBinIndex::Large(
            DigitBinIndexGeneric::<RoaringTreemap>::try_with_precision(precision)?,
        ))
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
        Self::try_with_precision(precision).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Validated variant of [`with_precision`](Self::with_precision).
    pub fn try_with_precision(precision: u8) -> Result<Self, DigitBinError> {
        Self::try_small(precision)
    }

    /// Adds an item with the given ID and weight to the index.
    ///
    /// The weight must be finite and in `(0, 1)`. It is rounded to the nearest
    /// representable decimal bin, with positive boundary values clamped to the
    /// smallest or largest bin. IDs must be unique across the complete index.
    ///
    /// # Arguments
    ///
    /// * `individual_id` - The unique ID of the item to add.
    /// * `weight` - A finite weight strictly between zero and one.
    ///
    /// # Returns
    ///
    /// An error for invalid weights, duplicate IDs, backend ID overflow, or
    /// aggregate overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// use digit_bin_index::DigitBinIndex;
    ///
    /// let mut index = DigitBinIndex::new();
    /// index.add(1, 0.5).unwrap();
    /// assert_eq!(index.count(), 1);
    /// ```    
    pub fn add(&mut self, id: u64, weight: f64) -> Result<(), DigitBinError> {
        match self {
            DigitBinIndex::Small(index) => index.add(id, weight),
            DigitBinIndex::Medium(index) => index.add(id, weight),
            DigitBinIndex::Large(index) => index.add(id, weight),
        }
    }

    /// Adds multiple items after validating the complete input.
    ///
    /// Validation is atomic: no item is inserted if any weight, ID, duplicate, or
    /// aggregate-capacity check fails. Tree insertion remains one radix traversal
    /// per item; this method does not currently group equal weight paths.
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
    /// index.add_many(&items_to_add).unwrap();
    ///
    /// assert_eq!(index.count(), 3);
    /// // The total weight should be 0.1 + 0.2 + 0.1 = 0.4
    /// assert!((index.total_weight() - 0.4).abs() < f64::EPSILON);
    /// ```
    pub fn add_many(&mut self, items: &[(u64, f64)]) -> Result<(), DigitBinError> {
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
    /// index.add(1, 0.5).unwrap();
    /// index.remove(1, 0.5).unwrap();
    /// assert_eq!(index.count(), 0);
    /// ```
    pub fn remove(&mut self, id: u64, weight: f64) -> Result<bool, DigitBinError> {
        match self {
            DigitBinIndex::Small(index) => index.remove(id, weight),
            DigitBinIndex::Medium(index) => index.remove(id, weight),
            DigitBinIndex::Large(index) => index.remove(id, weight),
        }
    }

    /// Removes multiple items by applying the requested removals in order.
    ///
    /// The `(id, weight)` pairs must match items that are currently in the index.
    /// The operation is partially mutating: valid matches before a missing pair
    /// remain removed. It returns `Ok(true)` only when every pair was removed,
    /// and `Ok(false)` otherwise. Invalid IDs or weights are validated before any
    /// removals begin.
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
    /// index.add_many(&items_to_add).unwrap();
    /// assert_eq!(index.count(), 4);
    ///
    /// let items_to_remove = vec![(2, 0.2), (3, 0.1)];
    /// index.remove_many(&items_to_remove).unwrap();
    ///
    /// assert_eq!(index.count(), 2); // Items 1 and 4 should remain
    /// // The total weight should be 0.1 + 0.3 = 0.4
    /// assert!((index.total_weight() - 0.4).abs() < f64::EPSILON);
    /// ```
    pub fn remove_many(&mut self, items: &[(u64, f64)]) -> Result<bool, DigitBinError> {
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
    /// index.add(1, 0.5).unwrap();
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
    /// index.add(1, 0.5).unwrap();
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
    pub fn select_and_remove_with_rng(&mut self, rng: &mut impl Rng) -> Option<(u64, f64)> {
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
    /// index.add(1, 0.3).unwrap();
    /// index.add(2, 0.7).unwrap();
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
    /// index.add(1, 0.3).unwrap();
    /// index.add(2, 0.7).unwrap();
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
    /// The count of items as a `u64`.
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
    /// index.add(1, 0.5).unwrap();
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
            }
            DigitBinIndex::Medium(idx) => {
                println!("- Index Type: Medium (RoaringBitmap)");
                idx.print_stats_generic();
            }
            DigitBinIndex::Large(idx) => {
                println!("- Index Type: Large (RoaringTreemap)");
                idx.print_stats_generic();
            }
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
/// but in return, radix-tree traversal is O(P), where P is the configured
/// precision. Backend leaf operations and exact Fisher batching have additional
/// costs described by their respective methods.
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
    root: Node<B>,
    precision: u8,
    scale: f64,
    rng: IndexRng,
    order_generation: u64,
    ids: RoaringTreemap,
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
        Self::try_with_precision(precision).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Constructs an index after validating that `precision` is in `1..=9`.
    pub fn try_with_precision(precision: u8) -> Result<Self, DigitBinError> {
        if precision == 0 || usize::from(precision) > MAX_PRECISION {
            return Err(DigitBinError::InvalidPrecision(u64::from(precision)));
        }
        Ok(Self {
            root: Node::new_internal(),
            precision,
            scale: 10f64.powi(precision as i32),
            rng: IndexRng(WyRand::from_os_rng()),
            order_generation: 0,
            ids: RoaringTreemap::new(),
        })
    }

    /// Returns the number of decimal digits used to quantize weights.
    pub fn precision(&self) -> u8 {
        self.precision
    }

    /// Quantizes a finite weight in `(0, 1)` to the nearest representable bin.
    /// Values within half a bin of either boundary are clamped to the smallest
    /// or largest representable positive weight.
    fn weight_to_digits(
        &self,
        weight: f64,
        digits: &mut [u8; MAX_PRECISION],
    ) -> Result<u64, DigitBinError> {
        if !weight.is_finite() || weight <= 0.0 || weight >= 1.0 {
            return Err(DigitBinError::InvalidWeight(weight));
        }

        let maximum = self.scale as u64 - 1;
        let scaled = (weight * self.scale).round() as u64;
        let scaled = scaled.clamp(1, maximum);

        let mut temp = scaled;
        for i in (0..self.precision as usize).rev() {
            digits[i] = (temp % 10) as u8;
            temp /= 10;
        }
        Ok(scaled)
    }

    // --- Standard Functions ---

    pub fn add(&mut self, individual_id: u64, weight: f64) -> Result<(), DigitBinError> {
        if individual_id > B::MAX_ID {
            return Err(DigitBinError::IdOutOfRange {
                id: individual_id,
                maximum: B::MAX_ID,
            });
        }
        let mut digits = [0u8; MAX_PRECISION];
        let scaled = self.weight_to_digits(weight, &mut digits)?;
        self.root
            .content_count
            .checked_add(1)
            .ok_or(DigitBinError::CapacityOverflow)?;
        self.root
            .accumulated_value
            .checked_add(scaled)
            .ok_or(DigitBinError::CapacityOverflow)?;
        if !self.ids.insert(individual_id) {
            return Err(DigitBinError::DuplicateId(individual_id));
        }

        if !Self::add_recurse(
            &mut self.root,
            individual_id,
            scaled,
            &digits,
            1,
            self.precision,
        ) {
            self.ids.remove(individual_id);
            return Err(DigitBinError::InvariantViolation);
        }
        self.order_generation = self.order_generation.wrapping_add(1);
        Ok(())
    }

    /// Recursive private method to handle adding individuals.
    fn add_recurse(
        node: &mut Node<B>,
        individual_id: u64,
        scaled: u64, // Scaled weight as u64
        digits: &[u8; MAX_PRECISION],
        current_depth: u8,
        max_depth: u8,
    ) -> bool {
        if current_depth > max_depth {
            if let NodeContent::DigitIndex(_) = &node.content {
                node.content = NodeContent::Bin(B::default());
            }
            if let NodeContent::Bin(bin) = &mut node.content {
                if !bin.insert(individual_id) {
                    return false;
                }
            }
            node.content_count = node
                .content_count
                .checked_add(1)
                .expect("root aggregate preflight guarantees descendant capacity");
            node.accumulated_value = node
                .accumulated_value
                .checked_add(scaled)
                .expect("root aggregate preflight guarantees descendant capacity");
            return true;
        }

        let digit = digits[current_depth as usize - 1] as usize;
        if let NodeContent::DigitIndex(children) = &mut node.content {
            let child_node = children.get_or_insert_with(digit, || {
                if current_depth == max_depth {
                    Node::new_bin()
                } else {
                    Node::new_internal()
                }
            });
            if !Self::add_recurse(
                child_node,
                individual_id,
                scaled,
                digits,
                current_depth + 1,
                max_depth,
            ) {
                return false;
            }
        }
        node.content_count = node
            .content_count
            .checked_add(1)
            .expect("root aggregate preflight guarantees descendant capacity");
        node.accumulated_value = node
            .accumulated_value
            .checked_add(scaled)
            .expect("root aggregate preflight guarantees descendant capacity");
        true
    }

    /// Validates a batch atomically, then performs one radix traversal per item.
    /// Equal weight paths are not currently grouped for insertion.
    ///
    /// # Arguments
    ///
    /// * `items` - A slice of `(individual_id, weight)` tuples to add to the index.
    ///
    pub fn add_many(&mut self, items: &[(u64, f64)]) -> Result<(), DigitBinError> {
        if items.is_empty() {
            return Ok(());
        }

        let mut pending_ids = RoaringTreemap::new();
        let mut prepared = Vec::with_capacity(items.len());
        let mut total_scaled = 0u64;
        for &(id, weight) in items {
            if id > B::MAX_ID {
                return Err(DigitBinError::IdOutOfRange {
                    id,
                    maximum: B::MAX_ID,
                });
            }
            if !pending_ids.insert(id) {
                return Err(DigitBinError::DuplicateId(id));
            }
            let mut digits = [0u8; MAX_PRECISION];
            let scaled = self.weight_to_digits(weight, &mut digits)?;
            total_scaled = total_scaled
                .checked_add(scaled)
                .ok_or(DigitBinError::CapacityOverflow)?;
            prepared.push((id, scaled, digits));
        }
        let added_count =
            u64::try_from(prepared.len()).map_err(|_| DigitBinError::CapacityOverflow)?;
        self.root
            .content_count
            .checked_add(added_count)
            .ok_or(DigitBinError::CapacityOverflow)?;
        self.root
            .accumulated_value
            .checked_add(total_scaled)
            .ok_or(DigitBinError::CapacityOverflow)?;
        if !self.ids.is_disjoint(&pending_ids) {
            let duplicate = pending_ids
                .iter()
                .find(|&id| self.ids.contains(id))
                .expect("non-disjoint ID sets have an intersection");
            return Err(DigitBinError::DuplicateId(duplicate));
        }

        for (id, scaled, digits) in prepared {
            if !Self::add_recurse(&mut self.root, id, scaled, &digits, 1, self.precision) {
                return Err(DigitBinError::InvariantViolation);
            }
        }
        self.ids |= pending_ids;
        self.order_generation = self.order_generation.wrapping_add(1);
        Ok(())
    }

    pub fn remove(&mut self, individual_id: u64, weight: f64) -> Result<bool, DigitBinError> {
        if individual_id > B::MAX_ID {
            return Err(DigitBinError::IdOutOfRange {
                id: individual_id,
                maximum: B::MAX_ID,
            });
        }
        let mut digits = [0u8; MAX_PRECISION];
        let scaled = self.weight_to_digits(weight, &mut digits)?;
        if !self.ids.contains(individual_id) {
            return Ok(false);
        }
        let removed = Self::remove_recurse(
            &mut self.root,
            individual_id,
            scaled,
            &digits,
            1,
            self.precision,
        )?;
        if removed {
            if !self.ids.remove(individual_id) {
                return Err(DigitBinError::InvariantViolation);
            }
            self.order_generation = self.order_generation.wrapping_add(1);
        }
        Ok(removed)
    }

    /// Recursive private method to handle removing individuals.
    fn remove_recurse(
        node: &mut Node<B>,
        individual_id: u64,
        scaled: u64,
        digits: &[u8; MAX_PRECISION],
        current_depth: u8,
        max_depth: u8,
    ) -> Result<bool, DigitBinError> {
        if current_depth > max_depth {
            if let NodeContent::Bin(bin) = &mut node.content {
                if bin.remove(individual_id) {
                    node.content_count = node
                        .content_count
                        .checked_sub(1)
                        .ok_or(DigitBinError::InvariantViolation)?;
                    node.accumulated_value = node
                        .accumulated_value
                        .checked_sub(scaled)
                        .ok_or(DigitBinError::InvariantViolation)?;
                    return Ok(true);
                }
            }
            return Ok(false);
        }

        let digit = digits[current_depth as usize - 1] as usize;
        if let NodeContent::DigitIndex(children) = &mut node.content {
            let (removed, child_is_empty) = if let Some(child_node) = children.get_mut(digit) {
                let removed = Self::remove_recurse(
                    child_node,
                    individual_id,
                    scaled,
                    digits,
                    current_depth + 1,
                    max_depth,
                )?;
                (removed, removed && child_node.content_count == 0)
            } else {
                (false, false)
            };
            if removed {
                if child_is_empty {
                    children.remove_digit(digit);
                }
                node.content_count = node
                    .content_count
                    .checked_sub(1)
                    .ok_or(DigitBinError::InvariantViolation)?;
                node.accumulated_value = node
                    .accumulated_value
                    .checked_sub(scaled)
                    .ok_or(DigitBinError::InvariantViolation)?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Applies removals in input order after validating every ID and weight.
    ///
    /// The `(id, weight)` pairs must match items that are currently in the index.
    /// Returns `Ok(true)` only if every pair is removed. A missing pair makes the
    /// result `Ok(false)` but does not roll back earlier successful removals.
    ///
    /// # Arguments
    ///
    /// * `items` - A slice of `(id, weight)` tuples to remove from the index.
    ///
    pub fn remove_many(&mut self, items: &[(u64, f64)]) -> Result<bool, DigitBinError> {
        if items.is_empty() {
            return Ok(false);
        }

        for &(id, weight) in items {
            if id > B::MAX_ID {
                return Err(DigitBinError::IdOutOfRange {
                    id,
                    maximum: B::MAX_ID,
                });
            }
            let mut digits = [0u8; MAX_PRECISION];
            self.weight_to_digits(weight, &mut digits)?;
        }

        let mut success = true;
        for &(id, weight) in items {
            success &= self.remove(id, weight)?;
        }
        Ok(success)
    }

    // --- Selection Functions ---

    pub fn select(&mut self) -> Option<(u64, f64)> {
        let Self {
            root,
            precision,
            scale,
            rng,
            order_generation,
            ..
        } = self;
        Self::select_with_rng_inner::<false, _>(
            root,
            *precision,
            *scale,
            *order_generation,
            &mut rng.0,
        )
    }

    pub fn select_with_rng(&mut self, rng: &mut impl Rng) -> Option<(u64, f64)> {
        Self::select_with_rng_inner::<false, _>(
            &mut self.root,
            self.precision,
            self.scale,
            self.order_generation,
            rng,
        )
    }

    pub fn select_many(&mut self, num_to_draw: u64) -> Option<Vec<(u64, f64)>> {
        self.select_many_and_optionally_remove(num_to_draw, false)
    }

    pub fn select_and_remove(&mut self) -> Option<(u64, f64)> {
        let Self {
            root,
            precision,
            scale,
            rng,
            order_generation,
            ids,
        } = self;
        let selected = Self::select_with_rng_inner::<true, _>(
            root,
            *precision,
            *scale,
            *order_generation,
            &mut rng.0,
        )?;
        if !ids.remove(selected.0) {
            return None;
        }
        Some(selected)
    }

    pub fn select_and_remove_with_rng(&mut self, rng: &mut impl Rng) -> Option<(u64, f64)> {
        let selected = Self::select_with_rng_inner::<true, _>(
            &mut self.root,
            self.precision,
            self.scale,
            self.order_generation,
            rng,
        )?;
        if !self.ids.remove(selected.0) {
            return None;
        }
        Some(selected)
    }

    pub fn select_wallenius_many_and_remove(
        &mut self,
        num_to_draw: u64,
    ) -> Option<Vec<(u64, f64)>> {
        if num_to_draw > self.count() {
            return None;
        }
        let Self {
            root,
            precision,
            scale,
            rng,
            order_generation,
            ids,
        } = self;
        let mut selected = Vec::with_capacity(num_to_draw as usize);
        for _ in 0..num_to_draw {
            let item = Self::select_with_rng_inner::<true, _>(
                root,
                *precision,
                *scale,
                *order_generation,
                &mut rng.0,
            )?;
            if !ids.remove(item.0) {
                return None;
            }
            selected.push(item);
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
        let order_generation = self.order_generation;
        let mut selected = Vec::with_capacity(num_to_draw as usize);
        for _ in 0..num_to_draw {
            let item = Self::select_with_rng_inner::<true, _>(
                &mut self.root,
                precision,
                scale,
                order_generation,
                rng,
            )?;
            if !self.ids.remove(item.0) {
                return None;
            }
            selected.push(item);
        }
        Some(selected)
    }

    // Wrapper function to handle both select and select_and_remove
    pub fn select_and_optionally_remove(&mut self, with_removal: bool) -> Option<(u64, f64)> {
        if with_removal {
            self.select_and_remove()
        } else {
            self.select()
        }
    }

    fn select_with_rng_inner<const REMOVE: bool, R: Rng>(
        root: &mut Node<B>,
        precision: u8,
        scale: f64,
        order_generation: u64,
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
            order_generation,
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
        order_generation: u64,
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
                    node.content_count = node.content_count.checked_sub(1)?;
                    node.accumulated_value = node.accumulated_value.checked_sub(scaled_prefix)?;
                }
                return Some((selected_id, scaled_prefix));
            }
            return None;
        }

        if let NodeContent::DigitIndex(children) = &mut node.content {
            children.ensure_order(order_generation);
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
                            order_generation,
                            rng,
                        ) {
                            if REMOVE {
                                node.content_count = node.content_count.checked_sub(1)?;
                                node.accumulated_value =
                                    node.accumulated_value.checked_sub(scaled_weight)?;
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
    pub fn select_many_and_optionally_remove(
        &mut self,
        num_to_draw: u64,
        with_removal: bool,
    ) -> Option<Vec<(u64, f64)>> {
        if num_to_draw > self.count() || num_to_draw == 0 {
            return if num_to_draw == 0 {
                Some(Vec::new())
            } else {
                None
            };
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
        if Self::use_unique_fisher_path(num_to_draw, self.root.accumulated_value, max_scaled_weight)
        {
            return self.select_fisher_by_unique_rejection(num_to_draw, with_removal, rng);
        }
        let mut bins = Vec::new();
        Self::collect_fisher_bins(&self.root, &mut bins);
        let counts = Self::sample_fisher_counts(&bins, num_to_draw, self.count(), rng);
        let mut selected: Vec<(u64, f64)> = Vec::with_capacity(num_to_draw as usize);
        let mut leaf_index = 0;
        let extraction_settings = FisherExtractionSettings {
            with_removal,
            scale: self.scale,
            order_generation: self.order_generation,
        };
        Self::extract_fisher_sample(
            &mut self.root,
            &counts,
            &mut leaf_index,
            &mut selected,
            rng,
            extraction_settings,
        )?;
        if selected.len() == num_to_draw as usize && leaf_index == counts.len() {
            if with_removal {
                for &(id, _) in &selected {
                    if !self.ids.remove(id) {
                        return None;
                    }
                }
            }
            Some(selected)
        } else {
            None
        }
    }

    fn use_unique_fisher_path(num_to_draw: u64, total_weight: u64, max_scaled_weight: u64) -> bool {
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
                    &mut self.root,
                    target,
                    1,
                    self.precision,
                    0,
                    self.order_generation,
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
            self.order_generation = self.order_generation.wrapping_add(1);
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
                )
                .ok()?
                {
                    return None;
                }
                if !self.ids.remove(id) {
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
        node: &mut Node<B>,
        target: u64,
        current_depth: u8,
        max_depth: u8,
        scaled_prefix: u64,
        order_generation: u64,
        rng: &mut impl Rng,
    ) -> Option<(u64, u64)> {
        if current_depth > max_depth {
            if let NodeContent::Bin(bin) = &node.content {
                return bin.get_random(rng).map(|id| (id, scaled_prefix));
            }
            return None;
        }

        if let NodeContent::DigitIndex(children) = &mut node.content {
            children.ensure_order(order_generation);
            let mut cumulative = 0;
            for position in 0..children.active_len as usize {
                let digit = children.order[position];
                let child = children.children[digit as usize].as_mut()?;
                if target < cumulative + child.accumulated_value {
                    return Self::select_scaled_recurse(
                        child,
                        target - cumulative,
                        current_depth + 1,
                        max_depth,
                        scaled_prefix * 10 + u64::from(digit),
                        order_generation,
                        rng,
                    );
                }
                cumulative += child.accumulated_value;
            }
        }
        None
    }

    fn scaled_to_digits(scaled_weight: u64, precision: u8, digits: &mut [u8; MAX_PRECISION]) {
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
                if use_complement {
                    -log_weight
                } else {
                    log_weight
                }
            })
            .collect();
        let log_tilt = Self::find_fisher_log_tilt(bins, &log_weights, target);
        let probabilities: Vec<f64> = log_weights
            .iter()
            .map(|&log_weight| Self::logistic(log_tilt + log_weight))
            .collect();

        if let Some(mut proposal) =
            Self::sample_fisher_counts_by_partial_convolution(bins, &probabilities, target, rng)
        {
            if use_complement {
                for (sample, bin) in proposal.iter_mut().zip(bins) {
                    *sample = bin.count - *sample;
                }
            }
            return proposal;
        }

        let distributions: Vec<Binomial> = bins
            .iter()
            .zip(&probabilities)
            .map(|(bin, &probability)| {
                Binomial::new(bin.count, probability).expect("valid Fisher binomial parameters")
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

    fn sample_fisher_counts_by_partial_convolution(
        bins: &[FisherBin],
        probabilities: &[f64],
        target: u64,
        rng: &mut impl Rng,
    ) -> Option<Vec<u64>> {
        if bins.len() < FISHER_CONVOLUTION_MIN_BINS
            || bins.len() != probabilities.len()
            || bins
                .iter()
                .any(|bin| bin.count > FISHER_CONVOLUTION_MAX_BIN_COUNT)
        {
            return None;
        }

        let target = usize::try_from(target).ok()?;
        let mut blocks = Vec::with_capacity(bins.len().div_ceil(FISHER_CONVOLUTION_BLOCK_SIZE));
        for start in (0..bins.len()).step_by(FISHER_CONVOLUTION_BLOCK_SIZE) {
            let end = (start + FISHER_CONVOLUTION_BLOCK_SIZE).min(bins.len());
            blocks.push(FisherConvolutionBlock::build(
                start,
                &bins[start..end],
                &probabilities[start..end],
            )?);
        }

        let block_totals = FisherBlockConvolutionNode::sample_block_totals(
            &blocks,
            target,
            FISHER_CONVOLUTION_TAIL_BLOCKS,
            rng,
        )?;

        let mut counts = vec![0u64; bins.len()];
        for (block, &block_total) in blocks.iter().zip(&block_totals) {
            let start = block.root.start;
            let end = block.root.end;
            block.sample_conditioned(block_total, &mut counts, rng)?;
            debug_assert_eq!(counts[start..end].iter().sum::<u64>(), block_total as u64,);
        }
        Some(counts)
    }

    fn find_fisher_log_tilt(bins: &[FisherBin], log_weights: &[f64], target: u64) -> f64 {
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
        settings: FisherExtractionSettings,
    ) -> Option<()> {
        match &mut node.content {
            NodeContent::DigitIndex(children) => {
                for digit in 0..10 {
                    let child_is_empty = if let Some(child) = children.children[digit].as_mut() {
                        Self::extract_fisher_sample(
                            child, counts, leaf_index, selected, rng, settings,
                        )?;
                        settings.with_removal && child.content_count == 0
                    } else {
                        false
                    };
                    if child_is_empty {
                        children.remove_digit(digit);
                    }
                }
            }
            NodeContent::Bin(bin) if !bin.is_empty() => {
                let count = *counts.get(*leaf_index)?;
                *leaf_index += 1;
                let scaled_weight = node.accumulated_value / node.content_count;
                let weight = scaled_weight as f64 / settings.scale;
                let ids =
                    bin.get_random_many(rng, usize::try_from(count).ok()?, settings.with_removal)?;
                selected.extend(ids.into_iter().map(|id| (id, weight)));
                if settings.with_removal {
                    node.content_count = node.content_count.checked_sub(count)?;
                    node.accumulated_value = node
                        .accumulated_value
                        .checked_sub(scaled_weight.checked_mul(count)?)?;
                }
            }
            NodeContent::Bin(_) => {}
        }
        if settings.with_removal {
            if let NodeContent::DigitIndex(children) = &mut node.content {
                node.content_count = children
                    .iter()
                    .flatten()
                    .try_fold(0u64, |total, child| total.checked_add(child.content_count))?;
                node.accumulated_value =
                    children.iter().flatten().try_fold(0u64, |total, child| {
                        total.checked_add(child.accumulated_value)
                    })?;
                children.rebuild_order(settings.order_generation);
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
            include_node_storage: bool,
        ) {
            stats.node_count += 1;
            if include_node_storage {
                stats.mem_nodes += std::mem::size_of::<Node<B>>();
            }

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

                    for child in children.iter().flatten() {
                        // Child nodes live inline in the boxed ten-slot array,
                        // whose complete allocation was counted above.
                        traverse(child, stats, scale, false);
                    }
                }
                NodeContent::Bin(bin) => {
                    stats.bin_count += 1;
                    let bin_size = bin.len();
                    stats.bin_sizes.push(bin_size);
                    stats.total_bin_items += bin_size as u64;

                    stats.mem_bins += bin.estimated_heap_bytes();

                    if bin_size == 0 {
                        stats.empty_bin_count += 1;
                    } else {
                        // All items in a bin share the same weight.
                        let scaled_weight = node.accumulated_value / node.content_count;
                        let weight = scaled_weight as f64 / scale;
                        stats.min_weight =
                            Some(stats.min_weight.map_or(weight, |min| min.min(weight)));
                        stats.max_weight =
                            Some(stats.max_weight.map_or(weight, |max| max.max(weight)));
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

        traverse(&self.root, &mut stats, self.scale, true);

        // --- Calculations ---
        let fill_ratio = if stats.node_count > 0 {
            stats.non_empty_node_count as f64 / stats.node_count as f64 * 100.0
        } else {
            0.0
        };

        // NEW: Calculate average branching factor
        let avg_branching_factor = if stats.internal_node_count > 0 {
            stats.child_slots_used as f64 / stats.internal_node_count as f64
        } else {
            0.0
        };

        let avg_bin_size = if stats.bin_count > 0 {
            stats.total_bin_items as f64 / stats.bin_count as f64
        } else {
            0.0
        };

        let std_dev_bin_size = if stats.bin_count > 1 {
            let variance = stats
                .bin_sizes
                .iter()
                .map(|&size| (size as f64 - avg_bin_size).powi(2)) // Corrected: removed the `*`
                .sum::<f64>()
                / (stats.bin_count - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        // NEW: Calculate bin size quartiles
        let (q1_bin_size, median_bin_size, q3_bin_size) = if !stats.bin_sizes.is_empty() {
            let mut sorted_sizes = stats.bin_sizes.clone();
            sorted_sizes.sort_unstable();
            let q1 = sorted_sizes
                .get(sorted_sizes.len() / 4)
                .cloned()
                .unwrap_or(0);
            let median = sorted_sizes
                .get(sorted_sizes.len() / 2)
                .cloned()
                .unwrap_or(0);
            let q3 = sorted_sizes
                .get(sorted_sizes.len() * 3 / 4)
                .cloned()
                .unwrap_or(0);
            (q1, median, q3)
        } else {
            (0, 0, 0)
        };

        let ids_mem = self.ids.serialized_size();
        let total_mem_mb = (stats.mem_nodes + stats.mem_bins + ids_mem) as f64 / (1024.0 * 1024.0);
        let nodes_mem_mb = stats.mem_nodes as f64 / (1024.0 * 1024.0);
        let bins_mem_mb = stats.mem_bins as f64 / (1024.0 * 1024.0);
        let ids_mem_mb = ids_mem as f64 / (1024.0 * 1024.0);

        // NEW: Calculate average weight
        let avg_weight = if self.count() > 0 {
            self.total_weight() / self.count() as f64
        } else {
            0.0
        };

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
        println!("- Global ID Set:        {:.2} MB", ids_mem_mb);
        println!("- Total Estimated:      {:.2} MB", total_mem_mb);

        println!("\n[Items & Bins]");
        println!("- Total Items:          {}", stats.total_bin_items);
        println!("- Total Bins (Leaves):  {}", stats.bin_count);
        println!("- Empty Bins:           {}", stats.empty_bin_count);
        println!("- Avg Items per Bin:    {:.2}", avg_bin_size);
        println!("- Std Dev of Bin Size:  {:.2}", std_dev_bin_size);
        println!(
            "- Bin Size (min/max):   {} / {}",
            stats.bin_sizes.iter().min().map_or(0, |v| *v),
            stats.bin_sizes.iter().max().map_or(0, |v| *v)
        );
        println!(
            "- Bin Size (Q1/Med/Q3): {} / {} / {}",
            q1_bin_size, median_bin_size, q3_bin_size
        ); // NEW

        println!("\n[Weights]");
        println!(
            "- Smallest Weight:      {}",
            stats.min_weight.map_or("-".to_string(), |v| format!(
                "{:.prec$}",
                v,
                prec = self.precision as usize
            ))
        );
        println!(
            "- Largest Weight:       {}",
            stats.max_weight.map_or("-".to_string(), |v| format!(
                "{:.prec$}",
                v,
                prec = self.precision as usize
            ))
        );
        println!(
            "- Average Weight:       {:.prec$}",
            avg_weight,
            prec = self.precision as usize
        ); // NEW
    }
}

#[cfg(feature = "python-bindings")]
mod python {
    use super::*;
    use pyo3::exceptions::{PyOverflowError, PyRuntimeError, PyValueError};
    use pyo3::prelude::*;

    fn python_precision(precision: i64) -> PyResult<u8> {
        if !(1..=MAX_PRECISION as i64).contains(&precision) {
            return Err(PyValueError::new_err(format!(
                "precision must be between 1 and {MAX_PRECISION}, got {precision}"
            )));
        }
        Ok(precision as u8)
    }

    fn python_error(error: DigitBinError) -> PyErr {
        match error {
            DigitBinError::InvalidPrecision(_) | DigitBinError::InvalidWeight(_) => {
                PyValueError::new_err(error.to_string())
            }
            DigitBinError::IdOutOfRange { .. } | DigitBinError::CapacityOverflow => {
                PyOverflowError::new_err(error.to_string())
            }
            DigitBinError::DuplicateId(_) => PyValueError::new_err(error.to_string()),
            DigitBinError::InvariantViolation => PyRuntimeError::new_err(error.to_string()),
        }
    }

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
        fn with_precision(precision: i64) -> PyResult<Self> {
            let precision = python_precision(precision)?;
            Ok(PyDigitBinIndex {
                index: DigitBinIndex::try_with_precision(precision).map_err(python_error)?,
            })
        }

        /// Create a DigitBinIndex with a specific precision and expected capacity.
        #[staticmethod]
        fn with_precision_and_capacity(precision: i64, capacity: u64) -> PyResult<Self> {
            let precision = python_precision(precision)?;
            Ok(PyDigitBinIndex {
                index: DigitBinIndex::try_with_precision_and_capacity(precision, capacity)
                    .map_err(python_error)?,
            })
        }

        /// Create a DigitBinIndex with Vec<u32> bins and the specified precision.
        #[staticmethod]
        fn small(precision: i64) -> PyResult<Self> {
            let precision = python_precision(precision)?;
            Ok(PyDigitBinIndex {
                index: DigitBinIndex::try_small(precision).map_err(python_error)?,
            })
        }

        /// Create a DigitBinIndex with RoaringBitmap bins and the specified precision.
        #[staticmethod]
        fn medium(precision: i64) -> PyResult<Self> {
            let precision = python_precision(precision)?;
            Ok(PyDigitBinIndex {
                index: DigitBinIndex::try_medium(precision).map_err(python_error)?,
            })
        }

        /// Create a DigitBinIndex with RoaringTreemap bins and the specified precision.
        #[staticmethod]
        fn large(precision: i64) -> PyResult<Self> {
            let precision = python_precision(precision)?;
            Ok(PyDigitBinIndex {
                index: DigitBinIndex::try_large(precision).map_err(python_error)?,
            })
        }

        fn add(&mut self, id: u64, weight: f64) -> PyResult<()> {
            self.index.add(id, weight).map_err(python_error)
        }

        fn add_many(&mut self, items: Vec<(u64, f64)>) -> PyResult<()> {
            self.index.add_many(&items).map_err(python_error)
        }

        fn remove(&mut self, id: u64, weight: f64) -> PyResult<bool> {
            self.index.remove(id, weight).map_err(python_error)
        }

        fn remove_many(&mut self, items: Vec<(u64, f64)>) -> PyResult<bool> {
            self.index.remove_many(&items).map_err(python_error)
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

    fn assert_mass_ordered<B: DigitBin>(node: &mut Node<B>, order_generation: u64) {
        if let NodeContent::DigitIndex(children) = &mut node.content {
            children.ensure_order(order_generation);
            let ordered = children.ordered_digits();
            let active_digits: HashSet<_> = ordered.iter().copied().collect();
            assert_eq!(active_digits.len(), ordered.len());

            let expected_digits: HashSet<_> = children
                .iter()
                .enumerate()
                .filter(|(_, child)| {
                    child
                        .as_ref()
                        .is_some_and(|child| child.accumulated_value > 0)
                })
                .map(|(digit, _)| digit as u8)
                .collect();
            assert_eq!(active_digits, expected_digits);
            let expected_mask = expected_digits
                .iter()
                .fold(0u16, |mask, &digit| mask | (1u16 << digit));
            assert_eq!(children.active_mask, expected_mask);

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
                    left_mass > right_mass || (left_mass == right_mass && left_digit > right_digit),
                    "digits {left_digit} and {right_digit} are out of mass order"
                );
            }

            for child in children.children.iter_mut().flatten() {
                assert_mass_ordered(child, order_generation);
            }
        }
    }

    fn aggregate_tree<B: DigitBin>(node: &Node<B>) -> (u64, u64, usize) {
        match &node.content {
            NodeContent::DigitIndex(children) => {
                let mut count = 0u64;
                let mut mass = 0u64;
                let mut nodes = 1usize;
                for child in children.iter().flatten() {
                    let (child_count, child_mass, child_nodes) = aggregate_tree(child);
                    count = count.checked_add(child_count).unwrap();
                    mass = mass.checked_add(child_mass).unwrap();
                    nodes += child_nodes;
                }
                assert_eq!(node.content_count, count);
                assert_eq!(node.accumulated_value, mass);
                (count, mass, nodes)
            }
            NodeContent::Bin(bin) => {
                assert_eq!(node.content_count, bin.len() as u64);
                if node.content_count == 0 {
                    assert_eq!(node.accumulated_value, 0);
                } else {
                    assert_eq!(node.accumulated_value % node.content_count, 0);
                }
                (node.content_count, node.accumulated_value, 1)
            }
        }
    }

    fn assert_index_invariants<B: DigitBin>(index: &DigitBinIndexGeneric<B>) {
        let (count, mass, _) = aggregate_tree(&index.root);
        assert_eq!(count, index.count());
        assert_eq!(mass, index.root.accumulated_value);
        assert_eq!(count, index.ids.len());
    }

    fn exercise_backend_contract<B: DigitBin>() {
        let mut index = DigitBinIndexGeneric::<B>::with_precision(2);
        index.add(7, 0.29).unwrap();
        assert_eq!(index.add(7, 0.29), Err(DigitBinError::DuplicateId(7)));
        assert_eq!(index.add(7, 0.71), Err(DigitBinError::DuplicateId(7)));
        assert_eq!(index.count(), 1);
        assert_eq!(index.total_weight(), 0.29);
        assert!(!index.remove(7, 0.30).unwrap());
        assert_index_invariants(&index);
        assert!(index.remove(7, 0.29).unwrap());
        assert_eq!(index.count(), 0);
        assert_index_invariants(&index);
    }

    #[test]
    fn test_duplicate_and_wrong_weight_contract_across_backends() {
        exercise_backend_contract::<Vec<u32>>();
        exercise_backend_contract::<RoaringBitmap>();
        exercise_backend_contract::<RoaringTreemap>();
    }

    #[test]
    fn test_id_boundaries_for_all_public_constructors() {
        let mut u32_indexes = vec![
            DigitBinIndex::new(),
            DigitBinIndex::with_precision(2),
            DigitBinIndex::small(2),
            DigitBinIndex::medium(2),
            DigitBinIndex::with_precision_and_capacity(2, 100),
            DigitBinIndex::with_precision_and_capacity(1, 100_010),
        ];
        for index in &mut u32_indexes {
            index.add(u32::MAX as u64, 0.5).unwrap();
            assert_eq!(
                index.add(u32::MAX as u64 + 1, 0.5),
                Err(DigitBinError::IdOutOfRange {
                    id: u32::MAX as u64 + 1,
                    maximum: u32::MAX as u64,
                })
            );
            assert_eq!(index.count(), 1);
        }

        let mut u64_indexes = vec![
            DigitBinIndex::large(2),
            DigitBinIndex::with_precision_and_capacity(1, 10_000_000_010),
        ];
        for index in &mut u64_indexes {
            index.add(u32::MAX as u64, 0.5).unwrap();
            index.add(u32::MAX as u64 + 1, 0.5).unwrap();
            assert_eq!(index.count(), 2);
        }
    }

    #[test]
    fn test_validated_weight_contract_and_rounding() {
        let mut index = DigitBinIndexGeneric::<Vec<u32>>::with_precision(2);
        for weight in [f64::NAN, f64::NEG_INFINITY, f64::INFINITY, -0.1, 0.0, 1.0] {
            assert!(matches!(
                index.add(1, weight),
                Err(DigitBinError::InvalidWeight(_))
            ));
            assert_eq!(index.count(), 0);
        }

        index.add(1, f64::MIN_POSITIVE).unwrap();
        index.add(2, 0.29).unwrap();
        index.add(3, 0.999_999).unwrap();
        assert_eq!(index.count(), 3);
        assert!((index.total_weight() - 1.29).abs() < f64::EPSILON);
        assert_index_invariants(&index);
    }

    #[test]
    fn test_precision_and_aggregate_boundaries() {
        assert_eq!(
            DigitBinIndexGeneric::<Vec<u32>>::try_with_precision(0).unwrap_err(),
            DigitBinError::InvalidPrecision(0)
        );
        assert_eq!(
            DigitBinIndexGeneric::<Vec<u32>>::try_with_precision(10).unwrap_err(),
            DigitBinError::InvalidPrecision(10)
        );
        assert_eq!(
            DigitBinIndex::try_with_precision_and_capacity(0, u64::MAX).unwrap_err(),
            DigitBinError::InvalidPrecision(0)
        );

        let mut count_overflow = DigitBinIndexGeneric::<Vec<u32>>::with_precision(1);
        count_overflow.root.content_count = u64::MAX;
        assert_eq!(
            count_overflow.add(1, 0.1),
            Err(DigitBinError::CapacityOverflow)
        );

        let mut mass_overflow = DigitBinIndexGeneric::<Vec<u32>>::with_precision(1);
        mass_overflow.root.accumulated_value = u64::MAX;
        assert_eq!(
            mass_overflow.add(1, 0.1),
            Err(DigitBinError::CapacityOverflow)
        );
    }

    #[test]
    fn test_add_many_is_atomic_on_validation_errors() {
        let mut index = DigitBinIndexGeneric::<Vec<u32>>::with_precision(2);
        assert_eq!(
            index.add_many(&[(1, 0.1), (1, 0.2)]),
            Err(DigitBinError::DuplicateId(1))
        );
        assert_eq!(index.count(), 0);
        assert_index_invariants(&index);

        assert!(matches!(
            index.add_many(&[(1, 0.1), (2, f64::NAN)]),
            Err(DigitBinError::InvalidWeight(_))
        ));
        assert_eq!(index.count(), 0);
        assert_index_invariants(&index);
    }

    #[test]
    fn test_removal_prunes_empty_weight_paths() {
        let mut index = DigitBinIndexGeneric::<Vec<u32>>::with_precision(4);
        let items: Vec<_> = (1..=2_000).map(|id| (id, id as f64 / 10_000.0)).collect();
        index.add_many(&items).unwrap();
        let (_, _, populated_nodes) = aggregate_tree(&index.root);
        assert!(populated_nodes > 2_000);

        assert!(index.remove_many(&items).unwrap());
        let (_, _, remaining_nodes) = aggregate_tree(&index.root);
        assert_eq!(remaining_nodes, 1);
        assert_eq!(index.count(), 0);
        assert_index_invariants(&index);
    }

    #[test]
    fn test_zero_draws_preserve_aggregates() {
        let mut index = DigitBinIndexGeneric::<Vec<u32>>::with_precision(2);
        index.add_many(&[(1, 0.2), (2, 0.8)]).unwrap();
        let before = (index.count(), index.root.accumulated_value);
        assert_eq!(index.select_many(0), Some(Vec::new()));
        assert_eq!(index.select_many_and_remove(0), Some(Vec::new()));
        assert_eq!((index.count(), index.root.accumulated_value), before);
        assert_index_invariants(&index);
    }

    #[test]
    fn test_digit_scan_order_tracks_mass() {
        let mut index = DigitBinIndexGeneric::<Vec<u32>>::with_precision(1);
        for id in 0..100 {
            index.add(id, 0.1).unwrap();
        }
        index.add(100, 0.9).unwrap();

        let NodeContent::DigitIndex(children) = &mut index.root.content else {
            panic!("root must be an internal node");
        };
        assert_ne!(children.order_generation, index.order_generation);
        children.ensure_order(index.order_generation);
        assert_eq!(children.order_generation, index.order_generation);
        assert_eq!(children.ordered_digits(), &[1, 9]);
        assert_mass_ordered(&mut index.root, index.order_generation);

        for id in 0..99 {
            assert!(index.remove(id, 0.1).unwrap());
        }
        let NodeContent::DigitIndex(children) = &mut index.root.content else {
            panic!("root must be an internal node");
        };
        children.ensure_order(index.order_generation);
        assert_eq!(children.ordered_digits(), &[9, 1]);
        assert_mass_ordered(&mut index.root, index.order_generation);
    }

    #[test]
    fn test_lazy_order_reactivates_an_empty_path() {
        let mut index = DigitBinIndexGeneric::<Vec<u32>>::with_precision(1);
        index.add(1, 0.1).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        assert_eq!(index.select_and_remove_with_rng(&mut rng), Some((1, 0.1)));
        assert_eq!(index.count(), 0);

        index.add(2, 0.1).unwrap();
        assert_eq!(index.select_with_rng(&mut rng), Some((2, 0.1)));
        assert_mass_ordered(&mut index.root, index.order_generation);
    }

    #[test]
    fn test_mass_ordered_scan_preserves_weight_intervals() {
        let mut index = DigitBinIndexGeneric::<Vec<u32>>::with_precision(2);
        for (id, weight) in [(1, 0.01), (2, 0.01), (3, 0.17), (4, 0.52), (5, 0.99)] {
            index.add(id, weight).unwrap();
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
                index.order_generation,
                &mut rng,
            )
            .unwrap();
            *observed.entry(scaled_weight).or_insert(0u64) += 1;
        }

        assert_eq!(observed.get(&1), Some(&2));
        assert_eq!(observed.get(&17), Some(&17));
        assert_eq!(observed.get(&52), Some(&52));
        assert_eq!(observed.get(&99), Some(&99));
        assert_mass_ordered(&mut index.root, index.order_generation);
    }

    #[test]
    fn test_fisher_small_exact_distribution() {
        const SIMULATIONS: usize = 13_000;
        let mut frequencies = [0usize; 3];
        let mut rng = StdRng::seed_from_u64(0xF15E_2026);

        for _ in 0..SIMULATIONS {
            let mut index = DigitBinIndexGeneric::<Vec<u32>>::with_precision(1);
            index.add(0, 0.1).unwrap();
            index.add(1, 0.1).unwrap();
            index.add(2, 0.2).unwrap();
            index.add(3, 0.2).unwrap();

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
    fn test_fisher_convolution_tree_matches_direct_distribution() {
        let bins = [
            FisherBin {
                count: 2,
                scaled_weight: 1,
            },
            FisherBin {
                count: 1,
                scaled_weight: 2,
            },
            FisherBin {
                count: 3,
                scaled_weight: 3,
            },
        ];
        let probabilities = [0.2, 0.5, 0.7];
        let root = FisherConvolutionNode::build(0, &bins, &probabilities).unwrap();

        fn choose(n: usize, k: usize) -> f64 {
            (0..k.min(n - k)).fold(1.0, |value, index| {
                value * (n - index) as f64 / (index + 1) as f64
            })
        }

        let mut expected = vec![0.0; 7];
        let mut conditioned_probability = 0.0;
        let mut conditioned_first_count = 0.0;
        for first in 0..=2 {
            for second in 0..=1 {
                for third in 0..=3 {
                    let counts = [first, second, third];
                    let capacities = [2, 1, 3];
                    let probability = counts
                        .into_iter()
                        .zip(capacities)
                        .zip(probabilities)
                        .map(|((count, capacity), probability)| {
                            choose(capacity, count)
                                * probability.powi(count as i32)
                                * (1.0 - probability).powi((capacity - count) as i32)
                        })
                        .product::<f64>();
                    expected[first + second + third] += probability;
                    if first + second + third == 3 {
                        conditioned_probability += probability;
                        conditioned_first_count += first as f64 * probability;
                    }
                }
            }
        }

        for (observed, expected) in root.pmf.iter().zip(&expected) {
            assert!((observed - expected).abs() < 1e-12);
        }

        let expected_first_count = conditioned_first_count / conditioned_probability;
        let mut observed_first_count = 0u64;
        let mut rng = StdRng::seed_from_u64(0xC011_D170);
        for _ in 0..20_000 {
            let mut counts = [0u64; 3];
            root.sample_conditioned(3, &mut counts, &mut rng).unwrap();
            assert_eq!(counts.iter().sum::<u64>(), 3);
            observed_first_count += counts[0];
        }
        let observed_first_count = observed_first_count as f64 / 20_000.0;
        assert!((observed_first_count - expected_first_count).abs() < 0.02);
    }

    #[test]
    fn test_partial_fisher_convolution_returns_conditioned_counts() {
        let bins = vec![
            FisherBin {
                count: 1,
                scaled_weight: 1,
            };
            FISHER_CONVOLUTION_MIN_BINS
        ];
        let probabilities = vec![0.5; bins.len()];
        let target = bins.len() as u64 / 2;
        let mut rng = StdRng::seed_from_u64(0xC011_2026);
        let counts = DigitBinIndexGeneric::<Vec<u32>>::sample_fisher_counts_by_partial_convolution(
            &bins,
            &probabilities,
            target,
            &mut rng,
        )
        .unwrap();

        assert_eq!(counts.len(), bins.len());
        assert_eq!(counts.iter().sum::<u64>(), target);
        assert!(counts.iter().all(|&count| count <= 1));
    }

    #[test]
    fn test_fisher_tail_tree_matches_conditioned_block_distribution() {
        const SIMULATIONS: usize = 40_000;
        let probabilities = [0.1_f64, 0.3, 0.6, 0.8];
        let bins = [FisherBin {
            count: 1,
            scaled_weight: 1,
        }];
        let blocks: Vec<_> = probabilities
            .iter()
            .enumerate()
            .map(|(index, &probability)| {
                FisherConvolutionBlock::build(index, &bins, &[probability]).unwrap()
            })
            .collect();

        let mut expected = [0.0; 16];
        let mut expected_total = 0.0;
        for mask in 0u8..16 {
            if mask.count_ones() != 2 {
                continue;
            }
            let probability = probabilities
                .iter()
                .enumerate()
                .map(|(index, &success_probability)| {
                    if mask & (1 << index) != 0 {
                        success_probability
                    } else {
                        1.0 - success_probability
                    }
                })
                .product::<f64>();
            expected_total += probability;
            expected[mask as usize] = probability;
        }
        for probability in &mut expected {
            *probability /= expected_total;
        }

        let mut observed = [0usize; 16];
        let mut rng = StdRng::seed_from_u64(0x7A11_2026);
        for _ in 0..SIMULATIONS {
            let totals =
                FisherBlockConvolutionNode::sample_block_totals(&blocks, 2, 2, &mut rng).unwrap();
            assert_eq!(totals.iter().sum::<usize>(), 2);
            let mask = totals
                .iter()
                .enumerate()
                .fold(0usize, |mask, (index, &count)| mask | (count << index));
            observed[mask] += 1;
        }

        for mask in 0usize..16 {
            let observed = observed[mask] as f64 / SIMULATIONS as f64;
            assert!(
                (observed - expected[mask]).abs() < 0.006,
                "mask {mask:04b}: observed {observed:.4}, expected {:.4}",
                expected[mask]
            );
        }
    }

    #[test]
    fn test_fisher_selection_is_unique_without_removal() {
        let mut index = DigitBinIndexGeneric::<Vec<u32>>::with_precision(3);
        for id in 0..100 {
            index.add(id, 0.5).unwrap();
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
            index.add(id, 0.1).unwrap();
        }
        for id in 5..10 {
            index.add(id, 0.2).unwrap();
        }
        let mut rng = StdRng::seed_from_u64(7);
        let selected = index.select_many_with_rng(9, true, &mut rng).unwrap();
        let removed_weight: f64 = selected.iter().map(|(_, weight)| weight).sum();

        assert_eq!(selected.len(), 9);
        assert_eq!(index.count(), 1);
        assert!((index.total_weight() - (1.5 - removed_weight)).abs() < f64::EPSILON);
        assert_mass_ordered(&mut index.root, index.order_generation);
    }

    #[test]
    fn test_wallenius_batch_matches_repeated_draws() {
        let mut original = DigitBinIndexGeneric::<Vec<u32>>::with_precision(3);
        for id in 0..100 {
            original.add(id, (id % 9 + 1) as f64 / 10.0).unwrap();
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
            .map(|_| {
                repeated_index
                    .select_and_remove_with_rng(&mut repeated_rng)
                    .unwrap()
            })
            .collect();
        let removed_weight: f64 = batch.iter().map(|(_, weight)| weight).sum();

        assert_eq!(batch, repeated);
        assert_eq!(batch_index.count(), 50);
        assert_eq!(batch_index.count(), repeated_index.count());
        assert_eq!(batch_index.total_weight(), repeated_index.total_weight());
        assert!((batch_index.total_weight() - (initial_weight - removed_weight)).abs() < 1e-12);
        assert_mass_ordered(&mut batch_index.root, batch_index.order_generation);
        assert_mass_ordered(&mut repeated_index.root, repeated_index.order_generation);
    }

    #[test]
    fn test_cloned_indexes_reproduce_persistent_rng_stream() {
        let mut original = DigitBinIndexGeneric::<Vec<u32>>::with_precision(3);
        for id in 0..20 {
            original.add(id, (id + 1) as f64 / 21.0).unwrap();
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
        index.add(1, 0.085).unwrap();
        index.add(2, 0.205).unwrap();
        index.add(3, 0.346).unwrap();
        index.add(4, 0.364).unwrap();
        index.print_stats();
        println!(
            "Initial state: {} individuals, total weight = {}",
            index.count(),
            index.total_weight()
        );
        if let Some((id, weight)) = index.select_and_remove() {
            println!("Selected ID: {} with weight: {}", id, weight);
        }
        assert!(
            index.count() == 3,
            "The count is now {} and not 3 as expected",
            index.count()
        );
        println!(
            "Intermediate state: {} individuals, total weight = {}",
            index.count(),
            index.total_weight()
        );
        if let Some(selection) = index.select_many_and_remove(2) {
            println!("Selection: {:?}", selection);
        }
        assert!(
            index.count() == 1,
            "The count is now {} and not 1 as expected",
            index.count()
        );
        println!(
            "Final state: {} individuals, total weight = {}",
            index.count(),
            index.total_weight()
        );
    }

    #[test]
    fn test_wallenius_distribution_is_correct() {
        // --- Setup: Create a controlled population ---
        const ITEMS_PER_GROUP: u64 = 1000;
        const TOTAL_ITEMS: u64 = ITEMS_PER_GROUP * 2;
        const NUM_DRAWS: u64 = TOTAL_ITEMS / 2;

        let low_risk_weight = 0.1f64; // 0.1
        let high_risk_weight = 0.2f64; // 0.2

        // --- Execution: Run many simulations to average out randomness ---
        const NUM_SIMULATIONS: u32 = 100;
        let mut total_high_risk_selected = 0;

        for _ in 0..NUM_SIMULATIONS {
            let mut index = DigitBinIndex::with_precision_and_capacity(3, TOTAL_ITEMS);
            for i in 0..ITEMS_PER_GROUP {
                index.add(i, low_risk_weight).unwrap();
            }
            for i in ITEMS_PER_GROUP..TOTAL_ITEMS {
                index.add(i, high_risk_weight).unwrap();
            }

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
        assert!(
            (avg_high_risk - wallenius_mean).abs() < tolerance,
            "Wallenius test failed: Result {avg_high_risk:.2} was not close to {wallenius_mean:.2}"
        );

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

        let low_risk_weight = 0.1f64; // 0.1
        let high_risk_weight = 0.2f64; // 0.2

        const NUM_SIMULATIONS: u32 = 100;
        let mut total_high_risk_selected = 0;

        for _ in 0..NUM_SIMULATIONS {
            let mut index = DigitBinIndex::with_precision_and_capacity(3, TOTAL_ITEMS);
            for i in 0..ITEMS_PER_GROUP {
                index.add(i, low_risk_weight).unwrap();
            }
            for i in ITEMS_PER_GROUP..TOTAL_ITEMS {
                index.add(i, high_risk_weight).unwrap();
            }

            // Call the new method
            if let Some(selected_ids) = index.select_many_and_remove(NUM_DRAWS) {
                let high_risk_in_this_run = selected_ids
                    .iter()
                    .filter(|&&(id, _)| id >= ITEMS_PER_GROUP)
                    .count();
                total_high_risk_selected += high_risk_in_this_run as u32;
            }
        }

        let avg_high_risk = total_high_risk_selected as f64 / NUM_SIMULATIONS as f64;
        let fishers_mean = 585.829_352_955_203;
        let tolerance = fishers_mean * 0.02;

        assert!(
            (avg_high_risk - fishers_mean).abs() < tolerance,
            "Fisher's test failed: Result {:.2} was not close to the expected mean of {:.2}",
            avg_high_risk,
            fishers_mean
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
    if let Ok(scaled) = index.weight_to_digits(0.123, &mut digits) {
        assert_eq!(scaled, 123);
        assert_eq!(digits[0..3], [1, 2, 3]);
        assert_eq!(digits[3..], [0; 6]); // Remaining digits should be zero-padded
    } else {
        panic!("expected a valid quantized weight");
    }

    // Test invalid weights
    assert!(index.weight_to_digits(0.0, &mut digits).is_err());
    assert!(index.weight_to_digits(-0.1, &mut digits).is_err());
    assert_eq!(index.weight_to_digits(0.0000001, &mut digits), Ok(1));

    // Test overflow (though unlikely for weights <1, but for completeness)
    assert!(index.weight_to_digits(2.0, &mut digits).is_err());
}

#[cfg(test)]
#[test]
fn test_add_many() {
    const CAPACITY: u64 = 10_000;
    let mut index_one_at_a_time = DigitBinIndex::with_precision_and_capacity(3, CAPACITY);
    let mut index_all_at_once = DigitBinIndex::with_precision_and_capacity(3, CAPACITY);
    let mut population = Vec::with_capacity(CAPACITY as usize);
    let mut rng = WyRand::from_os_rng();
    for i in 0..CAPACITY {
        let weight: f64 = rng.random_range(0.001..=0.999);
        population.push((i, weight));
        index_one_at_a_time.add(i, weight).unwrap();
    }
    index_all_at_once.add_many(&population).unwrap();
    assert_eq!(index_one_at_a_time.count(), CAPACITY);
    assert_eq!(index_all_at_once.count(), CAPACITY);
    assert_eq!(
        index_one_at_a_time.total_weight(),
        index_all_at_once.total_weight()
    );

    assert!(index_one_at_a_time.remove_many(&population).unwrap());
    assert!(index_all_at_once.remove_many(&population).unwrap());
    assert_eq!(index_one_at_a_time.count(), 0);
    assert_eq!(index_all_at_once.count(), 0);
}
#[test]
fn test_add() {
    let mut index = DigitBinIndex::new();
    index.add(1, 0.5).unwrap();
    assert_eq!(index.count(), 1);
    assert_eq!(index.total_weight(), 0.5);
}
