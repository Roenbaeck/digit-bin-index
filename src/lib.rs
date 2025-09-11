use rand::{rngs::ThreadRng, Rng};
use roaring::RoaringBitmap;
use std::vec;

// Constants
const DEFAULT_PRECISION: u8 = 3;
const MAX_PRECISION: usize = 8; // Reduced to ensure u64 supports 1B individuals
const MAX_LEVELS: usize = 7; // Max radix-16 levels for precision=8 (~ceil(8 * log(10)/log(16)) ≈ 7)

#[derive(Debug, Clone)]
pub enum NodeContent {
    Internal(Vec<Node>),
    Leaf(RoaringBitmap),
}

#[derive(Debug, Clone)]
pub struct Node {
    pub content: NodeContent,
    pub accumulated_value: u64, // Changed to u64
    pub content_count: u32,
}

impl Node {
    fn new_internal() -> Self {
        Self {
            content: NodeContent::Internal(vec![]),
            accumulated_value: 0,
            content_count: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DigitBinIndex {
    pub root: Node,
    pub precision: u8, // Decimal precision for input weights
    pub levels: u8,   // Radix-16 tree depth
    scale: f64
}

impl Default for DigitBinIndex {
    fn default() -> Self {
        Self::with_precision(DEFAULT_PRECISION)
    }
}

impl DigitBinIndex {
    pub fn new() -> Self {
        Self::with_precision(DEFAULT_PRECISION)
    }

    pub fn with_precision(precision: u8) -> Self {
        assert!(precision > 0, "Precision must be at least 1.");
        assert!(precision <= MAX_PRECISION as u8, "Precision cannot be larger than {}.", MAX_PRECISION);
        // Compute radix-16 levels to match decimal precision
        let levels = ((precision as f32 * 3.321928) / 4.0).ceil() as u8; // log2(10)/log2(16) ≈ 0.830482
        let scale = 10f64.powi(precision as i32);
        Self {
            root: Node::new_internal(),
            precision,
            levels,
            scale
        }
    }

    fn weight_to_mantissa_and_digits(&self, weight: f64) -> Option<(u32, [u8; MAX_LEVELS])> {
        if weight <= 0.0 {
            return None;
        }
        let mantissa = (weight * self.scale) as u32;
        if mantissa == 0 {
            return None;
        }
        let mut digits = [0u8; MAX_LEVELS];
        let mut temp = mantissa;
        for i in 0..self.levels as usize {
            digits[i] = (temp & 0xF) as u8;
            temp >>= 4;
        }
        Some((mantissa, digits))
    }

    pub fn add(&mut self, individual_id: u32, weight: f64) -> bool {
        if let Some((mantissa, digits)) = self.weight_to_mantissa_and_digits(weight) {
            Self::add_recurse(&mut self.root, individual_id, mantissa, &digits, 0, self.levels);
            true
        } else {
            false
        }
    }

    fn add_recurse(
        node: &mut Node,
        individual_id: u32,
        mantissa: u32,
        digits: &[u8; MAX_LEVELS],
        current_level: u8,
        max_levels: u8,
    ) {
        node.content_count += 1;
        node.accumulated_value += mantissa as u64;

        if current_level >= max_levels {
            if let NodeContent::Internal(_) = &node.content {
                node.content = NodeContent::Leaf(RoaringBitmap::new());
            }
            if let NodeContent::Leaf(bitmap) = &mut node.content {
                bitmap.insert(individual_id);
            }
            return;
        }

        let digit = digits[current_level as usize] as usize;
        if let NodeContent::Internal(children) = &mut node.content {
            if children.len() <= digit {
                children.resize_with(digit + 1, Node::new_internal);
            }
            Self::add_recurse(&mut children[digit], individual_id, mantissa, digits, current_level + 1, max_levels);
        }
    }

    pub fn remove(&mut self, individual_id: u32, weight: f64) {
        if let Some((mantissa, digits)) = self.weight_to_mantissa_and_digits(weight) {
            self.remove_with_digits(individual_id, mantissa, digits);
        }
    }

    fn remove_with_digits(&mut self, individual_id: u32, mantissa: u32, digits: [u8; MAX_LEVELS]) {
        Self::remove_recurse(&mut self.root, individual_id, mantissa, &digits, 0, self.levels);
    }

    fn remove_recurse(
        node: &mut Node,
        individual_id: u32,
        mantissa: u32,
        digits: &[u8; MAX_LEVELS],
        current_level: u8,
        max_levels: u8,
    ) -> bool {
        if current_level >= max_levels {
            if let NodeContent::Leaf(bitmap) = &mut node.content {
                if bitmap.remove(individual_id) {
                    node.content_count -= 1;
                    node.accumulated_value -= mantissa as u64;
                    return true;
                }
            }
            return false;
        }

        let digit = digits[current_level as usize] as usize;
        if let NodeContent::Internal(children) = &mut node.content {
            if children.len() > digit && Self::remove_recurse(&mut children[digit], individual_id, mantissa, digits, current_level + 1, max_levels) {
                node.content_count -= 1;
                node.accumulated_value -= mantissa as u64;
                return true;
            }
        }
        false
    }

    pub fn select(&mut self) -> Option<(u32, f64)> {
        self.select_and_optionally_remove(false)
    }

    pub fn select_many(&mut self, num_to_draw: u32) -> Option<Vec<(u32, f64)>> {
        self.select_many_and_optionally_remove(num_to_draw, false)
    }

    pub fn select_and_remove(&mut self) -> Option<(u32, f64)> {
        self.select_and_optionally_remove(true)
    }

    fn select_and_optionally_remove(&mut self, with_removal: bool) -> Option<(u32, f64)> {
        if self.root.content_count == 0 {
            return None;
        }
        let mut rng = rand::thread_rng();
        let random_target = rng.gen_range(0..self.root.accumulated_value); // Use u64 directly
        if let Some((id, mantissa)) = Self::select_and_optionally_remove_recurse(&mut self.root, random_target, 0, self.levels, &mut rng, with_removal) {
            Some((id, mantissa as f64 / self.scale))
        } else {
            None
        }
    }

    fn select_and_optionally_remove_recurse(
        node: &mut Node,
        mut target: u64,
        current_level: u8,
        max_levels: u8,
        rng: &mut ThreadRng,
        with_removal: bool,
    ) -> Option<(u32, u32)> {
        if current_level >= max_levels {
            if let NodeContent::Leaf(bitmap) = &mut node.content {
                if bitmap.is_empty() {
                    return None;
                }
                let rand_index = rng.gen_range(0..node.content_count);
                if let Some(selected_id) = bitmap.select(rand_index) {
                    let mantissa = (node.accumulated_value / node.content_count as u64) as u32;
                    if with_removal {
                        bitmap.remove(selected_id);
                        node.content_count -= 1;
                        node.accumulated_value -= mantissa as u64;
                    }
                    return Some((selected_id, mantissa));
                }
            }
            return None;
        }

        if let NodeContent::Internal(children) = &mut node.content {
            for child in children.iter_mut() {
                if child.accumulated_value == 0 {
                    continue;
                }
                if target < child.accumulated_value {
                    if let Some((selected_id, mantissa)) = Self::select_and_optionally_remove_recurse(
                        child,
                        target,
                        current_level + 1,
                        max_levels,
                        rng,
                        with_removal,
                    ) {
                        if with_removal {
                            node.content_count -= 1;
                            node.accumulated_value -= mantissa as u64;
                        }
                        return Some((selected_id, mantissa));
                    }
                    return None;
                }
                target -= child.accumulated_value;
            }
        }
        None
    }

    pub fn select_many_and_remove(&mut self, num_to_draw: u32) -> Option<Vec<(u32, f64)>> {
        self.select_many_and_optionally_remove(num_to_draw, true)
    }

    fn select_many_and_optionally_remove(&mut self, num_to_draw: u32, with_removal: bool) -> Option<Vec<(u32, f64)>> {
        if num_to_draw > self.count() || num_to_draw == 0 {
            return if num_to_draw == 0 { Some(Vec::new()) } else { None };
        }
        let mut rng = rand::thread_rng();
        let mut selected = Vec::with_capacity(num_to_draw as usize);
        let total_weight = self.root.accumulated_value;
        Self::select_many_and_optionally_remove_recurse(
            &mut self.root,
            num_to_draw,
            total_weight,
            &mut selected,
            &mut rng,
            0,
            self.levels,
            with_removal,
        );
        if selected.len() == num_to_draw as usize {
            Some(selected.into_iter().map(|(id, mantissa)| (id, mantissa as f64 / self.scale)).collect())
        } else {
            None
        }
    }

    fn select_many_and_optionally_remove_recurse(
        node: &mut Node,
        m: u32,
        subtree_total: u64,
        selected: &mut Vec<(u32, u32)>,
        rng: &mut ThreadRng,
        current_level: u8,
        max_levels: u8,
        with_removal: bool,
    ) {
        if m == 0 {
            return;
        }
        if current_level >= max_levels {
            if let NodeContent::Leaf(bitmap) = &mut node.content {
                let mantissa = if node.content_count > 0 {
                    (node.accumulated_value / node.content_count as u64) as u32
                } else {
                    0
                };
                let mut picked = 0;
                while picked < m && !bitmap.is_empty() {
                    let rand_index = rng.gen_range(0..bitmap.len() as u32);
                    if let Some(id) = bitmap.select(rand_index) {
                        if with_removal {
                            bitmap.remove(id);
                        }
                        selected.push((id, mantissa));
                        picked += 1;
                    }
                }
                if with_removal {
                    node.content_count -= picked;
                    node.accumulated_value -= mantissa as u64 * picked as u64;
                }
            }
            return;
        }

        if let NodeContent::Internal(children) = &mut node.content {
            let mut child_assigned = vec![0u32; children.len()];
            let mut child_rel_targets: Vec<Vec<u64>> = vec![Vec::new(); children.len()];

            let mut assigned = 0u32;
            while assigned < m {
                let target = rng.gen_range(0..subtree_total);
                let mut cum = 0u64;
                let mut chosen_child = None;
                for (i, child) in children.iter().enumerate() {
                    if target < cum + child.accumulated_value {
                        if child_assigned[i] + 1 <= child.content_count {
                            chosen_child = Some(i);
                        }
                        break;
                    }
                    cum += child.accumulated_value;
                }
                if let Some(idx) = chosen_child {
                    child_assigned[idx] += 1;
                    let rel_target = target - cum;
                    child_rel_targets[idx].push(rel_target);
                    assigned += 1;
                }
            }

            for (i, child) in children.iter_mut().enumerate() {
                let child_m = child_assigned[i];
                if child_m > 0 {
                    Self::select_many_and_optionally_remove_recurse(
                        child,
                        child_m,
                        child.accumulated_value,
                        selected,
                        rng,
                        current_level + 1,
                        max_levels,
                        with_removal,
                    );
                }
            }

            if with_removal {
                node.content_count = children.iter().map(|c| c.content_count).sum();
                node.accumulated_value = children.iter().map(|c| c.accumulated_value).sum();
            }
        }
    }

    pub fn count(&self) -> u32 {
        self.root.content_count
    }

    pub fn total_weight(&self) -> f64 {
        self.root.accumulated_value as f64 / self.scale
    }

    pub fn count_nodes(&self) -> (usize, usize) {
        fn count_recurse(node: &Node, level: u8, max_levels: u8) -> (usize, usize) {
            if level >= max_levels {
                return (0, 1);
            }
            match &node.content {
                NodeContent::Internal(children) => {
                    let mut internal = 1;
                    let mut leaves = 0;
                    for child in children {
                        let (i, l) = count_recurse(child, level + 1, max_levels);
                        internal += i;
                        leaves += l;
                    }
                    (internal, leaves)
                }
                NodeContent::Leaf(_) => (0, 1),
            }
        }
        count_recurse(&self.root, 0, self.levels)
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
        fn new(precision: u32) -> PyResult<Self> {
            if precision > MAX_PRECISION as u32 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Precision cannot be larger than {}.", MAX_PRECISION),
                ));
            }
            Ok(PyDigitBinIndex {
                index: DigitBinIndex::with_precision(precision as u8),
            })
        }

        fn add(&mut self, id: u32, weight: f64) -> bool {
            self.index.add(id, weight)
        }

        fn remove(&mut self, id: u32, weight: f64) {
            self.index.remove(id, weight);
        }

        fn select(&self) -> Option<(u32, f64)> {
            self.index.select()
        }

        fn select_many(&self, n: u32) -> Option<Vec<(u32, f64)>> {
            self.index.select_many(n)
        }

        fn select_and_remove(&mut self) -> Option<(u32, f64)> {
            self.index.select_and_remove()
        }

        fn select_many_and_remove(&mut self, n: u32) -> Option<Vec<(u32, f64)>> {
            self.index.select_many_and_remove(n)
        }

        fn count(&self) -> u32 {
            self.index.count()
        }

        fn total_weight(&self) -> f64 {
            self.index.total_weight()
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
        if let Some((id, weight)) = index.select_and_remove() {
            println!("Selected ID: {} with weight: {}", id, weight);
        }
        assert_eq!(index.count(), 3);
        if let Some(selection) = index.select_many_and_remove(2) {
            println!("Selection: {:?}", selection);
        }
        assert_eq!(index.count(), 1);
    }

    #[test]
    fn test_wallenius_distribution_is_correct() {
        const ITEMS_PER_GROUP: u32 = 1000;
        const TOTAL_ITEMS: u32 = ITEMS_PER_GROUP * 2;
        const NUM_DRAWS: u32 = TOTAL_ITEMS / 2;

        let low_risk_weight = 0.1;
        let high_risk_weight = 0.2;

        const NUM_SIMULATIONS: u32 = 100;
        let mut total_high_risk_selected = 0;

        for _ in 0..NUM_SIMULATIONS {
            let mut index = DigitBinIndex::with_precision(3);
            for i in 0..ITEMS_PER_GROUP {
                index.add(i, low_risk_weight);
            }
            for i in ITEMS_PER_GROUP..TOTAL_ITEMS {
                index.add(i, high_risk_weight);
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

        let avg_high_risk = total_high_risk_selected as f64 / NUM_SIMULATIONS as f64;
        let uniform_mean = NUM_DRAWS as f64 * 0.5;
        let fishers_mean = NUM_DRAWS as f64 * (2.0 / 3.0);

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
            "Test passed: The Wallenius mean is {:.2}",
            avg_high_risk
        )
    }

    #[test]
    fn test_fisher_distribution_is_correct() {
        const ITEMS_PER_GROUP: u32 = 1000;
        const TOTAL_ITEMS: u32 = ITEMS_PER_GROUP * 2;
        const NUM_DRAWS: u32 = TOTAL_ITEMS / 2;

        let low_risk_weight = 0.1;
        let high_risk_weight = 0.2;

        const NUM_SIMULATIONS: u32 = 100;
        let mut total_high_risk_selected = 0;

        for _ in 0..NUM_SIMULATIONS {
            let mut index = DigitBinIndex::with_precision(3);
            for i in 0..ITEMS_PER_GROUP {
                index.add(i, low_risk_weight);
            }
            for i in ITEMS_PER_GROUP..TOTAL_ITEMS {
                index.add(i, high_risk_weight);
            }

            if let Some(selected_ids) = index.select_many_and_remove(NUM_DRAWS) {
                let high_risk_in_this_run = selected_ids.iter().filter(|&&(id, _)| id >= ITEMS_PER_GROUP).count();
                total_high_risk_selected += high_risk_in_this_run as u32;
            }
        }

        let avg_high_risk = total_high_risk_selected as f64 / NUM_SIMULATIONS as f64;
        let fishers_mean = NUM_DRAWS as f64 * (2.0 / 3.0);
        let tolerance = fishers_mean * 0.02;

        assert!(
            (avg_high_risk - fishers_mean).abs() < tolerance,
            "Fisher's test failed: Result {:.2} was not close to the expected mean of {:.2}",
            avg_high_risk, fishers_mean
        );

        println!(
            "Test passed: The Fischer mean is {:.2}",
            avg_high_risk
        )
    }
}