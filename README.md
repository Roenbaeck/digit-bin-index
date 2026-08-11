# DigitBinIndex

A `DigitBinIndex` is a tree-based data structure designed for efficient weighted random selection and removal from large collections of items. It is optimized for scenarios involving millions of items where quantized weights or probabilities are acceptable and high performance is critical, including independent event simulation, [Wallenius' noncentral hypergeometric distribution](https://en.wikipedia.org/wiki/Wallenius%27_noncentral_hypergeometric_distribution), [Fisher's noncentral hypergeometric distribution](https://en.wikipedia.org/wiki/Fisher%27s_noncentral_hypergeometric_distribution), and fixed-size PPS sampling.

This library provides high-performance solutions for four complementary
weighted sampling designs:

*   **Independent Event Sampling (Bernoulli)**: Modeled by `select_bernoulli_many` and `select_bernoulli_many_and_remove`, where each weight is an absolute event probability and the sample size is random.
*   **Sequential Sampling (Wallenius')**: Modeled by `select_and_remove`, where items are selected and removed one at a time.
*   **Simultaneous Sampling (Fisher's)**: Modeled by `select_many_and_remove`, where a batch of unique items is selected and removed together.
*   **Marginal Inclusion Sampling (PPS)**: Modeled by `select_pps_many` and `select_pps_many_and_remove`, where weights specify desired first-order inclusion probabilities in a fixed-size sample.

### The Core Problem

In simulations, forecasts, or statistical models (e.g., [mortality models](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4060603/), Monte Carlo simulations, or machine learning sampling), managing a large, dynamic set of probabilities is common. A key task is to randomly select items based on their weights, often removing them afterward, and repeat this process efficiently. For datasets with millions of items, achieving high performance while maintaining reasonable accuracy is a significant challenge, especially for complex distributions like Wallenius' or Fisher's.

### How It Works

`DigitBinIndex` is a radix tree that organizes items into bins based on the decimal digits of their rescaled probabilities, enabling fast weighted random selection and updates.

1.  **Digit-based Tree Structure**: Each level of the tree corresponds to a decimal place of the rescaled weight. For example, a weight of `0.543` at precision 3 is rescaled to `543` and placed by traversing the path: `root -> child[5] -> child[4] -> child[3]`.

2.  **Selectable Bin Storage**: Leaf nodes store IDs in `Vec<u32>`, `RoaringBitmap`, or `RoaringTreemap`. A capacity hint selects a backend from expected average occupancy; explicitly choose a backend for highly skewed weights or when full `u64` IDs are required.

3.  **Accumulated Value Index**: Each node tracks the `accumulated_value` (sum of weights beneath it), supporting O(P) weighted random selection, where P is the configured precision (number of decimal places).

4.  **Exact Binned Bernoulli Sampling**: Each occupied bin draws its selected count from a binomial distribution using the bin's quantized probability, then samples that many IDs uniformly. This is exactly equivalent to an independent Bernoulli trial per item while avoiding a population-wide probability check.

5.  **Adaptive Exact Fisher Sampling**: Batch draws operate on occupied weight bins rather than expanding the population. Small draws use weighted proposals conditioned on all IDs being unique. Larger draws sample exponentially tilted binomial counts for each occupied bin and condition their sum on the requested sample size. Both paths produce Fisher's noncentral hypergeometric law for the quantized weights.

6.  **Fixed-Size PPS Sampling**: PPS draws water-fill certainty items, randomly order the remaining occupied bins, and use exact integer systematic rounding to allocate the requested sample size. Uniform sampling without replacement inside each bin gives every item its requested marginal inclusion probability under the quantized weights.

7.  **Persistent Sequential Sampling**: Each index seeds a fast `WyRand` generator once and reuses it for sequential convenience methods. Rust callers can provide their own generator through `select_with_rng` and `select_and_remove_with_rng`. `select_wallenius_many_and_remove` performs many sequential removals in one call, avoiding repeated Python/Rust boundary crossings while preserving Wallenius' law.

8.  **Lazy Mass-Optimal Child Scans**: Internal nodes keep direct digit lookup for updates and a separate scan order sorted by descending accumulated mass. Mutations invalidate ordering cheaply; only nodes reached by a later selection sort their at-most-ten active children. Selection removals maintain the visited path locally.

### Features

*   **High Performance**: Outperforms general-purpose data structures like Fenwick Trees for both sequential and simultaneous weighted sampling.
*   **Four Explicit Sampling Designs**: Independent Bernoulli events (`select_bernoulli_many`), optimized sequential Wallenius draws (`select_and_remove`), exact Fisher draws over quantized weights (`select_many_and_remove`), and fixed-size PPS draws with exact first-order inclusion probabilities (`select_pps_many`).
*   **Bounded Tree Traversal**: Radix traversal is O(P), where P is the fixed precision. Explicit `Vec`-leaf removal is linear in that leaf's occupancy, and exact Fisher batches also depend on occupied bins.
*   **Memory Efficiency**: Combines a sparse radix tree with Roaring Bitmaps for efficient storage, especially for sparse or clustered weight distributions.
*   **Python Integration**: Seamless Python bindings via `pyo3` for cross-language support.

---

### Choosing the Selection Model

Choose a method from how the events are generated, not from what happens to the
selected IDs afterward. In particular, a weight is not automatically an
absolute event probability.

| Question being modeled | Method | Simple example |
| :-- | :-- | :-- |
| Who wins one weighted draw? | `select` or `select_and_remove` | Choose one job from a queue, with urgency as its relative weight. Remove it if the job must not be chosen again. |
| Who wins each of `k` independent draws when the same item may appear repeatedly? | Call `select()` `k` times without changing the index | Allocate `k` ad impressions independently; the resulting per-item counts follow a multinomial law. |
| Who fills `k` places awarded one at a time? | `select_wallenius_many_and_remove(k)` | Award `k` appointments sequentially. At each step, every remaining person's chance is proportional to their priority weight. |
| Which `k` independent binary events occurred, given that their total is `k`? | `select_many(k)` or `select_many_and_remove(k)` (Fisher) | Components fail independently with different probabilities; after learning that exactly `k` failed, sample their identities. Use failure **odds**, `p / (1 - p)`, as relative weights. |
| Who belongs to a fixed cohort when marginal inclusion chances must be proportional to size? | `select_pps_many(k)` or `select_pps_many_and_remove(k)` | Audit exactly `k` invoices, giving an invoice with twice the value twice the marginal chance of inclusion. |
| Does each item independently experience an event during this period? | `select_bernoulli_many()` or `select_bernoulli_many_and_remove()` | Simulate whether each person dies or moves using that person's period probability. The number of events remains random. |

The versions without `and_remove` leave the index unchanged. The versions with
`and_remove` use the same sampling law and then remove the selected IDs.
Repeated calls to `select()` already provide weighted sampling with replacement;
a native batch wrapper would improve call overhead but would not introduce a
new distribution. Equal weights reduce the fixed-size methods to their ordinary
unweighted counterparts. In survey-sampling terminology, independent Bernoulli
inclusion is also called [Poisson sampling](https://www150.statcan.gc.ca/n1/pub/12-001-x/2016001/article/14543/04-eng.htm),
where independent selections produce a random sample size.

#### Mortality Over Time

Suppose person `i` has probability `p_i(t)` of dying during period `t`. When
the model treats people as conditionally independent given those probabilities,
Bernoulli selection performs one trial per living person using the index's
quantized probabilities:

```python
deaths = mortality_index.select_bernoulli_many_and_remove()
```

The index accepts probabilities strictly between zero and one. Leave `p=0`
items out of the selectable index and handle `p=1` events deterministically.

The number of deaths is random, with expected value equal to the sum of the
quantized probabilities. A fixed-size method should not replace this draw: PPS
would force a chosen total and rescale the marginal probabilities, while Fisher
and Wallenius would impose different dependence between people.

There are two related mortality models that do use this library:

* If an external model fixes the period total at exactly `k` deaths, and the
  desired model is independent deaths conditioned on that total, use Fisher's
  `select_many_and_remove(k)`. Convert each probability to odds
  `p_i / (1 - p_i)`, then multiply all odds by the same positive constant so
  the largest index weight is below one. Common scaling does not change the
  Fisher law.
* In a continuous-time, event-driven model with instantaneous mortality hazards
  `h_i(t)`, `select_and_remove` can choose the identity of the next death using
  hazards as relative weights. Scale all hazards by the same positive constant
  if needed to fit the index's weight range. Calendar time is a separate draw:
  while hazards are constant, the waiting time is exponential with rate
  `sum(h_i)`. Update hazards whenever the simulated state or time changes.

Fisher's distribution is the conditional distribution of independent binomial
variates given their sum; see [Fog, “Noncentral hypergeometric
distributions”](https://www.agner.org/random/distrib.pdf).

#### Migration Over Time

Use the same rule for migration because the subject matter does not determine
the sampling law:

* If `p_i(t)` is person `i`'s probability of moving during the period, use
  `select_bernoulli_many()` or `select_bernoulli_many_and_remove()`. The number
  of movers is random.
* If exactly `k` independent move decisions are known to have occurred, use
  Fisher with odds `p_i / (1 - p_i)`.
* If `k` scarce relocation places are filled sequentially from a changing pool,
  use Wallenius with relative priority or propensity weights.
* If exactly `k` people must be included and the requirement is that marginal
  inclusion chances be proportional to a risk, exposure, or policy score, use
  PPS.

For both mortality and migration, removal only describes the state update. A
death is removed from the population; a mover is normally removed from the
origin index and added to a destination index. It does not by itself determine
which sampling distribution is appropriate.

---

### Performance

`DigitBinIndex` trades a small, controllable amount of precision by binning
weights or probabilities to achieve significant performance gains. The
Criterion suite compares Bernoulli selection with an item-level probability
scan and compares the fixed-size designs with general-purpose weighted data
structures in high-churn simulations.

The churn benchmarks start with a large population (1M or 10M items), then
simulate a high volume of activity:

*   **Churn**: A significant number of items are selected and removed.
*   **Acquisition**: New items are added to the population.

The historical churn tables were measured on a desktop with an Intel i7, 16 GB
RAM, and Rust 1.75. Newer local-development measurements are labeled as such
and should be rerun on the documented release hardware before publication.

---

#### Independent Bernoulli Draw

For a bin containing `n` items with quantized probability `p`, Bernoulli
selection draws `K ~ Binomial(n, p)` and then chooses a uniform `K`-subset of
the bin. Every particular subset therefore has the same probability it would
have under independent item-level Bernoulli trials. Counts are drawn
independently across bins.

`select_bernoulli_many()` leaves the index unchanged;
`select_bernoulli_many_and_remove()` removes the events. Both return a
random-size sample, and an empty vector is a normal successful result. For
quantized probabilities `p_i`:

```text
expected_sample_size = sum(p_i)
sample_size_variance = sum(p_i * (1 - p_i))
```

The one-million-item Criterion workload uses probabilities from the smallest
positive bin through 5% and includes removal of selected IDs:

| Precision | Binned Bernoulli | Item-level scan | Result |
| :-- | --: | --: | :-- |
| **3** | **0.56 ms** | 1.22 ms | **2.2x faster** |
| **5** | 1.88 ms | **1.16 ms** | 1.6x slower |

At p=3, only 50 probability bins are occupied and binomial aggregation avoids
almost all item-level trials. At p=5, roughly 5,000 bins are occupied, so bin
traversal and distribution setup cost more than a tight item scan. These are
local development measurements and show why p=3 or p=4 should remain the
default unless finer probability resolution is material.

---

#### Wallenius' Draw (Sequential Churn)

This benchmark simulates sequential selection by removing 100,000 items one-by-one, then adding 110,000 new items. This is a common pattern in agent-based models or iterative simulations. `DigitBinIndex`'s O(P) complexity gives it a decisive advantage as the population scales.

| Scenario (N items)         | `DigitBinIndex` Time | `FenwickTree` Time | **Speedup Factor** |
| :------------------------- | :------------------- | :----------------- | :----------------- |
| **1 Million Items** (p=3)  | **~27.5 ms**         | ~82.2 ms           | **~3.0x faster**   |
| **1 Million Items** (p=5)  | **~39.7 ms**         | ~77.7 ms           | **~2.0x faster**   |
| **10 Million Items** (p=3) | **~551.0 ms**        | ~1723.1 ms         | **~3.1x faster**   |

*   **Key Takeaway**: `DigitBinIndex` is **over 3.0 times faster** than the `FenwickTree` for sequential operations on large datasets. Its performance is dependent on precision (`P`) and not the number of items (`N`), allowing it to scale far more effectively.

These are historical v0.4.2 measurements. The current implementation reuses RNG state, propagates integer weights through the hot path, removes runtime selection/removal branching, and scans children in descending accumulated-mass order. Updated numbers should replace this table after rerunning the Criterion suite on the documented hardware. The suite now includes a heavily skewed workload whose dominant mass occupies a low-digit branch.

---

#### Fisher's Draw (Batch Churn)

Batch sampling now implements the exact Fisher law for the quantized weights. It uses two adaptive engines:

*   **Small samples**: weighted item proposals conditioned on uniqueness. Expected work is close to O(kP) while collision probability is low.
*   **Larger samples**: conditional binomial sampling over the B occupied weight bins, followed by uniform sampling within each selected bin. High-bin-count workloads use partial convolution blocks so each exact-sum proposal samples block totals rather than every individual bin. Runtime depends on B rather than population size N. Draws over half the population sample the smaller complement with reciprocal weights.

The previous Fisher timing table measured the earlier capped-multinomial approximation and is not comparable to the exact implementation. The Criterion suite retains 1M and 10M batch scenarios at precision 3 and 5; updated numbers should be published after benchmarking the exact sampler on the documented hardware. Precision 3 has at most 999 positive-weight bins and is expected to remain the strongest large-batch configuration. High precision with many occupied bins is intentionally covered as a stress case.

---

#### Fixed-Size PPS Draw

PPS sampling answers a different question from Fisher or Wallenius. In Fisher
and Wallenius draws, weights are **odds** that determine a joint distribution.
In PPS sampling, weights are **size measures** that determine each item's
marginal probability of appearing in a fixed-size sample. This is useful when
selecting an audit, inspection, or evaluation cohort where an item with twice
the exposure should have twice the inclusion probability.

For a requested sample size `k`, the PPS methods target

```text
inclusion_probability_i = min(1, lambda * quantized_weight_i)
```

where `lambda` is chosen so that the inclusion probabilities sum to `k`.
Items whose probability reaches one are certainty items; their slots and mass
are removed before `lambda` is recomputed for the remaining population. This is
the standard feasibility treatment required when a strictly proportional
probability would exceed one.

`select_pps_many(k)` returns `k` unique IDs without changing the index.
`select_pps_many_and_remove(k)` uses the same design and removes them. The
implementation randomly orders occupied equal-weight bins, applies systematic
rounding with an exact integer random start, and samples uniformly without
replacement inside each bin. Consequently:

* the sample size is always exactly `k`;
* every returned ID is unique;
* first-order inclusion probabilities are exact for the index's quantized weights;
* the joint law is the documented randomized systematic PPS design, not Fisher's or Wallenius' law.

In the Criterion suite's 1-million-item churn scenario (select and remove
100,000, then add 110,000), the binned PPS implementation avoids expanding and
shuffling the full population:

| Precision | `DigitBinIndex` PPS | Item-level systematic PPS baseline | Speedup |
| :-- | --: | --: | --: |
| **3** | **16.9 ms** | 42.9 ms | **2.5x** |
| **5** | **25.1 ms** | 43.5 ms | **1.7x** |

These are local development measurements and should be treated as
hardware-dependent; the benchmark source is included for reproducibility.

Systematic unequal-probability sampling with a random start is a longstanding
fixed-size PPS design; see [Hartley (1966), “Systematic Sampling with Unequal
Probability and without Replacement”](https://doi.org/10.1080/01621459.1966.10480902).
Applications that require design-based variance estimates must also account for
the design's second-order inclusion probabilities rather than treating selected
items as independent.

---

### When to Choose DigitBinIndex

Use `DigitBinIndex` when:

*   You need to simulate many independent binary events from quantized per-item probabilities without scanning every item.
*   You need high-performance sampling for Wallenius' or Fisher's distributions.
*   You need a fixed-size PPS cohort whose marginal inclusion probabilities are proportional to an exposure, value, or risk measure.
*   Your dataset is large (N > 100,000).
*   Probabilities are approximate, as is common in empirical data, simulations, or machine learning models.
*   Performance is more critical than perfect precision.

Consider a Fenwick Tree if you require exact precision and your weights differ only at high decimal places (e.g., 0.12345 vs. 0.12346), though this comes at the cost of O(log N) complexity and higher memory usage for large datasets.

---

### Choosing a Precision

The `precision` parameter controls the radix tree's depth, balancing **accuracy**, **performance**, and **memory**. Higher precision improves sampling accuracy but increases memory usage (up to 10x per additional level) and slightly impacts runtime.

#### The Rule of Thumb

**A precision of 3 or 4 (default: 3) is recommended for most applications.** This captures sufficient detail for typical weight distributions while maintaining excellent performance and low memory usage.

#### The Mathematical Intuition

Each decimal place contributes exponentially less to a weight’s value. For a weight of `0.12345`:

*   1st digit (`1`): `0.1`
*   2nd digit (`2`): `0.02`
*   3rd digit (`3`): `0.003`
*   4th digit (`4`): `0.0004`

Rounding to 3 digits limits the error per item to at most 0.0005, except at the positive boundaries where values are clamped to the smallest or largest representable bin.

#### Guidance

| Precision | Typical Use Case                                     | Trade-offs                                                   |
| :-------- | :--------------------------------------------------- | :----------------------------------------------------------- |
| **1-2**   | Maximum performance, minimal memory usage.           | Best for coarse weights (e.g., `0.1`, `0.5`). Loses accuracy with fine-grained data. |
| **3-4**   | **Recommended Default.** Optimal for most scenarios. | Captures sufficient detail for simulation or model data. Negligible performance/memory cost. |
| **5+**    | High-fidelity scenarios with very close weights.     | Distinguishes weights like `0.12345` vs. `0.12346`. Increases memory (up to 10x per level) and slightly impacts performance. |

---

### Mutation Contract and Capacity

Weights must be finite and strictly between zero and one. Rust mutation methods
return `Result`; Python raises `ValueError` for invalid weights and duplicates,
and `OverflowError` for an ID that does not fit the selected backend. Batch
addition validates the complete input before changing the tree. Batch removal
is ordered and partially mutating when a valid `(ID, weight)` pair is missing.

Counts and scaled-weight sums use `u64` and are checked before growth. At
precision 9, repeatedly adding weights near one reaches the scaled-mass limit at
roughly 18.4 billion items; the insertion that would exceed the limit returns an
error rather than wrapping.

---

## Internal Storage and Capacity

The `DigitBinIndex` is designed to handle a vast range of use cases, from a few thousand items to trillions, by automatically selecting the most appropriate internal storage engine.

### Item Capacity

The index accepts **`u64`** item IDs. `Small` and `Medium` return an error for IDs above `u32::MAX`; `Large` supports the complete `u64` range. IDs are unique across the complete index, including across different weight bins.

To provide the best balance of performance and memory usage, the library's `DigitBinIndex` is an enum that automatically switches between three different backends (`Small`, `Medium`, and `Large`) when you use the `with_precision_and_capacity()` constructor or the explicit constructors `small()`, `medium()`, and `large()`.

The selection is based on a simple heuristic: the **average number of items expected per bin**, which is calculated as `capacity / 10^precision`.

The capacity hint cannot infer ID width or weight skew. Use `large` whenever an
ID may exceed `u32::MAX`. If many items share a small number of weights, compare
`small` and `medium` with representative data; the Criterion suite includes a
single-bin backend benchmark for this case.

1.  `Small` (**`Vec<u32>`**):
    *   **Constructor:** `small(precision: u8)`.
    *   **Backend Datatype:** `u32` (max 4 billion).
    *   **Capacity Trigger:** Low average items per bin (<= 1,000).
    *   **Best for:** Small to medium-sized problems where `select_and_remove` speed is the absolute priority (O(1) `swap_remove`).
    *   ***Warning:*** Rejects IDs above `u32::MAX`. Explicit removal is linear in the selected leaf's occupancy.

2.  `Medium` (**`RoaringBitmap`**):
    *   **Constructor:** `medium(precision: u8)`.
    *   **Backend Datatype:** `u32` (max 4 billion).
    *   **Capacity Trigger:** Medium to large average items per bin (> 1,000).
    *   **Best for:** Large-scale problems where IDs fit within `u32`, especially heavily occupied or clustered bins.
    *   ***Warning:*** Rejects IDs above `u32::MAX`.

3.  `Large` (**`RoaringTreemap`**):
    *   **Constructor:** `large(precision: u8)`.
    *   **Backend Datatype:** `u64` (max 18 quintillion = 18 billion billions).
    *   **Capacity Trigger:** Extremely large average items per bin (> 1,000,000,000). This is used as a heuristic to detect that full `u64` support is required.
    *   **Best for:** Massive-scale simulations or any dataset that requires the full 64-bit ID space.

### Examples of Engine Selection

Here are some practical examples of how calling `with_precision_and_capacity` translates into a specific internal engine.

#### Example 1: `Small (Vec<u32>)` is Chosen

You are simulating a population of 100,000 individuals with `u32` IDs.

```rust
// Expecting 100,000 items with 3-digit precision
let index = DigitBinIndex::with_precision_and_capacity(3, 100_000);
```

*   **Calculation:** The number of bins is `10^3 = 1,000`. The average items per bin is `100,000 / 1,000 = 100`.
*   **Result:** Since 100 <= 1,000, the `Small` variant is chosen. IDs above `u32::MAX` will be rejected rather than truncated.

#### Example 2: `Medium (RoaringBitmap)` is Chosen

You need to index 50 million product IDs, all of which fit within `u32`.

```rust
// Expecting 50 million items with 3-digit precision
let index = DigitBinIndex::with_precision_and_capacity(3, 50_000_000);
```

*   **Calculation:** The average items per bin is `50,000,000 / 1,000 = 50,000`.
*   **Result:** This is > 1,000. The `Medium` variant is selected, using `RoaringBitmap`; IDs above `u32::MAX` will be rejected.

#### Example 3: `Large (RoaringTreemap)` is Chosen

You are working with a massive dataset where item IDs are 64-bit, and you expect trillions of entries.

```rust
// Expecting 5 trillion items with 3-digit precision
let index = DigitBinIndex::with_precision_and_capacity(3, 5_000_000_000_000);
```

*   **Calculation:** The average items per bin is `5_000_000_000_000 / 1,000 = 5,000,000,000`.
*   **Result:** This is > 1,000,000,000. The `Large` variant is chosen. The heuristic correctly identifies this as a `u64`-scale problem and selects the only backend, `RoaringTreemap`, that provides full 64-bit ID support.

---

### Usage & Installation

`DigitBinIndex` is available as a Python library on [PyPI](https://pypi.org/project/digit-bin-index/) or as a Rust crate on [Crates.io](https://crates.io/crates/digit-bin-index). The Python bindings require Python 3.8+.

#### For Python 🐍

Install from PyPI:

```bash
pip install digit-bin-index
```

Example usage:

```python
from digit_bin_index import DigitBinIndex

def main():
    # Create an index with precision 3 (default).
    index = DigitBinIndex()

    # With custom precision
    index_5 = DigitBinIndex.with_precision(5)

    # With custom precision and capacity hint for large datasets
    # This might choose a more memory-efficient internal storage.
    index_3_xl = DigitBinIndex.with_precision_and_capacity(3, 10_000_000)

    # Add items with IDs and weights.
    index.add(id=101, weight=0.123)  # Low weight
    index.add(id=202, weight=0.800)  # High weight
    index.add(id=303, weight=0.755)  # High weight
    index.add(id=404, weight=0.110)  # Low weight

    # Independent Bernoulli draw: each stored weight is the item's absolute
    # event probability. The result size is random; the index is unchanged.
    period_events = index.select_bernoulli_many()
    print(f"Bernoulli events: {period_events}")

    # Sequential (Wallenius') Draw: Select and remove one item.
    # Higher-weighted items (202, 303) are more likely.
    selected_item = index.select_and_remove()
    if selected_item:
        # The returned weight is a float, representing the bin's average weight
        item_id, weight = selected_item
        print(f"Wallenius draw: ID {item_id}, Weight ~{weight:.3f}")
    
    print(f"Items remaining: {index.count()}")  # 3

    # For many sequential Wallenius draws, use one native batch call:
    # churned_items = index.select_wallenius_many_and_remove(100_000)

    # Fixed-size PPS draw: weights control marginal inclusion probabilities.
    # This returns unique IDs without changing the index.
    pps_items = index.select_pps_many(2)
    if pps_items:
        print(f"PPS draw: {pps_items}")

    # Simultaneous (Fisher's) Draw: Select and remove 2 unique items.
    selected_items = index.select_many_and_remove(2)
    if selected_items:
        print(f"Fisher's draw: {selected_items}")
    
    print(f"Items remaining: {index.count()}")  # 1

if __name__ == "__main__":
    main()
```

#### For Rust 🦀

Add to your `Cargo.toml`:

```toml
[dependencies]
digit-bin-index = "0.4.2" # Replace with the latest version from crates.io
```

Example usage:

```rust
use digit_bin_index::DigitBinIndex;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create an index with precision 3.
    let mut index = DigitBinIndex::with_precision(3);

    // Add items with IDs and f64 weights.
    index.add(101, 0.123)?; // Low weight
    index.add(202, 0.800)?; // High weight
    index.add(303, 0.755)?; // High weight
    index.add(404, 0.110)?; // Low weight

    // Independent Bernoulli draw: each weight is an absolute event
    // probability. The result size is random; the index is unchanged.
    let period_events = index.select_bernoulli_many().unwrap();
    println!("Bernoulli events: {:?}", period_events);

    // Sequential (Wallenius') Draw: Select and remove one item.
    if let Some((id, weight)) = index.select_and_remove() {
        println!("Wallenius draw: ID {}, Weight ~{}", id, weight);
    }
    println!("Items remaining: {}", index.count()); // 3

    // For repeated sequential draws, reuse the index RNG in one call:
    // let churned_items = index.select_wallenius_many_and_remove(100_000);

    // Fixed-size PPS draw: weights control marginal inclusion probabilities.
    // The index is not changed.
    if let Some(items) = index.select_pps_many(2) {
        println!("PPS draw: {:?}", items);
    }

    // Simultaneous (Fisher's) Draw: Select and remove 2 unique items.
    if let Some(items) = index.select_many_and_remove(2) {
        println!("Fisher's draw: {:?}", items);
    }
    println!("Items remaining: {}", index.count()); // 1
    Ok(())
}
```

### License

This project is licensed under the [MIT License](LICENSE), a permissive open-source license allowing free use, modification, and distribution.
