# DigitBinIndex

A `DigitBinIndex` is a tree-based data structure designed for efficient weighted random selection and removal from large collections of items. It is optimized for scenarios involving millions of items where probabilities are approximate and high performance is critical, such as simulations for [Wallenius' noncentral hypergeometric distribution](https://en.wikipedia.org/wiki/Wallenius%27_noncentral_hypergeometric_distribution) or [Fisher's noncentral hypergeometric distribution](https://en.wikipedia.org/wiki/Fisher%27s_noncentral_hypergeometric_distribution).

This library provides high-performance solutions for both major types of noncentral hypergeometric distributions:

* **Sequential Sampling (Wallenius')**: Modeled by `select_and_remove`, where items are selected and removed one at a time.
* **Simultaneous Sampling (Fisher's)**: Modeled by `select_many_and_remove`, where a batch of unique items is selected and removed together.

### The Core Problem

In simulations, forecasts, or statistical models (e.g., [mortality models](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4060603/), Monte Carlo simulations, or machine learning sampling), managing a large, dynamic set of probabilities is common. A key task is to randomly select items based on their weights, often removing them afterward, and repeat this process efficiently. For datasets with millions of items, achieving high performance while maintaining reasonable accuracy is a significant challenge, especially for complex distributions like Wallenius' or Fisher's.

### How It Works

`DigitBinIndex` is a radix tree that organizes items into bins based on the decimal digits of their rescaled probabilities, enabling fast weighted random selection and updates.

1. **Digit-based Tree Structure**: Each level of the tree corresponds to a decimal place of the rescaled weight. For example, a weight of `0.543` at precision 3 is rescaled to `543` and placed by traversing the path: `root -> child[5] -> child[4] -> child[3]`.

2. **Roaring Bitmap Bins**: Leaf nodes act as bins, storing item IDs in a [Roaring Bitmap](https://roaringbitmap.org/), a compressed data structure optimized for fast set operations. This enables efficient unique sampling, particularly for Fisher's distribution.

3. **Accumulated Value Index**: Each node tracks the `accumulated_value` (sum of weights beneath it), supporting O(P) weighted random selection, where P is the configured precision (number of decimal places).

### Features

* **High Performance**: Outperforms general-purpose data structures like Fenwick Trees for both sequential and simultaneous weighted sampling.
* **Dual-Model Support**: Optimized methods for Wallenius' (`select_and_remove`) and Fisher's (`select_many_and_remove`) distributions.
* **O(P) Complexity**: Core operations (add, remove, select) have a time complexity of O(P), where P is the fixed precision, effectively constant for a given configuration.
* **Memory Efficiency**: Combines a sparse radix tree with Roaring Bitmaps for efficient storage, especially for sparse or clustered weight distributions.
* **Python Integration**: Seamless Python bindings via `pyo3` for cross-language support.

---

### Performance

`DigitBinIndex` trades a small, controllable amount of precision by binning probabilities to achieve significant performance gains. Below are benchmarks comparing `DigitBinIndex` against an optimized Fenwick Tree implementation, run on a standard desktop (Intel i7, 16GB RAM, Rust 1.75) with typical weight distributions.

#### Wallenius' Draw (Sequential Selections)

Measures the total time for 1,000 `select_and_remove` operations, selecting and removing one item at a time. `DigitBinIndex`'s O(P) complexity provides a significant advantage as dataset size grows.

| Number of Items (N) | `DigitBinIndex` Loop Time | `FenwickTree` Loop Time | **Speedup Factor** |
| :------------------ | :------------------------ | :---------------------- | :----------------- |
| 100,000             | **~0.46 ms**              | ~1.77 ms                | **~3.9x faster**   |
| 1,000,000           | **~0.52 ms**              | ~13.58 ms               | **~26.1x faster**  |

#### Fisher's Draw (Simultaneous Selections)

Measures the time to select a batch of unique items (1% of the total population, e.g., k=1,000 for N=100,000). `DigitBinIndex` uses batched rejection sampling, optimized for efficiency.

| Scenario (N items, draw k) | `DigitBinIndex` Time | `FenwickTree` Time | **Speedup Factor** |
| :------------------------- | :------------------- | :----------------- | :----------------- |
| N=100k, k=1k               | **~0.47 ms**         | ~1.87 ms           | **~4.0x faster**   |
| N=1M, k=10k                | **~5.48 ms**         | ~20.16 ms          | **~3.7x faster**   |

Higher precision settings slightly increase runtime but remain O(P), independent of N. Memory usage depends on the weight distribution and precision, with sparse distributions benefiting most from Roaring Bitmaps.

---

### When to Choose DigitBinIndex

Use `DigitBinIndex` when:

* You need high-performance sampling for Wallenius' or Fisher's distributions.
* Your dataset is large (N > 100,000).
* Probabilities are approximate, as is common in empirical data, simulations, or machine learning models.
* Performance is more critical than perfect precision.

Consider a Fenwick Tree if you require exact precision and your weights differ only at high decimal places (e.g., 0.12345 vs. 0.12346), though this comes at the cost of O(log N) complexity and higher memory usage for large datasets.

---

### Choosing a Precision

The `precision` parameter controls the radix tree's depth, balancing **accuracy**, **performance**, and **memory**. Higher precision improves sampling accuracy but increases memory usage (up to 10x per additional level) and slightly impacts runtime.

#### The Rule of Thumb

**A precision of 3 or 4 (default: 3) is recommended for most applications.** This captures sufficient detail for typical weight distributions while maintaining excellent performance and low memory usage.

#### The Mathematical Intuition

Each decimal place contributes exponentially less to a weight’s value. For a weight of `0.12345`:

* 1st digit (`1`): `0.1`
* 2nd digit (`2`): `0.02`
* 3rd digit (`3`): `0.003`
* 4th digit (`4`): `0.0004`

Truncating at 3 digits limits the error per item to <0.001. In large populations, these errors average out, minimally affecting the selection distribution.

#### Guidance

| Precision | Typical Use Case                                     | Trade-offs                                                   |
| :-------- | :--------------------------------------------------- | :----------------------------------------------------------- |
| **1-2**   | Maximum performance, minimal memory usage.           | Best for coarse weights (e.g., `0.1`, `0.5`). Loses accuracy with fine-grained data. |
| **3-4**   | **Recommended Default.** Optimal for most scenarios. | Captures sufficient detail for simulation or model data. Negligible performance/memory cost. |
| **5+**    | High-fidelity scenarios with very close weights.     | Distinguishes weights like `0.12345` vs. `0.12346`. Increases memory (up to 10x per level) and slightly impacts performance. |

---

### Usage & Installation

`DigitBinIndex` is available as a Python library on [PyPI](https://pypi.org/project/digit-bin-index/) or as a Rust crate on [Crates.io](https://crates.io/crates/digit-bin-index). Ensure Python 3.6+ for Python bindings or Rust 1.75+ for the Rust crate.

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
    index = DigitBinIndex(precision=3)

    # Add items with IDs and weights.
    index.add(id=101, weight=0.123)  # Low weight
    index.add(id=202, weight=0.800)  # High weight
    index.add(id=303, weight=0.755)  # High weight
    index.add(id=404, weight=0.110)  # Low weight

    # Sequential (Wallenius') Draw: Select and remove one item.
    # Higher-weighted items (202, 303) are more likely.
    selected_item = index.select_and_remove()
    if selected_item:
        print(f"Wallenius draw: ID {selected_item[0]}, Weight ~{selected_item[1]}")
    
    print(f"Items remaining: {index.count()}")  # 3

    # Simultaneous (Fisher's) Draw: Select and remove 2 unique items.
    selected_ids = index.select_many_and_remove(2)
    if selected_ids:
        print(f"Fisher's draw: {selected_ids}")
    
    print(f"Items remaining: {index.count()}")  # 1

if __name__ == "__main__":
    main()
```

#### For Rust 🦀

Add to your `Cargo.toml`:

```toml
[dependencies]
digit-bin-index = "0.2.3"    # Replace with the latest version from crates.io
rust_decimal = "1.37"
rust_decimal_macros = "1.37"  # Optional, for convenient decimal literals
```

Example usage:

```rust
use digit_bin_index::DigitBinIndex;
use rust_decimal_macros::dec;

fn main() {
    // Create an index with precision 3.
    let mut index = DigitBinIndex::with_precision(3);

    // Add items with IDs and weights.
    index.add(101, dec!(0.123)); // Low weight
    index.add(202, dec!(0.800)); // High weight
    index.add(303, dec!(0.755)); // High weight
    index.add(404, dec!(0.110)); // Low weight

    // Sequential (Wallenius') Draw: Select and remove one item.
    if let Some((id, weight)) = index.select_and_remove() {
        println!("Wallenius draw: ID {}, Weight ~{}", id, weight);
    }
    println!("Items remaining: {}", index.count()); // 3

    // Simultaneous (Fisher's) Draw: Select and remove 2 unique items.
    if let Some(ids) = index.select_many_and_remove(2) {
        println!("Fisher's draw: {:?}", ids);
    }
    println!("Items remaining: {}", index.count()); // 1
}
```

### License

This project is licensed under the [MIT License](LICENSE), a permissive open-source license allowing free use, modification, and distribution.