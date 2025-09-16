# DigitBinIndex

A `DigitBinIndex` is a tree-based data structure designed for efficient weighted random selection and removal from large collections of items. It is optimized for scenarios involving millions of items where probabilities are approximate and high performance is critical, such as simulations for [Wallenius' noncentral hypergeometric distribution](https://en.wikipedia.org/wiki/Wallenius%27_noncentral_hypergeometric_distribution) or [Fisher's noncentral hypergeometric distribution](https://en.wikipedia.org/wiki/Fisher%27s_noncentral_hypergeometric_distribution).

This library provides high-performance solutions for both major types of noncentral hypergeometric distributions:

*   **Sequential Sampling (Wallenius')**: Modeled by `select_and_remove`, where items are selected and removed one at a time.
*   **Simultaneous Sampling (Fisher's)**: Modeled by `select_many_and_remove`, where a batch of unique items is selected and removed together.

### The Core Problem

In simulations, forecasts, or statistical models (e.g., [mortality models](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4060603/), Monte Carlo simulations, or machine learning sampling), managing a large, dynamic set of probabilities is common. A key task is to randomly select items based on their weights, often removing them afterward, and repeat this process efficiently. For datasets with millions of items, achieving high performance while maintaining reasonable accuracy is a significant challenge, especially for complex distributions like Wallenius' or Fisher's.

### How It Works

`DigitBinIndex` is a radix tree that organizes items into bins based on the decimal digits of their rescaled probabilities, enabling fast weighted random selection and updates.

1.  **Digit-based Tree Structure**: Each level of the tree corresponds to a decimal place of the rescaled weight. For example, a weight of `0.543` at precision 3 is rescaled to `543` and placed by traversing the path: `root -> child[5] -> child[4] -> child[3]`.

2.  **Adaptive Bin Storage**: Leaf nodes act as bins, storing item IDs in either a fast `Vec<u32>` (for small bins) or a compressed [Roaring Bitmap](https://roaringbitmap.org/) (for large bins). The bin type is chosen automatically for optimal performance and memory use if a capacity hint is given.

3.  **Accumulated Value Index**: Each node tracks the `accumulated_value` (sum of weights beneath it), supporting O(P) weighted random selection, where P is the configured precision (number of decimal places).

### Features

*   **High Performance**: Outperforms general-purpose data structures like Fenwick Trees for both sequential and simultaneous weighted sampling.
*   **Dual-Model Support**: Optimized methods for Wallenius' (`select_and_remove`) and Fisher's (`select_many_and_remove`) distributions.
*   **O(P) Complexity**: Core operations (add, remove, select) have a time complexity of O(P), where P is the fixed precision, effectively constant for a given configuration.
*   **Memory Efficiency**: Combines a sparse radix tree with Roaring Bitmaps for efficient storage, especially for sparse or clustered weight distributions.
*   **Python Integration**: Seamless Python bindings via `pyo3` for cross-language support.

---

### Performance

`DigitBinIndex` trades a small, controllable amount of precision by binning probabilities to achieve significant performance gains. The following benchmarks compare `DigitBinIndex` against an optimized `FenwickTree` implementation in a realistic, high-churn simulation.

The benchmark scenario starts with a large population (1M or 10M items), then simulates a high volume of activity:
*   **Churn**: A significant number of items are selected and removed.
*   **Acquisition**: New items are added to the population.

This measures the real-world throughput of both data structures under dynamic conditions. The tests were run on a standard desktop (Intel i7, 16GB RAM, Rust 1.75).

---

#### Wallenius' Draw (Sequential Churn)

This benchmark simulates sequential selection by removing 100,000 items one-by-one, then adding 110,000 new items. This is a common pattern in agent-based models or iterative simulations. `DigitBinIndex`'s O(P) complexity gives it a decisive advantage as the population scales.

| Scenario (N items)         | `DigitBinIndex` Time | `FenwickTree` Time | **Speedup Factor** |
| :------------------------- | :------------------- | :----------------- | :----------------- |
| **1 Million Items** (p=3)  | **~27.5 ms**         | ~82.2 ms           | **~3.0x faster**   |
| **1 Million Items** (p=5)  | **~39.7 ms**         | ~77.7 ms           | **~2.0x faster**   |
| **10 Million Items** (p=3) | **~551.0 ms**        | ~1723.1 ms         | **~3.1x faster**   |

*   **Key Takeaway**: `DigitBinIndex` is **over 3.0 times faster** than the `FenwickTree` for sequential operations on large datasets. Its performance is dependent on precision (`P`) and not the number of items (`N`), allowing it to scale far more effectively.

---

#### Fisher's Draw (Batch Churn)

This benchmark simulates simultaneous selection by removing a large batch of 100,000 unique items at once, followed by the acquisition of 110,000 new items. `DigitBinIndex` uses a highly optimized batch rejection sampling method.

| Scenario (N items)         | `DigitBinIndex` Time | `FenwickTree` Time | **Speedup Factor** |
| :------------------------- | :------------------- | :----------------- | :----------------- |
| **1 Million Items** (p=3)  | **~19.5 ms**         | ~104.2 ms          | **~5.3x faster**   |
| **1 Million Items** (p=5)  | **~39.9 ms**         | ~106.6 ms          | **~2.7x faster**   |
| **10 Million Items** (p=3) | **~389.0 ms**        | ~1649.9 ms         | **~4.2x faster**   |

**Key Takeaway**: For batch selections, `DigitBinIndex` is even more efficient, performing **up to 5.3 times faster**. The batched nature of the operation further highlights the architectural advantages of the radix tree approach for this use case.

---

### When to Choose DigitBinIndex

Use `DigitBinIndex` when:

*   You need high-performance sampling for Wallenius' or Fisher's distributions.
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

Truncating at 3 digits limits the error per item to <0.001. In large populations, these errors average out, minimally affecting the selection distribution.

#### Guidance

| Precision | Typical Use Case                                     | Trade-offs                                                   |
| :-------- | :--------------------------------------------------- | :----------------------------------------------------------- |
| **1-2**   | Maximum performance, minimal memory usage.           | Best for coarse weights (e.g., `0.1`, `0.5`). Loses accuracy with fine-grained data. |
| **3-4**   | **Recommended Default.** Optimal for most scenarios. | Captures sufficient detail for simulation or model data. Negligible performance/memory cost. |
| **5+**    | High-fidelity scenarios with very close weights.     | Distinguishes weights like `0.12345` vs. `0.12346`. Increases memory (up to 10x per level) and slightly impacts performance. |

---

## Internal Storage and Capacity

The `DigitBinIndex` is designed to handle a vast range of use cases, from a few thousand items to trillions, by automatically selecting the most appropriate internal storage engine.

### Item Capacity

The index accepts **`u64`** for individual item IDs. However, the internal storage of these IDs depends on the backend chosen.

> [!WARNING]
> **`u64` ID Truncation:** The `Small (Vec<u32>)` and `Medium (RoaringBitmap)` backends are optimized for `u32` IDs. If you insert an ID that is larger than `u32::MAX` (4,294,967,295) into one of these backends, **it will be silently truncated to a `u32`**.
>
> Only the `Large (RoaringTreemap)` backend provides full `u64` ID support. The constructor `with_precision_and_capacity` attempts to guess if you need `u64` support based on the total capacity. If you know you need to store `u64` IDs, it is recommended to manually construct the `Large` variant:
>
> ```rust
> use digit_bin_index::{DigitBinIndex, DigitBinIndexGeneric};
> use roaring::RoaringTreemap;
> 
> // Manually create an index with full u64 support
> let mut index = DigitBinIndex::Large(DigitBinIndexGeneric::<RoaringTreemap>::with_precision(3));
> ```

### Automatic Engine Selection

To provide the best balance of performance and memory usage, the library's `DigitBinIndex` is an enum that automatically switches between three different backends (`Small`, `Medium`, and `Large`) when you use the `with_precision_and_capacity()` constructor.

The selection is based on a simple heuristic: the **average number of items expected per bin**, which is calculated as `capacity / 10^precision`.

1.  `Small` (**`Vec<u32>`**):
    *   **Trigger:** Low average items per bin (<= 1,000).
    *   **Best for:** Small to medium-sized problems where `select_and_remove` speed is the absolute priority (O(1) `swap_remove`).
    *   **Warning:** Truncates `u64` IDs. `remove_many` can be O(N) per bin.

2.  `Medium` (**`RoaringBitmap`**):
    *   **Trigger:** Medium to large average items per bin (> 1,000).
    *   **Best for:** Large-scale problems (millions to billions of items) where IDs fit within `u32`. Provides excellent memory compression and fast set operations (including `remove_many`).
    *   **Warning:** Truncates `u64` IDs.

3.  `Large` (**`RoaringTreemap`**):
    *   **Trigger:** Extremely large average items per bin (> 1,000,000,000). This is used as a heuristic to detect that full `u64` support is required.
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
*   **Result:** Since 100 <= 1,000, the `Small` variant is chosen. This provides the fastest O(1) `select_and_remove` performance. (This would truncate `u64` IDs).

#### Example 2: `Medium (RoaringBitmap)` is Chosen

You need to index 50 million product IDs, all of which fit within `u32`.

```rust
// Expecting 50 million items with 3-digit precision
let index = DigitBinIndex::with_precision_and_capacity(3, 50_000_000);
```

*   **Calculation:** The average items per bin is `50,000,000 / 1,000 = 50,000`.
*   **Result:** This is > 1,000. The `Medium` variant is selected, using `RoaringBitmap`. This will be highly memory-efficient and very fast for all operations, including `remove_many`. (This would also truncate `u64` IDs).

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

    # Sequential (Wallenius') Draw: Select and remove one item.
    # Higher-weighted items (202, 303) are more likely.
    selected_item = index.select_and_remove()
    if selected_item:
        # The returned weight is a float, representing the bin's average weight
        item_id, weight = selected_item
        print(f"Wallenius draw: ID {item_id}, Weight ~{weight:.3f}")
    
    print(f"Items remaining: {index.count()}")  # 3

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
digit-bin-index = "0.4.0" # Replace with the latest version from crates.io
```

Example usage:

```rust
use digit_bin_index::DigitBinIndex;

fn main() {
    // Create an index with precision 3.
    let mut index = DigitBinIndex::with_precision(3);

    // Add items with IDs and f64 weights.
    index.add(101, 0.123); // Low weight
    index.add(202, 0.800); // High weight
    index.add(303, 0.755); // High weight
    index.add(404, 0.110); // Low weight

    // Sequential (Wallenius') Draw: Select and remove one item.
    if let Some((id, weight)) = index.select_and_remove() {
        println!("Wallenius draw: ID {}, Weight ~{}", id, weight);
    }
    println!("Items remaining: {}", index.count()); // 3

    // Simultaneous (Fisher's) Draw: Select and remove 2 unique items.
    if let Some(items) = index.select_many_and_remove(2) {
        println!("Fisher's draw: {:?}", items);
    }
    println!("Items remaining: {}", index.count()); // 1
}
```

### License

This project is licensed under the [MIT License](LICENSE), a permissive open-source license allowing free use, modification, and distribution.