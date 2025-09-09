# DigitBinIndex

A `DigitBinIndex` is a tree-based data structure that organizes a large collection of weighted items to enable highly efficient weighted random selection and removal. It is a specialized tool, purpose-built for scenarios with millions of items where probabilities are approximate and high performance is critical.

This library provides state-of-the-art, high-performance solutions for both major types of noncentral hypergeometric distributions:
*   **Sequential Sampling (Wallenius'):** Modeled by `select_and_remove`.
*   **Simultaneous Sampling (Fisher's):** Modeled by `select_many_and_remove`.

### The Core Problem

In many simulations, forecasts, or statistical models, one needs to manage a large, dynamic set of probabilities. A common task is to randomly select items based on their weight, remove them, and repeat. Doing this efficiently with millions of items is a non-trivial performance challenge, especially when modeling complex behaviors like [Wallenius'](https://en.wikipedia.org/wiki/Wallenius%27_noncentral_hypergeometric_distribution) or [Fisher's](https://en.wikipedia.org/wiki/Fisher%27s_noncentral_hypergeometric_distribution) distributions, which are common in agent-based simulations like [mortality models](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4060603/).

### How It Works

`DigitBinIndex` is a radix tree where the path is determined by the decimal digits of the probabilities. This structure allows it to group items into "bins" based on a configurable level of precision.

1.  **Digit-based Tree Structure**: The index builds a tree where each level corresponds to a decimal place. For a probability like `0.543`, an item would be placed by traversing the path: `root -> child[5] -> child[4] -> child[3]`.

2.  **Roaring Bitmap Bins**: The node at the end of a path acts as a "bin." Instead of a simple list, it holds a [**Roaring Bitmap**](https://roaringbitmap.org/), a highly optimized data structure for storing and performing set operations on integers. This is the key to the library's high performance for simultaneous (Fisher's) draws.

3.  **Accumulated Value Index**: Each node in the tree stores the `accumulated_value` (the sum of all probabilities beneath it). This allows for extremely fast `O(P)` weighted random selection, where `P` is the configured precision.

### Features

*   **State-of-the-Art Performance:** Outperforms standard, general-purpose data structures for both sequential and simultaneous weighted sampling.
*   **Dual-Model Support:** Provides optimized methods for both Wallenius' (`select_and_remove`) and Fisher's (`select_many_and_remove`) distributions.
*   **Effectively O(1) Complexity:** Core operations have a time complexity of `O(P)`, where `P` is the configured precision. This is effectively constant time, independent of the number of items.
*   **Memory Efficient:** The combination of a sparse tree and Roaring Bitmaps makes it highly memory-efficient for most datasets.

---

### Performance

`DigitBinIndex` makes a deliberate engineering trade-off: it sacrifices a small, controllable amount of precision by binning probabilities to gain significant improvements in speed.

The standard alternative is a **Fenwick Tree**, which is perfectly accurate but has a slower `O(log N)` complexity. The benchmarks below compare `DigitBinIndex` against a highly optimized Fenwick Tree implementation.

#### Wallenius' Draw (Sequential Selections)

This benchmark measures the total time to perform a loop of 1,000 `select_and_remove` operations. The results show `DigitBinIndex`'s superior `O(P)` complexity provides a massive and growing advantage as the dataset size increases.

| Number of Items (N) | `DigitBinIndex` Loop Time | `FenwickTree` Loop Time | **Speedup Factor** |
| :------------------ | :---------------------- | :-------------------- | :----------------- |
| 100,000             | **~0.46 ms**            | ~1.77 ms              | **~3.9x faster**   |
| 1,000,000           | **~0.52 ms**            | ~13.58 ms             | **~26.1x faster**  |

#### Fisher's Draw (Simultaneous Selections)

This benchmark measures the time to select a single batch of unique items (1% of the total population). After algorithmic improvements, `DigitBinIndex` now uses a batched rejection sampling approach that is significantly more efficient than its previous method and faster than the Fenwick Tree's equivalent.

| Scenario (N items, draw k) | `DigitBinIndex` Time | `FenwickTree` Time | **Speedup Factor** |
| :------------------------- | :------------------- | :----------------- | :----------------- |
| N=100k, k=1k               | **~0.47 ms**         | ~1.87 ms           | **~4.0x faster**   |
| N=1M, k=10k                | **~5.48 ms**         | ~20.16 ms          | **~3.7x faster**   |

As the results show, `DigitBinIndex` outperforms the Fenwick Tree in both sequential and simultaneous batched selection scenarios, making it a highly effective tool for large-scale weighted random sampling simulations.

---

### When to Choose DigitBinIndex

This structure is the preferred choice when your scenario matches these conditions:
*   **You need high-performance Wallenius' or Fisher's sampling.**
*   **Your dataset is large (`N` > 100,000).**
*   **Your probabilities are approximate.** If your weights come from empirical data, simulations, or ML models, the precision beyond a few decimal places is often meaningless.
*   **Performance is more critical than perfect precision.**

You should consider a more general-purpose data structure (like a Fenwick Tree) only if you require perfect, lossless precision *and* your data is "digitally incompressible" (e.g., all items differ only at a very high decimal place).

---

### Choosing a Precision

The `precision` parameter controls the depth of the radix tree and represents the core trade-off of the library: **Accuracy vs. Performance & Memory**. Choosing the right value is key to getting the most out of `DigitBinIndex`.

#### The Rule of Thumb

**For most applications, a precision of 3 or 4 is an excellent starting point.** This provides a great balance, capturing the vast majority of the weight distribution while remaining extremely fast.

#### The Mathematical Intuition

The impact of each decimal place on an item's probability diminishes exponentially. Consider a weight of `0.12345`:

*   The 1st digit (`1`) contributes `0.1` to the value.
*   The 2nd digit (`2`) contributes `0.02`.
*   The 3rd digit (`3`) contributes `0.003`.
*   The 4th digit (`4`) contributes only `0.0004`.

By truncating at 3 digits, the maximum error for any single item is less than `0.001`. When sampling from a large population, these small, random errors tend to average out, having a negligible effect on the final distribution of selections. The first few digits capture almost all of the meaningful relative differences between item weights, which is what drives a weighted random draw.

#### Guidance

Here is a summary to help guide your choice:

| Precision | Typical Use Case                                     | Trade-offs                                                                      |
| :-------- | :--------------------------------------------------- | :------------------------------------------------------------------------------ |
| **1-2**   | Maximum performance, minimal memory usage.           | Best for coarse-grained weights (e.g., `0.1`, `0.5`, `0.9`). Loses significant accuracy with finely-grained data. |
| **3-4**   | **Recommended Default.** The optimal balance for most scenarios. | Captures sufficient detail for typical floating-point data from simulations or models, with negligible performance cost. |
| **5+**    | High-fidelity scenarios where weights are very close. | Use if you must distinguish between weights like `0.12345` and `0.12346`. This increases memory usage and is slightly slower, but still `O(P)`. |

---

### Usage

First, add `digit-bin-index` to your project's dependencies in `Cargo.toml`. You will also need `rust_decimal` because it is used in the public API for defining item weights.

```toml
[dependencies]
digit-bin-index = "0.2.2"    # Use the latest version from crates.io
rust_decimal = "1.37"        # The decimal type used in the API
rust_decimal_macros = "1.37" # Recommended for easily creating decimals
```

Then, you can use `DigitBinIndex` in your project to perform both sequential (Wallenius') and simultaneous (Fisher's) draws.

```rust
use digit_bin_index::DigitBinIndex;
use rust_decimal_macros::dec; // A convenient macro, dec!

fn main() {
    // Create a new index with a precision of 3 decimal places.
    let mut index = DigitBinIndex::with_precision(3);

    // Add individuals with their ID and a specific weight.
    index.add(101, dec!(0.123)); // Low weight
    index.add(202, dec!(0.800)); // High weight
    index.add(303, dec!(0.755)); // High weight
    index.add(404, dec!(0.110)); // Low weight

    // --- Example 1: Sequential (Wallenius') Draw ---
    // Select one item, which is removed from the pool for subsequent draws.
    // The higher weighted items (202, 303) are more likely to be chosen.
    if let Some((selected_id, approx_weight)) = index.select_and_remove() {
        println!("Wallenius draw selected: ID {}, Weight ~{}", selected_id, approx_weight);
    }
    println!("Items remaining: {}", index.count()); // Will be 3

    // --- Example 2: Simultaneous (Fisher's) Draw ---
    // Select a batch of 2 unique items.
    // This is more efficient than calling select_and_remove() in a loop.
    if let Some(selected_ids) = index.select_many_and_remove(2) {
        println!("Fisher's draw selected IDs: {:?}", selected_ids);
    }
    println!("Items remaining: {}", index.count()); // Will be 1
}
```

### License

This project is licensed under the [MIT License](LICENSE).
