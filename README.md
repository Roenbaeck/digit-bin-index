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

This benchmark measures the total time to perform a loop of 1,000 `select_and_remove` operations, simulating a real-world workload. The results show `DigitBinIndex`'s superior `O(P)` complexity provides a massive and growing advantage as the dataset size increases.

| Number of Items (N) | `DigitBinIndex` Loop Time | `FenwickTree` Loop Time | **Speedup Factor** |
| :------------------ | :---------------------- | :-------------------- | :----------------- |
| 100,000             | **~0.99 ms**            | ~2.93 ms              | **~3.0x faster**   |
| 1,000,000           | **~1.18 ms**            | ~22.06 ms             | **~18.7x faster**  |

#### Fisher's Draw (Simultaneous Selections)

This benchmark measures the time to select a single batch of unique items (1% of the total population). `DigitBinIndex` uses a "bin-aware" selection algorithm powered by Roaring Bitmaps, while the `FenwickTree` uses naive rejection sampling.

| Scenario (N items, draw k) | `DigitBinIndex` Time | `FenwickTree` Time | **Winner & Speedup** |
| :------------------------- | :------------------- | :----------------- | :------------------- |
| N=100k, k=1k               | **~1.52 ms**         | ~2.95 ms           | **`DigitBinIndex` (~1.9x faster)** |
| N=1M, k=10k                | ~61.7 ms             | **~35.1 ms**         | **`FenwickTree` (~1.75x faster)** |

---

### When to Choose DigitBinIndex

This structure is the preferred choice when your scenario matches these conditions:
*   **You need high-performance Wallenius' or Fisher's sampling.**
*   **Your dataset is large (`N` > 100,000).**
*   **Your probabilities are approximate.** If your weights come from empirical data, simulations, or ML models, the precision beyond a few decimal places is often meaningless.
*   **Performance is more critical than perfect precision.**

You should consider a more general-purpose data structure (like a Fenwick Tree) only if you require perfect, lossless precision *and* your data is "digitally incompressible" (e.g., all items differ only at a very high decimal place).

### Usage

First, add `digit-bin-index` to your `Cargo.toml`:

```toml
[dependencies]
digit-bin-index = "0.2.0" # Use the latest version
fraction = "0.14"
rand = "0.8"
roaring = "0.10"
```

Then, you can use it in your project:

```rust
use digit_bin_index::DigitBinIndex;
use fraction::Decimal;

fn main() {
    // Create an index with a precision of 3 decimal places.
    let mut index = DigitBinIndex::with_precision(3);

    // Add individuals with unique IDs and associated weights.
    // Note: 0.12345 will be binned as 0.123 due to the precision.
    index.add(101, Decimal::from(0.543));
    index.add(102, Decimal::from(0.120));
    index.add(103, Decimal::from(0.543));
    index.add(104, Decimal::from(0.12345));

    println!("Initial state: {} individuals", index.count());

    // --- Wallenius' Draw (Sequential) ---
    if let Some((id, _)) = index.select_and_remove() {
        println!("Selected one individual (Wallenius'): {}", id);
    }
    
    // --- Fisher's Draw (Simultaneous) ---
    // Draw 2 unique individuals from the remaining population.
    if let Some(ids) = index.select_many_and_remove(2) {
        println!("Selected two individuals (Fisher's): {:?}", ids);
    }
    
    println!("Final state: {} individuals", index.count());
}
```

### License

This project is licensed under the [MIT License](LICENSE).
