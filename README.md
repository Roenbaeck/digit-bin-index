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

Of course. Those results are not just an improvement; they are a complete reversal and a resounding success. The change in algorithm for the Fisher's draw was clearly the right move. Congratulations!

Here is the updated performance section for your README, populated with the new, phenomenal results.

***

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

### Usage

First, add `digit-bin-index` and its dependencies to your `Cargo.toml`. The library uses the `rust_decimal` crate for high-performance, precise decimal arithmetic.

```toml
[dependencies]
digit-bin-index = "0.2.1" # Use the latest version
rust_decimal = { version = "1.32", features = ["rand"] }
roaring = "0.10"
rand = "0.8"
```

Then, you can use `DigitBinIndex` in your project to perform both sequential (Wallenius') and simultaneous (Fisher's) draws.

```rust
use digit_bin_index::DigitBinIndex;
use rust_decimal::Decimal;
use rust_decimal_macros::dec; // For easy decimal creation

fn main() {
    // Create an index with a precision of 3 decimal places.
    let mut index = DigitBinIndex::with_precision(3);

    // Add individuals with unique IDs and associated weights.
    // The `dec!` macro is a convenient way to create decimals.
    index.add(101, dec!(0.543));
    index.add(102, dec!(0.120));
    index.add(103, dec!(0.543)); // A duplicate weight is fine.
    index.add(104, dec!(0.810));
    index.add(105, dec!(0.12345)); // Note: this will be binned as 0.123

    println!(
        "Initial state: {} individuals, total weight = {}",
        index.count(),
        index.total_weight()
    );

    // --- 1. Wallenius' Draw (Sequential) ---
    // Select one item, which is immediately removed. The odds change for the next draw.
    if let Some((id, _)) = index.select_and_remove() {
        println!("\nSelected one individual (Wallenius' draw): {}", id);
        println!("State after one draw: {} individuals", index.count());
    }
    
    // --- 2. Fisher's Draw (Simultaneous) ---
    // Select a batch of 2 unique individuals from the remaining population.
    // This is a single, atomic operation.
    if let Some(ids) = index.select_many_and_remove(2) {
        println!("\nSelected two individuals (Fisher's draw): {:?}", ids);
    }
    
    println!("\nFinal state: {} individuals", index.count());
}
```

### License

This project is licensed under the [MIT License](LICENSE).
