# DigitBinIndex

A `DigitBinIndex` is a tree-based data structure that organizes a large collection of weighted items to enable highly efficient weighted random selection and removal. It is a specialized tool, purpose-built for scenarios with millions of items where probabilities are approximate and high performance is critical.

### The Core Problem

In many simulations, forecasts, or statistical models, one needs to manage a large, dynamic set of probabilities. A common task is to randomly select an item based on its weight (probability), remove it from the set, and repeat this process thousands of times. Doing this efficiently with millions of items is a non-trivial performance challenge.

The use case for which this was originally developed is to perform fast selections in a [Wallenius' noncentral hypergeometric distribution](https://en.wikipedia.org/wiki/Wallenius%27_noncentral_hypergeometric_distribution). This sequential sampling model is common in complex agent-based simulations, such as [mortality models](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4060603/).

### How It Works

`DigitBinIndex` is a radix tree where the path is determined by the decimal digits of the probabilities. This structure allows it to group items into "bins" based on a configurable level of precision.

1.  **Digit-based Tree Structure**: The index builds a tree where each level corresponds to a decimal place. For a probability like `0.543`, an item would be placed by traversing the path: `root -> child[5] -> child[4] -> child[3]`.

2.  **Binning**: The node at the end of a path acts as a "bin," holding a list of all individuals whose probabilities are truncated to that value. For example, with a precision of 3, probabilities `0.543`, `0.5432`, and `0.5439` would all be placed in the `0.543` bin.

3.  **Accumulated Value Index**: Each node in the tree stores the `accumulated_value` (the sum of all probabilities beneath it). This is the key to its speed. To select an item, a random number is generated between 0 and the root's total value. The tree is then traversed, "spending" the random number on the accumulated values of the branches until a leaf bin is selected.

### Features

*   **Fast Selection**: Weighted random selection is an **O(P)** operation, where P is the configured precision. This is effectively constant time, independent of the number of items.
*   **Fast Updates**: Adding and removing items are also **O(P)** operations.
*   **Configurable Precision**: The desired precision can be set during instantiation, allowing you to balance accuracy with performance and memory.
*   **Memory Efficient**: For datasets where many items share the same effective probability (up to the chosen precision), this structure is highly memory efficient.

---

### Choosing the Right Tool: DigitBinIndex vs. General-Purpose Structures

`DigitBinIndex` is a specialized data structure. Its design makes a deliberate engineering trade-off: it sacrifices a small, controllable amount of precision to gain significant improvements in speed and memory usage for its target use case.

The standard, general-purpose tool for this type of problem is a **Fenwick Tree** (or Binary Indexed Tree), which can store the exact probability for every individual. Here is how they compare conceptually:

| Feature | `DigitBinIndex` (This Crate) | Fenwick Tree (General-Purpose) |
| :--- | :--- | :--- |
| **Time Complexity** | **O(P)** <br>*(P = configured precision)* | **O(log N)** <br>*(N = number of individuals)* |
| **Accuracy** | **Binned (Approximate)** <br>Quantizes probabilities to `P` decimal places. | **Perfect (Exact)** <br>Stores the precise probability for every item. |
| **Ideal Data**| Empirical probabilities (from medicine, ML, etc.) where precision beyond a few digits is noise. | Theoretical probabilities (from physics, crypto, etc.) where high precision is meaningful. |

This difference in time complexity leads to a dramatic performance gap in practice, as shown by the benchmark results below.

### Performance

The following benchmarks compare the time for a single `select_and_remove` operation for both data structures across a growing number of individuals (`N`).

| Number of Items (N) | `DigitBinIndex` Time | `FenwickTree` Time | **Speedup Factor** |
| :------------------ | :------------------- | :----------------- | :----------------- |
| 100,000             | **~10.3 µs**         | ~5,579 µs (5.58 ms)  | **~542x faster**   |
| 1,000,000           | **~346.1 µs**        | ~59,041 µs (59.0 ms) | **~171x faster**   |
| 10,000,000          | **~739.1 µs**        | ~624,050 µs (624 ms) | **~844x faster**   |

As the table shows, `DigitBinIndex` is not just faster; it is **orders of magnitude faster** for its intended use case.

#### ✅ When to Choose DigitBinIndex

This structure is the preferred choice when your scenario matches these conditions:
*   **You have a very large number of items (`N` is in the millions).**
*   **Performance is critical.**
*   **Your probabilities are approximate.** If your weights come from empirical data, simulations, or machine learning models, the precision beyond a few decimal places is often meaningless.
*   **Many items share the same effective probability.**

#### ❌ When to Consider an Alternative (like a Fenwick Tree)

You should use a more general-purpose data structure if:
*   **You require perfect, lossless precision.** If all your items have unique probabilities that only differ at a high decimal place (e.g., the 15th digit), you would need to set `P` so high that the performance and memory benefits would be lost.

---

### Usage

First, add `digit-bin-index` to your `Cargo.toml`:

```toml
[dependencies]
digit-bin-index = "0.1.0" # Replace with the actual version
fraction = "0.15" # This structure relies on the Decimal type
rand = "0.9"
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
    index.add(101, Decimal::from(0.543));    // 0.543
    index.add(102, Decimal::from(0.120));    // 0.120
    index.add(103, Decimal::from(0.543));    // another 0.543
    index.add(104, Decimal::from(0.12345));  // 0.12345

    println!(
        "Initial state: {} individuals, total weight = {}",
        index.count(),
        index.total_weight()
    );

    // select_and_remove() performs a weighted random selection and removes the item.
    if let Some((selected_id, original_weight)) = index.select_and_remove() {
        println!("\nSelected individual {} (weight ~{})", selected_id, original_weight);
    }

    println!(
        "Final state: {} individuals, total weight = {}",
        index.count(),
        index.total_weight()
    );
}
```

### License

This project is licensed under the [MIT License](LICENSE).