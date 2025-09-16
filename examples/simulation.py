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
