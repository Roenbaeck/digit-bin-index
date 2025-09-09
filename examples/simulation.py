# Now inside Python
from digit_bin_index import DigitBinIndex

# The class name is "DigitBinIndex" because we used #[pyclass(name = "DigitBinIndex")]
index = DigitBinIndex(precision=3)

index.add(id=101, weight=0.123)
index.add(id=202, weight=0.800)
index.add(id=303, weight=0.755)

print(f"Item count: {index.count()}")

# Select and remove a single item
item = index.select_and_remove()
print(f"Selected item: {item}")

# Select and remove multiple items
items = index.select_many_and_remove(2)
print(f"Selected items: {items}")

print(f"Remaining items: {index.count()}")

# Exit python
exit()