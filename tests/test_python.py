import math
import unittest

from digit_bin_index import DigitBinIndex


class DigitBinIndexPythonTests(unittest.TestCase):
    def test_invalid_precision_raises_value_error(self):
        constructors = [
            lambda value: DigitBinIndex.with_precision(value),
            lambda value: DigitBinIndex.with_precision_and_capacity(value, 100),
            lambda value: DigitBinIndex.small(value),
            lambda value: DigitBinIndex.medium(value),
            lambda value: DigitBinIndex.large(value),
        ]
        for constructor in constructors:
            for precision in (-1, 0, 10, 256):
                with self.subTest(precision=precision):
                    with self.assertRaises(ValueError):
                        constructor(precision)

    def test_invalid_weights_and_duplicates_raise_value_error(self):
        index = DigitBinIndex.with_precision(2)
        for weight in (math.nan, math.inf, -math.inf, -0.1, 0.0, 1.0):
            with self.subTest(weight=weight):
                with self.assertRaises(ValueError):
                    index.add(1, weight)
                self.assertEqual(index.count(), 0)

        index.add(1, 0.29)
        self.assertAlmostEqual(index.total_weight(), 0.29)
        with self.assertRaises(ValueError):
            index.add(1, 0.71)

    def test_id_boundaries_are_checked(self):
        small = DigitBinIndex.small(2)
        small.add(2**32 - 1, 0.5)
        with self.assertRaises(OverflowError):
            small.add(2**32, 0.5)

        large = DigitBinIndex.large(2)
        large.add(2**32, 0.5)
        self.assertEqual(large.count(), 1)
        with self.assertRaises(OverflowError):
            large.add(2**64, 0.5)

    def test_mutation_and_selection_smoke(self):
        index = DigitBinIndex()
        index.add_many([(1, 0.1), (2, 0.2), (3, 0.3)])
        self.assertEqual(index.count(), 3)
        self.assertFalse(index.remove(1, 0.2))
        self.assertTrue(index.remove(1, 0.1))
        selected = index.select_many_and_remove(2)
        self.assertEqual({item_id for item_id, _ in selected}, {2, 3})
        self.assertEqual(index.count(), 0)


if __name__ == "__main__":
    unittest.main()
