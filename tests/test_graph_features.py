
import unittest

from src.graph.features import (
    FEATURE_COLUMNS,
    RECEIVER_HISTORY_QUERY,
)


class TestGraphFeatures(unittest.TestCase):

    def test_strict_past_filter(self):
        # Aynı veya gelecek zaman adımlarını dışarıda bıraktık.
        self.assertIn(
            "past.time_step < current.time_step",
            RECEIVER_HISTORY_QUERY,
        )

    def test_seven_day_boundary(self):
        # Yedi günlük pencerenin 168 saat olduğunu doğruladık.
        self.assertIn(
            "current.time_step - 168",
            RECEIVER_HISTORY_QUERY,
        )

    def test_no_forbidden_fields(self):
        # Sorgunun yasaklı alanları okumadığını kontrol ettik.
        for forbidden in (
            "label_fraud",
            "label_flagged",
            "src_balance_before",
            "src_balance_after",
            "dst_balance_before",
            "dst_balance_after",
        ):
            self.assertNotIn(
                forbidden,
                RECEIVER_HISTORY_QUERY,
            )

    def test_feature_contract(self):
        self.assertEqual(len(FEATURE_COLUMNS), 5)
        self.assertEqual(len(set(FEATURE_COLUMNS)), 5)


if __name__ == "__main__":
    unittest.main()
