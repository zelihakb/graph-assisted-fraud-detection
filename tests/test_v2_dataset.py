
import unittest

import pandas as pd

from src.data.v2_dataset import (
    MODEL_COLUMNS,
    MODEL_INPUTS,
    assemble_model_chunk,
    partition_model_chunk,
)
from src.graph.features import FEATURE_COLUMNS


class TestV2Dataset(unittest.TestCase):

    def setUp(self):
        # Split sınırlarını temsil eden dört örnek hazırladık.
        self.raw = pd.DataFrame({
            "tx_id": ["raw_1", "raw_2", "raw_3", "raw_4"],
            "time_step": [322, 323, 376, 377],
            "tx_type": ["TRANSFER"] * 4,
            "amount": [100.0, 200.0, 300.0, 400.0],
            "label_fraud": [0, 1, 0, 1],
            "src_balance_before": [999.0] * 4,
        })

        self.features = pd.DataFrame({
            "tx_id": self.raw["tx_id"],
            **{
                column: [0.0] * 4
                for column in FEATURE_COLUMNS
            },
        })

    def test_frozen_boundaries(self):
        model = assemble_model_chunk(
            self.raw, self.features
        )
        splits = partition_model_chunk(model)

        self.assertEqual(
            splits["train"]["tx_id"].tolist(),
            ["raw_1"],
        )
        self.assertEqual(
            splits["validation"]["tx_id"].tolist(),
            ["raw_2", "raw_3"],
        )
        self.assertEqual(
            splits["test"]["tx_id"].tolist(),
            ["raw_4"],
        )

    def test_forbidden_fields_excluded(self):
        model = assemble_model_chunk(
            self.raw, self.features
        )

        self.assertEqual(
            model.columns.tolist(), MODEL_COLUMNS
        )

        self.assertNotIn(
            "src_balance_before", model.columns
        )
        self.assertNotIn("label_fraud", MODEL_INPUTS)
        self.assertNotIn("time_step", MODEL_INPUTS)
        self.assertNotIn("tx_id", MODEL_INPUTS)

    def test_missing_transaction_rejected(self):
        wrong_features = self.features.copy()
        wrong_features.loc[0, "tx_id"] = "raw_missing"

        with self.assertRaises(ValueError):
            assemble_model_chunk(
                self.raw, wrong_features
            )


if __name__ == "__main__":
    unittest.main()
