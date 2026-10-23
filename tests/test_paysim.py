import unittest
import pandas as pd

from src.data.paysim import prepare_scoped_paysim


class TestPaySimPreparation(unittest.TestCase):

    def setUp(self):
        # Testler için gerçek veri setinden bağımsız küçük bir örnek oluşturduk.
        self.raw = pd.DataFrame({
            "step": [1, 1, 2],
            "type": ["PAYMENT", "CASH_OUT", "TRANSFER"],
            "amount": [100, 200, 300],
            "nameOrig": ["C1", "C2", "C3"],
            "nameDest": ["C4", "C5", "C6"],
            "oldbalanceOrg": [500, 500, 500],
            "newbalanceOrig": [400, 300, 200],
            "oldbalanceDest": [0, 0, 0],
            "newbalanceDest": [100, 200, 300],
            "isFraud": [0, 0, 1],
            "isFlaggedFraud": [0, 0, 0],
        })

    def test_filter_and_ids(self):
        # Filtreleme sonrasında orijinal satır kimliklerinin korunduğunu doğruladık.
        result = prepare_scoped_paysim(self.raw)

        self.assertEqual(len(result), 2)
        self.assertEqual(result["tx_id"].tolist(), ["raw_2", "raw_3"])
        self.assertEqual(
            result["tx_type"].tolist(),
            ["CASH_OUT", "TRANSFER"]
        )
        self.assertTrue(result["tx_id"].is_unique)

    def test_missing_column(self):
        # Zorunlu bir kolon eksik olduğunda hata üretildiğini kontrol ettik.
        incomplete = self.raw.drop(columns=["isFraud"])

        with self.assertRaises(ValueError):
            prepare_scoped_paysim(incomplete)

    def test_canonical_values_preserved(self):
        # Dönüşüm sırasında işlem tutarlarını ve bakiyeleri koruduğumuzu doğruladık.
        result = prepare_scoped_paysim(self.raw)

        self.assertEqual(result["amount"].tolist(), [200, 300])
        self.assertEqual(result["src_balance_before"].tolist(), [500, 500])
        self.assertEqual(result["src_balance_after"].tolist(), [300, 200])
        self.assertEqual(result["dst_balance_before"].tolist(), [0, 0])
        self.assertEqual(result["dst_balance_after"].tolist(), [200, 300])

        # Fraud etiketlerinin doğru işlemlerle eşleştiğini kontrol ettik.
        self.assertEqual(result["label_fraud"].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()