import unittest
import pandas as pd

from src.graph.tables import build_graph_tables


class TestGraphTables(unittest.TestCase):

    def setUp(self):
        # İki işlemin aynı alıcı hesabına ulaştığı örnek veriyi hazırladık.
        self.chunk = pd.DataFrame({
            "tx_id": ["raw_1", "raw_2"],
            "src_id": ["C1", "C3"],
            "dst_id": ["C2", "C2"],
            "time_step": [1, 2],
            "tx_type": ["TRANSFER", "CASH_OUT"],
            "amount": [500, 200],
        })

    def test_graph_structure(self):
        # İşlemleri graph düğümlerine ve ilişkilerine dönüştürdük.
        tables = build_graph_tables(self.chunk)

        # Tekrarlanan alıcıyı yalnızca bir hesap olarak koruduğumuzu doğruladık.
        self.assertEqual(len(tables["accounts"]), 3)
        self.assertEqual(
            set(tables["accounts"]["account_id"]),
            {"C1", "C2", "C3"}
        )

        # Her işlemin ayrı bir Transaction düğümü olduğunu kontrol ettik.
        self.assertEqual(len(tables["transactions"]), 2)

        # Her işlem için bir SENT ve bir TO ilişkisi oluşturduk.
        self.assertEqual(len(tables["sent_relationships"]), 2)
        self.assertEqual(len(tables["to_relationships"]), 2)

        # Her iki işlemin de doğru alıcı hesaba bağlandığını doğruladık.
        self.assertEqual(
            tables["to_relationships"]["to_account_id"].tolist(),
            ["C2", "C2"]
        )

    def test_duplicate_transaction_ids(self):
        # Tekrarlanan işlem kimliklerinin kabul edilmediğini doğruladık.
        duplicated = self.chunk.copy()
        duplicated.loc[1, "tx_id"] = "raw_1"

        with self.assertRaises(ValueError):
            build_graph_tables(duplicated)