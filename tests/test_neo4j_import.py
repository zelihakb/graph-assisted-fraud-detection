import unittest
from unittest.mock import MagicMock

import pandas as pd

from src.graph.neo4j_import import import_graph_chunk


class TestNeo4jImport(unittest.TestCase):

    def setUp(self):
        # İki işlemden oluşan küçük bir canonical veri örneği hazırladık.
        self.chunk = pd.DataFrame({
            "tx_id": ["raw_1", "raw_2"],
            "src_id": ["C1", "C3"],
            "dst_id": ["C2", "C2"],
            "time_step": [1, 2],
            "tx_type": ["TRANSFER", "CASH_OUT"],
            "amount": [500.0, 200.0],
        })

    def test_batch_import(self):
        # Gerçek Neo4j bağlantısı yerine sahte bir session kullandık.
        session = MagicMock()
        fake_tx = MagicMock()

        session.execute_write.side_effect = (
            lambda func, tables: func(fake_tx, tables)
        )

        # Batch boyutunu 1 seçerek işlemleri ayrı gruplarda işledik.
        imported = import_graph_chunk(
            session,
            self.chunk,
            batch_size=1
        )

        self.assertEqual(imported, 2)
        self.assertEqual(session.execute_write.call_count, 2)

        # Her batch için dört graph sorgusu çalıştırıldığını kontrol ettik.
        self.assertEqual(fake_tx.run.call_count, 8)

        calls = fake_tx.run.call_args_list

        # Transaction ID'lerinin batch'ler arasında korunduğunu doğruladık.
        self.assertEqual(
            calls[1].kwargs["rows"][0]["tx_id"],
            "raw_1"
        )
        self.assertEqual(
            calls[5].kwargs["rows"][0]["tx_id"],
            "raw_2"
        )

    def test_invalid_batch_size(self):
        # Geçersiz batch boyutlarının reddedildiğini doğruladık.
        session = MagicMock()

        with self.assertRaises(ValueError):
            import_graph_chunk(
                session,
                self.chunk,
                batch_size=0
            )