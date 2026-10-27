import unittest
from unittest.mock import MagicMock

from src.graph.schema import ensure_graph_constraints


class TestGraphSchema(unittest.TestCase):

    def test_uniqueness_constraints(self):
        # Gerçek Neo4j bağlantısı kullanmadan constraint sorgularını doğruladık.
        session = MagicMock()

        ensure_graph_constraints(session)

        self.assertEqual(session.run.call_count, 2)

        queries = [
            call.args[0]
            for call in session.run.call_args_list
        ]

        # Hem Account hem Transaction kimliklerini kontrol ettik.
        self.assertIn("a.account_id IS UNIQUE", queries[0])
        self.assertIn("t.tx_id IS UNIQUE", queries[1])

        # Tekrarlanan kurulumlarda kullanılacak koşulu doğruladık.
        self.assertTrue(
            all("IF NOT EXISTS" in query for query in queries)
        )

        self.assertEqual(
            session.run.return_value.consume.call_count,
            2
        )