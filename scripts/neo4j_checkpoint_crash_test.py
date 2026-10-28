
import tempfile
from getpass import getpass
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pandas as pd
from neo4j import GraphDatabase

from scripts import neo4j_full_import as full_import
from scripts.neo4j_real_sample import URI
from src.graph.neo4j_import import import_graph_chunk


def inspect_test_graph(session, tx_ids, account_ids):
    # Test kimliklerine ait kayıtları ve bağlantıları kontrol ettik.
    queries = {
        "accounts": (
            "MATCH (a:Account) WHERE a.account_id IN $ids "
            "RETURN count(a) AS n"
        ),
        "transactions": (
            "MATCH (t:Transaction) WHERE t.tx_id IN $ids "
            "RETURN count(t) AS n"
        ),
        "sent": (
            "MATCH (:Account)-[r:SENT]->(t:Transaction) "
            "WHERE t.tx_id IN $ids RETURN count(r) AS n"
        ),
        "to": (
            "MATCH (t:Transaction)-[r:TO]->(:Account) "
            "WHERE t.tx_id IN $ids RETURN count(r) AS n"
        ),
    }

    counts = {}

    for name, query in queries.items():
        ids = account_ids if name == "accounts" else tx_ids
        counts[name] = session.run(query, ids=ids).single()["n"]

    paths = {}

    for tx_id in tx_ids:
        paths[tx_id] = session.run("""
            MATCH (sender:Account)-[:SENT]->
                  (t:Transaction {tx_id: $tx_id})
                  -[:TO]->(receiver:Account)
            RETURN
                sender.account_id AS sender,
                receiver.account_id AS receiver,
                t.time_step AS time_step,
                t.amount AS amount
        """, tx_id=tx_id).data()

    return counts, paths


def main():
    prefix = f"crash_test_{uuid4().hex}"

    tx_ids = [f"{prefix}_tx1", f"{prefix}_tx2"]
    account_ids = [
        f"{prefix}_sender1",
        f"{prefix}_sender2",
        f"{prefix}_receiver",
    ]

    # Gerçek PaySim işlemlerinden bağımsız test verileri hazırladık.
    demo = pd.DataFrame({
        "tx_id": tx_ids,
        "src_id": account_ids[:2],
        "dst_id": [account_ids[2], account_ids[2]],
        "time_step": [1, 2],
        "tx_type": ["TRANSFER", "CASH_OUT"],
        "amount": [500.0, 200.0],
    })

    expected_counts = {
        "accounts": 3,
        "transactions": 2,
        "sent": 2,
        "to": 2,
    }

    expected_paths = {
        tx_ids[0]: [{
            "sender": account_ids[0],
            "receiver": account_ids[2],
            "time_step": 1,
            "amount": 500.0,
        }],
        tx_ids[1]: [{
            "sender": account_ids[1],
            "receiver": account_ids[2],
            "time_step": 2,
            "amount": 200.0,
        }],
    }

    password = getpass("V2 Neo4j password: ")

    # Gerçek checkpoint yerine geçici bir dosya kullandık.
    with tempfile.TemporaryDirectory() as temp_dir:
        temporary_checkpoint = (
            Path(temp_dir) / "checkpoint.json"
        )

        with (
            patch.object(
                full_import,
                "CHECKPOINT_PATH",
                temporary_checkpoint,
            ),
            patch.object(
                full_import,
                "source_info",
                return_value={
                    "path": "synthetic-crash-test",
                    "size": 2,
                },
            ),
        ):
            full_import.save_checkpoint(0)

            with GraphDatabase.driver(
                URI,
                auth=("neo4j", password),
            ) as driver:

                driver.verify_connectivity()

                with driver.session(database="neo4j") as session:

                    empty_counts = {
                        key: 0 for key in expected_counts
                    }

                    before, _ = inspect_test_graph(
                        session, tx_ids, account_ids
                    )

                    if before != empty_counts:
                        raise RuntimeError(
                            "Test kimlikleri önceden mevcut."
                        )

                    try:
                        # İlk batch'i başarıyla veritabanına yazdık.
                        first = import_graph_chunk(
                            session, demo, batch_size=2
                        )

                        first_counts, first_paths = inspect_test_graph(
                            session, tx_ids, account_ids
                        )

                        assert first == 2
                        assert first_counts == expected_counts
                        assert first_paths == expected_paths

                        print("PASS: First batch committed.")

                        # Checkpoint yazılmadan kesinti yaşanmasını taklit ettik.
                        # Bu aşamada save_checkpoint(2) çağırmadık.
                        assert full_import.load_checkpoint() == 0

                        print("PASS: Checkpoint still at 0.")

                        # Yeniden başlatıldığında kayıtlı konumdan devam ettik.
                        resume_from = full_import.load_checkpoint()

                        replayed = import_graph_chunk(
                            session,
                            demo.iloc[resume_from:],
                            batch_size=2,
                        )

                        second_counts, second_paths = inspect_test_graph(
                            session, tx_ids, account_ids
                        )

                        assert replayed == 2
                        assert second_counts == expected_counts
                        assert second_paths == expected_paths

                        print("PASS: Replay created no duplicates.")

                        # Tekrarlanan aktarımın ardından ilerlemeyi kaydettik.
                        full_import.save_checkpoint(2)

                        assert full_import.load_checkpoint() == 2

                        print("PASS: Checkpoint recovered to 2.")

                    finally:
                        # Yalnızca bu teste ait işlem düğümlerini temizledik.
                        session.run("""
                            MATCH (t:Transaction)
                            WHERE t.tx_id IN $ids
                            DETACH DELETE t
                        """, ids=tx_ids).consume()

                        # Test hesaplarını, bağlantıları yoksa sildik.
                        session.run("""
                            MATCH (a:Account)
                            WHERE a.account_id IN $ids
                              AND NOT EXISTS {
                                  MATCH (a)--()
                              }
                            DELETE a
                        """, ids=account_ids).consume()

                    after, _ = inspect_test_graph(
                        session, tx_ids, account_ids
                    )

                    assert after == empty_counts

                    print("PASS: Temporary graph data cleaned.")

    print("\nCHECKPOINT CRASH-CONSISTENCY TEST PASSED!")


if __name__ == "__main__":
    main()
