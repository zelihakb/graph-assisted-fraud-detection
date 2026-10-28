
from getpass import getpass
from math import isclose
from uuid import uuid4

from neo4j import GraphDatabase

from src.graph.features import (
    FEATURE_COLUMNS,
    extract_receiver_history_batch,
)


URI = "bolt://127.0.0.1:17687"


def main():
    # Gerçek PaySim kimlikleriyle çakışmayan test hesapları oluşturduk.
    prefix = f"boundary_{uuid4().hex}"
    receiver = f"{prefix}_receiver"

    cases = [
        ("t100", receiver, 100, 7.0),
        ("t101a", receiver, 101, 10.0),
        ("t101b", receiver, 101, 20.0),
        ("t350", receiver, 350, 50.0),
        ("t269", receiver, 269, 30.0),
        ("t270", receiver, 270, 40.0),
        ("new_account", f"{prefix}_new", 250, 400.0),
    ]

    rows = [
        {
            "tx_id": f"{prefix}_{name}",
            "receiver_id": account,
            "time_step": step,
            "amount": amount,
        }
        for name, account, step, amount in cases
    ]

    # Beklenen sonuçları Neo4j'den bağımsız olarak elle hesapladık.
    # Sıra: count, total, average, 7d_count, 7d_total.
    expected = {
        "t100": (0, 0.0, 0.0, 0, 0.0),
        "t101a": (1, 7.0, 7.0, 1, 7.0),
        "t101b": (1, 7.0, 7.0, 1, 7.0),
        "t269": (3, 37.0, 37.0 / 3, 2, 30.0),
        "t270": (4, 67.0, 67.0 / 4, 1, 30.0),
        "t350": (5, 107.0, 107.0 / 5, 2, 70.0),
        "new_account": (0, 0.0, 0.0, 0, 0.0),
    }

    password = getpass("V2 Neo4j password: ")

    with GraphDatabase.driver(
        URI, auth=("neo4j", password)
    ) as driver:

        driver.verify_connectivity()

        with driver.session(database="neo4j") as session:
            tx = session.begin_transaction()

            try:
                # Test düğümlerini yalnızca geçici transaction içinde hazırladık.
                tx.run("""
                    UNWIND $rows AS row
                    MERGE (a:Account {
                        account_id: row.receiver_id
                    })
                    MERGE (t:Transaction {
                        tx_id: row.tx_id
                    })
                    SET t.time_step = row.time_step,
                        t.amount = row.amount
                    MERGE (t)-[:TO]->(a)
                """, rows=rows).consume()

                ids = [row["tx_id"] for row in rows]

                results = extract_receiver_history_batch(
                    tx, ids
                )

                actual = {
                    row["tx_id"]: row
                    for row in results
                }

                for name, *_, in cases:
                    tx_id = f"{prefix}_{name}"
                    observed = actual[tx_id]
                    reference = expected[name]

                    for column, wanted in zip(
                        FEATURE_COLUMNS, reference
                    ):
                        found = observed[column]

                        if not isclose(
                            float(found),
                            float(wanted),
                            rel_tol=1e-9,
                            abs_tol=1e-6,
                        ):
                            raise AssertionError(
                                f"{name} / {column}: "
                                f"expected={wanted}, actual={found}"
                            )

                    print(f"PASS: {name}")

            finally:
                # Test kayıtlarını kalıcılaştırmadan geri aldık.
                tx.rollback()

            remaining = session.run("""
                UNWIND $ids AS id
                MATCH (t:Transaction {tx_id: id})
                RETURN count(t) AS n
            """, ids=[row["tx_id"] for row in rows]).single()["n"]

            if remaining != 0:
                raise AssertionError(
                    "Geçici test kayıtları veritabanında kaldı."
                )

    print("\nSTRICT-PAST BOUNDARY TEST PASSED!")
    print("Temporary test data rolled back.")


if __name__ == "__main__":
    main()
