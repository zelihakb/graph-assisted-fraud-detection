from getpass import getpass

import pandas as pd
from neo4j import GraphDatabase

from src.graph.schema import ensure_graph_constraints
from src.graph.neo4j_import import import_graph_chunk


URI = "bolt://127.0.0.1:17687"

# Gerçek PaySim verilerini kullanmadan iki test işlemi hazırladık.
demo = pd.DataFrame({
    "tx_id": ["smoke_tx_1", "smoke_tx_2"],
    "src_id": ["smoke_C1", "smoke_C3"],
    "dst_id": ["smoke_C2", "smoke_C2"],
    "time_step": [1, 2],
    "tx_type": ["TRANSFER", "CASH_OUT"],
    "amount": [500.0, 200.0],
})


def get_counts(session):
    # Graph'taki düğüm ve ilişki sayılarını kontrol ettik.
    queries = {
        "accounts": "MATCH (n:Account) RETURN count(n) AS n",
        "transactions": "MATCH (n:Transaction) RETURN count(n) AS n",
        "sent": "MATCH ()-[r:SENT]->() RETURN count(r) AS n",
        "to": "MATCH ()-[r:TO]->() RETURN count(r) AS n",
    }

    return {
        name: session.run(query).single()["n"]
        for name, query in queries.items()
    }


password = getpass("V2 Neo4j password: ")

with GraphDatabase.driver(
    URI,
    auth=("neo4j", password)
) as driver:

    driver.verify_connectivity()

    with driver.session(database="neo4j") as session:
        ensure_graph_constraints(session)

        # Teste başlamadan önce veritabanının boş olduğunu doğruladık.
        assert all(value == 0 for value in get_counts(session).values())

        try:
            # Aynı işlemleri iki kez aktararak tekrar davranışını test ettik.
            for attempt in range(1, 3):
                imported = import_graph_chunk(
                    session, demo, batch_size=1
                )

                counts = get_counts(session)

                assert imported == 2
                assert counts == {
                    "accounts": 3,
                    "transactions": 2,
                    "sent": 2,
                    "to": 2,
                }

                print(f"Import {attempt}: {counts}")

        finally:
            # Yalnızca oluşturduğumuz test işlemlerini temizledik.
            session.run(
                "MATCH (t:Transaction) "
                "WHERE t.tx_id IN $ids "
                "DETACH DELETE t",
                ids=demo["tx_id"].tolist(),
            ).consume()

            session.run(
                "MATCH (a:Account) "
                "WHERE a.account_id IN $ids "
                "DETACH DELETE a",
                ids=pd.unique(
                    pd.concat([
                        demo["src_id"], demo["dst_id"]
                    ])
                ).tolist(),
            ).consume()

            print("After cleanup:", get_counts(session))

print("Neo4j integration smoke test passed!")