
from getpass import getpass
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

from src.data.full_history import iter_scoped_paysim
from src.graph.neo4j_import import import_graph_chunk
from src.graph.schema import ensure_graph_constraints


CSV_PATH = Path("data/paysim.csv")
URI = "bolt://127.0.0.1:17687"
SAMPLE_SIZE = 1_000


def load_sample():
    # İlk 1.000 uygun PaySim işlemini özgün kimlikleriyle hazırladık.
    pieces = []
    remaining = SAMPLE_SIZE

    for chunk in iter_scoped_paysim(CSV_PATH, chunk_size=50_000):
        piece = chunk.head(remaining)
        pieces.append(piece)
        remaining -= len(piece)

        if remaining == 0:
            break

    if remaining > 0:
        raise ValueError("CSV dosyasında yeterli uygun işlem bulunamadı.")

    sample = pd.concat(pieces, ignore_index=True)

    if sample["tx_id"].duplicated().any():
        raise ValueError("Sample içinde tekrar eden transaction ID bulundu.")

    return sample


def get_counts(session):
    # Neo4j'deki düğüm ve ilişki sayılarını ayrı ayrı kontrol ettik.
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


def main():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV bulunamadı: {CSV_PATH}")

    sample = load_sample()

    account_ids = pd.unique(
        pd.concat(
            [sample["src_id"], sample["dst_id"]],
            ignore_index=True,
        )
    )

    expected = {
        "accounts": len(account_ids),
        "transactions": SAMPLE_SIZE,
        "sent": SAMPLE_SIZE,
        "to": SAMPLE_SIZE,
    }

    print("Hazırlanan işlem:", len(sample))
    print("Beklenen graph:", expected)

    password = getpass("V2 Neo4j password: ")

    with GraphDatabase.driver(
        URI,
        auth=("neo4j", password),
    ) as driver:

        driver.verify_connectivity()

        with driver.session(database="neo4j") as session:
            ensure_graph_constraints(session)

            # Teste boş V2 ortamında başladığımızı doğruladık.
            before = get_counts(session)

            if any(before.values()):
                raise RuntimeError(
                    f"V2 veritabanı boş değil: {before}. "
                    "Mevcut verileri silmeden işlemi durdurduk."
                )

            for attempt in range(1, 3):
                imported = import_graph_chunk(
                    session,
                    sample,
                    batch_size=250,
                )

                counts = get_counts(session)

                if imported != SAMPLE_SIZE or counts != expected:
                    raise AssertionError(
                        f"Import {attempt} doğrulaması başarısız: {counts}"
                    )

                print(f"Import {attempt}: {counts}")

    print("Real PaySim sample import passed!")


if __name__ == "__main__":
    main()
