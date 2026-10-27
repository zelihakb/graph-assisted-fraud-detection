
from getpass import getpass
from time import perf_counter

import pandas as pd
from neo4j import GraphDatabase

from src.data.full_history import iter_scoped_paysim
from src.graph.neo4j_import import import_graph_chunk
from src.graph.schema import ensure_graph_constraints
from scripts.neo4j_real_sample import CSV_PATH, URI, get_counts


SAMPLE_SIZE = 10_000
BATCH_SIZE = 1_000


def load_benchmark_sample():
    # İlk 10.000 uygun işlemi orijinal kimlikleriyle hazırladık.
    chunks = []
    remaining = SAMPLE_SIZE

    for chunk in iter_scoped_paysim(CSV_PATH, chunk_size=50_000):
        selected = chunk.head(remaining)
        chunks.append(selected)
        remaining -= len(selected)

        if remaining == 0:
            break

    if remaining:
        raise ValueError("Yeterli PaySim işlemi bulunamadı.")

    return pd.concat(chunks, ignore_index=True)


def main():
    sample = load_benchmark_sample()

    if sample["tx_id"].duplicated().any():
        raise ValueError("Tekrarlanan transaction ID bulundu.")

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

    print("Beklenen graph:", expected)

    password = getpass("V2 Neo4j password: ")

    with GraphDatabase.driver(
        URI,
        auth=("neo4j", password),
    ) as driver:

        driver.verify_connectivity()

        with driver.session(database="neo4j") as session:
            ensure_graph_constraints(session)

            # Var olan işlemlerin bu örnekleme ait olduğunu kontrol ettik.
            before = get_counts(session)

            if before["transactions"] > SAMPLE_SIZE:
                raise RuntimeError(
                    f"Beklenenden fazla işlem var: {before}"
                )

            existing_ids = {
                record["tx_id"]
                for record in session.run(
                    "MATCH (t:Transaction) RETURN t.tx_id AS tx_id"
                )
            }

            unexpected = existing_ids - set(sample["tx_id"])

            if unexpected:
                raise RuntimeError(
                    f"Örnekleme ait olmayan {len(unexpected)} işlem var."
                )

            print("Başlangıç:", before)

            # Aynı veriyi iki kez göndererek süreleri ve tekrar davranışını ölçtük.
            for attempt in range(1, 3):
                start = perf_counter()

                processed = import_graph_chunk(
                    session,
                    sample,
                    batch_size=BATCH_SIZE,
                )

                elapsed = perf_counter() - start
                counts = get_counts(session)

                if processed != SAMPLE_SIZE or counts != expected:
                    raise AssertionError(
                        f"Import {attempt} başarısız: {counts}"
                    )

                print(f"\nImport {attempt}")
                print(f"Processed: {processed:,}")
                print(f"Elapsed: {elapsed:.2f} seconds")
                print(f"Rate: {processed / elapsed:.1f} tx/s")
                print(f"Graph: {counts}")

    print("\n10k benchmark passed!")


if __name__ == "__main__":
    main()
