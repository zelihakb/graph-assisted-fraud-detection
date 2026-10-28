
import argparse
import random
from getpass import getpass
from pathlib import Path
from time import perf_counter

from neo4j import GraphDatabase, READ_ACCESS

from src.data.full_history import iter_scoped_paysim
from src.graph.features import extract_receiver_history_batch


CSV_PATH = Path("data/paysim.csv")
URI = "bolt://127.0.0.1:17687"

TOTAL_TRANSACTIONS = 2_770_409
MAX_HOURS = 6.0


def select_sample_ids(sample_size):
    # Sabit seed ile tüm zaman aralığından örnek işlem kimlikleri seçtik.
    positions = sorted(
        random.Random(42).sample(
            range(TOTAL_TRANSACTIONS),
            sample_size,
        )
    )

    ids = []
    seen = 0
    index = 0

    for chunk in iter_scoped_paysim(CSV_PATH):
        end = seen + len(chunk)

        while index < len(positions) and positions[index] < end:
            local_index = positions[index] - seen
            ids.append(chunk["tx_id"].iloc[local_index])
            index += 1

        seen = end

        if index == len(positions):
            break

    if len(ids) != sample_size:
        raise RuntimeError("Örneklem eksik oluşturuldu.")

    return ids


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-size",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
    )

    args = parser.parse_args()

    if not 0 < args.sample_size <= TOTAL_TRANSACTIONS:
        parser.error("Geçersiz sample-size.")

    if args.batch_size <= 0:
        parser.error("batch-size pozitif olmalı.")

    if not CSV_PATH.is_file():
        raise FileNotFoundError(CSV_PATH)

    print("Örneklem hazırlanıyor...", flush=True)
    tx_ids = select_sample_ids(args.sample_size)

    password = getpass("V2 Neo4j password: ")

    with GraphDatabase.driver(
        URI,
        auth=("neo4j", password),
    ) as driver:

        driver.verify_connectivity()

        with driver.session(
            database="neo4j",
            default_access_mode=READ_ACCESS,
        ) as session:

            # Yalnızca Neo4j feature extraction süresini ölçtük.
            started = perf_counter()
            completed = 0

            for start in range(0, len(tx_ids), args.batch_size):
                batch = tx_ids[start:start + args.batch_size]

                features = extract_receiver_history_batch(
                    session,
                    batch,
                )

                completed += len(features)

                if completed % 2_000 == 0 or completed == len(tx_ids):
                    print(
                        f"Feature extraction: {completed:,}/{len(tx_ids):,}",
                        flush=True,
                    )

            elapsed = perf_counter() - started

    rate = completed / elapsed
    projected_hours = (
        TOTAL_TRANSACTIONS / rate / 3600
    )

    print(f"\nExtracted: {completed:,}")
    print(f"Elapsed: {elapsed:.2f} seconds")
    print(f"Rate: {rate:.1f} tx/s")
    print(f"Projected full extraction: {projected_hours:.2f} hours")
    print(f"Predefined limit: {MAX_HOURS:.1f} hours")

    if projected_hours > MAX_HOURS:
        print("Preliminary NO-GO: alternatif yöntemi inceleyeceğiz.")
    else:
        print("Preliminary GO: daha büyük ve hot-account testi gerekli.")

    print("Benchmark completed!")


if __name__ == "__main__":
    main()
