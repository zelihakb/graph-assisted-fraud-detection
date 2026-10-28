
import argparse
from getpass import getpass
from math import isfinite
from pathlib import Path
from time import perf_counter

import pandas as pd
from neo4j import GraphDatabase, READ_ACCESS

from src.data.full_history import iter_scoped_paysim
from src.graph.features import (
    FEATURE_COLUMNS,
    extract_receiver_history_batch,
)


CSV_PATH = Path("data/paysim.csv")
OUTPUT_DIR = Path(".local/v2_graph_features")
URI = "bolt://127.0.0.1:17687"

EXPECTED_TOTAL = 2_770_409
CHUNK_SIZE = 50_000
BATCH_SIZE = 200

OUTPUT_COLUMNS = ["tx_id", *FEATURE_COLUMNS]


def validate_saved_part(path, expected_ids):
    # Kaydedilmiş parçanın kimliklerini ve kolonlarını doğruladık.
    saved = pd.read_csv(path)

    if list(saved.columns) != OUTPUT_COLUMNS:
        raise RuntimeError(f"Geçersiz kolonlar: {path}")

    if saved["tx_id"].tolist() != expected_ids:
        raise RuntimeError(f"İşlem kimlikleri uyuşmuyor: {path}")

    if saved.isna().any().any():
        raise RuntimeError(f"Eksik feature bulundu: {path}")


def extract_part(session, tx_ids):
    # Neo4j'den özellikleri küçük batch'lerle hesapladık.
    records = []

    for start in range(0, len(tx_ids), BATCH_SIZE):
        batch = tx_ids[start:start + BATCH_SIZE]

        records.extend(
            extract_receiver_history_batch(session, batch)
        )

    features = pd.DataFrame(records)

    # Neo4j sonuçlarını kaynak CSV'nin işlem sırasına eşitledik.
    features = (
        features
        .set_index("tx_id")
        .loc[tx_ids]
        .reset_index()
    )

    features = features[OUTPUT_COLUMNS]

    if features.isna().any().any():
        raise RuntimeError("Feature extraction eksik değer üretti.")

    for column in FEATURE_COLUMNS:
        if not features[column].map(
            lambda value: isfinite(float(value))
        ).all():
            raise RuntimeError(f"Geçersiz feature: {column}")

    return features


def save_part(features, output_path):
    # Dosyayı tamamlanmadan geçerli çıktı olarak işaretlemedik.
    temporary_path = output_path.with_suffix(".tmp")

    features.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit-parts",
        type=int,
        default=None,
        help="Yalnızca ilk N parçayı işler.",
    )

    args = parser.parse_args()

    if args.limit_parts is not None and args.limit_parts <= 0:
        parser.error("--limit-parts pozitif olmalı.")

    if not CSV_PATH.is_file():
        raise FileNotFoundError(CSV_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    password = getpass("V2 Neo4j password: ")

    total = 0
    newly_extracted = 0
    reused = 0
    expected_paths = set()

    started = perf_counter()

    with GraphDatabase.driver(
        URI,
        auth=("neo4j", password),
    ) as driver:

        driver.verify_connectivity()

        with driver.session(
            database="neo4j",
            default_access_mode=READ_ACCESS,
        ) as session:

            for part_index, chunk in enumerate(
                iter_scoped_paysim(
                    CSV_PATH,
                    chunk_size=CHUNK_SIZE,
                )
            ):
                if (
                    args.limit_parts is not None
                    and part_index >= args.limit_parts
                ):
                    break

                tx_ids = chunk["tx_id"].tolist()

                if len(tx_ids) != len(set(tx_ids)):
                    raise RuntimeError("Tekrarlanan transaction ID.")

                output_path = (
                    OUTPUT_DIR / f"part_{part_index:04d}.csv"
                )

                expected_paths.add(output_path)

                if output_path.exists():
                    # Önceden tamamlanan parçaları yeniden hesaplamadık.
                    validate_saved_part(output_path, tx_ids)
                    reused += len(tx_ids)

                    print(
                        f"SKIP part {part_index:04d}: "
                        f"{len(tx_ids):,} işlem",
                        flush=True,
                    )

                else:
                    features = extract_part(session, tx_ids)

                    if len(features) != len(tx_ids):
                        raise RuntimeError(
                            "Eksik feature extraction."
                        )

                    save_part(features, output_path)

                    # Kaydedilen çıktıyı yeniden okuyup kontrol ettik.
                    validate_saved_part(output_path, tx_ids)

                    newly_extracted += len(tx_ids)

                    print(
                        f"SAVED part {part_index:04d}: "
                        f"{len(tx_ids):,} işlem",
                        flush=True,
                    )

                total += len(tx_ids)

                elapsed = perf_counter() - started

                print(
                    f"Toplam tamamlanan: {total:,} "
                    f"| Süre: {elapsed:.1f} sn",
                    flush=True,
                )

    if args.limit_parts is None:
        if total != EXPECTED_TOTAL:
            raise AssertionError(
                f"Eksik feature: {total:,}/{EXPECTED_TOTAL:,}"
            )

        extra_paths = (
            set(OUTPUT_DIR.glob("part_*.csv"))
            - expected_paths
        )

        if extra_paths:
            raise RuntimeError(
                f"Beklenmeyen çıktı dosyaları: {extra_paths}"
            )

    elapsed = perf_counter() - started

    print(f"\nToplam satır: {total:,}")
    print(f"Yeni çıkarılan: {newly_extracted:,}")
    print(f"Önceden kullanılan: {reused:,}")
    print(f"Süre: {elapsed:.2f} sn")
    print(f"Çıktı klasörü: {OUTPUT_DIR}")

    if args.limit_parts is not None:
        print("PARTIAL FEATURE EXPORT PASSED!")
    else:
        print("FULL FEATURE EXTRACTION PASSED!")


if __name__ == "__main__":
    main()
