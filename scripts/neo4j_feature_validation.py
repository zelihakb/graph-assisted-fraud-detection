
from getpass import getpass
from math import isclose
from time import perf_counter

import pandas as pd
from neo4j import GraphDatabase, READ_ACCESS

from src.data.full_history import iter_scoped_paysim
from src.graph.features import (
    FEATURE_COLUMNS,
    extract_receiver_history_batch,
)
from scripts.neo4j_feature_benchmark import (
    URI,
    CSV_PATH,
    select_sample_ids,
)


def get_hot_ids(session):
    # En yoğun on alıcı hesabı veritabanından belirledik.
    accounts = session.run("""
        MATCH (t:Transaction)-[:TO]->(a:Account)
        WITH a, count(t) AS incoming
        ORDER BY incoming DESC, a.account_id
        LIMIT 10
        RETURN a.account_id AS account_id, incoming
    """).data()

    account_ids = [row["account_id"] for row in accounts]

    records = session.run("""
        UNWIND $ids AS account_id
        MATCH (a:Account {account_id: account_id})
              <-[:TO]-(t:Transaction)
        RETURN t.tx_id AS tx_id
        ORDER BY tx_id
    """, ids=account_ids)

    hot_ids = [record["tx_id"] for record in records]

    print("En yoğun hesap:", accounts[0], flush=True)
    print("Hot-account işlemleri:", len(hot_ids), flush=True)

    return hot_ids


def extract_batches(session, ids, batch_size=200):
    # Feature sorgusunu aynı batch yöntemiyle çalıştırdık.
    result = []

    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        result.extend(
            extract_receiver_history_batch(session, batch)
        )

    return result


def load_csv_reference(target_ids):
    # Örnek işlemleri orijinal CSV'den bağımsız olarak bulduk.
    targets = {}

    for chunk in iter_scoped_paysim(CSV_PATH):
        selected = chunk.loc[
            chunk["tx_id"].isin(target_ids),
            ["tx_id", "dst_id", "time_step"],
        ]

        for row in selected.to_dict("records"):
            targets[row["tx_id"]] = row

    if len(targets) != len(target_ids):
        raise AssertionError("CSV'de bazı örnek işlemler bulunamadı.")

    # Seçilen alıcıların TÜM geçmişini CSV'den topladık.
    receiver_ids = {
        row["dst_id"] for row in targets.values()
    }

    history_parts = []

    for chunk in iter_scoped_paysim(CSV_PATH):
        selected = chunk.loc[
            chunk["dst_id"].isin(receiver_ids),
            ["dst_id", "time_step", "amount"],
        ]

        if not selected.empty:
            history_parts.append(selected)

    history = pd.concat(history_parts, ignore_index=True)

    groups = {
        account_id: group
        for account_id, group in history.groupby("dst_id")
    }

    return targets, groups


def pandas_features(transaction, groups):
    # Neo4j sorgusunu kullanmadan geçmiş özelliklerini hesapladık.
    receiver = transaction["dst_id"]
    step = transaction["time_step"]

    history = groups[receiver]

    past = history.loc[
        history["time_step"] < step
    ]

    recent = past.loc[
        past["time_step"] >= step - 168
    ]

    prior_count = len(past)
    prior_total = float(past["amount"].sum())

    return {
        "receiver_prior_tx_count": prior_count,
        "receiver_prior_total_amount": prior_total,
        "receiver_prior_avg_amount": (
            prior_total / prior_count if prior_count else 0.0
        ),
        "receiver_prior_7d_tx_count": len(recent),
        "receiver_prior_7d_total_amount": float(
            recent["amount"].sum()
        ),
    }


def main():
    random_ids = select_sample_ids(100)

    password = getpass("V2 Neo4j password: ")

    with GraphDatabase.driver(
        URI, auth=("neo4j", password)
    ) as driver:

        driver.verify_connectivity()

        with driver.session(
            database="neo4j",
            default_access_mode=READ_ACCESS,
        ) as session:

            hot_ids = get_hot_ids(session)

            # Hot-account extraction süresini ayrı ölçtük.
            started = perf_counter()
            hot_results = extract_batches(session, hot_ids)
            elapsed = perf_counter() - started

            print(
                f"Hot extraction: {len(hot_results)} işlem "
                f"| {elapsed:.2f} saniye",
                flush=True,
            )

            # Rastgele ve yoğun hesap örneklerini birleştirdik.
            target_ids = list(
                dict.fromkeys(random_ids + hot_ids)
            )

            neo4j_rows = extract_batches(session, target_ids)

    print("Bağımsız CSV hesaplaması başlıyor...", flush=True)

    targets, groups = load_csv_reference(set(target_ids))

    actual = {
        row["tx_id"]: row
        for row in neo4j_rows
    }

    if len(actual) != len(target_ids):
        raise AssertionError("Neo4j sonuç sayısı uyuşmuyor.")

    checked = 0

    for tx_id in target_ids:
        expected = pandas_features(targets[tx_id], groups)
        found = actual[tx_id]

        for column in FEATURE_COLUMNS:
            if not isclose(
                float(found[column]),
                float(expected[column]),
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                raise AssertionError(
                    f"{tx_id} / {column}: "
                    f"Neo4j={found[column]}, "
                    f"pandas={expected[column]}"
                )

        checked += 1

    print(f"Neo4j-pandas eşleşmesi: {checked}/{checked}")
    print("FEATURE VALIDATION PASSED!")


if __name__ == "__main__":
    main()
