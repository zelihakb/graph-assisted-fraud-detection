
import random
from getpass import getpass
from math import isclose
from pathlib import Path

from neo4j import GraphDatabase, READ_ACCESS

from src.data.full_history import iter_scoped_paysim


CSV_PATH = Path("data/paysim.csv")
URI = "bolt://127.0.0.1:17687"

EXPECTED_TOTAL = 2_770_409
EXPECTED_FRAUD = 8_213


def collect_source_samples():
    # CSV'yi baştan okuyup sayıları ve karşılaştırma örneklerini hazırladık.
    rng = random.Random(42)

    positions = sorted(
        {0, EXPECTED_TOTAL - 1}
        | set(rng.sample(range(EXPECTED_TOTAL), 150))
    )

    selected = []
    fraud_samples = []
    total = 0
    fraud_total = 0
    index = 0

    for chunk in iter_scoped_paysim(CSV_PATH):
        chunk_end = total + len(chunk)

        while index < len(positions) and positions[index] < chunk_end:
            local_index = positions[index] - total
            selected.append(chunk.iloc[local_index].to_dict())
            index += 1

        # Fraud etiketlerini ayrıca kontrol etmek için örnekledik.
        for row in chunk.loc[chunk["label_fraud"] == 1].to_dict("records"):
            fraud_total += 1

            if len(fraud_samples) < 10:
                fraud_samples.append(row)
            else:
                position = rng.randrange(fraud_total)
                if position < 10:
                    fraud_samples[position] = row

        total = chunk_end

    if total != EXPECTED_TOTAL or fraud_total != EXPECTED_FRAUD:
        raise AssertionError(
            f"CSV beklenen sayılarla uyuşmuyor: "
            f"transactions={total}, fraud={fraud_total}"
        )

    samples = {
        row["tx_id"]: row
        for row in selected + fraud_samples
    }

    return samples


def verify_counts(session):
    # Kaynak CSV sayılarıyla Neo4j kayıt sayılarını karşılaştırdık.
    queries = {
        "transactions": "MATCH (t:Transaction) RETURN count(t) AS n",
        "sent": "MATCH ()-[r:SENT]->() RETURN count(r) AS n",
        "to": "MATCH ()-[r:TO]->() RETURN count(r) AS n",
        "fraud": (
            "MATCH (t:Transaction) "
            "WHERE t.label_fraud = 1 RETURN count(t) AS n"
        ),
    }

    counts = {
        name: session.run(query).single()["n"]
        for name, query in queries.items()
    }

    expected = {
        "transactions": EXPECTED_TOTAL,
        "sent": EXPECTED_TOTAL,
        "to": EXPECTED_TOTAL,
        "fraud": EXPECTED_FRAUD,
    }

    print("Neo4j counts:", counts, flush=True)

    if counts != expected:
        raise AssertionError(
            f"Graph sayıları uyuşmuyor. Beklenen: {expected}"
        )


def verify_samples(session, samples):
    # Gerçek CSV alanlarını ve ilişki uçlarını tek tek doğruladık.
    query = """
    UNWIND $ids AS id
    OPTIONAL MATCH (t:Transaction {tx_id: id})
    OPTIONAL MATCH (sender:Account)-[s:SENT]->(t)
    OPTIONAL MATCH (t)-[r:TO]->(receiver:Account)

    RETURN
        id,
        t.tx_id AS found_id,
        t.time_step AS time_step,
        t.tx_type AS tx_type,
        t.amount AS amount,
        t.src_id AS src_id,
        t.dst_id AS dst_id,
        t.label_fraud AS label_fraud,
        collect(DISTINCT sender.account_id) AS senders,
        collect(DISTINCT receiver.account_id) AS receivers,
        count(DISTINCT s) AS sent_edges,
        count(DISTINCT r) AS to_edges
    """

    results = {
        record["id"]: record.data()
        for record in session.run(query, ids=list(samples))
    }

    if set(results) != set(samples):
        raise AssertionError("Örnek işlem kimlikleri uyuşmuyor.")

    for tx_id, source in samples.items():
        actual = results[tx_id]

        for field in (
            "time_step",
            "tx_type",
            "src_id",
            "dst_id",
            "label_fraud",
        ):
            if actual[field] != source[field]:
                raise AssertionError(
                    f"{tx_id}: {field} uyuşmuyor."
                )

        if actual["found_id"] != tx_id:
            raise AssertionError(f"{tx_id}: işlem bulunamadı.")

        if actual["amount"] is None or not isclose(
            float(actual["amount"]),
            float(source["amount"]),
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise AssertionError(f"{tx_id}: amount uyuşmuyor.")

        if (
            actual["senders"] != [source["src_id"]]
            or actual["receivers"] != [source["dst_id"]]
            or actual["sent_edges"] != 1
            or actual["to_edges"] != 1
        ):
            raise AssertionError(
                f"{tx_id}: graph ilişkileri uyuşmuyor."
            )

    print(f"CSV spot-check: {len(samples)} işlem doğrulandı.", flush=True)


def verify_structure(session):
    # Tüm işlem düğümlerinin ilişki bütünlüğünü kontrol ettik.
    checks = {
        "invalid_sent": """
            MATCH (t:Transaction)
            WHERE COUNT { (t)<-[:SENT]-(:Account) } <> 1
            RETURN count(t) AS n
        """,
        "invalid_to": """
            MATCH (t:Transaction)
            WHERE COUNT { (t)-[:TO]->(:Account) } <> 1
            RETURN count(t) AS n
        """,
        "orphan_accounts": """
            MATCH (a:Account)
            WHERE NOT (a)--()
            RETURN count(a) AS n
        """,
    }

    for name, query in checks.items():
        print(f"Kontrol ediliyor: {name}", flush=True)

        invalid = session.run(query).single()["n"]
        print(f"{name}: {invalid}", flush=True)

        if invalid != 0:
            raise AssertionError(
                f"{name} kontrolü başarısız: {invalid}"
            )


def main():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(CSV_PATH)

    print("CSV bağımsız olarak okunuyor...", flush=True)
    samples = collect_source_samples()
    print(f"CSV örnekleri hazır: {len(samples)}", flush=True)

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

            verify_counts(session)
            verify_samples(session, samples)
            verify_structure(session)

    print("\nFULL-HISTORY INTEGRITY AUDIT PASSED!")


if __name__ == "__main__":
    main()
