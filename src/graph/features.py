
FEATURE_COLUMNS = (
    "receiver_prior_tx_count",
    "receiver_prior_total_amount",
    "receiver_prior_avg_amount",
    "receiver_prior_7d_tx_count",
    "receiver_prior_7d_total_amount",
)

RECEIVER_HISTORY_QUERY = """
UNWIND $ids AS requested_id

MATCH (current:Transaction {tx_id: requested_id})
      -[:TO]->(receiver:Account)

OPTIONAL MATCH (past:Transaction)-[:TO]->(receiver)
WHERE past.time_step < current.time_step

WITH
    requested_id,
    current,
    count(past) AS prior_count,
    coalesce(sum(past.amount), 0.0) AS prior_total,
    sum(
        CASE
            WHEN past.time_step >= current.time_step - 168
            THEN 1 ELSE 0
        END
    ) AS count_7d,
    sum(
        CASE
            WHEN past.time_step >= current.time_step - 168
            THEN past.amount ELSE 0.0
        END
    ) AS total_7d

RETURN
    requested_id AS tx_id,
    prior_count AS receiver_prior_tx_count,
    prior_total AS receiver_prior_total_amount,
    CASE
        WHEN prior_count = 0 THEN 0.0
        ELSE prior_total / prior_count
    END AS receiver_prior_avg_amount,
    count_7d AS receiver_prior_7d_tx_count,
    total_7d AS receiver_prior_7d_total_amount
"""


def extract_receiver_history_batch(session, tx_ids):
    # İşlem kimliklerini tek sorguda işleyerek bağlantı maliyetini azalttık.
    if not tx_ids:
        return []

    records = [
        dict(record)
        for record in session.run(
            RECEIVER_HISTORY_QUERY,
            ids=tx_ids,
        )
    ]

    # Eksik veya tekrar eden sonuçları sessizce kabul etmedik.
    returned_ids = [row["tx_id"] for row in records]

    if len(records) != len(tx_ids) or set(returned_ids) != set(tx_ids):
        raise RuntimeError(
            "Feature sorgusu eksik veya tekrar eden işlem döndürdü."
        )

    return records
