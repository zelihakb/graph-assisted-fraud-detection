
import numpy as np
import pandas as pd

from src.graph.features import FEATURE_COLUMNS


# Daha önce dondurduğumuz zaman sınırlarını değiştirmedik.
SPLIT_RANGES = {
    "train": (1, 322),
    "validation": (323, 376),
    "test": (377, 743),
}

EXPECTED_COUNTS = {
    "train": {"rows": 1_938_484, "fraud": 3_633},
    "validation": {"rows": 415_628, "fraud": 560},
    "test": {"rows": 416_297, "fraud": 4_020},
}

RAW_COLUMNS = [
    "tx_id", "time_step", "tx_type",
    "amount", "label_fraud",
]

MODEL_COLUMNS = [
    "tx_id", "time_step", "tx_type", "amount",
    *FEATURE_COLUMNS, "label_fraud",
]

MODEL_INPUTS = [
    "amount", "tx_type", *FEATURE_COLUMNS,
]


def assemble_model_chunk(raw_chunk, graph_features):
    # İşlemleri ve graph özelliklerini kalıcı tx_id ile eşleştirdik.
    expected_graph_columns = ["tx_id", *FEATURE_COLUMNS]

    if list(graph_features.columns) != expected_graph_columns:
        raise ValueError("Graph feature kolonları uyuşmuyor.")

    if (
        raw_chunk["tx_id"].duplicated().any()
        or graph_features["tx_id"].duplicated().any()
    ):
        raise ValueError("Tekrarlanan transaction ID bulundu.")

    if len(raw_chunk) != len(graph_features):
        raise ValueError("İşlem ve feature satır sayıları farklı.")

    result = raw_chunk[RAW_COLUMNS].merge(
        graph_features,
        on="tx_id",
        how="left",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )

    if not (result["_merge"] == "both").all():
        raise ValueError("Eşleşmeyen transaction bulundu.")

    if result["tx_id"].tolist() != raw_chunk["tx_id"].tolist():
        raise ValueError("İşlem sırası değişti.")

    result = result.drop(columns="_merge")

    numeric_columns = [
        "time_step", "amount", "label_fraud",
        *FEATURE_COLUMNS,
    ]

    # Sayısal özelliklerde eksik ve sonsuz değerleri reddettik.
    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column], errors="raise"
        )

        if not np.isfinite(
            result[column].to_numpy(dtype=float)
        ).all():
            raise ValueError(f"Geçersiz sayısal değer: {column}")

    if not result["label_fraud"].isin([0, 1]).all():
        raise ValueError("Fraud etiketi ikili değil.")

    if not result["time_step"].between(1, 743).all():
        raise ValueError("Zaman adımı beklenen aralıkta değil.")

    if not result["tx_type"].isin(
        ["CASH_OUT", "TRANSFER"]
    ).all():
        raise ValueError("Kapsam dışı işlem türü bulundu.")

    if result["amount"].lt(0).any():
        raise ValueError("Negatif işlem tutarı bulundu.")

    for column in FEATURE_COLUMNS:
        if result[column].lt(0).any():
            raise ValueError(f"Negatif graph feature: {column}")

    if (
        result["receiver_prior_7d_tx_count"]
        > result["receiver_prior_tx_count"]
    ).any():
        raise ValueError("7 günlük işlem sayısı geçmiş toplamını aşıyor.")

    # Kimlik ve zaman alanlarını koruduk; model girdilerini ayrıca tanımladık.
    return result[MODEL_COLUMNS]


def partition_model_chunk(frame):
    # Fraud etiketine göre değil, yalnızca dondurulmuş zamana göre böldük.
    return {
        split: frame.loc[
            frame["time_step"].between(start, end)
        ].copy()
        for split, (start, end) in SPLIT_RANGES.items()
    }
