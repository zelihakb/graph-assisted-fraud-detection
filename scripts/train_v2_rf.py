
import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from scripts.train_v2_logreg import (
    GRAPH_INPUTS,
    TABULAR_INPUTS,
    TARGET,
    load_split,
    select_threshold,
)


OUTPUT_DIR = Path(".local/v2_models")

# İlk Random Forest deneyinin ayarlarını sonuçları görmeden sabitledik.
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 14,
    "min_samples_leaf": 20,
    "class_weight": "balanced_subsample",
    "max_features": "sqrt",
    "n_jobs": 2,
    "random_state": 42,
}



def build_rf(inputs):
    numeric = [
        column for column in inputs
        if column != "tx_type"
    ]

    # İşlem türü varsa kategorik dönüşümü ekledik.
    transformers = [
        ("numeric", "passthrough", numeric),
    ]

    if "tx_type" in inputs:
        transformers.insert(
            0,
            (
                "transaction_type",
                OneHotEncoder(handle_unknown="ignore"),
                ["tx_type"],
            ),
        )

    preprocessing = ColumnTransformer(
        transformers=transformers,
        sparse_threshold=0,
    )

    return Pipeline([
        ("preprocess", preprocessing),
        ("model", RandomForestClassifier(**RF_PARAMS)),
    ])


def fit_model(name, train, inputs, validation=None):
    print(f"\nTraining Random Forest: {name}", flush=True)

    pipeline = build_rf(inputs)

    started = perf_counter()

    pipeline.fit(
        train[inputs],
        train[TARGET],
    )

    elapsed = perf_counter() - started
    print(f"Training time: {elapsed:.2f} seconds", flush=True)

    # Pilot aşamasında validation değerlendirmesi yapmadık.
    if validation is None:
        return

    y_true = validation[TARGET].to_numpy()
    probabilities = pipeline.predict_proba(
        validation[inputs]
    )[:, 1]

    threshold = select_threshold(y_true, probabilities)
    predictions = (probabilities >= threshold).astype(int)

    tp = int(((predictions == 1) & (y_true == 1)).sum())
    fp = int(((predictions == 1) & (y_true == 0)).sum())

    metrics = {
        "model": f"rf_{name}",
        "training_seconds": round(elapsed, 2),
        "threshold": threshold,
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "true_positives": tp,
        "false_positives": fp,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        pipeline,
        OUTPUT_DIR / f"rf_{name}.joblib",
    )

    (OUTPUT_DIR / f"rf_{name}_validation.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    print("Training data yükleniyor...", flush=True)
    train = load_split("train")

    if args.pilot:
        # Eğitim süresini görmek için sabit bir alt örneklem kullandık.
        train = train.sample(
            n=200_000,
            random_state=42,
        )
        validation = None
        print("PILOT MODE: 200.000 train işlemi", flush=True)
    else:
        validation = load_split("validation")
        print(f"Full train: {len(train):,}", flush=True)
        print(f"Validation: {len(validation):,}", flush=True)

    fit_model(
        "tabular",
        train,
        TABULAR_INPUTS,
        validation,
    )

    fit_model(
        "graph",
        train,
        GRAPH_INPUTS,
        validation,
    )

    if args.pilot:
        print("\nRF PILOT COMPLETED!")
    else:
        print("\nRF VALIDATION COMPLETED!")
        print("Test split was not accessed.")


if __name__ == "__main__":
    main()
