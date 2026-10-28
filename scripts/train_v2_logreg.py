
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

from src.data.v2_dataset import (
    EXPECTED_COUNTS,
    MODEL_INPUTS,
)


DATA_DIR = Path(".local/v2_model_data")
OUTPUT_DIR = Path(".local/v2_models")

TABULAR_INPUTS = ["amount", "tx_type"]
GRAPH_INPUTS = list(MODEL_INPUTS)

TARGET = "label_fraud"


def load_split(name):
    # Split dosyalarını yalnızca gerekli sütunlarla birleştirdik.
    paths = sorted((DATA_DIR / name).glob("part_*.csv"))

    if not paths:
        raise FileNotFoundError(f"Split bulunamadı: {name}")

    columns = list(dict.fromkeys([*GRAPH_INPUTS, TARGET]))

    frame = pd.concat(
        [
            pd.read_csv(path, usecols=columns)
            for path in paths
        ],
        ignore_index=True,
    )

    expected = EXPECTED_COUNTS[name]

    if len(frame) != expected["rows"]:
        raise AssertionError(f"{name}: satır sayısı uyuşmuyor.")

    if int(frame[TARGET].sum()) != expected["fraud"]:
        raise AssertionError(f"{name}: fraud sayısı uyuşmuyor.")

    if frame[columns].isna().any().any():
        raise ValueError(f"{name}: eksik değer bulundu.")

    return frame


def build_pipeline(feature_names):
    numeric = [
        name for name in feature_names
        if name != "tx_type"
    ]

    # Sayısal dağılımların çarpıklığını log dönüşümüyle azalttık.
    numeric_pipeline = Pipeline([
        (
            "log_transform",
            FunctionTransformer(np.log1p),
        ),
        ("scaler", StandardScaler()),
    ])

    # Dönüşümleri yalnızca eğitim verisinden öğrendik.
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        (
            "transaction_type",
            OneHotEncoder(handle_unknown="ignore"),
            ["tx_type"],
        ),
    ])

    return Pipeline([
        ("preprocess", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=500,
                solver="lbfgs",
                random_state=42,
            ),
        ),
    ])


def select_threshold(y_true, probabilities):
    # Validation F1 değerini maksimize eden eşiği seçtik.
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f1_values))

    return float(thresholds[best_index])


def train_and_evaluate(name, train, validation, inputs):
    print(f"\nTraining: {name}", flush=True)

    pipeline = build_pipeline(inputs)

    X_train = train[inputs]
    y_train = train[TARGET].to_numpy()

    X_val = validation[inputs]
    y_val = validation[TARGET].to_numpy()

    started = perf_counter()

    # Modeli yalnızca train ayrımına uyarladık.
    pipeline.fit(X_train, y_train)

    training_seconds = perf_counter() - started

    # Validation üzerinden tahminleri ve karar eşiğini hesapladık.
    probabilities = pipeline.predict_proba(X_val)[:, 1]

    threshold = select_threshold(y_val, probabilities)
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "model": name,
        "train_seconds": round(training_seconds, 2),
        "threshold": threshold,
        "average_precision": float(
            average_precision_score(y_val, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_val, probabilities)
        ),
        "precision": float(
            precision_score(y_val, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_val, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_val, predictions, zero_division=0)
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        pipeline,
        OUTPUT_DIR / f"logreg_{name}.joblib",
    )

    print(json.dumps(metrics, indent=2), flush=True)

    return metrics


def main():
    if not (DATA_DIR / "manifest.json").exists():
        raise FileNotFoundError("V2 model manifest bulunamadı.")

    print("Train ve validation yükleniyor...", flush=True)

    train = load_split("train")
    validation = load_split("validation")

    print(f"Train: {len(train):,}")
    print(f"Validation: {len(validation):,}")

    results = [
        train_and_evaluate(
            "tabular",
            train,
            validation,
            TABULAR_INPUTS,
        ),
        train_and_evaluate(
            "graph",
            train,
            validation,
            GRAPH_INPUTS,
        ),
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "logreg_validation.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nLOGISTIC REGRESSION VALIDATION COMPLETED!")
    print("Test split was not accessed.")


if __name__ == "__main__":
    main()
