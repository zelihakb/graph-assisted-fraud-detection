
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.full_history import iter_scoped_paysim
from src.data.v2_dataset import (
    EXPECTED_COUNTS,
    MODEL_COLUMNS,
    MODEL_INPUTS,
    SPLIT_RANGES,
    assemble_model_chunk,
    partition_model_chunk,
)
from src.graph.features import FEATURE_COLUMNS


CSV_PATH = Path("data/paysim.csv")
FEATURE_DIR = Path(".local/v2_graph_features")
OUTPUT_DIR = Path(".local/v2_model_data")


def verify_saved(path, expected):
    # Önceden tamamlanmış dosyaların kaynakla uyumunu kontrol ettik.
    saved = pd.read_csv(path)

    if saved.columns.tolist() != MODEL_COLUMNS:
        raise ValueError(f"Kolon uyuşmazlığı: {path}")

    if len(saved) != len(expected):
        raise ValueError(f"Satır sayısı uyuşmazlığı: {path}")

    for column in ("tx_id", "tx_type"):
        if saved[column].tolist() != expected[column].tolist():
            raise ValueError(f"{column} uyuşmazlığı: {path}")

    numeric = [
        "time_step", "amount", "label_fraud",
        *FEATURE_COLUMNS,
    ]

    for column in numeric:
        if not np.allclose(
            saved[column].to_numpy(dtype=float),
            expected[column].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-7,
            equal_nan=False,
        ):
            raise ValueError(f"{column} uyuşmazlığı: {path}")


def save_part(frame, path):
    # Tamamlanmamış parçaları geçerli çıktı olarak işaretlemedik.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")

    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(CSV_PATH)

    counts = {
        name: {"rows": 0, "fraud": 0}
        for name in SPLIT_RANGES
    }

    total = 0

    for part_index, raw_chunk in enumerate(
        iter_scoped_paysim(CSV_PATH, chunk_size=50_000)
    ):
        feature_path = (
            FEATURE_DIR / f"part_{part_index:04d}.csv"
        )

        if not feature_path.is_file():
            raise FileNotFoundError(feature_path)

        graph_features = pd.read_csv(feature_path)

        model_chunk = assemble_model_chunk(
            raw_chunk, graph_features
        )

        partitions = partition_model_chunk(model_chunk)

        for split, frame in partitions.items():
            if frame.empty:
                continue

            output_path = (
                OUTPUT_DIR
                / split
                / f"part_{part_index:04d}.csv"
            )

            if output_path.exists():
                verify_saved(output_path, frame)
                action = "SKIP"
            else:
                save_part(frame, output_path)
                verify_saved(output_path, frame)
                action = "SAVED"

            counts[split]["rows"] += len(frame)
            counts[split]["fraud"] += int(
                frame["label_fraud"].sum()
            )

            print(
                f"{action} {split}/part_{part_index:04d}: "
                f"{len(frame):,} satır",
                flush=True,
            )

        total += len(model_chunk)

    # Dondurduğumuz split dağılımını yeniden doğruladık.
    if total != 2_770_409:
        raise AssertionError(
            f"Toplam işlem uyuşmuyor: {total}"
        )

    if counts != EXPECTED_COUNTS:
        raise AssertionError(
            f"Frozen split uyuşmuyor.\n"
            f"Beklenen: {EXPECTED_COUNTS}\n"
            f"Gerçek: {counts}"
        )

    manifest = {
        "total_transactions": total,
        "split_ranges": SPLIT_RANGES,
        "counts": counts,
        "model_inputs": MODEL_INPUTS,
        "feature_columns": list(FEATURE_COLUMNS),
        "target": "label_fraud",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / "manifest.json"
    temp_manifest = OUTPUT_DIR / "manifest.tmp"

    temp_manifest.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    temp_manifest.replace(manifest_path)

    print("\nFROZEN TEMPORAL SPLITS VERIFIED!")

    for split, values in counts.items():
        prevalence = (
            100 * values["fraud"] / values["rows"]
        )
        print(
            f"{split}: {values['rows']:,} işlem, "
            f"{values['fraud']:,} fraud, "
            f"%{prevalence:.4f}"
        )

    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
