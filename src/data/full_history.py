from pathlib import Path
from collections.abc import Iterator

import pandas as pd

from src.data.paysim import prepare_scoped_paysim


def iter_scoped_paysim(
    csv_path: str | Path,
    chunk_size: int = 50_000
) -> Iterator[pd.DataFrame]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    row_offset = 0

    # Ham PaySim verisini belleğe tamamen almadan parçalara ayırdık.
    for raw_chunk in pd.read_csv(csv_path, chunksize=chunk_size):

        # Her parçaya orijinal CSV satır konumlarını geri atadık.
        raw_chunk.index = pd.RangeIndex(
            start=row_offset,
            stop=row_offset + len(raw_chunk)
        )

        row_offset += len(raw_chunk)

        # Mevcut canonical dönüşüm fonksiyonumuzu yeniden kullandık.
        prepared = prepare_scoped_paysim(raw_chunk)

        # CASH_OUT veya TRANSFER içermeyen parçaları atladık.
        if not prepared.empty:
            yield prepared