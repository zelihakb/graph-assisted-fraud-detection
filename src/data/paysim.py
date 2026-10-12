import pandas as pd

COLUMN_MAP = {
    "step": "time_step",
    "type": "tx_type",
    "nameOrig": "src_id",
    "nameDest": "dst_id",
    "oldbalanceOrg": "src_balance_before",
    "newbalanceOrig": "src_balance_after",
    "oldbalanceDest": "dst_balance_before",
    "newbalanceDest": "dst_balance_after",
    "isFraud": "label_fraud",
    "isFlaggedFraud": "label_flagged",
}


def prepare_scoped_paysim(raw_df: pd.DataFrame) -> pd.DataFrame:
    missing = set(COLUMN_MAP) - set(raw_df.columns)

    if missing:
        raise ValueError(f"Missing PaySim columns: {sorted(missing)}")

    if not isinstance(raw_df.index, pd.RangeIndex):
        raise ValueError("Expected original PaySim CSV row indices.")

    scoped = raw_df.loc[
        raw_df["type"].isin(["CASH_OUT", "TRANSFER"])
    ].copy()

    scoped = scoped.rename(columns=COLUMN_MAP)

    # Stable IDs based on original CSV row positions
    scoped.insert(
        0,
        "tx_id",
        [f"raw_{i + 1}" for i in scoped.index]
    )

    return scoped.reset_index(drop=True)