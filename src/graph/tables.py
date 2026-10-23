import pandas as pd


REQUIRED_COLUMNS = {
    "tx_id",
    "src_id",
    "dst_id",
    "time_step",
    "tx_type",
    "amount",
}


def build_graph_tables(chunk: pd.DataFrame) -> dict[str, pd.DataFrame]:

    # Graph oluşturmak için gerekli kolonları doğruladık.
    missing = REQUIRED_COLUMNS - set(chunk.columns)

    if missing:
        raise ValueError(f"Missing graph columns: {sorted(missing)}")

    if chunk["tx_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs in chunk.")

    # Gönderen ve alıcı hesapları tek bir Account tablosunda birleştirdik.
    accounts = pd.DataFrame({
        "account_id": pd.unique(
            pd.concat(
                [chunk["src_id"], chunk["dst_id"]],
                ignore_index=True
            )
        )
    })

    # İşlem bilgilerini ayrı Transaction tablosunda koruduk.
    transactions = chunk.copy()

    # Gönderen hesap ile işlem arasındaki SENT ilişkisini oluşturduk.
    sent = chunk[["src_id", "tx_id"]].rename(columns={
        "src_id": "from_account_id",
        "tx_id": "to_tx_id",
    }).copy()

    # İşlem ile alıcı hesap arasındaki TO ilişkisini oluşturduk.
    to = chunk[["tx_id", "dst_id"]].rename(columns={
        "tx_id": "from_tx_id",
        "dst_id": "to_account_id",
    }).copy()

    return {
        "accounts": accounts,
        "transactions": transactions,
        "sent_relationships": sent,
        "to_relationships": to,
    }