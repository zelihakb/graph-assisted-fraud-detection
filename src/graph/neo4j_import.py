from src.graph.tables import build_graph_tables


QUERIES = {
    "accounts": """
        UNWIND $rows AS row
        MERGE (a:Account {account_id: row.account_id})
    """,

    "transactions": """
        UNWIND $rows AS row
        MERGE (t:Transaction {tx_id: row.tx_id})
        SET t += row
    """,

    "sent_relationships": """
        UNWIND $rows AS row
        MATCH (a:Account {account_id: row.from_account_id})
        MATCH (t:Transaction {tx_id: row.to_tx_id})
        MERGE (a)-[:SENT]->(t)
    """,

    "to_relationships": """
        UNWIND $rows AS row
        MATCH (t:Transaction {tx_id: row.from_tx_id})
        MATCH (a:Account {account_id: row.to_account_id})
        MERGE (t)-[:TO]->(a)
    """,
}


def _write_graph_tables(tx, tables):
    # Dört graph tablosunu aynı veritabanı işlemi içinde aktardık.
    for table_name, query in QUERIES.items():
        rows = tables[table_name].to_dict("records")

        if rows:
            tx.run(query, rows=rows).consume()


def import_graph_chunk(session, chunk, batch_size=1000):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    imported_count = 0

    # Büyük veri parçalarını daha küçük Neo4j batch'lerine ayırdık.
    for start in range(0, len(chunk), batch_size):
        batch = chunk.iloc[start:start + batch_size]

        # Mevcut graph tablosu oluşturma fonksiyonumuzu kullandık.
        tables = build_graph_tables(batch)

        # Her batch'in dört tablosunu tek transaction içinde işledik.
        session.execute_write(_write_graph_tables, tables)

        imported_count += len(batch)

    return imported_count