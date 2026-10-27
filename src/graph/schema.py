CONSTRAINT_QUERIES = (
    """
    CREATE CONSTRAINT account_id_unique IF NOT EXISTS
    FOR (a:Account)
    REQUIRE a.account_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT transaction_id_unique IF NOT EXISTS
    FOR (t:Transaction)
    REQUIRE t.tx_id IS UNIQUE
    """,
)


def ensure_graph_constraints(session):
    # Hesap ve işlem kimliklerinin benzersizliğini sağlayan kuralları hazırladık.
    for query in CONSTRAINT_QUERIES:
        session.run(query).consume()