from __future__ import annotations
import os
import psycopg2
import psycopg2.extras
from typing import Any
 
 
# ============================================================
# Connection helper
# ============================================================
 
def get_connection(connection_string: str | None = None) -> psycopg2.extensions.connection:
    """
    Open a PostgreSQL connection.
    Uses PG_CONNECTION_STRING from .env if no argument is given.
    """
    conn_str = connection_string or os.getenv("PG_CONNECTION_STRING")
    if not conn_str:
        raise ValueError(
            "No PostgreSQL connection string found.\n"
            "Set PG_CONNECTION_STRING in your .env file.\n"
            "Example: postgresql://postgres:password@localhost:5432/sunglasses_store"
        )
    conn = psycopg2.connect(conn_str)
    # READ ONLY — prevents any accidental INSERT/UPDATE/DELETE
    conn.set_session(readonly=True, autocommit=True)
    return conn
 
 
# ============================================================
# reads from PostgreSQL tables
# ============================================================
 
def _fetch_table_info(conn: psycopg2.extensions.connection, table_name: str) -> dict:
    """
    Fetch column names, types, and a few sample rows from a table.
    Returns a dict with keys: columns, rows, row_count
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # Get column names and data types from PostgreSQL catalog
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
        """, (table_name,))
        columns = [{"name": row["column_name"], "type": row["data_type"]} for row in cur.fetchall()]
 
        # Get total row count
        cur.execute(f"SELECT COUNT(*) AS cnt FROM {table_name}")
        row_count = cur.fetchone()["cnt"]
 
        # Get up to 3 sample rows so the LLM sees real values
        cur.execute(f"SELECT * FROM {table_name} LIMIT 3")
        sample_rows = [dict(r) for r in cur.fetchall()]
 
    return {
        "columns": columns,
        "row_count": row_count,
        "sample_rows": sample_rows,
    }
 
 
def build_schema_for_table(conn: psycopg2.extensions.connection, table_name: str) -> str:
    """
    Build a text description of one table.
    Mirrors inv_utils.build_schema_for_table() but reads from PostgreSQL.
    """
    info = _fetch_table_info(conn, table_name)
 
    lines = [f"TABLE: {table_name}", "COLUMNS:"]
    for col in info["columns"]:
        lines.append(f"  - {col['name']}: {col['type']}")
 
    lines.append(f"ROWS: {info['row_count']}")
    lines.append(f"PREVIEW (first 3 rows):")
    for row in info["sample_rows"]:
        lines.append(f"  {dict(row)}")
 
    return "\n".join(lines)
 
 
def build_schema_block(conn: psycopg2.extensions.connection) -> str:
    """
    Build a combined schema description for both tables.
    This is injected into the LLM prompt exactly as in the notebook.
    """
    inv = build_schema_for_table(conn, "inventory")
    tx  = build_schema_for_table(conn, "transactions")
 
    notes = (
        "NOTES:\n"
        "- inventory.price is in USD (numeric).\n"
        "- inventory.quantity_in_stock > 0 means the item is available.\n"
        "- inventory.name is the style name (e.g., 'Classic', 'Moon', 'Aviator').\n"
        "- inventory.description contains keywords like 'round', 'wraparound', etc.\n"
        "- transactions.created_at is a PostgreSQL TIMESTAMP.\n"
        "- This is a READ-ONLY connection. Do NOT write INSERT/UPDATE/DELETE.\n"
    )
    return f"{inv}\n\n{tx}\n\n{notes}"
 
 
# ============================================================
# Quick test — run this file directly to verify your DB setup
# ============================================================
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
 
    print("Testing PostgreSQL connection...\n")
    try:
        conn = get_connection()
        schema = build_schema_block(conn)
        print(schema)
        conn.close()
        print("\n✅  Connection and schema fetch successful!")
    except Exception as e:
        print(f"\n❌  Error: {e}")
        print("\nMake sure:")
        print("  1. PostgreSQL is running (pgAdmin 4 shows green icon)")
        print("  2. You ran db_setup.sql in pgAdmin 4 Query Tool")
        print("  3. PG_CONNECTION_STRING in .env is correct")