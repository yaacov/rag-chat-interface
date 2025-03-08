import sqlite3
import json


def setup_sqlite_db(db_path):
    """Set up SQLite database for query logging."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table for storing queries and responses
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS query_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        original_query TEXT,
        augmented_query TEXT,
        response TEXT,
        sources TEXT,
        execution_time_ms INTEGER
    )
    """
    )

    conn.commit()
    return conn


def log_query(conn, question, augmented_query, response, sources, execution_time_ms):
    """Log a query and its response to the SQLite database."""
    if conn is None:
        return

    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO query_logs (original_query, augmented_query, response, sources, execution_time_ms) VALUES (?, ?, ?, ?, ?)",
        (question, augmented_query, response, json.dumps(sources), execution_time_ms),
    )
    conn.commit()


def get_query_history(conn, limit=10, offset=0):
    """Retrieve query history from SQLite database."""
    if conn is None:
        return []

    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, timestamp, original_query, response, sources, execution_time_ms FROM query_logs ORDER BY timestamp DESC LIMIT ? OFFSET ?",
        (limit, offset),
    )

    history = []
    for row in cursor.fetchall():
        history.append(
            {
                "id": row[0],
                "timestamp": row[1],
                "query": row[2],
                "response": row[3],
                "sources": json.loads(row[4]) if row[4] else [],
                "execution_time_ms": row[5],
            }
        )

    return history
