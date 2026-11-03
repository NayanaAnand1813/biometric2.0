import sys
import os

# Ensure backend directory is in python search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import get_db_connection


CREATE_USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(255) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def init_database() -> None:
    """
    Initializes the PostgreSQL database schema by creating the `users` table
    if it does not already exist.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_USERS_TABLE_SQL)
        conn.commit()
        print("Table 'users' verified / created successfully in biometricDB.")
    except Exception as e:
        conn.rollback()
        print(f"Error creating 'users' table: {e}")
        raise e
    finally:
        conn.close()


if __name__ == "__main__":
    init_database()

