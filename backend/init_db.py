import sys
import os

# Ensure backend directory is in python search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import get_db_connection


CREATE_USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(255) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    embedding REAL[]
);

ALTER TABLE users ADD COLUMN IF NOT EXISTS embedding REAL[];
ALTER TABLE users ADD COLUMN IF NOT EXISTS full_name VARCHAR(255);
ALTER TABLE users ADD COLUMN IF NOT EXISTS email VARCHAR(255);
ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(100);
ALTER TABLE users ADD COLUMN IF NOT EXISTS phone_number VARCHAR(50);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'check_embedding_dim'
    ) THEN
        ALTER TABLE users ADD CONSTRAINT check_embedding_dim 
        CHECK (embedding IS NULL OR (array_ndims(embedding) = 1 AND array_length(embedding, 1) = 512));
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'uq_users_email'
    ) THEN
        ALTER TABLE users ADD CONSTRAINT uq_users_email UNIQUE (email);
    END IF;
END $$;
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

