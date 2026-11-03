import os
import psycopg2
from psycopg2 import pool
from psycopg2.extensions import connection as PGConnection
from dotenv import load_dotenv
from contextlib import contextmanager

# Load environment variables from .env located in the backend directory
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=dotenv_path, override=True)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "biometricDB")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")

_connection_pool: pool.ThreadedConnectionPool | None = None


def get_connection_pool(minconn: int = 1, maxconn: int = 20) -> pool.ThreadedConnectionPool:
    """
    Returns a thread-safe connection pool for concurrent database operations.
    Initializes the pool if not already initialized.
    """
    global _connection_pool
    if _connection_pool is None or _connection_pool.closed:
        if not DB_PASSWORD:
            raise ValueError(
                "DB_PASSWORD environment variable is not set. "
                "Please create a .env file in the backend directory with DB_PASSWORD defined."
            )
        _connection_pool = pool.ThreadedConnectionPool(
            minconn,
            maxconn,
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
    return _connection_pool


@contextmanager
def get_pooled_connection():
    """
    Context manager that checks out a database connection from the connection pool
    and reliably returns it to the pool when the block exits.
    """
    pool_instance = get_connection_pool()
    conn = pool_instance.getconn()
    try:
        yield conn
    finally:
        pool_instance.putconn(conn)


def get_db_connection() -> PGConnection:
    """
    Establishes and returns an independent connection to the PostgreSQL database
    using credentials loaded from environment variables.
    
    Raises:
        ValueError: If DB_PASSWORD environment variable is missing.
        psycopg2.OperationalError: If connection to PostgreSQL fails.
    """
    if not DB_PASSWORD:
        raise ValueError(
            "DB_PASSWORD environment variable is not set. "
            "Please create a .env file in the backend directory with DB_PASSWORD defined."
        )

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    return conn


