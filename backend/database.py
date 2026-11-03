import os
import psycopg2
from psycopg2.extensions import connection as PGConnection
from dotenv import load_dotenv

# Load environment variables from .env located in the backend directory
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=dotenv_path, override=True)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "biometricDB")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")


def get_db_connection() -> PGConnection:
    """
    Establishes and returns a connection to the PostgreSQL database
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

