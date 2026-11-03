import os
import sys
import psycopg2
from psycopg2 import errors

# Ensure backend directory is in python search path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from database import get_pooled_connection
from generate_embedding import generate_embedding, extract_face_embedding
import numpy as np


class UserAlreadyExistsError(Exception):
    """Raised when an attempt is made to register a user_id that already exists in the database."""
    pass


class InvalidUserError(Exception):
    """Raised when user data, ID, or image is invalid."""
    pass


def register_user_with_embedding(
    user_id: str,
    image_path: str = None,
    face_bgr: np.ndarray = None,
    save_json: bool = False
) -> dict:
    """
    Registers a new user in the PostgreSQL database with their 512-dimensional FaceNet embedding.

    Workflow:
    1. Validates user ID.
    2. Generates the 512-dimensional FaceNet embedding from either the in-memory cropped
       face image (face_bgr) or file system path (image_path).
    3. Validates embedding length and values.
    4. Connects to PostgreSQL via the connection pool and inserts the user record
       using parameterized queries.
    5. Confirms successful transaction commit.
    6. Synchronizes embedding to FAISS index ONLY post-commit.

    Parameters:
        user_id (str): Unique user identifier.
        image_path (str, optional): File system path to cropped face image.
        face_bgr (np.ndarray, optional): In-memory cropped face image in BGR format.
        save_json (bool): Whether to create a permanent JSON backup file (default False).

    Returns:
        dict: Registration confirmation containing user_id, created_at, and embedding dimension.

    Raises:
        InvalidUserError: If user_id is empty or image is invalid.
        UserAlreadyExistsError: If user_id already exists in the database (preventing silent overwrite).
        ValueError: If embedding dimensions are not exactly 512.
        Exception: If database connection or query execution fails.
    """
    if not user_id or not str(user_id).strip():
        raise InvalidUserError("User ID must be a non-empty string.")
    clean_user_id = str(user_id).strip()

    # Step 1: Generate FaceNet embedding via in-memory image or file path
    if face_bgr is not None:
        try:
            embedding_arr = extract_face_embedding(face_bgr)
            embedding_vector = [float(x) for x in embedding_arr]
            embedding_dim = len(embedding_vector)
        except Exception as emb_err:
            raise InvalidUserError(f"Failed to generate embedding: {emb_err}") from emb_err
    else:
        try:
            embedding_result = generate_embedding(
                user_id=clean_user_id,
                image_path=image_path,
                save_json=save_json
            )
        except FileNotFoundError as fnf_err:
            raise InvalidUserError(f"Cropped face image not found: {fnf_err}") from fnf_err
        except ValueError as val_err:
            raise InvalidUserError(f"Failed to generate embedding: {val_err}") from val_err

        embedding_vector = embedding_result["embedding"]
        embedding_dim = embedding_result["embedding_dimension"]

    # Step 2: Validate embedding dimension before database insertion
    if embedding_dim != 512 or len(embedding_vector) != 512:
        raise ValueError(
            f"Pre-insertion validation failed: Expected 512 dimensions, got {len(embedding_vector)}."
        )

    # Step 3: Insert user and embedding into PostgreSQL with parameterized SQL query
    insert_sql = """
    INSERT INTO users (user_id, embedding)
    VALUES (%s, %s)
    RETURNING user_id, created_at, array_length(embedding, 1);
    """

    with get_pooled_connection() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(insert_sql, (clean_user_id, embedding_vector))
                row = cur.fetchone()
            conn.commit()

            confirmed_uid = row[0]
            confirmed_created_at = row[1]
            confirmed_dim = row[2]

            print(f"[DB SUCCESS] Registered user '{confirmed_uid}' with {confirmed_dim}-D embedding.")

            # Step 4: Safe FAISS synchronization - Add to FAISS ONLY after successful DB commit
            try:
                from faiss_service import get_faiss_manager
                faiss_mgr = get_faiss_manager()
                faiss_mgr.add_user(confirmed_uid, embedding_vector)
            except Exception as faiss_err:
                print(f"[FAISS SYNC WARNING] Post-commit FAISS sync for '{confirmed_uid}': {faiss_err}. Initiating index reconciliation...")
                try:
                    faiss_mgr.initialize_from_database()
                except Exception as recon_err:
                    print(f"[FAISS RECONCILIATION ERROR] Index reconciliation failed: {recon_err}")

            return {
                "status": "success",
                "user_id": confirmed_uid,
                "created_at": confirmed_created_at.isoformat() if hasattr(confirmed_created_at, "isoformat") else str(confirmed_created_at),
                "embedding_dimension": confirmed_dim,
                "message": "User registered and embedding stored successfully in PostgreSQL."
            }

        except errors.UniqueViolation as uv_err:
            conn.rollback()
            raise UserAlreadyExistsError(
                f"User ID '{clean_user_id}' already exists in database. Registration rejected."
            ) from uv_err
        except Exception as e:
            conn.rollback()
            print(f"[DB ERROR] Failed to store user '{clean_user_id}': {e}")
            raise e


def get_user_embedding(user_id: str) -> dict | None:
    """
    Retrieves a user's record and face embedding from PostgreSQL by user ID.

    Parameters:
        user_id (str): Unique user identifier.

    Returns:
        dict | None: User data including 512-D embedding list if found, otherwise None.
    """
    if not user_id or not str(user_id).strip():
        return None
    clean_user_id = str(user_id).strip()

    select_sql = """
    SELECT user_id, created_at, embedding, array_length(embedding, 1)
    FROM users
    WHERE user_id = %s;
    """

    with get_pooled_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(select_sql, (clean_user_id,))
            row = cur.fetchone()

    if not row:
        return None

    user_id_val, created_at_val, raw_embedding, dim = row

    embedding_list = [float(x) for x in raw_embedding] if raw_embedding is not None else None

    return {
        "user_id": user_id_val,
        "created_at": created_at_val.isoformat() if hasattr(created_at_val, "isoformat") else str(created_at_val),
        "embedding": embedding_list,
        "embedding_dimension": dim
    }


def delete_user(user_id: str) -> bool:
    """
    Safely removes a test user from PostgreSQL by user ID and synchronizes with FAISS.
    Used for test cleanup and idempotency.
    """
    if not user_id:
        return False
    clean_uid = str(user_id).strip()
    with get_pooled_connection() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM users WHERE user_id = %s;", (clean_uid,))
            conn.commit()

            # Keep FAISS in sync on delete
            try:
                from faiss_service import get_faiss_manager
                faiss_mgr = get_faiss_manager()
                if faiss_mgr.is_initialized():
                    faiss_mgr.remove_user(clean_uid)
            except Exception as faiss_err:
                print(f"[FAISS REMOVE WARNING] Post-delete FAISS sync for '{clean_uid}': {faiss_err}")

            return True
        except Exception:
            conn.rollback()
            return False

