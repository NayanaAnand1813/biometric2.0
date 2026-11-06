import os
import sys
import psycopg2
from psycopg2 import errors

# Ensure backend directory is in python search path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from database import get_pooled_connection
from generate_embedding import extract_face_embedding
import numpy as np


class UserAlreadyExistsError(Exception):
    """Raised when an attempt is made to register a user_id that already exists in the database."""
    pass


class EmailAlreadyExistsError(Exception):
    """Raised when an attempt is made to register an email address that already exists in the database."""
    pass


class InvalidUserError(Exception):
    """Raised when user data, ID, or image is invalid."""
    pass


class UserNotFoundError(Exception):
    """Raised when a specified user does not exist in the database."""
    pass


def register_user_with_embedding(
    user_id: str,
    full_name: str = None,
    email: str = None,
    role: str = None,
    phone_number: str = None,
    face_bgr: np.ndarray = None,
    image_path: str = None,
    save_json: bool = False
) -> dict:
    """
    Registers a new user in the PostgreSQL database with their profile information
    and their 512-dimensional FaceNet embedding.

    Workflow:
    1. Validates user ID.
    2. Normalizes email (lowercase, stripped) and profile fields.
    3. Strictly enforces in-memory live webcam capture (face_bgr).
       Rejects file-based registration (image_path) to guarantee biometric authenticity.
    4. Generates the 512-dimensional FaceNet embedding using extract_face_embedding().
    5. Validates embedding length (512) and finite values.
    6. Connects to PostgreSQL via the connection pool and inserts the user record
       with profile data using parameterized queries.
    7. Confirms successful transaction commit.
    8. Synchronizes embedding to FAISS index ONLY post-commit.

    Parameters:
        user_id (str): Unique user identifier.
        full_name (str, optional): User's full name.
        email (str, optional): User's work email address.
        role (str, optional): User's role or designation.
        phone_number (str, optional): User's contact phone number.
        face_bgr (np.ndarray): In-memory cropped face image in BGR format from live webcam.
        image_path (str, optional): Forbidden legacy parameter; if provided, registration is rejected.
        save_json (bool): Deprecated legacy flag (retained for backward compatibility).

    Returns:
        dict: Registration confirmation containing user_id, profile fields, created_at, and embedding dimension.

    Raises:
        InvalidUserError: If user_id is empty, face_bgr is missing/invalid, or image_path is supplied.
        UserAlreadyExistsError: If user_id already exists in the database.
        EmailAlreadyExistsError: If email already exists in the database.
        ValueError: If embedding dimensions are not exactly 512.
        Exception: If database connection or query execution fails.
    """
    if not user_id or not str(user_id).strip():
        raise InvalidUserError("User ID must be a non-empty string.")
    clean_user_id = str(user_id).strip()
    clean_full_name = str(full_name).strip() if full_name and str(full_name).strip() else None
    clean_email = str(email).strip().lower() if email and str(email).strip() else None
    clean_role = str(role).strip() if role and str(role).strip() else None
    clean_phone = str(phone_number).strip() if phone_number and str(phone_number).strip() else None

    # Step 1: Reject file-based registration and strictly enforce in-memory live webcam frame
    if image_path is not None:
        raise InvalidUserError(
            "File-based registration via image_path is strictly prohibited. "
            "Face images must originate exclusively from verified live webcam capture."
        )

    if face_bgr is None or not isinstance(face_bgr, np.ndarray) or face_bgr.size == 0:
        raise InvalidUserError(
            "A valid in-memory webcam face frame (face_bgr) is strictly required for registration."
        )

    try:
        embedding_arr = extract_face_embedding(face_bgr)
        embedding_vector = [float(x) for x in embedding_arr]
        embedding_dim = len(embedding_vector)
    except Exception as emb_err:
        raise InvalidUserError(f"Failed to generate embedding: {emb_err}") from emb_err

    # Step 2: Validate embedding dimension before database insertion
    if embedding_dim != 512 or len(embedding_vector) != 512:
        raise ValueError(
            f"Pre-insertion validation failed: Expected 512 dimensions, got {len(embedding_vector)}."
        )

    # Step 3: Insert user, profile, and embedding into PostgreSQL with parameterized SQL query
    insert_sql = """
    INSERT INTO users (user_id, full_name, email, role, phone_number, embedding)
    VALUES (%s, %s, %s, %s, %s, %s)
    RETURNING user_id, full_name, email, role, phone_number, created_at, array_length(embedding, 1);
    """

    with get_pooled_connection() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    insert_sql,
                    (clean_user_id, clean_full_name, clean_email, clean_role, clean_phone, embedding_vector)
                )
                row = cur.fetchone()
            conn.commit()

            confirmed_uid = row[0]
            confirmed_name = row[1]
            confirmed_email = row[2]
            confirmed_role = row[3]
            confirmed_phone = row[4]
            confirmed_created_at = row[5]
            confirmed_dim = row[6]

            print(f"[DB SUCCESS] Registered user '{confirmed_uid}' with profile and {confirmed_dim}-D embedding.")

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
                "full_name": confirmed_name or "",
                "email": confirmed_email or "",
                "role": confirmed_role or "",
                "phone_number": confirmed_phone or "",
                "created_at": confirmed_created_at.isoformat() if hasattr(confirmed_created_at, "isoformat") else str(confirmed_created_at),
                "embedding_dimension": confirmed_dim,
                "message": "User registered and profile stored successfully in PostgreSQL."
            }

        except errors.UniqueViolation as uv_err:
            conn.rollback()
            err_msg = str(uv_err).lower()
            if "uq_users_email" in err_msg or "email" in err_msg:
                raise EmailAlreadyExistsError(
                    f"Email '{clean_email}' is already registered to another user."
                ) from uv_err
            else:
                raise UserAlreadyExistsError(
                    f"User ID '{clean_user_id}' already exists in database. Registration rejected."
                ) from uv_err
        except Exception as e:
            conn.rollback()
            print(f"[DB ERROR] Failed to store user '{clean_user_id}': {e}")
            raise e


def get_user_profile(user_id: str) -> dict | None:
    """
    Retrieves safe public profile for user without exposing embeddings or database internals.
    Guarantees no biometric leakage: embedding is never selected or returned.

    Parameters:
        user_id (str): Unique user identifier.

    Returns:
        dict | None: Profile dictionary with user_id, full_name, email, role, phone_number,
                     and created_at (or None if user not found).
    """
    if not user_id or not str(user_id).strip():
        return None
    clean_user_id = str(user_id).strip()

    select_sql = """
    SELECT user_id, full_name, email, role, phone_number, created_at
    FROM users
    WHERE user_id = %s;
    """

    with get_pooled_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(select_sql, (clean_user_id,))
            row = cur.fetchone()

    if not row:
        return None

    uid, fname, mail, urole, phone, created_at_val = row
    return {
        "user_id": str(uid),
        "full_name": str(fname) if fname else "",
        "email": str(mail) if mail else "",
        "role": str(urole) if urole else "",
        "phone_number": str(phone) if phone else "",
        "created_at": created_at_val.isoformat() if hasattr(created_at_val, "isoformat") else str(created_at_val)
    }


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


def get_all_users() -> list[dict]:
    """
    Retrieves all registered users from PostgreSQL, ordered chronologically (newest first).
    Does not expose raw embeddings.

    Returns:
        list[dict]: List of user summary objects with user_id, created_at, embedding_dimension, and has_embedding.
    """
    select_sql = """
    SELECT user_id, created_at, array_length(embedding, 1)
    FROM users
    ORDER BY created_at DESC;
    """
    with get_pooled_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(select_sql)
            rows = cur.fetchall()

    users = []
    for uid, cat, dim in rows:
        users.append({
            "user_id": str(uid),
            "created_at": cat.isoformat() if hasattr(cat, "isoformat") else str(cat),
            "embedding_dimension": dim,
            "has_embedding": (dim == 512)
        })
    return users


def delete_user_completely(user_id: str) -> dict:
    """
    Completely and safely deletes a registered user across all persistent layers:
    1. Validates non-empty user identifier.
    2. Verifies user existence in PostgreSQL (raises UserNotFoundError if missing).
    3. Deletes user row and 512-D FaceNet embedding from PostgreSQL in an ACID transaction.
    4. Removes corresponding vector entry from the in-memory FAISS index (or triggers safe DB rebuild).
    5. Deletes all associated cropped face images from backend/cropped_faces/.
    6. Guarantees strict PostgreSQL and FAISS index synchronization.

    Parameters:
        user_id (str): Unique user identifier.

    Returns:
        dict: Deletion report with status, user_id, and details of purged resources.

    Raises:
        InvalidUserError: If user_id is empty or whitespace.
        UserNotFoundError: If user does not exist in PostgreSQL.
        Exception: If database operation fails.
    """
    if not user_id or not str(user_id).strip():
        raise InvalidUserError("User ID must be a non-empty string.")
    clean_user_id = str(user_id).strip()

    # Step 1: Delete PostgreSQL database record (including 512-D embedding column)
    with get_pooled_connection() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT user_id FROM users WHERE user_id = %s;", (clean_user_id,))
                row = cur.fetchone()
                if not row:
                    raise UserNotFoundError(f"User '{clean_user_id}' does not exist in database.")

                cur.execute("DELETE FROM users WHERE user_id = %s;", (clean_user_id,))
            conn.commit()
            print(f"[DB SUCCESS] Deleted user record for '{clean_user_id}'.")
        except UserNotFoundError:
            conn.rollback()
            raise
        except Exception as e:
            conn.rollback()
            print(f"[DB ERROR] Failed to delete user '{clean_user_id}': {e}")
            raise e

    # Step 2: Remove FAISS vector entry and synchronize index
    faiss_removed = False
    try:
        from faiss_service import get_faiss_manager
        faiss_mgr = get_faiss_manager()
        if faiss_mgr.is_initialized():
            faiss_removed = faiss_mgr.remove_user(clean_user_id)
            print(f"[FAISS SUCCESS] Removed user '{clean_user_id}' from FAISS index.")
    except Exception as faiss_err:
        print(f"[FAISS WARNING] Error removing user '{clean_user_id}': {faiss_err}. Rebuilding FAISS index from PostgreSQL...")
        try:
            faiss_mgr.rebuild_index_from_database()
            faiss_removed = True
        except Exception as recon_err:
            print(f"[FAISS RECONCILIATION ERROR] Rebuild failed: {recon_err}")

    # Step 3: Delete associated cropped face images
    deleted_images = []
    cropped_dir = os.path.join(BASE_DIR, "cropped_faces")
    if os.path.isdir(cropped_dir):
        prefix_pattern = f"{clean_user_id}_"
        for fname in os.listdir(cropped_dir):
            if fname.startswith(prefix_pattern) or fname == f"{clean_user_id}.jpg" or fname == f"{clean_user_id}.png":
                fpath = os.path.join(cropped_dir, fname)
                try:
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                        deleted_images.append(fname)
                        print(f"[STORAGE] Deleted cropped face image: {fname}")
                except Exception as f_err:
                    print(f"[STORAGE WARNING] Failed to delete file '{fpath}': {f_err}")

    return {
        "status": "success",
        "message": f"User '{clean_user_id}' and all associated biometric embeddings and face images deleted successfully.",
        "user_id": clean_user_id,
        "deleted": {
            "database_record": True,
            "faiss_index": faiss_removed,
            "cropped_images": deleted_images,
            "embedding_files": []
        }
    }


def delete_user(user_id: str) -> bool:
    """
    Safely removes a user from PostgreSQL, FAISS, and filesystem.
    Backward-compatible helper function used by test suites and cleanup scripts.
    """
    if not user_id:
        return False
    try:
        res = delete_user_completely(user_id)
        return res.get("status") == "success"
    except UserNotFoundError:
        return False
    except Exception:
        return False

