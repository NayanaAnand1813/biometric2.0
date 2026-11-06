import os
import sys
import time
import asyncio
import httpx
import cv2
import numpy as np

# Ensure backend directory is in python search path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from main import app
from user_service import (
    register_user_with_embedding,
    get_user_embedding,
    delete_user,
    get_all_users
)
from faiss_service import get_faiss_manager
from database import get_pooled_connection


async def run_cleanup_verification():
    print("=" * 75)
    print("     POST-CLEANUP VERIFICATION SUITE: POSTGRESQL & FAISS WORKFLOW      ")
    print("=" * 75)

    fixture_image_path = os.path.join(BASE_DIR, "cropped_faces", "1804_face.jpg")
    assert os.path.exists(fixture_image_path), f"Fixture face image missing: {fixture_image_path}"
    with open(fixture_image_path, "rb") as f:
        face_bytes = f.read()

    # Verify embeddings directory does not exist
    emb_dir = os.path.join(BASE_DIR, "embeddings")
    assert not os.path.exists(emb_dir), f"Obsolete embeddings directory still exists at: {emb_dir}"
    print("\n[Sanity Check] Confirmed obsolete backend/embeddings directory is completely absent.")

    # -------------------------------------------------------------------------
    # Verification 1: Existing registered users remain accessible
    # -------------------------------------------------------------------------
    print("\n[Check 1/4] Verifying existing registered users remain accessible...")
    existing_user_id = "1804"
    existing_rec = get_user_embedding(existing_user_id)
    assert existing_rec is not None, f"Existing registered user '{existing_user_id}' not found in PostgreSQL!"
    assert existing_rec["user_id"] == existing_user_id
    assert existing_rec["embedding_dimension"] == 512
    assert len(existing_rec["embedding"]) == 512
    print(f"  [PASS] Existing user '{existing_user_id}' verified in PostgreSQL (512-D embedding).")

    # Verify existing user is queryable via GET /users
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp_users = await client.get("/users")
        assert resp_users.status_code == 200
        user_ids = [u["user_id"] for u in resp_users.json()["users"]]
        assert existing_user_id in user_ids, f"User '{existing_user_id}' missing from /users response"
        print(f"  [PASS] Existing user '{existing_user_id}' returned by /users endpoint.")

        # ---------------------------------------------------------------------
        # Verification 2: FAISS works without JSON embedding files
        # ---------------------------------------------------------------------
        print("\n[Check 2/4] Verifying FAISS initializes and operates without JSON embedding files...")
        faiss_mgr = get_faiss_manager()
        count = faiss_mgr.rebuild_index_from_database()
        assert count > 0, "FAISS failed to load embeddings from PostgreSQL!"
        assert faiss_mgr.contains_user(existing_user_id), f"FAISS does not contain user '{existing_user_id}'"
        assert not os.path.exists(emb_dir), "FAISS initialization unexpectedly recreated embeddings directory!"
        print(f"  [PASS] FAISS index rebuilt directly from PostgreSQL with {count} user vectors. Zero JSON files used.")

        # ---------------------------------------------------------------------
        # Verification 3: Login for existing registered user 1804
        # ---------------------------------------------------------------------
        print("\n[Check 3/4] Verifying login endpoint successfully matches face embeddings via FAISS...")

        # 3a: Existing user 1804 authenticates with 1804_face.jpg
        login_files_1804 = {"image": ("login.jpg", face_bytes, "image/jpeg")}
        login_data_1804 = {"user_id": existing_user_id}
        resp_login_1804 = await client.post("/login", data=login_data_1804, files=login_files_1804)
        assert resp_login_1804.status_code == 200, f"Expected 200, got {resp_login_1804.status_code}"
        assert resp_login_1804.json()["authenticated"] is True
        assert resp_login_1804.json()["user_id"] == existing_user_id
        print(f"  [PASS] Login successful for existing registered user '{existing_user_id}'.")

        # 3b: Denied login for non-existent / impostor user ID
        impostor_data = {"user_id": "impostor_user_99999"}
        resp_login_impostor = await client.post(
            "/login",
            data=impostor_data,
            files={"image": ("login.jpg", face_bytes, "image/jpeg")}
        )
        assert resp_login_impostor.status_code == 401
        assert resp_login_impostor.json()["detail"]["authenticated"] is False
        print("  [PASS] Login correctly denied for unregistered user ID.")

        # ---------------------------------------------------------------------
        # Verification 4: New registration saves embeddings in PostgreSQL
        # ---------------------------------------------------------------------
        print("\n[Check 4/4] Verifying new user registration persists embedding to PostgreSQL...")
        test_new_uid = f"test_cleanup_reg_{int(time.time())}"
        delete_user(test_new_uid)

        try:
            # Register via user_service using in-memory frame
            fixture_bgr = cv2.imread(fixture_image_path)
            reg_result = register_user_with_embedding(test_new_uid, face_bgr=fixture_bgr)
            assert reg_result["status"] == "success"
            assert reg_result["user_id"] == test_new_uid
            assert reg_result["embedding_dimension"] == 512

            # Check directly in PostgreSQL database table
            with get_pooled_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT user_id, array_length(embedding, 1) FROM users WHERE user_id = %s;",
                        (test_new_uid,)
                    )
                    db_row = cur.fetchone()

            assert db_row is not None, f"New user '{test_new_uid}' not found in PostgreSQL table!"
            assert db_row[0] == test_new_uid
            assert db_row[1] == 512, f"Expected 512 dimensions in PostgreSQL, got {db_row[1]}"
            print(f"  [PASS] Verified embedding directly in PostgreSQL table `users` for '{test_new_uid}'.")

            # Verify no JSON file was created on disk
            assert not os.path.exists(emb_dir), "Registration created an obsolete embeddings directory!"
            print("  [PASS] Confirmed zero JSON files written to disk during registration.")

            # Verify FAISS was synchronized with new user
            assert faiss_mgr.contains_user(test_new_uid), f"FAISS was not updated with new user '{test_new_uid}'"
            print(f"  [PASS] FAISS synchronized with new user '{test_new_uid}'.")

            # Verify newly registered user can authenticate
            login_files_new = {"image": ("login.jpg", face_bytes, "image/jpeg")}
            login_data_new = {"user_id": test_new_uid}
            resp_login_new = await client.post("/login", data=login_data_new, files=login_files_new)
            assert resp_login_new.status_code == 200
            assert resp_login_new.json()["authenticated"] is True
            print(f"  [PASS] Login successful for newly registered user '{test_new_uid}'.")

        finally:
            # Cleanup temporary test user
            print("\n[Cleanup] Cleaning up temporary test user...")
            del_res = delete_user(test_new_uid)
            assert del_res is True
            assert get_user_embedding(test_new_uid) is None
            assert not faiss_mgr.contains_user(test_new_uid)
            print("  [PASS] Temporary test user cleaned up.")

        # Ensure existing user 1804 is still intact and accessible
        assert get_user_embedding(existing_user_id) is not None
        assert faiss_mgr.contains_user(existing_user_id)
        print(f"  [PASS] Existing user '{existing_user_id}' remains intact and accessible.")

    print("\n" + "=" * 75)
    print("      ALL POST-CLEANUP VERIFICATION TESTS COMPLETED SUCCESSFULLY!       ")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(run_cleanup_verification())
