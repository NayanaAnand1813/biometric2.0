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

from main import app, ADMIN_API_KEY
from user_service import (
    register_user_with_embedding,
    get_user_embedding,
    delete_user,
    get_all_users
)
from faiss_service import get_faiss_manager


async def test_delete_user_suite():
    print("=" * 70)
    print("        DELETE USER FUNCTIONALITY & SYNCHRONIZATION TEST SUITE       ")
    print("=" * 70)

    test_image_path = os.path.join(BASE_DIR, "cropped_faces", "1804_face.jpg")
    if not os.path.exists(test_image_path):
        test_image_path = os.path.join(BASE_DIR, "cropped_faces", "1234_face.jpg")
    if not os.path.exists(test_image_path):
        test_image_path = os.path.join(BASE_DIR, "test_fixtures", "test_face.jpg")
    assert os.path.exists(test_image_path), "Test face image fixture missing."
    with open(test_image_path, "rb") as f:
        valid_jpeg_bytes = f.read()

    timestamp = int(time.time())
    user_victim = f"del_user_{timestamp}"
    user_survivor = f"survivor_{timestamp}"

    # Clean pre-existing test fixtures
    delete_user(user_victim)
    delete_user(user_survivor)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:

        # -------------------------------------------------------------
        # Setup: Register two users
        # -------------------------------------------------------------
        print("\n[Setup] Registering victim and survivor users...")
        reg1_resp = await client.post(
            "/upload",
            data={"user_id": user_victim},
            files={"image": ("face.jpg", valid_jpeg_bytes, "image/jpeg")}
        )
        assert reg1_resp.status_code == 200, f"Victim setup registration failed: {reg1_resp.text}"

        reg2_resp = await client.post(
            "/upload",
            data={"user_id": user_survivor},
            files={"image": ("face.jpg", valid_jpeg_bytes, "image/jpeg")}
        )
        assert reg2_resp.status_code == 200, f"Survivor setup registration failed: {reg2_resp.text}"

        faiss_mgr = get_faiss_manager()
        assert faiss_mgr.contains_user(user_victim), "Victim not indexed in FAISS!"
        assert faiss_mgr.contains_user(user_survivor), "Survivor not indexed in FAISS!"

        victim_img_path = os.path.join(BASE_DIR, "cropped_faces", f"{user_victim}_face.jpg")
        assert os.path.exists(victim_img_path), "Victim face image could not be created."
        print(f"  - Registered '{user_victim}' and created fixture image: {victim_img_path}")
        print(f"  - Registered '{user_survivor}' as control subject.")

        # -------------------------------------------------------------
        # Test 1: Verify GET /users endpoint
        # -------------------------------------------------------------
        print("\n[Check 1/6] Testing GET /users endpoint...")
        resp_list = await client.get("/users")
        assert resp_list.status_code == 200, f"Expected 200, got {resp_list.status_code}"
        users_data = resp_list.json()
        uids = [u["user_id"] for u in users_data.get("users", [])]
        assert user_victim in uids, f"Victim '{user_victim}' not in GET /users output"
        assert user_survivor in uids, f"Survivor '{user_survivor}' not in GET /users output"
        print(f"  [PASS] GET /users verified. Total registered users: {users_data['total']}")

        # -------------------------------------------------------------
        # Test 2: Admin Authorization Enforcement
        # -------------------------------------------------------------
        print("\n[Check 2/6] Testing rejection of unauthorized DELETE request (Missing / Invalid Key)...")
        # 2a: No header
        resp_unauth = await client.delete(f"/users/{user_victim}")
        assert resp_unauth.status_code == 401, f"Expected 401, got {resp_unauth.status_code}"
        print("  [PASS] DELETE without admin key correctly rejected with HTTP 401.")

        # 2b: Invalid header
        resp_wrong = await client.delete(f"/users/{user_victim}", headers={"X-Admin-Key": "wrong_key_123"})
        assert resp_wrong.status_code == 401, f"Expected 401, got {resp_wrong.status_code}"
        print("  [PASS] DELETE with invalid admin key correctly rejected with HTTP 401.")

        # -------------------------------------------------------------
        # Test 3: Missing User Handling (404 Not Found)
        # -------------------------------------------------------------
        print("\n[Check 3/6] Testing deletion of non-existent user...")
        resp_notfound = await client.delete(
            f"/users/ghost_user_nonexistent_{timestamp}",
            headers={"X-Admin-Key": ADMIN_API_KEY}
        )
        assert resp_notfound.status_code == 404, f"Expected 404, got {resp_notfound.status_code}"
        print("  [PASS] Non-existent user correctly returned HTTP 404 Not Found.")

        # -------------------------------------------------------------
        # Test 4: Successful User Deletion with Admin Authorization
        # -------------------------------------------------------------
        print(f"\n[Check 4/6] Executing authorized DELETE /users/{user_victim}...")
        resp_del = await client.delete(
            f"/users/{user_victim}",
            headers={"X-Admin-Key": ADMIN_API_KEY}
        )
        print(f"  - Status Code: {resp_del.status_code}")
        print(f"  - Response JSON: {resp_del.json()}")

        assert resp_del.status_code == 200, f"Expected 200, got {resp_del.status_code}"
        del_data = resp_del.json()
        assert del_data["status"] == "success"
        assert del_data["user_id"] == user_victim
        assert del_data["deleted"]["database_record"] is True
        assert del_data["deleted"]["faiss_index"] is True
        print("  [PASS] Deletion endpoint returned 200 Success with detailed purge report.")

        # -------------------------------------------------------------
        # Test 5: Verify Persistence Layer Purge (PG, FAISS, Filesystem)
        # -------------------------------------------------------------
        print("\n[Check 5/6] Verifying user is purged from PostgreSQL, FAISS, and Filesystem...")
        # 5a: PostgreSQL check
        assert get_user_embedding(user_victim) is None, "Victim still exists in PostgreSQL!"
        print("  [PASS] PostgreSQL record and 512-D embedding confirmed purged.")

        # 5b: FAISS check
        assert not faiss_mgr.contains_user(user_victim), "Victim still indexed in FAISS!"
        print("  [PASS] FAISS index entry confirmed removed and index synchronized.")

        # 5c: Filesystem check
        assert not os.path.exists(victim_img_path), f"Cropped face image {victim_img_path} was not deleted!"
        print("  [PASS] Associated cropped face image confirmed deleted from disk.")

        # -------------------------------------------------------------
        # Test 6: Verify Survivor User Can Still Authenticate (No Side Effects)
        # -------------------------------------------------------------
        print("\n[Check 6/6] Verifying survivor user remains completely intact and can authenticate...")
        survivor_record = get_user_embedding(user_survivor)
        assert survivor_record is not None, "Survivor user corrupted or deleted!"
        assert survivor_record["embedding_dimension"] == 512
        assert faiss_mgr.contains_user(user_survivor), "Survivor user lost from FAISS index!"

        # Attempt login with survivor user
        with open(test_image_path, "rb") as f:
            survivor_img_bytes = f.read()

        login_files = {"image": ("login_capture.jpg", survivor_img_bytes, "image/jpeg")}
        login_data = {"user_id": user_survivor}

        resp_survivor_login = await client.post("/login", data=login_data, files=login_files)
        print(f"  - Survivor Login Status: {resp_survivor_login.status_code}")
        print(f"  - Survivor Login JSON: {resp_survivor_login.json()}")
        assert resp_survivor_login.status_code == 200
        assert resp_survivor_login.json()["authenticated"] is True
        print(f"  [PASS] Survivor user '{user_survivor}' successfully authenticated without any side effect.")

        # Final cleanup of survivor
        delete_user(user_survivor)
        print("\n[CLEANUP] Cleaned up survivor test record.")

    print("\n" + "=" * 70)
    print("  [ALL TESTS PASSED] DELETE USER FUNCTIONALITY COMPLETELY VERIFIED! ")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_delete_user_suite())

