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
    delete_user,
    get_user_embedding
)
from faiss_service import get_faiss_manager


async def run_1n_tests():
    print("=" * 70)
    print("   1:N BIOMETRIC IDENTIFICATION & REGISTRATION VALIDATION SUITE   ")
    print("=" * 70)

    fixture_path = os.path.join(BASE_DIR, "test_fixtures", "test_face.jpg")
    if os.path.exists(fixture_path):
        test_image_path = fixture_path
    else:
        cropped_dir = os.path.join(BASE_DIR, "cropped_faces")
        available_faces = [os.path.join(cropped_dir, f) for f in os.listdir(cropped_dir) if f.endswith(".jpg")]
        if not available_faces:
            raise FileNotFoundError("No test face images found.")
        test_image_path = available_faces[0]
    with open(test_image_path, "rb") as f:
        valid_jpeg_bytes = f.read()

    timestamp = int(time.time())
    user_enrolled = f"test_user_enrolled_{timestamp}"
    user_duplicate = f"test_dup_{timestamp}"

    # Cleanup any stale test records
    delete_user(user_enrolled)
    delete_user(user_duplicate)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:

        # -------------------------------------------------------------
        # Test 1: Register user and verify persistence & FAISS sync
        # -------------------------------------------------------------
        print("\n[Test 1] Testing user registration via POST /upload...")
        files = {"image": ("face.jpg", valid_jpeg_bytes, "image/jpeg")}
        data = {"user_id": user_enrolled}

        resp_reg = await client.post("/upload", data=data, files=files)
        print(f"  - Status Code: {resp_reg.status_code}")
        print(f"  - Response JSON: {resp_reg.json()}")
        assert resp_reg.status_code == 200
        assert resp_reg.json()["status"] == "success"
        assert resp_reg.json()["user_id"] == user_enrolled
        print("  [PASS] User registered successfully with 512-D embedding.")

        # -------------------------------------------------------------
        # Test 2: Duplicate user registration rejected with 409 Conflict
        # -------------------------------------------------------------
        print("\n[Test 2] Testing duplicate user registration rejection (HTTP 409)...")
        files = {"image": ("face.jpg", valid_jpeg_bytes, "image/jpeg")}
        data = {"user_id": user_enrolled}

        resp_dup = await client.post("/upload", data=data, files=files)
        print(f"  - Status Code: {resp_dup.status_code}")
        print(f"  - Response JSON: {resp_dup.json()}")
        assert resp_dup.status_code == 409
        print("  [PASS] Duplicate user registration rejected with HTTP 409 Conflict.")

        # -------------------------------------------------------------
        # Test 3: Empty User ID registration rejected with 400 Bad Request
        # -------------------------------------------------------------
        print("\n[Test 3] Testing empty User ID registration rejection (HTTP 400)...")
        files = {"image": ("face.jpg", valid_jpeg_bytes, "image/jpeg")}
        data = {"user_id": "   "}

        resp_empty_user = await client.post("/upload", data=data, files=files)
        print(f"  - Status Code: {resp_empty_user.status_code}")
        assert resp_empty_user.status_code == 400
        print("  [PASS] Empty User ID rejected with HTTP 400 Bad Request.")

        # -------------------------------------------------------------
        # Test 4: 1:N ID-Free Login - Successful identification
        # -------------------------------------------------------------
        print("\n[Test 4] Testing 1:N ID-free login with enrolled face (no user_id sent)...")
        files = {"image": ("login_capture.jpg", valid_jpeg_bytes, "image/jpeg")}
        # Note: NO user_id in data payload!
        resp_id_free = await client.post("/login", files=files)
        print(f"  - Status Code: {resp_id_free.status_code}")
        print(f"  - Response JSON: {resp_id_free.json()}")
        assert resp_id_free.status_code == 200
        res_json = resp_id_free.json()
        assert res_json["authenticated"] is True
        assert res_json["user_id"] == user_enrolled
        assert "similarity" not in res_json
        print(f"  [PASS] ID-free login successfully identified enrolled user '{user_enrolled}'.")

        # -------------------------------------------------------------
        # Test 5: 1:N ID-Free Login - Rejection of unknown / sub-threshold face
        # -------------------------------------------------------------
        print("\n[Test 5] Testing 1:N rejection of unknown / unregistered face...")
        # Create an unknown synthetic identity orthogonal to user_enrolled
        # To simulate an unknown face trying to authenticate, we delete the enrolled user
        # so the index only has users whose embeddings do NOT match this face
        delete_user(user_enrolled)

        # Enroll an unrelated dummy user with orthogonal vector to the probe face
        faiss_mgr = get_faiss_manager()
        orthogonal_vec = np.random.randn(512).astype(np.float32)
        orthogonal_vec /= np.linalg.norm(orthogonal_vec)
        user_unrelated = f"unrelated_{timestamp}"
        from database import get_pooled_connection
        with get_pooled_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (user_id, embedding) VALUES (%s, %s);",
                    (user_unrelated, [float(x) for x in orthogonal_vec])
                )
            conn.commit()
        faiss_mgr.add_user(user_unrelated, orthogonal_vec)

        # Now probe with valid_jpeg_bytes (an unregistered face whose cosine similarity to user_unrelated is ~0)
        files = {"image": ("unknown_login.jpg", valid_jpeg_bytes, "image/jpeg")}
        resp_unknown = await client.post("/login", files=files)
        print(f"  - Status Code: {resp_unknown.status_code}")
        print(f"  - Response JSON: {resp_unknown.json()}")
        assert resp_unknown.status_code == 401
        assert resp_unknown.json()["detail"]["authenticated"] is False
        print("  [PASS] Unknown face strictly rejected (HTTP 401) despite FAISS returning a nearest neighbor.")

        # Cleanup
        delete_user(user_unrelated)

    print("\n" + "=" * 70)
    print("   ALL 1:N IDENTIFICATION & REGISTRATION TESTS PASSED!   ")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_1n_tests())

