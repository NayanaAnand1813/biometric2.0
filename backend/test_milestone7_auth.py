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


async def run_milestone7_auth_suite():
    print("=" * 70)
    print("      MILESTONE 7 (WAVE 2): BIOMETRIC LOGIN ACCEPTANCE SUITE      ")
    print("=" * 70)

    test_image_path = os.path.join(BASE_DIR, "cropped_faces", "1804_face.jpg")
    if not os.path.exists(test_image_path):
        test_image_path = os.path.join(BASE_DIR, "cropped_faces", "1234_face.jpg")
    with open(test_image_path, "rb") as f:
        valid_jpeg_bytes = f.read()

    # Create in-memory blank non-face image
    blank_img = np.zeros((300, 300, 3), dtype=np.uint8)
    _, blank_jpeg_bytes = cv2.imencode(".jpg", blank_img)
    blank_bytes = blank_jpeg_bytes.tobytes()

    timestamp = int(time.time())
    user_alice = f"m7_alice_{timestamp}"
    user_bob = f"m7_bob_{timestamp}"
    ghost_user = f"m7_ghost_{timestamp}"

    # Pre-test cleanup
    delete_user(user_alice)
    delete_user(user_bob)
    delete_user(ghost_user)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:

        # ----------------------------------------------------------------------
        # Setup: Enroll User Alice with valid face image
        # ----------------------------------------------------------------------
        print("\n[Setup] Enrolling test users into PostgreSQL & FAISS...")
        reg_alice = register_user_with_embedding(user_alice, image_path=test_image_path)
        assert reg_alice["status"] == "success"

        # Enroll User Bob with an orthogonal embedding vector
        np.random.seed(99)
        raw_vec = np.random.randn(512).astype(np.float32)
        alice_vec = np.array(reg_alice["embedding"] if "embedding" in reg_alice else get_user_embedding(user_alice)["embedding"], dtype=np.float32)
        # Gram-Schmidt orthogonalization to ensure zero cosine similarity with Alice
        bob_vec = raw_vec - np.dot(raw_vec, alice_vec) * alice_vec
        bob_vec = bob_vec / np.linalg.norm(bob_vec)

        # Insert bob directly into PostgreSQL & FAISS
        from database import get_pooled_connection
        with get_pooled_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (user_id, embedding) VALUES (%s, %s);",
                    (user_bob, [float(x) for x in bob_vec])
                )
            conn.commit()

        faiss_mgr = get_faiss_manager()
        faiss_mgr.add_user(user_bob, bob_vec)
        print(f"  - Registered '{user_alice}' with face image fixture.")
        print(f"  - Registered '{user_bob}' with orthogonal embedding (zero similarity to Alice).")

        # ----------------------------------------------------------------------
        # Check 1: Genuine User Login Succeeds
        # ----------------------------------------------------------------------
        print("\n[Check 1/8] Testing genuine user login (Alice claiming Alice)...")
        files = {"image": ("login_capture.jpg", valid_jpeg_bytes, "image/jpeg")}
        data = {"user_id": user_alice}

        resp_genuine = await client.post("/login", data=data, files=files)
        print(f"  - Status Code: {resp_genuine.status_code}")
        print(f"  - Response JSON: {resp_genuine.json()}")

        assert resp_genuine.status_code == 200, f"Expected 200, got {resp_genuine.status_code}"
        res_json = resp_genuine.json()
        assert res_json["status"] == "success"
        assert res_json["authenticated"] is True
        assert res_json["user_id"] == user_alice

        # Verify no similarity score or raw embedding is exposed
        assert "similarity" not in res_json, "Security leak: similarity score exposed to client!"
        assert "score" not in res_json, "Security leak: score exposed to client!"
        assert "embedding" not in res_json, "Security leak: raw embedding exposed to client!"
        print("  [PASS] Genuine user authenticated successfully with no score leakage.")

        # ----------------------------------------------------------------------
        # Check 2: Another user's face claiming first user's ID is denied (Identity Mismatch)
        # ----------------------------------------------------------------------
        print("\n[Check 2/8] Testing impostor / identity mismatch (Alice's face claiming Bob's ID)...")
        files = {"image": ("login_capture.jpg", valid_jpeg_bytes, "image/jpeg")}
        data = {"user_id": user_bob}

        resp_mismatch = await client.post("/login", data=data, files=files)
        print(f"  - Status Code: {resp_mismatch.status_code}")
        print(f"  - Response JSON: {resp_mismatch.json()}")

        assert resp_mismatch.status_code == 401, f"Expected 401 Unauthorized, got {resp_mismatch.status_code}"
        detail_json = resp_mismatch.json().get("detail", {})
        assert detail_json.get("authenticated") is False
        assert "similarity" not in detail_json, "Security leak in error detail!"
        print("  [PASS] Identity mismatch correctly rejected with HTTP 401 Unauthorized.")

        # ----------------------------------------------------------------------
        # Check 3: Unknown / Unregistered user ID is denied
        # ----------------------------------------------------------------------
        print("\n[Check 3/8] Testing unknown user ID rejection...")
        files = {"image": ("login_capture.jpg", valid_jpeg_bytes, "image/jpeg")}
        data = {"user_id": ghost_user}

        resp_ghost = await client.post("/login", data=data, files=files)
        print(f"  - Status Code: {resp_ghost.status_code}")
        print(f"  - Response JSON: {resp_ghost.json()}")

        assert resp_ghost.status_code == 401, f"Expected 401, got {resp_ghost.status_code}"
        print("  [PASS] Unknown user ID correctly denied access.")

        # ----------------------------------------------------------------------
        # Check 4: Image containing no face is rejected
        # ----------------------------------------------------------------------
        print("\n[Check 4/8] Testing non-face image rejection...")
        files = {"image": ("blank.jpg", blank_bytes, "image/jpeg")}
        data = {"user_id": user_alice}

        resp_noface = await client.post("/login", data=data, files=files)
        print(f"  - Status Code: {resp_noface.status_code}")
        print(f"  - Response JSON: {resp_noface.json()}")

        assert resp_noface.status_code == 400, f"Expected 400 Bad Request, got {resp_noface.status_code}"
        detail_noface = resp_noface.json().get("detail", {})
        assert "no face" in detail_noface.get("message", "").lower() or "no_face" in str(detail_noface).lower()
        print("  [PASS] Non-face image correctly rejected with HTTP 400 Bad Request.")

        # ----------------------------------------------------------------------
        # Check 5: Corrupt and empty images are rejected
        # ----------------------------------------------------------------------
        print("\n[Check 5/8] Testing corrupt and empty image payloads...")
        # Empty image
        resp_empty_img = await client.post("/login", data={"user_id": user_alice}, files={"image": ("empty.jpg", b"", "image/jpeg")})
        assert resp_empty_img.status_code == 400
        # Corrupt bytes
        resp_corrupt = await client.post("/login", data={"user_id": user_alice}, files={"image": ("bad.jpg", b"corrupted data", "image/jpeg")})
        assert resp_corrupt.status_code == 400
        # Empty user_id
        resp_empty_uid = await client.post("/login", data={"user_id": "  "}, files={"image": ("login.jpg", valid_jpeg_bytes, "image/jpeg")})
        assert resp_empty_uid.status_code == 400
        print("  [PASS] Empty, corrupted, and invalid payloads rejected with HTTP 400.")

        # ----------------------------------------------------------------------
        # Check 6: Zero disk writes & zero database alterations during login
        # ----------------------------------------------------------------------
        print("\n[Check 6/8] Verifying zero disk writes and zero database insertions during login...")
        cropped_dir = os.path.join(BASE_DIR, "cropped_faces")
        initial_file_count = len(os.listdir(cropped_dir))

        with get_pooled_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM users;")
                initial_db_count = cur.fetchone()[0]

        # Execute genuine login
        files = {"image": ("login_capture.jpg", valid_jpeg_bytes, "image/jpeg")}
        data = {"user_id": user_alice}
        resp_zero_leak = await client.post("/login", data=data, files=files)
        assert resp_zero_leak.status_code == 200

        # Check directory and DB counts
        final_file_count = len(os.listdir(cropped_dir))
        with get_pooled_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM users;")
                final_db_count = cur.fetchone()[0]

        assert final_file_count == initial_file_count, f"Disk leak: file count changed from {initial_file_count} to {final_file_count}!"
        assert final_db_count == initial_db_count, f"Database leak: user count changed from {initial_db_count} to {final_db_count}!"
        print("  [PASS] Verified zero disk writes and zero DB record alterations during login.")

        # ----------------------------------------------------------------------
        # Check 7: Deterministic Threshold Rejection and Acceptance Verification
        # ----------------------------------------------------------------------
        print("\n[Check 7/8] Testing deterministic threshold rejection and acceptance...")
        user_charlie = f"m7_charlie_{timestamp}"
        delete_user(user_charlie)

        # 1. Synthesize an enrolled vector with an exact controlled cosine similarity of 0.60 to Alice's face
        # probe vector = alice_vec (unit vector from 1234_face.jpg)
        np.random.seed(12345)
        raw_noise = np.random.randn(512).astype(np.float32)
        proj_noise = raw_noise - np.dot(raw_noise, alice_vec) * alice_vec
        ortho_unit = proj_noise / np.linalg.norm(proj_noise)

        target_sim = 0.60
        beta_weight = np.sqrt(1.0 - target_sim**2)
        charlie_vec = (target_sim * alice_vec + beta_weight * ortho_unit).astype(np.float32)
        charlie_vec /= np.linalg.norm(charlie_vec)

        # Verify exact mathematical cosine similarity
        exact_sim = float(np.dot(charlie_vec, alice_vec))
        assert abs(exact_sim - 0.60) < 1e-5, f"Vector construction error: expected 0.60, got {exact_sim}"

        # 2. Temporarily remove Alice from FAISS so Charlie is the top/sole candidate
        faiss_mgr.remove_user(user_alice)

        # 3. Enroll Charlie with this controlled vector into DB and FAISS
        with get_pooled_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (user_id, embedding) VALUES (%s, %s);",
                    (user_charlie, [float(x) for x in charlie_vec])
                )
            conn.commit()
        faiss_mgr.add_user(user_charlie, charlie_vec)
        print(f"  - Enrolled '{user_charlie}' as sole candidate with controlled embedding (exact similarity = {exact_sim:.4f} to probe image).")

        # ----------------------------------------------------------------------
        # Case A: Claimed ID matches top match, but similarity (0.60) is DELIBERATELY BELOW threshold (0.75)
        # Expected: HTTP 401 Unauthorized, authenticated=False
        # ----------------------------------------------------------------------
        os.environ["BIOMETRIC_COSINE_THRESHOLD"] = "0.75"
        resp_sub_thresh = await client.post(
            "/login",
            data={"user_id": user_charlie},
            files={"image": ("login.jpg", valid_jpeg_bytes, "image/jpeg")}
        )
        print(f"  - Case A (Sub-Threshold: sim={exact_sim:.2f} < threshold=0.75): HTTP {resp_sub_thresh.status_code}")
        assert resp_sub_thresh.status_code == 401, f"Expected 401 for sub-threshold login, got {resp_sub_thresh.status_code}"
        detail_sub = resp_sub_thresh.json().get("detail", {})
        assert detail_sub.get("authenticated") is False
        assert "similarity" not in detail_sub and "score" not in detail_sub and "embedding" not in detail_sub
        print("  [PASS] Case A: Sub-threshold probe correctly denied (HTTP 401, authenticated=False, no score leak).")

        # ----------------------------------------------------------------------
        # Case B: Claimed ID matches top match, and similarity (0.60) is ABOVE threshold (0.50)
        # Expected: HTTP 200 OK, authenticated=True
        # ----------------------------------------------------------------------
        os.environ["BIOMETRIC_COSINE_THRESHOLD"] = "0.50"
        resp_above_thresh = await client.post(
            "/login",
            data={"user_id": user_charlie},
            files={"image": ("login.jpg", valid_jpeg_bytes, "image/jpeg")}
        )
        print(f"  - Case B (Above-Threshold: sim={exact_sim:.2f} >= threshold=0.50): HTTP {resp_above_thresh.status_code}")
        assert resp_above_thresh.status_code == 200, f"Expected 200 for above-threshold login, got {resp_above_thresh.status_code}"
        res_above = resp_above_thresh.json()
        assert res_above.get("authenticated") is True
        assert res_above.get("status") == "success"
        assert res_above.get("user_id") == user_charlie
        assert "similarity" not in res_above and "score" not in res_above and "embedding" not in res_above
        print("  [PASS] Case B: Above-threshold probe correctly authenticated (HTTP 200, authenticated=True, no score leak).")

        # ----------------------------------------------------------------------
        # Clean up Charlie and re-enroll Alice for remaining tests
        # ----------------------------------------------------------------------
        delete_user(user_charlie)
        faiss_mgr.remove_user(user_charlie)
        faiss_mgr.add_user(user_alice, alice_vec)
        os.environ["BIOMETRIC_COSINE_THRESHOLD"] = "0.70"
        print("  [PASS] Both authentication gates independently verified with mathematical determinism.")

        # ----------------------------------------------------------------------
        # Check 8: Concurrent Login Requests Isolation
        # ----------------------------------------------------------------------
        print("\n[Check 8/8] Testing 5 simultaneous login requests for concurrency safety...")
        async def send_login_request(req_id: int):
            f = {"image": (f"login_{req_id}.jpg", valid_jpeg_bytes, "image/jpeg")}
            d = {"user_id": user_alice}
            return await client.post("/login", data=d, files=f)

        tasks = [send_login_request(i) for i in range(5)]
        responses = await asyncio.gather(*tasks)

        for i, r in enumerate(responses):
            assert r.status_code == 200, f"Request {i} failed with status {r.status_code}"
            assert r.json()["authenticated"] is True
        print("  [PASS] All 5 concurrent login requests succeeded without error or FAISS corruption.")

        # ----------------------------------------------------------------------
        # Cleanup
        # ----------------------------------------------------------------------
        delete_user(user_alice)
        delete_user(user_bob)
        faiss_mgr.remove_user(user_alice)
        faiss_mgr.remove_user(user_bob)
        print("\n[CLEANUP] Test users cleaned up from PostgreSQL & FAISS.")

    print("\n" + "=" * 70)
    print("  [ALL ACCEPTANCE CRITERIA PASSED] MILESTONE 7 WAVE 2 VERIFIED!  ")
    print("=================================================================")


if __name__ == "__main__":
    asyncio.run(run_milestone7_auth_suite())

