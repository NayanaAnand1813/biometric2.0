"""
Comprehensive Validation Suite for Liveness-First Biometric Pipeline.

Tests all requirements specified in Section 7:
1. Genuine face: liveness passes (score >= 0.85), verified frame captured,
   MediaPipe BlazeFace crops face, FaceNet generates 512-D embedding.
2. Printed photograph spoof (image_F1.jpg): rejected before MediaPipe/FaceNet.
3. Mobile screen / replay spoof (image_F2.jpg): rejected before MediaPipe/FaceNet.
4. Uncertain / sub-threshold confidence: strictly rejected.
5. Invalid or missing face (blank / non-face): rejected before FaceNet.
6. Failed registration: guarantees zero database write, zero FAISS insertion, zero disk image saved.
7. Failed login: guarantees no FAISS authentication.
8. Successful registration: PostgreSQL and FAISS strictly synchronized.
9. Successful login: correct user authenticated, profile returned.
10. Processing order enforcement: spies verify MediaPipe and FaceNet are never called on failed liveness.
11. Dedicated /check-liveness endpoint tests.
"""

import os
import sys
import time
import asyncio
import unittest.mock as mock
import cv2
import numpy as np
import httpx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from main import app, _process_registration_sync, _process_login_sync, detector
from user_service import delete_user, get_user_embedding, get_user_profile
from faiss_service import get_faiss_manager
from liveness_service import get_liveness_service


async def run_liveness_pipeline_tests():
    print("=" * 75)
    print("   LIVENESS-FIRST FACE REGISTRATION & LOGIN PIPELINE VALIDATION   ")
    print("=" * 75)

    # Load test image fixtures
    real_face_path = os.path.join(BASE_DIR, "test_fixtures", "test_face.jpg")
    fake_print_path = os.path.join(BASE_DIR, "test_fixtures", "fake_print.jpg")
    fake_replay_path = os.path.join(BASE_DIR, "test_fixtures", "fake_replay.jpg")

    assert os.path.isfile(real_face_path), f"Missing {real_face_path}"
    assert os.path.isfile(fake_print_path), f"Missing {fake_print_path}"
    assert os.path.isfile(fake_replay_path), f"Missing {fake_replay_path}"

    with open(real_face_path, "rb") as f:
        real_jpeg_bytes = f.read()

    with open(fake_print_path, "rb") as f:
        fake_print_bytes = f.read()

    with open(fake_replay_path, "rb") as f:
        fake_replay_bytes = f.read()

    # Synthetic non-face / noise image
    blank_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    _, blank_bytes_arr = cv2.imencode(".jpg", blank_bgr)
    blank_jpeg_bytes = blank_bytes_arr.tobytes()

    timestamp = int(time.time())
    user_genuine = f"live_genuine_{timestamp}"
    user_spoof = f"live_spoof_{timestamp}"

    delete_user(user_genuine)
    delete_user(user_spoof)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:

        # ----------------------------------------------------------------------
        # Test 1: Dedicated /check-liveness endpoint
        # ----------------------------------------------------------------------
        print("\n[Test 1] Dedicated /check-liveness endpoint verification...")
        resp_t1 = await client.post("/check-liveness", files={"image": ("real.jpg", real_jpeg_bytes, "image/jpeg")})
        assert resp_t1.status_code == 200
        res_t1 = resp_t1.json()
        assert res_t1["is_real"] is True
        assert res_t1["score"] >= 0.85
        assert res_t1["label"] == "Real"
        print(f"  [PASS] Genuine face verified: is_real=True, score={res_t1['score']:.4f} >= 0.85")

        resp_f1 = await client.post("/check-liveness", files={"image": ("fake_print.jpg", fake_print_bytes, "image/jpeg")})
        assert resp_f1.status_code == 200
        res_f1 = resp_f1.json()
        assert res_f1["is_real"] is False
        assert res_f1["label"] == "Fake"
        print(f"  [PASS] Printed photograph spoof rejected: is_real=False, score={res_f1['score']:.4f}, label={res_f1['label']}")

        resp_f2 = await client.post("/check-liveness", files={"image": ("fake_replay.jpg", fake_replay_bytes, "image/jpeg")})
        assert resp_f2.status_code == 200
        res_f2 = resp_f2.json()
        assert res_f2["is_real"] is False
        assert res_f2["label"] == "Fake"
        print(f"  [PASS] Screen replay spoof rejected: is_real=False, score={res_f2['score']:.4f}, label={res_f2['label']}")

        resp_blank = await client.post("/check-liveness", files={"image": ("blank.jpg", blank_jpeg_bytes, "image/jpeg")})
        assert resp_blank.status_code == 200
        res_blank = resp_blank.json()
        assert res_blank["is_real"] is False
        assert res_blank["status"] == "no_face"
        print("  [PASS] Blank non-face image: status='no_face', is_real=False")

        # ----------------------------------------------------------------------
        # Test 2: Spoof Registration Rejection (Print Photo)
        # ----------------------------------------------------------------------
        print("\n[Test 2] Testing registration rejection of printed photograph (POST /upload)...")
        files_spoof = {"image": ("spoof.jpg", fake_print_bytes, "image/jpeg")}
        data_spoof = {
            "user_id": user_spoof,
            "full_name": "Spoof Attacker",
            "email": "attacker@example.com",
            "role": "Fraudster",
            "phone_number": "+1 555-0199"
        }
        resp_reg_spoof = await client.post("/upload", data=data_spoof, files=files_spoof)
        print(f"  - Status Code: {resp_reg_spoof.status_code}")
        print(f"  - Response JSON: {resp_reg_spoof.json()}")
        assert resp_reg_spoof.status_code == 400
        err_detail = resp_reg_spoof.json()["detail"]
        assert err_detail["error"] == "liveness_failed"
        assert err_detail["liveness"]["is_real"] is False

        # Verify integrity: zero write to DB, zero FAISS update, zero saved image
        assert get_user_embedding(user_spoof) is None, "Spoof user written to DB!"
        faiss_mgr = get_faiss_manager()
        assert not faiss_mgr.contains_user(user_spoof), "Spoof user inserted into FAISS!"
        spoof_img_path = os.path.join(BASE_DIR, "cropped_faces", f"{user_spoof}_face.jpg")
        assert not os.path.exists(spoof_img_path), "Spoof image saved to disk!"
        print("  [PASS] Spoof registration rejected: PostgreSQL, FAISS, and filesystem remain completely unchanged.")

        # ----------------------------------------------------------------------
        # Test 3: Spoof Login Rejection (Screen Replay)
        # ----------------------------------------------------------------------
        print("\n[Test 3] Testing login rejection of screen replay attempt (POST /login)...")
        files_login_replay = {"image": ("replay.jpg", fake_replay_bytes, "image/jpeg")}
        resp_login_replay = await client.post("/login", files=files_login_replay)
        print(f"  - Status Code: {resp_login_replay.status_code}")
        print(f"  - Response JSON: {resp_login_replay.json()}")
        assert resp_login_replay.status_code == 400
        replay_detail = resp_login_replay.json()["detail"]
        assert replay_detail["error"] == "liveness_failed"
        assert replay_detail["authenticated"] is False
        print("  [PASS] Replay login strictly rejected with authenticated=False before FAISS.")

        # ----------------------------------------------------------------------
        # Test 4: Uncertain / Sub-Threshold Confidence Rejection
        # ----------------------------------------------------------------------
        print("\n[Test 4] Testing rejection of uncertain confidence face (score < threshold)...")
        # Test with threshold = 0.99999 (higher than genuine score 0.999906)
        liveness_svc = get_liveness_service()
        real_img = cv2.imread(real_face_path)
        strict_result = liveness_svc.verify_liveness(real_img, threshold=0.99999)
        assert strict_result["is_real"] is False
        assert strict_result["status"] == "uncertain"
        print(f"  [PASS] Face below strict threshold (0.99999 vs {strict_result['score']:.6f}) classified as uncertain and rejected.")

        # ----------------------------------------------------------------------
        # Test 5: Missing / Invalid Face in Registration & Login
        # ----------------------------------------------------------------------
        print("\n[Test 5] Testing registration & login rejection with blank / non-face frame...")
        resp_reg_blank = await client.post(
            "/upload",
            data={"user_id": f"blank_{timestamp}"},
            files={"image": ("blank.jpg", blank_jpeg_bytes, "image/jpeg")}
        )
        assert resp_reg_blank.status_code == 400
        assert resp_reg_blank.json()["detail"]["error"] == "no_face"

        resp_login_blank = await client.post(
            "/login",
            files={"image": ("blank.jpg", blank_jpeg_bytes, "image/jpeg")}
        )
        assert resp_login_blank.status_code == 400
        assert resp_login_blank.json()["detail"]["error"] == "no_face"
        print("  [PASS] Both registration and login reject missing face with HTTP 400 'no_face'.")

        # ----------------------------------------------------------------------
        # Test 6: Processing Order Enforcement via Mocks / Spies
        # ----------------------------------------------------------------------
        print("\n[Test 6] Verifying processing order: MediaPipe & FaceNet are NOT called on failed liveness...")
        with mock.patch("main.detector.detect") as spy_detector, \
             mock.patch("main.extract_face_embedding") as spy_embedder:

            fake_bgr = cv2.imread(fake_print_path)

            # Registration worker test with fake input
            reg_out = _process_registration_sync(
                clean_user_id=f"test_spy_{timestamp}",
                full_name="Spy Test",
                email="spy@test.com",
                role="Tester",
                phone_number="+1 555-0100",
                img=fake_bgr
            )
            assert reg_out["status"] == "liveness_failed"
            assert spy_detector.call_count == 0, "MediaPipe BlazeFace was called despite liveness failure!"
            assert spy_embedder.call_count == 0, "FaceNet embedder was called despite liveness failure!"
            print("  [PASS] In registration: MediaPipe detect calls = 0, FaceNet extract calls = 0 when liveness fails.")

            # Login worker test with fake input
            login_out = _process_login_sync(
                clean_user_id=None,
                img=fake_bgr,
                image_size_bytes=len(fake_print_bytes)
            )
            assert login_out["status"] == "liveness_failed"
            assert spy_detector.call_count == 0, "MediaPipe BlazeFace was called on login liveness failure!"
            assert spy_embedder.call_count == 0, "FaceNet embedder was called on login liveness failure!"
            print("  [PASS] In login: MediaPipe detect calls = 0, FaceNet extract calls = 0 when liveness fails.")

        # ----------------------------------------------------------------------
        # Test 7: Successful Registration of Genuine Face
        # ----------------------------------------------------------------------
        print("\n[Test 7] Testing successful registration with genuine live face...")
        files_genuine = {"image": ("live_face.jpg", real_jpeg_bytes, "image/jpeg")}
        data_genuine = {
            "user_id": user_genuine,
            "full_name": "Jane Doe",
            "email": f"jane_{timestamp}@example.com",
            "role": "Security Lead",
            "phone_number": "+1 555-0188"
        }
        resp_reg_genuine = await client.post("/upload", data=data_genuine, files=files_genuine)
        assert resp_reg_genuine.status_code == 200, f"Registration failed: {resp_reg_genuine.text}"
        res_genuine = resp_reg_genuine.json()
        assert res_genuine["status"] == "success"
        assert res_genuine["user_id"] == user_genuine
        assert res_genuine["liveness"]["is_real"] is True
        assert res_genuine["liveness"]["score"] >= 0.85
        assert res_genuine["embedding_dimension"] == 512

        # Verify PostgreSQL & FAISS synchronization
        user_record = get_user_embedding(user_genuine)
        assert user_record is not None, "User missing from PostgreSQL!"
        assert len(user_record["embedding"]) == 512
        assert faiss_mgr.contains_user(user_genuine), "User missing from FAISS index!"
        saved_crop = os.path.join(BASE_DIR, "cropped_faces", f"{user_genuine}_face.jpg")
        assert os.path.exists(saved_crop), "Cropped face image not saved on disk!"
        print(f"  [PASS] Genuine user '{user_genuine}' registered. PostgreSQL, FAISS, and filesystem synchronized.")

        # ----------------------------------------------------------------------
        # Test 8: Successful 1:N and 1:1 Login with Genuine Face
        # ----------------------------------------------------------------------
        print("\n[Test 8] Testing successful biometric login with genuine face...")
        # 8a: 1:N ID-free login
        files_login = {"image": ("login_live.jpg", real_jpeg_bytes, "image/jpeg")}
        resp_login_1n = await client.post("/login", files=files_login)
        assert resp_login_1n.status_code == 200, f"1:N login failed: {resp_login_1n.text}"
        login_1n_data = resp_login_1n.json()
        assert login_1n_data["status"] == "success"
        assert login_1n_data["authenticated"] is True
        assert login_1n_data["user_id"] == user_genuine
        assert login_1n_data["liveness"]["is_real"] is True
        assert login_1n_data["user_profile"]["full_name"] == "Jane Doe"
        print(f"  [PASS] 1:N Identification successful. Authenticated: '{login_1n_data['user_id']}', Profile: {login_1n_data['user_profile']['full_name']}")

        # 8b: 1:1 Claimed Identity verification
        data_11 = {"user_id": user_genuine}
        files_11 = {"image": ("login_live.jpg", real_jpeg_bytes, "image/jpeg")}
        resp_login_11 = await client.post("/login", data=data_11, files=files_11)
        assert resp_login_11.status_code == 200
        assert resp_login_11.json()["authenticated"] is True
        print(f"  [PASS] 1:1 Verification successful for claimed ID '{user_genuine}'.")

        # ----------------------------------------------------------------------
        # Test 9: Fail-Closed Security Verification
        # ----------------------------------------------------------------------
        print("\n[Test 9] Verifying fail-closed security when liveness model is unavailable...")
        with mock.patch.object(liveness_svc, "is_available", return_value=False):
            fc_reg = _process_registration_sync(
                clean_user_id=f"fail_closed_{timestamp}",
                full_name="Fail Test",
                email="fc@test.com",
                role="Tester",
                phone_number="+1 555-0100",
                img=real_img
            )
            assert fc_reg["status"] == "liveness_failed"
            assert fc_reg["liveness"]["status"] == "model_unavailable"

            fc_login = _process_login_sync(
                clean_user_id=None,
                img=real_img,
                image_size_bytes=len(real_jpeg_bytes)
            )
            assert fc_login["status"] == "liveness_failed"
            assert fc_login["authenticated"] is False
            print("  [PASS] Fail-closed verified: when liveness model is unavailable, both registration and login are rejected.")

        # ----------------------------------------------------------------------
        # Test 10: Cleanup & Post-Verification
        # ----------------------------------------------------------------------
        print("\n[Test 10] Cleaning up test records and verifying user deletion...")
        del_resp = await client.delete(
            f"/users/{user_genuine}",
            headers={"X-Admin-Key": os.getenv("ADMIN_API_KEY", "admin_secret_key_2026")}
        )
        assert del_resp.status_code == 200
        assert not faiss_mgr.contains_user(user_genuine)
        assert get_user_embedding(user_genuine) is None
        assert not os.path.exists(saved_crop)
        print(f"  [PASS] Cleaned up user '{user_genuine}' from PostgreSQL, FAISS, and filesystem.")

    print("\n" + "=" * 75)
    print("   ALL LIVENESS-FIRST PIPELINE TESTS COMPLETED & PASSED!   ")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(run_liveness_pipeline_tests())

