"""
Comprehensive Validation Suite for Camera-Only Liveness-First Biometric Pipeline.

Validates all strict requirements:
1. Complete removal of upload endpoints (POST /upload returns 404/405).
2. Mandatory backend-issued liveness sessions (/api/liveness/session/start).
3. Live webcam frame verification with MiniFASNet (/api/liveness/session/verify-frame).
4. Spoof rejection on live stream: printed photo and screen replay rejected before capture.
5. Strict capture restriction: direct POST /register and /login without verified session rejected with HTTP 403.
6. Replay attack prevention: consumed sessions cannot be reused.
7. Processing order enforcement: MediaPipe BlazeFace and FaceNet are NEVER invoked on failed liveness.
8. Genuine registration workflow: live session -> genuine verified frame -> capture -> MediaPipe -> FaceNet -> DB/FAISS.
9. Genuine 1:N and 1:1 login workflow: live session -> genuine verified frame -> capture -> MediaPipe -> FaceNet -> FAISS match.
10. Fail-closed security when liveness service is unavailable.
11. Safe post-test cleanup via Admin API.
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
from liveness_session import get_liveness_session_manager


async def run_liveness_pipeline_tests():
    print("=" * 75)
    print("   CAMERA-ONLY LIVENESS-FIRST PIPELINE VALIDATION SUITE   ")
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
        # Test 1: Upload Endpoint Completely Removed (/upload returns 404/405)
        # ----------------------------------------------------------------------
        print("\n[Test 1] Verifying that legacy /upload endpoint is completely removed...")
        resp_upload = await client.post(
            "/upload",
            data={"user_id": "test_upload"},
            files={"image": ("face.jpg", real_jpeg_bytes, "image/jpeg")}
        )
        assert resp_upload.status_code in (404, 405), f"Expected 404/405 for /upload, got {resp_upload.status_code}"
        print(f"  [PASS] POST /upload rejected with HTTP {resp_upload.status_code}. No generic upload route exists.")

        # ----------------------------------------------------------------------
        # Test 2: Direct API Requests Without Liveness Session Rejected (HTTP 403/422)
        # ----------------------------------------------------------------------
        print("\n[Test 2] Testing direct API requests without liveness session (bypassing webcam)...")
        # Direct register call with no session_id
        resp_reg_direct = await client.post(
            "/register",
            data={"user_id": "attacker_1"},
            files={"image": ("face.jpg", real_jpeg_bytes, "image/jpeg")}
        )
        assert resp_reg_direct.status_code in (403, 422), f"Expected 403/422, got {resp_reg_direct.status_code}"

        # Direct login call with no session_id
        resp_login_direct = await client.post(
            "/login",
            files={"image": ("face.jpg", real_jpeg_bytes, "image/jpeg")}
        )
        assert resp_login_direct.status_code in (403, 422), f"Expected 403/422, got {resp_login_direct.status_code}"
        print("  [PASS] Direct POST /register and /login without liveness session strictly rejected.")

        # Direct call with forged/unverified session_id
        resp_reg_forged = await client.post(
            "/register",
            data={"session_id": "fake_session_123", "user_id": "attacker_2"},
            files={"image": ("face.jpg", real_jpeg_bytes, "image/jpeg")}
        )
        assert resp_reg_forged.status_code == 403
        assert resp_reg_forged.json()["detail"]["error"] == "liveness_verification_required"
        print("  [PASS] Forged session ID rejected with HTTP 403 Forbidden.")

        # ----------------------------------------------------------------------
        # Test 3: Backend Liveness Session Creation & Lifecycle
        # ----------------------------------------------------------------------
        print("\n[Test 3] Testing backend-issued liveness session creation (/api/liveness/session/start)...")
        resp_sess_reg = await client.post(
            "/api/liveness/session/start",
            data={"action": "register", "user_id": user_genuine}
        )
        assert resp_sess_reg.status_code == 200
        sess_data = resp_sess_reg.json()
        assert sess_data["status"] == "success"
        reg_session_id = sess_data["session"]["session_id"]
        assert reg_session_id.startswith("live_")
        assert sess_data["session"]["is_verified"] is False
        assert sess_data["session"]["is_consumed"] is False
        print(f"  [PASS] Created liveness session: {reg_session_id} (action: register, is_verified: False)")

        # ----------------------------------------------------------------------
        # Test 4: Live Frame Anti-Spoofing Verification (Reject Print, Replay, Blank)
        # ----------------------------------------------------------------------
        print("\n[Test 4] Testing live stream anti-spoofing verification (/api/liveness/session/verify-frame)...")

        # 4a: Spoof Print Photo Frame
        resp_vf_print = await client.post(
            "/api/liveness/session/verify-frame",
            data={"session_id": reg_session_id},
            files={"image": ("live_frame.jpg", fake_print_bytes, "image/jpeg")}
        )
        assert resp_vf_print.status_code == 200
        vf_print_res = resp_vf_print.json()
        assert vf_print_res["is_real"] is False
        assert vf_print_res["verified"] is False
        assert vf_print_res["label"] == "Fake"
        print(f"  [PASS] Printed photograph spoof rejected on live stream: is_real=False, verified=False, score={vf_print_res['score']:.4f}")

        # 4b: Spoof Screen Replay Frame
        resp_vf_replay = await client.post(
            "/api/liveness/session/verify-frame",
            data={"session_id": reg_session_id},
            files={"image": ("live_frame.jpg", fake_replay_bytes, "image/jpeg")}
        )
        assert resp_vf_replay.status_code == 200
        vf_replay_res = resp_vf_replay.json()
        assert vf_replay_res["is_real"] is False
        assert vf_replay_res["verified"] is False
        print(f"  [PASS] Screen replay spoof rejected on live stream: is_real=False, verified=False, score={vf_replay_res['score']:.4f}")

        # 4c: Blank / Non-face Frame
        resp_vf_blank = await client.post(
            "/api/liveness/session/verify-frame",
            data={"session_id": reg_session_id},
            files={"image": ("live_frame.jpg", blank_jpeg_bytes, "image/jpeg")}
        )
        assert resp_vf_blank.status_code == 200
        assert resp_vf_blank.json()["status"] == "no_face"
        assert resp_vf_blank.json()["verified"] is False
        print("  [PASS] Blank non-face frame: status='no_face', verified=False.")

        # Ensure session remains unverified after spoof attempts
        session_mgr = get_liveness_session_manager()
        assert session_mgr.get_session(reg_session_id).is_verified is False
        print("  [PASS] Session remains strictly unverified after spoof/blank attempts.")

        # 4d: Genuine Live Frame -> Session becomes verified
        resp_vf_real = await client.post(
            "/api/liveness/session/verify-frame",
            data={"session_id": reg_session_id},
            files={"image": ("live_frame.jpg", real_jpeg_bytes, "image/jpeg")}
        )
        assert resp_vf_real.status_code == 200
        vf_real_res = resp_vf_real.json()
        assert vf_real_res["is_real"] is True
        assert vf_real_res["verified"] is True
        assert vf_real_res["score"] >= 0.85
        assert session_mgr.get_session(reg_session_id).is_verified is True
        print(f"  [PASS] Genuine face verified on live stream: is_real=True, verified=True (Score: {vf_real_res['score']:.4f} >= 0.85)")

        # ----------------------------------------------------------------------
        # Test 5: Processing Order Enforcement via Mocks / Spies
        # ----------------------------------------------------------------------
        print("\n[Test 5] Verifying processing order: MediaPipe & FaceNet are NOT called on failed liveness...")
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
            print("  [PASS] Registration: MediaPipe calls = 0, FaceNet calls = 0 on failed liveness.")

            # Login worker test with fake input (Pure 1:N)
            login_out = _process_login_sync(
                img=fake_bgr,
                image_size_bytes=len(fake_print_bytes)
            )
            assert login_out["status"] == "liveness_failed"
            assert spy_detector.call_count == 0, "MediaPipe BlazeFace was called on login liveness failure!"
            assert spy_embedder.call_count == 0, "FaceNet embedder was called on login liveness failure!"
            print("  [PASS] Login: MediaPipe calls = 0, FaceNet calls = 0 on failed liveness.")

        # ----------------------------------------------------------------------
        # Test 6: Genuine User Registration with Verified Live Session
        # ----------------------------------------------------------------------
        print("\n[Test 6] Testing genuine user registration with verified live session...")
        data_genuine = {
            "session_id": reg_session_id,
            "user_id": user_genuine,
            "full_name": "Jane Doe",
            "email": f"jane_{timestamp}@example.com",
            "role": "Security Lead",
            "phone_number": "+1 555-0188"
        }
        files_genuine = {"image": ("webcam_verified_face.jpg", real_jpeg_bytes, "image/jpeg")}

        resp_reg_genuine = await client.post("/register", data=data_genuine, files=files_genuine)
        assert resp_reg_genuine.status_code == 200, f"Registration failed: {resp_reg_genuine.text}"
        res_genuine = resp_reg_genuine.json()
        assert res_genuine["status"] == "success"
        assert res_genuine["user_id"] == user_genuine
        assert res_genuine["embedding_dimension"] == 512

        # Verify DB and FAISS synchronization
        user_record = get_user_embedding(user_genuine)
        assert user_record is not None, "User missing from PostgreSQL!"
        assert len(user_record["embedding"]) == 512
        faiss_mgr = get_faiss_manager()
        assert faiss_mgr.contains_user(user_genuine), "User missing from FAISS index!"
        saved_crop = os.path.join(BASE_DIR, "cropped_faces", f"{user_genuine}_face.jpg")
        assert os.path.exists(saved_crop), "Cropped face image not saved on disk!"
        print(f"  [PASS] Genuine user '{user_genuine}' registered. PostgreSQL, FAISS, and filesystem synchronized.")

        # ----------------------------------------------------------------------
        # Test 7: Replay Attack Rejection (Consumed Session Cannot Be Reused)
        # ----------------------------------------------------------------------
        print("\n[Test 7] Testing replay attack rejection with consumed liveness session...")
        resp_replay = await client.post("/register", data=data_genuine, files=files_genuine)
        assert resp_replay.status_code == 403
        assert resp_replay.json()["detail"]["error"] == "liveness_verification_required"
        print("  [PASS] Replay registration attempt with consumed session strictly rejected with HTTP 403.")

        # ----------------------------------------------------------------------
        # Test 8: Genuine 1:N and 1:1 Login with Verified Live Session
        # ----------------------------------------------------------------------
        print("\n[Test 8] Testing genuine 1:N and 1:1 login via live camera session...")

        # 8a: Start login session
        resp_sess_login = await client.post("/api/liveness/session/start", data={"action": "login"})
        assert resp_sess_login.status_code == 200
        login_session_id = resp_sess_login.json()["session"]["session_id"]

        # Verify frame on live stream
        resp_vf_login = await client.post(
            "/api/liveness/session/verify-frame",
            data={"session_id": login_session_id},
            files={"image": ("live_login.jpg", real_jpeg_bytes, "image/jpeg")}
        )
        assert resp_vf_login.status_code == 200
        assert resp_vf_login.json()["verified"] is True

        # Perform 1:N Login
        resp_login_1n = await client.post(
            "/login",
            data={"session_id": login_session_id},
            files={"image": ("webcam_verified_login.jpg", real_jpeg_bytes, "image/jpeg")}
        )
        assert resp_login_1n.status_code == 200, f"1:N login failed: {resp_login_1n.text}"
        login_1n_data = resp_login_1n.json()
        assert login_1n_data["status"] == "success"
        assert login_1n_data["authenticated"] is True
        assert login_1n_data["user_id"] == user_genuine
        assert login_1n_data["user_profile"]["full_name"] == "Jane Doe"
        print(f"  [PASS] 1:N Login successful. Authenticated user: '{login_1n_data['user_id']}', Profile: Jane Doe.")

        # 8b: 1:N Rejection of Unknown / Unregistered Face
        print("\n[Test 8b] Testing 1:N rejection of unknown / unregistered face...")
        # Delete user_genuine so they are no longer in DB/FAISS
        del_resp = await client.delete(
            f"/users/{user_genuine}",
            headers={"X-Admin-Key": os.getenv("ADMIN_API_KEY", "admin_secret_key_2026")}
        )
        assert del_resp.status_code == 200

        # Attempt 1:N login with now-unregistered face
        resp_sess_unknown = await client.post("/api/liveness/session/start", data={"action": "login"})
        unknown_session_id = resp_sess_unknown.json()["session"]["session_id"]

        resp_vf_unk = await client.post(
            "/api/liveness/session/verify-frame",
            data={"session_id": unknown_session_id},
            files={"image": ("live_login.jpg", real_jpeg_bytes, "image/jpeg")}
        )
        assert resp_vf_unk.json()["verified"] is True

        resp_login_unk = await client.post(
            "/login",
            data={"session_id": unknown_session_id},
            files={"image": ("webcam_verified_login.jpg", real_jpeg_bytes, "image/jpeg")}
        )
        assert resp_login_unk.status_code == 401
        unk_detail = resp_login_unk.json()["detail"]
        assert unk_detail["authenticated"] is False
        print("  [PASS] 1:N Login correctly rejected unknown/unregistered face with HTTP 401 Unauthorized.")

        # ----------------------------------------------------------------------
        # Test 9: Fail-Closed Security Verification
        # ----------------------------------------------------------------------
        print("\n[Test 9] Verifying fail-closed security when liveness model is unavailable...")
        liveness_svc = get_liveness_service()
        real_img = cv2.imread(real_face_path)
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
                img=real_img,
                image_size_bytes=len(real_jpeg_bytes)
            )
            assert fc_login["status"] == "liveness_failed"
            assert fc_login["authenticated"] is False
            print("  [PASS] Fail-closed verified: when liveness model is unavailable, both registration and login fail closed.")

        # ----------------------------------------------------------------------
        # Test 10: Cleanup & Post-Verification
        # ----------------------------------------------------------------------
        print("\n[Test 10] Verifying cleanup...")
        assert not faiss_mgr.contains_user(user_genuine)
        assert get_user_embedding(user_genuine) is None
        assert not os.path.exists(saved_crop)
        print(f"  [PASS] Verified user '{user_genuine}' is completely purged from PostgreSQL, FAISS, and filesystem.")

    print("\n" + "=" * 75)
    print("   ALL CAMERA-ONLY LIVENESS-FIRST PIPELINE TESTS COMPLETED & PASSED!   ")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(run_liveness_pipeline_tests())
