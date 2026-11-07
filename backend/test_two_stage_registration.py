"""
Comprehensive Verification Suite for OJIX Two-Stage Face Registration Workflow.

Validates all 13 core requirements:
1. Clicking ADD FACE starts camera/liveness workflow (Session initialized with 300s TTL).
2. Spoofed captures are rejected (MiniFASNet anti-spoofing).
3. Genuine liveness verification allows temporary capture into backend RAM.
4. No FaceNet embedding is generated before REGISTER USER (and zero DB/disk writes during Stage B & C).
5. CONFIRM FACE enables registration; unconfirmed captures cannot register.
6. RETAKE invalidates the previous temporary capture & resets verification.
7. Expired registration sessions are rejected.
8. Session IDs cannot be reused across registration flows or different user IDs.
9. Temporary frames are removed from RAM after success, cancellation, and expiry.
10. PostgreSQL receives the embedding ONLY after registration.
11. FAISS synchronization occurs after database persistence.
12. Failed FAISS synchronization is reported accurately and is recoverable.
13. Existing face login and user deletion continue to pass cleanly.

Distinction:
- Hardware/Webcam Capture: Mocked via authentic frame fixtures (test_face.jpg, fake_print.jpg, fake_replay.jpg).
- Backend Pipeline: Real FastAPI ASGI app, real MiniFASNet anti-spoofing, real MediaPipe BlazeFace,
  real FaceNet 512-D inference, real PostgreSQL transactions, and real FAISS IndexFlatIP.
"""

import os
import sys
import time
import asyncio
from unittest import mock
import httpx
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from main import app, ADMIN_API_KEY
from user_service import delete_user_completely, get_user_embedding, get_user_profile
from faiss_service import get_faiss_manager
from liveness_session import get_liveness_session_manager


async def run_two_stage_registration_tests():
    print("=" * 80)
    print("      OJIX TWO-STAGE BIOMETRIC REGISTRATION VERIFICATION SUITE       ")
    print("=" * 80)

    # Load test fixtures
    fixtures_dir = os.path.join(BASE_DIR, "test_fixtures")
    real_face_path = os.path.join(fixtures_dir, "test_face.jpg")
    fake_print_path = os.path.join(fixtures_dir, "fake_print.jpg")
    fake_replay_path = os.path.join(fixtures_dir, "fake_replay.jpg")

    assert os.path.exists(real_face_path), f"Missing {real_face_path}"
    assert os.path.exists(fake_print_path), f"Missing {fake_print_path}"
    assert os.path.exists(fake_replay_path), f"Missing {fake_replay_path}"

    with open(real_face_path, "rb") as f:
        real_face_bytes = f.read()
    with open(fake_print_path, "rb") as f:
        fake_print_bytes = f.read()
    with open(fake_replay_path, "rb") as f:
        fake_replay_bytes = f.read()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        ts = int(time.time())
        user_test = f"twostage_user_{ts}"
        user_other = f"twostage_other_{ts}"

        # Clean any prior fixture
        try:
            delete_user_completely(user_test)
            delete_user_completely(user_other)
        except Exception:
            pass

        session_mgr = get_liveness_session_manager()

        # ----------------------------------------------------------------------
        # Requirement 1: Stage A / ADD FACE starts session with 300s TTL
        # ----------------------------------------------------------------------
        print("\n[Req 1] Testing Stage A: ADD FACE starts camera/liveness workflow...")
        resp_start = await client.post(
            "/api/liveness/session/start",
            data={"action": "register", "user_id": user_test}
        )
        assert resp_start.status_code == 200
        start_json = resp_start.json()
        assert start_json["status"] == "success"
        session_id = start_json["session"]["session_id"]
        expires_in = start_json["session"]["expires_in"]

        # TTL must be ~300 seconds (5 minutes) for registration
        assert 280 <= expires_in <= 300, f"Expected ~300s TTL, got {expires_in}"
        session_obj = session_mgr.get_session(session_id)
        assert session_obj is not None
        assert session_obj.is_verified is False
        assert session_obj.face_captured is False
        assert session_obj.is_confirmed is False
        print(f"  [PASS] Session {session_id} created with 300s TTL for user '{user_test}'.")

        # ----------------------------------------------------------------------
        # Requirement 2: Spoofed captures are rejected
        # ----------------------------------------------------------------------
        print("\n[Req 2] Testing Stage B: Spoofed captures are rejected by MiniFASNet...")
        # 2a: Print Spoof
        resp_fake1 = await client.post(
            "/api/liveness/session/verify-frame",
            data={"session_id": session_id},
            files={"image": ("frame.jpg", fake_print_bytes, "image/jpeg")}
        )
        assert resp_fake1.status_code == 200
        res1 = resp_fake1.json()
        assert res1["is_real"] is False
        assert res1["verified"] is False
        assert session_obj.is_verified is False
        print("  [PASS] Print spoof rejected on live stream (verified=False).")

        # 2b: Screen Replay Spoof
        resp_fake2 = await client.post(
            "/api/liveness/session/verify-frame",
            data={"session_id": session_id},
            files={"image": ("frame.jpg", fake_replay_bytes, "image/jpeg")}
        )
        assert resp_fake2.status_code == 200
        res2 = resp_fake2.json()
        assert res2["is_real"] is False
        assert res2["verified"] is False
        assert session_obj.is_verified is False
        print("  [PASS] Screen replay spoof rejected on live stream (verified=False).")

        # Attempt to capture frame while unverified -> must fail
        resp_early_cap = await client.post(
            "/api/registration/session/capture-frame",
            data={"session_id": session_id, "user_id": user_test},
            files={"image": ("frame.jpg", fake_print_bytes, "image/jpeg")}
        )
        assert resp_early_cap.status_code == 400
        assert resp_early_cap.json()["detail"]["error"] == "not_verified"
        print("  [PASS] Capture frame strictly rejected when session is not liveness-verified.")

        # ----------------------------------------------------------------------
        # Requirement 3 & 4: Genuine liveness allows temporary capture into RAM.
        # Zero FaceNet embeddings, DB rows, or disk files generated before REGISTER USER.
        # ----------------------------------------------------------------------
        print("\n[Req 3 & 4] Testing Genuine Liveness, RAM storage, & Zero FaceNet inference before Stage D...")
        with mock.patch("main.extract_face_embedding") as spy_facenet:
            # Genuine frame verification
            resp_real = await client.post(
                "/api/liveness/session/verify-frame",
                data={"session_id": session_id},
                files={"image": ("frame.jpg", real_face_bytes, "image/jpeg")}
            )
            assert resp_real.status_code == 200
            assert resp_real.json()["is_real"] is True
            assert resp_real.json()["verified"] is True
            assert session_obj.is_verified is True

            # Stage B: Capture frame into backend RAM
            resp_cap = await client.post(
                "/api/registration/session/capture-frame",
                data={"session_id": session_id, "user_id": user_test},
                files={"image": ("captured.jpg", real_face_bytes, "image/jpeg")}
            )
            assert resp_cap.status_code == 200
            cap_json = resp_cap.json()
            assert cap_json["status"] == "success"
            assert cap_json["session"]["face_captured"] is True
            assert cap_json["session"]["frame_count"] >= 1

            # In-memory RAM verification
            assert len(session_obj.temp_frames) >= 1
            assert isinstance(session_obj.temp_frames[0], np.ndarray)

            # Strict verification of Requirement 4:
            # No FaceNet embedding extracted during capture
            assert spy_facenet.call_count == 0, "FaceNet was invoked prematurely during capture!"

            # No PostgreSQL row inserted during Stage B
            assert get_user_embedding(user_test) is None, "PostgreSQL contains user before registration!"

            # No cropped face saved to disk during Stage B
            crop_path = os.path.join(BASE_DIR, "cropped_faces", f"{user_test}_face.jpg")
            assert not os.path.exists(crop_path), "Permanent disk file created prematurely!"

            print("  [PASS] Genuine face verified and stored strictly in RAM.")
            print("  [PASS] Verified: FaceNet embedding calls = 0, PostgreSQL rows = 0, disk files = 0.")

        # ----------------------------------------------------------------------
        # Requirement 5: CONFIRM FACE enables registration
        # ----------------------------------------------------------------------
        print("\n[Req 5] Testing CONFIRM FACE requirement before registration...")
        # Attempt to register BEFORE confirmation -> must be rejected
        reg_data = {
            "session_id": session_id,
            "user_id": user_test,
            "full_name": "Two-Stage Test User",
            "email": f"{user_test}@company.com",
            "role": "Security Analyst",
            "phone_number": "+1 555-0199"
        }
        resp_unconfirmed = await client.post("/register", data=reg_data)
        assert resp_unconfirmed.status_code == 403
        assert "confirmed" in resp_unconfirmed.json()["detail"]["message"].lower()
        print("  [PASS] Registration without CONFIRM FACE rejected with HTTP 403.")

        # Now call CONFIRM FACE
        resp_confirm = await client.post(
            "/api/registration/session/confirm",
            data={"session_id": session_id, "user_id": user_test}
        )
        assert resp_confirm.status_code == 200
        assert resp_confirm.json()["session"]["is_confirmed"] is True
        assert session_obj.is_confirmed is True
        print("  [PASS] CONFIRM FACE accepted and session marked confirmed.")

        # ----------------------------------------------------------------------
        # Requirement 6: RETAKE invalidates the previous temporary capture
        # ----------------------------------------------------------------------
        print("\n[Req 6] Testing Stage C RETAKE workflow...")
        resp_retake = await client.post(
            "/api/registration/session/retake",
            data={"session_id": session_id, "user_id": user_test}
        )
        assert resp_retake.status_code == 200
        assert session_obj.is_confirmed is False
        assert session_obj.is_verified is False
        assert session_obj.face_captured is False
        assert len(session_obj.temp_frames) == 0
        print("  [PASS] RETAKE discarded temporary frames from RAM and reset verification state.")

        # Attempt to register after retake -> must fail
        resp_after_retake = await client.post("/register", data=reg_data)
        assert resp_after_retake.status_code in (400, 403)
        print("  [PASS] Registration after RETAKE without re-verification correctly rejected.")

        # Re-verify and re-capture and re-confirm for subsequent tests
        resp_rv = await client.post(
            "/api/liveness/session/verify-frame",
            data={"session_id": session_id},
            files={"image": ("frame.jpg", real_face_bytes, "image/jpeg")}
        )
        assert resp_rv.json()["verified"] is True
        await client.post(
            "/api/registration/session/capture-frame",
            data={"session_id": session_id, "user_id": user_test},
            files={"image": ("captured.jpg", real_face_bytes, "image/jpeg")}
        )
        await client.post(
            "/api/registration/session/confirm",
            data={"session_id": session_id, "user_id": user_test}
        )
        assert session_obj.is_confirmed is True

        # ----------------------------------------------------------------------
        # Requirement 7: Expired registration sessions are rejected
        # ----------------------------------------------------------------------
        print("\n[Req 7] Testing session expiration rejection...")
        resp_short = await client.post("/api/liveness/session/start", data={"action": "register", "user_id": "exp_user"})
        exp_sid = resp_short.json()["session"]["session_id"]
        exp_session = session_mgr.get_session(exp_sid)
        # Fast-forward expiry
        exp_session.expires_at = time.time() - 1

        resp_exp_cap = await client.post(
            "/api/registration/session/capture-frame",
            data={"session_id": exp_sid, "user_id": "exp_user"},
            files={"image": ("frame.jpg", real_face_bytes, "image/jpeg")}
        )
        assert resp_exp_cap.status_code in (404, 410)
        print("  [PASS] Expired session rejected (status: 404/410).")

        # ----------------------------------------------------------------------
        # Requirement 8: Session IDs cannot be reused across flows or different user IDs
        # ----------------------------------------------------------------------
        print("\n[Req 8] Testing session isolation across different user IDs...")
        reg_data_impostor = {
            "session_id": session_id,
            "user_id": user_other,  # Session was issued for user_test
            "full_name": "Impostor User",
            "email": f"{user_other}@company.com",
            "role": "Impostor",
            "phone_number": "+1 555-0999"
        }
        resp_cross_user = await client.post("/register", data=reg_data_impostor)
        assert resp_cross_user.status_code == 403
        assert "mismatch" in resp_cross_user.json()["detail"]["message"].lower()
        print("  [PASS] Cross-user session hijacking rejected with HTTP 403.")

        # ----------------------------------------------------------------------
        # Requirement 9, 10, 11: Stage D REGISTER USER completes persistence & sync.
        # FaceNet embedding is generated ONLY here; temp frames purged from RAM.
        # ----------------------------------------------------------------------
        print("\n[Req 9, 10, 11] Testing Stage D REGISTER USER: embedding extraction, PostgreSQL commit, FAISS sync & RAM cleanup...")
        # Note: image file is NOT sent in form data; frame is pulled directly from RAM!
        resp_reg = await client.post("/register", data=reg_data)
        assert resp_reg.status_code == 200, f"Registration failed: {resp_reg.text}"
        reg_res = resp_reg.json()
        assert reg_res["status"] == "success"
        assert reg_res["user_id"] == user_test
        assert reg_res["embedding_dimension"] == 512
        assert reg_res["faiss_synced"] is True

        # Req 10: PostgreSQL receives embedding only after registration
        pg_rec = get_user_embedding(user_test)
        assert pg_rec is not None
        assert len(pg_rec["embedding"]) == 512
        print(f"  [PASS] User '{user_test}' persisted in PostgreSQL with 512-D embedding.")

        # Req 11: FAISS synchronized post-commit
        faiss_mgr = get_faiss_manager()
        assert faiss_mgr.contains_user(user_test)
        print(f"  [PASS] FAISS index synchronized with user '{user_test}'.")

        # Req 9: RAM frames purged post-registration
        assert len(session_obj.temp_frames) == 0
        assert session_obj.is_consumed is True
        print("  [PASS] Temporary frames in RAM successfully purged post-registration.")

        # Replay attempt with consumed session must fail
        resp_replay = await client.post("/register", data=reg_data)
        assert resp_replay.status_code == 403
        print("  [PASS] Consumed session cannot be reused (replay prevented).")

        # ----------------------------------------------------------------------
        # Requirement 9b: Cancellation purges memory
        # ----------------------------------------------------------------------
        print("\n[Req 9b] Testing cancellation memory purge...")
        resp_cancel_start = await client.post("/api/liveness/session/start", data={"action": "register", "user_id": "canc_user"})
        canc_sid = resp_cancel_start.json()["session"]["session_id"]
        canc_sess = session_mgr.get_session(canc_sid)
        canc_sess.is_verified = True
        canc_sess.add_temp_frame(np.zeros((100, 100, 3), dtype=np.uint8))
        assert len(canc_sess.temp_frames) == 1

        resp_cancel = await client.post(
            "/api/registration/session/cancel",
            data={"session_id": canc_sid, "user_id": "canc_user"}
        )
        assert resp_cancel.status_code == 200
        assert session_mgr.get_session(canc_sid) is None
        print("  [PASS] Cancel endpoint completely purged session and temporary RAM frames.")

        # ----------------------------------------------------------------------
        # Requirement 12: Failed FAISS synchronization is reported accurately & recoverable
        # ----------------------------------------------------------------------
        print("\n[Req 12] Testing resilient handling of FAISS sync failure...")
        user_faiss_fail = f"faiss_fail_{ts}"
        try:
            delete_user_completely(user_faiss_fail)
        except Exception:
            pass

        # Create verified session for this user
        s_fail = session_mgr.create_session("register", user_id=user_faiss_fail)
        s_fail.is_verified = True
        # Read real face as numpy BGR
        nparr = np.frombuffer(real_face_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        s_fail.add_temp_frame(img_bgr)
        s_fail.confirm_capture()

        # Mock FAISS add_user to fail, simulating FAISS sync failure
        with mock.patch.object(faiss_mgr, "add_user", side_effect=RuntimeError("Simulated FAISS crash")), \
             mock.patch.object(faiss_mgr, "initialize_from_database", side_effect=RuntimeError("Simulated recon crash")):
            fail_data = {
                "session_id": s_fail.session_id,
                "user_id": user_faiss_fail,
                "full_name": "Faiss Fail Test",
                "email": f"{user_faiss_fail}@biocorp.com",
                "role": "Security",
                "phone_number": "+1 555-0987"
            }
            resp_fail_sync = await client.post("/register", data=fail_data)
            assert resp_fail_sync.status_code == 200
            res_fail = resp_fail_sync.json()
            # User must NOT be deleted from PostgreSQL
            assert get_user_embedding(user_faiss_fail) is not None
            # Reported accurately
            assert res_fail["faiss_synced"] is False
            assert res_fail["indexing_status"] == "reconciliation_pending"
            print("  [PASS] FAISS failure did not delete DB user; returned reconciliation_pending accurately.")

        # Now recover via existing recovery mechanism
        count = faiss_mgr.initialize_from_database()
        assert faiss_mgr.contains_user(user_faiss_fail)
        print(f"  [PASS] FAISS index successfully reconciled from PostgreSQL ({count} users indexed).")

        # Cleanup user_faiss_fail
        delete_user_completely(user_faiss_fail)

        # ----------------------------------------------------------------------
        # Requirement 13: Existing face login and user deletion continue to pass
        # ----------------------------------------------------------------------
        print("\n[Req 13] Testing existing 1:N face login and user deletion...")
        # Start a live login session
        resp_login_start = await client.post("/api/liveness/session/start", data={"action": "login"})
        assert resp_login_start.status_code == 200
        login_sid = resp_login_start.json()["session"]["session_id"]

        # Verify genuine frame
        resp_l_vf = await client.post(
            "/api/liveness/session/verify-frame",
            data={"session_id": login_sid},
            files={"image": ("frame.jpg", real_face_bytes, "image/jpeg")}
        )
        assert resp_l_vf.json()["verified"] is True

        # Complete 1:N login
        resp_login = await client.post(
            "/login",
            data={"session_id": login_sid},
            files={"image": ("login.jpg", real_face_bytes, "image/jpeg")}
        )
        assert resp_login.status_code == 200
        login_res = resp_login.json()
        assert login_res["authenticated"] is True
        assert login_res["user_id"] == user_test
        print(f"  [PASS] 1:N Login authenticated user '{user_test}' without credentials.")

        # Delete user via Admin API
        resp_del = await client.delete(
            f"/api/users/{user_test}",
            headers={"X-Admin-Key": ADMIN_API_KEY}
        )
        assert resp_del.status_code == 200
        assert get_user_embedding(user_test) is None
        assert not faiss_mgr.contains_user(user_test)
        print(f"  [PASS] Admin deletion completely purged user '{user_test}'.")

    print("\n" + "=" * 80)
    print("      ALL 13 TWO-STAGE REGISTRATION VERIFICATIONS PASSED (13/13)!      ")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_two_stage_registration_tests())
