"""
Verification Suite for OJIX Temporary RAM Session Cleanup & Global Memory Protection.

Validates all 11 core requirements:
1. Expired sessions are removed by periodic cleanup even when no user requests arrive.
2. Temporary NumPy frames are cleared when sessions expire.
3. The cleanup task starts exactly once (idempotent startup).
4. The cleanup task shuts down cleanly.
5. Registration sessions can be created until configured limit is reached.
6. The next registration attempt receives HTTP 429 Too Many Requests.
7. Login sessions do not count toward the registration-session limit.
8. Simultaneous session creation cannot exceed the configured limit (thread-safety / concurrency).
9. Expired sessions free capacity for new registration attempts.
10. Existing session ownership and isolation remain enforced.
11. FastAPI lifespan startup and shutdown integration works cleanly.
"""

import os
import sys
import time
import asyncio
import httpx
import numpy as np
from starlette.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from main import app
from liveness_session import (
    get_liveness_session_manager,
    LivenessSessionManager,
    RegistrationCapacityExceededError,
    MAX_CONCURRENT_REGISTRATION_SESSIONS
)


async def run_ram_cleanup_and_capacity_tests():
    print("=" * 80)
    print("   OJIX TEMPORARY RAM SESSION CLEANUP & MEMORY PROTECTION VERIFICATION   ")
    print("=" * 80)

    session_mgr = get_liveness_session_manager()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:

        # ----------------------------------------------------------------------
        # Test 1 & 2: Periodic background cleanup removes expired sessions and
        # clears NumPy frames without user requests
        # ----------------------------------------------------------------------
        print("\n[Test 1 & 2] Testing background periodic cleanup of expired sessions & frame purge...")
        # Reset any leftover sessions in singleton
        with session_mgr._lock:
            session_mgr._sessions.clear()

        # Create session with 1s TTL and attach a dummy RAM frame
        short_session = session_mgr.create_session(action="register", user_id="test_exp_user", ttl_seconds=1)
        dummy_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        short_session.add_temp_frame(dummy_frame)
        assert len(short_session.temp_frames) == 1
        assert short_session.face_captured is True

        sid = short_session.session_id
        assert sid in session_mgr._sessions

        # Start background cleanup task with 0.2s interval
        task = session_mgr.start_background_cleanup(interval_seconds=0.2)
        assert task is not None
        assert not task.done()

        # Wait 1.5s for session to expire and background task to run WITHOUT any requests
        await asyncio.sleep(1.5)

        # Verify session has been removed from _sessions
        assert sid not in session_mgr._sessions, "Expired session was not purged by background cleanup!"
        # Verify temporary frames were cleared on the object
        assert len(short_session.temp_frames) == 0, "NumPy frames were not purged on expiry!"
        assert short_session.face_captured is False

        # Stop background cleanup
        await session_mgr.stop_background_cleanup()
        assert session_mgr.get_cleanup_task() is None
        print("  [PASS] Background task purged expired session from RAM and cleared NumPy frames without user requests.")

        # ----------------------------------------------------------------------
        # Test 3 & 4: Cleanup task idempotency and clean shutdown
        # ----------------------------------------------------------------------
        print("\n[Test 3 & 4] Testing cleanup task startup idempotency & clean shutdown...")
        t1 = session_mgr.start_background_cleanup(interval_seconds=60)
        t2 = session_mgr.start_background_cleanup(interval_seconds=60)
        assert t1 is t2, "Multiple cleanup tasks were created instead of reusing existing one!"
        assert not t1.done()
        print("  [PASS] Startup is idempotent: multiple calls return the same active task.")

        await session_mgr.stop_background_cleanup()
        assert t1.cancelled() or t1.done()
        assert session_mgr.get_cleanup_task() is None
        print("  [PASS] Shutdown is clean: task cancelled and reference cleared.")

        # ----------------------------------------------------------------------
        # Test 5 & 6: Configurable limit enforcement & HTTP 429 response
        # ----------------------------------------------------------------------
        print("\n[Test 5 & 6] Testing registration capacity limit and HTTP 429 response...")
        with session_mgr._lock:
            session_mgr._sessions.clear()

        # Set a test capacity limit of 5
        orig_limit = session_mgr.max_concurrent_registration_sessions
        session_mgr.max_concurrent_registration_sessions = 5

        try:
            created_sids = []
            for i in range(5):
                resp = await client.post(
                    "/api/liveness/session/start",
                    data={"action": "register", "user_id": f"capacity_user_{i}"}
                )
                assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
                created_sids.append(resp.json()["session"]["session_id"])

            assert len(created_sids) == 5
            assert session_mgr.get_active_registration_count() == 5
            print(f"  [PASS] Successfully created {len(created_sids)} sessions up to capacity limit.")

            # Attempt 6th session -> must return HTTP 429 Too Many Requests
            resp_overflow = await client.post(
                "/api/liveness/session/start",
                data={"action": "register", "user_id": "capacity_user_overflow"}
            )
            assert resp_overflow.status_code == 429, f"Expected 429, got {resp_overflow.status_code}"
            err_json = resp_overflow.json()
            assert err_json["detail"]["error"] == "registration_capacity_exceeded"
            assert "capacity" in err_json["detail"]["message"].lower()
            assert "5" in err_json["detail"]["message"]
            print(f"  [PASS] Overflow attempt rejected with HTTP 429: {err_json['detail']['message']}")

            # ------------------------------------------------------------------
            # Test 7: Login sessions do not count toward registration limit
            # ------------------------------------------------------------------
            print("\n[Test 7] Testing that login sessions do NOT count toward registration limit...")
            # We are at 5/5 registration capacity; login session must still succeed!
            resp_login = await client.post(
                "/api/liveness/session/start",
                data={"action": "login"}
            )
            assert resp_login.status_code == 200, f"Login session rejected at registration cap: {resp_login.text}"
            assert resp_login.json()["session"]["action"] == "login"
            assert session_mgr.get_active_login_count() >= 1
            assert session_mgr.get_active_registration_count() == 5
            print("  [PASS] Login session succeeded and did not increment registration counter.")

            # ------------------------------------------------------------------
            # Test 8: Simultaneous session creation cannot exceed limit
            # ------------------------------------------------------------------
            print("\n[Test 8] Testing concurrent creation race condition safety...")
            with session_mgr._lock:
                session_mgr._sessions.clear()

            session_mgr.max_concurrent_registration_sessions = 8

            async def create_one(idx: int):
                return await client.post(
                    "/api/liveness/session/start",
                    data={"action": "register", "user_id": f"concurrent_user_{idx}"}
                )

            # Fire 20 requests concurrently
            responses = await asyncio.gather(*(create_one(i) for i in range(20)))
            successes = [r for r in responses if r.status_code == 200]
            throttled = [r for r in responses if r.status_code == 429]

            assert len(successes) == 8, f"Expected exactly 8 successes, got {len(successes)}"
            assert len(throttled) == 12, f"Expected exactly 12 throttled (429), got {len(throttled)}"
            assert session_mgr.get_active_registration_count() == 8
            print(f"  [PASS] Concurrency check: Exactly 8 succeeded and 12 throttled out of 20 concurrent requests.")

            # ------------------------------------------------------------------
            # Test 9: Expired sessions free capacity for new registrations
            # ------------------------------------------------------------------
            print("\n[Test 9] Testing that expired sessions automatically free capacity...")
            # Expire 3 sessions manually
            with session_mgr._lock:
                count = 0
                for s in session_mgr._sessions.values():
                    if s.action == "register" and count < 3:
                        s.expires_at = time.time() - 1
                        count += 1

            # Capacity should now allow 3 new sessions
            for i in range(3):
                resp_freed = await client.post(
                    "/api/liveness/session/start",
                    data={"action": "register", "user_id": f"freed_user_{i}"}
                )
                assert resp_freed.status_code == 200, f"Expected 200 after expiry, got {resp_freed.status_code}"

            # Next one must be rejected (back at 8/8)
            resp_overflow2 = await client.post(
                "/api/liveness/session/start",
                data={"action": "register", "user_id": "freed_user_overflow"}
            )
            assert resp_overflow2.status_code == 429
            print("  [PASS] Expired sessions freed capacity and limit was re-enforced at 8.")

        finally:
            session_mgr.max_concurrent_registration_sessions = orig_limit

        # ----------------------------------------------------------------------
        # Test 10: Session ownership and isolation remain enforced
        # ----------------------------------------------------------------------
        print("\n[Test 10] Testing session ownership and isolation remain intact...")
        with session_mgr._lock:
            session_mgr._sessions.clear()

        resp_owner = await client.post(
            "/api/liveness/session/start",
            data={"action": "register", "user_id": "owner_alice"}
        )
        owner_sid = resp_owner.json()["session"]["session_id"]
        sess_obj = session_mgr.get_session(owner_sid)
        sess_obj.is_verified = True

        # Impostor attempt to capture frame with alice's session
        resp_impostor_cap = await client.post(
            "/api/registration/session/capture-frame",
            data={"session_id": owner_sid, "user_id": "impostor_bob"}
        )
        assert resp_impostor_cap.status_code == 403
        assert "user_mismatch" in resp_impostor_cap.json()["detail"]["error"]

        # Alice's legitimate capture attempt
        dummy_bgr = np.ones((60, 60, 3), dtype=np.uint8) * 200
        sess_obj.add_temp_frame(dummy_bgr)

        # Impostor attempt to confirm
        resp_impostor_conf = await client.post(
            "/api/registration/session/confirm",
            data={"session_id": owner_sid, "user_id": "impostor_bob"}
        )
        assert resp_impostor_conf.status_code == 400
        assert "unauthorized" in resp_impostor_conf.json()["detail"]["message"].lower()

        # Owner legitimate confirm
        resp_owner_conf = await client.post(
            "/api/registration/session/confirm",
            data={"session_id": owner_sid, "user_id": "owner_alice"}
        )
        assert resp_owner_conf.status_code == 200
        print("  [PASS] Session ownership strictly preserved across all operations.")

        # ----------------------------------------------------------------------
        # Test 11: FastAPI lifespan integration
        # ----------------------------------------------------------------------
        print("\n[Test 11] Testing FastAPI lifespan context manager integration...")
        with TestClient(app) as test_c:
            # During lifespan, the background task should be running
            mgr = get_liveness_session_manager()
            active_task = mgr.get_cleanup_task()
            assert active_task is not None, "Cleanup task was not started by lifespan!"
            assert not active_task.done(), "Cleanup task is not running during lifespan!"
            resp_ping = test_c.get("/")
            assert resp_ping.status_code == 200

        # After TestClient exits, shutdown must have run
        assert mgr.get_cleanup_task() is None, "Cleanup task was not stopped during lifespan shutdown!"
        print("  [PASS] Lifespan integration verified: started on startup, cleanly stopped on shutdown.")

    print("\n" + "=" * 80)
    print("   ALL 11 TEMPORARY RAM CLEANUP & MEMORY PROTECTION TESTS PASSED!   ")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_ram_cleanup_and_capacity_tests())

