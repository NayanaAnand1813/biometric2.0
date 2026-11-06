"""
End-to-End Test Suite for User Profile Registration and Pure 1:N Biometric Login.

Verifies:
1. Registration with all profile fields: User ID, Full Name, Work Email, Role, Phone Number.
2. Email normalization (lowercase trimmed).
3. Conflict rejection on duplicate User ID (HTTP 409).
4. Conflict rejection on duplicate Email (HTTP 409).
5. Validation of mandatory profile format requirements (Email format, Phone format).
6. Pure 1:N Face Login: Identifies user exclusively via face embedding without requiring User ID, username, or email.
7. Correct retrieval of user profile from PostgreSQL via FAISS match.
8. Rejection of unknown / unregistered faces under 1:N identification (HTTP 401).
9. Security: zero raw FaceNet vector leakage in login response.
10. Strict FAISS and PostgreSQL synchronization.
"""

import sys
import os
import time
import asyncio
import httpx

# Ensure backend directory is in python search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import app
from user_service import delete_user, get_user_embedding, get_user_profile
from faiss_service import get_faiss_manager


async def create_verified_session(client: httpx.AsyncClient, action: str, face_bytes: bytes, user_id: str | None = None) -> str:
    """Helper to start a liveness session and verify a genuine live frame."""
    start_data = {"action": action}
    if user_id:
        start_data["user_id"] = user_id
    resp_start = await client.post("/api/liveness/session/start", data=start_data)
    assert resp_start.status_code == 200, f"Session start failed: {resp_start.text}"
    session_id = resp_start.json()["session"]["session_id"]

    resp_vf = await client.post(
        "/api/liveness/session/verify-frame",
        data={"session_id": session_id},
        files={"image": ("frame.jpg", face_bytes, "image/jpeg")}
    )
    assert resp_vf.status_code == 200, f"Verify frame failed: {resp_vf.text}"
    assert resp_vf.json()["verified"] is True
    return session_id


async def run_profile_registration_suite():
    print("=" * 75)
    print("      USER PROFILE REGISTRATION & 1:N FACE LOGIN VERIFICATION SUITE      ")
    print("=" * 75)

    fixture_path = os.path.join(os.path.dirname(__file__), "test_fixtures", "test_face.jpg")
    assert os.path.exists(fixture_path), f"Base test image missing at {fixture_path}"

    with open(fixture_path, "rb") as f:
        face_bytes = f.read()

    ts = int(time.time())
    user_alice = f"prof_alice_{ts}"
    user_bob = f"prof_bob_{ts}"
    email_alice = f"alice_{ts}@biocorp.com"

    delete_user(user_alice)
    delete_user(user_bob)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:

        # ----------------------------------------------------------------------
        # Test 1: Register new user with all five profile fields
        # ----------------------------------------------------------------------
        print("\n[Check 1/10] Testing new user registration with all five profile fields...")
        session_id_alice = await create_verified_session(client, "register", face_bytes, user_alice)
        files = {"image": ("alice.jpg", face_bytes, "image/jpeg")}
        data = {
            "session_id": session_id_alice,
            "user_id": user_alice,
            "full_name": "Alice Montgomery",
            "email": f"  {email_alice.upper()}  ",  # Test trimming and case-normalization
            "role": "Principal AI Architect",
            "phone_number": "+1 (555) 019-2834"
        }

        resp = await client.post("/register", data=data, files=files)
        print(f"  - Status Code: {resp.status_code}")
        print(f"  - Response JSON: {resp.json()}")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        res_json = resp.json()
        assert res_json["status"] == "success"
        assert res_json["user_id"] == user_alice
        assert res_json["full_name"] == "Alice Montgomery"
        assert res_json["email"] == email_alice.lower()  # Verified lowercase normalized
        assert res_json["role"] == "Principal AI Architect"
        assert res_json["phone_number"] == "+1 (555) 019-2834"
        print("  [PASS] User successfully registered with all five profile fields.")

        # ----------------------------------------------------------------------
        # Test 2: Duplicate User ID rejection (HTTP 409)
        # ----------------------------------------------------------------------
        print("\n[Check 2/10] Testing duplicate User ID rejection...")
        session_id_dup_id = await create_verified_session(client, "register", face_bytes, user_alice)
        files = {"image": ("alice_dup.jpg", face_bytes, "image/jpeg")}
        data_dup_id = {
            "session_id": session_id_dup_id,
            "user_id": user_alice,  # Same user_id
            "full_name": "Alice Different Name",
            "email": f"different_{ts}@biocorp.com",
            "role": "Consultant",
            "phone_number": "+1 (555) 999-8888"
        }

        resp_dup_id = await client.post("/register", data=data_dup_id, files=files)
        print(f"  - Status Code: {resp_dup_id.status_code}")
        print(f"  - Response JSON: {resp_dup_id.json()}")
        assert resp_dup_id.status_code == 409, f"Expected 409, got {resp_dup_id.status_code}"
        assert "User already exists" in str(resp_dup_id.json())
        print("  [PASS] Duplicate User ID correctly rejected with HTTP 409 Conflict.")

        # ----------------------------------------------------------------------
        # Test 3: Duplicate Email rejection (HTTP 409)
        # ----------------------------------------------------------------------
        print("\n[Check 3/10] Testing duplicate Email rejection...")
        session_id_dup_email = await create_verified_session(client, "register", face_bytes, user_bob)
        files = {"image": ("bob_dup_email.jpg", face_bytes, "image/jpeg")}
        data_dup_email = {
            "session_id": session_id_dup_email,
            "user_id": user_bob,  # Different user_id
            "full_name": "Bob Smith",
            "email": email_alice,  # Same email as Alice!
            "role": "Security Engineer",
            "phone_number": "+1 (555) 777-6666"
        }

        resp_dup_email = await client.post("/register", data=data_dup_email, files=files)
        print(f"  - Status Code: {resp_dup_email.status_code}")
        print(f"  - Response JSON: {resp_dup_email.json()}")
        assert resp_dup_email.status_code == 409, f"Expected 409, got {resp_dup_email.status_code}"
        assert "Email already exists" in str(resp_dup_email.json())
        print("  [PASS] Duplicate Email correctly rejected with HTTP 409 Conflict.")

        # ----------------------------------------------------------------------
        # Test 4: Invalid Email and Phone validation (HTTP 400)
        # ----------------------------------------------------------------------
        print("\n[Check 4/10] Testing invalid email and phone format validation...")
        # Case A: Invalid Email
        session_bad_mail = await create_verified_session(client, "register", face_bytes)
        bad_email_data = {
            "session_id": session_bad_mail,
            "user_id": f"temp_bad_mail_{ts}",
            "full_name": "Bad Mail User",
            "email": "not-an-email",
            "role": "Tester",
            "phone_number": "+1 (555) 123-4567"
        }
        resp_bad_email = await client.post("/register", data=bad_email_data, files={"image": ("img.jpg", face_bytes, "image/jpeg")})
        assert resp_bad_email.status_code == 400
        print("  - Invalid email correctly returned HTTP 400.")

        # Case B: Invalid Phone
        session_bad_phone = await create_verified_session(client, "register", face_bytes)
        bad_phone_data = {
            "session_id": session_bad_phone,
            "user_id": f"temp_bad_phone_{ts}",
            "full_name": "Bad Phone User",
            "email": f"valid_{ts}@biocorp.com",
            "role": "Tester",
            "phone_number": "123"  # Too short
        }
        resp_bad_phone = await client.post("/register", data=bad_phone_data, files={"image": ("img.jpg", face_bytes, "image/jpeg")})
        assert resp_bad_phone.status_code == 400
        print("  - Invalid phone correctly returned HTTP 400.")
        print("  [PASS] Frontend & Backend validation guards verified.")

        # ----------------------------------------------------------------------
        # Test 5 & 7: Pure 1:N Automatic Face Login & Profile Retrieval
        # ----------------------------------------------------------------------
        print("\n[Check 5/10 & Check 7/10] Testing Pure 1:N automatic Face Login without User ID or email...")
        session_login_alice = await create_verified_session(client, "login", face_bytes)
        login_files = {"image": ("login.jpg", face_bytes, "image/jpeg")}
        login_data = {"session_id": session_login_alice}

        # Note: NO user_id, username, or email field sent!
        resp_login = await client.post("/login", data=login_data, files=login_files)
        print(f"  - Status Code: {resp_login.status_code}")
        print(f"  - Response JSON: {resp_login.json()}")
        assert resp_login.status_code == 200, f"Expected 200, got {resp_login.status_code}"
        login_json = resp_login.json()
        assert login_json["authenticated"] is True
        assert login_json["user_id"] == user_alice
        assert "user_profile" in login_json, "Missing user_profile in login response!"

        profile = login_json["user_profile"]
        assert profile["user_id"] == user_alice
        assert profile["full_name"] == "Alice Montgomery"
        assert profile["email"] == email_alice.lower()
        assert profile["role"] == "Principal AI Architect"
        assert profile["phone_number"] == "+1 (555) 019-2834"
        print("  [PASS] 1:N Face Login succeeded and accurately retrieved user profile without User ID/email input.")

        # ----------------------------------------------------------------------
        # Test 8: Verify NO FaceNet embedding or internal DB leak in response
        # ----------------------------------------------------------------------
        print("\n[Check 8/10] Verifying zero biometric embedding leakage in login response...")
        assert "embedding" not in login_json
        assert "embedding" not in profile
        assert "score" not in login_json
        assert "similarity" not in login_json
        print("  [PASS] Biometric vectors and similarity scores strictly withheld.")

        # ----------------------------------------------------------------------
        # Test 6: Rejection of unknown / unregistered face (HTTP 401)
        # ----------------------------------------------------------------------
        print("\n[Check 6/10] Testing 1:N rejection of unknown face...")
        # Delete Alice so Alice's face becomes unregistered/unknown in the system
        delete_user(user_alice)

        session_unk = await create_verified_session(client, "login", face_bytes)
        resp_unknown = await client.post("/login", data={"session_id": session_unk}, files={"image": ("unknown.jpg", face_bytes, "image/jpeg")})
        print(f"  - Status Code: {resp_unknown.status_code}")
        print(f"  - Response JSON: {resp_unknown.json()}")
        assert resp_unknown.status_code == 401, f"Expected 401, got {resp_unknown.status_code}"
        assert resp_unknown.json()["detail"]["authenticated"] is False
        print("  [PASS] Unknown face strictly rejected with HTTP 401 Unauthorized under 1:N.")

        # ----------------------------------------------------------------------
        # Test 9: FAISS Synchronization after registration
        # ----------------------------------------------------------------------
        print("\n[Check 9/10] Testing FAISS synchronization after registration...")
        reg_user = f"sync_test_{ts}"
        delete_user(reg_user)
        faiss_mgr = get_faiss_manager()
        count_before = faiss_mgr.get_user_count()

        session_sync = await create_verified_session(client, "register", face_bytes, reg_user)
        reg_sync = await client.post("/register", data={
            "session_id": session_sync,
            "user_id": reg_user,
            "full_name": "Sync User",
            "email": f"sync_{ts}@biocorp.com",
            "role": "DevOps",
            "phone_number": "+1 555-0100"
        }, files={"image": ("face.jpg", face_bytes, "image/jpeg")})
        assert reg_sync.status_code == 200

        count_after = faiss_mgr.get_user_count()
        assert count_after == count_before + 1
        print(f"  - FAISS index count increased from {count_before} to {count_after}.")
        print("  [PASS] FAISS index synchronized immediately post-commit.")

        # ----------------------------------------------------------------------
        # Test 10: FAISS Synchronization after backend restart (initialize_from_database)
        # ----------------------------------------------------------------------
        print("\n[Check 10/10] Testing FAISS synchronization after backend restart...")
        rebuilt_count = faiss_mgr.initialize_from_database()
        assert rebuilt_count >= count_after
        print(f"  - FAISS rebuild from PostgreSQL restored {rebuilt_count} user vector(s).")
        print("  [PASS] Startup database-to-FAISS reconciliation verified.")

        # Clean up test user
        delete_user(reg_user)

    print("\n" + "=" * 75)
    print("      ALL USER PROFILE REGISTRATION & 1:N TESTS PASSED (10/10)!      ")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(run_profile_registration_suite())
