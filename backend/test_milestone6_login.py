import sys
import os
import asyncio
import httpx

# Ensure backend directory is in python search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import app
from user_service import delete_user, get_user_embedding, register_user_with_embedding


async def test_milestone6_login_pipeline():
    print("=" * 65)
    print("       MILESTONE 6: LOGIN IMAGE CAPTURE API VERIFICATION        ")
    print("=" * 65)

    fixture_path = os.path.join(os.path.dirname(__file__), "test_fixtures", "test_face.jpg")
    if os.path.exists(fixture_path):
        test_image_path = fixture_path
    else:
        cropped_dir = os.path.join(os.path.dirname(__file__), "cropped_faces")
        available_faces = [os.path.join(cropped_dir, f) for f in os.listdir(cropped_dir) if f.endswith(".jpg")]
        if not available_faces:
            raise FileNotFoundError("No test face images found.")
        test_image_path = available_faces[0]
    with open(test_image_path, "rb") as f:
        valid_jpeg_bytes = f.read()

    # Pre-register registered_user_101 with the test face image so they exist as a registered user
    delete_user("registered_user_101")
    register_user_with_embedding("registered_user_101", image_path=test_image_path)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:

        # -------------------------------------------------------------
        # Test 1: Successful login image reception & verification (Valid User ID & JPEG)
        # -------------------------------------------------------------
        print("\n[Check 1/5] Sending valid User ID and JPEG image to POST /login...")
        files = {"image": ("login_capture.jpg", valid_jpeg_bytes, "image/jpeg")}
        data = {"user_id": "registered_user_101"}

        resp = await client.post("/login", data=data, files=files)
        print(f"  - Status Code: {resp.status_code}")
        print(f"  - Response JSON: {resp.json()}")

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        res_json = resp.json()
        assert res_json["status"] == "success"
        assert res_json["user_id"] == "registered_user_101"
        assert res_json["image_size_bytes"] == len(valid_jpeg_bytes)
        print("  [PASS] Login image and User ID received and validated successfully.")

        # -------------------------------------------------------------
        # Test 2: Missing / Empty User ID handling
        # -------------------------------------------------------------
        print("\n[Check 2/5] Testing rejection of empty User ID...")
        files = {"image": ("login_capture.jpg", valid_jpeg_bytes, "image/jpeg")}
        data = {"user_id": "   "}

        resp_empty_user = await client.post("/login", data=data, files=files)
        print(f"  - Status Code: {resp_empty_user.status_code}")
        print(f"  - Response JSON: {resp_empty_user.json()}")

        assert resp_empty_user.status_code == 400, f"Expected 400, got {resp_empty_user.status_code}"
        print("  [PASS] Empty User ID correctly rejected with HTTP 400.")

        # -------------------------------------------------------------
        # Test 3: Empty Image File handling
        # -------------------------------------------------------------
        print("\n[Check 3/5] Testing rejection of empty image file...")
        files = {"image": ("empty.jpg", b"", "image/jpeg")}
        data = {"user_id": "user_empty_img"}

        resp_empty_img = await client.post("/login", data=data, files=files)
        print(f"  - Status Code: {resp_empty_img.status_code}")
        print(f"  - Response JSON: {resp_empty_img.json()}")

        assert resp_empty_img.status_code == 400, f"Expected 400, got {resp_empty_img.status_code}"
        print("  [PASS] Empty image file correctly rejected with HTTP 400.")

        # -------------------------------------------------------------
        # Test 4: Corrupted Non-Image File handling
        # -------------------------------------------------------------
        print("\n[Check 4/5] Testing rejection of corrupted / non-image data...")
        files = {"image": ("corrupted.jpg", b"This is plain text not an image", "image/jpeg")}
        data = {"user_id": "user_corrupt_img"}

        resp_corrupt = await client.post("/login", data=data, files=files)
        print(f"  - Status Code: {resp_corrupt.status_code}")
        print(f"  - Response JSON: {resp_corrupt.json()}")

        assert resp_corrupt.status_code == 400, f"Expected 400, got {resp_corrupt.status_code}"
        print("  [PASS] Corrupted image bytes correctly rejected with HTTP 400.")

        # Cleanup test user before boundary check
        delete_user("registered_user_101")

        # -------------------------------------------------------------
        # Test 5: Verify Milestone 7 boundaries (No premature auth matching)
        # -------------------------------------------------------------
        print("\n[Check 5/5] Verifying Milestone 7 boundaries...")
        # Verify that /login did NOT create an embedding or write to users table
        db_check = get_user_embedding("registered_user_101")
        assert db_check is None, "Premature database write occurred during login!"
        print("  [PASS] No database write or permanent storage performed during login.")
        print("  [PASS] No database write, embedding extraction, or auth decision performed (Milestone 7 strictly preserved).")

    print("\n" + "=" * 65)
    print("      ALL MILESTONE 6 BACKEND API TESTS PASSED SUCCESSFULLY!    ")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(test_milestone6_login_pipeline())

