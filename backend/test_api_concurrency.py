import sys
import os
import asyncio
import httpx

# Ensure backend directory is in python search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import app
from user_service import delete_user, get_user_embedding


async def run_api_concurrency_test():
    print("=" * 65)
    print("      FASTAPI ASYNC HTTP CONCURRENCY VERIFICATION        ")
    print("=" * 65)

    test_image_path = os.path.join(os.path.dirname(__file__), "cropped_faces", "1234_face.jpg")
    with open(test_image_path, "rb") as f:
        image_bytes = f.read()

    test_uids = [f"http_user_{i}" for i in range(1, 4)]
    for uid in test_uids:
        delete_user(uid)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Check health
        health_resp = await client.get("/")
        assert health_resp.status_code == 200
        print("[Check 1/3] Backend health check: OK")

        # Concurrent POST /upload requests
        print("\n[Check 2/3] Sending 3 simultaneous HTTP POST /upload requests...")

        async def send_reg(uid):
            files = {"image": ("face.jpg", image_bytes, "image/jpeg")}
            data = {"user_id": uid}
            return uid, await client.post("/upload", data=data, files=files)

        responses = await asyncio.gather(*(send_reg(uid) for uid in test_uids))

        for uid, resp in responses:
            print(f"  - User '{uid}' HTTP response: {resp.status_code}")
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
            res_json = resp.json()
            assert res_json["status"] == "success"
            assert res_json["embedding_dimension"] == 512
            print(f"    [PASS] Registered successfully via API: {res_json['message']}")

        # Test duplicate registration rejection via API
        print("\n[Check 3/3] Testing duplicate registration rejection (409 Conflict)...")
        files = {"image": ("face.jpg", image_bytes, "image/jpeg")}
        dup_resp = await client.post("/upload", data={"user_id": test_uids[0]}, files=files)
        print(f"  - Duplicate user HTTP status: {dup_resp.status_code}")
        assert dup_resp.status_code == 409, f"Expected 409 Conflict, got {dup_resp.status_code}"
        print(f"  - Response body: {dup_resp.json()}")
        print("  [PASS] Duplicate registration rejected with 409 Conflict.")

    # Cleanup
    for uid in test_uids:
        delete_user(uid)
    print("\n[CLEANUP] Cleaned up HTTP test users from database.")
    print("=" * 65)
    print("      FASTAPI ASYNC HTTP CONCURRENCY TEST PASSED!        ")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(run_api_concurrency_test())

