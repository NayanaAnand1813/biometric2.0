import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure backend directory is in python search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from user_service import (
    register_user_with_embedding,
    get_user_embedding,
    delete_user,
    UserAlreadyExistsError
)


def test_concurrent_registrations():
    print("=" * 65)
    print("      TASK 4: MULTIPLE SIMULTANEOUS USERS CONCURRENCY TEST       ")
    print("=" * 65)

    test_image_path = os.path.join(os.path.dirname(__file__), "cropped_faces", "1234_face.jpg")
    assert os.path.exists(test_image_path), f"Test image not found at {test_image_path}"

    concurrent_users = [f"concurrent_user_{i}_{int(time.time())}" for i in range(1, 6)]

    # Clean up any residual test records
    for uid in concurrent_users:
        delete_user(uid)

    try:
        # -------------------------------------------------------------
        # Part 1: Simultaneous Distinct Users Registration
        # -------------------------------------------------------------
        print(f"\n[Test 1/3] Launching 5 simultaneous user registrations...")
        results = {}
        start_time = time.time()

        def register_worker(uid):
            thread_id = time.strftime("%H:%M:%S")
            print(f"  [Thread {uid}] Starting registration at {thread_id}...")
            res = register_user_with_embedding(
                user_id=uid,
                image_path=test_image_path,
                save_json=False
            )
            return uid, res

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(register_worker, uid) for uid in concurrent_users]
            for future in as_completed(futures):
                uid, res = future.result()
                results[uid] = res
                print(f"  [Thread {uid}] Completed successfully: status={res['status']}, dim={res['embedding_dimension']}")

        duration = time.time() - start_time
        print(f"\n  [PASS] All 5 concurrent registrations completed in {duration:.2f} seconds.")

        # -------------------------------------------------------------
        # Part 2: Verify Strict Isolation & Accuracy in Database
        # -------------------------------------------------------------
        print("\n[Test 2/3] Verifying database records, dimensions, and data isolation...")
        for uid in concurrent_users:
            rec = get_user_embedding(uid)
            assert rec is not None, f"User '{uid}' not found in database!"
            assert rec["user_id"] == uid, f"User ID mismatch: expected {uid}, got {rec['user_id']}"
            assert rec["embedding_dimension"] == 512, f"Embedding dimension mismatch: {rec['embedding_dimension']}"
            assert len(rec["embedding"]) == 512, f"Vector length mismatch: {len(rec['embedding'])}"
            assert all(isinstance(v, float) for v in rec["embedding"]), "Non-float value found in embedding vector"
            print(f"  [PASS] User '{uid}' verified: exactly 512 float values stored correctly.")

        # -------------------------------------------------------------
        # Part 3: Race Condition on Duplicate User ID
        # -------------------------------------------------------------
        print("\n[Test 3/3] Testing simultaneous duplicate registration race condition...")
        duplicate_uid = f"dup_race_user_{int(time.time())}"
        delete_user(duplicate_uid)

        race_results = []
        race_errors = []

        def race_worker(attempt_id):
            try:
                res = register_user_with_embedding(
                    user_id=duplicate_uid,
                    image_path=test_image_path,
                    save_json=False
                )
                race_results.append((attempt_id, res))
            except UserAlreadyExistsError as e:
                race_errors.append((attempt_id, str(e)))

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(race_worker, "Worker-1")
            f2 = executor.submit(race_worker, "Worker-2")
            f1.result()
            f2.result()

        print(f"  - Successful insertions: {len(race_results)}")
        print(f"  - Rejected duplicates:   {len(race_errors)}")

        assert len(race_results) == 1, f"Expected exactly 1 success, got {len(race_results)}"
        assert len(race_errors) == 1, f"Expected exactly 1 UserAlreadyExistsError, got {len(race_errors)}"
        print(f"  [PASS] Race condition safely handled: 1 inserted, 1 rejected with UserAlreadyExistsError.")

        # Verify the database still has exactly one record
        rec_dup = get_user_embedding(duplicate_uid)
        assert rec_dup is not None, "Record missing after race test"
        assert rec_dup["embedding_dimension"] == 512
        print(f"  [PASS] Database integrity preserved for duplicate user.")

        delete_user(duplicate_uid)

    finally:
        # Clean up test users
        for uid in concurrent_users:
            delete_user(uid)
        print("\n[CLEANUP] All temporary concurrent test users removed from database.")

    print("\n" + "=" * 65)
    print("      TASK 4: ALL CONCURRENCY & MULTI-USER TESTS PASSED         ")
    print("=" * 65)


if __name__ == "__main__":
    test_concurrent_registrations()

