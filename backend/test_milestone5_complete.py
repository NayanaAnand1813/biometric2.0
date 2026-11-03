import sys
import os
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure backend directory is in python search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from user_service import (
    register_user_with_embedding,
    get_user_embedding,
    delete_user,
    UserAlreadyExistsError,
    InvalidUserError
)
from main import app, detector, _detector_lock
import mediapipe as mp


def test_milestone5_all():
    print("=" * 70)
    print("       MILESTONE 5: END-TO-END VERIFICATION & ACCEPTANCE SUITE       ")
    print("=" * 70)

    test_image_path = os.path.join(os.path.dirname(__file__), "cropped_faces", "1234_face.jpg")
    assert os.path.exists(test_image_path), f"Base test image missing at {test_image_path}"

    test_users_to_clean = []

    try:
        # ----------------------------------------------------------------------
        # Test 1 & 2: Register one sample user and store 512-D embedding in PG
        # ----------------------------------------------------------------------
        print("\n[Step 1/8] Registering sample user 1 and storing 512-D embedding in PostgreSQL...")
        user_1 = f"m5_user_alpha_{int(time.time())}"
        test_users_to_clean.append(user_1)
        delete_user(user_1)

        res_1 = register_user_with_embedding(user_id=user_1, image_path=test_image_path, save_json=False)
        assert res_1["status"] == "success"
        assert res_1["user_id"] == user_1
        assert res_1["embedding_dimension"] == 512
        print(f"  [PASS] User '{user_1}' registered. DB confirmation received: created_at={res_1['created_at']}")

        # ----------------------------------------------------------------------
        # Test 3 & 4: Retrieve embedding from PostgreSQL and verify exactly 512 values
        # ----------------------------------------------------------------------
        print("\n[Step 2/8] Retrieving embedding from PostgreSQL and verifying 512 dimensions...")
        user_1_record = get_user_embedding(user_1)
        assert user_1_record is not None, f"Failed to retrieve user '{user_1}'!"
        assert user_1_record["user_id"] == user_1, "User ID mismatch in retrieved record!"
        assert user_1_record["embedding_dimension"] == 512, f"Dimension mismatch: {user_1_record['embedding_dimension']}"
        assert len(user_1_record["embedding"]) == 512, f"Vector length mismatch: {len(user_1_record['embedding'])}"
        assert all(isinstance(v, float) for v in user_1_record["embedding"]), "Non-float in embedding"
        initial_embedding_snapshot = list(user_1_record["embedding"])
        print(f"  [PASS] Retrieved record verified: exactly 512 float values. Sample: {initial_embedding_snapshot[:3]}")

        # ----------------------------------------------------------------------
        # Test 5: Register second user and verify first user's data remains unchanged
        # ----------------------------------------------------------------------
        print("\n[Step 3/8] Registering second user and verifying data immutability of user 1...")
        user_2 = f"m5_user_beta_{int(time.time())}"
        test_users_to_clean.append(user_2)
        delete_user(user_2)

        res_2 = register_user_with_embedding(user_id=user_2, image_path=test_image_path, save_json=False)
        assert res_2["status"] == "success"
        assert res_2["user_id"] == user_2
        print(f"  [PASS] Second user '{user_2}' registered.")

        # Re-fetch user 1 and assert strict equality
        user_1_after = get_user_embedding(user_1)
        assert user_1_after["user_id"] == user_1, "User 1 ID altered!"
        assert user_1_after["created_at"] == user_1_record["created_at"], "User 1 created_at timestamp altered!"
        assert user_1_after["embedding"] == initial_embedding_snapshot, "User 1 embedding vector altered!"
        print("  [PASS] User 1 data integrity confirmed: ID, timestamp, and embedding values completely unchanged.")

        # ----------------------------------------------------------------------
        # Test 6: Test duplicate user ID handling
        # ----------------------------------------------------------------------
        print("\n[Step 4/8] Testing duplicate user ID rejection...")
        duplicate_caught = False
        try:
            register_user_with_embedding(user_id=user_1, image_path=test_image_path, save_json=False)
        except UserAlreadyExistsError as e:
            duplicate_caught = True
            print(f"  [PASS] Successfully rejected duplicate registration: {e}")
        assert duplicate_caught, "Failed to reject duplicate user ID!"

        # Ensure user 1 record still matches original snapshot
        user_1_check = get_user_embedding(user_1)
        assert user_1_check["embedding"] == initial_embedding_snapshot, "User 1 data corrupted by duplicate attempt!"
        print("  [PASS] User 1 was not overwritten by duplicate registration attempt.")

        # ----------------------------------------------------------------------
        # Test 7: Test invalid image handling
        # ----------------------------------------------------------------------
        print("\n[Step 5/8] Testing invalid image handling (missing image, corrupted image, empty ID)...")
        # 7a: Missing file
        missing_handled = False
        try:
            register_user_with_embedding(user_id="invalid_img_user", image_path="non_existent_path_xyz.jpg")
        except InvalidUserError as e:
            missing_handled = True
            print(f"  [PASS] Missing image rejected: {e}")
        assert missing_handled

        # 7b: Corrupted image file (empty or non-image content)
        corrupted_path = os.path.join(os.path.dirname(__file__), "cropped_faces", "temp_corrupt.jpg")
        with open(corrupted_path, "w") as f:
            f.write("Not an image data")

        corrupt_handled = False
        try:
            register_user_with_embedding(user_id="corrupt_img_user", image_path=corrupted_path)
        except InvalidUserError as e:
            corrupt_handled = True
            print(f"  [PASS] Corrupted image rejected: {e}")
        finally:
            if os.path.exists(corrupted_path):
                os.remove(corrupted_path)
        assert corrupt_handled

        # 7c: Empty user ID
        empty_id_handled = False
        try:
            register_user_with_embedding(user_id="   ", image_path=test_image_path)
        except InvalidUserError as e:
            empty_id_handled = True
            print(f"  [PASS] Empty user ID rejected: {e}")
        assert empty_id_handled

        # ----------------------------------------------------------------------
        # Test 8: Multiple simultaneous registration requests & association verification
        # ----------------------------------------------------------------------
        print("\n[Step 6/8] Testing 5 simultaneous registrations with verified user-to-embedding isolation...")
        concurrent_batch = [f"m5_batch_user_{k}_{int(time.time())}" for k in range(1, 6)]
        test_users_to_clean.extend(concurrent_batch)
        for u in concurrent_batch:
            delete_user(u)

        def worker(uid):
            return uid, register_user_with_embedding(user_id=uid, image_path=test_image_path, save_json=False)

        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = [ex.submit(worker, u) for u in concurrent_batch]
            for fut in as_completed(futures):
                uid, res = fut.result()
                assert res["status"] == "success"
                assert res["user_id"] == uid
                assert res["embedding_dimension"] == 512

        print("  [PASS] All 5 concurrent registrations succeeded.")

        # Cross-verify each user in PostgreSQL
        for uid in concurrent_batch:
            rec = get_user_embedding(uid)
            assert rec is not None, f"Concurrent user '{uid}' missing from database!"
            assert rec["user_id"] == uid, f"User association mismatch for '{uid}'"
            assert len(rec["embedding"]) == 512, f"Embedding dimension mismatch for '{uid}'"
            print(f"  [PASS] Verified user '{uid}': correctly associated with 512-D embedding in DB.")

        # ----------------------------------------------------------------------
        # Step 7: Regression verification - MediaPipe BlazeFace detection
        # ----------------------------------------------------------------------
        print("\n[Step 7/8] Regression check: Verifying MediaPipe BlazeFace face detector...")
        raw_bgr = cv2.imread(test_image_path)
        rgb_img = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        with _detector_lock:
            det = detector.detect(mp_img)
        assert len(det.detections) > 0, "BlazeFace failed to detect face on test image!"
        print(f"  [PASS] BlazeFace detected face: score={det.detections[0].categories[0].score:.4f}")

        # ----------------------------------------------------------------------
        # Step 8: Regression verification - Milestone 4 users table structure
        # ----------------------------------------------------------------------
        print("\n[Step 8/8] Regression check: Verifying database connection & primary key integrity...")
        assert get_user_embedding("non_existent_user_999999") is None
        print("  [PASS] Non-existent user query correctly returns None.")

    finally:
        # Cleanup all created test records
        print("\n[CLEANUP] Removing test users from PostgreSQL...")
        for u in test_users_to_clean:
            delete_user(u)
        print("  [CLEANUP COMPLETED] Test records cleaned up successfully.")

    print("\n" + "=" * 70)
    print("    [ALL ACCEPTANCE CRITERIA PASSED] MILESTONE 5 FULLY VERIFIED!     ")
    print("=" * 70)


if __name__ == "__main__":
    test_milestone5_all()

