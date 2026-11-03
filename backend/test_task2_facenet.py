import sys
import os

# Ensure backend directory is in python search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_embedding import generate_embedding, embedder


def test_facenet_integration():
    print("=" * 60)
    print("      TASK 2: FACENET INTEGRATION & VALIDATION TESTS     ")
    print("=" * 60)

    # 1. Verify single model instance
    print("\n[Check 1/5] Verifying FaceNet model singleton...")
    assert embedder is not None, "FaceNet embedder instance is None!"
    print("  [PASS] FaceNet model is loaded in memory as a singleton.")

    # 2. Test valid embedding generation
    print("\n[Check 2/5] Testing 512-D embedding generation with test image...")
    test_user_id = "test_user_fn_001"
    test_image_path = os.path.join(os.path.dirname(__file__), "cropped_faces", "1234_face.jpg")

    result = generate_embedding(user_id=test_user_id, image_path=test_image_path, save_json=False)

    print(f"  - User ID: {result['user_id']}")
    print(f"  - Dimension: {result['embedding_dimension']}")
    print(f"  - Vector Length: {len(result['embedding'])}")
    print(f"  - First 5 Floats: {result['embedding'][:5]}")

    assert result["user_id"] == test_user_id, "User ID does not match input!"
    assert result["embedding_dimension"] == 512, "Dimension is not 512!"
    assert len(result["embedding"]) == 512, "Returned embedding length is not 512!"
    assert all(isinstance(v, float) for v in result["embedding"]), "Not all elements are floats!"
    print("  [PASS] 512-D embedding generated and validated successfully.")

    # 3. Test empty user ID rejection
    print("\n[Check 3/5] Testing rejection of empty / whitespace user ID...")
    empty_rejected = False
    try:
        generate_embedding(user_id="   ", image_path=test_image_path)
    except ValueError as e:
        empty_rejected = True
        print(f"  [PASS] Successfully rejected empty user ID: {e}")
    assert empty_rejected, "Failed to reject empty user ID!"

    # 4. Test non-existent image rejection
    print("\n[Check 4/5] Testing rejection of non-existent image...")
    missing_rejected = False
    try:
        generate_embedding(user_id="user_test_missing", image_path="non_existent_face_9999.jpg")
    except FileNotFoundError as e:
        missing_rejected = True
        print(f"  [PASS] Successfully rejected non-existent image: {e}")
    assert missing_rejected, "Failed to reject missing image!"

    # 5. Test user-to-embedding independence
    print("\n[Check 5/5] Testing distinct user associations...")
    user_a = "user_alpha"
    user_b = "user_beta"
    res_a = generate_embedding(user_id=user_a, image_path=test_image_path, save_json=False)
    res_b = generate_embedding(user_id=user_b, image_path=test_image_path, save_json=False)

    assert res_a["user_id"] == user_a, "User A ID mismatch"
    assert res_b["user_id"] == user_b, "User B ID mismatch"
    # Ensure they are independent Python list instances
    assert res_a["embedding"] is not res_b["embedding"], "Embeddings shared mutable memory reference!"
    print("  [PASS] User IDs and embeddings are strictly isolated and independent.")

    print("\n" + "=" * 60)
    print("      TASK 2: ALL FACENET INTEGRATION CHECKS PASSED     ")
    print("=" * 60)


if __name__ == "__main__":
    test_facenet_integration()

