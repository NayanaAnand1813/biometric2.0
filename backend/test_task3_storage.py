import sys
import os

# Ensure backend directory is in python search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from user_service import register_user_with_embedding, get_user_embedding, delete_user


def test_task3_database_storage():
    print("=" * 60)
    print("      TASK 3: DATABASE STORAGE FUNCTION VERIFICATION     ")
    print("=" * 60)

    test_user_id = "test_user_task3_storage"
    test_image_path = os.path.join(os.path.dirname(__file__), "cropped_faces", "1234_face.jpg")

    # Clean up prior test data if any
    delete_user(test_user_id)

    try:
        # Step 1: Register sample user with face image
        print(f"\n[Step 1/4] Registering sample user: '{test_user_id}'...")
        reg_result = register_user_with_embedding(
            user_id=test_user_id,
            image_path=test_image_path,
            save_json=False
        )

        print(f"  - Status: {reg_result['status']}")
        print(f"  - User ID: {reg_result['user_id']}")
        print(f"  - Timestamp: {reg_result['created_at']}")
        print(f"  - Embedding Dimension: {reg_result['embedding_dimension']}")
        print(f"  - Message: {reg_result['message']}")

        assert reg_result["status"] == "success"
        assert reg_result["user_id"] == test_user_id
        assert reg_result["embedding_dimension"] == 512
        print("  [PASS] Registration confirmed by database transaction commit.")

        # Step 2: Retrieve embedding from PostgreSQL
        print(f"\n[Step 2/4] Retrieving embedding from PostgreSQL for user: '{test_user_id}'...")
        retrieved = get_user_embedding(test_user_id)

        assert retrieved is not None, f"Failed to retrieve user '{test_user_id}' from database!"
        assert retrieved["user_id"] == test_user_id, "Retrieved user_id mismatch!"
        print(f"  - Retrieved User: {retrieved['user_id']}")
        print(f"  - Retrieved Timestamp: {retrieved['created_at']}")
        print(f"  - Retrieved Vector Dimensions: {retrieved['embedding_dimension']}")
        print(f"  - First 5 Floating Values: {retrieved['embedding'][:5]}")
        print("  [PASS] Record retrieved successfully from database.")

        # Step 3: Verify that exactly 512 floating-point values are returned
        print("\n[Step 3/4] Verifying vector integrity and element count...")
        emb = retrieved["embedding"]
        assert isinstance(emb, list), "Retrieved embedding is not a list!"
        assert len(emb) == 512, f"Expected exactly 512 values, got {len(emb)}!"
        assert all(isinstance(v, float) for v in emb), "Elements in embedding are not float type!"
        print(f"  [PASS] Verified exactly 512 floating-point numbers stored and returned.")

        # Step 4: Verify non-duplicate JSON disk footprint
        print("\n[Step 4/4] Verifying no duplicate JSON file created when save_json=False...")
        json_path = os.path.join(os.path.dirname(__file__), "embeddings", f"{test_user_id}.json")
        assert not os.path.exists(json_path), f"Unexpected duplicate JSON file created at {json_path}!"
        print("  [PASS] No redundant JSON file written to disk.")

    finally:
        # Clean up test user
        delete_user(test_user_id)
        print(f"\n[CLEANUP] Deleted test user '{test_user_id}' from database.")

    print("\n" + "=" * 60)
    print("      TASK 3: ALL DATABASE STORAGE CHECKS PASSED     ")
    print("=" * 60)


if __name__ == "__main__":
    test_task3_database_storage()

