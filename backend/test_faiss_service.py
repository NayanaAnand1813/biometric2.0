import os
import sys
import time
import threading
import numpy as np
import cv2

# Ensure backend directory is in python search path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from generate_embedding import extract_face_embedding, generate_embedding
from faiss_service import FAISSIndexManager, get_faiss_manager
from user_service import (
    register_user_with_embedding,
    delete_user,
    get_user_embedding,
    UserAlreadyExistsError
)


def test_shared_face_embedding_pipeline():
    print("=" * 65)
    print(" [Check 1/6] Testing Shared FaceNet Preprocessing & Normalization")
    print("=" * 65)

    test_image_path = os.path.join(BASE_DIR, "cropped_faces", "1234_face.jpg")
    img_bgr = cv2.imread(test_image_path)
    assert img_bgr is not None and img_bgr.size > 0, "Test face image could not be loaded."

    # 1. Direct extraction via in-memory BGR image
    embedding_arr = extract_face_embedding(img_bgr)
    assert isinstance(embedding_arr, np.ndarray), "Output must be a numpy ndarray."
    assert embedding_arr.shape == (512,), f"Expected shape (512,), got {embedding_arr.shape}."
    assert np.all(np.isfinite(embedding_arr)), "Embedding contains non-finite values."

    # 2. Verify unit L2 norm (necessary for FAISS IndexFlatIP cosine similarity)
    norm = float(np.linalg.norm(embedding_arr))
    assert abs(norm - 1.0) < 1e-5, f"Expected unit norm ~1.0, got {norm}."
    print(f"  [PASS] In-memory extract_face_embedding verified: 512-D, finite, L2 norm = {norm:.6f}")

    # 3. Path-based interface consistency
    res_path = generate_embedding("consistency_test_user", test_image_path, save_json=False)
    vec_path = np.array(res_path["embedding"], dtype=np.float32)

    # They must be mathematically identical since generate_embedding delegates to extract_face_embedding
    diff = np.max(np.abs(embedding_arr - vec_path))
    assert diff < 1e-6, f"Max difference between shared pipeline and path interface: {diff}"
    print("  [PASS] Path-based generate_embedding matches shared extract_face_embedding identically.")


def test_faiss_vector_math_properties():
    print("\n" + "=" * 65)
    print(" [Check 2/6] Testing FAISS IndexFlatIP Mathematical Properties")
    print("=" * 65)

    mgr = FAISSIndexManager()
    mgr.reset()

    # Create random unit vector v1
    np.random.seed(42)
    raw_v1 = np.random.randn(512).astype(np.float32)
    v1 = raw_v1 / np.linalg.norm(raw_v1)

    # Identical vector
    v1_identical = v1.copy()

    # Construct strictly orthogonal vector v2 via Gram-Schmidt
    raw_v2 = np.random.randn(512).astype(np.float32)
    v2_proj = raw_v2 - np.dot(raw_v2, v1) * v1
    v2_ortho = v2_proj / np.linalg.norm(v2_proj)
    dot_check = float(np.dot(v1, v2_ortho))
    assert abs(dot_check) < 1e-6, f"Gram-Schmidt orthogonalization failed: dot={dot_check}"

    # Add v1 to index
    mgr.add_user("test_subject_1", v1)
    mgr.add_user("test_subject_2", v2_ortho)

    # Search with v1_identical
    top1 = mgr.search_top_k(v1_identical, k=1)
    assert len(top1) == 1
    matched_user, similarity_identical = top1[0]
    assert matched_user == "test_subject_1"
    assert abs(similarity_identical - 1.0) < 1e-4, f"Identical vector similarity expected ~1.0, got {similarity_identical}"
    print(f"  [PASS] Identical vector search: matched '{matched_user}' with similarity = {similarity_identical:.6f} (~1.0)")

    # 1:1 similarity with identical vector
    sim_1to1 = mgr.compute_user_similarity("test_subject_1", v1_identical)
    assert abs(sim_1to1 - 1.0) < 1e-4, f"1:1 similarity expected ~1.0, got {sim_1to1}"
    print(f"  [PASS] 1:1 verification similarity for identical vector: {sim_1to1:.6f}")

    # Search with orthogonal vector against test_subject_1
    sim_ortho = mgr.compute_user_similarity("test_subject_1", v2_ortho)
    assert abs(sim_ortho) < 1e-5, f"Orthogonal vector similarity expected ~0.0, got {sim_ortho}"
    print(f"  [PASS] Orthogonal vector 1:1 similarity: {sim_ortho:.8f} (~0.0)")

    # Test top-2 search
    top2 = mgr.search_top_k(v1, k=2)
    assert len(top2) == 2
    assert top2[0][0] == "test_subject_1" and abs(top2[0][1] - 1.0) < 1e-4
    assert top2[1][0] == "test_subject_2" and abs(top2[1][1]) < 1e-5
    print("  [PASS] Top-k ranking correctly orders by exact inner product.")


def test_faiss_id_mapping_and_removal():
    print("\n" + "=" * 65)
    print(" [Check 3/6] Testing FAISS Bidirectional ID Mapping & Removal")
    print("=" * 65)

    mgr = FAISSIndexManager()
    mgr.reset()

    # Add 3 users
    uids = ["alice_001", "bob_002", "charlie_003"]
    for uid in uids:
        vec = np.random.randn(512).astype(np.float32)
        norm_vec = vec / np.linalg.norm(vec)
        mgr.add_user(uid, norm_vec)

    assert mgr.get_user_count() == 3
    for idx, uid in enumerate(uids):
        assert mgr.get_user_for_index(idx) == uid
        assert mgr.get_index_for_user(uid) == idx
        assert mgr.contains_user(uid) is True
    print("  [PASS] Initial 3-user bidirectional ID mapping verified.")

    # Remove the middle user ('bob_002', index 1)
    removed = mgr.remove_user("bob_002")
    assert removed is True
    assert mgr.get_user_count() == 2
    assert mgr.contains_user("bob_002") is False

    # Verify shifted indices: alice is still index 0, charlie is now index 1
    assert mgr.get_user_for_index(0) == "alice_001"
    assert mgr.get_index_for_user("alice_001") == 0

    assert mgr.get_user_for_index(1) == "charlie_003"
    assert mgr.get_index_for_user("charlie_003") == 1
    print("  [PASS] Removal of intermediate vector cleanly updates shifted index mapping.")


def test_faiss_initialization_from_postgres():
    print("\n" + "=" * 65)
    print(" [Check 4/6] Testing FAISS Initialization from PostgreSQL Database")
    print("=" * 65)

    # Register temporary test user in DB
    test_uid = f"faiss_db_init_test_{int(time.time())}"
    test_img = os.path.join(BASE_DIR, "cropped_faces", "1234_face.jpg")

    try:
        reg_res = register_user_with_embedding(test_uid, image_path=test_img)
        assert reg_res["status"] == "success"

        # Initialize global FAISS manager from PostgreSQL
        faiss_mgr = get_faiss_manager()
        count = faiss_mgr.initialize_from_database()
        assert count >= 1, "FAISS should have loaded at least 1 record from DB."
        assert faiss_mgr.contains_user(test_uid) is True
        print(f"  [PASS] Successfully initialized FAISS from PostgreSQL with {count} total vectors.")

        # Reconstructed vector check
        reconstructed = faiss_mgr.get_user_vector(test_uid)
        assert reconstructed is not None
        assert reconstructed.shape == (512,)
        assert abs(np.linalg.norm(reconstructed) - 1.0) < 1e-4
        print(f"  [PASS] Reconstructed vector from FAISS for '{test_uid}' is 512-D unit vector.")

    finally:
        # Cleanup
        delete_user(test_uid)
        faiss_mgr.remove_user(test_uid)
        print("  [CLEANUP] Cleaned up temporary database test user.")


def test_faiss_safe_synchronization_and_rollback():
    print("\n" + "=" * 65)
    print(" [Check 5/6] Testing Safe Synchronization & Failed-DB Inconsistency")
    print("=" * 65)

    faiss_mgr = get_faiss_manager()
    faiss_mgr.initialize_from_database()
    initial_count = faiss_mgr.get_user_count()

    conflict_uid = f"faiss_conflict_user_{int(time.time())}"
    test_img = os.path.join(BASE_DIR, "cropped_faces", "1234_face.jpg")

    # Step 1: Successful registration adds to DB and FAISS
    reg1 = register_user_with_embedding(conflict_uid, image_path=test_img)
    assert reg1["status"] == "success"
    assert faiss_mgr.get_user_count() == initial_count + 1
    assert faiss_mgr.contains_user(conflict_uid) is True
    print("  [PASS] Successful registration synced to FAISS post-commit.")

    # Step 2: Attempt duplicate registration -> Database raises 409 UserAlreadyExistsError
    count_before_conflict = faiss_mgr.get_user_count()
    duplicate_raised = False
    try:
        register_user_with_embedding(conflict_uid, image_path=test_img)
    except UserAlreadyExistsError:
        duplicate_raised = True

    assert duplicate_raised is True, "Expected UserAlreadyExistsError on duplicate registration."

    # Verify FAISS index was NOT touched or corrupted by the failed registration
    assert faiss_mgr.get_user_count() == count_before_conflict
    print("  [PASS] Failed registration did not alter FAISS index count or state.")

    # Cleanup
    delete_user(conflict_uid)
    assert faiss_mgr.contains_user(conflict_uid) is False
    print("  [PASS] User deletion synchronized to FAISS index.")


def test_faiss_thread_safe_concurrency():
    print("\n" + "=" * 65)
    print(" [Check 6/6] Testing Concurrent Searches and Index Modifications")
    print("=" * 65)

    mgr = FAISSIndexManager()
    mgr.reset()

    # Seed with 10 users
    for i in range(10):
        vec = np.random.randn(512).astype(np.float32)
        norm_vec = vec / np.linalg.norm(vec)
        mgr.add_user(f"concurrent_seed_{i}", norm_vec)

    num_threads = 8
    iterations_per_thread = 25
    errors = []

    def worker(thread_id: int):
        for it in range(iterations_per_thread):
            try:
                # Alternate between searches and additions
                if it % 2 == 0:
                    q = np.random.randn(512).astype(np.float32)
                    norm_q = q / np.linalg.norm(q)
                    res = mgr.search_top_k(norm_q, k=3)
                    assert len(res) <= 3
                else:
                    new_uid = f"dyn_user_t{thread_id}_it{it}"
                    v = np.random.randn(512).astype(np.float32)
                    norm_v = v / np.linalg.norm(v)
                    mgr.add_user(new_uid, norm_v)
            except Exception as e:
                errors.append(f"Thread {thread_id} failed: {e}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Encountered concurrency errors: {errors}"
    final_count = mgr.get_user_count()
    expected_added = (num_threads * (iterations_per_thread // 2))
    assert final_count == 10 + expected_added, f"Expected {10 + expected_added}, got {final_count}"
    print(f"  [PASS] 8 threads completed {8 * iterations_per_thread} operations without error. Final index count: {final_count}")


if __name__ == "__main__":
    print("=================================================================")
    print("      MILESTONE 7 (WAVE 1): SHARED PIPELINE & FAISS SERVICE      ")
    print("=================================================================")

    test_shared_face_embedding_pipeline()
    test_faiss_vector_math_properties()
    test_faiss_id_mapping_and_removal()
    test_faiss_initialization_from_postgres()
    test_faiss_safe_synchronization_and_rollback()
    test_faiss_thread_safe_concurrency()

    print("\n" + "=" * 65)
    print("  [ALL TESTS PASSED] WAVE 1 IMPLEMENTATION VERIFIED SUCCESSFULLY! ")
    print("=================================================================")

