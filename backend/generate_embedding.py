import os
import json
import threading
import cv2
import numpy as np
from keras_facenet import FaceNet

# ==============================================================================
# Configuration & Model Initialization
# ==============================================================================

# Directory where embeddings will be saved (backend/embeddings/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Load the FaceNet model only ONCE globally to avoid re-loading weights on every call
print("Loading FaceNet model (this will load only once)...")
embedder = FaceNet()
_embedder_lock = threading.Lock()
print("FaceNet model loaded successfully.")


# ==============================================================================
# Shared FaceNet Embedding Pipeline
# ==============================================================================

def extract_face_embedding(face_bgr: np.ndarray) -> np.ndarray:
    """
    Extracts a 512-dimensional, L2-normalized FaceNet embedding vector from an
    in-memory cropped face image in BGR format.

    Preserves the exact preprocessing pipeline used across registration and authentication:
    1. Input validation: ensures face_bgr is a valid non-empty numpy array.
    2. Resizing to 160x160 pixels (FaceNet's standard input dimensions).
    3. Color conversion from BGR (OpenCV standard) to RGB (FaceNet expected format).
    4. Thread-safe FaceNet inference protected by _embedder_lock.
    5. Verification of 512-dimensional output and finite numerical values.
    6. Explicit L2 normalization: vector / ||vector||_2 to guarantee exact cosine similarity.

    Parameters:
        face_bgr (np.ndarray): In-memory cropped face image in BGR format.

    Returns:
        np.ndarray: 512-dimensional L2-normalized float32 embedding vector.

    Raises:
        ValueError: If input is not a valid image, output dimensions != 512, contains
                    non-finite (NaN/Inf) values, or has zero norm.
    """
    if face_bgr is None or not isinstance(face_bgr, np.ndarray) or face_bgr.size == 0:
        raise ValueError("Invalid face image: input must be a non-empty numpy array.")

    # 1. Resize image to 160x160 pixels (FaceNet standard input dimensions)
    resized_image = cv2.resize(face_bgr, (160, 160))

    # 2. Convert color from BGR (OpenCV default) to RGB (FaceNet expected format)
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # 3. Generate face embedding using FaceNet (Thread-safe inference)
    with _embedder_lock:
        embeddings = embedder.embeddings([rgb_image])
    embedding_vector = np.asarray(embeddings[0], dtype=np.float32)

    # 4. Verify 512-dimensional embedding output & validate values
    dimension = len(embedding_vector)
    if dimension != 512:
        raise ValueError(f"Expected 512-dimensional embedding, but got {dimension} dimensions.")

    if not np.all(np.isfinite(embedding_vector)):
        raise ValueError("Embedding contains invalid non-finite (NaN or Inf) values.")

    # 5. Explicit L2 normalization to guarantee unit norm for exact FAISS cosine search
    norm = float(np.linalg.norm(embedding_vector))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError(f"Invalid embedding norm: {norm}")
    embedding_vector = embedding_vector / norm

    return embedding_vector


# ==============================================================================
# Embedding Generation Function (Path-Based Interface)
# ==============================================================================

def generate_embedding(user_id: str, image_path: str, save_json: bool = False) -> dict:
    """
    Generates a 512-dimensional FaceNet embedding for a given cropped face image
    and associates it with the provided user ID. Delegates to the shared
    extract_face_embedding function.

    Parameters:
        user_id (str): The unique identifier for the user.
        image_path (str): File system path to the cropped face image.
        save_json (bool): If True, saves a JSON copy of the embedding to backend/embeddings/.
                          Defaults to False to prevent disk clutter and redundant I/O.

    Returns:
        dict: A dictionary containing the user_id, embedding dimension, embedding list,
              and optional output JSON file path if save_json is True.
    """
    # Step 1: Validate user ID
    if not user_id or not str(user_id).strip():
        raise ValueError("Invalid user_id: User ID cannot be empty.")
    clean_user_id = str(user_id).strip()

    # Step 2: Validate file path existence
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file does not exist at: {image_path}")

    # Step 3: Read the cropped image using OpenCV
    image_bgr = cv2.imread(image_path)
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError(f"Failed to read image. The file may be corrupted or not a valid image: {image_path}")

    # Step 4: Extract 512-D L2-normalized embedding via shared pipeline
    embedding_vector = extract_face_embedding(image_bgr)
    dimension = len(embedding_vector)

    # Convert numpy array to standard Python list for database compatibility
    embedding_list = [float(x) for x in embedding_vector]

    output_filepath = None
    # Step 5: Optionally save to a JSON file inside backend/embeddings/
    if save_json:
        embedding_data = {
            "user_id": clean_user_id,
            "embedding_dimension": dimension,
            "embedding": embedding_list
        }
        output_filename = f"{clean_user_id}.json"
        output_filepath = os.path.join(EMBEDDINGS_DIR, output_filename)
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(embedding_data, f, indent=4)
        print(f"[SAVED] Stored embedding copy in: {output_filepath}")

    print(f"[SUCCESS] Generated 512-D embedding for User ID: {clean_user_id}")

    return {
        "user_id": clean_user_id,
        "embedding_dimension": dimension,
        "embedding_file": output_filepath,
        "embedding": embedding_list
    }


# ==============================================================================
# Standalone Execution / Test Block
# ==============================================================================

if __name__ == "__main__":
    print("-" * 60)
    print("Testing FaceNet embedding generation...")
    print("-" * 60)

    # Test with existing test image: backend/cropped_faces/1234_face.jpg and user ID 1234
    test_user_id = "1234"
    test_image_path = os.path.join(BASE_DIR, "cropped_faces", "1234_face.jpg")

    try:
        result = generate_embedding(user_id=test_user_id, image_path=test_image_path)
        print(f"\nUser ID: {result['user_id']}")
        print(f"Embedding Vector Dimensions: {result['embedding_dimension']}")
        print(f"Sample Embedding Values (first 5 elements): {result['embedding'][:5]}")
        print(f"Saved JSON File: {result['embedding_file']}")
    except Exception as e:
        print(f"[ERROR] Failed to generate embedding: {e}")

    # Demonstration of error handling with an invalid image path
    print("\n" + "-" * 60)
    print("Testing error handling with an invalid image path...")
    print("-" * 60)
    invalid_path = os.path.join(BASE_DIR, "cropped_faces", "non_existent.jpg")
    try:
        generate_embedding(user_id="9999", image_path=invalid_path)
    except Exception as e:
        print(f"[HANDLED ERROR as expected]: {e}")

