import os
import json
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
print("FaceNet model loaded successfully.")


# ==============================================================================
# Embedding Generation Function
# ==============================================================================

def generate_embedding(user_id: str, image_path: str) -> dict:
    """
    Generates a 512-dimensional FaceNet embedding for a given cropped face image
    and saves the user ID and embedding into a JSON file.

    Parameters:
        user_id (str): The unique identifier for the user.
        image_path (str): File system path to the cropped face image.

    Returns:
        dict: A dictionary containing the user_id, embedding length, and output JSON path.
    """
    # Step 1: Validate file path existence
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file does not exist at: {image_path}")

    # Step 2: Read the cropped image using OpenCV
    image_bgr = cv2.imread(image_path)
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError(f"Failed to read image. The file may be corrupted or not a valid image: {image_path}")

    # Step 3: Resize image to 160x160 pixels (FaceNet's standard input dimensions)
    resized_image = cv2.resize(image_bgr, (160, 160))

    # Step 4: Convert color from BGR (OpenCV default) to RGB (FaceNet expected format)
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # Step 5: Generate face embedding using FaceNet
    # embedder.embeddings expects a list or batch of RGB images
    embeddings = embedder.embeddings([rgb_image])
    embedding_vector = embeddings[0]

    # Step 6: Verify 512-dimensional embedding output
    dimension = len(embedding_vector)
    if dimension != 512:
        raise ValueError(f"Expected 512-dimensional embedding, but got {dimension} dimensions.")

    # Convert numpy array to standard Python list for JSON serialization
    embedding_list = embedding_vector.tolist()

    # Step 7: Prepare data associating user_id with the embedding
    embedding_data = {
        "user_id": str(user_id),
        "embedding_dimension": dimension,
        "embedding": embedding_list
    }

    # Step 8: Save to a JSON file inside backend/embeddings/
    output_filename = f"{user_id}.json"
    output_filepath = os.path.join(EMBEDDINGS_DIR, output_filename)

    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(embedding_data, f, indent=4)

    print(f"[SUCCESS] Generated 512-D embedding for User ID: {user_id}")
    print(f"[SAVED] Stored embedding in: {output_filepath}")

    return {
        "user_id": str(user_id),
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

