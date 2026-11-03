import os
import sys
import threading
import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

# Ensure backend directory is in python search path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from user_service import (
    register_user_with_embedding,
    get_user_embedding,
    UserAlreadyExistsError,
    InvalidUserError
)

# Folder to store cropped faces
CROPPED_FACES_DIR = os.path.join(BASE_DIR, "cropped_faces")
os.makedirs(CROPPED_FACES_DIR, exist_ok=True)

# MediaPipe Face Detector Initialization
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

_detector_lock = threading.Lock()
options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path=os.path.join(BASE_DIR, "blaze_face_short_range.tflite")
    ),
    running_mode=VisionRunningMode.IMAGE
)
detector = FaceDetector.create_from_options(options)
print("BlazeFace initialized successfully!")

app = FastAPI(title="Biometric Face Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Biometric Backend is Running"}


def _process_registration_sync(clean_user_id: str, img: np.ndarray) -> dict:
    """
    Synchronous worker executed in worker threadpool to prevent event loop blocking:
    Performs face detection, cropping, FaceNet 512-D embedding generation,
    and PostgreSQL persistence in isolated transaction.
    """
    # 1. Convert BGR to RGB for MediaPipe
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    # 2. Thread-safe face detection
    with _detector_lock:
        detection_result = detector.detect(mp_image)

    if not detection_result.detections:
        return {"status": "no_face", "message": "No face detected"}

    # 3. Crop first detected face
    detection = detection_result.detections[0]
    bbox = detection.bounding_box
    x = max(0, bbox.origin_x)
    y = max(0, bbox.origin_y)
    w = max(1, bbox.width)
    h = max(1, bbox.height)

    cropped_face = img[y:y+h, x:x+w]
    if cropped_face.size == 0:
        return {"status": "no_face", "message": "No face detected"}

    # 4. Save cropped face image (isolated per user ID)
    output_path = os.path.join(CROPPED_FACES_DIR, f"{clean_user_id}_face.jpg")
    cv2.imwrite(output_path, cropped_face)

    # 5. Generate FaceNet 512-D embedding and store in PostgreSQL
    reg_result = register_user_with_embedding(
        user_id=clean_user_id,
        image_path=output_path,
        save_json=False
    )

    return {
        "status": "success",
        "message": "Face detected, embedding generated, and user registered successfully",
        "user_id": clean_user_id,
        "saved_image": output_path,
        "embedding_dimension": reg_result["embedding_dimension"],
        "created_at": reg_result["created_at"]
    }


@app.post("/upload")
@app.post("/register")
async def register_user_endpoint(
    user_id: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Registration endpoint:
    Receives user_id and face image, validates input, detects and crops face,
    generates 512-D FaceNet embedding, and commits record into PostgreSQL.
    Prevents duplicate user_id overwrites with 409 Conflict.
    """
    clean_user_id = user_id.strip() if user_id else ""
    if not clean_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid user_id", "message": "User ID cannot be empty."}
        )

    # Read uploaded image bytes
    image_data = await image.read()
    if not image_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Empty file", "message": "Uploaded image file is empty."}
        )

    # Decode image using OpenCV
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if img is None or img.size == 0:
        return {"error": "Invalid image", "message": "Failed to decode image"}

    # Offload detection, embedding generation, and DB insertion to threadpool
    try:
        result = await run_in_threadpool(_process_registration_sync, clean_user_id, img)
        return result
    except UserAlreadyExistsError as exist_err:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": "User already exists", "message": str(exist_err)}
        )
    except InvalidUserError as inv_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid user data", "message": str(inv_err)}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Registration failed", "message": str(e)}
        )


@app.get("/users/{user_id}")
async def get_user_endpoint(user_id: str):
    """
    Retrieves user registration details and embedding dimension from PostgreSQL.
    """
    user_data = await run_in_threadpool(get_user_embedding, user_id)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "Not Found", "message": f"User '{user_id}' not found."}
        )
    return user_data


@app.post("/login")
async def login_image_endpoint(
    user_id: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Milestone 6 Login Capture endpoint:
    Receives User ID and face image from the frontend login interface.
    Validates user_id and image integrity. Confirms receipt to frontend.
    Note: Face detection, embedding generation, FAISS similarity matching,
    and authentication decisions are strictly deferred to Milestone 7.
    """
    clean_user_id = user_id.strip() if user_id else ""
    if not clean_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid user_id", "message": "User ID cannot be empty."}
        )

    # Read uploaded image bytes
    image_data = await image.read()
    if not image_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Empty file", "message": "Uploaded image file is empty."}
        )

    # Decode and validate image using OpenCV
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if img is None or img.size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid image", "message": "Failed to decode image. Expected valid JPEG/PNG format."}
        )

    return {
        "status": "success",
        "message": f"Login image received successfully for User ID '{clean_user_id}'",
        "user_id": clean_user_id,
        "image_size_bytes": len(image_data)
    }

