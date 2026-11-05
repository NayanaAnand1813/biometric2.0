import os
import sys
import threading
import re
import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

# Ensure backend directory is in python search path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from user_service import (
    register_user_with_embedding,
    get_user_profile,
    get_user_embedding,
    get_all_users,
    delete_user_completely,
    UserAlreadyExistsError,
    EmailAlreadyExistsError,
    InvalidUserError,
    UserNotFoundError
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


@app.on_event("startup")
def startup_event():
    """
    On FastAPI server startup, initialize the in-memory FAISS IndexFlatIP(512)
    from PostgreSQL. PostgreSQL remains the permanent source of truth.
    """
    try:
        from faiss_service import get_faiss_manager
        faiss_mgr = get_faiss_manager()
        count = faiss_mgr.initialize_from_database()
        print(f"[STARTUP] Initialized FAISS index with {count} enrolled user embeddings.")
    except Exception as e:
        print(f"[STARTUP WARNING] Could not initialize FAISS from database: {e}")


@app.get("/")
def home():
    return {"message": "Biometric Backend is Running"}


EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
PHONE_REGEX = re.compile(r"^\+?[0-9\s\-\(\)]{7,20}$")


def _process_registration_sync(
    clean_user_id: str,
    full_name: str | None,
    email: str | None,
    role: str | None,
    phone_number: str | None,
    img: np.ndarray
) -> dict:
    """
    Synchronous worker executed in worker threadpool to prevent event loop blocking:
    Performs face detection, cropping, FaceNet 512-D embedding generation,
    and PostgreSQL persistence in an isolated transaction.
    Cleans up any cropped face image file if database or FAISS insertion fails.
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
    try:
        reg_result = register_user_with_embedding(
            user_id=clean_user_id,
            full_name=full_name,
            email=email,
            role=role,
            phone_number=phone_number,
            image_path=output_path
        )
    except Exception as reg_err:
        # Avoid partial registration: remove cropped face image if DB insertion or FAISS sync failed
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass
        raise reg_err

    return {
        "status": "success",
        "message": "Face detected, embedding generated, and user profile registered successfully",
        "user_id": clean_user_id,
        "full_name": reg_result.get("full_name", ""),
        "email": reg_result.get("email", ""),
        "role": reg_result.get("role", ""),
        "phone_number": reg_result.get("phone_number", ""),
        "saved_image": output_path,
        "embedding_dimension": reg_result["embedding_dimension"],
        "created_at": reg_result["created_at"]
    }


@app.post("/upload")
@app.post("/register")
async def register_user_endpoint(
    user_id: str = Form(...),
    full_name: str | None = Form(None),
    email: str | None = Form(None),
    role: str | None = Form(None),
    phone_number: str | None = Form(None),
    image: UploadFile = File(...)
):
    """
    Registration endpoint:
    Receives user_id, profile fields (full_name, email, role, phone_number), and face image.
    Validates input format, detects and crops face, generates 512-D FaceNet embedding,
    and commits record into PostgreSQL with FAISS synchronization.
    Prevents duplicate user_id and email overwrites with HTTP 409 Conflict.
    """
    clean_user_id = user_id.strip() if user_id else ""
    if not clean_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid user_id", "message": "User ID cannot be empty."}
        )

    # Check and validate profile fields if any profile field is provided
    has_profile_input = any(f is not None for f in (full_name, email, role, phone_number))
    clean_full_name: str | None = None
    clean_email: str | None = None
    clean_role: str | None = None
    clean_phone: str | None = None

    if has_profile_input:
        clean_full_name = full_name.strip() if full_name else ""
        clean_email = email.strip().lower() if email else ""
        clean_role = role.strip() if role else ""
        clean_phone = phone_number.strip() if phone_number else ""

        if not clean_full_name or not clean_email or not clean_role or not clean_phone:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Missing fields",
                    "message": "All five fields (User ID, Full Name, Work Email, Role, Phone Number) are mandatory."
                }
            )

        if not EMAIL_REGEX.match(clean_email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Invalid email", "message": "Invalid email format. Please provide a valid work email address."}
            )

        digits_only = re.sub(r"\D", "", clean_phone)
        if not PHONE_REGEX.match(clean_phone) or len(digits_only) < 7:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Invalid phone", "message": "Invalid phone number format. Please provide a valid contact number."}
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
        result = await run_in_threadpool(
            _process_registration_sync,
            clean_user_id,
            clean_full_name,
            clean_email,
            clean_role,
            clean_phone,
            img
        )
        return result
    except UserAlreadyExistsError as exist_err:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": "User already exists", "message": str(exist_err)}
        )
    except EmailAlreadyExistsError as mail_err:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": "Email already exists", "message": str(mail_err)}
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


ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "admin_secret_key_2026")


def verify_admin_key(
    x_admin_key: str | None = Header(None, alias="X-Admin-Key"),
    authorization: str | None = Header(None)
):
    """
    Validates administrator credentials via X-Admin-Key header or Authorization: Bearer token.
    Restricts administrative endpoints (such as user deletion) to authorized personnel.
    """
    token = x_admin_key
    if not token and authorization:
        if authorization.startswith("Bearer "):
            token = authorization[7:].strip()
        else:
            token = authorization.strip()

    if not token or token != ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Unauthorized", "message": "Admin authorization required to perform this action."}
        )
    return True


@app.get("/users")
@app.get("/api/users")
async def list_users_endpoint():
    """
    Retrieves the list of all registered biometric users.
    Returns user_id, registration timestamp, and embedding presence.
    """
    try:
        users = await run_in_threadpool(get_all_users)
        return {"status": "success", "users": users, "total": len(users)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Database error", "message": str(e)}
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


@app.delete("/users/{user_id}")
@app.delete("/api/users/{user_id}")
async def delete_user_endpoint(
    user_id: str,
    _admin: bool = Depends(verify_admin_key)
):
    """
    Milestone 8 User Deletion Endpoint:
    Completely and safely deletes a registered user:
    - Verifies administrator authorization via X-Admin-Key or Authorization header.
    - Purges the user record and 512-D FaceNet embedding from PostgreSQL.
    - Removes their corresponding vector from the in-memory FAISS IndexFlatIP.
    - Deletes their cropped face images from disk.
    - Safely handles missing users (404) and database failures (500).
    - Preserves all other registered users and maintains PostgreSQL-FAISS synchronization.
    """
    clean_user_id = user_id.strip() if user_id else ""
    if not clean_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid user_id", "message": "User ID cannot be empty."}
        )

    try:
        result = await run_in_threadpool(delete_user_completely, clean_user_id)
        return result
    except UserNotFoundError as nf_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "Not Found", "message": str(nf_err)}
        )
    except InvalidUserError as inv_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid user data", "message": str(inv_err)}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Deletion failed", "message": str(e)}
        )


from generate_embedding import extract_face_embedding
from faiss_service import get_faiss_manager


def get_biometric_threshold() -> float:
    """
    Retrieves the configurable cosine similarity threshold.
    NOTE: The default (0.70) is an empirical test threshold and MUST undergo
    formal empirical calibration (FAR/FRR/EER evaluation using genuine and impostor
    distributions) prior to production deployment.
    """
    raw_val = os.getenv("BIOMETRIC_COSINE_THRESHOLD", "0.70")
    try:
        val = float(raw_val)
        if not (0.0 < val < 1.0):
            raise ValueError("Threshold must be strictly between 0.0 and 1.0.")
        return val
    except Exception as e:
        print(f"[CONFIG WARNING] Invalid BIOMETRIC_COSINE_THRESHOLD '{raw_val}': {e}. Falling back to 0.70.")
        return 0.70


def _process_login_sync(clean_user_id: str | None, img: np.ndarray, image_size_bytes: int) -> dict:
    """
    Synchronous worker executed in worker threadpool for biometric login:
    1. BlazeFace face detection in memory (thread-safe under _detector_lock).
    2. In-memory face cropping (never saved to disk).
    3. FaceNet 512-D L2-normalized embedding extraction via shared extract_face_embedding.
    4. Exact 1:N FAISS IndexFlatIP cosine similarity search under _lock.
    5. Identity matching:
       - If clean_user_id is None (1:N identification): Top matching FAISS identity
         is verified against the biometric threshold. Unknown/unregistered faces are rejected.
       - If clean_user_id is provided (1:1 verification): Top matching FAISS user_id
         must equal clean_user_id and similarity must meet or exceed threshold.
    6. Return clean authenticated / denied response (no raw similarity scores exposed).
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
        return {
            "status": "error",
            "error_type": "no_face",
            "message": "No face detected in the captured image. Please center your face and try again."
        }

    # 3. Crop first detected face in memory (NO disk write)
    detection = detection_result.detections[0]
    bbox = detection.bounding_box
    x = max(0, bbox.origin_x)
    y = max(0, bbox.origin_y)
    w = max(1, bbox.width)
    h = max(1, bbox.height)

    cropped_face = img[y:y+h, x:x+w]
    if cropped_face.size == 0:
        return {
            "status": "error",
            "error_type": "no_face",
            "message": "No face detected in the captured image."
        }

    # 4. Extract 512-D L2-normalized embedding using shared pipeline (in memory)
    try:
        probe_embedding = extract_face_embedding(cropped_face)
    except Exception as emb_err:
        return {
            "status": "error",
            "error_type": "embedding_failed",
            "message": f"Failed to extract biometric features: {emb_err}"
        }

    # 5. Search FAISS index for top matching enrolled identity
    faiss_mgr = get_faiss_manager()
    top_matches = faiss_mgr.search_top_k(probe_embedding, k=1)
    threshold = get_biometric_threshold()

    if not top_matches:
        # FAISS index is empty or has no enrolled users
        claimed_info = f" for claimed ID '{clean_user_id}'" if clean_user_id else ""
        print(f"[AUTH DENIED] FAISS index is empty. Denying authentication{claimed_info}.")
        return {
            "status": "failed",
            "authenticated": False,
            "message": "Biometric verification failed. Face does not match registered user."
        }

    top_user_id, similarity = top_matches[0]

    # 6. Verify identity match
    if clean_user_id:
        # Dual-gate 1:1 check: Top match must match claimed ID AND score >= threshold
        is_authenticated = (top_user_id == clean_user_id and similarity >= threshold)
        matched_user = clean_user_id
    else:
        # 1:N Identification check: Top match must meet or exceed threshold
        is_authenticated = (similarity >= threshold)
        matched_user = top_user_id

    if is_authenticated:
        profile = get_user_profile(matched_user)
        if not profile:
            profile = {
                "user_id": matched_user,
                "full_name": "",
                "email": "",
                "role": "",
                "phone_number": "",
                "created_at": ""
            }
        display_name = profile.get("full_name") or matched_user
        print(f"[AUTH SUCCESS] User '{matched_user}' authenticated. Score: {similarity:.4f} >= Threshold: {threshold:.4f}")
        return {
            "status": "success",
            "authenticated": True,
            "user_id": matched_user,
            "user_profile": profile,
            "message": f"Biometric authentication successful. Welcome, {display_name}!",
            "image_size_bytes": image_size_bytes
        }
    else:
        print(
            f"[AUTH DENIED] Claimed: '{clean_user_id}', Top Match: '{top_user_id}', "
            f"Similarity: {similarity:.4f}, Threshold: {threshold:.4f}"
        )
        return {
            "status": "failed",
            "authenticated": False,
            "message": "Biometric verification failed. Face does not match registered user."
        }


@app.post("/login")
async def login_endpoint(
    user_id: str | None = Form(None),
    image: UploadFile = File(...)
):
    """
    Biometric Authentication Endpoint:
    Supports both 1:N face identification (no user_id required) and 1:1 claimed identity verification.
    Offloads in-memory face detection, shared FaceNet embedding, and FAISS exact
    search to the worker threadpool.
    Does not save login images or embeddings to disk or database.
    Does not expose raw similarity scores to the client.
    """
    clean_user_id: str | None = None
    if user_id is not None:
        trimmed = user_id.strip()
        if not trimmed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Invalid user_id", "message": "User ID cannot be empty."}
            )
        clean_user_id = trimmed

    # Read uploaded image bytes
    image_data = await image.read()
    if not image_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Empty file", "message": "Uploaded image file is empty."}
        )

    # Decode and validate image in memory using OpenCV (no disk write)
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if img is None or img.size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid image", "message": "Failed to decode image. Expected valid JPEG/PNG format."}
        )

    # Offload CPU-bound face detection, FaceNet inference, and FAISS search to threadpool
    result = await run_in_threadpool(_process_login_sync, clean_user_id, img, len(image_data))

    if result.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": result.get("error_type", "Processing error"), "message": result.get("message")}
        )

    if not result.get("authenticated", False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Authentication failed",
                "authenticated": False,
                "message": result.get("message", "Biometric verification failed. Face does not match registered user.")
            }
        )

    return result

