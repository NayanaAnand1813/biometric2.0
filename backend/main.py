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
from liveness_service import get_liveness_service, get_liveness_threshold
from liveness_session import get_liveness_session_manager

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
    On FastAPI server startup:
    1. Initialize in-memory FAISS IndexFlatIP(512) from PostgreSQL.
    2. Initialize MiniFASNetV1SE liveness model (fail-closed security).
    """
    try:
        from faiss_service import get_faiss_manager
        faiss_mgr = get_faiss_manager()
        count = faiss_mgr.initialize_from_database()
        print(f"[STARTUP] Initialized FAISS index with {count} enrolled user embeddings.")
    except Exception as e:
        print(f"[STARTUP WARNING] Could not initialize FAISS from database: {e}")

    try:
        liveness_svc = get_liveness_service()
        if liveness_svc.is_available():
            print(f"[STARTUP] MiniFASNetV1SE liveness model loaded (REAL threshold: {get_liveness_threshold()}).")
        else:
            print("[STARTUP CRITICAL] MiniFASNet liveness service failed to load. Pipeline will fail closed.")
    except Exception as e:
        print(f"[STARTUP CRITICAL] Could not initialize liveness service: {e}")


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
    Synchronous worker executed in worker threadpool enforcing Liveness-First registration:
    Step 2: MiniFASNetV1SE face anti-spoofing liveness check.
            Rejects fake, replay, uncertain, or low-confidence inputs immediately.
    Step 3: Exact verified frame retained upon REAL confirmation.
    Step 4: MediaPipe BlazeFace face detection and cropping.
    Step 5: FaceNet 512-D embedding extraction via shared pipeline.
    Step 6: PostgreSQL atomic persistence and FAISS IndexFlatIP synchronization.
    """
    # --------------------------------------------------------------------------
    # Step 2: MiniFASNet Liveness Detection
    # --------------------------------------------------------------------------
    liveness_svc = get_liveness_service()
    liveness_res = liveness_svc.verify_liveness(img)

    if liveness_res.get("status") == "no_face":
        return {
            "status": "no_face",
            "message": "No face detected in camera frame. Please align your face with the camera.",
            "liveness": liveness_res
        }

    if not liveness_res.get("is_real", False):
        return {
            "status": "liveness_failed",
            "message": liveness_res.get("message", "Liveness check failed: Real face could not be verified."),
            "liveness": liveness_res
        }

    # --------------------------------------------------------------------------
    # Step 3: Capture / Retain Verified Image (exact frame that passed liveness)
    # --------------------------------------------------------------------------
    verified_img = img

    # --------------------------------------------------------------------------
    # Step 4: MediaPipe BlazeFace Face Detection and Cropping
    # --------------------------------------------------------------------------
    rgb_image = cv2.cvtColor(verified_img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    with _detector_lock:
        detection_result = detector.detect(mp_image)

    if not detection_result.detections:
        return {
            "status": "no_face",
            "message": "MediaPipe BlazeFace did not detect a valid face in the verified frame.",
            "liveness": liveness_res
        }

    detection = detection_result.detections[0]
    bbox = detection.bounding_box
    x = max(0, bbox.origin_x)
    y = max(0, bbox.origin_y)
    w = max(1, bbox.width)
    h = max(1, bbox.height)

    cropped_face = verified_img[y:y+h, x:x+w]
    if cropped_face.size == 0:
        return {
            "status": "no_face",
            "message": "Invalid face crop from verified frame.",
            "liveness": liveness_res
        }

    # --------------------------------------------------------------------------
    # Step 5 & 6: FaceNet Embedding Generation & PostgreSQL / FAISS Registration
    # --------------------------------------------------------------------------
    output_path = os.path.join(CROPPED_FACES_DIR, f"{clean_user_id}_face.jpg")
    cv2.imwrite(output_path, cropped_face)

    try:
        reg_result = register_user_with_embedding(
            user_id=clean_user_id,
            full_name=full_name,
            email=email,
            role=role,
            phone_number=phone_number,
            face_bgr=cropped_face
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
        "message": "Real face verified, embedding generated, and user profile registered successfully",
        "user_id": clean_user_id,
        "full_name": reg_result.get("full_name", ""),
        "email": reg_result.get("email", ""),
        "role": reg_result.get("role", ""),
        "phone_number": reg_result.get("phone_number", ""),
        "saved_image": output_path,
        "embedding_dimension": reg_result["embedding_dimension"],
        "created_at": reg_result["created_at"],
        "liveness": liveness_res
    }


@app.post("/api/liveness/session/start")
async def start_liveness_session_endpoint(
    action: str = Form(...),
    user_id: str | None = Form(None)
):
    """
    Step 1: Starts a backend-issued liveness verification session for live webcam capture.
    Enforces camera-only face registration and login workflows.
    """
    clean_action = action.strip().lower() if action else ""
    if clean_action not in ("register", "login"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_action", "message": "Action must be 'register' or 'login'."}
        )

    clean_user = user_id.strip() if user_id and user_id.strip() else None
    session_mgr = get_liveness_session_manager()
    session = session_mgr.create_session(action=clean_action, user_id=clean_user)

    return {
        "status": "success",
        "message": f"Live webcam liveness session started for '{clean_action}'.",
        "session": session.to_dict()
    }


@app.post("/api/liveness/session/verify-frame")
async def verify_liveness_frame_endpoint(
    session_id: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Step 2: Receives live webcam frames during an active session and executes
    MiniFASNetV1SE anti-spoofing verification.
    Only updates session to 'verified' when a genuine real face meets or exceeds threshold.
    """
    clean_session_id = session_id.strip() if session_id else ""
    session_mgr = get_liveness_session_manager()
    session = session_mgr.get_session(clean_session_id)

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "session_not_found", "message": "Liveness session not found or expired. Please restart the camera."}
        )

    if session.is_expired():
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail={"error": "session_expired", "message": "Liveness session has expired. Please restart the camera."}
        )

    if session.is_consumed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "session_consumed", "message": "Liveness session was already used."}
        )

    image_data = await image.read()
    if not image_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "empty_frame", "message": "Webcam frame data is empty."}
        )

    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if img is None or img.size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_frame", "message": "Failed to decode camera frame."}
        )

    liveness_svc = get_liveness_service()
    liveness_res = await run_in_threadpool(liveness_svc.verify_liveness, img)

    is_verified, msg, session_dict = session_mgr.record_verification_result(clean_session_id, liveness_res)

    return {
        "status": liveness_res.get("status"),
        "is_real": liveness_res.get("is_real", False),
        "score": liveness_res.get("score", 0.0),
        "threshold": liveness_res.get("threshold", 0.85),
        "label": liveness_res.get("label", "Unknown"),
        "message": msg,
        "verified": is_verified,
        "session": session_dict
    }


@app.post("/register")
async def register_user_endpoint(
    session_id: str = Form(...),
    user_id: str = Form(...),
    full_name: str | None = Form(None),
    email: str | None = Form(None),
    role: str | None = Form(None),
    phone_number: str | None = Form(None),
    image: UploadFile = File(...)
):
    """
    Registration endpoint:
    Enforces camera-only face registration following successful live liveness detection.
    Direct API requests without a verified backend liveness session are strictly rejected (HTTP 403).
    Receives user_id, profile fields (full_name, email, role, phone_number), and the verified live webcam frame.
    Validates input format, detects and crops face, generates 512-D FaceNet embedding,
    and commits record into PostgreSQL with FAISS synchronization.
    Prevents duplicate user_id and email overwrites with HTTP 409 Conflict.
    """
    session_mgr = get_liveness_session_manager()
    is_valid_session, session_err, session = session_mgr.validate_and_consume(
        session_id=session_id,
        expected_action="register"
    )
    if not is_valid_session:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "liveness_verification_required", "message": session_err}
        )

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
        if result.get("status") == "liveness_failed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "liveness_failed",
                    "message": result.get("message"),
                    "liveness": result.get("liveness")
                }
            )
        if result.get("status") == "no_face":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "no_face",
                    "message": result.get("message"),
                    "liveness": result.get("liveness")
                }
            )
        return result
    except HTTPException:
        raise
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


def _process_login_sync(img: np.ndarray, image_size_bytes: int) -> dict:
    """
    Synchronous worker executed in worker threadpool for biometric 1:N identification enforcing
    Liveness-First verification:
    Step 2: MiniFASNetV1SE face anti-spoofing liveness check.
            Rejects spoof, fake photograph, screen replay, or low-confidence inputs immediately.
            MediaPipe BlazeFace, FaceNet, and FAISS are strictly NOT called if liveness fails.
    Step 3: Exact verified frame retained upon REAL confirmation.
    Step 4: MediaPipe BlazeFace face detection and in-memory cropping (never saved to disk).
    Step 5: FaceNet 512-D L2-normalized embedding extraction via shared extract_face_embedding.
    Step 6: Pure 1:N FAISS IndexFlatIP cosine similarity identification against all enrolled users.
            Retrieves matched user's profile from PostgreSQL on threshold match.
    """
    # --------------------------------------------------------------------------
    # Step 2: MiniFASNet Liveness Detection
    # --------------------------------------------------------------------------
    liveness_svc = get_liveness_service()
    liveness_res = liveness_svc.verify_liveness(img)

    if liveness_res.get("status") == "no_face":
        return {
            "status": "error",
            "error_type": "no_face",
            "authenticated": False,
            "message": "No face detected in the captured image. Please center your face and try again.",
            "liveness": liveness_res
        }

    if not liveness_res.get("is_real", False):
        return {
            "status": "liveness_failed",
            "error_type": "liveness_failed",
            "authenticated": False,
            "message": liveness_res.get("message", "Liveness verification failed: Real face could not be verified."),
            "liveness": liveness_res
        }

    # --------------------------------------------------------------------------
    # Step 3: Retain Verified Image (exact frame that passed liveness)
    # --------------------------------------------------------------------------
    verified_img = img

    # --------------------------------------------------------------------------
    # Step 4: MediaPipe BlazeFace Face Detection and In-Memory Cropping
    # --------------------------------------------------------------------------
    rgb_image = cv2.cvtColor(verified_img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    # Thread-safe face detection
    with _detector_lock:
        detection_result = detector.detect(mp_image)

    if not detection_result.detections:
        return {
            "status": "error",
            "error_type": "no_face",
            "authenticated": False,
            "message": "MediaPipe BlazeFace did not detect a face in the verified frame. Please center your face.",
            "liveness": liveness_res
        }

    # Crop first detected face in memory (NO disk write)
    detection = detection_result.detections[0]
    bbox = detection.bounding_box
    x = max(0, bbox.origin_x)
    y = max(0, bbox.origin_y)
    w = max(1, bbox.width)
    h = max(1, bbox.height)

    cropped_face = verified_img[y:y+h, x:x+w]
    if cropped_face.size == 0:
        return {
            "status": "error",
            "error_type": "no_face",
            "authenticated": False,
            "message": "Failed to crop face region from verified frame.",
            "liveness": liveness_res
        }

    # --------------------------------------------------------------------------
    # Step 5: FaceNet 512-D Embedding Generation
    # --------------------------------------------------------------------------
    try:
        probe_embedding = extract_face_embedding(cropped_face)
    except Exception as emb_err:
        return {
            "status": "error",
            "error_type": "embedding_failed",
            "authenticated": False,
            "message": f"Failed to extract biometric features: {emb_err}",
            "liveness": liveness_res
        }

    # --------------------------------------------------------------------------
    # Step 6: Pure 1:N FAISS Index Identification & Authentication
    # --------------------------------------------------------------------------
    faiss_mgr = get_faiss_manager()
    top_matches = faiss_mgr.search_top_k(probe_embedding, k=1)
    threshold = get_biometric_threshold()

    if not top_matches:
        print("[AUTH DENIED] FAISS index is empty. Denying authentication.")
        return {
            "status": "failed",
            "authenticated": False,
            "message": "Biometric verification failed. Face does not match registered user.",
            "liveness": liveness_res
        }

    top_user_id, similarity = top_matches[0]

    # Pure 1:N Identification check: Top match across all enrolled users must meet or exceed threshold
    is_authenticated = (similarity >= threshold)

    if is_authenticated:
        profile = get_user_profile(top_user_id)
        if not profile:
            profile = {
                "user_id": top_user_id,
                "full_name": "",
                "email": "",
                "role": "",
                "phone_number": "",
                "created_at": ""
            }
        display_name = profile.get("full_name") or top_user_id
        print(f"[AUTH SUCCESS] User '{top_user_id}' identified via 1:N search. Score: {similarity:.4f} >= Threshold: {threshold:.4f}")
        return {
            "status": "success",
            "authenticated": True,
            "user_id": top_user_id,
            "user_profile": profile,
            "message": f"Biometric authentication successful. Welcome, {display_name}!",
            "image_size_bytes": image_size_bytes,
            "liveness": liveness_res
        }
    else:
        print(
            f"[AUTH DENIED] Top Match: '{top_user_id}', "
            f"Similarity: {similarity:.4f} < Threshold: {threshold:.4f}. Unknown or unverified face."
        )
        return {
            "status": "failed",
            "authenticated": False,
            "message": "Biometric verification failed. Face does not match registered user.",
            "liveness": liveness_res
        }


@app.post("/login")
async def login_endpoint(
    session_id: str = Form(...),
    image: UploadFile = File(...)
):
    """
    1:N Biometric Authentication Endpoint:
    Identifies the individual exclusively through 1:N facial search across the enrolled FAISS index.
    Does not require or accept User ID, username, or email.
    Enforces camera-only face login following successful live liveness detection.
    Direct API requests without a verified backend liveness session are strictly rejected (HTTP 403).
    Offloads in-memory face detection, shared FaceNet embedding, and FAISS exact
    1:N search to the worker threadpool.
    Does not save login images or embeddings to disk or database.
    Does not expose raw similarity scores to the client.
    """
    session_mgr = get_liveness_session_manager()
    is_valid_session, session_err, session = session_mgr.validate_and_consume(
        session_id=session_id,
        expected_action="login"
    )
    if not is_valid_session:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "liveness_verification_required", "message": session_err}
        )

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

    # Offload CPU-bound face detection, FaceNet inference, and FAISS 1:N search to threadpool
    result = await run_in_threadpool(_process_login_sync, img, len(image_data))

    if result.get("status") == "liveness_failed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "liveness_failed",
                "authenticated": False,
                "message": result.get("message"),
                "liveness": result.get("liveness")
            }
        )

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


@app.post("/check-liveness")
@app.post("/api/liveness")
async def check_liveness_endpoint(image: UploadFile = File(...)):
    """
    Dedicated endpoint to evaluate face anti-spoofing liveness using MiniFASNetV1SE.
    Enables pre-verification checks without invoking FaceNet, database mutations,
    or FAISS operations.
    """
    image_data = await image.read()
    if not image_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Empty file", "message": "Uploaded image file is empty."}
        )

    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if img is None or img.size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid image", "message": "Failed to decode image. Expected valid JPEG/PNG."}
        )

    liveness_svc = get_liveness_service()
    res = await run_in_threadpool(liveness_svc.verify_liveness, img)
    return res


