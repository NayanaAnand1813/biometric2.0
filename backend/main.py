import cv2
import mediapipe as mp
import numpy as np
import os
from fastapi import FastAPI, UploadFile, File, Form

# Folder to store cropped faces
CROPPED_FACES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cropped_faces"
)

os.makedirs(CROPPED_FACES_DIR, exist_ok=True)

app = FastAPI()


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Biometric Backend is Running"}

@app.post("/upload")
async def upload_image(
    user_id: str = Form(...),
    image: UploadFile = File(...)
):
    # Read uploaded image
    image_data = await image.read()

    # Convert image bytes into OpenCV image
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    # Detect face
    detection_result = detector.detect(mp_image)

    if not detection_result.detections:
        return {"message": "No face detected"}

    # Get first detected face
    detection = detection_result.detections[0]
    bbox = detection.bounding_box

    x = max(0, bbox.origin_x)
    y = max(0, bbox.origin_y)
    w = bbox.width
    h = bbox.height

    # Crop face
    cropped_face = img[y:y+h, x:x+w]

    # Save cropped face
    output_path = os.path.join(
        CROPPED_FACES_DIR,
        f"{user_id}_face.jpg"
    )

    cv2.imwrite(output_path, cropped_face)

    return {
        "message": "Face detected and cropped successfully",
        "user_id": user_id,
        "saved_image": output_path
    }
# MediaPipe Face Detector
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "blaze_face_short_range.tflite"
        )
    ),
    running_mode=VisionRunningMode.IMAGE
)

detector = FaceDetector.create_from_options(options)

print("BlazeFace initialized successfully!")