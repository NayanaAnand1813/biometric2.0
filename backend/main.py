
from fastapi import FastAPI, UploadFile, File, Form

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
    image_data = await image.read()

    print("User ID:", user_id)
    print("Image received:", image.filename)
    print("Image size:", len(image_data), "bytes")

    return {
        "message": "Image received successfully",
        "user_id": user_id,
        "filename": image.filename,
        "image_size": len(image_data)
    }
