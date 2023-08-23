from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pathlib import Path
import cv2
import numpy as np
import io

# Initialize the FastAPI app
app = FastAPI()

# CORS middleware settings (Optional, only if you want to enable CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Function to crop face; assumes this function already exists and works as intended
def crop_face(image: np.ndarray):
    script_path = Path(__file__)
    cascade_path = script_path.parent.parent / "DataSetsPre" / "haarcascade_frontalface_default.xml"
    face_classifier = cv2.CascadeClassifier(str(cascade_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        return image[y:y + h, x:x + w]

    return None


@app.post("/crop_face/")
async def crop_face_endpoint(image: UploadFile = File(...)):
    image_stream = io.BytesIO(await image.read())
    image = Image.open(image_stream)
    image_np = np.array(image)

    cropped_face = crop_face(image_np)
    if cropped_face is None:
        return {"error": "No face detected"}

    # Convert the NumPy array back to a PIL Image
    img_pil = Image.fromarray(cropped_face)

    # Save PIL Image to BytesIO object and get the byte array
    img_byte_arr = io.BytesIO()
    img_pil.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    img_byte_io = io.BytesIO(img_byte_arr)
    return StreamingResponse(img_byte_io, media_type="image/png")

