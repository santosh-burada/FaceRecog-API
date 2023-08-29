import numpy
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
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
        return JSONResponse(content={"error": "No face detected"}, status_code=400)
    print(cropped_face,cropped_face.shape)

    serialized_face = cropped_face.tobytes()

    headers = {
        "shape": f"{cropped_face.shape[0]},{cropped_face.shape[1]},{cropped_face.shape[2]}"
    }

    # return JSONResponse(content={"data": serialized_face,
    #                              "shape": f"{cropped_face.shape[0]},{cropped_face.shape[1]},{cropped_face.shape[2]}"},
    #                     headers=headers)
    return StreamingResponse(io.BytesIO(serialized_face), media_type="application/octet-stream", headers=headers)

