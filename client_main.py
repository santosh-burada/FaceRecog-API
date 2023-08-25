from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import requests
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

API_ENDPOINT = "http://127.0.0.1:8000/crop_face/"  # Endpoint of main.py for face cropping

SAVED_IMAGES_BASE_DIR = "saved_cropped_faces"
os.makedirs(SAVED_IMAGES_BASE_DIR, exist_ok=True)


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/save_image/")
async def save_cropped_image(username: str):
    logging.info(f"Received username: {username}")

    # Check if we have a recent cropped face image in memory (you can expand this logic)
    if not recent_cropped_image:
        logging.error("No recent cropped image found")
        raise HTTPException(status_code=404, detail="No recent cropped image found")

    # Save the cropped face in a directory with the user's name
    user_dir = os.path.join(SAVED_IMAGES_BASE_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_path = os.path.join(user_dir, f"cropped_face_{timestamp}.jpeg")
    with open(image_path, "wb") as img_file:
        img_file.write(recent_cropped_image)

    logging.info(f"Image saved at {image_path}")
    return {"status": "success", "message": "Image saved successfully"}


@app.websocket("/ws/camera/")
async def camera_feed(websocket: WebSocket):
    global recent_cropped_image  # to store the most recent cropped face image
    recent_cropped_image = None
    await websocket.accept()
    while True:
        frame_data = await websocket.receive_bytes()

        # Send frame to main.py for processing
        response = requests.post(API_ENDPOINT, files={'image': ('image.jpeg', frame_data)})

        # Save the cropped face locally and set it as recent_cropped_image
        if response.ok:
            recent_cropped_image = response.content
            await websocket.send_bytes(recent_cropped_image)
        else:
            logging.error(f"Error processing image: {response.text}")
            await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)