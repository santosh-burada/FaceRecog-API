import io
import os

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, make_response
from werkzeug.datastructures import FileStorage
from werkzeug.utils import send_file

app = Flask(__name__)


def crop_face(image):
    """
    Crop the face from the input image using the Haar Cascade Classifier.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.

    Returns:
        numpy.ndarray: The cropped face as a NumPy array, or None if no face is detected.
    """
    face_classifier = cv2.CascadeClassifier("../DataSetsPre/haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        return image[y:y + h, x:x + w]

    return None


@app.route('/detectface', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return "No image found in request files", 400

    image_data = request.files['image'].read()
    image = Image.open(io.BytesIO(image_data))

    try:
        face = crop_face(np.array(image))
        if face is not None:
            face = cv2.resize(face, (450, 450))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        else:
            return "No face detected in the image", 400
    except Exception as e:
        return f"Error while processing image: {str(e)}", 400

    output = io.BytesIO()
    face_pil = Image.fromarray(face)
    face_pil = face_pil.resize((450, 450)).convert('L')
    face_pil.save(output, format='JPEG')
    output.seek(0)

    response = make_response(output.getvalue())
    response.headers.set('Content-Type', 'image/jpeg')
    response.headers.set('Content-Disposition', 'attachment', filename='processed_image.jpeg')
    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
