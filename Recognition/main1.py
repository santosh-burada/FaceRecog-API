import os
import numpy as np
import cv2
import pickle
import face_recognition
from pymongo import MongoClient, errors
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


# Constants for file paths
FACE_DETECTOR_PATH = ['face_detector', "deploy.prototxt"]
FACE_MODEL_PATH = ['face_detector', "res10_300x300_ssd_iter_140000.caffemodel"]
LIVENESS_MODEL_PATH = ['model', "liveness.model"]
LABEL_ENCODER_PATH = ['model', "le.pickle"]
DATABASE_URI = "mongodb+srv://santuburada99:L7T3TUVD1KOkLtLJ@train-facerec.8dl5kmd.mongodb.net/?retryWrites=true&w=majority&appName=Train-faceRec"
DATABASE_NAME = "Train-faceRec"

# Initialize Flask app
app = Flask(__name__)

# Load face detector
print("[INFO] Loading face detector...")
net = cv2.dnn.readNetFromCaffe(os.path.sep.join(FACE_DETECTOR_PATH), os.path.sep.join(FACE_MODEL_PATH))

# Load liveness detector model and label encoder
print("[INFO] Loading liveness detector...")
model = load_model(os.path.sep.join(LIVENESS_MODEL_PATH))
le = pickle.loads(open(os.path.sep.join(LABEL_ENCODER_PATH), "rb").read())

def connect_to_mongodb(uri):
    """Connect to MongoDB and return the database object."""
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        db = client[DATABASE_NAME]
        db.command("ping")
        print("Connected successfully to MongoDB.")
        return db
    except errors.ConnectionFailure as e:
        print(f"Failed to connect to MongoDB: {e}")
        exit()

# Connect to MongoDB
db = connect_to_mongodb(DATABASE_URI)
train_collection = db["Training_Data"]

def find_best_match(input_encoding, features):
    """Find the best match for the input encoding in the features dictionary."""
    min_distance = None
    best_match_name = None
    
    for name, encodings in features.items():
        # Ensure there are encodings to compare against
        if len(encodings) == 0:
            continue  # Skip this name if there are no encodings
        
        # Calculate distances from the input encoding to all encodings of this person
        distances = face_recognition.face_distance(encodings, input_encoding)
        
        # Proceed only if distances array is not empty
        if len(distances) > 0:
            closest_distance = np.min(distances)
            
            # Update the best match if this is the closest distance so far
            if min_distance is None or closest_distance < min_distance:
                min_distance = closest_distance
                best_match_name = name
    
    # You can adjust the threshold based on your requirements
    threshold = 0.6
    if min_distance is not None and min_distance <= threshold:
        return best_match_name, min_distance
    else:
        return "Unknown", None

@app.route('/recognize', methods=['POST'])
def recognize_face():
    """Face recognition endpoint."""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_data = np.frombuffer(request.files['image'].read(), np.uint8)
    frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    recognized_faces = []
    existing_doc = train_collection.find_one({"_id": "santu.burada99@gmail.com"})
    features = existing_doc.get('features', {}) if existing_doc else {}
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int").clip(min=0)
            face = frame[startY:endY, startX:endX]
            faceLocation = face_recognition.face_locations(frame)
            if faceLocation:
                inputFeatures = face_recognition.face_encodings(frame, faceLocation, model='cnn')
                for input_encoding in inputFeatures:
                    best_match_name, _ = find_best_match(input_encoding, features)
                    face = cv2.resize(face, (32, 32)).astype("float") / 255.0
                    face = np.expand_dims(img_to_array(face), axis=0)
                    preds = model.predict(face)[0]
                    label = le.classes_[np.argmax(preds)]
                    recognized_faces.append({
                        "name": best_match_name,
                        "liveness": "real" if label == "fake" else "fake",  # Assuming a mistake in the original label logic
                        "confidence": float(confidence)
                    })
    return jsonify({"faces": recognized_faces})

if __name__ == "__main__":
    app.run(debug=True, port=5005)
