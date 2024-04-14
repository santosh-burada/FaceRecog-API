import os
import cv2
import pickle
import jwt
import face_recognition
from pymongo import MongoClient, errors
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

executor = ThreadPoolExecutor(max_workers=5)
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

print("Packages are Imported")

print("[INFO] loading face detector...")
protoPath = os.path.sep.join(['face_detector', "deploy.prototxt"])
modelPath = os.path.sep.join(['face_detector',
                                "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")
model = load_model(os.path.sep.join(['model', "liveness.model"]))
le = pickle.loads(open(os.path.sep.join(['model', "le.pickle"]), "rb").read())
# Entries expire after 5 minutes

def connect_to_mongodb_atlas(connection_string, database_name):
    """Connect to MongoDB Atlas and return the database object."""
    try:
        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)  # 5-second timeout
        db = client[database_name]
        db.command("ping")  # Quick operation to check the connection
        print("Connected successfully to MongoDB Atlas.")
        return db
    except errors.ConnectionFailure as e:
        print(f"Failed to connect to MongoDB Atlas: {e}")
        exit()

connection_string = os.getenv('MONGO_URI')
database_name = os.getenv('DATABASE_NAME')
db = connect_to_mongodb_atlas(connection_string, database_name)
train_collection = db["Training_Data"]
users_collection = db.users


def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        email = request.headers.get('Email')
        if not email:
            return jsonify({'message': 'Email header is missing!'}), 403
        
        user = users_collection.find_one({"email": email})
        if not user:
            return jsonify({'message': 'User not found!'}), 404
        
        token = user.get('token')
        if not token:
            return jsonify({'message': 'Token not found for user!'}), 403
        
        try:
            jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token expired, please log in again.'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token. Please log in again.'}), 403

        return f(*args, **kwargs)

    return decorated_function

def fetch_document_async(collection, document_id):
    """Asynchronously fetch a document from MongoDB."""
    try:
        document_future = executor.submit(collection.find_one, {"_id": document_id})
        return document_future.result(timeout=5)  # Adjust timeout as necessary
    except Exception as e:
        print(f"Error fetching document: {e}")
        return None

def find_best_match(input_encoding, features):
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

@app.route('/health', methods=['GET'])
def health_check():
    # Simple health check endpoint
    return jsonify({"status": "UP"}), 200

@app.route('/recognize', methods=['POST'])
@token_required
def FaceRecognition():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_data = request.files['image'].read()
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    x, y, w, h = 0, 0, 0, 0

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    # print(h,w)
    
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))
    
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    c = 0
    k = 5
    recognized_faces = []
    best_match_name = "Unknown"  # Default value
    label = "fake"
    # loop over the detections
    email = request.headers.get('Email')
    existing_doc_future = executor.submit(fetch_document_async, train_collection, email)
    existing_doc = existing_doc_future.result()  # Wait for the background task to complete
    features = existing_doc.get('features', {}) if existing_doc else {}

    for i in range(0, detections.shape[2]):
        c += 1
        k += 1
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the face and extract the face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # # ensure the detected bounding box does fall outside the
            # # dimensions of the frame
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # # extract the face ROI and then preproces it in the exact
            # # same manner as our training data
            face = frame[startY:endY, startX:endX]
            if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
                continue 

            faceLocation = face_recognition.face_locations(frame)

            if len(faceLocation) > 0:

                inputFeatures = face_recognition.face_encodings(frame, faceLocation, model='cnn')
                   # Here we wil have the input face features
            #     # matching the input feature with the loaded images features.

                for input_encoding in inputFeatures:
                    # Use the find_best_match function to find the closest match in your database
                    best_match_name, distance = find_best_match(input_encoding, features)
                
                face = cv2.resize(face, (32, 32))
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)

                preds = model.predict(face)[0]
                print(preds)
                j = np.argmax(preds)
                label = le.classes_[j]
                if label =="fake":
                    label="real"
                else:
                    label="fake"
            recognized_faces.append({
                "name": best_match_name,
                "liveness": label,
                "confidence": float(confidence)
            })
    return jsonify({"faces": recognized_faces})

if __name__ == "__main__":
    app.run(debug=False, port=8005)
