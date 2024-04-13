import os
import cv2
# import imutils
import pickle
import face_recognition
from pymongo import MongoClient, errors
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=5)
app = Flask(__name__)

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

# MongoDB setup
connection_string = "mongodb+srv://santuburada99:L7T3TUVD1KOkLtLJ@train-facerec.8dl5kmd.mongodb.net/?retryWrites=true&w=majority&ssl=true&appName=Train-faceRec"
database_name = "Train-faceRec"
db = connect_to_mongodb_atlas(connection_string, database_name)
train_collection = db["Training_Data"]
# existing_doc = train_collection.find_one({"_id": "santu.burada99@gmail.com"})
# features = existing_doc.get('features', {})
def fetch_document_async(collection, document_id):
    """Asynchronously fetch a document from MongoDB."""
    try:
        document_future = executor.submit(collection.find_one, {"_id": document_id})
        return document_future.result(timeout=5)  # Adjust timeout as necessary
    except Exception as e:
        print(f"Error fetching document: {e}")
        return None
def fetch_document_async(collection, document_id):
    """Fetch a document from MongoDB asynchronously."""
    try:
        return collection.find_one({"_id": document_id})
    except Exception as e:
        print(f"Error fetching document: {e}")
        return None

# print(features)

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

@app.route('/recognize', methods=['POST'])
def FaceRecognition():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_data = request.files['image'].read()
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    x, y, w, h = 0, 0, 0, 0

   
    # frame = imutils.resize(frame, width=600)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    # print(h,w)
    
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))
    
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    # print(detections.shape[2])
    c = 0
    k = 5
    recognized_faces = []
    best_match_name = "Unknown"  # Default value
    label = "fake"
    # loop over the detections
    existing_doc_future = executor.submit(fetch_document_async, train_collection, "santu.burada99@gmail.com")
    existing_doc = existing_doc_future.result()  # Wait for the background task to complete
    features = existing_doc.get('features', {}) if existing_doc else {}

    for i in range(0, detections.shape[2]):
        c += 1
        k += 1
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # print(confidence)

        # filter out weak detections
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the face and extract the face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # print(startX, startY, endX, endY)
            # # ensure the detected bounding box does fall outside the
            # # dimensions of the frame
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # # extract the face ROI and then preproces it in the exact
            # # same manner as our training data
            face = frame[startY:endY, startX:endX]
            # cv2.imshow("imae",face)
            faceLocation = face_recognition.face_locations(frame)
            # print(faceLocation)
            if len(faceLocation) > 0:

                inputFeatures = face_recognition.face_encodings(frame, faceLocation, model='cnn')
                # print(inputFeatures)
                   # Here we wil have the input face features
            #     # matching the input feature with the loaded images features.

                for input_encoding in inputFeatures:
                    # Use the find_best_match function to find the closest match in your database
                    best_match_name, distance = find_best_match(input_encoding, features)
                    # print(f"Match found: {best_match_name} with a distance of {distance}")
                
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
    app.run(debug=False, port=5005)
