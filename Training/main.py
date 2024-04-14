from flask import Flask, request, jsonify
import numpy as np
import cv2
import face_recognition
from pymongo import MongoClient, errors
from functools import wraps
import jwt
import os
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

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
connection_string= os.getenv('MONGO_URI')
database_name = os.getenv('DATABASE_NAME')
db = connect_to_mongodb_atlas(connection_string, database_name) 
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

@app.route('/health', methods=['GET'])
def health_check():
    # Simple health check endpoint
    return jsonify({"status": "UP"}), 200


@app.route('/process_images', methods=['POST'])
@token_required
def process_images():
    # The client should send the images as a list of lists (image arrays serialized into JSON)
    data = request.get_json()
 
    
    if not data or 'images' not in data:
        return jsonify({"error": "No data provided"}), 400
    # print("Images data type:", type(data['images']))  # Check the data type of 'images'
    # print("First element type:", type(data['images']["Santosh"]))
    response_data = {}
    try:
        # Convert the received data back into numpy arrays
     
        for name, images_list in data['images'].items():
            images = [np.array(image, dtype=np.uint8) for image in images_list]
            featuresOfImages, face_locations_messages = features(images)
            response_data[name] = [features.tolist() for features in featuresOfImages]
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def features(images):
    featuresOfImages = []
    face_locations_messages = []
    for img in images:
        imgs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(imgs)
        if len(face_locations) > 0:
            # face_locations_messages.append(str(face_locations[0]))
            print(face_locations[0])
        try:
            featuresOfImg = face_recognition.face_encodings(imgs, model='cnn')[0]
            featuresOfImages.append(featuresOfImg)
            print("faceencoding is done")
        except IndexError as e:
            face_locations_messages.append("Some Faces are not detected by dlib")
    
    return np.array(featuresOfImages), face_locations_messages

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8003)

        