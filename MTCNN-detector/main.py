from flask import Flask, request, Response, jsonify
from mtcnn.mtcnn import MTCNN
import numpy as np
import cv2
import os
from pymongo import MongoClient, errors
from functools import wraps
import jwt
from dotenv import load_dotenv

load_dotenv()


app = Flask(__name__)
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

@app.route('/crop_face_mtcnn', methods=['POST'])
@token_required
def crop_face_mtcnn():
    try:
        image_data = request.files['image'].read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        detector = MTCNN()

        results = detector.detect_faces(image)

        if results:
            x, y, w, h = results[0]['box']
            cropped_face = image[y:y + h, x:x + w]
            gray = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
            print(cropped_face)
            serialized_face = gray.tobytes()

            headers = {
                "Content-Type": "application/octet-stream",
                "shape": f"{gray.shape[0]},{gray.shape[1]}"
            }

            return Response(serialized_face, headers=headers)

        else:
            return jsonify({"error": "No face detected"}), 400
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    # Simple health check endpoint
    return jsonify({"status": "UP"}), 200

if __name__ == '__main__':
    app.run(port=8001)
