from flask import Flask, request, jsonify
import numpy as np
import cv2
import face_recognition
import json

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    # Simple health check endpoint
    return jsonify({"status": "UP"}), 200

@app.route('/process_images', methods=['POST'])
def process_images():
    # The client should send the images as a list of lists (image arrays serialized into JSON)
    data = request.get_json()
 
    
    if not data or 'images' not in data:
        return jsonify({"error": "No data provided"}), 400
    print("Images data type:", type(data['images']))  # Check the data type of 'images'
    print("First element type:", type(data['images']["Santosh"]))
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
    app.run(host='0.0.0.0', port=8002)

        