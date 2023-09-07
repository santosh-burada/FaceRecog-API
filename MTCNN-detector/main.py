from flask import Flask, request, Response, jsonify
from mtcnn.mtcnn import MTCNN
import numpy as np
import cv2

app = Flask(__name__)


@app.route('/crop_face_mtcnn', methods=['POST'])
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


if __name__ == '__main__':
    app.run(port=8001)
