url = 'http://localhost:5000/face_recognition'

# Initialize video stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame to reduce size.
    frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    _, img_encoded = cv2.imencode('.jpg', frame_small)
    
    # Send frame to server
    response = requests.post(url, files={"image": img_encoded.tobytes()})
    
    # Print server response
    print(response.json())
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break

cap.release()