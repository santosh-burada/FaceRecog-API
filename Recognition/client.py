import cv2
import requests
import time
from concurrent.futures import ThreadPoolExecutor

server_url = 'http://192.168.1.159:30005/recognize'
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
executor = ThreadPoolExecutor(max_workers=5)  # Adjust max_workers as needed

def send_frame_async(frame):
    """Function to send a frame to the server asynchronously."""
    try:
        # Encode frame as JPEG to reduce size and send as image
        _, buffer = cv2.imencode('.jpg', frame)
        
        # Convert to bytes and send to server using a POST request
        response = requests.post(
            server_url,
            files={"image": buffer.tobytes()},
        )
        
        # Process the server response
        print(response.json())
    except Exception as e:
        print(f"Error sending frame: {e}")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Resize frame to reduce size and speed up transmission time
        height, width = frame.shape[:2]
        new_width = 600
        aspect_ratio = new_width / float(width)
        new_height = int(height * aspect_ratio)
        frame= cv2.resize(frame, (new_width, new_height))
        # Optional: Display the frame being sent
        cv2.imshow('Sending Frame', frame)
        
        # Use the executor to send the frame asynchronously
        executor.submit(send_frame_async, frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Throttle to avoid overwhelming the server and the executor
        time.sleep(0.2)
finally:
    cap.release()
    cv2.destroyAllWindows()
