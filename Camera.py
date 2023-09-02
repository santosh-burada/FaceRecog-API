import os
import pickle
from io import BytesIO
import cv2
import numpy as np
import requests


class FaceCapture:
    """Captures webcam frames, processes for face detection and cropping, then saves the data.

    Attributes:
        current_name (str): The name of the person whose face is being captured.
        face_data_list (list): List storing face data.
        folder_path (str): The path where the captured data will be stored.
    """

    def __init__(self):
        self.current_name = None
        self.face_data_list = []
        self.folder_path = "CapturedData"
        os.makedirs(self.folder_path, exist_ok=True)

    def capture_and_process_frames(self):
        """Capture webcam frames and process them for face cropping.
        Press 'c' to capture a frame and 's' to save the captured data to a file.
        Press 'Esc' to exit the program.

        The captured data will be stored in a file named <name>_data.pkl in the CapturedData folder.

        The data will be stored as a list of numpy arrays.

        Example:
            [
                array([[[  0,   0,   0],
                        [  0,   0,   0],
                        [  0,   0,   0],
                        ...,
                        [  0,   0,   0],
                        [  0,   0,   0],
                        [  0,   0,   0]],
                ...,
                array([[[  0,   0,   0],
                        [  0,   0,   0],
                        [  0,   0,   0],
                        ...,
                        [  0,   0,   0],
                        [  0,   0,   0],
                        [  0,   0,   0]],
                ]
            ]
        """

        send_url = 'http://127.0.0.1:8000/crop_face'
        cap = cv2.VideoCapture(0)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture image")
                    break

                cv2.imshow("frame", frame)
                user_input = cv2.waitKey(30) & 0xFF

                self._process_frame(send_url, frame, user_input)

                if user_input == 27:  # Escape key
                    self._save_data()
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _process_frame(self, url, frame, user_input):
        """Internal method to process each frame.

        Args:
            url (str): The URL of the server.
            frame (numpy.ndarray): The frame to process.
            user_input (int): The key pressed by the user.
        """

        _, image_enc = cv2.imencode(".jpeg", frame)
        image_file = BytesIO(image_enc.tobytes())

        try:
            response = requests.post(url, files={'image': ('image.jpeg', image_file)})
            response.raise_for_status()
            self._handle_response(response, user_input)
        except requests.RequestException as e:
            error_message = f"Request failed: {e}"
            cv2.putText(frame, error_message, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def _handle_response(self, response, user_input):
        """Internal method to handle the server's response.

           Args:
            response (requests.Response): The response from the server.
            user_input (int): The key pressed by the user.
        """

        content_type = response.headers.get("Content-Type")

        if content_type == "application/octet-stream":
            shape_str = response.headers.get("shape")
            shape = tuple(map(int, shape_str.split(',')))
            face_array = np.frombuffer(response.content, dtype=np.uint8).reshape(shape)
            cv2.imshow("cropped image", face_array)

            if user_input == ord('c'):  # 'c' key
                self.face_data_list.append(face_array)
                print("Data Captured")

        elif content_type == "application/json":
            json_data = response.json()
            if "error" in json_data:
                print(f"Error: {json_data['error']}")

        if user_input == ord('s'):  # 's' key
            self.current_name = input("Enter full name of person: ")
            self._save_data()
            self.face_data_list.clear()

    def _save_data(self):
        """Internal method to save the captured face data.
        The data will be stored in a file named <name>_data.pkl in the CapturedData folder.
        The data will be stored as a list of numpy arrays.

        """

        if self.current_name and self.face_data_list:
            file_path = os.path.join(self.folder_path, f"{self.current_name}_data.pkl")

            existing_data = self._load_existing_data(file_path)
            existing_data.extend(self.face_data_list)

            with open(file_path, 'wb') as f:
                pickle.dump(existing_data, f)
            print("File updated")

    def _load_existing_data(self, file_path):
        """Internal method to load existing data if available."""

        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        return []


if __name__ == "__main__":
    try:
        face_capture = FaceCapture()
        face_capture.capture_and_process_frames()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
