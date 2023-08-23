import os
from io import BytesIO

import cv2
import numpy
import requests
from PIL import Image


def create_folder(parent_folder, folder_name, exist_ok):
    """
        Create a new folder with the given name inside the specified parent folder.

        Args:
            parent_folder (str): The path to the parent folder where the new folder will be created.
            folder_name (str): The name of the new folder to be created.
            exist_ok (bool): If True, the function will not raise an error if the folder already exists.
                             If False, the user will be prompted to decide whether to continue with the
                             existing folder or provide a new folder name.

        Returns:
            str: The full path of the created folder.

        Raises:
            OSError: If there is an error creating the folder (e.g., invalid path, insufficient permissions).
        """
    try:
        full_path = os.path.join(parent_folder, folder_name)
        os.makedirs(full_path, exist_ok=exist_ok)
        print(f"Folder '{folder_name}' created successfully in '{parent_folder}'.")
    except FileExistsError:
        print(f"Folder '{full_path}' already exists.")
        n = input("you want to continue saving the image in the existed folder[Yes/No]: ")
        if n.lower() == "yes":
            create_folder(parent_folder, folder_name, True)
        if n.lower() == "no":
            folder_name = input("Enter new Folder name: ")
            create_folder(full_path, folder_name, False)
    except OSError as e:
        print(f"Error creating folder '{full_path}': {e}")
    return full_path


def capture_and_process_frames():
    """
    Capture video frames from the default camera and send them to a face detection API.
    Display the original and processed frames, and save the processed frames upon user input.

    The function continuously captures frames from the default camera using OpenCV and sends
    each frame to a face detection API. If the API successfully processes the frame, the
    processed frame is displayed alongside the original frame.

    When the user presses the 'c' key, the processed frame is saved to a folder with a name
    provided by the user. If the folder does not exist, it is created.

    The loop continues until the user presses the 'ESC' key (ASCII code 27), at which point
    the function releases the camera and closes all OpenCV windows.
    """
    send_url = 'http://127.0.0.1:8000/crop_face'
    counter = 0
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        k = cv2.waitKey(30) & 0xFF
        _, image_enc = cv2.imencode(".jpeg", frame)
        image_file = BytesIO(image_enc)
        try:
            response = requests.post(send_url, files={'image': ('image.jpeg', image_file)})
            if response.ok:
                counter += 1
                if response.headers["Content-Type"] == "image/png":
                    processedImage = Image.open(BytesIO(response.content))
                    cv2.imshow("detected", numpy.array(processedImage))
                # else:
                #     print(response.content)
                if k == ord('c'):
                    name = input("Enter full name of person: ")
                    filepath = create_folder("./CapturedData", name, False)
                    image_name = str(name + "_" + str(counter) + ".jpeg")
                    cv2.imwrite(os.path.join(filepath, image_name), numpy.array(processedImage))
        except Exception as e:
            print(e)
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    capture_and_process_frames()


if __name__ == "__main__":
    main()
