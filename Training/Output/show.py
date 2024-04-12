import cv2
import pickle
with open('namelist.pkl', 'rb') as file:
    images = pickle.load(file)
print(images)

# for img in images:
#     # If the image is not in the correct color format (e.g., BGR for OpenCV), convert it
#     # For example, if the image is in RGB format, convert it to BGR
    
#     # Display the image
#     # print(img)
#     # print("_____________________________________")
#     cv2.imshow('Image', img)
#     cv2.waitKey(0)  # Wait for a key press to move to the next image

# cv2.destroyAllWindows()