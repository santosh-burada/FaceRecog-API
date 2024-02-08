import os
import pickle
import requests
import json

# Initialize a dictionary to hold the name of the data and the images as lists
images_dict = {}

# Walk through the directory structure
for root, dirs, files in os.walk("../CapturedData", topdown=False):
    # Filter for pickle files
    pkl_files = [file for file in files if file.endswith('.pkl')]
    for name in pkl_files:
        # Construct the full file path
        file_path = os.path.join(root, name)
        
        # Open the pickle file and load its contents
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        
        # Extract the base name without "_data.pkl" for dictionary key
        base_name = name.replace("_data.pkl", "")
        
        # Convert images to lists and store them in the dictionary
        images_dict[base_name] = [image.tolist() for image in data]
        
        # Optional: Print information about the loaded data
        print(f"{base_name}: {len(data)} images loaded")




url = "http://<local_Ip>:30002/process_images"

data = {"images": images_dict}
print(type(data))
headers = {
    "Content-Type": "application/json"
}


response = requests.post(url, headers=headers,json=data, verify=False)

# Check the response from the server
if response.status_code == 200:
    print("Success:", response.json())
    
else:
    print("Error:", response.text)

