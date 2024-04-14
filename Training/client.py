import os
import pickle
import requests
import json
import numpy as np
from pymongo import MongoClient, errors
from dotenv import load_dotenv

load_dotenv()

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
connection_string = os.getenv('MONGO_URI')
database_name = os.getenv('DATABASE_NAME')
 

# Function to load processed files' records
def load_processed_files(record_file):
    if os.path.exists(record_file) and os.path.getsize(record_file) > 0:
        with open(record_file, 'r') as file:
            return json.load(file)
    else:
        return {}

# Function to save processed files' records
def save_processed_files(processed_files, record_file):
    with open(record_file, 'w') as file:
        json.dump(processed_files, file, indent=4)

# Function to get file modification time
def get_file_modification_time(file_path):
    return os.path.getmtime(file_path)

def update_or_insert_data(db, email, new_data):
    train_collection = db["Training_Data"]
    existing_doc = train_collection.find_one({"_id": email})

    if existing_doc:
        # Document exists, so update it
        features = existing_doc.get('features', {})
        
        # Update existing features with new_data or add new_data if not present
        for key, value in new_data.items():
            if key in features:
                # Update or extend the existing key's value
                features[key].extend(value)
            else:
                # Add the new key-value pair
                features[key] = value
        
        train_collection.update_one({"_id": email}, {"$set": {"features": features}})
    else:
        # No existing document, so insert a new one
        data = {"_id": email, 'features': new_data}
        train_collection.insert_one(data)

def main():
    # Initialize variables
    images_dict = {}
    processed_files = load_processed_files('processed_files.json')
    new_or_updated_files = False

    # Input email address
    email = input("Enter email address: ")

    # Walk through the directory structure
    for root, dirs, files in os.walk("../CapturedData", topdown=False):
        # Filter for pickle files
        pkl_files = [file for file in files if file.endswith('.pkl')]
        for name in pkl_files:
            file_path = os.path.join(root, name)
            file_mod_time = get_file_modification_time(file_path)

            # Check if the file is new or has been updated
            if name not in processed_files or processed_files[name] < file_mod_time:
                # Open the pickle file and load its contents
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)

                # Extract the base name for dictionary key
                base_name = name.replace("_data.pkl", "")

                # Convert images to lists and store in the dictionary
                images_dict[base_name] = [image.tolist() for image in data]

                # Mark that we've processed new or updated files
                new_or_updated_files = True

                # Update the processed files record
                processed_files[name] = file_mod_time

                # Optional: Print information about the loaded data
                print(f"{base_name}: {len(data)} images loaded")

    # Only proceed if there are new or updated files
    if new_or_updated_files:
        # Prepare and send the request
        url = "http://127.0.0.1:8003/process_images"
        data = {"images": images_dict}
        headers = {
            "Content-Type": "application/json",
            'Email': email
        }

        response = requests.post(url, headers=headers, json=data, verify=False)

        # Check the response from the server
        if response.status_code == 200:
            db = connect_to_mongodb_atlas(connection_string, database_name)
            processed_data = {name: images_list for name, images_list in response.json().items()}
            
            # Update or insert data into MongoDB
            update_or_insert_data(db, email, processed_data)
            
            print("Success:", response.json())
            numpylist = []
            namelist = []
            for name, images_list in response.json().items():
                for i in images_list:
                    numpylist.append(np.array(i))
                    namelist.append(name)
            with open('Output/numpylist.pkl', 'wb') as f:
                pickle.dump(numpylist, f)
            with open('Output/namelist.pkl', 'wb') as f:
                pickle.dump(namelist, f)

            # Save the updated record of processed files
            save_processed_files(processed_files, 'processed_files.json')
        else:
            print("Error:", response.text)
    else:
        print("No new or updated files to process.")

if __name__ == "__main__":
    main()
