import pickle
from pymongo import MongoClient, errors
from bson.objectid import ObjectId

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

def create_collection_if_not_exists(db, collection_name):
    """Create a collection if it does not exist in the specified database."""
    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name)
        print(f"Collection '{collection_name}' created successfully.")
    else:
        print(f"Collection '{collection_name}' already exists.")

def insert_pickle_data_as_document(db, collection_name, pkl_file_path, key_name):
    """Load data from a .pkl file, wrap it under a specified key in a dictionary, and insert into MongoDB."""
    # Load data from the pickle file
    with open(pkl_file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    
    # Wrap the loaded data in a dictionary under the specified key
    document = {"_id": key_name, "values": loaded_data}
    
    # Insert the document into the specified MongoDB collection
    collection = db[collection_name]
    result = collection.insert_one(document)
    print(f"Data inserted with ID: {result.inserted_id}")
def retrieve_single_document(db, collection_name, query):
    """Retrieve a single document from the specified MongoDB collection."""
    collection = db[collection_name]
    document = collection.find_one(query)
    if document:
        print("Retrieved document:", document)
    else:
        print("No document matches the query.")

if __name__ == "__main__":
    connection_string = "mongodb+srv://santuburada99:L7T3TUVD1KOkLtLJ@train-facerec.8dl5kmd.mongodb.net/?retryWrites=true&w=majority&appName=Train-faceRec"
    database_name = "Train-faceRec"
    collection_name = "Training_Data"
    pkl_file_path = 'Output/numpylist.pkl'
    key_name = "numpylist" 

    db = connect_to_mongodb_atlas(connection_string, database_name)
    create_collection_if_not_exists(db, collection_name)
    insert_pickle_data_as_document(db, collection_name, pkl_file_path, key_name)

    collections = db.list_collection_names()
    print("Collections in the database:", collections)

    # retrive
    query = {"_id": key_name}
    retrieve_single_document(db, collection_name, query)
