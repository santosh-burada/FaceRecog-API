

# Developing a Scalable and Distributed Face Recognition System using Kubernetes and Docker

## Project Description
This project aims to build a robust, scalable, and distributed face recognition system leveraging the power of Kubernetes and Docker. Prepared by Santosh Burada, this initiative addresses the growing needs of efficient face recognition in various applications like security, identity verification, and more.

## Features
- **Scalability**: The system can handle an increasing number of face recognition requests without degrading performance.
- **Distributed Architecture**: Utilizes multiple nodes to distribute the computational load.
- **High Availability**: Designed to minimize downtime and ensure continuous operation.
- **Modularity**: Each component is containerized for easy deployment and scaling.
- **User-friendly Interface**: Provides a simple and intuitive user interface for interaction.

## Technologies Used
- **Kubernetes**: For orchestrating and managing containerized applications.
- **Docker**: For containerizing each component of the system.
- **Face Recognition Algorithms**: Implementing state-of-the-art algorithms for accurate face recognition.
- **Cloud Providers**: Support for deployment on various cloud platforms.

## Python version 3.11.0

## Installation Guide

### Step 1: Setting up the MTCNN-detector

1. **Navigate to the MTCNN-detector Folder**: Open the terminal in the MTCNN-detector directory.

2. **Create and Activate a Python Virtual Environment**:
   - Create the environment:
     ```bash
     python3 -m venv myenv-name
     ```
   - Activate the environment:
     - For Windows:
       ```bash
       myenv-name\Scripts\activate
       ```
     - For macOS/Linux:
       ```bash
       source myenv-name/bin/activate
       ```

3. **Install TensorFlow**:
   - TensorFlow will automatically manage package dependencies based on your operating system. Install it using:
     ```bash
     pip install tensorflow
     ```

4. **Install Remaining Dependencies**:
   - Install the packages specified in `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```