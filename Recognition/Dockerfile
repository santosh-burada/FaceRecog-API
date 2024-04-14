# Use an official Python runtime as a parent image
FROM python:3.11.0-slim-bullseye

# Set environment variables for Python to run Flask
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV CUDA_VISIBLE_DEVICES=-1

# Set the working directory in the container
WORKDIR /app

# RUN apt-get update && apt-get install -y \
#     libsm6 libxext6 libxrender-dev \
#     libgl1-mesa-dev \
#     ca-certificates \
#     libglib2.0-0 \
#     libgl1-mesa-glx \
#     pkg-config \
#     libhdf5-dev \
#     cmake \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev \
    libgl1-mesa-dev \
    ca-certificates \
    libglib2.0-0 \
    libgl1-mesa-glx \
    pkg-config \
    libhdf5-dev \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*


# Install TensorFlow
RUN pip install tensorflow==2.15.0

# Copy the current directory contents into the container at /app
COPY docker-req.txt .
COPY face_detector ./face_detector
COPY model ./model

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r docker-req.txt

# Copy the API code into the container
COPY main.py .

# Make port 8001 available to the world outside this container
EXPOSE 8005

# Run main.py when the container launches
CMD ["flask", "run", "--port=8005"]