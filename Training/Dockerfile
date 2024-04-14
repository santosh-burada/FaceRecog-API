# Use an official lightweight Python image.
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app
# Install system dependencies required for dlib, OpenCV, and other Python packages.
# Note: Debian uses apt-get, not apk (Alpine package manager).
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    zlib1g-dev \
    ca-certificates \
    g++ \
    make \
    liblapack-dev \
    gfortran \
    libstdc++6 \
    libblas-dev \
    liblapack-dev \
    libkrb5-dev \
    # Cleanup the apt cache to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the Python scripts and requirements file into the container at /app
COPY main.py requirements.txt ./

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8002 available to the world outside this container
EXPOSE 8002

# Define environment variable to improve Python logging
ENV PYTHONUNBUFFERED=1

# Command to run the Flask application
CMD ["python", "main.py"]
