# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container to /app
WORKDIR /app

# Install necessary OS libraries for OpenCV, Git, and general operation
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libgl1-mesa-glx \
    git  # Ensure git is installed

# Clone YOLOv5 and install dependencies
RUN git clone --depth 1 --branch v6.0 https://github.com/ultralytics/yolov5.git yolov5
WORKDIR /app/yolov5
RUN pip install -r requirements.txt

# Install YOLOv5 dependencies from the cloned directory
RUN pip install --no-cache-dir -r /app/yolov5/requirements.txt

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt /app/
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV MODEL_PATH=best.pt  

# Set PYTHONPATH to include the YOLOv5 folder
ENV PYTHONPATH="${PYTHONPATH}:/app/yolov5"

# Run gunicorn as the entry point, change to match your Flask app initialization
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
