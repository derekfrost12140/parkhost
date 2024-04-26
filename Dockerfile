# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container to /app
WORKDIR /app

# Install necessary OS libraries for OpenCV and general operation
RUN apt-get update && apt-get install -y libopencv-dev libgl1-mesa-glx

# Copy the current directory contents into the container at /app
# This includes requirements.txt, app.py, and the YOLO model file (best.pt)
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV MODEL_PATH=best.pt  # Assuming best.pt is in the /app directory

# Run app.py when the container launches
CMD ["python", "app.py"]

