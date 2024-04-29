# Use an official Python runtime as a base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install necessary OS libraries for OpenCV
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0

# Install any needed packages specified in requirements.txt
# First copy only the requirements.txt to leverage Docker cache
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt 

# Copy the rest of your application's code
COPY . /app

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV MODEL_PATH=best.pt

# Run gunicorn with the app using fewer workers to save memory
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "webpython:app"]


