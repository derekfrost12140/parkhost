# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container to /app
WORKDIR /app

# Install necessary OS libraries for OpenCV and general operation
RUN apt-get update && apt-get install -y libopencv-dev libgl1-mesa-glx

# Copy only the necessary files
COPY requirements.txt .
COPY app.py .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable for the model URL
ENV MODEL_URL=your_model_url_here

# Run app.py when the container launches
CMD ["python", "app.py"]
