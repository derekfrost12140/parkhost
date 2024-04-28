# Use an official Python runtime as a parent image
FROM python:3.8-slim as builder

# Set the working directory
WORKDIR /app

# Install necessary OS libraries for OpenCV and general operation
RUN apt-get update && \
    apt-get install -y libopencv-dev libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Copy only the requirements.txt at first to leverage Docker cache
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt ultralytics

# Stage 2: Setup the runtime environment
FROM python:3.8-slim
WORKDIR /app

# Copy from builder stage
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY --from=builder /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu
COPY . /app

# Install necessary runtime libraries only
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV MODEL_PATH=best.pt  

# Run app.py when the container launches
# Final line in Dockerfile
CMD ["gunicorn", "webpython:app", "-b", "0.0.0.0:$PORT"]

