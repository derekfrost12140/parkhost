# Use an official Python runtime as a base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
# First copy only the requirements.txt to leverage Docker cache
COPY requirements.txt /app/
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the rest of your application's code
COPY . /app

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV MODEL_PATH=best.pt

# Run gunicorn with the app
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "webpython:_flask"]
