# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Create a cache directory for pip
RUN mkdir -p /root/.cache/pip

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# Install git (required for cloning repositories)
RUN apt-get update && apt-get install -y git

# Run the application
CMD ["python", "indexing_script.py"]