# Uses an official Python runtime as a parent image
FROM python:3.9-slim

# Sets environment variables
ENV PYTHONUNBUFFERED=1

# Installs necessary system dependencies
RUN apt-get update && apt-get install -y git

# Sets the working directory
WORKDIR /app

# Copies project files
COPY . .

# Installs Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Exposes port for potential API use
EXPOSE 5000

# Command to run the main script
CMD ["python", "main.py"]
