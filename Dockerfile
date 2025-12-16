# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    torch \
    transformers \
    datasets \
    sentencepiece \
    langdetect \
    protobuf \
    accelerate \
    google-generativeai \
    python-dotenv

# Copy the entire workspace into the container
COPY . /app

# Expose port 8000 for the API
EXPOSE 8000

# Run the server
CMD ["uvicorn", "scripts.server:app", "--host", "0.0.0.0", "--port", "8000"]
