# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire workspace into the container
COPY . /app

# Ensure the start script is executable
RUN chmod +x ./scripts/start.sh

# Expose port 8000 for the API and 11434 for the LLM server
EXPOSE 8000
EXPOSE 11434

# Run the startup script
CMD ["./scripts/start.sh"]
