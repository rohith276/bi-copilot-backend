# Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for pandas and other libs
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create upload directory
RUN mkdir -p uploads

# Expose port
EXPOSE 8000

# Start command — $PORT is injected by Render at runtime
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
