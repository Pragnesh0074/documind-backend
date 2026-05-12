FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for ChromaDB and SQLite
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8080

# Set environment variables for production
ENV PYTHONUNBUFFERED=1

# Command to run - Cloud Run provides $PORT dynamically
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
