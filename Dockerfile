# Dockerfile for Wine Quality API
# Use a Python base image. For GPU support use the official PyTorch CUDA images.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# system deps for pip installs (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy requirements and install
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

# create mounted dirs to ensure they exist
RUN mkdir -p /app/Models/trained /app/data /app/logs || true

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
