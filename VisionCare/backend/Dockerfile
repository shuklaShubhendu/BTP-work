FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY . .

# Create directories for models and images (populated via volumes)
RUN mkdir -p /app/models /app/demo_images

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
