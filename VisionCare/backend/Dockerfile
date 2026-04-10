FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first (needs special index URL), then the rest
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.1.0+cpu torchvision==0.16.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir fastapi==0.115.0 uvicorn[standard]==0.30.6 \
    pydantic==2.9.2 python-multipart==0.0.9 Pillow==10.4.0 numpy==1.26.4

# RAG dependencies: FAISS + LangChain + embeddings + Gemini HTTP client
RUN pip install --no-cache-dir \
    faiss-cpu==1.8.0 \
    langchain==0.2.16 \
    langchain-community==0.2.17 \
    sentence-transformers==2.7.0 \
    transformers==4.40.2 \
    pypdf==4.3.1 \
    httpx==0.27.2

# Copy backend source
COPY . .

# Create directories for models, images, and RAG index
RUN mkdir -p /app/models /app/demo_images /app/rag_index

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
