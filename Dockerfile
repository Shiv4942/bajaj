# Multi-stage build for HackRX RAG API - Memory Optimized for 512MB
FROM python:3.11-slim as builder

# Set build environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Memory optimization environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Python memory optimizations
ENV PYTHONHASHSEED=0
ENV PYTHONOPTIMIZE=1
ENV MALLOC_TRIM_THRESHOLD_=10000
ENV MALLOC_MMAP_THRESHOLD_=1048576

# HuggingFace optimizations (for your embedding model)
ENV TOKENIZERS_PARALLELISM=false
ENV TRANSFORMERS_OFFLINE=0
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/transformers

# Application-specific optimizations
ENV MAX_WORKERS=1
ENV WEB_CONCURRENCY=1
ENV WORKER_CONNECTIONS=100
ENV TIMEOUT=120
ENV KEEP_ALIVE=5

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN useradd --create-home --shell /bin/bash --user-group app

# Create app directory and temp directories
WORKDIR /app
RUN mkdir -p /tmp/huggingface /tmp/transformers /app/temp && \
    chown -R app:app /app /tmp/huggingface /tmp/transformers

# Copy application code
COPY --chown=app:app . .

# Switch to non-root user
USER app

# Create startup script for better memory management
RUN echo '#!/bin/bash\n\
echo "🚀 Starting HackRX RAG API..."\n\
echo "Memory Info:"\n\
free -h\n\
echo "Environment:"\n\
echo "PORT: ${PORT:-10000}"\n\
echo "WORKERS: ${MAX_WORKERS:-1}"\n\
echo "TIMEOUT: ${TIMEOUT:-120}"\n\
\n\
# Start with memory optimizations\n\
exec python -O -m uvicorn main:app \\\n\
    --host 0.0.0.0 \\\n\
    --port ${PORT:-10000} \\\n\
    --workers ${MAX_WORKERS:-1} \\\n\
    --loop asyncio \\\n\
    --log-level warning \\\n\
    --no-access-log \\\n\
    --timeout-keep-alive ${KEEP_ALIVE:-5} \\\n\
    --limit-concurrency ${WORKER_CONNECTIONS:-100}\n' > start.sh && \
chmod +x start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-10000}/health || exit 1

# Expose port
EXPOSE ${PORT:-10000}

# Use the startup script
CMD ["./start.sh"]

---

# requirements.txt - Optimized for your HackRX API
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6

# Core dependencies from your code
pydantic==2.5.0
python-dotenv==1.0.0
requests==2.31.0

# LangChain - minimal installation
langchain-core==0.1.52
langchain-community==0.0.29
langchain-pinecone==0.1.0
langchain-groq==0.1.3

# Vector store and embeddings
pinecone-client==3.2.2
sentence-transformers==2.2.2  # For HuggingFaceEmbeddings

# PDF processing - lightweight
pypdf==4.0.1

# Text processing
tiktoken==0.6.0

# System monitoring for memory management
psutil==5.9.8

# HTTP client optimization
httpx==0.26.0

# For production
gunicorn==21.2.0

---

# docker-compose.yml - For local testing with memory limits
version: '3.8'

services:
  hackrx-api:
    build: .
    ports:
      - "10000:10000"
    
    # Memory constraints for testing 512MB limit
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '1.0'
        reservations:
          memory: 256M
          cpus: '0.5'
    
    environment:
      - PORT=10000
      - PYTHONUNBUFFERED=1
      - PYTHONOPTIMIZE=1
      - MAX_WORKERS=1
      - WEB_CONCURRENCY=1
      - TIMEOUT=120
      - KEEP_ALIVE=5
      
      # Your API keys (set these in .env file)
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - GROQ_API_KEY=${GROQ_API_KEY}
      
      # Memory optimizations
      - MALLOC_TRIM_THRESHOLD_=10000
      - TOKENIZERS_PARALLELISM=false
      - TRANSFORMERS_OFFLINE=0
    
    volumes:
      - ./temp:/app/temp:rw
      - /tmp/huggingface:/tmp/huggingface:rw
    
    # Health check
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:10000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    
    # Restart policy
    restart: unless-stopped

---

# .dockerignore - Reduce build context size
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
.env
pip-log.txt
.tox/
.coverage
.cache
nosetests.xml
*.cover
*.log
.git/
.gitignore
.mypy_cache/
.pytest_cache/
.hypothesis/
.DS_Store
*.sqlite3
*.db
node_modules/
dist/
build/
*.tar.gz
*.zip
temp/
uploads/
logs/
README.md
docs/
*.md
.vscode/
.idea/
*.swp
*.swo
*~
Thumbs.db
ehthumbs.db
Desktop.ini

# Model caches (will be downloaded at runtime)
models/
.cache/
huggingface_hub/
transformers_cache/

---

# Build and run commands (build.sh)
#!/bin/bash

echo "🏗️ Building HackRX RAG API Docker image..."

# Build the image
docker build -t hackrx-rag-api:latest .

echo "✅ Build complete!"
echo ""
echo "🚀 To run locally with memory constraints:"
echo "docker-compose up"
echo ""
echo "🚀 Or run directly:"
echo "docker run -p 10000:10000 --memory=512m \\"
echo "  -e PINECONE_API_KEY=your_key \\"
echo "  -e GROQ_API_KEY=your_key \\"
echo "  hackrx-rag-api:latest"
echo ""
echo "🔍 Monitor memory usage:"
echo "docker stats"

---

# Memory monitoring script (monitor-docker.sh)
#!/bin/bash

echo "🔍 Monitoring HackRX API Docker container..."

# Function to get container stats
get_stats() {
    local container_name=${1:-"hackrx-rag-api"}
    
    echo "=== Container Stats ==="
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}" $container_name
    
    echo ""
    echo "=== Application Health ==="
    curl -s http://localhost:10000/health | jq '.' 2>/dev/null || echo "Health check failed"
    
    echo ""
    echo "=== Memory Details ==="
    curl -s http://localhost:10000/status | jq '.index_stats' 2>/dev/null || echo "Status check failed"
    
    echo ""
    echo "================================="
    echo ""
}

# Monitor every 10 seconds
while true; do
    get_stats
    sleep 10
done
