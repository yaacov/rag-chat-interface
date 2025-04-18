# Use Red Hat UBI9 with Python 3.9
FROM registry.access.redhat.com/ubi9/python-39:latest

# Prevent creation of .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps (e.g., poppler-utils for PDF)
USER root
RUN dnf update -y \
    && dnf install -y poppler-utils \
    && dnf clean all

# Create the directory and give group‑write perms to GID 0
RUN mkdir -p /opt/app/data \
    && chgrp -R 0 /opt/app/data \
    && chmod -R g=u /opt/app/data

USER 1001

# Allow configuration via ENV and redirect all writable paths to /opt/app/data
ENV APP_SOURCE="" \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    DB_PATH=/opt/app/data/rag_milvus.db \
    MODELS_CACHE_DIR=/opt/app/data/models_cache \
    DOWNLOADS_DIR=/opt/app/data/downloads \
    CHUNK_SIZE=2000 \
    CHUNK_OVERLAP=200 \
    DEVICE="" \
    LLM_MODEL="" \
    LLM_API_URL="" \
    LLM_API_KEY="" \
    EMBEDDING_MODEL="" \
    EMBEDDING_API_URL="" \
    EMBEDDING_API_KEY=""

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application sources
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Default command reads flags from ENV, /opt/app is world‑writable so lock/db files can be created
CMD /opt/app-root/bin/python main.py \
  ${APP_SOURCE:+--source "$APP_SOURCE"} \
  --host "$APP_HOST" \
  --port $APP_PORT \
  --db-path "$DB_PATH" \
  --models-cache-dir "$MODELS_CACHE_DIR" \
  --downloads-dir "$DOWNLOADS_DIR" \
  --chunk_size $CHUNK_SIZE \
  --chunk_overlap $CHUNK_OVERLAP \
  ${DEVICE:+--device "$DEVICE"} \
  ${LLM_MODEL:+--llm-model "$LLM_MODEL"} \
  ${LLM_API_URL:+--llm-api-url "$LLM_API_URL"} \
  ${LLM_API_KEY:+--llm-api-key "$LLM_API_KEY"} \
  ${EMBEDDING_MODEL:+--embedding-model "$EMBEDDING_MODEL"} \
  ${EMBEDDING_API_URL:+--embedding-api-url "$EMBEDDING_API_URL"} \
  ${EMBEDDING_API_KEY:+--embedding-api-key "$EMBEDDING_API_KEY"}
