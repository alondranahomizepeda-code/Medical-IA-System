# ==============================================================================
# Multi-Stage Production Dockerfile — Medical AI System
# ==============================================================================
# Stage 1: Dependency Builder
# Stage 2: Minimal Production Runtime with Non-Root Security Context
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build-time OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt .

# Install dependencies into isolated wheels directory
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# ------------------------------------------------------------------------------
# Stage 2: Final Runtime Image
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS runner

# Metadata labels
LABEL maintainer="Equipo 3 - Software Engineering II" \
      version="1.0.0" \
      description="Medical Image Classification System with Transfer Learning & Vertebral Tracking"

WORKDIR /app

# Install minimal runtime OS dependencies (OpenCV headless & image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy built wheels from builder stage
COPY --from=builder /build/wheels /wheels
COPY --from=builder /build/requirements.txt .

# Install wheels without compiler overhead
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Create dedicated non-root application user for clinical data security compliance
RUN groupadd -r medicalgroup && useradd -r -g medicalgroup -d /app medicaluser

# Copy application source code
COPY . /app/

# Create persistent storage directories with proper permissions
RUN mkdir -p /app/data/uploads /app/data/processed /app/data/models /app/data/reports /app/logs && \
    chown -R medicaluser:medicalgroup /app

# Switch to non-root user
USER medicaluser

# Expose internal API service port
EXPOSE 8000

# Container healthcheck for orchestrator readiness / liveness
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production server via Uvicorn with standard ASGI workers
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
