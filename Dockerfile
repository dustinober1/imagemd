# Multi-stage Docker build for VisionPDF
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install OCR dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml ./
COPY README.md ./
COPY LICENSE ./

# Install VisionPDF with API dependencies
RUN pip install --upgrade pip && \
    pip install -e .[api,ocr]

# Copy application code
COPY vision_pdf/ ./vision_pdf/

# Create non-root user
RUN useradd --create-home --shell /bin/bash visionpdf && \
    chown -R visionpdf:visionpdf /app
USER visionpdf

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["vision-pdf-api", "--host", "0.0.0.0", "--port", "8000"]

# Production stage
FROM base as production

# Add production-specific configurations
ENV VISIONPDF_ENV=production

# Use production-ready startup script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]