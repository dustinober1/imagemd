#!/bin/bash

# Docker entrypoint script for VisionPDF API server

set -e

# Default values
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
WORKERS=${WORKERS:-1}
LOG_LEVEL=${LOG_LEVEL:-info}
RELOAD=${RELOAD:-false}

# Convert boolean string to actual boolean for uvicorn
if [ "$RELOAD" = "true" ]; then
    RELOAD_FLAG="--reload"
else
    RELOAD_FLAG=""
fi

# Check if we should run in development mode
if [ "$VISIONPDF_ENV" = "development" ]; then
    RELOAD_FLAG="--reload"
    WORKERS=1
fi

# Print startup information
echo "🚀 Starting VisionPDF API Server"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Workers: $WORKERS"
echo "   Log Level: $LOG_LEVEL"
echo "   Environment: ${VISIONPDF_ENV:-production}"
echo "   Reload: ${RELOAD_FLAG:-false}"
echo ""

# Run health check on dependencies
echo "🔍 Checking system health..."

# Check if Ollama is available (if configured)
if [ -n "$OLLAMA_BASE_URL" ]; then
    echo "   Checking Ollama at $OLLAMA_BASE_URL..."
    if curl -s "$OLLAMA_BASE_URL/api/version" > /dev/null 2>&1; then
        echo "   ✅ Ollama is available"
    else
        echo "   ⚠️  Ollama is not available at $OLLAMA_BASE_URL"
    fi
fi

# Check Python dependencies
echo "   Checking Python dependencies..."
python -c "import vision_pdf; print('   ✅ VisionPDF imported successfully')" || {
    echo "   ❌ Failed to import VisionPDF"
    exit 1
}

# Check API dependencies
python -c "import fastapi, uvicorn; print('   ✅ API dependencies available')" || {
    echo "   ❌ API dependencies not available"
    exit 1
}

echo ""
echo "📖 API Documentation will be available at: http://$HOST:$PORT/docs"
echo "📊 ReDoc Documentation will be available at: http://$HOST:$PORT/redoc"
echo "🔍 Health Check will be available at: http://$HOST:$PORT/health"
echo ""

# Start the API server
exec vision-pdf-api \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL" \
    $RELOAD_FLAG