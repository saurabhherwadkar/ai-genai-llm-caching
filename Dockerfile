# -------------------------------------------------------------------
# Dockerfile
# Multi-stage build for the LLM Caching application.
# -------------------------------------------------------------------

# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies into a target directory
RUN pip install --no-cache-dir --target=/app/dependencies -r requirements.txt

# Stage 2: Production image
FROM python:3.12-slim AS production

# Set working directory
WORKDIR /app

# Copy installed dependencies from builder stage
COPY --from=builder /app/dependencies /app/dependencies

# Add dependencies to Python path
ENV PYTHONPATH="/app/dependencies:/app"

# Copy application source code
COPY src/ ./src/
COPY config/ ./config/
COPY main.py .

# Create directories for data and logs
RUN mkdir -p data logs

# Expose the API port
EXPOSE 8000

# Set default environment variables
ENV LLM_CACHE_LLM__PROVIDER=mock
ENV LLM_CACHE_LOGGING__LEVEL=INFO

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application in server mode
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
