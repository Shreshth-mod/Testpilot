# Cryptographically pinned base image digest: python:3.11-slim
FROM python:3.11-slim@sha256:9635410349cba3531fabde484e07f14e625abb40e1bc8b23fffa046cb497ce2d

# Prevent interactive prompts and disable bytecode writes
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Copy statically linked Docker CLI from official docker:cli image
COPY --from=docker:cli /usr/local/bin/docker /usr/local/bin/docker

# Install essential system utilities: git for revision analysis, curl for health checks, ca-certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pinned Python dependencies
COPY docker/requirements-app.txt /tmp/requirements-app.txt
RUN pip install --no-cache-dir -r /tmp/requirements-app.txt && rm /tmp/requirements-app.txt

# Copy project source and install package in editable/development mode
COPY pyproject.toml README.md ./
COPY testpilot/ ./testpilot/
COPY testbed/ ./testbed/

RUN pip install --no-cache-dir --no-deps -e .

# Expose TestPilot web service port (8088 to avoid port conflicts with host services)
EXPOSE 8088

# Healthcheck validating the real /health endpoint
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8088/health || exit 1

# Launch production-grade ASGI web server
CMD ["uvicorn", "testpilot.web.api:app", "--host", "0.0.0.0", "--port", "8088"]
