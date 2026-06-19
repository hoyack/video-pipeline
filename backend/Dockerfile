# ---- Builder stage ----
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install dependencies first (cache-friendly layer ordering)
COPY backend/pyproject.toml backend/
RUN pip install --no-cache-dir -e "backend/[dev]"

# ---- Production stage ----
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application source
COPY config.yaml .
COPY backend/ backend/

# Install package in editable mode (registers entry points)
RUN pip install --no-cache-dir -e "backend/[dev]"

# Default Docker paths — overridable via env or .env
ENV VIDPIPE_STORAGE__DATABASE_URL="sqlite+aiosqlite:///data/vidpipe.db"
ENV VIDPIPE_STORAGE__TMP_DIR="/app/tmp"

EXPOSE 8000

CMD ["uvicorn", "vidpipe.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
