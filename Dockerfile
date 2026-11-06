# Build native dependencies separately from the serving image.
FROM python:3.11-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=2 \
    CMAKE_ARGS="-DGGML_NATIVE=OFF"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build
COPY requirements.txt .
# Prefer a CPU wheel for the pinned release; compile it if no matching wheel exists.
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir --prefer-binary \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    -r requirements.txt

FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# OpenMP is required by the CPU llama.cpp build; compilers stay in the builder.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=appuser:appuser app/ ./app/
USER appuser

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
