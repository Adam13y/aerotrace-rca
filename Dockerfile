# AeroTrace-RCA application image (Python + all requirements).
# The LLMs run in the separate Ollama image (docker/Dockerfile.ollama); see docker-compose.yml.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OLLAMA_BASE_URL=http://ollama:11434

WORKDIR /app

# Dependencies first so code changes do not invalidate the (large) dependency layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Run as an unprivileged user. Data directories are created (and owned) in the image so that
# named volumes mounted over them inherit the right ownership on first use.
RUN useradd --create-home --uid 1000 aerotrace \
    && mkdir -p data/raw data/processed data/reports data/vector_store \
    && chmod +x docker/entrypoint.sh \
    && chown -R aerotrace:aerotrace /app
USER aerotrace

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["python", "scripts/run_pipeline.py"]
