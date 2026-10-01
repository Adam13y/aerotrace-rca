#!/bin/sh
# AeroTrace-RCA container entrypoint.
#   1. Wait for the Ollama service to answer.
#   2. Build the RAG vector store if it is missing or out of date with data/historical_failures/
#      (it needs Ollama for embeddings, so it cannot be built at image build time).
#   3. Run the given command (default: scripts/run_pipeline.py).
set -eu

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://ollama:11434}"
WAIT_SECONDS="${OLLAMA_WAIT_SECONDS:-180}"
export OLLAMA_BASE_URL

echo "[entrypoint] waiting for Ollama at $OLLAMA_BASE_URL (up to ${WAIT_SECONDS}s)"
waited=0
until "$PYTHON" -c "import sys, urllib.request; urllib.request.urlopen(sys.argv[1] + '/api/tags', timeout=3)" "$OLLAMA_BASE_URL" 2>/dev/null; do
    waited=$((waited + 2))
    if [ "$waited" -ge "$WAIT_SECONDS" ]; then
        echo "[entrypoint] Ollama not reachable after ${WAIT_SECONDS}s - continuing; agents fall back to rule-based mode" >&2
        break
    fi
    sleep 2
done

if [ "${SKIP_KB_BUILD:-false}" != "true" ]; then
    if "$PYTHON" - <<'PY'
import sys
from config.settings import get_settings
from src.memory.historical_rag import HistoricalRAG, OllamaEmbedder, load_documents

settings = get_settings()
rag = HistoricalRAG(settings.historical_store_dir, OllamaEmbedder(settings.ollama_base_url, settings.embedding_model))
indexed = rag.stats()["cases"]
on_disk = len(load_documents(settings.historical_docs_dir))
print(f"[entrypoint] knowledge base: {on_disk} cases on disk, {indexed} indexed")
sys.exit(0 if indexed == on_disk and indexed > 0 else 1)
PY
    then
        echo "[entrypoint] vector store up to date"
    else
        echo "[entrypoint] building vector store (first run or documents changed)"
        "$PYTHON" scripts/build_knowledge_base.py --query "" \
            || echo "[entrypoint] vector store build failed - pipeline will run without historical context" >&2
    fi
fi

exec "$@"
