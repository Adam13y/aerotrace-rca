#!/bin/sh
# Build-time helper: start a temporary Ollama server, pull the given models into the image, stop it.
set -eu

if [ "$#" -eq 0 ]; then
    echo "usage: pull-models.sh MODEL [MODEL ...]" >&2
    exit 2
fi

ollama serve >/tmp/ollama-build.log 2>&1 &
server_pid=$!

attempts=0
until ollama list >/dev/null 2>&1; do
    attempts=$((attempts + 1))
    if [ "$attempts" -gt 60 ]; then
        echo "Ollama server did not start within 60 s:" >&2
        cat /tmp/ollama-build.log >&2
        kill "$server_pid" 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

for model in "$@"; do
    echo "Pulling $model ..."
    ollama pull "$model"
done
ollama list

kill "$server_pid"
wait "$server_pid" 2>/dev/null || true
rm -f /tmp/ollama-build.log
