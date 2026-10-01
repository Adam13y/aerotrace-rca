"""Static checks for the Docker packaging (Docker itself is not needed to run these)."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LINUX_FILES = ("docker/entrypoint.sh", "docker/pull-models.sh", "Dockerfile", "docker/Dockerfile.ollama",
               "docker-compose.yml", "docker-compose.cpu.yml")


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that understands Compose's !reset tag."""


_ComposeLoader.add_constructor("!reset", lambda loader, node: "!reset")


def _compose(name: str) -> dict:
    return yaml.load((ROOT / name).read_text(encoding="utf-8"), Loader=_ComposeLoader)


def test_container_files_use_lf_line_endings() -> None:
    for name in LINUX_FILES:
        assert b"\r" not in (ROOT / name).read_bytes(), f"{name} has CRLF line endings"


def test_gitattributes_forces_lf_for_container_files() -> None:
    rules = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    for pattern in ("*.sh", "Dockerfile", "Dockerfile.*", "docker-compose*.yml"):
        assert any(line.startswith(pattern) and "eol=lf" in line for line in rules.splitlines()), pattern


def test_dockerignore_keeps_secrets_and_local_state_out() -> None:
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    for entry in (".env", ".venv", ".git", "data/vector_store/", "data/reports/*"):
        assert entry in ignored, entry


def test_compose_wires_app_to_baked_ollama() -> None:
    services = _compose("docker-compose.yml")["services"]
    ollama, app = services["ollama"], services["aerotrace"]
    assert ollama["build"]["dockerfile"] == "docker/Dockerfile.ollama"
    assert "qwen2.5:14b-instruct-q4_K_M" in ollama["build"]["args"]["MODELS"]
    assert "nomic-embed-text" in ollama["build"]["args"]["MODELS"]
    assert ollama["healthcheck"]["test"] == ["CMD", "ollama", "list"]
    assert app["depends_on"]["ollama"]["condition"] == "service_healthy"
    env = app["environment"]
    assert env["OLLAMA_BASE_URL"] == "http://ollama:11434"
    assert "true" in env["OLLAMA_REPRODUCIBLE"] and "10240" in env["OLLAMA_NUM_CTX"]
    mounts = {v.split(":")[1] for v in app["volumes"]}
    assert {"/app/data/raw", "/app/data/reports", "/app/data/vector_store", "/app/data/processed"} <= mounts


def test_cpu_override_removes_gpu_reservation() -> None:
    assert _compose("docker-compose.cpu.yml")["services"]["ollama"]["deploy"] == "!reset"


def test_dockerfiles_reference_existing_files() -> None:
    app = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert app.startswith("# ") and "FROM python:3.12-slim" in app
    assert 'ENTRYPOINT ["/app/docker/entrypoint.sh"]' in app and "USER aerotrace" in app
    ollama = (ROOT / "docker" / "Dockerfile.ollama").read_text(encoding="utf-8")
    assert "COPY docker/pull-models.sh" in ollama and "pull-models.sh ${MODELS}" in ollama
    for script in ("docker/entrypoint.sh", "docker/pull-models.sh"):
        assert (ROOT / script).read_text(encoding="utf-8").startswith("#!/bin/sh\n"), script
