"""Set up the AeroTrace-RCA development environment.

Steps:
1. Verify the Python version (3.11+).
2. Create ``.venv`` (unless it exists).
3. Install ``requirements.txt`` into it.
4. Create ``.env`` from ``.env.example`` if missing.
5. Create the data directories.
6. Report whether Ollama is reachable and the configured model is installed.

Usage::

    python scripts/setup_env.py [--skip-install] [--pull-model]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import venv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = PROJECT_ROOT / ".venv"
MIN_PYTHON = (3, 11)


def venv_python() -> Path:
    """Path to the virtual environment's Python interpreter."""
    return VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def check_python() -> None:
    """Exit if the running interpreter is too old."""
    if sys.version_info < MIN_PYTHON:
        sys.exit(f"Python {'.'.join(map(str, MIN_PYTHON))}+ is required (found {sys.version.split()[0]}).")
    print(f"[ok] Python {sys.version.split()[0]}")


def create_venv() -> None:
    """Create the virtual environment if it does not exist."""
    if venv_python().exists():
        print(f"[ok] Virtual environment exists: {VENV_DIR}")
        return
    print(f"[..] Creating virtual environment in {VENV_DIR}")
    venv.EnvBuilder(with_pip=True, upgrade_deps=True).create(VENV_DIR)
    print("[ok] Virtual environment created")


def install_requirements() -> None:
    """Install requirements.txt into the virtual environment.

    Raises:
        subprocess.CalledProcessError: If pip fails.
    """
    print("[..] Installing requirements (this can take several minutes)")
    subprocess.run(
        [str(venv_python()), "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")],
        check=True,
    )
    print("[ok] Requirements installed")


def ensure_env_file() -> None:
    """Copy .env.example to .env when no .env exists."""
    env_file, example = PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.example"
    if env_file.exists():
        print("[ok] .env exists")
    elif example.exists():
        shutil.copyfile(example, env_file)
        print("[ok] Created .env from .env.example - add your ANTHROPIC_API_KEY to enable Claude")
    else:
        print("[warn] .env.example missing; skipping .env creation")


def ensure_directories() -> None:
    """Create the data directory tree."""
    for sub in ("raw", "processed", "reports"):
        (PROJECT_ROOT / "data" / sub).mkdir(parents=True, exist_ok=True)
    print("[ok] Data directories ready")


def read_env_value(key: str, default: str) -> str:
    """Read a value from the environment or .env without importing project code."""
    if os.environ.get(key):
        return os.environ[key]
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == key and not line.lstrip().startswith("#"):
                return value.strip().strip('"').strip("'")
    return default


def check_ollama(pull: bool) -> None:
    """Report Ollama status and optionally pull the configured model."""
    base_url = read_env_value("OLLAMA_BASE_URL", "http://localhost:11434")
    model = read_env_value("OLLAMA_MODEL", "qwen2.5:14b-instruct-q4_K_M")
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/tags", timeout=3) as response:  # noqa: S310
            models = [m.get("name") for m in json.loads(response.read()).get("models", [])]
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        print(f"[warn] Ollama not reachable at {base_url} ({exc}). Local agents will fall back to Claude/rules.")
        return
    if model in models:
        print(f"[ok] Ollama running with model {model}")
        return
    print(f"[warn] Ollama running but model '{model}' is not installed.")
    if pull and shutil.which("ollama"):
        print(f"[..] Pulling {model} (several GB)")
        subprocess.run(["ollama", "pull", model], check=False)
    else:
        print(f"       Install it with: ollama pull {model}")


def main() -> int:
    """Run the setup steps. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Set up the AeroTrace-RCA environment.")
    parser.add_argument("--skip-install", action="store_true", help="Do not run pip install.")
    parser.add_argument("--pull-model", action="store_true", help="Pull the configured Ollama model if missing.")
    args = parser.parse_args()

    check_python()
    create_venv()
    if not args.skip_install:
        try:
            install_requirements()
        except subprocess.CalledProcessError as exc:
            print(f"[error] pip install failed with exit code {exc.returncode}")
            return exc.returncode
    ensure_env_file()
    ensure_directories()
    check_ollama(args.pull_model)
    activate = r".venv\Scripts\activate" if os.name == "nt" else "source .venv/bin/activate"
    print(f"\nDone. Activate with:  {activate}\nThen run:             python scripts/run_pipeline.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
