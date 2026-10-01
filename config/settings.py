"""Central configuration for AeroTrace-RCA.

Values are read from environment variables and the project ``.env`` file via
pydantic-settings. Agent-level tuning lives in ``config/agents.yaml``.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.logger import setup_logging

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
ENV_FILE: Path = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE, override=False)

_PLACEHOLDER_KEYS = {"", "your_claude_api_key_here", "changeme"}


class Settings(BaseSettings):
    """Application settings loaded from the environment and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    anthropic_api_key: SecretStr | None = Field(default=None)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="qwen2.5:14b-instruct-q4_K_M")
    claude_model: str = Field(default="claude-opus-5-5")
    ollama_timeout: float = Field(default=120.0, gt=0)
    ollama_seed: int | None = Field(default=42)
    claude_max_tokens: int = Field(default=4096, gt=0)
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    report_dir: Path = Field(default=PROJECT_ROOT / "data" / "reports")
    log_level: str = Field(default="INFO")
    agents_config_path: Path = Field(default=PROJECT_ROOT / "config" / "agents.yaml")

    @field_validator("data_dir", "report_dir", "agents_config_path", mode="after")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        """Resolve relative paths against the project root instead of the CWD."""
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @field_validator("log_level", mode="after")
    @classmethod
    def _normalise_level(cls, value: str) -> str:
        """Validate the log level name."""
        level = value.upper()
        if level not in logging.getLevelNamesMapping():
            raise ValueError(f"Invalid LOG_LEVEL: {value}")
        return level

    @property
    def raw_dir(self) -> Path:
        """Directory for raw input telemetry."""
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Directory for normalised telemetry and the vector store."""
        return self.data_dir / "processed"

    @property
    def vector_store_dir(self) -> Path:
        """Directory holding the persistent knowledge-base vector store."""
        return self.processed_dir / "knowledge_base"

    @property
    def claude_api_key(self) -> str | None:
        """Return the Anthropic API key, or ``None`` if unset or a placeholder."""
        if self.anthropic_api_key is None:
            return None
        key = self.anthropic_api_key.get_secret_value().strip()
        return None if key in _PLACEHOLDER_KEYS else key

    def ensure_directories(self) -> None:
        """Create every data directory the pipeline writes to."""
        for directory in (self.data_dir, self.raw_dir, self.processed_dir, self.report_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def load_agents_config(self) -> dict[str, Any]:
        """Load ``agents.yaml``; returns an empty dict if the file is missing."""
        if not self.agents_config_path.exists():
            return {}
        with self.agents_config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{self.agents_config_path} must contain a mapping at top level")
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings


def configure_logging(level: str | None = None) -> logging.Logger:
    """Configure Rich logging using the settings' level unless overridden.

    Args:
        level: Optional level name that overrides ``LOG_LEVEL``.
    """
    settings = get_settings()
    return setup_logging(level or settings.log_level)


_settings = get_settings()
OLLAMA_BASE_URL: str = _settings.ollama_base_url
OLLAMA_MODEL: str = _settings.ollama_model
CLAUDE_MODEL: str = _settings.claude_model
ANTHROPIC_API_KEY: str | None = _settings.claude_api_key
DATA_DIR: Path = _settings.data_dir
REPORT_DIR: Path = _settings.report_dir
