"""Shared helpers: timestamp normalisation, retries, JSON extraction and LLM clients.

The LLM layer has two providers:

* :class:`OllamaClient` - local GPU worker models (e.g. Qwen 2.5 via Ollama).
* :class:`ClaudeClient` - the cloud supervisor model via the Anthropic API.

:class:`LLMRouter` picks a provider by preference ("local" or "cloud") and falls
back to the other one automatically when the preferred one is unavailable or fails.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, TypeVar

import numpy as np
import pandas as pd

from src.models.telemetry_schema import AnomalyEvent, TelemetryReading, sensor_display_name
from src.utils.logger import get_logger

T = TypeVar("T")
Provider = Literal["local", "cloud"]

logger = get_logger("utils.helpers")

DEFAULT_BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)
_EPOCH_SECONDS_MIN = 1e9  # ~2001-09-09; smaller numbers are treated as relative seconds
_EPOCH_MILLIS_MIN = 1e12
_EPOCH_MICROS_MIN = 1e15
_EPOCH_NANOS_MIN = 1e18


# --------------------------------------------------------------------------- #
# Timestamps and frames
# --------------------------------------------------------------------------- #
def normalize_timestamp(value: Any, base_time: datetime = DEFAULT_BASE_TIME) -> datetime:
    """Convert a raw timestamp into a timezone-aware UTC ``datetime``.

    Accepted inputs: ``datetime``, ISO-8601 strings, epoch seconds/ms/us/ns,
    and relative mission-elapsed seconds (``12.5``, ``"T+12.5"``, ``"T-3"``),
    which are offset from ``base_time``.

    Args:
        value: Raw timestamp value.
        base_time: Reference time used for relative timestamps.

    Raises:
        ValueError: If the value cannot be interpreted as a timestamp.
    """
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return _numeric_to_datetime(float(value), base_time)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Empty timestamp")
        relative = re.fullmatch(r"[Tt]\s*([+-])\s*(\d+(?:\.\d+)?)\s*s?", text)
        if relative:
            sign = 1.0 if relative.group(1) == "+" else -1.0
            return base_time + timedelta(seconds=sign * float(relative.group(2)))
        try:
            return _numeric_to_datetime(float(text), base_time)
        except ValueError:
            pass
        try:
            parsed = pd.Timestamp(text)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Unrecognised timestamp: {value!r}") from exc
        if pd.isna(parsed):
            raise ValueError(f"Unrecognised timestamp: {value!r}")
        return normalize_timestamp(parsed.to_pydatetime(), base_time)
    raise ValueError(f"Unsupported timestamp type: {type(value).__name__}")


def _numeric_to_datetime(number: float, base_time: datetime) -> datetime:
    """Interpret a number as epoch ns/us/ms/s, or relative seconds."""
    if not np.isfinite(number):
        raise ValueError("Non-finite timestamp")
    if abs(number) >= _EPOCH_NANOS_MIN:
        return datetime.fromtimestamp(number / 1e9, tz=UTC)
    if abs(number) >= _EPOCH_MICROS_MIN:
        return datetime.fromtimestamp(number / 1e6, tz=UTC)
    if abs(number) >= _EPOCH_MILLIS_MIN:
        return datetime.fromtimestamp(number / 1000.0, tz=UTC)
    if abs(number) >= _EPOCH_SECONDS_MIN:
        return datetime.fromtimestamp(number, tz=UTC)
    return base_time + timedelta(seconds=number)


def readings_to_frame(readings: Iterable[TelemetryReading]) -> pd.DataFrame:
    """Convert readings to a DataFrame sorted by sensor and time.

    Columns: ``timestamp``, ``sensor_id``, ``value``, ``unit``, ``imputed``, ``index``
    where ``index`` is the position in the original sequence.
    """
    rows = [
        {
            "timestamp": r.timestamp,
            "sensor_id": r.sensor_id,
            "value": r.value,
            "unit": r.unit,
            "imputed": bool(r.metadata.get("imputed", False)),
            "index": i,
        }
        for i, r in enumerate(readings)
    ]
    if not rows:
        return pd.DataFrame(columns=["timestamp", "sensor_id", "value", "unit", "imputed", "index"])
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values(["sensor_id", "timestamp"], kind="stable").reset_index(drop=True)


def robust_zscore(values: np.ndarray) -> np.ndarray:
    """Median/MAD based z-score that is insensitive to the outliers it hunts for."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    median = np.median(values)
    mad = np.median(np.abs(values - median)) * 1.4826
    if mad < 1e-12:
        std = np.std(values)
        mad = std if std > 1e-12 else 1.0
    return (values - median) / mad


def _normalise_reference(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def resolve_anomaly_reference(reference: Any, events: Sequence[AnomalyEvent]) -> AnomalyEvent | None:
    """Resolve an anomaly reference written by an LLM to a detected event.

    Accepts the exact human-readable label (case/whitespace-insensitive), the internal ID,
    or a near-copy of the label that still names the sensor and the exact MET (+/-0.15 s).
    Returns ``None`` when the reference does not identify exactly one event.
    """
    if not isinstance(reference, str) or not reference.strip():
        return None
    key = _normalise_reference(reference)
    for event in events:
        if _normalise_reference(event.label) == key or event.id.lower() == key:
            return event
    def names_sensor(event: AnomalyEvent) -> bool:
        return sensor_display_name(event.sensor_id).lower() in key or event.sensor_id.lower() in key

    match = re.search(r"T\s*\+\s*(\d+(?:\.\d+)?)", reference)
    if match:
        met = float(match.group(1))
        candidates = [
            e for e in events
            if e.mission_elapsed_s is not None and abs(e.mission_elapsed_s - met) <= 0.15 and names_sensor(e)
        ]
        return candidates[0] if len(candidates) == 1 else None
    # No time given (e.g. "lox_tank_pressure CRITICAL anomaly"): accept only if the sensor - and the
    # severity, when one is written - identify exactly one detected anomaly.
    candidates = [e for e in events if names_sensor(e)]
    written = [s for s in ("critical", "warning", "info") if re.search(rf"\b{s}\b", key)]
    if written:
        candidates = [e for e in candidates if e.severity.value.lower() in written]
    return candidates[0] if len(candidates) == 1 else None


def format_met(seconds: float | None) -> str:
    """Format mission elapsed time as ``T+123.4s``."""
    if seconds is None:
        return "T+?"
    sign = "+" if seconds >= 0 else "-"
    return f"T{sign}{abs(seconds):.1f}s"


# --------------------------------------------------------------------------- #
# Retries and JSON parsing
# --------------------------------------------------------------------------- #
def retry_call(
    func: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    description: str = "operation",
) -> T:
    """Call ``func`` with exponential back-off.

    Args:
        func: Zero-argument callable to execute.
        attempts: Total attempts (>= 1).
        base_delay: Initial delay in seconds, doubled after each failure.
        exceptions: Exception types that trigger a retry.
        description: Human-readable name used in log messages.

    Raises:
        The last exception raised by ``func`` once attempts are exhausted.
    """
    attempts = max(1, attempts)
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except exceptions as exc:
            if attempt == attempts:
                logger.error("%s failed after %d attempt(s): %s", description, attempt, exc)
                raise
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "%s failed (attempt %d/%d): %s - retrying in %.1fs",
                description, attempt, attempts, exc, delay,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover


def extract_json(text: str) -> Any:
    """Extract the first JSON object or array from free-form LLM output.

    Handles Markdown code fences and leading/trailing prose.

    Raises:
        ValueError: If no valid JSON document is found.
    """
    if not text or not text.strip():
        raise ValueError("Empty LLM response")
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        candidate = candidate.strip()
        for start, char in enumerate(candidate):
            if char not in "{[":
                continue
            try:
                document, _ = decoder.raw_decode(candidate[start:])
                return document
            except json.JSONDecodeError:
                continue
    raise ValueError("No JSON document found in LLM response")


# --------------------------------------------------------------------------- #
# LLM clients
# --------------------------------------------------------------------------- #
class LLMUnavailableError(RuntimeError):
    """Raised when no LLM provider can serve a request."""


@dataclass(slots=True)
class LLMResponse:
    """Text returned by an LLM together with the provider that produced it."""

    text: str
    provider: str
    model: str


def list_ollama_models(base_url: str, timeout: float = 3.0) -> list[str]:
    """Return the names of models installed on an Ollama server.

    Raises:
        LLMUnavailableError: If the server cannot be reached.
    """
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - configured URL
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise LLMUnavailableError(f"Ollama not reachable at {base_url}: {exc}") from exc
    return [str(model.get("name", "")) for model in payload.get("models", [])]


def is_ollama_available(base_url: str, timeout: float = 3.0) -> bool:
    """Return ``True`` if an Ollama server answers at ``base_url``."""
    try:
        list_ollama_models(base_url, timeout)
    except LLMUnavailableError:
        return False
    return True


def ollama_model_installed(model: str, installed: Sequence[str]) -> bool:
    """Check whether ``model`` (with or without ``:latest``) is in ``installed``."""
    wanted = {model, f"{model}:latest"} if ":" not in model else {model}
    return any(name in wanted for name in installed)


class OllamaClient:
    """Thin wrapper around the local Ollama chat API."""

    def __init__(self, base_url: str, model: str, timeout: float = 120.0, seed: int | None = 42) -> None:
        """Create a client.

        Args:
            base_url: Ollama server URL, e.g. ``http://localhost:11434``.
            model: Installed model tag.
            timeout: Request timeout in seconds.
            seed: Fixed sampling seed for repeatable output (``None`` = random).
        """
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.seed = seed
        self._available: bool | None = None
        self._client: Any = None

    def available(self, refresh: bool = False) -> bool:
        """Return whether the server is up and the configured model is installed."""
        if self._available is not None and not refresh:
            return self._available
        try:
            installed = list_ollama_models(self.base_url)
        except LLMUnavailableError as exc:
            logger.info("Local model unavailable: %s", exc)
            self._available = False
            return False
        self._available = ollama_model_installed(self.model, installed)
        if not self._available:
            logger.info(
                "Ollama is running but model '%s' is not installed (run: ollama pull %s)",
                self.model, self.model,
            )
        return self._available

    def chat(self, system: str, prompt: str, *, json_mode: bool = False, temperature: float = 0.2) -> str:
        """Send a single-turn chat request and return the assistant text.

        Raises:
            LLMUnavailableError: If the ``ollama`` package is missing or the call fails.
        """
        if self._client is None:
            try:
                import ollama
            except ImportError as exc:
                raise LLMUnavailableError("The 'ollama' package is not installed") from exc
            self._client = ollama.Client(host=self.base_url, timeout=self.timeout)
        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                format="json" if json_mode else "",
                options={"temperature": temperature, **({"seed": self.seed} if self.seed is not None else {})},
            )
        except Exception as exc:  # ollama raises httpx / ResponseError variants
            self._available = None
            raise LLMUnavailableError(f"Ollama request failed: {exc}") from exc
        return str(response["message"]["content"])


class ClaudeClient:
    """Thin wrapper around the Anthropic Messages API."""

    def __init__(self, api_key: str | None, model: str, max_tokens: int = 4096, timeout: float = 120.0) -> None:
        """Create a client.

        Args:
            api_key: Anthropic API key; ``None`` disables the client.
            model: Claude model ID.
            max_tokens: Response token budget.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client: Any = None

    def available(self) -> bool:
        """Return whether an API key is configured and the SDK is importable."""
        if not self.api_key:
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            logger.warning("The 'anthropic' package is not installed; Claude disabled")
            return False
        return True

    def chat(self, system: str, prompt: str, *, json_mode: bool = False, temperature: float = 0.2) -> str:
        """Send a single-turn request to Claude and return the text response.

        ``temperature`` is accepted for interface parity with :class:`OllamaClient`
        but not forwarded; Claude's default sampling is used.

        Raises:
            LLMUnavailableError: If Claude is not configured or the call fails.
        """
        if not self.available():
            raise LLMUnavailableError("Claude is not configured (missing ANTHROPIC_API_KEY)")
        import anthropic

        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout, max_retries=2)
        user_prompt = prompt
        if json_mode:
            user_prompt += "\n\nRespond with a single valid JSON document and nothing else."
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except anthropic.APIError as exc:
            raise LLMUnavailableError(f"Claude request failed: {exc}") from exc
        return "".join(block.text for block in message.content if getattr(block, "type", "") == "text")


class LLMRouter:
    """Routes requests to local (Ollama) or cloud (Claude) models with fallback."""

    def __init__(self, local: OllamaClient | None, cloud: ClaudeClient | None) -> None:
        """Create a router.

        Args:
            local: Local Ollama client, or ``None`` to disable local inference.
            cloud: Claude client, or ``None`` to disable cloud inference.
        """
        self.local = local
        self.cloud = cloud
        self.usage: dict[str, int] = {}

    @classmethod
    def from_settings(cls, settings: Any) -> LLMRouter:
        """Build a router from a :class:`config.settings.Settings` instance."""
        return cls(
            OllamaClient(settings.ollama_base_url, settings.ollama_model, settings.ollama_timeout, settings.ollama_seed),
            ClaudeClient(settings.claude_api_key, settings.claude_model, settings.claude_max_tokens),
        )

    def available_providers(self) -> list[str]:
        """List providers that are currently usable."""
        providers = []
        if self.local is not None and self.local.available():
            providers.append("local")
        if self.cloud is not None and self.cloud.available():
            providers.append("cloud")
        return providers

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        prefer: Provider = "local",
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Run a completion on the preferred provider, falling back to the other.

        Raises:
            LLMUnavailableError: If every provider is unavailable or fails.
        """
        order: list[tuple[str, OllamaClient | ClaudeClient | None]] = (
            [("local", self.local), ("cloud", self.cloud)]
            if prefer == "local"
            else [("cloud", self.cloud), ("local", self.local)]
        )
        errors: list[str] = []
        for name, client in order:
            if client is None or not client.available():
                errors.append(f"{name}: unavailable")
                continue
            try:
                text = client.chat(system, prompt, json_mode=json_mode, temperature=temperature)
            except LLMUnavailableError as exc:
                logger.warning("%s provider failed, trying fallback: %s", name, exc)
                errors.append(f"{name}: {exc}")
                continue
            self.usage[name] = self.usage.get(name, 0) + 1
            if name != prefer:
                logger.info("Served by fallback provider '%s' (%s)", name, client.model)
            return LLMResponse(text=text, provider=name, model=client.model)
        raise LLMUnavailableError("; ".join(errors) or "no providers configured")

    def complete_json(
        self,
        system: str,
        prompt: str,
        *,
        prefer: Provider = "local",
        temperature: float = 0.2,
    ) -> tuple[Any, LLMResponse]:
        """Like :meth:`complete` but parses the response as JSON.

        If the preferred provider returns unparsable output, the other provider is tried.

        Raises:
            LLMUnavailableError: If no provider yields valid JSON.
        """
        response = self.complete(system, prompt, prefer=prefer, json_mode=True, temperature=temperature)
        try:
            return extract_json(response.text), response
        except ValueError as exc:
            logger.warning("Invalid JSON from %s: %s", response.provider, exc)
        alternate: Provider = "cloud" if response.provider == "local" else "local"
        client = self.cloud if alternate == "cloud" else self.local
        if client is None or not client.available():
            raise LLMUnavailableError("Provider returned invalid JSON and no fallback is available")
        text = client.chat(system, prompt, json_mode=True, temperature=temperature)
        self.usage[alternate] = self.usage.get(alternate, 0) + 1
        fallback = LLMResponse(text=text, provider=alternate, model=client.model)
        try:
            return extract_json(text), fallback
        except ValueError as exc:
            raise LLMUnavailableError(f"No provider returned valid JSON: {exc}") from exc
