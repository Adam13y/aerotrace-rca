"""Supervisor agent: orchestrates the full AeroTrace-RCA pipeline.

Pipeline::

    ingest -> analyse (telemetry analyst) -> detect (statistics) -> contextualise
           -> triage (local Ollama, Claude fallback) -> hypothesise (Claude, Ollama fallback)
           -> remember (knowledge base) -> report (Markdown / JSON)

Every step is executed through :meth:`Supervisor._step`, which retries transient
failures with exponential back-off, applies a fallback when a non-critical step keeps
failing, and records every decision in an auditable decision log.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypeVar

from config.settings import Settings, get_settings
from src.agents.anomaly_detector import AnomalyDetector, DetectorConfig
from src.agents.hypothesis_generator import HypothesisConfig, HypothesisGenerator
from src.agents.report_writer import ReportConfig, ReportWriter
from src.agents.sensor_fault_classifier import SENSOR_FAULT, SensorFaultClassifier, SensorFaultConfig
from src.agents.telemetry_analyst import AnalystConfig, TelemetryAnalyst
from src.ingestion import CSVParserConfig, export_normalized_csv, load_telemetry
from src.memory.historical_rag import HistoricalRAG, OllamaEmbedder
from src.memory.knowledge_base import KnowledgeBase
from src.models.telemetry_schema import AnomalyEvent, Hypothesis, RCAReport, TelemetryAnalysis, TelemetryReading
from src.utils.helpers import LLMRouter
from src.utils.logger import get_logger

logger = get_logger("agents.supervisor")

T = TypeVar("T")
Mode = Literal["analyze", "report"]


class PipelineError(RuntimeError):
    """Raised when a critical pipeline step fails after all retries."""


@dataclass(slots=True)
class Decision:
    """One entry in the supervisor's decision log."""

    timestamp: str
    agent: str
    decision: str
    detail: str = ""


@dataclass(slots=True)
class PipelineResult:
    """Everything produced by one pipeline run."""

    run_id: str
    source_file: str
    mode: str
    readings: list[TelemetryReading] = field(default_factory=list)
    analysis: TelemetryAnalysis | None = None
    anomalies: list[AnomalyEvent] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    report: RCAReport | None = None
    output_paths: dict[str, Path] = field(default_factory=dict)
    decisions: list[Decision] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def actionable_anomalies(self) -> list[AnomalyEvent]:
        """Anomalies that are not planned/expected transients."""
        return [a for a in self.anomalies if not a.expected]


class Supervisor:
    """Coordinates the analyst, detector, hypothesis and report agents."""

    def __init__(
        self,
        settings: Settings | None = None,
        agents_config: dict[str, Any] | None = None,
        use_llm: bool | None = None,
        router: LLMRouter | None = None,
        knowledge_base: KnowledgeBase | None = None,
        parser_config: CSVParserConfig | None = None,
    ) -> None:
        """Initialise every agent from settings and ``agents.yaml``.

        Args:
            settings: Application settings (defaults to :func:`get_settings`).
            agents_config: Parsed ``agents.yaml``; loaded from settings if omitted.
            use_llm: Override ``pipeline.use_llm``; ``False`` runs fully offline.
            router: Pre-built LLM router (mainly for tests).
            knowledge_base: Pre-built knowledge base (mainly for tests).
            parser_config: Ingestion options.
        """
        self.settings = settings or get_settings()
        self.config = agents_config if agents_config is not None else self.settings.load_agents_config()
        pipeline_cfg = self.config.get("pipeline", {}) or {}
        self.max_retries = int(pipeline_cfg.get("max_retries", 3))
        self.retry_delay = float(pipeline_cfg.get("retry_base_delay_seconds", 1.5))
        self.use_llm = bool(pipeline_cfg.get("use_llm", True)) if use_llm is None else use_llm
        self.parser_config = parser_config or CSVParserConfig()
        self.decisions: list[Decision] = []

        self.router = router if router is not None else (LLMRouter.from_settings(self.settings) if self.use_llm else None)
        self.knowledge_base = knowledge_base or self._init_knowledge_base()

        analyst_cfg = AnalystConfig.from_dict(self.config.get("telemetry_analyst"))
        self.analyst = TelemetryAnalyst(analyst_cfg)
        self.sensor_fault_classifier = SensorFaultClassifier(
            SensorFaultConfig.from_dict(self.config.get("sensor_fault_classifier"))
        )
        self.detector = AnomalyDetector(
            DetectorConfig.from_dict(self.config.get("anomaly_detector")), router=self.router, use_llm=self.use_llm
        )
        rag_cfg = self.config.get("historical_rag", {}) or {}
        self.historical_rag: HistoricalRAG | None = None
        if self.use_llm and rag_cfg.get("enabled", True):
            self.historical_rag = HistoricalRAG(
                self.settings.historical_store_dir,
                OllamaEmbedder(self.settings.ollama_base_url, self.settings.embedding_model),
                chunk_tokens=int(rag_cfg.get("chunk_tokens", 500)),
                overlap_tokens=int(rag_cfg.get("overlap_tokens", 50)),
            )
        self.hypothesis_generator = HypothesisGenerator(
            self.knowledge_base,
            historical_rag=self.historical_rag,
            router=self.router,
            config=HypothesisConfig.from_dict(
                self.config.get("hypothesis_generator"),
                analyst_cfg.correlation_window_seconds,
                analyst_cfg.flight_phases,
            ),
            use_llm=self.use_llm,
        )
        self.report_config = ReportConfig.from_dict(self.config.get("report_writer"))
        self.report_writer = ReportWriter(self.settings.report_dir, router=self.router, config=self.report_config, use_llm=self.use_llm)

    def _init_knowledge_base(self) -> KnowledgeBase:
        try:
            return KnowledgeBase(self.settings.vector_store_dir)
        except Exception as exc:  # vector backends raise many error types
            logger.warning("Persistent knowledge base unavailable (%s); using JSON-backed store", exc)
            from src.memory.vector_store import VectorStore

            return KnowledgeBase(self.settings.vector_store_dir, store=VectorStore(self.settings.vector_store_dir, prefer_chroma=False))

    # ------------------------------------------------------------------ #
    # Decision log and step execution
    # ------------------------------------------------------------------ #
    def log_decision(self, agent: str, decision: str, detail: str = "") -> None:
        """Record and log a supervisor decision."""
        entry = Decision(datetime.now(UTC).isoformat(timespec="milliseconds"), agent, decision, detail)
        self.decisions.append(entry)
        logger.info("[%s] %s%s", agent, decision, f" - {detail}" if detail else "")

    def _step(
        self,
        name: str,
        func: Callable[[], T],
        *,
        attempts: int = 1,
        fallback: Callable[[], T] | None = None,
        errors: list[str] | None = None,
    ) -> T:
        """Execute a pipeline step with retries and optional fallback.

        Raises:
            PipelineError: If the step fails and no fallback is provided.
        """
        last_error: Exception | None = None
        for attempt in range(1, max(1, attempts) + 1):
            started = time.perf_counter()
            try:
                result = func()
            except Exception as exc:  # the supervisor is the error boundary for every agent
                last_error = exc
                self.log_decision("supervisor", f"Step '{name}' failed (attempt {attempt}/{attempts})", repr(exc))
                if attempt < attempts:
                    time.sleep(self.retry_delay * (2 ** (attempt - 1)))
                continue
            logger.debug("Step '%s' finished in %.2fs", name, time.perf_counter() - started)
            return result
        message = f"Step '{name}' failed after {attempts} attempt(s): {last_error}"
        if errors is not None:
            errors.append(message)
        if fallback is not None:
            self.log_decision("supervisor", f"Using fallback for step '{name}'", str(last_error))
            return fallback()
        raise PipelineError(message) from last_error

    def check_providers(self) -> list[str]:
        """Probe LLM providers and record the routing decision.

        Returns:
            Names of available providers (``local``, ``cloud``).
        """
        if not self.use_llm or self.router is None:
            self.log_decision("supervisor", "LLM usage disabled - running deterministic offline pipeline")
            return []
        providers = self.router.available_providers()
        local_model = self.router.local.model if self.router.local else "-"
        cloud_model = self.router.cloud.model if self.router.cloud else "-"
        if not providers:
            self.log_decision(
                "supervisor", "No LLM providers available - falling back to rule-based triage and hypotheses",
                f"local={local_model}, cloud={cloud_model}",
            )
        elif "local" not in providers:
            self.log_decision("supervisor", "Local Ollama unavailable - routing local work to Claude", f"cloud={cloud_model}")
        elif "cloud" not in providers:
            self.log_decision("supervisor", "Claude unavailable - routing cloud work to local Ollama", f"local={local_model}")
        else:
            self.log_decision("supervisor", "Both providers available", f"local={local_model}, cloud={cloud_model}")
        return providers

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #
    def run(
        self,
        input_path: str | Path,
        mode: Mode = "report",
        output_formats: Sequence[str] = ("markdown",),
        mission_context: str | None = None,
    ) -> PipelineResult:
        """Execute the pipeline on a telemetry file.

        Args:
            input_path: CSV, JSON/JSONL or FITS telemetry file.
            mode: ``"analyze"`` stops after triage; ``"report"`` runs hypotheses and reporting too.
            output_formats: Report formats - ``"markdown"`` and/or ``"json"``.
            mission_context: Free-text mission description for the LLM agents.

        Raises:
            PipelineError: If a critical step (ingestion, detection) fails.
        """
        started = time.perf_counter()
        input_path = Path(input_path)
        run_id = f"RUN-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"
        self.decisions = []
        result = PipelineResult(run_id=run_id, source_file=str(input_path), mode=mode)
        self.log_decision("supervisor", f"Starting run {run_id}", f"input={input_path.name}, mode={mode}")
        context = mission_context or f"Mission: {self.report_config.mission_name}. Vehicle: {self.report_config.vehicle}."

        result.providers = self.check_providers()

        result.readings = self._step(
            "ingest", lambda: load_telemetry(input_path, self.parser_config), attempts=2, errors=result.errors
        )
        self.log_decision("ingestion", f"Loaded {len(result.readings)} readings", input_path.name)
        self._step(
            "export_normalized",
            lambda: export_normalized_csv(result.readings, self.settings.processed_dir / f"{input_path.stem}_normalized.csv"),
            fallback=lambda: None,
            errors=result.errors,
        )

        result.analysis = self._step(
            "analyse", lambda: self.analyst.analyze(result.readings, source=input_path.name), errors=result.errors
        )
        mission_start = TelemetryAnalyst.mission_start(result.readings)

        detected = self._step(
            "detect", lambda: self.detector.detect(result.readings, mission_start=mission_start), errors=result.errors
        )
        self.log_decision(
            "anomaly_detector", f"Detected {len(detected)} anomalies",
            f"{sum(not a.expected for a in detected)} actionable, {sum(a.expected for a in detected)} expected",
        )
        contextualised = self.analyst.contextualize(detected)
        result.anomalies = self._step(
            "triage",
            lambda: self.detector.classify(contextualised),
            attempts=self.max_retries,
            fallback=lambda: AnomalyDetector(self.detector.config, use_llm=False).classify(contextualised),
            errors=result.errors,
        )
        self.log_decision("anomaly_detector", "Triage complete", f"classifier={self.detector.last_classifier}")

        # Deterministic sensor-fault screen: runs after triage, before any hypothesis generation.
        result.anomalies = self._step(
            "sensor_fault_screen",
            lambda: self.sensor_fault_classifier.classify(result.anomalies),
            fallback=lambda: result.anomalies,
            errors=result.errors,
        )
        for anomaly in result.anomalies:
            fault_class = anomaly.classification.get("fault_class")
            if fault_class:
                previous = anomaly.classification.get("triage_category")
                relabel = f" (triage said '{previous}')" if previous else ""
                self.log_decision(
                    "sensor_fault_classifier", f"{anomaly.label}: {fault_class}{relabel}",
                    f"id={anomaly.id}; {anomaly.classification.get('fault_class_reason', '')}",
                )

        if mode == "analyze":
            result.duration_s = time.perf_counter() - started
            self.log_decision("supervisor", f"Analysis-only run finished in {result.duration_s:.1f}s")
            result.decisions = list(self.decisions)
            return result

        result.hypotheses = self._step(
            "hypothesise",
            lambda: self.hypothesis_generator.generate(result.anomalies, result.analysis, context),
            attempts=self.max_retries,
            fallback=lambda: HypothesisGenerator(self.knowledge_base, use_llm=False, config=self.hypothesis_generator.config)
            .generate(result.anomalies, result.analysis, context),
            errors=result.errors,
        )
        generator = self.hypothesis_generator
        if generator.last_historical_cases:
            self.log_decision(
                "historical_rag", f"Injected {len(generator.last_historical_cases)} historical cases into the prompt",
                "; ".join(f"{c.title} [{c.source_file}] similarity={c.score:.3f}" for c in generator.last_historical_cases),
            )
        elif generator.last_historical_error:
            self.log_decision("historical_rag", "Historical context skipped", generator.last_historical_error)
        for drop in generator.last_drops:
            self.log_decision(
                "hypothesis_validator", f"Dropped citation '{drop.reference}' from '{drop.title}'",
                f"{drop.reason}; anomaly_id={drop.anomaly_id}",
            )
        for retry in generator.last_retries:
            outcome = f"accepted: {retry.accepted}" if retry.accepted else "no valid hypothesis returned"
            self.log_decision(
                "hypothesis_generator", f"Coverage retry {retry.attempt} for unexplained '{retry.label}'",
                f"id={retry.anomaly_id}; {outcome}; reasons_sent={retry.reasons_sent}",
            )
        for flag in self.hypothesis_generator.last_flags:
            self.log_decision(
                "hypothesis_validator", f"Flagged hypothesis '{flag.title}' as low-confidence",
                f"{flag.reason}; cited_ids={flag.cited_ids}",
            )
        for rejection in self.hypothesis_generator.last_rejections:
            self.log_decision(
                "hypothesis_validator", f"Rejected hypothesis '{rejection.title}'",
                f"{rejection.reason}; cited={rejection.cited}; cited_ids={rejection.cited_ids}",
            )
        self.log_decision(
            "hypothesis_generator", f"Generated {len(result.hypotheses)} hypotheses",
            f"source={self.hypothesis_generator.last_source}, "
            f"rejected={len(self.hypothesis_generator.last_rejections)}, "
            f"flagged={len(self.hypothesis_generator.last_flags)}, "
            f"dropped_citations={len(generator.last_drops)}, coverage_retries={len(generator.last_retries)}, "
            f"sensor_faults={sum(a.classification.get('fault_class') == SENSOR_FAULT for a in result.anomalies)}, "
            "anomaly_ids=" + "; ".join(f"{h.title}: {h.related_anomalies}" for h in result.hypotheses),
        )

        self._step(
            "remember",
            lambda: self.knowledge_base.record_run(result.anomalies, result.hypotheses, input_path.name),
            fallback=lambda: 0,
            errors=result.errors,
        )

        metadata = {
            "run_id": run_id,
            "source_file": input_path.name,
            "historical_cases": [
                {"title": c.title, "source_file": c.source_file, "similarity": round(c.score, 3)}
                for c in self.hypothesis_generator.last_historical_cases
            ],
        }
        result.report = self._step(
            "build_report",
            lambda: self.report_writer.build_report(result.anomalies, result.hypotheses, result.analysis, metadata),
            fallback=lambda: ReportWriter(self.settings.report_dir, config=self.report_config, use_llm=False).build_report(
                result.anomalies, result.hypotheses, result.analysis, metadata
            ),
            errors=result.errors,
        )
        # Recorded after report assembly so the executive-summary call is included.
        result.report.metadata["providers"] = self._provider_labels(sorted(self.router.usage) if self.router else [])
        stamp = datetime.now(UTC)
        result.output_paths = self._step(
            "write_report",
            lambda: self.report_writer.write(result.report, result.readings, output_formats, timestamp=stamp),
            attempts=2,
            errors=result.errors,
        )
        result.duration_s = time.perf_counter() - started
        self.log_decision("supervisor", f"Run {run_id} finished in {result.duration_s:.1f}s", f"errors={len(result.errors)}")
        result.output_paths["decision_log"] = self._write_decision_log(result, stamp)
        result.decisions = list(self.decisions)
        return result

    def _provider_labels(self, used: list[str]) -> list[str]:
        labels = []
        if self.router is None:
            return labels
        if "local" in used and self.router.local:
            labels.append(f"Ollama {self.router.local.model}")
        if "cloud" in used and self.router.cloud:
            labels.append(f"Claude {self.router.cloud.model}")
        return labels

    def _write_decision_log(self, result: PipelineResult, stamp: datetime) -> Path:
        path = self.settings.report_dir / f"decision_log_{stamp.strftime('%Y%m%d_%H%M%S')}.json"
        payload = {
            "run_id": result.run_id,
            "source_file": result.source_file,
            "mode": result.mode,
            "duration_s": round(result.duration_s, 3),
            "providers_available": result.providers,
            "llm_calls": dict(self.router.usage) if self.router else {},
            "errors": result.errors,
            "decisions": [asdict(d) for d in self.decisions],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote decision log: %s", path)
        return path
