"""Hypothesis generation agent.

Builds an engineering context (anomalies, cross-sensor correlation groups, flight
phases and knowledge-base matches) and asks Claude - the cloud supervisor model - to
produce ranked root-cause hypotheses. Falls back to the local Ollama model and, if no
LLM is reachable, to deterministic knowledge-base-driven hypotheses.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from src.agents.evidence_rules import UNOBSERVABLE_LABEL, EvidenceRules, normalise_type
from src.agents.sensor_fault_classifier import SENSOR_FAULT
from src.agents.telemetry_analyst import FlightPhase, TelemetryAnalyst
from src.memory.historical_rag import (
    DEFAULT_SENSOR_CATEGORIES,
    DEFAULT_SENSOR_CATEGORY_PATTERNS,
    HistoricalCase,
    HistoricalRAG,
    anomaly_query,
    format_historical_context,
    sensor_category,
)
from src.memory.knowledge_base import KnowledgeBase
from src.memory.vector_store import SearchResult
from src.models.telemetry_schema import AnomalyEvent, Hypothesis, Severity, TelemetryAnalysis, anomaly_labels
from src.utils.helpers import LLMRouter, LLMUnavailableError, format_met, resolve_anomaly_reference
from src.utils.logger import get_logger

logger = get_logger("agents.hypothesis_generator")

SYSTEM_PROMPT = """You are a senior aerospace failure analysis engineer with 20 years of experience in rocket propulsion systems. You have worked on Apollo, Space Shuttle, Falcon 9, and Starship programs.

When analyzing telemetry anomalies:
1. Always identify the FIRST anomaly in time as the likely root cause
2. Look for causal chains (e.g., pressure drop → cavitation → vibration)
3. Use precise aerospace terminology (LOX, turbopump, cavitation, combustion instability)
4. Consider known failure modes from historical incidents
5. Rank hypotheses by: (a) temporal precedence, (b) number of anomalies explained, (c) physical plausibility

Output format:
- Hypothesis title (concise, technical)
- Confidence percentage (0-100%)
- Causal chain explanation (what caused what)
- Supporting evidence (which anomalies support this)
- Counter-evidence (what doesn't fit)
- Recommended next steps for investigation"""

HYPOTHESIS_PROMPT_TEMPLATE = """## Mission context
{mission_context}

## Data quality notes
{quality_notes}

## Detected anomalies (non-expected, ordered by time)
{anomalies}

## Instrumentation glitches (already explained by the sensor-fault classifier - do NOT cite or use in causal chains)
{sensor_faults}

## Correlation groups (anomalies within {window}s of each other on different sensors)
{groups}

## Leading correlation group (the main failure chain)
{leading_group}
Note: anomalies in a correlation group are close in time. That alone does NOT mean they are causally related.

## Vehicle sensors by subsystem
{subsystem_sensors}

## Physically plausible causal couplings between subsystems (cause -> effect)
{couplings}

## Flight phases (mission elapsed time windows)
{phases}

## Expected transients (planned events, for context only)
{expected}

## Knowledge-base matches (prior failure modes and past anomalies)
{kb_matches}

## Historical Context (most similar past aerospace failures, retrieved from the failure database)
Use these as prior art to recognise causal chains and telemetry signatures. They are NOT evidence about this
flight: every claim must be supported by this flight's telemetry, and never cite a historical case in
"related_anomalies".

{historical_context}

Generate up to {max_hypotheses} root-cause hypotheses covering every correlation group, ranked by confidence.

COMPETING HYPOTHESES: for the leading correlation group, give AT LEAST {min_competing} DISTINCT, ranked explanations.
Each has a "type":
- "root_cause": a physical mechanism (different mechanisms, not rewordings of the same one);
- "independent_events": the group's anomalies are coincident in time but causally unrelated (cite all of them);
- "insufficient_evidence": the data cannot discriminate between the candidate causes (say which ones).
Only propose mechanisms the telemetry supports: do not pad the list with causes in subsystems that show no anomaly.
For each explanation, fill "distinguishing_evidence" with the observations that would confirm it and rule out the others.

IMPORTANT: When building causal chains, anomalies must be ordered by time. The FIRST anomaly in time is the most likely root cause. Do not connect anomalies more than 30 seconds apart unless there is a clear physical mechanism connecting them.

Citation rules (hypotheses that break them are rejected automatically):
- In "related_anomalies", copy anomaly labels EXACTLY as written in the "anomaly" field above.
- Cite only anomalies that your description and supporting evidence actually discuss, naming their sensors.
- Cite anomalies from different correlation groups only if you quote each one's time (e.g. "at T+142.4s").
- If you link anomalies more than 30 seconds apart, state the delay and the mechanism that explains it.
- Only name a flight phase (e.g. Max-Q, throttle bucket, engine cutoff) if a cited anomaly occurred during it.
- Only name a sensor in the title (e.g. "Avionics Bus Under-Voltage") if you cite that sensor's anomaly.
- Never cite expected transients or instrumentation glitches.
- A subsystem you name as part of the cause must have a cited anomaly from one of its sensors (see the sensor list);
  a subsystem with no sensors can only be proposed as unobservable, with low confidence.
- A causal link between anomalies of different subsystems must follow a listed coupling (cause -> effect).

Respond with JSON exactly in this shape:
{{"hypotheses": [{{
  "type": "<root_cause | independent_events | insufficient_evidence>",
  "title": "<short name of the root cause>",
  "description": "<mechanism, causal chain and how it explains the timeline>",
  "confidence": <0.0-1.0>,
  "subsystem": "<subsystem>",
  "related_anomalies": ["<anomaly label, copied exactly>", ...],
  "supporting_evidence": ["<evidence from the telemetry>", ...],
  "counter_evidence": ["<evidence against, or what would falsify it>", ...],
  "distinguishing_evidence": ["<observation that would confirm this cause and rule out the competing ones>", ...],
  "recommended_actions": ["<concrete next step>", ...]
}}]}}"""


COVERAGE_PROMPT_TEMPLATE = """## Mission context
{mission_context}

## Flight phases (mission elapsed time windows)
{phases}

## UNEXPLAINED anomaly - explain this one
{anomaly}

## Other real anomalies in this flight (context; cite only if causally linked within 30 s)
{others}

## Knowledge-base matches for the unexplained anomaly
{kb_matches}

## Historical Context (similar past aerospace failures - prior art only, never cite them as anomalies)
{historical_context}

## Why earlier hypotheses for this flight were rejected or trimmed
{reasons}

None of the accepted root-cause hypotheses validly explains the anomaly "{label}".
Propose up to 2 root-cause hypotheses for it, avoiding the mistakes listed above.

Rules (hypotheses that break them are rejected automatically):
- "related_anomalies" MUST include "{label}", copied exactly.
- Name the sensor of every anomaly you cite in the description, and quote its time (e.g. "at T+209.9s").
- Only name a sensor in the title if you cite that sensor's anomaly.
- Only name a flight phase if the cited anomaly occurred during it.
- IMPORTANT: When building causal chains, anomalies must be ordered by time. The FIRST anomaly in time is the most likely root cause. Do not connect anomalies more than 30 seconds apart unless there is a clear physical mechanism connecting them.

Respond with JSON exactly in this shape:
{{"hypotheses": [{{
  "type": "<root_cause | independent_events | insufficient_evidence>",
  "title": "<short name of the root cause>",
  "description": "<mechanism, causal chain and how it explains the timeline>",
  "confidence": <0.0-1.0>,
  "subsystem": "<subsystem>",
  "related_anomalies": ["<anomaly label, copied exactly>", ...],
  "supporting_evidence": ["<evidence from the telemetry>", ...],
  "counter_evidence": ["<evidence against, or what would falsify it>", ...],
  "distinguishing_evidence": ["<observation that would confirm this cause and rule out the competing ones>", ...],
  "recommended_actions": ["<concrete next step>", ...]
}}]}}"""


ALTERNATIVE_PROMPT_TEMPLATE = """## Mission context
{mission_context}

## Flight phases (mission elapsed time windows)
{phases}

## Leading correlation group (the main failure chain) - needs a competing explanation
{group}

## Root causes ALREADY proposed for this group - do NOT repeat or reword them
{existing}

## Knowledge-base matches for the group's first anomaly
{kb_matches}

## Historical Context (similar past aerospace failures - prior art only, never cite them as anomalies)
{historical_context}

## Why earlier hypotheses for this flight were rejected or trimmed
{reasons}

Only {count} distinct root cause(s) were proposed for the leading correlation group. Propose up to 2 DIFFERENT,
physically distinct root causes that could also explain it, ranked by confidence.

Rules (hypotheses that break them are rejected automatically):
- "related_anomalies" MUST include "{label}", copied exactly, plus any other group anomalies your chain explains.
- The mechanism must differ from every root cause listed above; a rewording of one of them is rejected.
- Fill "distinguishing_evidence" with the observations that would confirm this cause and rule out the ones above.
- Name the sensor of every anomaly you cite in the description, and quote its time (e.g. "at T+142.4s").
- Only name a sensor in the title if you cite that sensor's anomaly.
- Only name a flight phase if the cited anomaly occurred during it.
- IMPORTANT: When building causal chains, anomalies must be ordered by time. The FIRST anomaly in time is the most likely root cause. Do not connect anomalies more than 30 seconds apart unless there is a clear physical mechanism connecting them.

Respond with JSON exactly in this shape:
{{"hypotheses": [{{
  "type": "<root_cause | independent_events | insufficient_evidence>",
  "title": "<short name of the root cause>",
  "description": "<mechanism, causal chain and how it explains the timeline>",
  "confidence": <0.0-1.0>,
  "subsystem": "<subsystem>",
  "related_anomalies": ["<anomaly label, copied exactly>", ...],
  "supporting_evidence": ["<evidence from the telemetry>", ...],
  "counter_evidence": ["<evidence against, or what would falsify it>", ...],
  "distinguishing_evidence": ["<observation that would confirm this cause and rule out the ones above>", ...],
  "recommended_actions": ["<concrete next step>", ...]
}}]}}"""

#: Words ignored when comparing hypothesis titles for near-duplicates.
_TITLE_STOPWORDS = frozenset(
    "a an and the of to from due in on by or with for at via caused induced system failure fault anomaly".split()
)


def title_tokens(title: str) -> set[str]:
    """Significant lower-case words of a hypothesis title."""
    return {w for w in re.findall(r"[a-z0-9]+", title.lower()) if w not in _TITLE_STOPWORDS}


def near_duplicate(a: str, b: str) -> bool:
    """Heuristic: two titles name the same root cause (word overlap >= 50% or one contains the other).

    This compares wording, not physics; it catches rewordings such as "LOX Tank Pressurisation Leak"
    vs "LOX Tank Pressurization System Leak", but cannot judge whether two differently worded
    mechanisms are physically the same.
    """
    ta, tb = title_tokens(a), title_tokens(b)
    if not ta or not tb:
        return a.strip().lower() == b.strip().lower()
    return len(ta & tb) / len(ta | tb) >= 0.5 or ta <= tb or tb <= ta


#: Sensor-name words too generic to identify a sensor on their own.
_GENERIC_SENSOR_WORDS = frozenset(
    {"tank", "pressure", "temp", "temperature", "bus", "bay", "sensor", "engine", "main", "system", "level", "rate"}
)
#: Extra phrases that count as naming a sensor word.
_SENSOR_SYNONYMS: dict[str, tuple[str, ...]] = {
    "lox": ("liquid oxygen", "oxidizer", "oxidiser"),
    "temp": ("temperature", "thermal"),
    "voltage": ("under-voltage", "undervoltage", "brownout"),
    "vibration": ("vibrations",),
}
#: How far (s) a time quoted in the hypothesis may be from a cited anomaly and still be consistent.
_TIME_TOLERANCE_S = 10.0
_MET_PATTERN = re.compile(r"T\s*\+\s*(\d+(?:\.\d+)?)")
#: How close (s) a quoted time must be to an anomaly's interval to count as quoting that anomaly.
_QUOTE_TOLERANCE_S = 1.0
#: Cited anomalies further apart than this (s) need an explicit explanation of the delay.
_MAX_UNEXPLAINED_GAP_S = 30.0
#: Penalty for hypotheses that link distant anomalies without explaining the gap.
_GAP_CONFIDENCE_FACTOR = 0.5
_GAP_CONFIDENCE_CAP = 0.3
_GAP_EXPLANATION = re.compile(
    r"\b(?:\d+(?:\.\d+)?\s*(?:s|sec|secs|seconds)\s+(?:later|after|delay|lag|gap)"
    r"|delay(?:ed)?|latency|lag(?:ged)?|time gap|propagat\w*|accumulat\w*|gradual\w*|build-?up"
    r"|over (?:the )?(?:following|next) \d+)",
    re.IGNORECASE,
)


#: A cited anomaly may start/end this close (s) to a named phase's window and still count as inside it.
_PHASE_TOLERANCE_S = 2.0
#: Extra ways a phase-name fragment may be written in hypothesis text.
_PHASE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "max-q": (r"max[\s-]?q", r"maximum dynamic pressure"),
    "engine cutoff": (r"engine cut-?off", r"meco", r"main engine cut-?off"),
    "liftoff": (r"lift[\s-]?off",),
    "throttle bucket": (r"throttle[\s-]bucket",),
}


def phase_patterns(phase: FlightPhase) -> list[re.Pattern[str]]:
    """Regexes that detect a mention of ``phase`` in free text.

    The phase name from ``agents.yaml`` is split on ``/``, ``&`` and ``,`` so that
    "Throttle bucket / Max-Q" matches either "throttle bucket" or "Max-Q".
    """
    fragments = [f.strip().lower() for f in re.split(r"[/&,]", phase.name) if f.strip()]
    patterns: list[str] = []
    for fragment in fragments:
        patterns.extend(_PHASE_SYNONYMS.get(fragment, (re.escape(fragment).replace(r"\ ", r"[\s-]+"),)))
    return [re.compile(rf"\b(?:{p})\b", re.IGNORECASE) for p in patterns]


def phase_problem(anomalies: list[AnomalyEvent], text: str, phases: list[FlightPhase]) -> str | None:
    """Check that every flight phase named in ``text`` contains at least one cited anomaly.

    Returns:
        A description of the inconsistency, or ``None`` if consistent.
    """
    for phase in phases:
        mention = next((m for p in phase_patterns(phase) if (m := p.search(text))), None)
        if mention is None:
            continue
        inside = [
            a for a in anomalies
            if (a.mission_elapsed_s or 0.0) <= phase.end + _PHASE_TOLERANCE_S
            and (a.mission_elapsed_s or 0.0) + a.duration_s >= phase.start - _PHASE_TOLERANCE_S
        ]
        if not inside:
            window = f"T+{phase.start:g}s to " + ("end of record" if phase.end >= 1e8 else f"T+{phase.end:g}s")
            cited = "; ".join(f"'{a.label}'" for a in anomalies)
            return (
                f"names the '{mention.group(0)}' phase ({phase.name}: {window}) but none of its cited anomalies "
                f"occurred in that window ({cited})"
            )
    return None


def mentions_sensor(sensor_id: str, text: str) -> bool:
    """Whether ``text`` names ``sensor_id`` (see :func:`sensor_phrases`)."""
    lowered = text.lower().replace("_", " ")
    return any(re.search(rf"\b{re.escape(p)}\b", lowered) for p in sensor_phrases(sensor_id))


def topic_problem(title: str, cited: list[AnomalyEvent], sensors: set[str]) -> str | None:
    """Reject titles that name a sensor whose anomaly the hypothesis does not cite.

    Args:
        title: Hypothesis title.
        cited: The hypothesis's (valid) cited anomalies.
        sensors: Sensors that have actionable anomalies in this flight.
    """
    cited_sensors = {a.sensor_id for a in cited}
    named = sorted(s for s in sensors if mentions_sensor(s, title))
    missing = [s for s in named if s not in cited_sensors]
    if missing:
        names = ", ".join(s.replace("_", " ") for s in missing)
        cited_text = "; ".join(f"'{a.label}'" for a in cited) or "nothing"
        return f"title names the {names} sensor but the hypothesis does not cite its anomaly (cites {cited_text})"
    return None


def quotes_time_of(anomaly: AnomalyEvent, text: str) -> bool:
    """Whether ``text`` quotes a mission time (``T+...``) inside the anomaly's interval (+/-1 s)."""
    start = anomaly.mission_elapsed_s or 0.0
    end = start + anomaly.duration_s
    return any(
        start - _QUOTE_TOLERANCE_S <= float(t) <= end + _QUOTE_TOLERANCE_S for t in _MET_PATTERN.findall(text)
    )


def max_time_gap(anomalies: list[AnomalyEvent]) -> float:
    """Largest gap (s) between the intervals of any two anomalies (0 if they overlap)."""
    if len(anomalies) < 2:
        return 0.0
    starts = [a.mission_elapsed_s or 0.0 for a in anomalies]
    ends = [s + a.duration_s for s, a in zip(starts, anomalies)]
    return max(0.0, max(starts) - min(ends))


def sensor_phrases(sensor_id: str) -> set[str]:
    """Phrases whose presence in text counts as mentioning ``sensor_id``."""
    tokens = [t for t in sensor_id.lower().replace("-", "_").split("_") if t]
    phrases = {" ".join(tokens)}
    if len(tokens) >= 2:
        phrases.add(" ".join(tokens[:2]))
    for token in tokens:
        if token not in _GENERIC_SENSOR_WORDS:
            phrases.add(token)
        phrases.update(_SENSOR_SYNONYMS.get(token, ()))
    return phrases


def citation_problem(anomaly: AnomalyEvent, text: str) -> str | None:
    """Check that hypothesis ``text`` is consistent with a cited anomaly.

    The text must name the anomaly's sensor, and if it quotes mission times (``T+...``) at
    least one must lie within ``_TIME_TOLERANCE_S`` of the anomaly's interval.

    Returns:
        A description of the inconsistency, or ``None`` if consistent.
    """
    lowered = text.lower().replace("_", " ")
    if not any(re.search(rf"\b{re.escape(p)}\b", lowered) for p in sensor_phrases(anomaly.sensor_id)):
        return f"the text never mentions the {anomaly.sensor_id.replace('_', ' ')} sensor"
    quoted = [float(t) for t in _MET_PATTERN.findall(text)]
    start = anomaly.mission_elapsed_s or 0.0
    end = start + anomaly.duration_s
    if quoted and not any(start - _TIME_TOLERANCE_S <= t <= end + _TIME_TOLERANCE_S for t in quoted):
        times = ", ".join(f"T+{t:g}s" for t in quoted[:4])
        return f"the text quotes {times}, none near the anomaly at {format_met(start)}"
    return None


@dataclass(slots=True)
class HypothesisRejection:
    """A hypothesis discarded by the validation layer."""

    title: str
    reason: str
    cited: list[str] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CategoryPreference:
    """Which knowledge-base categories get the retrieval bonus, and why."""

    categories: set[str]
    leading_group: list[AnomalyEvent]
    counts: dict[str, int]


@dataclass(slots=True)
class DroppedReference:
    """A single bad citation removed from an otherwise acceptable hypothesis."""

    title: str
    reference: str
    reason: str
    anomaly_id: str | None = None


@dataclass(slots=True)
class CoverageRetry:
    """One re-ask of the LLM for an anomaly left unexplained after validation."""

    anomaly_id: str
    label: str
    attempt: int
    reasons_sent: list[str]
    accepted: list[str] = field(default_factory=list)
    historical_cases: list[str] = field(default_factory=list)
    kind: str = "coverage"  # "coverage" (unexplained anomaly) or "alternative" (competing root cause)


@dataclass(slots=True)
class HypothesisFlag:
    """A hypothesis accepted with reduced confidence by the validation layer."""

    title: str
    reason: str
    original_confidence: float
    adjusted_confidence: float
    cited_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class HypothesisConfig:
    """Configuration for :class:`HypothesisGenerator`."""

    provider_preference: str = "cloud"
    max_hypotheses: int = 6
    knowledge_base_matches: int = 3
    temperature: float = 0.0
    coverage_retries: int = 1
    historical_cases: int = 5
    #: Minimum number of distinct competing root causes for the leading correlated group.
    min_competing_hypotheses: int = 2
    #: Extra LLM calls to ask for a competing explanation when too few are valid (0 = report the shortfall).
    alternative_retries: int = 0
    evidence_rules: EvidenceRules = field(default_factory=EvidenceRules)
    correlation_window_seconds: float = 5.0
    flight_phases: list[FlightPhase] = field(default_factory=list)
    #: Sensor -> knowledge-base category, used for the retrieval category bonus.
    sensor_categories: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SENSOR_CATEGORIES))
    sensor_category_patterns: list[tuple[str, str]] = field(
        default_factory=lambda: list(DEFAULT_SENSOR_CATEGORY_PATTERNS)
    )
    #: "dominant": only the leading group's most frequent category; "all": every category in the group.
    category_preference: str = "dominant"

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        correlation_window: float = 5.0,
        flight_phases: list[FlightPhase] | None = None,
        rag: dict[str, Any] | None = None,
    ) -> HypothesisConfig:
        """Build from the ``hypothesis_generator`` section of ``agents.yaml``.

        ``flight_phases`` come from the ``telemetry_analyst`` section and drive the
        phase-consistency check; ``rag`` is the ``historical_rag`` section (sensor categories).
        """
        data = data or {}
        rag = rag or {}
        mapping = {str(k).lower(): str(v).lower() for k, v in (rag.get("sensor_categories") or {}).items()}
        patterns = [(str(p).lower(), str(c).lower()) for p, c in (rag.get("sensor_category_patterns") or [])]
        preference = str(rag.get("category_preference", "dominant")).lower()
        if preference not in {"dominant", "all"}:
            raise ValueError(f"historical_rag.category_preference must be 'dominant' or 'all', got {preference!r}")
        return cls(
            provider_preference=str(data.get("provider_preference", "cloud")),
            max_hypotheses=int(data.get("max_hypotheses", 6)),
            knowledge_base_matches=int(data.get("knowledge_base_matches", 3)),
            temperature=float(data.get("temperature", 0.0)),
            coverage_retries=int(data.get("coverage_retries", 1)),
            historical_cases=int(data.get("historical_cases", 5)),
            min_competing_hypotheses=int(data.get("min_competing_hypotheses", 2)),
            alternative_retries=int(data.get("alternative_retries", 0)),
            evidence_rules=EvidenceRules.from_dict(
                data.get("evidence_rules"), mapping or None, patterns or None
            ),
            correlation_window_seconds=correlation_window,
            flight_phases=list(flight_phases or []),
            sensor_categories=mapping or dict(DEFAULT_SENSOR_CATEGORIES),
            sensor_category_patterns=patterns or list(DEFAULT_SENSOR_CATEGORY_PATTERNS),
            category_preference=preference,
        )


class HypothesisGenerator:
    """Generate ranked root-cause hypotheses for anomaly events."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        router: LLMRouter | None = None,
        config: HypothesisConfig | None = None,
        use_llm: bool = True,
        historical_rag: HistoricalRAG | None = None,
    ) -> None:
        """Create the agent.

        Args:
            knowledge_base: Knowledge base used for similarity search.
            router: LLM router (Claude preferred); ``None`` means rule-based only.
            config: Agent configuration.
            use_llm: Disable LLM use even if a router is supplied.
            historical_rag: Optional retriever of historical failure cases (RAG).
        """
        self.kb = knowledge_base
        self.historical_rag = historical_rag
        self.last_historical_cases: list[HistoricalCase] = []
        self.last_historical_error: str | None = None
        self.router = router
        self.config = config or HypothesisConfig()
        self.use_llm = use_llm and router is not None
        self.last_source = "none"
        self.last_rejections: list[HypothesisRejection] = []
        self.last_flags: list[HypothesisFlag] = []
        self.last_drops: list[DroppedReference] = []
        self.last_retries: list[CoverageRetry] = []

    def generate(
        self,
        anomalies: list[AnomalyEvent],
        analysis: TelemetryAnalysis | None = None,
        mission_context: str = "",
        dataset_id: str | None = None,
        source_file: str | None = None,
    ) -> list[Hypothesis]:
        """Produce hypotheses ranked by confidence (highest first).

        ``dataset_id``/``source_file`` identify the dataset under analysis; past anomalies saved
        from that same dataset are excluded from retrieval so a re-run never sees its own output.

        Flow: instrumentation-glitch hypotheses for sensor faults; one LLM call for the physical
        anomalies; validation; a focused coverage retry for each physical anomaly still
        unexplained; rule-based fallback if the LLM yields nothing usable.

        Args:
            anomalies: Classified, contextualised anomaly events.
            analysis: Optional dataset overview for context.
            mission_context: Free-text description of the vehicle and mission.
        """
        actionable = sorted((a for a in anomalies if not a.expected), key=lambda a: a.timestamp)
        self.last_rejections, self.last_flags, self.last_drops, self.last_retries = [], [], [], []
        self.last_historical_cases, self.last_historical_error = [], None
        self.last_category_preference: CategoryPreference | None = None
        self.last_leading_group: list[AnomalyEvent] = []
        self.last_competing: list[str] = []
        self.last_retrieval_query: str | None = None
        self._vehicle_sensors = (
            sorted(s.sensor_id for s in analysis.sensors) if analysis is not None
            else sorted({a.sensor_id for a in anomalies})
        )
        if not actionable:
            logger.info("No actionable anomalies - no hypotheses generated")
            self.last_source = "none"
            return []

        # Sensor faults (from the sensor-fault classifier) get a dedicated instrumentation hypothesis
        # and are kept out of the physical root-cause reasoning.
        faults = [a for a in actionable if a.classification.get("fault_class") == SENSOR_FAULT]
        physical = [a for a in actionable if a not in faults]
        glitches = [self._sensor_fault_hypothesis(a) for a in faults]

        hypotheses: list[Hypothesis] = []
        self.last_source = "sensor-fault-classifier"
        if physical:
            self.kb.excluded_ids = set()
            kb_matches = {
                a.id: self.kb.context_for(
                    a, k=self.config.knowledge_base_matches, exclude_dataset=dataset_id, exclude_source=source_file
                )
                for a in physical
            }
            if self.use_llm and self.router is not None:
                self.last_category_preference = self.category_preference(physical)
                self.last_retrieval_query = anomaly_query(physical)
                self.last_historical_cases = self._retrieve_history(
                    physical, self.config.historical_cases, self.last_category_preference
                )
                try:
                    self.last_leading_group = self.category_preference(physical).leading_group
                    hypotheses = self._llm_hypotheses(anomalies, physical, kb_matches, analysis, mission_context)
                    if hypotheses:
                        hypotheses += self._coverage_retries(anomalies, physical, hypotheses, kb_matches, mission_context)
                        hypotheses = self._drop_duplicate_competitors(hypotheses, self.last_leading_group)
                        hypotheses += self._alternative_retries(
                            anomalies, physical, hypotheses, self.last_leading_group, kb_matches, mission_context
                        )
                except (LLMUnavailableError, ValidationError, ValueError) as exc:
                    logger.warning("LLM hypothesis generation failed, using knowledge-base fallback: %s", exc)
            if not hypotheses:
                hypotheses = self._rule_based(physical, kb_matches)
                self.last_source = "rule-based"

        competing = self.competing(hypotheses, self.last_leading_group)
        self.last_competing = [h.title for h in competing]
        kept = self._select(
            hypotheses, physical, max(1, self.config.max_hypotheses - len(glitches)),
            must_keep=competing[: self.config.min_competing_hypotheses],
        )
        ranked = sorted(kept + glitches, key=lambda h: h.confidence, reverse=True)
        logger.info("Generated %d hypotheses via %s (+%d instrumentation glitch)", len(kept), self.last_source, len(glitches))
        return ranked

    def category_preference(self, anomalies: list[AnomalyEvent]) -> CategoryPreference:
        """Knowledge-base categories that get the retrieval bonus for ``anomalies``.

        Categories come from the sensor names (``sensor_categories`` in ``agents.yaml``), not from
        the LLM triage labels. Only the leading correlated group - the one with the most anomalies,
        ties broken by the earliest group - counts. In ``dominant`` mode only that group's most
        frequent category is preferred (ties go to the category of its earliest anomaly), so one
        secondary anomaly in the chain cannot pull in cases from its own category.
        """
        groups = TelemetryAnalyst.correlation_groups(anomalies)
        if not groups:
            return CategoryPreference(set(), [], {})
        lead = max(groups, key=lambda g: (len(g), -min(a.timestamp for a in g).timestamp()))
        lead = sorted(lead, key=lambda a: a.timestamp)
        by_anomaly = {
            a.id: sensor_category(a.sensor_id, self.config.sensor_categories, self.config.sensor_category_patterns)
            for a in lead
        }
        counts: dict[str, int] = {}
        for anomaly in lead:
            category = by_anomaly[anomaly.id]
            if category:
                counts[category] = counts.get(category, 0) + 1
        if not counts:
            return CategoryPreference(set(), lead, counts)
        if self.config.category_preference == "all":
            return CategoryPreference(set(counts), lead, counts)
        top = max(counts.values())
        earliest_first = [by_anomaly[a.id] for a in lead if by_anomaly[a.id] and counts[by_anomaly[a.id]] == top]
        return CategoryPreference({earliest_first[0]}, lead, counts)

    def _retrieve_history(
        self, anomalies: list[AnomalyEvent], k: int, preference: CategoryPreference | None = None
    ) -> list[HistoricalCase]:
        """Retrieve similar historical cases; RAG is optional, so failures only log a warning."""
        if self.historical_rag is None or k <= 0:
            return []
        preference = preference or self.category_preference(anomalies)
        try:
            cases = self.historical_rag.search_for_anomalies(anomalies, k=k, prefer_categories=preference.categories)
        except Exception as exc:  # retrieval is best-effort context (embedder/Chroma errors vary)
            self.last_historical_error = str(exc)
            logger.warning("Historical RAG unavailable: %s", exc)
            return []
        logger.info(
            "Historical context: %s",
            "; ".join(f"{c.title} ({c.category}, {c.source_type}, {c.score:.2f})" for c in cases),
        )
        return cases

    @staticmethod
    def competing(hypotheses: list[Hypothesis], lead: list[AnomalyEvent]) -> list[Hypothesis]:
        """Distinct hypotheses citing the leading group, best first (near-duplicate titles count once)."""
        lead_ids = {a.id for a in lead}
        distinct: list[Hypothesis] = []
        for hypothesis in sorted(hypotheses, key=lambda h: h.confidence, reverse=True):
            if lead_ids & set(hypothesis.related_anomalies) and not any(
                near_duplicate(hypothesis.title, kept.title) for kept in distinct
            ):
                distinct.append(hypothesis)
        return distinct

    def _drop_duplicate_competitors(self, hypotheses: list[Hypothesis], lead: list[AnomalyEvent]) -> list[Hypothesis]:
        """Remove lower-confidence rewordings of the same leading-group root cause (recorded as rejections)."""
        lead_ids = {a.id for a in lead}
        kept: list[Hypothesis] = []
        for hypothesis in sorted(hypotheses, key=lambda h: h.confidence, reverse=True):
            twin = next(
                (k for k in kept
                 if lead_ids & set(k.related_anomalies) and lead_ids & set(hypothesis.related_anomalies)
                 and near_duplicate(hypothesis.title, k.title)),
                None,
            )
            if twin is not None:
                self.last_rejections.append(HypothesisRejection(
                    hypothesis.title, f"near-duplicate of '{twin.title}' (same root cause reworded)",
                    [], hypothesis.related_anomalies,
                ))
                logger.info("Dropped near-duplicate hypothesis '%s' (of '%s')", hypothesis.title, twin.title)
                continue
            kept.append(hypothesis)
        return kept

    def _alternative_retries(
        self,
        anomalies: list[AnomalyEvent],
        physical: list[AnomalyEvent],
        accepted: list[Hypothesis],
        lead: list[AnomalyEvent],
        kb_matches: dict[str, list[SearchResult]],
        mission_context: str,
    ) -> list[Hypothesis]:
        """Ask for competing root causes when the leading group has too few distinct ones.

        Uses the coverage-retry flow (same validation, same retry budget, recorded in
        ``last_retries`` with ``kind="alternative"``). Answers are kept only if they cite the
        leading group and are not rewordings of an existing root cause. Nothing is generated
        from rules: if the LLM does not supply an alternative, the shortfall is just recorded.
        """
        needed = self.config.min_competing_hypotheses
        if not lead or needed <= 1:
            return []
        added: list[Hypothesis] = []
        root = lead[0]
        for attempt in range(1, self.config.alternative_retries + 1):
            existing = self.competing([*accepted, *added], lead)
            if len(existing) >= needed:
                break
            reasons = self._reasons_for(root)
            record = CoverageRetry(root.id, root.label, attempt, reasons, kind="alternative")
            self.last_retries.append(record)
            logger.info("Alternative retry %d: leading group has %d distinct root cause(s)", attempt, len(existing))
            history = self._retrieve_history(lead, min(3, self.config.historical_cases))
            record.historical_cases = [c.title for c in history]
            prompt = self._alternative_prompt(lead, existing, physical, kb_matches.get(root.id, []), reasons,
                                              mission_context, history)
            try:
                candidates = self._ask_and_validate(prompt, anomalies, physical)
            except (LLMUnavailableError, ValueError) as exc:
                logger.warning("Alternative retry failed: %s", exc)
                continue
            lead_ids = {a.id for a in lead}
            for hypothesis in candidates:
                twin = next((h for h in [*existing, *added] if near_duplicate(hypothesis.title, h.title)), None)
                if not lead_ids & set(hypothesis.related_anomalies):
                    reason = "alternative answer does not cite the leading correlation group"
                elif twin is not None:
                    reason = f"alternative answer repeats '{twin.title}' (near-duplicate title)"
                else:
                    added.append(hypothesis)
                    record.accepted.append(hypothesis.title)
                    continue
                self.last_rejections.append(HypothesisRejection(hypothesis.title, reason, [], hypothesis.related_anomalies))
        return added

    def _alternative_prompt(
        self,
        lead: list[AnomalyEvent],
        existing: list[Hypothesis],
        physical: list[AnomalyEvent],
        kb_hits: list[SearchResult],
        reasons: list[str],
        mission_context: str,
        history: list[HistoricalCase],
    ) -> str:
        """Render the prompt asking for a competing root cause for the leading group."""
        labels = anomaly_labels(physical)
        return ALTERNATIVE_PROMPT_TEMPLATE.format(
            mission_context=mission_context or "Not provided.",
            phases=self._phase_lines(),
            group=json.dumps([a.to_prompt_dict(labels) for a in lead], indent=1, default=str),
            existing="\n".join(f"- {h.title}: {h.description[:240]}" for h in existing) or "- none",
            kb_matches="\n".join(
                f"- [{h.metadata.get('kind', '?')}] {h.metadata.get('title', h.id)} (similarity {h.score:.2f}) - {h.text[:300]}"
                for h in kb_hits
            ) or "- none",
            historical_context=format_historical_context(history),
            reasons="\n".join(f"- {r}" for r in reasons) or "- none",
            count=len(existing),
            label=lead[0].label,
        )

    @staticmethod
    def _select(
        hypotheses: list[Hypothesis],
        physical: list[AnomalyEvent],
        slots: int,
        must_keep: Sequence[Hypothesis] = (),
    ) -> list[Hypothesis]:
        """Keep ``must_keep`` (competing root causes) and the best hypothesis for every covered
        anomaly, then fill the remaining slots by confidence."""
        ordered = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)
        chosen: list[Hypothesis] = [h for h in must_keep if h in hypotheses]
        for anomaly in physical:
            best = next((h for h in ordered if anomaly.id in h.related_anomalies), None)
            if best is not None and best not in chosen:
                chosen.append(best)
        for hypothesis in ordered:
            if len(chosen) >= slots:
                break
            if hypothesis not in chosen:
                chosen.append(hypothesis)
        return chosen

    # ------------------------------------------------------------------ #
    def build_prompt(
        self,
        anomalies: list[AnomalyEvent],
        actionable: list[AnomalyEvent],
        kb_matches: dict[str, list[SearchResult]],
        analysis: TelemetryAnalysis | None,
        mission_context: str,
        historical: list[HistoricalCase] | None = None,
    ) -> str:
        """Render the hypothesis prompt (exposed for inspection and testing)."""
        labels = anomaly_labels(anomalies)
        lead = self.category_preference(actionable).leading_group if actionable else []
        vehicle = getattr(self, "_vehicle_sensors", None) or (
            [s.sensor_id for s in analysis.sensors] if analysis is not None else sorted({a.sensor_id for a in anomalies})
        )
        sensor_lines, coupling_lines = self.config.evidence_rules.prompt_lines(vehicle)
        groups = TelemetryAnalyst.correlation_groups(actionable)
        group_lines = [f"- Group {i}: " + " -> ".join(a.label for a in group) for i, group in enumerate(groups, 1)]
        kb_lines: list[str] = []
        for anomaly_id, hits in kb_matches.items():
            for hit in hits:
                kb_lines.append(
                    f"- for {labels.get(anomaly_id, anomaly_id)}: [{hit.metadata.get('kind', '?')}] {hit.metadata.get('title', hit.id)} "
                    f"(similarity {hit.score:.2f}) - {hit.text[:300]}"
                )
        expected = [a for a in anomalies if a.expected]
        faults = [a for a in anomalies if not a.expected and a.classification.get("fault_class") == SENSOR_FAULT]
        context = mission_context or "Not provided."
        if analysis is not None:
            sensors = ", ".join(f"{s.sensor_id} [{s.unit}] @ {s.sample_rate_hz:g} Hz" for s in analysis.sensors)
            context += f"\nRecord: {analysis.duration_s:.1f}s, {analysis.total_readings} samples. Sensors: {sensors}."
        return HYPOTHESIS_PROMPT_TEMPLATE.format(
            mission_context=context,
            quality_notes="\n".join(f"- {n}" for n in (analysis.data_quality_notes if analysis else [])) or "- none",
            anomalies=json.dumps([a.to_prompt_dict(labels) for a in actionable], indent=1, default=str),
            window=self.config.correlation_window_seconds,
            sensor_faults="\n".join(
                f"- {a.label}: {a.classification.get('fault_class_reason', 'instrumentation glitch')}" for a in faults
            ) or "- none",
            groups="\n".join(group_lines) or "- none",
            phases=self._phase_lines(),
            expected="\n".join(f"- {a.label}: {a.description}" for a in expected) or "- none",
            kb_matches="\n".join(kb_lines) or "- none",
            historical_context=format_historical_context(historical or []),
            leading_group=" -> ".join(a.label for a in lead) or "- none",
            subsystem_sensors=sensor_lines,
            couplings=coupling_lines,
            min_competing=self.config.min_competing_hypotheses,
            max_hypotheses=self.config.max_hypotheses,
        )

    def _llm_hypotheses(
        self,
        anomalies: list[AnomalyEvent],
        actionable: list[AnomalyEvent],
        kb_matches: dict[str, list[SearchResult]],
        analysis: TelemetryAnalysis | None,
        mission_context: str,
    ) -> list[Hypothesis]:
        """Run the main hypothesis prompt and return the hypotheses that pass validation."""
        prompt = self.build_prompt(
            anomalies, actionable, kb_matches, analysis, mission_context, self.last_historical_cases
        )
        return self._ask_and_validate(prompt, anomalies, actionable)

    def _ask_and_validate(
        self, prompt: str, anomalies: list[AnomalyEvent], actionable: list[AnomalyEvent]
    ) -> list[Hypothesis]:
        """Send ``prompt`` to the LLM and validate every returned hypothesis."""
        assert self.router is not None
        prefer = "local" if self.config.provider_preference == "local" else "cloud"
        document, response = self.router.complete_json(SYSTEM_PROMPT, prompt, prefer=prefer, temperature=self.config.temperature)
        items = document.get("hypotheses") if isinstance(document, dict) else document
        if not isinstance(items, list) or not items:
            raise ValueError("LLM response contained no hypotheses")
        source = f"{response.provider}:{response.model}"
        group_of = {
            a.id: index for index, group in enumerate(TelemetryAnalyst.correlation_groups(actionable)) for a in group
        }
        accepted: list[Hypothesis] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            hypothesis = self._validate(item, actionable, source, group_of, anomalies)
            if hypothesis is not None:
                accepted.append(hypothesis)
        if accepted:
            self.last_source = source
        return accepted

    def _coverage_retries(
        self,
        anomalies: list[AnomalyEvent],
        physical: list[AnomalyEvent],
        accepted: list[Hypothesis],
        kb_matches: dict[str, list[SearchResult]],
        mission_context: str,
    ) -> list[Hypothesis]:
        """Re-ask the LLM about each physical anomaly that no accepted hypothesis cites.

        Each retry prompt contains only the unexplained anomaly (plus context) and the reasons
        earlier hypotheses were rejected or trimmed. Only hypotheses that cite the target are kept.
        """
        added: list[Hypothesis] = []
        for target in physical:
            for attempt in range(1, self.config.coverage_retries + 1):
                if any(target.id in h.related_anomalies for h in [*accepted, *added] if h.hypothesis_type == "root_cause"):
                    break
                reasons = self._reasons_for(target)
                record = CoverageRetry(target.id, target.label, attempt, reasons)
                self.last_retries.append(record)
                logger.info("Coverage retry %d for unexplained anomaly %s", attempt, target.label)
                history = self._retrieve_history([target], min(2, self.config.historical_cases))
                record.historical_cases = [c.title for c in history]
                prompt = self._coverage_prompt(
                    target, physical, kb_matches.get(target.id, []), reasons, mission_context, history
                )
                try:
                    candidates = self._ask_and_validate(prompt, anomalies, physical)
                except (LLMUnavailableError, ValueError) as exc:
                    logger.warning("Coverage retry for %s failed: %s", target.label, exc)
                    continue
                for hypothesis in candidates:
                    if target.id in hypothesis.related_anomalies:
                        added.append(hypothesis)
                        record.accepted.append(hypothesis.title)
                    else:
                        self.last_rejections.append(HypothesisRejection(
                            hypothesis.title, f"coverage retry answer does not cite the target '{target.label}'",
                            [], hypothesis.related_anomalies,
                        ))
        return added

    def _reasons_for(self, target: AnomalyEvent, limit: int = 8) -> list[str]:
        """Rejection/trim reasons from earlier attempts, those about ``target`` first."""
        about: list[str] = []
        other: list[str] = []
        for rejection in self.last_rejections:
            line = f"'{rejection.title}' rejected: {rejection.reason}"
            (about if target.id in rejection.cited_ids or target.label in rejection.reason else other).append(line)
        for drop in self.last_drops:
            line = f"'{drop.title}': citation '{drop.reference}' dropped - {drop.reason}"
            (about if drop.anomaly_id == target.id else other).append(line)
        return (about + other)[:limit]

    def _coverage_prompt(
        self,
        target: AnomalyEvent,
        physical: list[AnomalyEvent],
        kb_hits: list[SearchResult],
        reasons: list[str],
        mission_context: str,
        history: list[HistoricalCase] | None = None,
    ) -> str:
        """Render the focused prompt for one unexplained anomaly."""
        labels = anomaly_labels(physical)
        return COVERAGE_PROMPT_TEMPLATE.format(
            mission_context=mission_context or "Not provided.",
            phases=self._phase_lines(),
            anomaly=json.dumps(target.to_prompt_dict(labels), indent=1, default=str),
            others="\n".join(f"- {a.label}" for a in physical if a.id != target.id) or "- none",
            kb_matches="\n".join(
                f"- [{h.metadata.get('kind', '?')}] {h.metadata.get('title', h.id)} (similarity {h.score:.2f}) - {h.text[:300]}"
                for h in kb_hits
            ) or "- none",
            reasons="\n".join(f"- {r}" for r in reasons) or "- no hypothesis cited this anomaly",
            historical_context=format_historical_context(history or []),
            label=target.label,
        )

    def _phase_lines(self) -> str:
        return "\n".join(
            f"- {p.name}: T+{p.start:g}s to " + ("end of record" if p.end >= 1e8 else f"T+{p.end:g}s")
            for p in self.config.flight_phases
        ) or "- not defined"

    def _validate(
        self,
        item: dict[str, Any],
        actionable: list[AnomalyEvent],
        source: str,
        group_of: dict[str, int] | None = None,
        all_anomalies: list[AnomalyEvent] | None = None,
    ) -> Hypothesis | None:
        """Validate one LLM hypothesis; return ``None`` (and record why) if it is rejected.

        Citation-level problems drop just that citation (recorded in ``last_drops``):
        unknown anomalies, expected transients, classified sensor faults, and anomalies whose
        sensor/time the hypothesis text does not support. The hypothesis is rejected when:

        * no valid citations remain;
        * its title names a sensor whose anomaly it does not cite (topic check);
        * it names a flight phase in which none of its cited anomalies occurred;
        * it cites anomalies from different correlated groups without quoting each one's time.

        A hypothesis linking anomalies more than 30 s apart without explaining the delay is kept
        but flagged: confidence halved (capped at 0.3) and a counter-evidence note added.
        """
        group_of = group_of or {}
        all_anomalies = all_anomalies or actionable
        expected = [a for a in all_anomalies if a.expected]
        faults = [a for a in all_anomalies if not a.expected and a.classification.get("fault_class") == SENSOR_FAULT]
        title = str(item.get("title") or "Untitled hypothesis")
        cited = _as_list(item.get("related_anomalies"))
        text = " ".join([title, str(item.get("description", "")), *_as_list(item.get("supporting_evidence"))])
        dropped: list[DroppedReference] = []

        def reject(reason: str, cited_ids: list[str] | None = None) -> None:
            self.last_rejections.append(HypothesisRejection(title, reason, cited, cited_ids or []))
            logger.warning("Rejected hypothesis '%s': %s", title, reason)

        def drop(reference: str, reason: str, anomaly: AnomalyEvent | None = None) -> None:
            dropped.append(DroppedReference(title, reference, reason, anomaly.id if anomaly else None))

        resolved: list[AnomalyEvent] = []
        for reference in cited:
            event = resolve_anomaly_reference(reference, actionable)
            if event is None:
                if (transient := resolve_anomaly_reference(reference, expected)) is not None:
                    drop(reference, f"'{transient.label}' is an expected transient (planned event)", transient)
                elif (fault := resolve_anomaly_reference(reference, faults)) is not None:
                    drop(reference, f"'{fault.label}' is an instrumentation glitch (explained separately)", fault)
                else:
                    drop(reference, "not a detected anomaly")
                continue
            if event in resolved:
                continue
            problem = citation_problem(event, text)
            if problem:
                drop(reference, f"cites '{event.label}' but {problem}", event)
                continue
            resolved.append(event)

        if not resolved:
            details = "; ".join(d.reason for d in dropped) or "cites no anomalies"
            reject(f"no valid citations remain ({details})" if dropped else "cites no anomalies",
                   [d.anomaly_id for d in dropped if d.anomaly_id])
            return None
        cited_ids = [e.id for e in resolved]
        hypothesis_type = normalise_type(item.get("type"))
        sensors = {a.sensor_id for a in all_anomalies if not a.expected}
        for problem in (
            topic_problem(title, resolved, sensors),
            phase_problem(resolved, text, self.config.flight_phases),
        ):
            if problem:
                reject(problem, cited_ids)
                return None
        rules = self.config.evidence_rules
        vehicle = getattr(self, "_vehicle_sensors", None) or sorted({a.sensor_id for a in all_anomalies})
        unobservable: list[str] = []
        if hypothesis_type != "insufficient_evidence":
            # Rule 1: named subsystems need evidence (the causal chain = title + description).
            problem, unobservable = rules.subsystem_check(f"{title}. {item.get('description', '')}", resolved, vehicle)
            if problem:
                reject(problem, cited_ids)
                return None
        if hypothesis_type == "root_cause":
            # Rule 2: cross-subsystem links need a listed physical coupling, whatever the time gap.
            problem = rules.coupling_check(resolved)
            if problem:
                reject(problem, cited_ids)
                return None
        if hypothesis_type == "independent_events" and len(resolved) < 2:
            reject("'independent_events' must cite at least two anomalies", cited_ids)
            return None
        cited_groups = {group_of.get(e.id, -1) for e in resolved}
        if len(cited_groups) > 1 and hypothesis_type == "root_cause":
            unquoted = [e for e in resolved if not quotes_time_of(e, text)]
            if unquoted:
                reject(
                    f"links anomalies from {len(cited_groups)} different correlated groups without quoting the time of "
                    + "; ".join(f"'{e.label}'" for e in unquoted),
                    cited_ids,
                )
                return None
        try:
            hypothesis = Hypothesis(
                title=title,
                description=str(item.get("description", "")),
                confidence=item.get("confidence", 0.0),
                subsystem=str(item.get("subsystem", "unknown")),
                related_anomalies=cited_ids,
                supporting_evidence=_as_list(item.get("supporting_evidence")),
                counter_evidence=_as_list(item.get("counter_evidence")),
                recommended_actions=_as_list(item.get("recommended_actions")),
                distinguishing_evidence=_as_list(item.get("distinguishing_evidence")),
                hypothesis_type=hypothesis_type,
                source=source,
            )
        except ValidationError as exc:
            reject(f"malformed fields ({exc.errors()[0].get('msg', exc)})", cited_ids)
            return None
        for item_drop in dropped:
            self.last_drops.append(item_drop)
            logger.info("Hypothesis '%s': dropped citation '%s' (%s)", title, item_drop.reference, item_drop.reason)
        if unobservable:
            label = f"{UNOBSERVABLE_LABEL}: {', '.join(unobservable)}"
            adjusted = round(min(hypothesis.confidence, rules.unobservable_cap), 2)
            reason = f"names {', '.join(unobservable)} but the vehicle has no sensor for it; " \
                     f"confidence capped {hypothesis.confidence:.0%} -> {adjusted:.0%}"
            self.last_flags.append(HypothesisFlag(title, reason, hypothesis.confidence, adjusted, cited_ids))
            logger.info("Flagged hypothesis '%s': %s", title, reason)
            hypothesis = hypothesis.model_copy(update={"confidence": adjusted, "labels": [*hypothesis.labels, label]})
        gap = max_time_gap(resolved)
        if gap > _MAX_UNEXPLAINED_GAP_S and not _GAP_EXPLANATION.search(text):
            adjusted = round(min(hypothesis.confidence * _GAP_CONFIDENCE_FACTOR, _GAP_CONFIDENCE_CAP), 2)
            reason = (
                f"cites anomalies {gap:.0f}s apart without explaining the time gap; "
                f"confidence reduced {hypothesis.confidence:.0%} -> {adjusted:.0%}"
            )
            self.last_flags.append(HypothesisFlag(title, reason, hypothesis.confidence, adjusted, cited_ids))
            logger.warning("Flagged hypothesis '%s': %s", title, reason)
            hypothesis = hypothesis.model_copy(
                update={
                    "confidence": adjusted,
                    "counter_evidence": [*hypothesis.counter_evidence, f"Validator: {reason}."],
                }
            )
        return hypothesis

    def _rule_based(self, actionable: list[AnomalyEvent], kb_matches: dict[str, list[SearchResult]]) -> list[Hypothesis]:
        """Deterministic hypotheses: one per correlation group, driven by knowledge-base matches."""
        hypotheses: list[Hypothesis] = []
        for group in TelemetryAnalyst.correlation_groups(actionable):
            lead = group[0]
            ids = [a.id for a in group]
            categories = {a.classification.get("category") for a in group}
            if len(group) == 1 and categories == {"sensor_fault"}:
                hypotheses.append(self._sensor_fault_hypothesis(lead))
                continue

            # Aggregate KB similarity across the group; the initiating (earliest) anomaly weighs most,
            # because downstream deviations are more likely consequences than causes.
            scores: dict[str, tuple[float, SearchResult]] = {}
            weights = [1.5] + [1.0] * (len(group) - 1)
            for anomaly, weight in zip(group, weights):
                for hit in kb_matches.get(anomaly.id, []):
                    if hit.metadata.get("kind") != "failure_mode":
                        continue
                    previous = scores.get(hit.id, (0.0, hit))[0]
                    scores[hit.id] = (previous + weight * hit.score, hit)
            severity_weight = max(a.severity.rank for a in group) / Severity.CRITICAL.rank
            timeline = "; ".join(a.label for a in group)
            ranked = sorted(scores.values(), key=lambda pair: pair[0], reverse=True)[:2]
            # Keep an alternative only if it matches nearly as well as the best candidate.
            ranked = [pair for pair in ranked if pair[0] >= 0.75 * ranked[0][0]] if ranked else ranked
            if not ranked:
                hypotheses.append(
                    Hypothesis(
                        title=f"Unexplained {lead.classification.get('subsystem', lead.sensor_id)} anomaly",
                        description=f"No matching failure mode in the knowledge base. Timeline: {timeline}.",
                        confidence=0.2 * severity_weight,
                        subsystem=str(lead.classification.get("subsystem", "unknown")),
                        related_anomalies=ids,
                        supporting_evidence=[a.description for a in group],
                        counter_evidence=["No historical precedent found for this signature."],
                        recommended_actions=["Escalate to subsystem engineers for manual review of raw telemetry."],
                        source="rule-based",
                    )
                )
                continue
            for rank, (total, hit) in enumerate(ranked):
                mode = self.kb.failure_mode(hit.id)
                similarity = total / sum(weights)
                multi_sensor_bonus = 0.15 if len(group) > 1 else 0.0
                confidence = min(0.9, (0.35 + similarity * 0.6 + multi_sensor_bonus) * severity_weight) * (1.0 if rank == 0 else 0.7)
                supporting = [f"{a.label}: {a.description}" for a in group]
                if len(group) > 1:
                    supporting.append(f"{len(group)} sensors deviated within the correlation window, first: {lead.sensor_id}.")
                hypotheses.append(
                    Hypothesis(
                        title=mode.title if mode else str(hit.metadata.get("title", hit.id)),
                        description=(
                            f"{mode.mechanism if mode else hit.text} Observed sequence: {timeline}. "
                            f"Known signature: {mode.signature if mode else 'n/a'}."
                        ),
                        confidence=round(confidence, 2),
                        subsystem=mode.subsystem if mode else str(hit.metadata.get("subsystem", "unknown")),
                        related_anomalies=ids,
                        supporting_evidence=supporting,
                        counter_evidence=[
                            "Rule-based match on telemetry signature only; not reviewed by an LLM or engineer.",
                            "Confirm with additional channels (pressurant, accelerometers, currents) not present in this dataset.",
                        ],
                        recommended_actions=list(mode.recommended_actions) if mode else KnowledgeBase.actions_from_metadata(hit.metadata),
                        source="rule-based",
                    )
                )
        return hypotheses

    @staticmethod
    def _sensor_fault_hypothesis(anomaly: AnomalyEvent) -> Hypothesis:
        # Confident about the mechanism, but scaled by severity so glitches never outrank real events.
        severity_weight = anomaly.severity.rank / Severity.CRITICAL.rank
        return Hypothesis(
            title=f"Instrumentation glitch on {anomaly.sensor_id.replace('_', ' ')}",
            description=(
                f"The {anomaly.label} is a "
                f"{anomaly.classification.get('fault_class_reason', 'very short spike with no corroborating deviation')}. "
                "This signature is consistent with a data-acquisition bit error, EMI pickup or an intermittent "
                "transducer connection rather than a physical event."
            ),
            confidence=round(0.7 * severity_weight, 2),
            subsystem="instrumentation",
            related_anomalies=[anomaly.id],
            supporting_evidence=[
                anomaly.description,
                "No correlated anomalies on other sensors within the correlation window.",
            ],
            counter_evidence=["A real but very brief physical transient cannot be excluded without redundant sensors."],
            recommended_actions=[
                f"Check DAQ CRC/parity error counters for {anomaly.sensor_id} at the event time.",
                "Compare against redundant or neighbouring transducers.",
                "Flag the channel for post-flight calibration and connector inspection.",
            ],
            source="sensor-fault-classifier" if anomaly.classification.get("fault_class") == SENSOR_FAULT else "rule-based",
        )


def _as_list(value: Any) -> list[str]:
    """Coerce an LLM field into a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)]
