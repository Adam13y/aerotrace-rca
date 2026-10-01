"""Hypothesis generation agent.

Builds an engineering context (anomalies, cross-sensor correlation groups, flight
phases and knowledge-base matches) and asks Claude - the cloud supervisor model - to
produce ranked root-cause hypotheses. Falls back to the local Ollama model and, if no
LLM is reachable, to deterministic knowledge-base-driven hypotheses.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from src.agents.sensor_fault_classifier import SENSOR_FAULT
from src.agents.telemetry_analyst import TelemetryAnalyst
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

## Expected transients (planned events, for context only)
{expected}

## Knowledge-base matches (prior failure modes and past anomalies)
{kb_matches}

Generate up to {max_hypotheses} root-cause hypotheses covering every correlation group, ranked by confidence.

IMPORTANT: When building causal chains, anomalies must be ordered by time. The FIRST anomaly in time is the most likely root cause. Do not connect anomalies more than 30 seconds apart unless there is a clear physical mechanism connecting them.

Citation rules (hypotheses that break them are rejected automatically):
- In "related_anomalies", copy anomaly labels EXACTLY as written in the "anomaly" field above.
- Cite only anomalies that your description and supporting evidence actually discuss, naming their sensors.
- Cite anomalies from different correlation groups only if you quote each one's time (e.g. "at T+142.4s").
- If you link anomalies more than 30 seconds apart, state the delay and the mechanism that explains it.

Respond with JSON exactly in this shape:
{{"hypotheses": [{{
  "title": "<short name of the root cause>",
  "description": "<mechanism, causal chain and how it explains the timeline>",
  "confidence": <0.0-1.0>,
  "subsystem": "<subsystem>",
  "related_anomalies": ["<anomaly label, copied exactly>", ...],
  "supporting_evidence": ["<evidence from the telemetry>", ...],
  "counter_evidence": ["<evidence against, or what would falsify it>", ...],
  "recommended_actions": ["<concrete next step>", ...]
}}]}}"""


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
    temperature: float = 0.2
    correlation_window_seconds: float = 5.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, correlation_window: float = 5.0) -> HypothesisConfig:
        """Build from the ``hypothesis_generator`` section of ``agents.yaml``."""
        data = data or {}
        return cls(
            provider_preference=str(data.get("provider_preference", "cloud")),
            max_hypotheses=int(data.get("max_hypotheses", 6)),
            knowledge_base_matches=int(data.get("knowledge_base_matches", 3)),
            temperature=float(data.get("temperature", 0.2)),
            correlation_window_seconds=correlation_window,
        )


class HypothesisGenerator:
    """Generate ranked root-cause hypotheses for anomaly events."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        router: LLMRouter | None = None,
        config: HypothesisConfig | None = None,
        use_llm: bool = True,
    ) -> None:
        """Create the agent.

        Args:
            knowledge_base: Knowledge base used for similarity search.
            router: LLM router (Claude preferred); ``None`` means rule-based only.
            config: Agent configuration.
            use_llm: Disable LLM use even if a router is supplied.
        """
        self.kb = knowledge_base
        self.router = router
        self.config = config or HypothesisConfig()
        self.use_llm = use_llm and router is not None
        self.last_source = "none"
        self.last_rejections: list[HypothesisRejection] = []
        self.last_flags: list[HypothesisFlag] = []

    def generate(
        self,
        anomalies: list[AnomalyEvent],
        analysis: TelemetryAnalysis | None = None,
        mission_context: str = "",
    ) -> list[Hypothesis]:
        """Produce hypotheses ranked by confidence (highest first).

        Args:
            anomalies: Classified, contextualised anomaly events.
            analysis: Optional dataset overview for context.
            mission_context: Free-text description of the vehicle and mission.
        """
        actionable = sorted((a for a in anomalies if not a.expected), key=lambda a: a.timestamp)
        self.last_rejections = []
        self.last_flags = []
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
            kb_matches = {a.id: self.kb.context_for(a, k=self.config.knowledge_base_matches) for a in physical}
            if self.use_llm and self.router is not None:
                try:
                    hypotheses = self._llm_hypotheses(anomalies, physical, kb_matches, analysis, mission_context)
                except (LLMUnavailableError, ValidationError, ValueError) as exc:
                    logger.warning("LLM hypothesis generation failed, using knowledge-base fallback: %s", exc)
            if not hypotheses:
                hypotheses = self._rule_based(physical, kb_matches)
                self.last_source = "rule-based"

        # Always keep the instrumentation-glitch hypotheses; fill the remaining slots by confidence.
        slots = max(1, self.config.max_hypotheses - len(glitches))
        kept = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)[:slots]
        ranked = sorted(kept + glitches, key=lambda h: h.confidence, reverse=True)
        logger.info("Generated %d hypotheses via %s (+%d instrumentation glitch)", len(kept), self.last_source, len(glitches))
        return ranked

    # ------------------------------------------------------------------ #
    def build_prompt(
        self,
        anomalies: list[AnomalyEvent],
        actionable: list[AnomalyEvent],
        kb_matches: dict[str, list[SearchResult]],
        analysis: TelemetryAnalysis | None,
        mission_context: str,
    ) -> str:
        """Render the hypothesis prompt (exposed for inspection and testing)."""
        labels = anomaly_labels(anomalies)
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
            expected="\n".join(f"- {a.label}: {a.description}" for a in expected) or "- none",
            kb_matches="\n".join(kb_lines) or "- none",
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
        assert self.router is not None
        prompt = self.build_prompt(anomalies, actionable, kb_matches, analysis, mission_context)
        prefer = "local" if self.config.provider_preference == "local" else "cloud"
        document, response = self.router.complete_json(SYSTEM_PROMPT, prompt, prefer=prefer, temperature=self.config.temperature)
        items = document.get("hypotheses") if isinstance(document, dict) else document
        if not isinstance(items, list) or not items:
            raise ValueError("LLM response contained no hypotheses")
        source = f"{response.provider}:{response.model}"
        group_of = {
            a.id: index for index, group in enumerate(TelemetryAnalyst.correlation_groups(actionable)) for a in group
        }
        faults = [a for a in anomalies if not a.expected and a.classification.get("fault_class") == SENSOR_FAULT]
        hypotheses: list[Hypothesis] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            hypothesis = self._validate(item, actionable, source, group_of, faults)
            if hypothesis is not None:
                hypotheses.append(hypothesis)
        if not hypotheses:
            raise ValueError(f"All {len(items)} LLM hypotheses failed validation")
        self.last_source = source
        return hypotheses

    def _validate(
        self,
        item: dict[str, Any],
        actionable: list[AnomalyEvent],
        source: str,
        group_of: dict[str, int] | None = None,
        faults: list[AnomalyEvent] | None = None,
    ) -> Hypothesis | None:
        """Validate one LLM hypothesis; record a rejection and return ``None`` if it fails.

        A hypothesis is rejected when it:

        * cites no anomalies, or an anomaly that was not detected (or is a classified sensor fault);
        * cites an anomaly whose sensor/time its own text does not support;
        * cites anomalies from different correlated groups without quoting each one's time.

        A hypothesis that links anomalies more than 30 s apart without explaining the delay is
        accepted but flagged: its confidence is halved (capped at 0.3) and a counter-evidence note added.
        """
        group_of = group_of or {}
        title = str(item.get("title") or "Untitled hypothesis")
        cited = _as_list(item.get("related_anomalies"))

        def reject(reason: str, cited_ids: list[str] | None = None) -> None:
            self.last_rejections.append(HypothesisRejection(title, reason, cited, cited_ids or []))
            logger.warning("Rejected hypothesis '%s': %s", title, reason)

        if not cited:
            reject("cites no anomalies")
            return None
        resolved: list[AnomalyEvent] = []
        for reference in cited:
            event = resolve_anomaly_reference(reference, actionable)
            if event is None:
                fault = resolve_anomaly_reference(reference, faults or [])
                if fault is not None:
                    reject(
                        f"cites '{fault.label}', which the sensor-fault classifier identified as an instrumentation "
                        "glitch (explained separately)",
                        [e.id for e in resolved] + [fault.id],
                    )
                else:
                    reject(f"cites '{reference}', which is not a detected anomaly", [e.id for e in resolved])
                return None
            if event not in resolved:
                resolved.append(event)
        text = " ".join([title, str(item.get("description", "")), *_as_list(item.get("supporting_evidence"))])
        for event in resolved:
            problem = citation_problem(event, text)
            if problem:
                reject(f"cites '{event.label}' but {problem}", [e.id for e in resolved])
                return None
        cited_ids = [e.id for e in resolved]
        cited_groups = {group_of.get(e.id, -1) for e in resolved}
        if len(cited_groups) > 1:
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
                related_anomalies=[e.id for e in resolved],
                supporting_evidence=_as_list(item.get("supporting_evidence")),
                counter_evidence=_as_list(item.get("counter_evidence")),
                recommended_actions=_as_list(item.get("recommended_actions")),
                source=source,
            )
        except ValidationError as exc:
            reject(f"malformed fields ({exc.errors()[0].get('msg', exc)})", cited_ids)
            return None
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
