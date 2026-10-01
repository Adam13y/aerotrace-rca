"""Physical-evidence rules applied to LLM hypotheses.

Rule 1 - subsystem evidence: every subsystem a hypothesis names as part of its cause (title or
causal chain) must be backed by at least one cited anomaly from a sensor that observes that
subsystem. If the vehicle has *no* sensor for the subsystem, the claim cannot be checked: its
confidence is capped and it is labelled "unobservable - no sensor coverage". Otherwise the
hypothesis is rejected.

Rule 2 - cross-subsystem coupling: when a hypothesis links anomalies from different subsystems,
each later anomaly's subsystem must be directly reachable from an earlier cited anomaly's
subsystem through the coupling table. Chains of couplings are deliberately not followed: almost
every subsystem reaches every other within a few hops, which would make the rule meaningless.
Sensors with no subsystem mapping have no couplings at all.

Subsystem names are detected with a configurable term list (longest phrase first, sensor names
stripped from the text beforehand). This is a vocabulary heuristic, not semantic understanding.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.memory.historical_rag import DEFAULT_SENSOR_CATEGORIES, DEFAULT_SENSOR_CATEGORY_PATTERNS, sensor_category
from src.models.telemetry_schema import AnomalyEvent, sensor_display_name

#: Phrases that name a subsystem as a cause. A trailing "*" matches any word ending (stem).
DEFAULT_SUBSYSTEM_TERMS: dict[str, list[str]] = {
    "propulsion": [
        "turbopump", "turbo pump", "pump cavitation", "cavitat*", "combustion", "injector", "nozzle",
        "gas generator", "thrust", "engine", "chamber", "propellant feed", "feed line", "feed system",
        "inlet pressure", "npsh", "npsp", "hot gas", "hot-gas", "plume", "regenerative cooling",
        "mixture ratio", "igniter", "ignition", "preburner",
    ],
    "pressurization": [
        "pressuris*", "pressuriz*", "pressurant", "regulator", "helium", "copv", "ullage",
        "relief valve", "vent valve", "overpressur*", "over-pressur*",
    ],
    "thermal": [
        "thermal", "overheat*", "heat soak", "heating", "insulation", "thermal protection", "tps",
        "cooling failure", "heat leak",
    ],
    "electrical": [
        "electrical", "avionics bus", "power bus", "voltage", "under-voltage", "undervoltage", "brownout",
        "battery", "batteries", "cell", "short circuit", "short-circuit", "power distribution", "power supply",
        "load transient", "relay", "wiring", "harness", "bus", "emi",
    ],
    "avionics": [
        "avionics", "imu", "inertial", "gps", "flight computer", "guidance", "navigation", "software",
        "telemetry dropout", "sensor dropout", "star tracker", "data bus",
    ],
    "structural": [
        "structural", "strut", "crack*", "fatigue", "buckl*", "weld*", "fastener", "delaminat*", "fractur*",
    ],
    "aerodynamic": [
        "aerodynamic*", "max-q", "max q", "buffet*", "flutter", "wind shear", "angle of attack", "dynamic pressure",
    ],
    "separation": [
        "separation", "pyrotechnic*", "pyro", "fairing", "frangible", "staging",
    ],
}
#: Location phrases that contain subsystem words but do not name a cause.
DEFAULT_NEUTRAL_PHRASES: list[str] = ["engine bay", "avionics bay", "engine compartment", "equipment bay"]
#: Sensors that observe more than one subsystem (all others observe only their primary category).
DEFAULT_SENSOR_EVIDENCE: dict[str, list[str]] = {
    "lox_tank_pressure": ["propulsion", "pressurization"],
    "fuel_tank_pressure": ["propulsion", "pressurization"],
}
#: Directed, physically plausible subsystem -> subsystem causal links.
DEFAULT_COUPLINGS: dict[str, list[str]] = {
    "pressurization": ["propulsion", "structural", "thermal"],
    "propulsion": ["thermal", "structural", "aerodynamic", "pressurization"],
    "thermal": ["structural", "electrical", "avionics", "pressurization"],
    "electrical": ["avionics", "thermal", "propulsion"],
    "avionics": ["propulsion", "aerodynamic", "separation", "structural"],
    "structural": ["pressurization", "propulsion", "thermal", "aerodynamic", "electrical", "separation"],
    "aerodynamic": ["structural", "thermal"],
    "separation": ["structural", "propulsion", "avionics"],
}
HYPOTHESIS_TYPES = ("root_cause", "independent_events", "insufficient_evidence")
UNOBSERVABLE_LABEL = "unobservable - no sensor coverage"


@dataclass(slots=True)
class EvidenceRules:
    """Configuration and checks for the subsystem-evidence and coupling rules."""

    subsystem_terms: dict[str, list[str]] = field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_SUBSYSTEM_TERMS.items()})
    neutral_phrases: list[str] = field(default_factory=lambda: list(DEFAULT_NEUTRAL_PHRASES))
    sensor_evidence: dict[str, list[str]] = field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_SENSOR_EVIDENCE.items()})
    couplings: dict[str, list[str]] = field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_COUPLINGS.items()})
    unobservable_cap: float = 0.3
    sensor_categories: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SENSOR_CATEGORIES))
    sensor_category_patterns: list[tuple[str, str]] = field(default_factory=lambda: list(DEFAULT_SENSOR_CATEGORY_PATTERNS))

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        sensor_categories: dict[str, str] | None = None,
        sensor_category_patterns: list[tuple[str, str]] | None = None,
    ) -> EvidenceRules:
        """Build from the ``hypothesis_generator.evidence_rules`` section of ``agents.yaml``."""
        data = data or {}
        rules = cls()
        if data.get("subsystem_terms"):
            rules.subsystem_terms = {str(k).lower(): [str(t).lower() for t in v] for k, v in data["subsystem_terms"].items()}
        if data.get("neutral_phrases"):
            rules.neutral_phrases = [str(p).lower() for p in data["neutral_phrases"]]
        if data.get("sensor_evidence"):
            rules.sensor_evidence = {str(k).lower(): [str(c).lower() for c in v] for k, v in data["sensor_evidence"].items()}
        if data.get("couplings"):
            rules.couplings = {str(k).lower(): [str(c).lower() for c in (v or [])] for k, v in data["couplings"].items()}
        rules.unobservable_cap = float(data.get("unobservable_confidence_cap", rules.unobservable_cap))
        if sensor_categories:
            rules.sensor_categories = dict(sensor_categories)
        if sensor_category_patterns:
            rules.sensor_category_patterns = list(sensor_category_patterns)
        return rules

    # ------------------------------------------------------------------ sensors
    def primary_category(self, sensor_id: str) -> str:
        """Subsystem a sensor belongs to (``unmapped:<sensor>`` when unknown: no couplings)."""
        category = sensor_category(sensor_id, self.sensor_categories, self.sensor_category_patterns)
        return category or f"unmapped:{sensor_id.lower()}"

    def evidence_categories(self, sensor_id: str) -> set[str]:
        """Subsystems a sensor can provide evidence for."""
        explicit = self.sensor_evidence.get(sensor_id.lower())
        if explicit:
            return set(explicit)
        category = sensor_category(sensor_id, self.sensor_categories, self.sensor_category_patterns)
        return {category} if category else set()

    # ------------------------------------------------------------------ rule 1
    def named_subsystems(self, text: str, sensor_ids: Iterable[str]) -> set[str]:
        """Subsystems named in ``text`` (sensor names stripped first, longest phrase first)."""
        lowered = " " + text.lower().replace("_", " ") + " "
        names: set[str] = set()
        for sensor in sensor_ids:
            names.add(sensor.lower().replace("_", " "))
            names.add(sensor_display_name(sensor).lower())
        for name in sorted(names, key=len, reverse=True):
            lowered = re.sub(rf"\b{re.escape(name)}s?\b", " ", lowered)
        phrases = [(term, category) for category, terms in self.subsystem_terms.items() for term in terms]
        phrases += [(phrase, None) for phrase in self.neutral_phrases]
        found: set[str] = set()
        for term, category in sorted(phrases, key=lambda pair: len(pair[0].rstrip("*")), reverse=True):
            stem = term.rstrip("*")
            pattern = rf"\b{re.escape(stem)}[a-z-]*" if term.endswith("*") else rf"\b{re.escape(stem)}s?\b"
            if re.search(pattern, lowered):
                if category:
                    found.add(category)
                lowered = re.sub(pattern, " ", lowered)
        return found

    def subsystem_check(
        self, text: str, cited: Sequence[AnomalyEvent], vehicle_sensors: Iterable[str]
    ) -> tuple[str | None, list[str]]:
        """Apply rule 1.

        Returns:
            ``(rejection_reason, unobservable_subsystems)``; the reason is ``None`` when the hypothesis passes.
        """
        vehicle_sensors = list(vehicle_sensors)
        named = self.named_subsystems(text, vehicle_sensors)
        cited_evidence = set().union(*(self.evidence_categories(a.sensor_id) for a in cited)) if cited else set()
        observable = set().union(*(self.evidence_categories(s) for s in vehicle_sensors)) if vehicle_sensors else set()
        unobservable: list[str] = []
        for subsystem in sorted(named):
            if subsystem in cited_evidence:
                continue
            if subsystem in observable:
                sensors = sorted(s for s in vehicle_sensors if subsystem in self.evidence_categories(s))
                return (
                    f"names {subsystem} as part of the cause but cites no anomaly from its {subsystem} sensors "
                    f"({', '.join(sensors)}), which show no supporting deviation",
                    [],
                )
            unobservable.append(subsystem)
        return None, unobservable

    # ------------------------------------------------------------------ rule 2
    def coupling_check(self, cited: Sequence[AnomalyEvent]) -> str | None:
        """Apply rule 2: every cross-subsystem link must follow a direct coupling (earlier -> later)."""
        ordered = sorted(cited, key=lambda a: a.timestamp)
        categories = [self.primary_category(a.sensor_id) for a in ordered]
        for i in range(1, len(ordered)):
            later = categories[i]
            earlier = categories[:i]
            if later in earlier:
                continue
            if any(later in self.couplings.get(source, []) for source in earlier):
                continue
            sources = sorted(set(earlier))
            return (
                f"links {', '.join(sources)} anomalies to '{ordered[i].label}' ({later}) but the coupling table lists "
                f"no physical path {' or '.join(f'{s} -> {later}' for s in sources)}"
            )
        return None

    # ------------------------------------------------------------------ prompt support
    def prompt_lines(self, vehicle_sensors: Iterable[str]) -> tuple[str, str]:
        """Render (sensors by subsystem, coupling table) for the hypothesis prompt."""
        by_subsystem: dict[str, list[str]] = {}
        for sensor in sorted(set(vehicle_sensors)):
            for category in sorted(self.evidence_categories(sensor)) or ["(no subsystem - cannot be causally linked)"]:
                by_subsystem.setdefault(category, []).append(sensor)
        sensors = "\n".join(f"- {cat}: {', '.join(names)}" for cat, names in sorted(by_subsystem.items())) or "- none"
        missing = sorted(set(self.subsystem_terms) - set(by_subsystem))
        if missing:
            sensors += f"\n- no sensors (unobservable): {', '.join(missing)}"
        couplings = "\n".join(f"- {src} -> {', '.join(dst)}" for src, dst in sorted(self.couplings.items()) if dst)
        return sensors, couplings or "- none"


def normalise_type(value: Any) -> str:
    """Map an LLM-provided hypothesis type onto :data:`HYPOTHESIS_TYPES` (default ``root_cause``)."""
    text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if text in HYPOTHESIS_TYPES:
        return text
    if "independent" in text or "coincident" in text:
        return "independent_events"
    if "insufficient" in text or "discriminat" in text:
        return "insufficient_evidence"
    return "root_cause"
