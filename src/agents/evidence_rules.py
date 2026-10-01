"""Physical-evidence rules applied to LLM hypotheses.

Rule 1 - evidence for the named cause. Only the hypothesis TITLE and its PRIMARY CAUSE (the first
sentence of the causal chain) are checked; passing mentions later in the chain are not.

* Component level first: a named component (e.g. turbopump cavitation) must be backed by cited
  anomalies on the sensors it requires (AND of OR-groups, glob patterns allowed). If a required
  sensor exists on the vehicle but is not cited, the hypothesis is rejected. If the vehicle has
  no such sensor, the component is "local_component_inferred" (when allowed: confidence capped
  at ``inferred_confidence_cap``) or unobservable (capped at ``unobservable_confidence_cap``).
* Subsystem level as fallback when no listed component is named: every named subsystem needs a
  cited anomaly from a sensor observing it; unobservable subsystems are capped and labelled
  "unobservable - no sensor coverage"; otherwise the hypothesis is rejected.

Rule 2 - cross-subsystem coupling: when a hypothesis links anomalies from different subsystems,
each later anomaly's subsystem must be directly reachable from an earlier cited anomaly's
subsystem through the coupling table. Chains of couplings are deliberately not followed: almost
every subsystem reaches every other within a few hops, which would make the rule meaningless.
Sensors with no subsystem mapping have no couplings at all.

Subsystem names are detected with a configurable term list (longest phrase first, sensor names
stripped from the text beforehand). This is a vocabulary heuristic, not semantic understanding.
"""

from __future__ import annotations

import fnmatch
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
#: Components with their own evidence requirements (checked before the subsystem fallback).
DEFAULT_COMPONENTS: dict[str, dict[str, Any]] = {
    "turbopump_cavitation": {
        "terms": ["cavitat*"],
        "requires": [["turbopump_vibration"], ["lox_tank_pressure", "fuel_tank_pressure"]],
    },
    "combustion_instability": {
        "terms": ["combustion instabilit*", "combustion oscillat*", "chugging", "screech*"],
        "requires": [["chamber_pressure"]],
    },
    "thermal_failure": {
        "terms": ["thermal control", "thermal failure", "overheat*", "insulation", "heat soak", "cooling failure"],
        "requires": [["engine_bay_temp"]],
    },
    "electrical_failure": {
        "terms": ["battery", "batteries", "cell", "under-voltage", "undervoltage", "brownout", "short circuit",
                  "short-circuit", "load transient", "power distribution"],
        "requires": [["avionics_bus_voltage"]],
    },
    "camera_failure": {"terms": ["camera"], "requires": [["camera_*"]], "allow_inferred": True},
}
#: Subsystem each component belongs to (for the cross-subsystem mechanism check).
DEFAULT_COMPONENT_SUBSYSTEMS: dict[str, str] = {
    "turbopump_cavitation": "propulsion",
    "combustion_instability": "propulsion",
    "thermal_failure": "thermal",
    "electrical_failure": "electrical",
    "camera_failure": "avionics",
}
#: Mechanism keywords per subsystem. A term may belong to several subsystems ("hot gas" comes from
#: the engine and heats its surroundings), in which case naming it needs evidence from each.
DEFAULT_MECHANISM_TERMS: dict[str, list[str]] = {
    "propulsion": ["plume", "combustion", "thrust", "cavitat*", "injector", "turbopump", "nozzle", "gas generator",
                   "regenerative cooling", "hot gas", "hot-gas"],
    "thermal": ["radiative", "convective", "ablat*", "hot gas", "hot-gas", "heat soak", "insulation"],
    "pressurization": ["regulator", "helium", "copv", "ullage", "pressurant", "relief valve", "vent valve"],
    "electrical": ["short circuit", "short-circuit", "battery", "brownout", "emi", "under-voltage", "undervoltage",
                   "power distribution"],
    "avionics": ["software", "imu", "guidance", "gps", "flight computer", "star tracker"],
    "structural": ["crack*", "fatigue", "strut", "buckl*", "fractur*"],
    "aerodynamic": ["buffet*", "flutter", "wind shear", "aerodynamic*"],
    "separation": ["pyrotechnic*", "pyro", "separation"],
}
_UNIT_WORDS = frozenset({"hz", "psi", "v", "degc", "g", "k", "pa", "kpa", "a", "w", "rpm", "s"})
HYPOTHESIS_TYPES = ("root_cause", "independent_events", "insufficient_evidence")
INFERRED_LABEL = "local_component_inferred"
UNOBSERVABLE_LABEL = "unobservable - no sensor coverage"


@dataclass(slots=True)
class EvidenceRules:
    """Configuration and checks for the subsystem-evidence and coupling rules."""

    subsystem_terms: dict[str, list[str]] = field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_SUBSYSTEM_TERMS.items()})
    neutral_phrases: list[str] = field(default_factory=lambda: list(DEFAULT_NEUTRAL_PHRASES))
    sensor_evidence: dict[str, list[str]] = field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_SENSOR_EVIDENCE.items()})
    couplings: dict[str, list[str]] = field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_COUPLINGS.items()})
    unobservable_cap: float = 0.3
    components: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        k: {**v, "subsystem": DEFAULT_COMPONENT_SUBSYSTEMS.get(k)} for k, v in DEFAULT_COMPONENTS.items()
    })
    inferred_cap: float = 0.5
    mechanism_terms: dict[str, list[str]] = field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_MECHANISM_TERMS.items()})
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
        rules.inferred_cap = float(data.get("inferred_confidence_cap", rules.inferred_cap))
        if data.get("components"):
            rules.components = {
                str(name).lower(): {
                    "terms": [str(t).lower() for t in spec.get("terms", [])],
                    "requires": [[str(s).lower() for s in group] for group in spec.get("requires", [])],
                    "allow_inferred": bool(spec.get("allow_inferred", False)),
                    "subsystem": str(spec["subsystem"]).lower() if spec.get("subsystem") else None,
                }
                for name, spec in data["components"].items()
            }
        if data.get("mechanism_terms"):
            rules.mechanism_terms = {str(k).lower(): [str(t).lower() for t in v] for k, v in data["mechanism_terms"].items()}
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
    @staticmethod
    def primary_cause_text(title: str, description: str) -> str:
        """Title plus the first sentence of the causal chain - the part that states the cause."""
        first = re.split(r"(?<=[.!?])\s+", description.strip(), maxsplit=1)[0] if description else ""
        return f"{title}. {first}"

    def _find_terms(self, text: str, sensor_ids: Iterable[str], terms: dict[str, list[str]]) -> dict[str, str]:
        """Keys of ``terms`` named in ``text`` -> the phrase that matched.

        Sensor names (with and without a trailing unit word) are stripped first, then phrases are
        matched longest first; a phrase listed under several keys counts for all of them.
        """
        lowered = " " + text.lower().replace("_", " ") + " "
        names: set[str] = set()
        for sensor in sensor_ids:
            for name in (sensor.lower().replace("_", " "), sensor_display_name(sensor).lower()):
                names.add(name)
                words = name.split()
                if len(words) > 1 and words[-1] in _UNIT_WORDS:
                    names.add(" ".join(words[:-1]))
        for name in sorted(names, key=len, reverse=True):
            lowered = re.sub(rf"\b{re.escape(name)}s?\b", " ", lowered)
        keys_by_term: dict[str, set[str | None]] = {}
        for key, values in terms.items():
            for term in values:
                keys_by_term.setdefault(term, set()).add(key)
        for phrase in self.neutral_phrases:
            keys_by_term.setdefault(phrase, set()).add(None)
        phrases = list(keys_by_term.items())
        found: dict[str, str] = {}
        for term, keys in sorted(phrases, key=lambda pair: len(pair[0].rstrip("*")), reverse=True):
            stem = term.rstrip("*")
            pattern = rf"\b{re.escape(stem)}[a-z-]*" if term.endswith("*") else rf"\b{re.escape(stem)}s?\b"
            if re.search(pattern, lowered):
                for key in keys:
                    if key:
                        found.setdefault(key, stem)
                lowered = re.sub(pattern, " ", lowered)
        return found

    def named_subsystems(self, text: str, sensor_ids: Iterable[str]) -> set[str]:
        """Subsystems named in ``text``."""
        return set(self._find_terms(text, sensor_ids, self.subsystem_terms))

    def named_components(self, text: str, sensor_ids: Iterable[str]) -> set[str]:
        """Components (from ``components``) named in ``text``."""
        return set(self._find_terms(text, sensor_ids, {k: v.get("terms", []) for k, v in self.components.items()}))

    def component_check(
        self, text: str, cited: Sequence[AnomalyEvent], vehicle_sensors: Iterable[str]
    ) -> tuple[str | None, list[tuple[str, float]], bool]:
        """Apply the component-level part of rule 1 to ``text`` (title + primary cause).

        Returns:
            ``(rejection_reason, [(label, confidence_cap)], any_component_named)``.
        """
        vehicle = list(vehicle_sensors)
        named = sorted(self.named_components(text, vehicle))
        cited_sensors = [a.sensor_id.lower() for a in cited]
        caps: list[tuple[str, float]] = []
        for name in named:
            spec = self.components[name]
            for group in spec.get("requires", []):
                if any(fnmatch.fnmatch(s, p) for s in cited_sensors for p in group):
                    continue
                on_vehicle = sorted({s for s in vehicle for p in group if fnmatch.fnmatch(s.lower(), p)})
                requirement = " AND ".join(
                    "(" + " OR ".join(g) + ")" if len(g) > 1 else g[0] for g in spec.get("requires", [])
                )
                if on_vehicle:
                    return (
                        f"names {name.replace('_', ' ')} as the cause, which requires cited anomalies on {requirement}, "
                        f"but cites none on {' / '.join(on_vehicle)} (cited: {', '.join(cited_sensors) or 'nothing'})",
                        [], True,
                    )
                if spec.get("allow_inferred"):
                    caps.append((f"{INFERRED_LABEL}: {name}", self.inferred_cap))
                else:
                    caps.append((f"{UNOBSERVABLE_LABEL}: {name}", self.unobservable_cap))
                break
            # Cross-subsystem mechanism check: a component in subsystem A whose stated cause also uses a
            # mechanism from subsystem B needs evidence from B as well.
            home = spec.get("subsystem")
            others = {k: v for k, v in self.mechanism_terms.items() if k != home}
            cited_evidence = set().union(*(self.evidence_categories(a.sensor_id) for a in cited)) if cited else set()
            observable = set().union(*(self.evidence_categories(s) for s in vehicle)) if vehicle else set()
            for subsystem, term in sorted(self._find_terms(text, vehicle, others).items()):
                if subsystem in cited_evidence:
                    continue
                if subsystem in observable:
                    return (
                        f"cross-subsystem hypothesis: names {name.replace('_', ' ')} ({home}) but also a {subsystem} "
                        f"mechanism ('{term}'); evidence from both subsystems is required and no {subsystem} anomaly "
                        f"is cited",
                        [], True,
                    )
                caps.append((f"{UNOBSERVABLE_LABEL}: {subsystem} (cross-subsystem)", self.unobservable_cap))
        return None, caps, bool(named)

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
