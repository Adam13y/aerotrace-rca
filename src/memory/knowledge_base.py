"""Engineering knowledge base of known failure modes and past anomalies.

The base is seeded with generic launch-vehicle failure modes (the kind of entries a
propulsion/avionics team keeps in its lessons-learned database). Every pipeline run
also stores its anomalies, so later runs can retrieve similar historical events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.memory.vector_store import SearchResult, VectorStore
from src.models.telemetry_schema import AnomalyEvent, Hypothesis
from src.utils.logger import get_logger

logger = get_logger("memory.knowledge_base")


@dataclass(frozen=True, slots=True)
class FailureMode:
    """A known failure mode with its telemetry signature."""

    id: str
    title: str
    subsystem: str
    signature: str
    mechanism: str
    recommended_actions: tuple[str, ...] = field(default_factory=tuple)

    def as_text(self) -> str:
        """Text used for embedding."""
        return f"{self.title}. Subsystem: {self.subsystem}. Signature: {self.signature}. Mechanism: {self.mechanism}"


SEED_FAILURE_MODES: tuple[FailureMode, ...] = (
    FailureMode(
        "FM-PRESS-001", "Tank pressurisation system leak or regulator failure", "propulsion/pressurisation",
        "LOX or fuel tank ullage pressure drops rapidly, then recovers slowly as the regulator compensates; "
        "may coincide with pressurant bottle pressure decay",
        "Leak in pressurant line, COPV, or ullage; regulator poppet sticking or bang-bang valve failing to cycle",
        ("Review pressurant bottle and regulator outlet pressure for the same interval",
         "Inspect pressurisation lines, COPV mounts and fittings for leak paths",
         "Re-run regulator acceptance test and check valve cycle counts"),
    ),
    FailureMode(
        "FM-STRUCT-002", "Structural failure of internal tank component or strut", "structures",
        "Sudden tank pressure drop coincident with an acceleration or vibration transient, followed by thermal "
        "or pressure cascade in adjacent compartments",
        "Overloaded strut or bracket releases an internal component (e.g. pressurant vessel), rupturing lines",
        ("Correlate accelerometer channels at the event time for an impulsive shock signature",
         "Review material certifications and load margins for internal tank hardware",
         "Inspect recovered hardware or perform fault-tree analysis of structural load paths"),
    ),
    FailureMode(
        "FM-TPUMP-003", "Turbopump bearing or rotor dynamic instability", "propulsion/turbomachinery",
        "Broadband or tonal vibration increase on turbopump accelerometers, possible speed fluctuation and "
        "temperature rise in the engine bay or bearing coolant",
        "Bearing wear, cavitation from low inlet pressure (NPSP shortfall) or rotor rub",
        ("Perform spectral analysis of turbopump vibration around the event",
         "Check pump inlet pressure against NPSP requirements - low tank pressure can induce cavitation",
         "Inspect bearings and seals; review prior hot-fire vibration signatures"),
    ),
    FailureMode(
        "FM-THERM-004", "Engine bay thermal excursion from hot-gas leak", "thermal/propulsion",
        "Engine bay temperature rises rapidly by tens of degrees, often following a propulsion transient",
        "Hot-gas or propellant leak from a cracked duct, gasket or seal; fire in the engine compartment",
        ("Review engine-bay thermocouple spatial distribution to localise the heat source",
         "Inspect hot-gas ducts, gaskets and seals near the hottest thermocouple",
         "Check fire-suppression / purge flow telemetry"),
    ),
    FailureMode(
        "FM-CAV-005", "Propellant pump cavitation due to low inlet pressure", "propulsion",
        "Tank pressure dip followed by turbopump vibration burst and possible chamber pressure oscillation",
        "Ullage pressure falls below the net positive suction pressure margin, vapour bubbles collapse in the pump",
        ("Compare tank pressure minimum with pump NPSP requirement at the flight acceleration",
         "Review chamber pressure for oscillations at the same time",
         "Increase pressurisation margin or tighten regulator band"),
    ),
    FailureMode(
        "FM-ELEC-006", "Avionics bus under-voltage from load transient or battery cell fault", "electrical/avionics",
        "Short sag of the main bus voltage below the regulated band, recovering within a second",
        "Inrush current from actuator or heater switching, failing battery cell, or intermittent connector",
        ("Correlate bus voltage with current sensors and command logs for switching events",
         "Check battery cell voltages and connector continuity on the affected harness",
         "Confirm downstream avionics did not reset or brown out"),
    ),
    FailureMode(
        "FM-SENS-007", "Instrumentation glitch / sensor fault", "instrumentation",
        "Single-sample or very short spike on one channel with no corroborating change on related sensors",
        "Data acquisition bit error, EMI pickup, loose connector, or transducer fault",
        ("Check DAQ parity/CRC error counters and redundant sensors",
         "Inspect the transducer connector and cable shielding",
         "Flag channel for post-flight calibration verification"),
    ),
    FailureMode(
        "FM-COMB-008", "Combustion instability", "propulsion/combustion",
        "High-frequency oscillation on chamber pressure with elevated vibration and injector temperature",
        "Acoustic coupling between combustion and chamber modes, injector damage",
        ("Run high-rate spectral analysis on chamber pressure",
         "Review injector face and baffle condition",
         "Compare against hot-fire stability margins"),
    ),
    FailureMode(
        "FM-POGO-009", "POGO longitudinal oscillation", "propulsion/structures",
        "Low-frequency coupled oscillation of tank pressure, chamber pressure and axial acceleration",
        "Feed-system and structural modes couple through thrust; accumulator mistuned",
        ("Analyse coherence between axial acceleration and propellant line pressures",
         "Verify POGO accumulator charge pressure"),
    ),
    FailureMode(
        "FM-MAXQ-010", "Aerodynamic loading near Max-Q", "structures/aerodynamics",
        "Vibration and structural loads peak in the transonic / Max-Q region; usually expected and bounded",
        "Aerodynamic buffet, wind shear and transonic shock oscillation",
        ("Compare measured loads with pre-flight load predictions and day-of-launch winds",),
    ),
)


class KnowledgeBase:
    """Similarity search over seeded failure modes and historical anomalies."""

    #: Extra results fetched when excluding same-dataset records (one flight rarely has more).
    _SELF_RECORD_HEADROOM = 50

    def __init__(self, persist_dir: str | Path, store: VectorStore | None = None, seed: bool = True) -> None:
        """Open the knowledge base, seeding failure modes if they are not present.

        Args:
            persist_dir: Directory for the underlying vector store.
            store: Optional pre-built store (useful in tests).
            seed: Insert :data:`SEED_FAILURE_MODES` on first use.
        """
        self.store = store or VectorStore(persist_dir)
        #: Record IDs skipped by the most recent same-dataset exclusions (reset by callers per run).
        self.excluded_ids: set[str] = set()
        self._modes = {mode.id: mode for mode in SEED_FAILURE_MODES}
        if seed:
            self._seed()

    def _seed(self) -> None:
        self.store.upsert(
            ids=[mode.id for mode in SEED_FAILURE_MODES],
            texts=[mode.as_text() for mode in SEED_FAILURE_MODES],
            metadatas=[
                {"kind": "failure_mode", "title": m.title, "subsystem": m.subsystem,
                 "recommended_actions": list(m.recommended_actions)}
                for m in SEED_FAILURE_MODES
            ],
        )
        logger.debug("Knowledge base ready (%d documents, backend=%s)", self.store.count(), self.store.backend)

    def failure_mode(self, mode_id: str) -> FailureMode | None:
        """Return a seeded failure mode by ID."""
        return self._modes.get(mode_id)

    @staticmethod
    def anomaly_query_text(anomaly: AnomalyEvent) -> str:
        """Build the search text that represents an anomaly."""
        sensor = anomaly.sensor_id.replace("_", " ")
        category = anomaly.classification.get("category", "")
        return f"{sensor} {category} {anomaly.description}"

    def search(self, text: str, k: int = 3, kind: str | None = None, min_score: float = 0.05) -> list[SearchResult]:
        """Search the knowledge base.

        Args:
            text: Free-text query.
            k: Maximum number of results.
            kind: Optional filter, ``"failure_mode"`` or ``"anomaly"``.
            min_score: Minimum cosine similarity.
        """
        return self.store.query(text, k=k, min_score=min_score, where={"kind": kind} if kind else None)

    @staticmethod
    def is_same_dataset(metadata: dict[str, Any], dataset_id: str | None, source: str | None) -> bool:
        """Whether a stored record came from the dataset being analysed.

        Records carry the dataset fingerprint; legacy records without one are matched by file name.
        """
        stored_id = metadata.get("dataset_id")
        if stored_id:
            return bool(dataset_id) and stored_id == dataset_id
        return bool(source) and metadata.get("source") == source

    def similar_to(
        self,
        anomaly: AnomalyEvent,
        k: int = 3,
        kind: str | None = None,
        exclude_dataset: str | None = None,
        exclude_source: str | None = None,
    ) -> list[SearchResult]:
        """Return knowledge-base entries similar to an anomaly.

        Excludes the anomaly itself and, when ``exclude_dataset``/``exclude_source`` are given,
        every record saved from that dataset, so a re-analysis never retrieves its own output.
        Excluded record IDs are collected in :attr:`excluded_ids`.
        """
        excluding = bool(exclude_dataset or exclude_source)
        # Over-fetch so that filtering out same-dataset records still leaves k results.
        fetch = k + 1 + (self._SELF_RECORD_HEADROOM if excluding else 0)
        hits = self.search(self.anomaly_query_text(anomaly), k=fetch, kind=kind)
        kept: list[SearchResult] = []
        for hit in hits:
            if hit.id == anomaly.id:
                continue
            if excluding and self.is_same_dataset(hit.metadata, exclude_dataset, exclude_source):
                self.excluded_ids.add(hit.id)
                continue
            kept.append(hit)
        return kept[:k]

    def context_for(
        self,
        anomaly: AnomalyEvent,
        k: int = 3,
        exclude_dataset: str | None = None,
        exclude_source: str | None = None,
    ) -> list[SearchResult]:
        """Top-``k`` failure modes plus up to two similar past anomalies from *other* datasets."""
        return self.similar_to(anomaly, k, kind="failure_mode") + self.similar_to(
            anomaly, 2, kind="anomaly", exclude_dataset=exclude_dataset, exclude_source=exclude_source
        )

    def record_run(
        self,
        anomalies: list[AnomalyEvent],
        hypotheses: list[Hypothesis],
        source: str,
        dataset_id: str | None = None,
    ) -> int:
        """Store a run's anomalies (with their leading hypothesis) for future retrieval.

        Records are keyed ``<dataset_id>:<anomaly id>`` so that different flights with
        coinciding anomaly IDs (same sensor and timestamp) do not overwrite each other.

        Returns:
            Number of anomalies stored.
        """
        stored = [a for a in anomalies if not a.expected]
        if not stored:
            return 0
        lead = {aid: h.title for h in hypotheses for aid in h.related_anomalies}
        timestamp = datetime.now(UTC).isoformat()
        self.store.upsert(
            ids=[f"{dataset_id}:{a.id}" if dataset_id else a.id for a in stored],
            texts=[self.anomaly_query_text(a) for a in stored],
            metadatas=[
                {
                    "kind": "anomaly",
                    "title": f"{a.sensor_id} {a.severity.value} anomaly ({source})",
                    "severity": a.severity.value,
                    "sensor_id": a.sensor_id,
                    "source": source,
                    **({"dataset_id": dataset_id} if dataset_id else {}),
                    "anomaly_id": a.id,
                    "recorded_at": timestamp,
                    "leading_hypothesis": lead.get(a.id, ""),
                }
                for a in stored
            ],
        )
        logger.info("Stored %d anomalies in the knowledge base", len(stored))
        return len(stored)

    @staticmethod
    def actions_from_metadata(metadata: dict[str, Any]) -> list[str]:
        """Decode recommended actions stored as JSON in vector-store metadata."""
        raw = metadata.get("recommended_actions")
        if isinstance(raw, list):
            return [str(x) for x in raw]
        if isinstance(raw, str) and raw:
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError:
                return [raw]
            return [str(x) for x in decoded] if isinstance(decoded, list) else [str(decoded)]
        return []
