"""Generate unseen evaluation flights with known faults, plus their ground truth.

These flights test whether the pipeline generalises beyond ``sample_flight_nasa_long.csv``,
on which all earlier tuning was done. They use the same vehicle and mission profile (so the
planned events in ``agents.yaml`` still apply) but new noise seeds, different faults, and one
sensor the pipeline has never seen: ``imu_heartbeat_hz`` (IMU data rate, nominally 100 Hz).

Outputs (never read by the pipeline itself):
    data/eval/flights/<flight>.csv     NASA-style long CSV, the pipeline input
    data/eval/ground_truth.json        injected faults and scoring rules, read only by
                                       scripts/evaluate_generalization.py

Usage::

    python scripts/generate_eval_flights.py
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.helpers import DEFAULT_BASE_TIME, normalize_timestamp  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"
UNITS = {
    "chamber_pressure": "psi", "lox_tank_pressure": "psi", "fuel_tank_pressure": "psi",
    "engine_bay_temp": "degC", "turbopump_vibration": "g", "avionics_bus_voltage": "V",
    "imu_heartbeat_hz": "Hz",
}


def _smoothstep(t: np.ndarray, start: float, end: float) -> np.ndarray:
    x = np.clip((t - start) / (end - start), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def nominal_flight(seed: int, duration_s: float = 300.0, rate_hz: float = 10.0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Nominal ascent of the reference vehicle (same profile as the sample flight, new noise)."""
    rng = np.random.default_rng(seed)
    t = np.round(np.arange(0.0, duration_s, 1.0 / rate_hz), 6)
    n = t.size
    ignition = _smoothstep(t, 0.0, 3.0)
    throttle = 1.0 - 0.28 * _smoothstep(t, 55.0, 60.0) + 0.28 * _smoothstep(t, 78.0, 83.0)
    engine = ignition * throttle * (1.0 - _smoothstep(t, 284.0, 287.0))
    sensors = {
        "chamber_pressure": 1400.0 * engine + rng.normal(0, 4.0, n),
        "lox_tank_pressure": 50.0 + 0.6 * np.sin(2 * np.pi * t / 37.0) - 0.004 * t + rng.normal(0, 0.12, n),
        "fuel_tank_pressure": 45.0 - 0.006 * t + 0.3 * np.sin(2 * np.pi * t / 51.0) + rng.normal(0, 0.10, n),
        "engine_bay_temp": 20.0 + 100.0 * (1 - np.exp(-t / 90.0)) * (0.3 + 0.7 * ignition) + rng.normal(0, 0.4, n),
        "turbopump_vibration": (0.3 + 1.7 * engine) + 1.5 * np.exp(-0.5 * ((t - 70.0) / 7.0) ** 2)
        + np.abs(rng.normal(0, 0.12, n)),
        "avionics_bus_voltage": 28.0 - 0.15 * ignition + rng.normal(0, 0.04, n),
        "imu_heartbeat_hz": 100.0 + rng.normal(0, 0.15, n),
    }
    return t, sensors


def _decay_after(t: np.ndarray, start: float, ramp: float, tau: float) -> np.ndarray:
    """0 -> 1 over ``ramp`` seconds from ``start``, then exponential decay with time constant ``tau``."""
    return _smoothstep(t, start, start + ramp) * np.exp(-np.clip(t - (start + ramp), 0, None) / tau)


# ---------------------------------------------------------------------------- scenarios
def fuel_turbopump(t: np.ndarray, s: dict[str, np.ndarray]) -> None:
    """Fuel inlet pressure loss -> fuel turbopump cavitation -> thrust dip. LOX side nominal."""
    s["fuel_tank_pressure"] -= 4.5 * _decay_after(t, 118.0, 1.2, 10.0)
    s["turbopump_vibration"] += 4.5 * np.exp(-0.5 * ((t - 119.6) / 0.4) ** 2)
    s["chamber_pressure"] -= 84.0 * _decay_after(t, 120.0, 0.8, 2.5)


def thermal_only(t: np.ndarray, s: dict[str, np.ndarray]) -> None:
    """Engine-bay temperature excursion with no propulsion or other sensor involvement."""
    s["engine_bay_temp"] += 30.0 * _decay_after(t, 165.0, 5.0, 40.0)


def electrical_cascade(t: np.ndarray, s: dict[str, np.ndarray]) -> None:
    """Bus under-voltage that browns out the IMU: its data rate drops to zero, then recovers."""
    sag = (t >= 188.0) & (t < 189.6)
    s["avionics_bus_voltage"][sag] -= 2.2
    dropout = (t >= 188.3) & (t < 190.8)
    s["imu_heartbeat_hz"][dropout] = 0.0


def nominal(t: np.ndarray, s: dict[str, np.ndarray]) -> None:
    """No faults."""


def two_independent(t: np.ndarray, s: dict[str, np.ndarray]) -> None:
    """LOX regulator pressure drop and an unrelated bus sag 2.5 s apart (inside the 5 s correlation window)."""
    s["lox_tank_pressure"] -= 4.0 * _decay_after(t, 150.0, 1.5, 12.0)
    sag = (t >= 152.5) & (t < 153.5)
    s["avionics_bus_voltage"][sag] -= 2.0


# ---------------------------------------------------------------------------- ground truth
PROPULSION_TERMS = ["turbopump", "cavitation", "combustion", "injector", "propellant", "hot-gas", "hot gas",
                    "exhaust", "plume", "nozzle"]

FLIGHTS: list[dict] = [
    {
        "flight": "eval_fuel_turbopump", "seed": 101, "inject": fuel_turbopump,
        "description": "Fuel-side turbopump problem: fuel inlet pressure drops first, LOX side nominal.",
        "faults": [
            {"sensor": "fuel_tank_pressure", "start": 118.0, "end": 130.0, "cause_group": "A", "role": "root"},
            {"sensor": "turbopump_vibration", "start": 118.6, "end": 120.6, "cause_group": "A", "role": "consequence"},
            {"sensor": "chamber_pressure", "start": 120.0, "end": 125.0, "cause_group": "A", "role": "consequence"},
        ],
        "top_must_cite_any_of": ["fuel_tank_pressure"],
        "top_terms_any": ["fuel"],
        "top_terms_none": ["lox", "liquid oxygen", "oxidizer", "oxidiser"],
    },
    {
        "flight": "eval_thermal_only", "seed": 102, "inject": thermal_only,
        "description": "Pure thermal event in the engine bay with no propulsion cause.",
        "faults": [{"sensor": "engine_bay_temp", "start": 165.0, "end": 185.0, "cause_group": "A", "role": "root"}],
        "top_must_cite_any_of": ["engine_bay_temp"],
        "top_terms_any": ["thermal", "heat", "temperature", "insulation", "cooling"],
        "top_terms_none": PROPULSION_TERMS,
    },
    {
        "flight": "eval_electrical_cascade", "seed": 103, "inject": electrical_cascade,
        "description": "Bus under-voltage cascades into an IMU data dropout (new sensor imu_heartbeat_hz).",
        "faults": [
            {"sensor": "avionics_bus_voltage", "start": 188.0, "end": 189.6, "cause_group": "A", "role": "root"},
            {"sensor": "imu_heartbeat_hz", "start": 188.3, "end": 190.8, "cause_group": "A", "role": "consequence"},
        ],
        "top_must_cite_any_of": ["avionics_bus_voltage"],
        "top_terms_any": ["voltage", "electrical", "power", "bus", "battery"],
        "top_terms_none": [],
    },
    {
        "flight": "eval_nominal", "seed": 104, "inject": nominal,
        "description": "Nominal flight, no injected faults.",
        "faults": [],
        "expect_no_critical": True,
    },
    {
        "flight": "eval_two_independent", "seed": 105, "inject": two_independent,
        "description": "Two independent faults 2.5 s apart: LOX regulator pressure drop and an unrelated bus sag.",
        "faults": [
            {"sensor": "lox_tank_pressure", "start": 150.0, "end": 165.0, "cause_group": "A", "role": "root"},
            {"sensor": "avionics_bus_voltage", "start": 152.5, "end": 153.5, "cause_group": "B", "role": "root"},
        ],
        "top_must_cite_any_of": ["lox_tank_pressure", "avionics_bus_voltage"],
        "top_terms_any": [],
        "top_terms_none": [],
    },
]

SCORING_RULES = {
    "detection": "each injected fault must produce an actionable (non-expected) anomaly on its sensor whose "
                 "interval overlaps [start - 3 s, end + 5 s]",
    "false_positive": "any actionable anomaly not matching an injected fault",
    "root_cause_ranked_first": "the top-ranked hypothesis cites an anomaly on a 'top_must_cite_any_of' sensor, its "
                               "title+description contains one of 'top_terms_any' (if given) and none of "
                               "'top_terms_none'; for eval_two_independent it must not cite both cause groups; "
                               "for eval_nominal the check is replaced by 'no critical outcome'",
    "no_critical_outcome": "eval_nominal: no CRITICAL actionable anomaly and no hypothesis with confidence >= 50%",
    "all_explained": "every detected injected-fault anomaly is cited by at least one hypothesis",
    "false_link": "a hypothesis cites anomalies from different cause groups, or a fault anomaly together with a "
                  "false-positive anomaly",
}


def main() -> int:
    """Write the evaluation flights and ground truth."""
    flights_dir = EVAL_DIR / "flights"
    flights_dir.mkdir(parents=True, exist_ok=True)
    truth: dict = {"rules": SCORING_RULES, "flights": {}}
    for spec in FLIGHTS:
        t, sensors = nominal_flight(spec["seed"])
        spec["inject"](t, sensors)
        rng = np.random.default_rng(spec["seed"] + 1000)
        timestamps = [normalize_timestamp(float(x), DEFAULT_BASE_TIME).isoformat(timespec="milliseconds") for x in t]
        rows = []
        for sensor, values in sensors.items():
            values = np.round(values, 4).astype(float)
            mask = rng.random(t.size) < 0.004  # sparse missing samples, never inside a fault window
            for fault in spec["faults"]:
                if fault["sensor"] == sensor:
                    mask &= ~((t >= fault["start"] - 3) & (t <= fault["end"] + 3))
            values[mask] = np.nan
            rows.append(pd.DataFrame({"timestamp": timestamps, "sensor_id": sensor, "value": values,
                                      "unit": UNITS[sensor]}))
        path = flights_dir / f"{spec['flight']}.csv"
        pd.concat(rows, ignore_index=True).to_csv(path, index=False)
        truth["flights"][spec["flight"]] = {k: v for k, v in spec.items() if not isinstance(v, Callable)}
        print(f"wrote {path}  ({len(spec['faults'])} injected fault(s))")
    (EVAL_DIR / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(f"wrote {EVAL_DIR / 'ground_truth.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
