"""Generate unseen evaluation flights with known faults, plus their ground truth.

These flights test whether the pipeline generalises beyond ``sample_flight_nasa_long.csv``,
on which all earlier tuning was done. They use the same vehicle and mission profile (so the
planned events in ``agents.yaml`` still apply) but new noise seeds, different faults, and one
sensor the pipeline has never seen: ``imu_heartbeat_hz`` (IMU data rate, nominally 100 Hz).

Sets (outputs are never read by the pipeline itself, only by scripts/evaluate_generalization.py):
    eval     data/eval/flights/*.csv + data/eval/ground_truth.json            (first unseen set)
    holdout  data/eval_holdout/flights/*.csv + data/eval_holdout/ground_truth.json
             (second unseen set: new seeds and new fault types, incl. a sensor with no subsystem)
    sample   data/eval/sample_ground_truth.json - ground truth for data/raw/sample_flight_nasa_long.csv

Usage::

    python scripts/generate_eval_flights.py [--set eval|holdout|sample]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.helpers import DEFAULT_BASE_TIME, normalize_timestamp  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVAL_DIR = DATA_DIR / "eval"
UNITS = {
    "camera_frame_rate_hz": "Hz",
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

# ---------------------------------------------------------------------------- held-out scenarios (seeds 201-205)
def combustion_instability(t: np.ndarray, s: dict[str, np.ndarray]) -> None:
    """Chamber-pressure oscillation grows first, then turbopump vibration and engine-bay heating follow."""
    envelope = _smoothstep(t, 200.0, 202.0) * (1.0 - _smoothstep(t, 204.0, 205.0))
    s["chamber_pressure"] += 70.0 * envelope * np.sin(2 * np.pi * 1.5 * (t - 200.0))
    s["turbopump_vibration"] += 2.5 * _smoothstep(t, 201.0, 201.5) * (1.0 - _smoothstep(t, 204.5, 205.5))
    s["engine_bay_temp"] += 15.0 * _decay_after(t, 203.0, 5.0, 30.0)


def thermal_and_fuel_independent(t: np.ndarray, s: dict[str, np.ndarray]) -> None:
    """Unrelated engine-bay heating and fuel tank pressure drop 3 s apart (inside the correlation window)."""
    s["engine_bay_temp"] += 25.0 * _decay_after(t, 130.0, 4.0, 30.0)
    s["fuel_tank_pressure"] -= 4.0 * _decay_after(t, 133.0, 1.2, 10.0)


def vibration_and_camera(t: np.ndarray, s: dict[str, np.ndarray]) -> None:
    """Turbopump vibration burst plus an unrelated on-board camera dropout (sensor with no subsystem) 2 s later."""
    s["turbopump_vibration"] += 3.5 * np.exp(-0.5 * ((t - 174.5) / 0.35) ** 2)
    camera = 30.0 + np.random.default_rng(777).normal(0, 0.1, t.size)
    camera[(t >= 176.0) & (t < 180.0)] = 0.0
    s["camera_frame_rate_hz"] = camera


def regulator_overpressure(t: np.ndarray, s: dict[str, np.ndarray]) -> None:
    """LOX regulator fails open: tank pressure RISES, then relief/venting brings it back slowly."""
    s["lox_tank_pressure"] += 5.0 * _decay_after(t, 100.0, 3.0, 20.0)


def battery_cell_step(t: np.ndarray, s: dict[str, np.ndarray]) -> None:
    """Loss of one battery cell: permanent bus-voltage step down, nothing else affected."""
    s["avionics_bus_voltage"] -= 0.9 * _smoothstep(t, 230.0, 230.3)


HOLDOUT_FLIGHTS: list[dict] = [
    {
        "flight": "holdout_combustion_instability", "seed": 201, "inject": combustion_instability,
        "description": "Combustion instability: chamber-pressure oscillation first, then vibration and bay heating.",
        "faults": [
            {"sensor": "chamber_pressure", "start": 200.0, "end": 205.0, "cause_group": "A", "role": "root"},
            {"sensor": "turbopump_vibration", "start": 201.0, "end": 205.5, "cause_group": "A", "role": "consequence"},
            {"sensor": "engine_bay_temp", "start": 203.0, "end": 225.0, "cause_group": "A", "role": "consequence"},
        ],
        "top_must_cite_any_of": ["chamber_pressure"],
        "top_terms_any": ["combustion", "instabilit", "oscillat", "injector", "chamber"],
        "top_terms_none": ["regulator", "helium", "pressuris", "pressuriz"],
    },
    {
        "flight": "holdout_thermal_fuel_independent", "seed": 202, "inject": thermal_and_fuel_independent,
        "description": "Two independent faults 3 s apart: engine-bay heating and a fuel tank pressure drop.",
        "faults": [
            {"sensor": "engine_bay_temp", "start": 130.0, "end": 150.0, "cause_group": "A", "role": "root"},
            {"sensor": "fuel_tank_pressure", "start": 133.0, "end": 145.0, "cause_group": "B", "role": "root"},
        ],
        "top_must_cite_any_of": ["engine_bay_temp", "fuel_tank_pressure"],
        "top_terms_any": [],
        "top_terms_none": [],
    },
    {
        "flight": "holdout_vibration_camera", "seed": 203, "inject": vibration_and_camera,
        "description": "Turbopump vibration burst and an unrelated camera dropout (unmapped sensor) 1.5 s later.",
        "faults": [
            {"sensor": "turbopump_vibration", "start": 173.5, "end": 175.5, "cause_group": "A", "role": "root"},
            {"sensor": "camera_frame_rate_hz", "start": 176.0, "end": 180.0, "cause_group": "B", "role": "root"},
        ],
        "top_must_cite_any_of": ["turbopump_vibration", "camera_frame_rate_hz"],
        "top_terms_any": [],
        "top_terms_none": [],
    },
    {
        "flight": "holdout_regulator_overpressure", "seed": 204, "inject": regulator_overpressure,
        "description": "LOX regulator fails open: tank pressure rises (not drops), no other effect.",
        "faults": [{"sensor": "lox_tank_pressure", "start": 100.0, "end": 130.0, "cause_group": "A", "role": "root"}],
        "top_must_cite_any_of": ["lox_tank_pressure"],
        "top_terms_any": ["regulator", "overpressur", "over-pressur", "pressuris", "pressuriz", "relief", "vent"],
        "top_terms_none": ["cavitation", "combustion"],
    },
    {
        "flight": "holdout_battery_cell_step", "seed": 205, "inject": battery_cell_step,
        "description": "Battery cell loss: permanent bus-voltage step down, no cascade.",
        "faults": [{"sensor": "avionics_bus_voltage", "start": 230.0, "end": 232.0, "cause_group": "A", "role": "root"}],
        "top_must_cite_any_of": ["avionics_bus_voltage"],
        "top_terms_any": ["battery", "cell", "voltage", "electrical", "power", "bus"],
        "top_terms_none": PROPULSION_TERMS,
    },
]

#: Ground truth for the original tuning flight (data/raw/sample_flight_nasa_long.csv), for regression checks.
SAMPLE_TRUTH = {
    "flight": "sample_flight_nasa_long",
    "path": "data/raw/sample_flight_nasa_long.csv",
    "description": "Original tuning flight: LOX pressurisation drop -> vibration -> bay heating; independent bus sag; "
                   "single-sample fuel pressure glitch.",
    "faults": [
        {"sensor": "lox_tank_pressure", "start": 142.0, "end": 160.0, "cause_group": "A", "role": "root"},
        {"sensor": "turbopump_vibration", "start": 143.0, "end": 145.0, "cause_group": "A", "role": "consequence"},
        {"sensor": "engine_bay_temp", "start": 145.0, "end": 175.0, "cause_group": "A", "role": "consequence"},
        {"sensor": "avionics_bus_voltage", "start": 210.0, "end": 211.0, "cause_group": "B", "role": "root"},
        {"sensor": "fuel_tank_pressure", "start": 230.0, "end": 230.5, "cause_group": "C", "role": "root"},
    ],
    "top_must_cite_any_of": ["lox_tank_pressure"],
    "top_terms_any": [],
    "top_terms_none": [],
}

SCORING_RULES = {
    "version": "2 (adds hypothesis types; decided before the regression and held-out runs)",
    "hypothesis_types": "'independent_events' and 'insufficient_evidence' hypotheses claim no causal link, so they "
                        "never count as false links and do not count as explaining an anomaly",
    "detection": "each injected fault must produce an actionable (non-expected) anomaly on its sensor whose "
                 "interval overlaps [start - 3 s, end + 5 s]",
    "false_positive": "any actionable anomaly not matching an injected fault",
    "root_cause_ranked_first": "the top-ranked hypothesis is a root cause that cites an anomaly on a "
                               "'top_must_cite_any_of' sensor, its title+description contains one of 'top_terms_any' "
                               "(if given) and none of 'top_terms_none', and it does not cite two cause groups. On "
                               "flights with independent faults, an 'independent_events' #1 citing both groups is "
                               "also correct. Nominal flights use 'no critical outcome' instead",
    "no_critical_outcome": "eval_nominal: no CRITICAL actionable anomaly and no hypothesis with confidence >= 50%",
    "all_explained": "every detected injected-fault anomaly is cited by at least one root-cause hypothesis",
    "false_link": "a hypothesis cites anomalies from different cause groups, or a fault anomaly together with a "
                  "false-positive anomaly",
}


def write_set(flights: list[dict], out_dir: Path) -> None:
    """Write one set of flights and its ground truth."""
    flights_dir = out_dir / "flights"
    flights_dir.mkdir(parents=True, exist_ok=True)
    truth: dict = {"rules": SCORING_RULES, "flights": {}}
    for spec in flights:
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
        entry = {k: v for k, v in spec.items() if not isinstance(v, Callable)}
        entry["path"] = path.relative_to(DATA_DIR.parent).as_posix()
        truth["flights"][spec["flight"]] = entry
        print(f"wrote {path}  ({len(spec['faults'])} injected fault(s))")
    (out_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'ground_truth.json'}")


def main() -> int:
    """Write the requested evaluation set."""
    parser = argparse.ArgumentParser(description="Generate evaluation flights with known faults.")
    parser.add_argument("--set", choices=("eval", "holdout", "sample"), default="eval")
    args = parser.parse_args()
    if args.set == "eval":
        write_set(FLIGHTS, EVAL_DIR)
    elif args.set == "holdout":
        write_set(HOLDOUT_FLIGHTS, DATA_DIR / "eval_holdout")
    else:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        truth = {"rules": SCORING_RULES, "flights": {SAMPLE_TRUTH["flight"]: SAMPLE_TRUTH}}
        (EVAL_DIR / "sample_ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
        print(f"wrote {EVAL_DIR / 'sample_ground_truth.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
