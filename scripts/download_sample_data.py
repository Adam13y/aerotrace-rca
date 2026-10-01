"""Obtain sample telemetry for AeroTrace-RCA.

By default this generates a realistic synthetic 300-second ascent (no network needed)
in three layouts: NASA-style long CSV, SpaceX-style wide CSV and channel JSON. Real
launch telemetry is rarely published as raw sensor data, so the synthetic flight -
with known injected faults - is the reference dataset for demos and tests.

Optionally, ``--url`` downloads an existing CSV/JSON telemetry file into ``data/raw``.

Usage::

    python scripts/download_sample_data.py [--seed 42] [--force]
    python scripts/download_sample_data.py --url https://example.org/telemetry.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import configure_logging, get_settings  # noqa: E402
from src.ingestion import SUPPORTED_SUFFIXES  # noqa: E402
from src.ingestion.csv_parser import INJECTED_ANOMALIES, SENSOR_UNITS, generate_sample_telemetry, write_sample_files  # noqa: E402

MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024


def write_channel_json(output_dir: Path, seed: int) -> Path:
    """Write the synthetic flight as a channel-oriented JSON document."""
    wide = generate_sample_telemetry(seed=seed)
    document = {
        "base_time": wide["timestamp"].iloc[0],
        "mission": "AeroTrace synthetic ascent",
        "sensors": {
            sensor: {
                "unit": unit,
                "samples": [
                    [float(t), None if v != v else float(v)]  # NaN -> null
                    for t, v in zip(wide["met_s"], wide[f"{sensor}_{unit}"])
                ],
            }
            for sensor, unit in SENSOR_UNITS.items()
        },
    }
    path = output_dir / "sample_flight_channels.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def download(url: str, output_dir: Path) -> Path:
    """Download a telemetry file over HTTP(S) into ``output_dir``.

    Raises:
        ValueError: For unsupported schemes/extensions or oversized files.
        urllib.error.URLError: On network failure.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http(s) URLs are supported")
    name = Path(parsed.path).name or "downloaded_telemetry.csv"
    if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type '{Path(name).suffix}' (supported: {sorted(SUPPORTED_SUFFIXES)})")
    destination = output_dir / name
    tmp = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as handle:  # noqa: S310 - user-supplied URL
        size = int(response.headers.get("Content-Length") or 0)
        if size > MAX_DOWNLOAD_BYTES:
            raise ValueError(f"File too large ({size} bytes)")
        copied = 0
        while chunk := response.read(1 << 16):
            copied += len(chunk)
            if copied > MAX_DOWNLOAD_BYTES:
                raise ValueError("File exceeded size limit during download")
            handle.write(chunk)
    shutil.move(tmp, destination)
    return destination


def main() -> int:
    """Generate or download sample data. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Generate or download sample telemetry.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the synthetic flight.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing sample files.")
    parser.add_argument("--url", help="Download a telemetry file from this URL instead of generating one.")
    args = parser.parse_args()

    configure_logging()
    raw_dir = get_settings().raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.url:
        try:
            path = download(args.url, raw_dir)
        except (ValueError, urllib.error.URLError, OSError) as exc:
            print(f"Download failed: {exc}")
            return 1
        print(f"Downloaded {path}")
        return 0

    existing = list(raw_dir.glob("sample_flight_*"))
    if existing and not args.force:
        print("Sample data already present (use --force to regenerate):")
        for path in existing:
            print(f"  {path}")
        return 0

    paths = write_sample_files(raw_dir, seed=args.seed)
    paths["channels_json"] = write_channel_json(raw_dir, args.seed)
    print("Generated synthetic 300 s ascent telemetry:")
    for kind, path in paths.items():
        print(f"  {kind:12} {path}")
    print("Injected faults (MET seconds): " + ", ".join(f"{s}@{t:g}" for s, t in INJECTED_ANOMALIES.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
