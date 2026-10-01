"""FITS binary-table telemetry ingestion (via astropy).

FITS is common for science-payload and spacecraft housekeeping archives. The reader
takes the first binary table HDU (or the one named by ``hdu``), uses a time column
(``TIME``, ``MET``, ``TIMESTAMP``, ...) and treats every other numeric scalar column
as a sensor channel. Units come from the ``TUNITn`` keywords. Relative times are
offset from the ``DATE-OBS``/``DATE-BEG`` header keyword when present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.csv_parser import CSVParserConfig, CSVTelemetryParser
from src.models.telemetry_schema import TelemetryReading
from src.utils.helpers import normalize_timestamp
from src.utils.logger import get_logger

logger = get_logger("ingestion.fits")

_TIME_COLUMNS = ("TIME", "MET", "TIMESTAMP", "T", "UTC", "DATE_TIME", "TSTART")


class FITSTelemetryReader:
    """Read telemetry from FITS binary tables."""

    def __init__(self, config: CSVParserConfig | None = None, hdu: int | str | None = None) -> None:
        """Create a reader.

        Args:
            config: Missing-value / base-time options shared with the CSV parser.
            hdu: Optional HDU index or name to read; defaults to the first binary table.
        """
        self.config = config or CSVParserConfig()
        self.hdu = hdu

    def parse(self, path: str | Path) -> list[TelemetryReading]:
        """Parse a FITS file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ImportError: If astropy is not installed.
            ValueError: If no suitable binary table or time column is found.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            from astropy.io import fits
        except ImportError as exc:
            raise ImportError("astropy is required to read FITS files: pip install astropy") from exc

        with fits.open(path, memmap=False) as hdul:
            table_hdu = self._select_hdu(hdul, fits)
            header = table_hdu.header
            data = table_hdu.data
            columns = table_hdu.columns
            base_time = self._base_time(header, hdul[0].header)
            time_name = next(
                (c.name for c in columns if c.name.upper() in _TIME_COLUMNS),
                None,
            )
            if time_name is None:
                raise ValueError(f"{path.name}: no time column (expected one of {_TIME_COLUMNS})")

            raw_times = np.asarray(data[time_name])
            if raw_times.dtype.kind in "SU":
                raw_times = raw_times.astype(str)
            timestamps = [normalize_timestamp(v.item() if hasattr(v, "item") else v, base_time) for v in raw_times]

            rows: list[pd.DataFrame] = []
            for column in columns:
                if column.name == time_name:
                    continue
                values = np.asarray(data[column.name])
                if values.ndim != 1 or values.dtype.kind not in "iuf":
                    logger.debug("Skipping non-scalar or non-numeric column %s", column.name)
                    continue
                rows.append(
                    pd.DataFrame(
                        {
                            "timestamp": timestamps,
                            "sensor_id": column.name.lower(),
                            "value": values.astype(float),
                            "unit": column.unit or "",
                        }
                    )
                )
        if not rows:
            raise ValueError(f"{path.name}: no numeric sensor columns found")

        parser = CSVTelemetryParser(
            CSVParserConfig(
                layout="long",
                missing_strategy=self.config.missing_strategy,
                max_interpolation_gap=self.config.max_interpolation_gap,
                base_time=base_time,
            )
        )
        readings = parser.parse_frame(pd.concat(rows, ignore_index=True), source=path.name)
        logger.info("Parsed %d readings from %s", len(readings), path.name)
        return readings

    def _select_hdu(self, hdul: object, fits: object) -> object:
        """Return the requested or first binary-table HDU."""
        if self.hdu is not None:
            return hdul[self.hdu]  # type: ignore[index]
        for item in hdul:  # type: ignore[attr-defined]
            if isinstance(item, fits.BinTableHDU):  # type: ignore[attr-defined]
                return item
        raise ValueError("FITS file contains no binary table HDU")

    def _base_time(self, *headers: object) -> object:
        """Find a reference epoch in the given headers, else use the configured base time."""
        for header in headers:
            for key in ("DATE-OBS", "DATE-BEG", "DATE"):
                value = header.get(key) if hasattr(header, "get") else None  # type: ignore[union-attr]
                if value:
                    try:
                        return normalize_timestamp(str(value))
                    except ValueError:
                        continue
        return self.config.base_time
