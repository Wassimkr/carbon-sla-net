"""Carbon intensity trace loader for Carbon-SLA-Net.

Maps each of the 11 infrastructure nodes to an electricity-grid zone and
provides either real Electricity Maps CSV data or a deterministic synthetic
fallback with zone-appropriate carbon intensity ranges.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Zone mappings and synthetic-trace parameters
# ---------------------------------------------------------------------------

#: Maps node_id → ISO 3166 zone code used by Electricity Maps.
NODE_TO_ZONE: Dict[int, str] = {
    0: "FI", 1: "FI",
    2: "SE", 3: "SE",
    4: "DE", 5: "FR", 6: "NO",
    7: "FR", 8: "DE", 9: "ES", 10: "PL",
}

#: Per-zone uniform ranges (gCO2eq/kWh) for the synthetic fallback.
_ZONE_RANGES: Dict[str, tuple[float, float]] = {
    "FI": (50.0,  200.0),
    "SE": (10.0,   80.0),
    "NO": (5.0,    30.0),
    "DE": (200.0, 500.0),
    "FR": (30.0,  150.0),
    "ES": (100.0, 300.0),
    "PL": (600.0, 900.0),
}

_SYNTHETIC_LENGTH = 8760  # one year of hourly data


def _zone_seed(zone: str) -> int:
    """Deterministic integer seed derived from zone ASCII characters."""
    return int.from_bytes(zone.encode("ascii"), "big")


class CarbonTraceLoader:
    """Loads or generates hourly carbon intensity traces for all 11 nodes.

    For each zone, the loader first tries to read a CSV from
    ``{data_dir}/{zone}_carbon_intensity.csv``.  If the file is absent it
    falls back to a zone-seeded synthetic trace so that results are always
    reproducible without external data.
    """

    def __init__(
        self,
        data_dir: str = "data/electricity_maps",
        T: int = 8,
    ) -> None:
        """Load (or generate) traces for every zone at construction time.

        Parameters
        ----------
        data_dir:
            Directory that may contain ``{ZONE}_carbon_intensity.csv`` files.
        T:
            Episode length in time slots.  Each ``sample_episode`` call
            returns arrays of this length.
        """
        self._T = T
        self._data_dir = data_dir

        # Build per-zone trace arrays once; sample_episode is just indexing.
        self._zone_traces: Dict[str, np.ndarray] = {}
        for zone in set(NODE_TO_ZONE.values()):
            self._zone_traces[zone] = self._load_zone(zone)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_zone(self, zone: str) -> np.ndarray:
        """Return a 1-D float array of carbon intensity values for *zone*."""
        csv_path = os.path.join(self._data_dir, f"{zone}_carbon_intensity.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
            return df["carbon_intensity_avg"].dropna().to_numpy(dtype=float)
        return self._synthetic_trace(zone)

    def _synthetic_trace(self, zone: str) -> np.ndarray:
        """Generate a deterministic synthetic trace seeded from the zone name."""
        rng = np.random.default_rng(_zone_seed(zone))
        lo, hi = _ZONE_RANGES[zone]
        return rng.uniform(lo, hi, size=_SYNTHETIC_LENGTH)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_episode(self, episode_idx: int) -> Dict[int, np.ndarray]:
        """Return carbon intensity arrays for all 11 nodes for one episode.

        Uses a sliding window:
            ``start = (episode_idx * T) % (len(trace) - T)``

        Parameters
        ----------
        episode_idx:
            Zero-based episode index.  Wraps around the trace via modulo so
            arbitrarily large indices are safe.

        Returns
        -------
        Dict mapping node_id (0–10) → numpy array of shape ``(T,)`` with
        carbon intensity values in gCO2eq/kWh.
        """
        result: Dict[int, np.ndarray] = {}
        for node_id, zone in NODE_TO_ZONE.items():
            trace = self._zone_traces[zone]
            window_count = len(trace) - self._T
            start = (episode_idx * self._T) % window_count
            result[node_id] = trace[start : start + self._T].copy()
        return result
