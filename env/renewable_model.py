"""Renewable energy model for Carbon-SLA-Net.

Models on-site renewable generation per node and episode.  Cloud nodes
(ids 7–10) carry no on-site renewables.  All randomness is episode-seeded
so the same episode always produces the same RE profile.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


#: Maximum possible renewable output per node per slot (Wh).
MAX_RE_PER_NODE: float = 100.0

#: Node ids that have no on-site renewable generation.
_CLOUD_NODE_IDS = frozenset({7, 8, 9, 10})

#: All node ids in the 11-node topology.
_ALL_NODE_IDS = list(range(11))

#: Capacity-factor uniform ranges per weather condition [lo, hi].
_CONDITION_RANGES: Dict[str, tuple[float, float]] = {
    "sunny":  (0.85, 1.00),
    "cloudy": (0.40, 0.70),
    "no_re":  (0.0,  0.0),
}


class RenewableModel:
    """Models on-site renewable energy availability per node and episode.

    For each episode a single capacity factor is drawn per non-cloud node,
    giving a constant (within the episode) renewable supply of
    ``factor × MAX_RE_PER_NODE`` Wh per slot.

    Parameters
    ----------
    condition:
        One of ``'sunny'``, ``'cloudy'``, ``'no_re'``.
    T:
        Episode length in time slots.
    seed:
        Base seed; the actual RNG seed per episode is ``seed + episode_idx``
        for full reproducibility.
    """

    #: Maximum possible renewable output per node per slot (Wh).
    MAX_RE_PER_NODE: float = 100.0

    def __init__(
        self,
        condition: str = "sunny",
        T: int = 8,
        seed: int = 0,
    ) -> None:
        """Validate condition and store parameters."""
        if condition not in _CONDITION_RANGES:
            raise ValueError(
                f"condition must be one of {list(_CONDITION_RANGES)}, got {condition!r}"
            )
        self._condition = condition
        self._T = T
        self._seed = seed

    def sample_episode(self, episode_idx: int) -> Dict[int, np.ndarray]:
        """Return renewable energy available (Wh) per node for one episode.

        Parameters
        ----------
        episode_idx:
            Zero-based episode index.  Combined with ``self._seed`` to form a
            per-episode RNG so results are fully deterministic.

        Returns
        -------
        Dict mapping node_id (0–10) → numpy array of shape ``(T,)``.
        Cloud nodes always contain all zeros.
        """
        rng = np.random.default_rng(self._seed + episode_idx)
        lo, hi = _CONDITION_RANGES[self._condition]

        result: Dict[int, np.ndarray] = {}
        for node_id in _ALL_NODE_IDS:
            if node_id in _CLOUD_NODE_IDS or self._condition == "no_re":
                result[node_id] = np.zeros(self._T, dtype=float)
            else:
                factor = rng.uniform(lo, hi)
                result[node_id] = np.full(self._T, factor * MAX_RE_PER_NODE, dtype=float)
        return result
