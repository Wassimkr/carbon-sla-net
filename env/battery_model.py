"""Battery storage model for Carbon-SLA-Net.

Implements Paper B Equations 3–5 for charge/discharge dynamics.
Nodes with zero capacity (cloud nodes) return all-zero outputs with no
division-by-zero risk.
"""

from __future__ import annotations

from typing import Dict


class BatteryModel:
    """Simulates a node-local battery with renewable-aware charge/discharge.

    Implements the three equations from Paper B:

    * **Eq. 4** – charge from renewable surplus, capped at remaining capacity.
    * **Eq. 5** – discharge up to ``delta`` fraction of current SoC, only
      when renewables are insufficient.
    * **Eq. 3** – SoC update with hard clipping to ``[0, capacity_wh]``.

    Parameters
    ----------
    capacity_wh:
        S_max_j — maximum battery capacity (Wh).  Use 0.0 for cloud nodes.
    initial_soc_fraction:
        Starting state of charge as a fraction of capacity.
    delta:
        Maximum fraction of current SoC that may be discharged in one slot.
    gamma:
        Renewable energy fraction parameter (fraction of RE used before
        drawing from/storing to battery).
    """

    def __init__(
        self,
        capacity_wh: float,
        initial_soc_fraction: float = 0.6,
        delta: float = 0.25,
        gamma: float = 0.80,
    ) -> None:
        """Initialise battery state."""
        self._capacity_wh = capacity_wh
        self._delta = delta
        self._gamma = gamma
        self.soc_wh: float = initial_soc_fraction * capacity_wh

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def soc_fraction(self) -> float:
        """Current state of charge as a fraction of capacity (0.0–1.0)."""
        if self._capacity_wh <= 0.0:
            return 0.0
        return self.soc_wh / self._capacity_wh

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, power_draw_w: float, re_available_wh: float) -> Dict[str, float]:
        """Advance battery state by one time slot.

        Parameters
        ----------
        power_draw_w:
            Node power consumption during this slot (W treated as Wh for a
            1-hour slot; caller is responsible for unit consistency).
        re_available_wh:
            Renewable energy available at this node during this slot (Wh).

        Returns
        -------
        Dict with keys ``b_charge_wh``, ``b_discharge_wh``, ``soc_wh``,
        ``soc_fraction``.
        """
        if self._capacity_wh <= 0.0:
            return {
                "b_charge_wh": 0.0,
                "b_discharge_wh": 0.0,
                "soc_wh": 0.0,
                "soc_fraction": 0.0,
            }

        re_effective = self._gamma * re_available_wh

        # Eq. 4: charge from renewable surplus
        surplus = re_effective - power_draw_w
        b_charge = min(max(0.0, surplus), self._capacity_wh - self.soc_wh)

        # Eq. 5: discharge when RE is insufficient
        deficit = power_draw_w - re_effective
        b_discharge = min(self._delta * self.soc_wh, max(0.0, deficit))

        # Eq. 3: SoC update with hard bounds
        new_soc = self.soc_wh + b_charge - b_discharge
        self.soc_wh = max(0.0, min(new_soc, self._capacity_wh))

        return {
            "b_charge_wh": b_charge,
            "b_discharge_wh": b_discharge,
            "soc_wh": self.soc_wh,
            "soc_fraction": self.soc_fraction,
        }

    def reset(self, initial_soc_fraction: float = 0.6) -> None:
        """Reset SoC to the given fraction of capacity.

        Parameters
        ----------
        initial_soc_fraction:
            Desired starting SoC as a fraction of ``capacity_wh``.
        """
        self.soc_wh = initial_soc_fraction * self._capacity_wh
