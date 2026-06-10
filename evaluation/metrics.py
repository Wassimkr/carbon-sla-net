"""Metric computation and aggregation for Carbon-SLA-Net evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class EpisodeMetrics:
    """Raw metrics for a single evaluation episode."""

    energy_wh: float
    carbon_gco2: float
    sla_violation: float
    re_used_wh: float
    inference_ms: float
    n_tasks_assigned: int
    reward: float


@dataclass
class AggregatedMetrics:
    """Statistics aggregated over multiple evaluation episodes."""

    mean_energy_wh: float
    std_energy_wh: float
    ci95_energy_wh: float
    mean_carbon_gco2: float
    std_carbon_gco2: float
    ci95_carbon_gco2: float
    mean_sla_violation: float
    std_sla_violation: float
    ci95_sla_violation: float
    mean_re_used_wh: float
    mean_inference_ms: float
    mean_reward: float
    n_episodes: int
    composite_energy: float   # mean_energy_wh + 0.1 * mean_carbon_gco2


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics(episodes: List[EpisodeMetrics]) -> AggregatedMetrics:
    """Compute mean, std, and 95% CI over a list of episode metrics.

    Parameters
    ----------
    episodes:
        Non-empty list of :class:`EpisodeMetrics`.

    Raises
    ------
    ValueError
        If *episodes* is empty.
    """
    if not episodes:
        raise ValueError("Cannot aggregate an empty list of EpisodeMetrics.")

    n = len(episodes)

    def _stats(values: List[float]):
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
        std = math.sqrt(var)
        ci95 = 1.96 * std / math.sqrt(n)
        return mean, std, ci95

    energies = [e.energy_wh for e in episodes]
    carbons = [e.carbon_gco2 for e in episodes]
    slas = [e.sla_violation for e in episodes]

    mean_e, std_e, ci95_e = _stats(energies)
    mean_c, std_c, ci95_c = _stats(carbons)
    mean_s, std_s, ci95_s = _stats(slas)
    mean_re = sum(e.re_used_wh for e in episodes) / n
    mean_ms = sum(e.inference_ms for e in episodes) / n
    mean_r = sum(e.reward for e in episodes) / n

    return AggregatedMetrics(
        mean_energy_wh=mean_e,
        std_energy_wh=std_e,
        ci95_energy_wh=ci95_e,
        mean_carbon_gco2=mean_c,
        std_carbon_gco2=std_c,
        ci95_carbon_gco2=ci95_c,
        mean_sla_violation=mean_s,
        std_sla_violation=std_s,
        ci95_sla_violation=ci95_s,
        mean_re_used_wh=mean_re,
        mean_inference_ms=mean_ms,
        mean_reward=mean_r,
        n_episodes=n,
        composite_energy=mean_e + 0.1 * mean_c,
    )


# ---------------------------------------------------------------------------
# Gap and table formatting
# ---------------------------------------------------------------------------

def milp_gap(method_energy: float, milp_energy: float) -> float:
    """Percentage gap of *method_energy* relative to *milp_energy*.

    Returns ``float('inf')`` when *milp_energy* is zero.
    """
    if milp_energy == 0.0:
        return float("inf")
    return (method_energy - milp_energy) / milp_energy * 100.0


def format_results_table(results: Dict[str, AggregatedMetrics]) -> str:
    """Format a results dict as a fixed-width ASCII table.

    Parameters
    ----------
    results:
        Mapping of method name → :class:`AggregatedMetrics`.  If ``'MILP'``
        is present it is used as the gap baseline; otherwise Gap shows ``N/A``.

    Returns
    -------
    Multi-line string ready for printing or saving.
    """
    col_w = [20, 13, 13, 13, 13, 13, 10]
    header = (
        f"{'Method':<{col_w[0]}}"
        f"{'E (Wh)':<{col_w[1]}}"
        f"{'CF (gCO2)':<{col_w[2]}}"
        f"{'SLA viol.':<{col_w[3]}}"
        f"{'RE (Wh)':<{col_w[4]}}"
        f"{'t (ms)':<{col_w[5]}}"
        f"{'Gap (%)':<{col_w[6]}}"
    )
    sep = "─" * sum(col_w)

    milp_comp = results["MILP"].composite_energy if "MILP" in results else None

    rows = [header, sep]
    for method, agg in results.items():
        gap_str: str
        if milp_comp is None:
            gap_str = "N/A"
        elif method == "MILP":
            gap_str = "0.0"
        else:
            g = milp_gap(agg.composite_energy, milp_comp)
            gap_str = f"{g:.1f}" if math.isfinite(g) else "inf"

        e_str = f"{agg.mean_energy_wh:.1f}±{agg.ci95_energy_wh:.1f}"
        c_str = f"{agg.mean_carbon_gco2:.1f}±{agg.ci95_carbon_gco2:.1f}"
        s_str = f"{agg.mean_sla_violation:.1f}±{agg.ci95_sla_violation:.1f}"
        re_str = f"{agg.mean_re_used_wh:.1f}"
        ms_str = f"{agg.mean_inference_ms:.1f}"

        rows.append(
            f"{method:<{col_w[0]}}"
            f"{e_str:<{col_w[1]}}"
            f"{c_str:<{col_w[2]}}"
            f"{s_str:<{col_w[3]}}"
            f"{re_str:<{col_w[4]}}"
            f"{ms_str:<{col_w[5]}}"
            f"{gap_str:<{col_w[6]}}"
        )

    return "\n".join(rows)
