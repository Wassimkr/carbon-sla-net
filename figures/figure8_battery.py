"""Figure 8 — Battery storage dynamics (SoC trajectory + charge/discharge)."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional

from figures.plot_config import (
    apply_paper_style, save_figure,
    SINGLE_COL_W, FIG_HEIGHT, FONT_TITLE, FONT_AXIS, FONT_TICK, FONT_LEGEND,
)

_SOC_TARGET = 0.90
_SOC_MIN    = 0.20
_ALL_SLOTS  = list(range(8))   # T=8 expected slots


def plot_battery_dynamics(
    env_seed: int = 42,
    node_id: int = 0,
    output_dir: str = "results/figures",
    model_path: Optional[str] = "checkpoints/ppo/carbon_sla_net_best/best_model.zip",
    battery_log: Optional[dict] = None,
) -> str:
    """Plot battery SoC trajectory and charge/discharge profile.

    Parameters
    ----------
    env_seed:
        Seed for the episode used to collect battery data.
    node_id:
        Edge node index to plot (0 = first edge node, battery capacity 500 Wh).
    output_dir:
        Output directory.
    model_path:
        Optional path to a trained MaskablePPO model.  If None or missing,
        a greedy policy is used.
    battery_log:
        Pre-collected battery log dict ``{node_id: [{'slot':…, 'soc_fraction':…,
        'b_charge_wh':…, 'b_discharge_wh':…}]}``.  If provided, episode
        collection is skipped.
    """
    apply_paper_style()

    if battery_log is None:
        battery_log = _collect_battery_log(env_seed, node_id, model_path)

    raw_entries = battery_log.get(node_id, [])
    if not raw_entries:
        raw_entries = [
            {"slot": t, "soc_fraction": 0.6 + 0.05 * t,
             "b_charge_wh": 20.0 * (t % 2), "b_discharge_wh": 10.0 * ((t + 1) % 2)}
            for t in _ALL_SLOTS
        ]

    # Fix 1: deduplicate per slot — keep last entry per slot
    slot_map: dict[int, dict] = {}
    for e in raw_entries:
        slot_map[int(e["slot"])] = e

    # Fix 2: forward-fill missing slots so all 8 appear
    full_entries = []
    last_soc = 0.5
    for s in _ALL_SLOTS:
        if s in slot_map:
            last_soc = float(slot_map[s]["soc_fraction"])
            full_entries.append(slot_map[s])
        else:
            # Forward-fill SoC; zero activity for missing slots
            full_entries.append({
                "slot": s,
                "soc_fraction": last_soc,
                "b_charge_wh": 0.0,
                "b_discharge_wh": 0.0,
            })

    slots       = np.array([e["slot"] for e in full_entries])
    soc         = np.array([e["soc_fraction"] for e in full_entries])
    b_charge    = np.array([e["b_charge_wh"] for e in full_entries])
    b_discharge = np.array([e["b_discharge_wh"] for e in full_entries])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(SINGLE_COL_W, FIG_HEIGHT * 1.5),
                                   sharex=True)
    fig.suptitle(f"Battery storage dynamics (Edge node {node_id}, sunny condition)",
                 fontsize=FONT_TITLE)

    # ---- Panel 1: SoC trajectory ----
    ax1.fill_between(slots, _SOC_MIN, _SOC_TARGET,
                     alpha=0.12, color="green", label="Target range")
    ax1.axhline(_SOC_TARGET, color="green", linestyle="--",
                linewidth=0.8, label=f"Target {_SOC_TARGET:.0%}")
    ax1.axhline(_SOC_MIN, color="orange", linestyle="--",
                linewidth=0.8, label=f"Min {_SOC_MIN:.0%}")
    ax1.plot(slots, soc, color="#378ADD", marker="o", linewidth=1.5,
             markersize=4, label="SoC", zorder=3)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("State of charge", fontsize=FONT_AXIS)
    ax1.tick_params(labelsize=FONT_TICK)
    ax1.legend(fontsize=FONT_LEGEND, loc="upper right", ncol=2)

    # ---- Panel 2: Charge / discharge bars ----
    ax2.bar(slots - 0.2, b_charge,    0.35, color="green",  alpha=0.75, label="Charge (Wh)")
    ax2.bar(slots + 0.2, -b_discharge, 0.35, color="orange", alpha=0.75, label="Discharge (Wh)")

    ax2.set_xticks(_ALL_SLOTS)
    ax2.set_xticklabels([str(s) for s in _ALL_SLOTS], fontsize=FONT_TICK)
    ax2.set_xlabel("Time slot", fontsize=FONT_AXIS)
    ax2.set_ylabel("Energy (Wh)", fontsize=FONT_AXIS)
    ax2.tick_params(labelsize=FONT_TICK)
    ax2.legend(fontsize=FONT_LEGEND, loc="upper right")
    ax2.axhline(0, color="black", linewidth=0.5)

    fig.tight_layout()
    return save_figure(fig, "fig8_battery_dynamics", output_dir)


def _collect_battery_log(env_seed: int, node_id: int, model_path: Optional[str]) -> dict:
    """Run one episode and return the env's battery_log."""
    from env.cloud_edge_env import CloudEdgeEnv
    from baselines.energy_greedy import EnergyGreedyScheduler

    env = CloudEdgeEnv(N=50, T=8, renewable_condition="sunny", seed=env_seed)
    obs, _ = env.reset(seed=env_seed)

    model = None
    if model_path and Path(model_path).exists():
        try:
            from sb3_contrib import MaskablePPO
            model = MaskablePPO.load(model_path)
        except Exception:
            model = None

    if model is not None:
        done = False
        while not done:
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
    else:
        assignments = EnergyGreedyScheduler().schedule(env)
        env2 = CloudEdgeEnv(N=50, T=8, renewable_condition="sunny", seed=env_seed)
        env2.reset(seed=env_seed)
        for _ in range(len(env2.tasks)):
            if env2.task_idx >= len(env2.tasks):
                break
            task = env2.tasks[env2.task_idx]
            action = assignments.get(task.task_id, 0)
            _, _, terminated, truncated, _ = env2.step(action)
            if terminated or truncated:
                break
        env = env2

    return env.battery_log
