"""Gymnasium environment for Carbon-SLA-Net Phase 1.

Wraps the Phase 0 data layer (infrastructure, workload, carbon, renewable,
battery) into a training-ready RL environment compatible with MaskablePPO
from sb3-contrib.  Only gymnasium is imported here — no SB3 dependency.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import gymnasium
import numpy as np

from env.infrastructure import Node, build_infrastructure
from env.workload_generator import Task, WorkloadGenerator
from env.carbon_trace import CarbonTraceLoader
from env.renewable_model import RenewableModel
from env.battery_model import BatteryModel


_DEFAULT_WEIGHTS: Dict[str, float] = {
    "energy": 0.3,
    "carbon": 0.3,
    "sla": 0.3,
    "renewable": 0.1,
}

_TIER_ID: Dict[str, float] = {"edge": 0.0, "far_edge": 0.5, "cloud": 1.0}

# J=11 nodes × 6 features + 5 task features + 3 global features
_OBS_DIM: int = 11 * 6 + 5 + 3  # = 74


class CloudEdgeEnv(gymnasium.Env):
    """Federated cloud-edge task-scheduling environment for DRL training.

    **Observation space** — ``Box(0, 1, (74,), float32)``:
      * 66 node features (6 per node, nodes 0–10)
      * 5 current-task features
      * 3 global-state features

    **Action space** — ``Discrete(11)``: index of node to assign current task.

    **Reward** — episodic (0.0 for every non-terminal step); negative scalar
    at termination combining energy, carbon, SLA violation, and RE utilization.

    **action_masks()** provides per-node feasibility for MaskablePPO (called
    by name via duck-typing — do not rename).
    """

    metadata: Dict[str, List] = {"render_modes": []}

    def __init__(
        self,
        N: int = 100,
        T: int = 8,
        carbon_data_dir: str = "data/electricity_maps",
        workload_pattern: str = "uniform",
        renewable_condition: str = "sunny",
        reward_weights: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ) -> None:
        """Build data-layer objects and define Gym spaces.

        Parameters
        ----------
        N:
            Number of tasks per episode.
        T:
            Number of time slots per episode.
        carbon_data_dir:
            Directory searched for ``{ZONE}_carbon_intensity.csv`` files.
        workload_pattern:
            Passed to ``WorkloadGenerator.sample_tasks()``.
        renewable_condition:
            One of ``'sunny'``, ``'cloudy'``, ``'no_re'``.
        reward_weights:
            Override any subset of ``{'energy', 'carbon', 'sla', 'renewable'}``.
        seed:
            Base seed for workload generation.
        """
        super().__init__()

        self.N = N
        self.T = T
        self.workload_pattern = workload_pattern

        self.reward_weights = dict(_DEFAULT_WEIGHTS)
        if reward_weights is not None:
            self.reward_weights.update(reward_weights)

        # Infrastructure
        self.nodes: List[Node] = build_infrastructure()
        self.J: int = len(self.nodes)  # must be 11

        # Data-layer objects
        self.wl_gen = WorkloadGenerator(seed=seed)
        self.carbon_loader = CarbonTraceLoader(data_dir=carbon_data_dir, T=T)
        self.re_model = RenewableModel(condition=renewable_condition, T=T, seed=seed)
        self.batteries: List[BatteryModel] = [
            BatteryModel(capacity_wh=n.battery_capacity_wh) for n in self.nodes
        ]

        # Gym spaces
        self.observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(_OBS_DIM,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Discrete(self.J)

        # Episode-level state — fully initialised in reset()
        self.episode_count: int = 0
        self.tasks: List[Task] = []
        self.task_idx: int = 0
        self.current_slot: int = 0
        self.carbon_episode: Dict[int, np.ndarray] = {}
        self.re_episode: Dict[int, np.ndarray] = {}
        self.node_cpu_used: np.ndarray = np.zeros(self.J, dtype=float)
        self.node_mem_used: np.ndarray = np.zeros(self.J, dtype=float)
        self.episode_energy_wh: float = 0.0
        self.episode_carbon_gco2: float = 0.0
        self.episode_sla_violation: float = 0.0
        self.episode_re_used_wh: float = 0.0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Any = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset to the start of a new episode.

        Parameters
        ----------
        seed:
            If not None, re-seeds the workload generator so the same tasks
            are produced on subsequent resets with this seed.
        options:
            Unused; present for API compatibility.

        Returns
        -------
        (obs, info) where obs has shape (74,) and info is an empty dict.
        """
        super().reset(seed=seed)

        if seed is not None:
            self.wl_gen = WorkloadGenerator(seed=seed)

        # Sample workload — sorted by t_start ascending
        self.tasks = self.wl_gen.sample_tasks(self.N, self.T, self.workload_pattern)
        self.task_idx = 0
        self.current_slot = self.tasks[0].t_start if self.tasks else 0

        # Sample per-episode environment signals (indexed by episode_count so
        # each episode sees a different carbon / RE profile)
        self.carbon_episode = self.carbon_loader.sample_episode(self.episode_count)
        self.re_episode = self.re_model.sample_episode(self.episode_count)

        # Reset accumulators
        self.episode_energy_wh = 0.0
        self.episode_carbon_gco2 = 0.0
        self.episode_sla_violation = 0.0
        self.episode_re_used_wh = 0.0

        # Reset batteries and node load counters
        for bat in self.batteries:
            bat.reset()
        self.node_cpu_used = np.zeros(self.J, dtype=float)
        self.node_mem_used = np.zeros(self.J, dtype=float)

        # Battery log: populated during step() for post-hoc analysis / Figure 8
        self.battery_log: dict = {}

        self.episode_count += 1
        return self._build_obs(), {}

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Assign the current task to the node identified by *action*.

        Parameters
        ----------
        action:
            Integer node index in ``[0, J-1]``.

        Returns
        -------
        ``(obs, reward, terminated, truncated, info)``

        Raises
        ------
        ValueError
            If *action* is outside ``[0, J-1]``.
        """
        if not (0 <= int(action) < self.J):
            raise ValueError(
                f"action={action} is out of range [0, {self.J - 1}]"
            )
        action = int(action)

        task = self.tasks[self.task_idx]
        self.current_slot = task.t_start

        # Resource accounting
        self.node_cpu_used[action] += task.cpu_req
        self.node_mem_used[action] += task.mem_req

        # Power draw: baseline + CPU-proportional load (10 W per normalised CPU unit)
        power_draw_w = self.nodes[action].baseline_power_w + task.cpu_req * 10.0

        # Battery step for the assigned node
        re_wh = float(self.re_episode[action][self.current_slot])
        bat_result = self.batteries[action].step(
            power_draw_w=power_draw_w,
            re_available_wh=re_wh,
        )

        # Carbon: power not covered by RE or battery discharge
        non_renewable_w = max(
            0.0,
            power_draw_w - bat_result["b_discharge_wh"] - re_wh,
        )
        carbon_ci = float(self.carbon_episode[action][self.current_slot])
        task_carbon_gco2 = non_renewable_w * carbon_ci / 1000.0

        # Battery log entry for Figure 8 / post-hoc analysis
        self.battery_log.setdefault(action, []).append({
            "slot": self.current_slot,
            "soc_fraction": self.batteries[action].soc_fraction,
            "b_charge_wh": bat_result["b_charge_wh"],
            "b_discharge_wh": bat_result["b_discharge_wh"],
        })

        # Update episode accumulators
        self.episode_energy_wh += power_draw_w
        self.episode_carbon_gco2 += task_carbon_gco2
        self.episode_re_used_wh += min(re_wh, power_draw_w)
        self.episode_sla_violation += self._compute_sla_violation(task, action)

        self.task_idx += 1
        terminated: bool = self.task_idx >= len(self.tasks)
        reward: float = self._compute_reward() if terminated else 0.0

        info: Dict[str, Any] = {
            "energy_wh": self.episode_energy_wh,
            "carbon_gco2": self.episode_carbon_gco2,
            "sla_violation": self.episode_sla_violation,
            "re_used_wh": self.episode_re_used_wh,
            "task_idx": self.task_idx,
            "terminated": terminated,
        }

        return self._build_obs(), reward, terminated, False, info

    def action_masks(self) -> np.ndarray:
        """Return a boolean feasibility mask of shape ``(J,)`` for the current task.

        A node ``j`` is feasible when all three conditions hold:
          1. Post-assignment CPU usage ≤ node capacity.
          2. Post-assignment memory usage ≤ node capacity.
          3. Projected post-assignment latency ≤ ``task.sla_latency_ms``.

        **Safety fallback:** if no node passes all checks, all nodes are set to
        ``True`` with a ``RuntimeWarning`` so MaskablePPO never crashes.

        Returns all ``True`` if the episode has already terminated.
        """
        if self.task_idx >= len(self.tasks):
            return np.ones(self.J, dtype=bool)

        task = self.tasks[self.task_idx]
        masks = np.ones(self.J, dtype=bool)

        for j, node in enumerate(self.nodes):
            # 1. CPU capacity
            if self.node_cpu_used[j] + task.cpu_req > node.cpu_capacity:
                masks[j] = False
                continue

            # 2. Memory capacity
            if self.node_mem_used[j] + task.mem_req > node.mem_capacity:
                masks[j] = False
                continue

            # 3. Latency SLA — linear contention model
            projected_util = (self.node_cpu_used[j] + task.cpu_req) / node.cpu_capacity
            projected_latency = (
                node.base_latency_ms
                + node.contention_coeff * projected_util * node.cpu_capacity
            )
            if projected_latency > task.sla_latency_ms:
                masks[j] = False

        if not np.any(masks):
            warnings.warn(
                f"No feasible node for task {self.task_idx} (all masked); "
                "enabling all nodes as fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            masks[:] = True

        return masks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_sla_violation(self, task: Task, node_idx: int) -> float:
        """Quantify SLA violation for *task* placed on *node_idx*.

        Uses the updated ``node_cpu_used`` (resource already incremented in
        ``step()``).  Implements Paper B Equations 18–20.

        Returns a non-negative scalar; 0.0 means perfect SLA compliance.
        """
        node = self.nodes[node_idx]
        util = self.node_cpu_used[node_idx] / node.cpu_capacity

        # Eq. 18 — latency (ms), grows linearly with utilisation
        latency = node.base_latency_ms + node.contention_coeff * util * node.cpu_capacity
        lat_viol = max(0.0, latency - task.sla_latency_ms)

        # Eq. 19 — reliability degrades above utilisation threshold
        over_thresh = max(0.0, util - node.utilization_thresh)
        reliability = node.reliability_base * (1.0 - node.reliability_degrad * over_thresh)
        rel_viol = max(0.0, task.sla_reliability - reliability)

        # Eq. 20 — throughput inversely proportional to cumulative load
        denom = max(self.node_cpu_used[node_idx], 0.01)
        throughput = (node.cpu_capacity / denom) * 50.0
        thr_viol = max(0.0, task.sla_throughput_rps - throughput)

        # Scale so each component is roughly 0–10 at moderate load
        return lat_viol + rel_viol * 100.0 + thr_viol * 0.1

    def _compute_reward(self) -> float:
        """Compute the episodic reward at termination.

        Normalises each objective by a fixed reference value (training-stable),
        then combines with the configured weights.  The reward is negative:
        energy, carbon, and SLA violation are costs; RE utilisation is a bonus.
        """
        w = self.reward_weights

        e_norm = self.episode_energy_wh / 5000.0
        c_norm = self.episode_carbon_gco2 / 1000.0
        s_norm = self.episode_sla_violation / (self.N * 100.0)

        total_re_available = sum(
            float(np.sum(self.re_episode[j]))
            for j in range(self.J)
        )
        r_norm = self.episode_re_used_wh / max(total_re_available, 1.0)

        reward = -(
            w["energy"] * e_norm
            + w["carbon"] * c_norm
            + w["sla"] * s_norm
            - w["renewable"] * r_norm
        )
        return float(reward)

    def _build_obs(self) -> np.ndarray:
        """Construct the 74-dimensional normalised observation vector.

        Returns an all-zero vector when the episode has terminated
        (``task_idx >= len(tasks)``).
        """
        if self.task_idx >= len(self.tasks):
            return np.zeros(_OBS_DIM, dtype=np.float32)

        obs = np.empty(_OBS_DIM, dtype=np.float32)
        ptr = 0

        # --- Node features: J × 6 = 66 ---
        for j, node in enumerate(self.nodes):
            obs[ptr]     = self.node_cpu_used[j] / node.cpu_capacity          # cpu_util
            obs[ptr + 1] = self.node_mem_used[j] / node.mem_capacity          # mem_util
            obs[ptr + 2] = float(self.batteries[j].soc_fraction)              # bat_soc
            obs[ptr + 3] = (                                                   # re_avail
                self.re_episode[j][self.current_slot] / RenewableModel.MAX_RE_PER_NODE
            )
            obs[ptr + 4] = self.carbon_episode[j][self.current_slot] / 1000.0 # carbon_ci
            obs[ptr + 5] = _TIER_ID[node.tier]                                # tier_id
            ptr += 6

        # --- Current task features: 5 ---
        task = self.tasks[self.task_idx]
        obs[ptr]     = task.cpu_req / 8.0
        obs[ptr + 1] = task.mem_req / 4.0
        obs[ptr + 2] = (task.t_deadline - task.t_start) / self.T
        obs[ptr + 3] = (task.priority - 2) / 6.0
        obs[ptr + 4] = task.sla_latency_ms / 84.0
        ptr += 5

        # --- Global state: 3 ---
        obs[ptr]     = self.current_slot / self.T
        obs[ptr + 1] = (self.N - self.task_idx) / self.N
        obs[ptr + 2] = min(self.episode_energy_wh / 5000.0, 1.0)

        return np.clip(obs, 0.0, 1.0)
