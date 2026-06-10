"""Energy-greedy scheduling baseline for Carbon-SLA-Net."""

from __future__ import annotations

from typing import Dict

import numpy as np

from baselines.renewable_greedy import _is_feasible


class EnergyGreedyScheduler:
    """Assign each task to the feasible node with the lowest power draw."""

    def schedule(self, env) -> Dict[int, int]:
        """Return ``{task_id: node_id}`` for all tasks.

        Selects the feasible node that minimises
        ``baseline_power_w + task.cpu_req * 10.0``.  Does not call
        ``env.step()`` or modify env state.
        """
        J = len(env.nodes)
        cpu_used = np.zeros(J, dtype=float)
        mem_used = np.zeros(J, dtype=float)
        assignments: Dict[int, int] = {}

        for task in env.tasks:
            best_j: int | None = None
            best_power = float("inf")

            for j, node in enumerate(env.nodes):
                if not _is_feasible(node, task, cpu_used[j], mem_used[j]):
                    continue
                power = node.baseline_power_w + task.cpu_req * 10.0
                if power < best_power:
                    best_power = power
                    best_j = j

            if best_j is None:
                best_j = int(
                    np.argmax([env.nodes[j].cpu_capacity - cpu_used[j] for j in range(J)])
                )

            assignments[task.task_id] = best_j
            cpu_used[best_j] += task.cpu_req
            mem_used[best_j] += task.mem_req

        return assignments
