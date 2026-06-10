"""SLA-priority scheduling baseline for Carbon-SLA-Net."""

from __future__ import annotations

from typing import Dict

import numpy as np

from baselines.renewable_greedy import _is_feasible


class SLAPriorityScheduler:
    """Assign each task to the feasible node with the lowest base latency."""

    def schedule(self, env) -> Dict[int, int]:
        """Return ``{task_id: node_id}`` for all tasks.

        Selects the feasible node that minimises ``base_latency_ms``.  Does
        not call ``env.step()`` or modify env state.
        """
        J = len(env.nodes)
        cpu_used = np.zeros(J, dtype=float)
        mem_used = np.zeros(J, dtype=float)
        assignments: Dict[int, int] = {}

        for task in env.tasks:
            best_j: int | None = None
            best_lat = float("inf")

            for j, node in enumerate(env.nodes):
                if not _is_feasible(node, task, cpu_used[j], mem_used[j]):
                    continue
                if node.base_latency_ms < best_lat:
                    best_lat = node.base_latency_ms
                    best_j = j

            if best_j is None:
                best_j = int(
                    np.argmax([env.nodes[j].cpu_capacity - cpu_used[j] for j in range(J)])
                )

            assignments[task.task_id] = best_j
            cpu_used[best_j] += task.cpu_req
            mem_used[best_j] += task.mem_req

        return assignments
