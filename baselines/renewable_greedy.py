"""Renewable-greedy scheduling baseline for Carbon-SLA-Net.

Replicates the RenewableGreedy heuristic from Paper B: each task is assigned
to the feasible node with the highest renewable energy at the task's start slot.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def _is_feasible(node, task, cpu_used: float, mem_used: float) -> bool:
    """Same three-check feasibility as CloudEdgeEnv.action_masks()."""
    if cpu_used + task.cpu_req > node.cpu_capacity:
        return False
    if mem_used + task.mem_req > node.mem_capacity:
        return False
    projected_latency = (
        node.base_latency_ms
        + node.contention_coeff * (cpu_used + task.cpu_req)
    )
    return projected_latency <= task.sla_latency_ms


class RenewableGreedyScheduler:
    """Assign each task to the feasible node with the most available RE."""

    def schedule(self, env) -> Dict[int, int]:
        """Return ``{task_id: node_id}`` for all tasks.

        Parameters
        ----------
        env:
            A reset :class:`~env.cloud_edge_env.CloudEdgeEnv`.  This method
            reads ``env.tasks``, ``env.nodes``, and ``env.re_episode`` but
            does **not** call ``env.step()`` or modify env state.
        """
        J = len(env.nodes)
        cpu_used = np.zeros(J, dtype=float)
        mem_used = np.zeros(J, dtype=float)
        assignments: Dict[int, int] = {}

        for task in env.tasks:
            best_j: int | None = None
            best_re = -1.0

            for j, node in enumerate(env.nodes):
                if not _is_feasible(node, task, cpu_used[j], mem_used[j]):
                    continue
                re = float(env.re_episode[j][task.t_start])
                if re > best_re:
                    best_re = re
                    best_j = j

            if best_j is None:
                # Fallback: most remaining CPU capacity
                best_j = int(
                    np.argmax([env.nodes[j].cpu_capacity - cpu_used[j] for j in range(J)])
                )

            assignments[task.task_id] = best_j
            cpu_used[best_j] += task.cpu_req
            mem_used[best_j] += task.mem_req

        return assignments
