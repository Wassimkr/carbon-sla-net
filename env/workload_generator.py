"""Workload generator for Carbon-SLA-Net.

Samples synthetic tasks whose statistical properties match Google Cluster
Traces v3, as described in Paper B.  All randomness is encapsulated in an
explicit numpy Generator — no global RNG state is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Task:
    """A computational task with resource requirements and SLA constraints."""

    task_id: int
    cpu_req: float            # normalized CPU units (log-normal)
    mem_req: float            # normalized memory units (uniform)
    priority: int             # integer in {2, 3, 4, 5, 6, 7, 8}
    sla_latency_ms: float     # maximum acceptable latency (ms)
    sla_reliability: float    # minimum required reliability
    sla_throughput_rps: float # minimum required throughput (req/s)
    t_start: int              # arrival time slot (0-indexed)
    t_deadline: int           # latest assignment slot (inclusive)


# SLA step sizes derived so that priority 2 → base value and priority 8 → max:
# latency:    84.0 ms → 44.0 ms  (range 40 over 6 priority steps)
# reliability: 0.92   → 0.98    (range 0.06 over 6 steps)
# throughput:  70 rps → 130 rps  (range 60 over 6 steps)
_LATENCY_STEP = 40.0 / 6.0   # ≈ 6.667 ms per priority level
_RELIABILITY_STEP = 0.06 / 6.0  # = 0.01 per priority level
_THROUGHPUT_STEP = 10.0       # rps per priority level


def _sla_from_priority(p: int) -> tuple[float, float, float]:
    """Compute (latency_ms, reliability, throughput_rps) from priority p."""
    offset = p - 2
    latency = 84.0 - offset * _LATENCY_STEP
    reliability = 0.92 + offset * _RELIABILITY_STEP
    throughput = 70.0 + offset * _THROUGHPUT_STEP
    return latency, reliability, throughput


class WorkloadGenerator:
    """Generates synthetic task workloads for the scheduler environment.

    All sampling uses ``numpy.random.default_rng`` seeded at construction
    time.  Calling ``sample_tasks`` with the same arguments always produces
    the same task list.
    """

    def __init__(self, seed: int = 42) -> None:
        """Store the seed; a fresh RNG is created per ``sample_tasks`` call."""
        self._seed = seed

    def sample_tasks(
        self,
        N: int,
        T: int,
        pattern: str = "uniform",
    ) -> List[Task]:
        """Sample N tasks over T time slots.

        Parameters
        ----------
        N:
            Number of tasks to generate.
        T:
            Number of time slots in the episode (0-indexed).
        pattern:
            Arrival pattern — one of ``'uniform'``, ``'bursty'``,
            ``'heavy'``, ``'light'``.

        Returns
        -------
        List of Task objects sorted by t_start ascending.
        """
        rng = np.random.default_rng(self._seed)

        # --- Resource requirements ---
        cpu_reqs = rng.lognormal(mean=np.log(0.40), sigma=0.80, size=N)
        cpu_reqs = np.clip(cpu_reqs, 0.05, 8.0)

        mem_reqs = rng.uniform(0.5, 4.0, size=N)

        # priorities: discrete uniform over {2, 3, 4, 5, 6, 7, 8}
        priorities = rng.integers(2, 9, size=N)  # 9 exclusive → [2, 8]

        # --- Pattern-based CPU scaling (applied before deadline sampling) ---
        if pattern == "heavy":
            cpu_reqs = np.clip(cpu_reqs * 1.5, 0.05, 8.0)
        elif pattern == "light":
            cpu_reqs = np.clip(cpu_reqs * 0.5, 0.05, 8.0)

        # --- Arrival window sampling ---
        n_burst = int(N * 0.7)
        tasks: List[Task] = []

        for i in range(N):
            p = int(priorities[i])
            sla_lat, sla_rel, sla_thr = _sla_from_priority(p)

            # t_start: always drawn from [0, T-2] inclusive so that a
            # deadline at least 1 slot later is always feasible.
            if pattern == "bursty":
                if i < n_burst:
                    # first 70 %: cluster in slots {0, 1, 2}
                    t_start = int(rng.integers(0, 3))           # [0, 2]
                else:
                    # remaining 30 %: spread over [3, T-2]
                    hi = max(4, T - 1)   # exclusive upper bound → [3, T-2]
                    t_start = int(rng.integers(3, hi))
            else:
                # uniform / heavy / light
                t_start = int(rng.integers(0, T - 1))           # [0, T-2]

            # t_deadline: t_start < t_deadline <= T-1
            max_offset = (T - 1) - t_start   # maximum allowed offset
            offset = int(rng.integers(1, max_offset + 1))       # [1, max_offset]
            t_deadline = t_start + offset

            tasks.append(Task(
                task_id=i,
                cpu_req=float(cpu_reqs[i]),
                mem_req=float(mem_reqs[i]),
                priority=p,
                sla_latency_ms=sla_lat,
                sla_reliability=sla_rel,
                sla_throughput_rps=sla_thr,
                t_start=t_start,
                t_deadline=t_deadline,
            ))

        tasks.sort(key=lambda t: t.t_start)
        return tasks
