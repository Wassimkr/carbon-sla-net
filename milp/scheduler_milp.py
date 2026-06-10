"""MILP oracle for Carbon-SLA-Net.

Implements the Paper B scheduling formulation using Gurobi.  Used as both a
performance upper-bound baseline and as a label generator for behavior cloning.

All Gurobi exceptions are caught internally; the caller always receives a
valid MILPResult regardless of solver availability or license status.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
    _GUROBI_AVAILABLE = True
except ImportError:
    _GUROBI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MILPResult:
    """Container for a single MILP solve outcome."""

    assignments: Dict[int, int]   # {task_id: node_id}
    obj_value: float              # final objective (inf if infeasible / no solve)
    mip_gap: float                # Gurobi MIPGap at termination
    runtime_s: float              # wall-clock solve time (seconds)
    status: int                   # Gurobi status code (-1 if not run)
    energy_wh: float              # sum of power_draw_w for all assignments
    carbon_gco2: float            # sum of task-level carbon contributions
    sla_violation: float          # sum of latency SLA penalty terms
    n_assigned: int               # number of tasks with a valid assignment
    timed_out: bool               # True when runtime_s >= time_limit


def _make_empty_result(runtime_s: float = 0.0, status: int = -1) -> MILPResult:
    return MILPResult(
        assignments={},
        obj_value=float("inf"),
        mip_gap=0.0,
        runtime_s=runtime_s,
        status=status,
        energy_wh=0.0,
        carbon_gco2=0.0,
        sla_violation=0.0,
        n_assigned=0,
        timed_out=True,
    )


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------

def solve_milp(
    tasks: list,
    nodes: list,
    carbon_per_node: Dict[int, np.ndarray],
    re_per_node: Dict[int, np.ndarray],
    T: int = 8,
    w1: float = 0.5,
    time_limit: float = 60.0,
    silent: bool = True,
) -> MILPResult:
    """Solve the Paper B MILP scheduling problem.

    Parameters
    ----------
    tasks:
        List of ``Task`` objects from the environment.
    nodes:
        List of ``Node`` objects from the environment.
    carbon_per_node:
        Mapping ``{node_id: ndarray(T,)}`` of carbon intensities (gCO2/kWh).
    re_per_node:
        Mapping ``{node_id: ndarray(T,)}`` of renewable energy available (Wh).
    T:
        Number of time slots.
    w1:
        Weight on the energy objective (``1 - w1`` goes to SLA).
    time_limit:
        Gurobi wall-clock time limit in seconds.
    silent:
        Suppress all Gurobi console output when True.

    Returns
    -------
    MILPResult — never raises; exceptions are caught and converted to an
    infeasible result.
    """
    if not _GUROBI_AVAILABLE:
        return _make_empty_result()

    N = len(tasks)
    J = len(nodes)

    if N == 0:
        return MILPResult(
            assignments={}, obj_value=0.0, mip_gap=0.0, runtime_s=0.0,
            status=2, energy_wh=0.0, carbon_gco2=0.0, sla_violation=0.0,
            n_assigned=0, timed_out=False,
        )

    # ------------------------------------------------------------------
    # Precompute per-(i, j) cost coefficients (pure Python floats)
    # ------------------------------------------------------------------
    energy_c: Dict = {}    # (i, j) → power_draw_w
    sla_c: Dict = {}       # (i, j) → latency SLA penalty
    carbon_c: Dict = {}    # (i, j) → carbon contribution

    for i, task in enumerate(tasks):
        for j, node in enumerate(nodes):
            power = node.baseline_power_w + task.cpu_req * 10.0
            energy_c[i, j] = power
            sla_c[i, j] = max(0.0, node.base_latency_ms - task.sla_latency_ms)
            ci = float(carbon_per_node[j][task.t_start])
            carbon_c[i, j] = node.emission_factor * ci * power / 1000.0

    # Precompute tasks_by_slot for capacity constraints
    tasks_by_slot: Dict[int, List[int]] = defaultdict(list)
    for i, task in enumerate(tasks):
        tasks_by_slot[task.t_start].append(i)

    try:
        model = gp.Model("carbon_sla_net")
        if silent:
            model.Params.OutputFlag = 0
        model.Params.TimeLimit = time_limit
        model.Params.MIPGap = 1e-4

        # ---------------------------------------------------------------
        # Variables
        # ---------------------------------------------------------------
        X = model.addVars(N, J, vtype=GRB.BINARY, name="X")
        Y = model.addVars(J, vtype=GRB.BINARY, name="Y")
        S = model.addVars(J, T, lb=0.0, name="S")
        Bch = model.addVars(J, T, lb=0.0, name="Bch")
        Bdis = model.addVars(J, T, lb=0.0, name="Bdis")

        # Tighten bounds for battery variables
        for j in range(J):
            cap = nodes[j].battery_capacity_wh
            for t in range(T):
                S[j, t].ub = cap
                if cap == 0.0:
                    Bch[j, t].ub = 0.0
                    Bdis[j, t].ub = 0.0

        # ---------------------------------------------------------------
        # Constraints
        # ---------------------------------------------------------------

        # C1: Every task must be assigned to exactly one node
        for i in range(N):
            model.addConstr(
                gp.quicksum(X[i, j] for j in range(J)) == 1,
                name=f"assign_{i}",
            )

        # C2: Node activation link
        for i in range(N):
            for j in range(J):
                model.addConstr(X[i, j] <= Y[j], name=f"act_{i}_{j}")

        # C3 & C4: CPU and memory capacity per (node, slot)
        for j in range(J):
            for t in range(T):
                idx_list = tasks_by_slot.get(t, [])
                if not idx_list:
                    continue
                model.addConstr(
                    gp.quicksum(tasks[i].cpu_req * X[i, j] for i in idx_list)
                    <= nodes[j].cpu_capacity,
                    name=f"cpu_{j}_{t}",
                )
                model.addConstr(
                    gp.quicksum(tasks[i].mem_req * X[i, j] for i in idx_list)
                    <= nodes[j].mem_capacity,
                    name=f"mem_{j}_{t}",
                )

        # C3-bis / C4-bis: Cumulative CPU and memory per node — mirrors
        # env.action_masks() which checks node_cpu_used + task.cpu_req <= cpu_capacity
        # across ALL tasks assigned to the node regardless of slot.
        for j in range(J):
            model.addConstr(
                gp.quicksum(tasks[i].cpu_req * X[i, j] for i in range(N))
                <= nodes[j].cpu_capacity,
                name=f"cum_cpu_{j}",
            )
            model.addConstr(
                gp.quicksum(tasks[i].mem_req * X[i, j] for i in range(N))
                <= nodes[j].mem_capacity,
                name=f"cum_mem_{j}",
            )

        # C5: Initial SoC at 60 %
        for j in range(J):
            model.addConstr(
                S[j, 0] == 0.6 * nodes[j].battery_capacity_wh,
                name=f"soc0_{j}",
            )

        # C7: SoC evolution  (C6 bounds handled via variable UBs above)
        for j in range(J):
            for t in range(T - 1):
                model.addConstr(
                    S[j, t + 1] == S[j, t] + Bch[j, t] - Bdis[j, t],
                    name=f"socevo_{j}_{t}",
                )

        # C8: Charge limited to 80 % of available RE (linearised)
        for j in range(J):
            for t in range(T):
                re_val = float(re_per_node[j][t])
                model.addConstr(
                    Bch[j, t] <= max(0.0, 0.80 * re_val),
                    name=f"charge_{j}_{t}",
                )

        # C9: Discharge fraction bounded by delta × SoC
        for j in range(J):
            for t in range(T):
                model.addConstr(
                    Bdis[j, t] <= 0.25 * S[j, t],
                    name=f"dis_frac_{j}_{t}",
                )

        # C10: Discharge magnitude capped at max node power (relaxed)
        for j in range(J):
            for t in range(T):
                model.addConstr(
                    Bdis[j, t] <= nodes[j].max_power_w,
                    name=f"dis_max_{j}_{t}",
                )

        # ---------------------------------------------------------------
        # Objective: minimise w1·E + (1−w1)·SLA + 0.1·CF
        # ---------------------------------------------------------------
        obj = gp.quicksum(
            (w1 * energy_c[i, j]
             + (1.0 - w1) * sla_c[i, j]
             + 0.1 * carbon_c[i, j]) * X[i, j]
            for i in range(N)
            for j in range(J)
        )
        model.setObjective(obj, GRB.MINIMIZE)

        # ---------------------------------------------------------------
        # Solve
        # ---------------------------------------------------------------
        model.optimize()
        runtime_s = float(model.Runtime)

        if model.SolCount == 0:
            return _make_empty_result(runtime_s=runtime_s, status=int(model.Status))

        # ---------------------------------------------------------------
        # Extract solution
        # ---------------------------------------------------------------
        assignments: Dict[int, int] = {}
        for i in range(N):
            for j in range(J):
                if X[i, j].X > 0.5:
                    assignments[tasks[i].task_id] = j
                    break

        energy_wh = 0.0
        carbon_gco2 = 0.0
        sla_violation = 0.0
        for i in range(N):
            tid = tasks[i].task_id
            if tid in assignments:
                j = assignments[tid]
                energy_wh += energy_c[i, j]
                carbon_gco2 += carbon_c[i, j]
                sla_violation += sla_c[i, j]

        try:
            mip_gap = float(model.MIPGap)
        except Exception:
            mip_gap = 0.0

        return MILPResult(
            assignments=assignments,
            obj_value=float(model.ObjVal),
            mip_gap=mip_gap,
            runtime_s=runtime_s,
            status=int(model.Status),
            energy_wh=energy_wh,
            carbon_gco2=carbon_gco2,
            sla_violation=sla_violation,
            n_assigned=len(assignments),
            timed_out=runtime_s >= time_limit * 0.99,
        )

    except Exception:
        return _make_empty_result()
