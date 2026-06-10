"""NSGA-II multi-objective scheduling baseline for Carbon-SLA-Net.

Implements the Paper B NSGA-II baseline using the DEAP evolutionary
framework.  Three objectives are minimised simultaneously: energy (Wh),
SLA violation, and carbon (gCO2).
"""

from __future__ import annotations

import functools
import random
import time
from typing import Dict, List, Tuple

import numpy as np

from deap import algorithms, base, creator, tools


# ---------------------------------------------------------------------------
# DEAP module-level creator registration
# Guard against duplicate registration when the module is imported
# multiple times (common in pytest sessions).
# ---------------------------------------------------------------------------

if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))

if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMulti)


# ---------------------------------------------------------------------------
# Fitness evaluation (module-level for pickling compatibility)
# ---------------------------------------------------------------------------

def _evaluate(
    individual: list,
    tasks: list,
    nodes: list,
    carbon_per_node: Dict[int, np.ndarray],
    re_per_node: Dict[int, np.ndarray],
    T: int,
) -> Tuple[float, float, float]:
    """Evaluate a scheduling individual on three objectives.

    Parameters
    ----------
    individual:
        List of length N where ``individual[i]`` is the node index (0..J-1)
        assigned to task i.

    Returns
    -------
    ``(energy_wh, sla_violation, carbon_gco2)`` — all minimised.
    """
    J = len(nodes)
    energy = 0.0
    sla_viol = 0.0
    carbon = 0.0

    # Track CPU load per (node, slot) for feasibility check
    cpu_load: List[List[float]] = [[0.0] * T for _ in range(J)]

    for i, task in enumerate(tasks):
        j = int(individual[i])
        node = nodes[j]
        power = node.baseline_power_w + task.cpu_req * 10.0
        energy += power
        sla_viol += max(0.0, node.base_latency_ms - task.sla_latency_ms)
        ci = float(carbon_per_node[j][task.t_start])
        carbon += node.emission_factor * ci * power / 1000.0
        cpu_load[j][task.t_start] += task.cpu_req

    # Infeasibility penalty: any node–slot CPU overload
    infeasible = any(
        cpu_load[j][t] > nodes[j].cpu_capacity
        for j in range(J)
        for t in range(T)
    )
    if infeasible:
        energy += 1e6
        sla_viol += 1e6
        carbon += 1e6

    return energy, sla_viol, carbon


# ---------------------------------------------------------------------------
# NSGA-II scheduler
# ---------------------------------------------------------------------------

class NSGA2Scheduler:
    """Multi-objective NSGA-II scheduler matching Paper B's baseline.

    Minimises three objectives jointly — energy (Wh), SLA violation, and
    carbon footprint (gCO2) — and returns the full Pareto front together
    with the best single-objective solutions for convenience.
    """

    def __init__(
        self,
        population_size: int = 100,
        n_generations: int = 50,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.2,
        seed: int = 42,
    ) -> None:
        """Store hyper-parameters; no computation at construction time."""
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.seed = seed

    def schedule(
        self,
        tasks: list,
        nodes: list,
        carbon_per_node: Dict[int, np.ndarray],
        re_per_node: Dict[int, np.ndarray],
        T: int = 8,
    ) -> Dict:
        """Run NSGA-II and return the Pareto front with convenience pointers.

        Parameters
        ----------
        tasks, nodes, carbon_per_node, re_per_node, T:
            Same semantics as in :func:`milp.scheduler_milp.solve_milp`.

        Returns
        -------
        Dict with keys:
          * ``pareto_front`` — list of dicts with assignments and metrics
          * ``best_energy`` — Pareto solution with lowest energy_wh
          * ``best_sla`` — Pareto solution with lowest sla_violation
          * ``best_carbon`` — Pareto solution with lowest carbon_gco2
          * ``runtime_s`` — wall-clock runtime
          * ``n_generations`` — number of generations run
        """
        # Seed for reproducibility — DEAP uses Python's built-in random
        random.seed(self.seed)
        np.random.seed(self.seed)

        N = len(tasks)
        J = len(nodes)

        # ------------------------------------------------------------------
        # DEAP toolbox setup
        # ------------------------------------------------------------------
        toolbox = base.Toolbox()
        toolbox.register("attr_int", random.randint, 0, J - 1)
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_int,
            n=N,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        eval_fn = functools.partial(
            _evaluate,
            tasks=tasks,
            nodes=nodes,
            carbon_per_node=carbon_per_node,
            re_per_node=re_per_node,
            T=T,
        )
        toolbox.register("evaluate", eval_fn)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register(
            "mutate", tools.mutUniformInt, low=0, up=J - 1, indpb=0.1
        )
        toolbox.register("select", tools.selNSGA2)

        # ------------------------------------------------------------------
        # Initialise and evaluate population
        # ------------------------------------------------------------------
        pop = toolbox.population(n=self.population_size)
        fits = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fits):
            ind.fitness.values = fit

        # ------------------------------------------------------------------
        # NSGA-II main loop via eaMuPlusLambda
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        pop, _ = algorithms.eaMuPlusLambda(
            pop,
            toolbox,
            mu=self.population_size,
            lambda_=self.population_size,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=self.n_generations,
            stats=None,
            halloffame=None,
            verbose=False,
        )
        runtime_s = time.perf_counter() - t0

        # ------------------------------------------------------------------
        # Extract Pareto front
        # ------------------------------------------------------------------
        pareto = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

        pareto_results: List[Dict] = []
        for ind in pareto:
            energy, sla_viol, carbon = ind.fitness.values
            assignments = {tasks[i].task_id: int(ind[i]) for i in range(N)}
            pareto_results.append(
                {
                    "assignments": assignments,
                    "energy_wh": float(energy),
                    "sla_violation": float(sla_viol),
                    "carbon_gco2": float(carbon),
                }
            )

        best_energy = min(pareto_results, key=lambda x: x["energy_wh"])
        best_sla = min(pareto_results, key=lambda x: x["sla_violation"])
        best_carbon = min(pareto_results, key=lambda x: x["carbon_gco2"])

        return {
            "pareto_front": pareto_results,
            "best_energy": best_energy,
            "best_sla": best_sla,
            "best_carbon": best_carbon,
            "runtime_s": float(runtime_s),
            "n_generations": self.n_generations,
        }
