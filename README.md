# Carbon-Aware Resource Scheduling for Federated Cloud-Edge Systems

> Code, data, and reproducibility artefacts for the paper:
>
> **"Carbon-Aware Resource Scheduling for Federated Cloud-Edge Systems: A
> Multi-Regime Evaluation on Real Grid-Intensity Data"**
> Wassim Kribaa, Petri Välisuo, Mohammed Elmusrati
> *Future Generation Computer Systems* (under review, 2026)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

This repository accompanies an empirical study of carbon-aware scheduling
for federated cloud-edge systems. We evaluate a mixed-integer linear
programming (MILP) scheduler against two lightweight heuristic baselines
(Greedy Spatial, WMA Temporal) using real Electricity Maps carbon-intensity
data from 12 grid zones across the full 2025 calendar year.

The headline empirical findings are:

- **Carbon savings vs. carbon-blind baseline** range from 18% (US,
  medium-asymmetry) to 88% (Nordic, low-carbon) for the MILP and from
  51% to 93% for the WMA Temporal heuristic.
- **The temporal-forecasting premium** of explicit hourly carbon modelling
  over a daily-mean carbon model is only 0.8 percentage points on average
  under hard-SLA semantics.
- **MILP's distinguishing value is formal constraint satisfaction**: zero
  SLA violations under tight capacity (vs. 1.80/episode for Greedy), zero
  carbon-cap violations (vs. 10% for WMA at the operational cap level),
  and clean infeasibility refusal when problems are unsolvable.

The paper synthesises these findings into an operator-oriented decision
rule for scheduler selection that depends on whether hard guarantees are
required and on the carbon-intensity asymmetry of the deployment.

---

## Repository structure

```
.
├── src/                Scheduler implementations
│   ├── schedulers/       MILP variants, heuristics, Kubernetes scheduler
│   ├── data/             Carbon-trace loader and workload generator
│   └── models/           Infrastructure, battery, and renewable-energy models
├── experiments/        One script per paper section (§4.2–§4.8)
├── figures/            Figure-generation scripts and final PDFs
├── data/               Carbon-intensity trace and infrastructure parameters
├── results/paper/      Canonical CSV outputs backing the paper's numbers
├── tests/              Unit and integration tests
├── requirements.txt    Pinned Python dependencies
├── LICENSE             MIT
└── CITATION.cff        Citation metadata (GitHub citation widget)
```

---

## Quick start

### Requirements

- Python 3.10 or newer
- [Gurobi Optimizer](https://www.gurobi.com/) 10.0+ (free academic licence is
  sufficient)
- Linux, macOS, or WSL2 (tested on Ubuntu 22.04 and macOS 14)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/fgcs-carbon-aware-scheduling.git
cd fgcs-carbon-aware-scheduling

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify Gurobi is accessible
python -c "import gurobipy; print(gurobipy.gurobi.version())"
```

### Run a single experiment

```bash
# §4.2 — Multi-regime carbon savings (Table 3, Figure 3)
python experiments/01_savings_by_regime.py

# Output:
#   results/paper/savings_by_regime.csv
#   figures/fig_savings_by_region.pdf
```

### Reproduce all paper results

```bash
# This runs all 7 experiments and regenerates all 8 figures.
# Expected total runtime: ~45-90 minutes on a modern workstation.
bash scripts/reproduce_paper.sh
```

After completion, all CSV outputs in `results/paper/` and PDF figures in
`figures/` should match the values and visuals in the submitted paper.

---

## Reproducing each paper experiment

| Paper section | Experiment | Script | Output |
|---|---|---|---|
| §4.2 | Multi-regime carbon savings | `experiments/01_savings_by_regime.py` | Table 3, Figure 3 |
| §4.3 | Temporal-forecasting premium | `experiments/02_temporal_premium.py` | Figure 4 |
| §4.4 | Constraint compliance | `experiments/03_formal_guarantees.py` | Table 4, Figure 5 |
| §4.5 | Weight sensitivity (β sweep) | `experiments/04_weight_sensitivity.py` | Figure 6 |
| §4.6 | Solver scalability | `experiments/05_scalability.py` | Table 5, Figure 7 |
| §4.7 | Battery + renewable coupling | `experiments/06_battery_re.py` | Figure 8 |
| §4.8 | Kubernetes runtime validation | `experiments/07_kubernetes_runtime.py` | Table 6 |

Each script accepts a `--seed` argument (default: 10 seeds, matching the
paper's evaluation) and a `--output-dir` argument (default: `results/paper/`).

### Headline numbers to expect

After running the full reproduction script, the following key values should
appear in the output CSVs (modulo small seed-dependent variation):

- **MILP Dynamic savings**: 51.1% (Set A), 87.7% (Set B), 17.6% (Set C)
- **WMA Temporal savings**: 89.4% (Set A), 93.1% (Set B), 51.1% (Set C)
- **Temporal-forecasting premium**: 0.8 pp on average across 27 configs
- **Greedy SLA violations**: 1.80/episode at the Tight capacity regime
  (FR × 0.15)
- **WMA carbon-cap violations**: 10% of episodes at the 100% cap level
- **MILP timeout rate**: 6.7% at N=150 (mean optimality gap 0.10%)
- **Battery+RE utilisation gap**: 17 pp (battery-aware vs. battery-blind)
- **Kubernetes runtime reduction**: 2.9% (carbon-aware vs. round-robin)

---

## Data

### Carbon-intensity trace

The evaluation uses real Electricity Maps data for the full 2025 calendar
year (8,760 hourly observations per zone) across 12 grid zones:

| Asymmetry regime | Spread | Zones |
|---|---|---|
| A — High (European) | 36:1 | FR, GB, DE, PL |
| B — Low (Nordic) | 7:1 | SE, NO-NO4, DK-DK1, FI |
| C — Medium (US) | 3:1 | US-CAL-CISO, US-NY-NYIS, US-TEX-ERCO, US-MIDW-MISO |

The trace files are provided in `data/carbon_intensity_2025/` as one CSV
per zone. Each row contains a UTC timestamp and a carbon-intensity value
in gCO₂/kWh.

> **Note on licensing**: Electricity Maps data is provided under the terms
> of the Electricity Maps API licence. The redistribution in this
> repository is for academic reproducibility of the published results only.
> For commercial use or fresh data, please obtain access directly from
> [electricitymaps.com](https://www.electricitymaps.com/).

### Workloads

Workloads are generated from statistical properties of Google Cluster
Traces v3. The repository contains the generator, not the raw GCT v3
traces. We evaluate three task counts (N ∈ {50, 100, 150}) and three
seasonal windows (Winter: mid-January 2025; Summer: mid-July 2025;
Shoulder: early April 2025), with 10 seeded trials per configuration.

### Infrastructure

The infrastructure model contains 11 heterogeneous nodes (4 edge, 3
far-edge, 4 cloud) mapped to the four zones of each evaluated regime.
Node capacities, baseline power values, and SLA targets are defined in
`data/infrastructure.json`.

---

## Kubernetes runtime experiment

Section 4.8 of the paper reports a runtime validation on a 10-node
Kubernetes cluster spanning three Electricity Maps zones. The cluster
setup, scheduler deployment manifest, and workload submission script are
in `experiments/07_kubernetes_runtime.py` and `src/schedulers/kubernetes/`.

If you do not have access to a multi-zone Kubernetes cluster, the script
can run against a local [kind](https://kind.sigs.k8s.io/) cluster with
zone labels injected for the three target zones. See
`experiments/07_kubernetes_runtime.py` header for instructions.

---

## Companion paper

The MILP formulation used in this paper builds on our companion work:

> W. Kribaa, M. Bagaa, I. Afolabi, A. Ksentini, P. Välisuo, M. Elmusrati,
> *A Comprehensive Model for Energy-Efficient and SLA-Aware Task Scheduling
> in the Cloud–Edge Computing Continuum*. Manuscript under review at IEEE
> Transactions on Sustainable Computing.

The companion paper introduces the MILP framework and evaluates it against
NSGA-II and naive greedy baselines using synthetic workloads derived from
Google Cluster Traces v3. The present paper extends that work by applying
the MILP to real Electricity Maps data across 12 grid zones and three
asymmetry regimes, and by comparing it against stronger heuristic baselines
(Greedy Spatial and WMA Temporal).

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{kribaa2026carbonaware,
  title   = {Carbon-Aware Resource Scheduling for Federated Cloud-Edge
             Systems: A Multi-Regime Evaluation on Real Grid-Intensity Data},
  author  = {Kribaa, Wassim and V{\"a}lisuo, Petri and Elmusrati, Mohammed},
  journal = {Future Generation Computer Systems},
  year    = {2026},
  note    = {Under review}
}
```

This entry will be updated with volume, issue, and DOI once the paper is
accepted.

---

## Licence

This code is released under the [MIT License](LICENSE).

The Electricity Maps carbon-intensity data redistributed in `data/` is
subject to Electricity Maps' own terms of use.

---

## Contact

For questions about the code or paper:

- **Wassim Kribaa** — `x6542082@student.uwasa.fi` (primary contact)
- **Petri Välisuo** — `petri.valisuo@uwasa.fi`
- **Mohammed Elmusrati** — `moel@uwasa.fi` (corresponding author)

University of Vaasa, Wolffintie 34, 65200 Vaasa, Finland

---