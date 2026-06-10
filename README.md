# Carbon-SLA-Net — Phase 0: Data Layer

Deep reinforcement learning scheduler for a federated cloud-edge computing continuum that jointly optimises energy, carbon emissions, SLA compliance, renewable energy utilisation, and battery storage.

## Phase 0 scope

Pure Python / NumPy data layer.  No ML libraries.  No Gymnasium environments.

| Module | Description |
|--------|-------------|
| `env/infrastructure.py` | Fixed 11-node topology (4 edge, 3 far-edge, 4 cloud) from Paper B |
| `env/workload_generator.py` | Synthetic tasks sampled from Google Cluster Traces v3 distributions |
| `env/carbon_trace.py` | Hourly carbon intensity per node (real CSV or synthetic fallback) |
| `env/renewable_model.py` | On-site renewable energy model (sunny / cloudy / no_re) |
| `env/battery_model.py` | Battery charge/discharge dynamics — Paper B Equations 3–5 |

## Quick start

```bash
pip install -r requirements.txt
python verify_phase0.py   # smoke-test all modules
pytest tests/ -v          # full test suite
```

## Data

Place Electricity Maps CSV exports as `data/electricity_maps/{ZONE}_carbon_intensity.csv`
(columns: `timestamp`, `carbon_intensity_avg`).  If absent, deterministic synthetic
traces are used automatically.

Supported zones: `FI`, `SE`, `NO`, `DE`, `FR`, `ES`, `PL`.
