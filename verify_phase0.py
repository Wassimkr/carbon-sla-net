"""Phase 0 verification script for Carbon-SLA-Net.

Run with:  python verify_phase0.py

Exercises every data-layer module and prints a clean summary.
No tracebacks should occur.
"""

import numpy as np

from env.infrastructure import build_infrastructure, node_summary
from env.workload_generator import WorkloadGenerator
from env.carbon_trace import CarbonTraceLoader, NODE_TO_ZONE
from env.renewable_model import RenewableModel
from env.battery_model import BatteryModel


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# 1. Infrastructure
# ---------------------------------------------------------------------------
section("1. Infrastructure (11-node topology)")
nodes = build_infrastructure()
header = f"{'id':>3}  {'tier':10}  {'cpu':>5}  {'mem':>5}  {'batt_wh':>8}  {'lat_ms':>7}  {'ef':>6}"
print(header)
print("-" * len(header))
for n in nodes:
    print(
        f"{n.node_id:>3}  {n.tier:10}  {n.cpu_capacity:>5.0f}  "
        f"{n.mem_capacity:>5.0f}  {n.battery_capacity_wh:>8.1f}  "
        f"{n.base_latency_ms:>7.1f}  {n.emission_factor:>6.2f}"
    )
print(f"\nTotal nodes: {len(nodes)}")


# ---------------------------------------------------------------------------
# 2. Workload generator
# ---------------------------------------------------------------------------
section("2. Workload generator (100 tasks, seed=42, uniform)")
gen = WorkloadGenerator(seed=42)
tasks = gen.sample_tasks(N=100, T=8, pattern="uniform")

cpu_vals = np.array([t.cpu_req for t in tasks])
print(f"  cpu_req  : min={cpu_vals.min():.3f}  max={cpu_vals.max():.3f}  mean={cpu_vals.mean():.3f}")

t_start_dist = {slot: 0 for slot in range(8)}
for t in tasks:
    t_start_dist[t.t_start] += 1
print("  t_start distribution across T=8 slots:")
for slot, count in t_start_dist.items():
    bar = "#" * count
    print(f"    slot {slot}: {count:3d}  {bar}")


# ---------------------------------------------------------------------------
# 3. Carbon traces
# ---------------------------------------------------------------------------
section("3. Carbon traces (episode 0)")
loader = CarbonTraceLoader(data_dir="data/electricity_maps", T=8)
ep0 = loader.sample_episode(0)

# Group by zone for compact output
zone_stats: dict = {}
for node_id, arr in ep0.items():
    zone = NODE_TO_ZONE[node_id]
    if zone not in zone_stats:
        zone_stats[zone] = []
    zone_stats[zone].append(arr)

print(f"  {'zone':>4}  {'min_ci':>8}  {'max_ci':>8}  (gCO2eq/kWh)")
print("  " + "-" * 30)
for zone in sorted(zone_stats):
    all_vals = np.concatenate(zone_stats[zone])
    print(f"  {zone:>4}  {all_vals.min():>8.1f}  {all_vals.max():>8.1f}")


# ---------------------------------------------------------------------------
# 4. Renewable model
# ---------------------------------------------------------------------------
section("4. Renewable model (edge nodes 0–3)")
edge_ids = [0, 1, 2, 3]
print(f"  {'condition':8}  {'mean RE (Wh)':>14}")
print("  " + "-" * 25)
for condition in ["sunny", "cloudy", "no_re"]:
    model = RenewableModel(condition=condition, T=8, seed=0)
    ep = model.sample_episode(0)
    mean_re = np.mean([ep[nid].mean() for nid in edge_ids])
    print(f"  {condition:8}  {mean_re:>14.2f}")


# ---------------------------------------------------------------------------
# 5. Battery model (20-step simulation)
# ---------------------------------------------------------------------------
section("5. Battery simulation (500 Wh, 60% initial SoC, 20 steps)")
battery = BatteryModel(capacity_wh=500.0, initial_soc_fraction=0.6)
rng = np.random.default_rng(0)
print(
    f"  {'step':>4}  {'power_W':>8}  {'RE_Wh':>7}  "
    f"{'charge_Wh':>10}  {'disch_Wh':>9}  {'SoC':>6}"
)
print("  " + "-" * 57)
for step in range(1, 21):
    power = rng.uniform(0.0, 150.0)
    re = rng.uniform(0.0, 200.0)
    r = battery.step(power, re)
    print(
        f"  {step:>4}  {power:>8.1f}  {re:>7.1f}  "
        f"{r['b_charge_wh']:>10.2f}  {r['b_discharge_wh']:>9.2f}  "
        f"{r['soc_fraction']:>6.3f}"
    )


# ---------------------------------------------------------------------------
print("\nPhase 0 complete. All modules verified.")
