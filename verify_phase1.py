"""Phase 1 verification script for Carbon-SLA-Net.

Run with:  python verify_phase1.py

Exercises CloudEdgeEnv end-to-end and confirms SB3 compatibility.
No tracebacks should occur.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.env_checker import check_env

from env.cloud_edge_env import CloudEdgeEnv


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# 1. Instantiate and print spaces
# ---------------------------------------------------------------------------
section("1. Environment spaces")
env = CloudEdgeEnv(N=100, T=8, seed=42)
print(f"  obs_dim   = {env.observation_space.shape[0]}")
print(f"  action_space = {env.action_space}")


# ---------------------------------------------------------------------------
# 2–4. One complete episode using first-feasible-node policy
# ---------------------------------------------------------------------------
section("2. Complete episode (first-feasible-node policy, seed=42)")
obs, _ = env.reset()
all_obs: list[np.ndarray] = [obs]
step_count = 0
total_reward = 0.0
info: dict = {}

while True:
    masks = env.action_masks()
    action = int(np.argmax(masks))   # first feasible node (deterministic)
    obs, reward, terminated, _, info = env.step(action)
    all_obs.append(obs)
    total_reward += reward
    step_count += 1
    if terminated:
        break

print(f"  Steps taken       : {step_count}")
print(f"  Terminal reward   : {total_reward:.6f}")
print(f"  energy_wh         : {info['energy_wh']:.2f}")
print(f"  carbon_gco2       : {info['carbon_gco2']:.4f}")
print(f"  sla_violation     : {info['sla_violation']:.4f}")
print(f"  re_used_wh        : {info['re_used_wh']:.2f}")


# ---------------------------------------------------------------------------
# 5. check_env
# ---------------------------------------------------------------------------
section("3. SB3 check_env")
env_check = CloudEdgeEnv(N=100, T=8, seed=42)
check_env(env_check, warn=True)
print("  check_env: PASSED (no exceptions raised)")


# ---------------------------------------------------------------------------
# 6. Three episodes with different seeds — rewards should all differ
# ---------------------------------------------------------------------------
section("4. Three episodes with different seeds")
rewards: list[float] = []
for seed in [42, 7, 123]:
    e = CloudEdgeEnv(N=100, T=8, seed=seed)
    e.reset()
    r = 0.0
    while True:
        masks = e.action_masks()
        action = int(np.argmax(masks))
        _, reward, terminated, _, _ = e.step(action)
        r += reward
        if terminated:
            break
    rewards.append(r)
    print(f"  seed={seed:3d}  reward={r:.6f}")

all_different = len(set(f"{r:.8f}" for r in rewards)) == len(rewards)
print(f"  All rewards differ: {all_different}")


# ---------------------------------------------------------------------------
# 7. obs min/max across all steps of one episode
# ---------------------------------------------------------------------------
section("5. Observation range across full episode")
obs_stack = np.stack(all_obs)
print(f"  obs min  = {obs_stack.min():.6f}  (should be >= 0.0)")
print(f"  obs max  = {obs_stack.max():.6f}  (should be <= 1.0)")
assert obs_stack.min() >= 0.0, "obs min < 0!"
assert obs_stack.max() <= 1.0, "obs max > 1!"
print("  Range check: PASSED")


# ---------------------------------------------------------------------------
print("\nPhase 1 complete. Environment verified and SB3-compatible.")
