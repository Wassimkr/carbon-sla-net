"""Tests for env/cloud_edge_env.py — Phase 1 Gymnasium environment."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from env.cloud_edge_env import CloudEdgeEnv
from env.workload_generator import Task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh(N: int = 100, seed: int = 42) -> CloudEdgeEnv:
    """Return a freshly reset CloudEdgeEnv."""
    env = CloudEdgeEnv(N=N, T=8, seed=seed)
    env.reset()
    return env


def _run_episode(env: CloudEdgeEnv, action: int = 0) -> dict:
    """Step through the full episode with a fixed action; return final info."""
    info: dict = {}
    while True:
        _, _, terminated, _, info = env.step(action)
        if terminated:
            return info


def _run_episode_masked(env: CloudEdgeEnv) -> tuple[dict, float]:
    """Run a full episode choosing the first feasible node each step."""
    total_reward = 0.0
    info: dict = {}
    all_obs: list[np.ndarray] = []
    obs, _ = env.reset()
    all_obs.append(obs)
    while True:
        masks = env.action_masks()
        action = int(np.argmax(masks))
        obs, reward, terminated, _, info = env.step(action)
        all_obs.append(obs)
        total_reward += reward
        if terminated:
            return info, total_reward, all_obs


# ---------------------------------------------------------------------------
# 1–2. Observation / action space shape
# ---------------------------------------------------------------------------

def test_obs_space_shape():
    assert CloudEdgeEnv().observation_space.shape == (74,)


def test_action_space_size():
    assert CloudEdgeEnv().action_space.n == 11


# ---------------------------------------------------------------------------
# 3. reset() obs shape and dtype
# ---------------------------------------------------------------------------

def test_reset_obs_shape_dtype():
    env = CloudEdgeEnv(N=100, T=8, seed=42)
    obs, info = env.reset()
    assert obs.shape == (74,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# 4. All obs in [0, 1] after reset
# ---------------------------------------------------------------------------

def test_obs_in_range_after_reset():
    env = CloudEdgeEnv(N=100, T=8, seed=42)
    obs, _ = env.reset()
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0), (
        f"obs out of [0,1]: min={obs.min():.4f} max={obs.max():.4f}"
    )


# ---------------------------------------------------------------------------
# 5. All obs in [0, 1] after 5 valid steps
# ---------------------------------------------------------------------------

def test_obs_in_range_after_steps():
    env = _fresh()
    for _ in range(5):
        masks = env.action_masks()
        action = int(np.argmax(masks))
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            break
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0), (
            f"obs out of [0,1]: min={obs.min():.4f} max={obs.max():.4f}"
        )


# ---------------------------------------------------------------------------
# 6. Terminal obs is all zeros
# ---------------------------------------------------------------------------

def test_obs_zeros_at_termination():
    env = CloudEdgeEnv(N=10, T=8, seed=42)
    env.reset()
    terminal_obs = None
    for _ in range(10):
        obs, _, terminated, _, _ = env.step(0)
        if terminated:
            terminal_obs = obs
            break
    assert terminal_obs is not None
    np.testing.assert_array_equal(terminal_obs, np.zeros(74, dtype=np.float32))


# ---------------------------------------------------------------------------
# 7. action_masks() shape and dtype
# ---------------------------------------------------------------------------

def test_action_masks_shape_dtype():
    env = _fresh()
    masks = env.action_masks()
    assert masks.shape == (11,)
    assert masks.dtype == bool


# ---------------------------------------------------------------------------
# 8. Overfilled node is masked
# ---------------------------------------------------------------------------

def test_action_mask_blocks_exceeded_cpu():
    env = _fresh()
    env.node_cpu_used[0] = env.nodes[0].cpu_capacity + 1.0
    masks = env.action_masks()
    assert not masks[0], "Node 0 should be masked when cpu_used exceeds capacity"


# ---------------------------------------------------------------------------
# 9. At least one unmasked node at episode start
# ---------------------------------------------------------------------------

def test_at_least_one_unmasked_at_start():
    env = _fresh()
    masks = env.action_masks()
    assert np.any(masks), "At least one node must be feasible at episode start"


# ---------------------------------------------------------------------------
# 10. Safety fallback returns all True
# ---------------------------------------------------------------------------

def test_safety_fallback_all_true():
    env = _fresh()
    for j, node in enumerate(env.nodes):
        env.node_cpu_used[j] = node.cpu_capacity + 1.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        masks = env.action_masks()
    assert np.all(masks), "Safety fallback must return all-True mask"
    assert any(issubclass(w.category, RuntimeWarning) for w in caught), (
        "Safety fallback should emit a RuntimeWarning"
    )


# ---------------------------------------------------------------------------
# 11. Full episode terminates after N steps
# ---------------------------------------------------------------------------

def test_full_episode_terminates():
    env = CloudEdgeEnv(N=10, T=8, seed=42)
    env.reset()
    terminated = False
    for _ in range(10):
        _, _, terminated, _, _ = env.step(0)
        if terminated:
            break
    assert terminated


# ---------------------------------------------------------------------------
# 12. Non-terminal reward is exactly 0.0
# ---------------------------------------------------------------------------

def test_non_terminal_reward_zero():
    env = CloudEdgeEnv(N=10, T=8, seed=42)
    env.reset()
    for i in range(9):
        _, reward, terminated, _, _ = env.step(0)
        assert not terminated
        assert reward == 0.0, f"Step {i+1}: expected reward=0, got {reward}"


# ---------------------------------------------------------------------------
# 13. Terminal reward is a finite negative float
# ---------------------------------------------------------------------------

def test_terminal_reward_finite_negative():
    env = CloudEdgeEnv(N=10, T=8, seed=42)
    env.reset()
    reward = 0.0
    for _ in range(10):
        _, reward, terminated, _, _ = env.step(0)
        if terminated:
            break
    assert np.isfinite(reward), f"Terminal reward {reward} is not finite"
    assert reward < 0.0, f"Terminal reward {reward} should be negative"


# ---------------------------------------------------------------------------
# 14. info dict at termination contains required keys
# ---------------------------------------------------------------------------

def test_info_keys_at_termination():
    env = CloudEdgeEnv(N=10, T=8, seed=42)
    env.reset()
    info = _run_episode(env, action=0)
    required = {"energy_wh", "carbon_gco2", "sla_violation", "re_used_wh"}
    assert required.issubset(info.keys())


# ---------------------------------------------------------------------------
# 15. energy_wh > 0, carbon_gco2 >= 0 after full episode
# ---------------------------------------------------------------------------

def test_energy_carbon_after_episode():
    env = CloudEdgeEnv(N=10, T=8, seed=42)
    env.reset()
    info = _run_episode(env, action=0)
    assert info["energy_wh"] > 0.0, "energy_wh should be positive"
    assert info["carbon_gco2"] >= 0.0, "carbon_gco2 must be non-negative"


# ---------------------------------------------------------------------------
# 16–17. ValueError for out-of-range actions
# ---------------------------------------------------------------------------

def test_invalid_action_negative():
    env = _fresh()
    with pytest.raises(ValueError):
        env.step(-1)


def test_invalid_action_too_large():
    env = _fresh()
    with pytest.raises(ValueError):
        env.step(11)


# ---------------------------------------------------------------------------
# 18. episode_count increments each reset
# ---------------------------------------------------------------------------

def test_episode_count_increments():
    env = CloudEdgeEnv(N=100, T=8, seed=42)
    assert env.episode_count == 0
    env.reset()
    assert env.episode_count == 1
    env.reset()
    assert env.episode_count == 2


# ---------------------------------------------------------------------------
# 19. Same seed → identical first obs (two fresh envs)
# ---------------------------------------------------------------------------

def test_same_seed_identical_first_obs():
    env1 = CloudEdgeEnv(N=100, T=8, seed=42)
    obs1, _ = env1.reset()
    env2 = CloudEdgeEnv(N=100, T=8, seed=42)
    obs2, _ = env2.reset()
    np.testing.assert_array_equal(obs1, obs2)


# ---------------------------------------------------------------------------
# 20. Different seeds → different first obs
# ---------------------------------------------------------------------------

def test_different_seeds_different_obs():
    env1 = CloudEdgeEnv(N=100, T=8, seed=42)
    obs1, _ = env1.reset()
    env2 = CloudEdgeEnv(N=100, T=8, seed=99)
    obs2, _ = env2.reset()
    assert not np.array_equal(obs1, obs2), "Different seeds should produce different observations"


# ---------------------------------------------------------------------------
# 21. Same seed → same total energy across two independent runs
# ---------------------------------------------------------------------------

def test_same_seed_same_energy():
    def run(seed: int) -> float:
        env = CloudEdgeEnv(N=10, T=8, seed=seed)
        env.reset()
        return _run_episode(env, action=0)["energy_wh"]

    assert run(42) == run(42)


# ---------------------------------------------------------------------------
# 22. _compute_sla_violation() == 0.0 for low-load edge node
# ---------------------------------------------------------------------------

def test_sla_violation_zero_on_empty_edge_node():
    env = CloudEdgeEnv(N=100, T=8, seed=42)
    env.reset()

    # Craft a low-priority task with relaxed SLAs — edge node 0 has base_latency=12ms
    task = Task(
        task_id=999, cpu_req=0.1, mem_req=0.5, priority=2,
        sla_latency_ms=84.0, sla_reliability=0.92,
        sla_throughput_rps=70.0, t_start=0, t_deadline=1,
    )
    # Simulate step: increment cpu_used by the task's requirement
    env.node_cpu_used[0] = task.cpu_req
    env.node_mem_used[0] = task.mem_req

    viol = env._compute_sla_violation(task, 0)
    assert viol == pytest.approx(0.0, abs=1e-6), (
        f"Expected 0 SLA violation on empty edge node, got {viol}"
    )


# ---------------------------------------------------------------------------
# 23. Battery SoC changes after a step on an edge node
# ---------------------------------------------------------------------------

def test_battery_soc_changes_on_edge_step():
    env = _fresh()
    initial_soc = env.batteries[0].soc_fraction
    env.step(0)  # assign first task to edge node 0
    assert env.batteries[0].soc_fraction != initial_soc, (
        f"Edge node SoC should change after a step "
        f"(was {initial_soc:.4f}, still {env.batteries[0].soc_fraction:.4f})"
    )


# ---------------------------------------------------------------------------
# 24. Cloud node battery SoC stays 0.0
# ---------------------------------------------------------------------------

def test_cloud_node_battery_stays_zero():
    env = _fresh()
    assert env.batteries[7].soc_fraction == pytest.approx(0.0)
    env.step(7)  # force assignment to cloud node 7 (ignores masks)
    assert env.batteries[7].soc_fraction == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 25. check_env passes (SB3 compatibility)
# ---------------------------------------------------------------------------

def test_check_env_sb3_compatibility():
    from stable_baselines3.common.env_checker import check_env

    env = CloudEdgeEnv(N=100, T=8, seed=42)
    check_env(env, warn=True)  # raises on failure
