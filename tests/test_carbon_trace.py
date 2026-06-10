"""Tests for env/carbon_trace.py."""

import numpy as np
import pytest

from env.carbon_trace import CarbonTraceLoader

T = 8
_loader = CarbonTraceLoader(data_dir="data/electricity_maps", T=T)


@pytest.fixture(scope="module")
def episode0():
    return _loader.sample_episode(0)


# 1. sample_episode returns a dict with 11 keys
def test_episode_has_11_nodes(episode0):
    assert len(episode0) == 11


# 2. Each value is a numpy array of shape (T,)
def test_episode_array_shapes(episode0):
    for node_id, arr in episode0.items():
        assert isinstance(arr, np.ndarray), f"Node {node_id}: expected ndarray"
        assert arr.shape == (T,), f"Node {node_id}: shape {arr.shape} != ({T},)"


# 3. All carbon intensity values > 0
def test_carbon_values_positive(episode0):
    for node_id, arr in episode0.items():
        assert np.all(arr > 0), f"Node {node_id} has non-positive carbon values"


# 4. FI nodes (0,1) consistently lower than PL node (10) across 10 episodes
def test_fi_lower_than_pl():
    for ep in range(10):
        ep_data = _loader.sample_episode(ep)
        fi_max = max(ep_data[0].max(), ep_data[1].max())
        pl_min = ep_data[10].min()
        assert fi_max < pl_min, (
            f"Episode {ep}: FI max {fi_max:.1f} should be < PL min {pl_min:.1f}"
        )


# 5. Same episode_idx always returns the same values
def test_determinism():
    a = _loader.sample_episode(3)
    b = _loader.sample_episode(3)
    for node_id in range(11):
        np.testing.assert_array_equal(a[node_id], b[node_id])


# 6. Different episode_idx returns different values
def test_different_episodes_differ():
    ep0 = _loader.sample_episode(0)
    ep1 = _loader.sample_episode(1)
    # At least one node should differ between episodes
    differ = any(not np.array_equal(ep0[nid], ep1[nid]) for nid in range(11))
    assert differ, "Episodes 0 and 1 returned identical arrays for every node"


# 7. No NaN or inf in any episode
def test_no_nan_or_inf():
    for ep in range(5):
        ep_data = _loader.sample_episode(ep)
        for node_id, arr in ep_data.items():
            assert np.all(np.isfinite(arr)), f"Node {node_id} ep {ep}: contains NaN/inf"


# 8. Sliding window does not go out of bounds for episode_idx up to 500
def test_no_out_of_bounds_large_episode():
    for ep_idx in [0, 100, 200, 300, 400, 500]:
        ep_data = _loader.sample_episode(ep_idx)
        for node_id, arr in ep_data.items():
            assert arr.shape == (T,), f"episode_idx={ep_idx} node {node_id}: bad shape"
            assert np.all(np.isfinite(arr))
