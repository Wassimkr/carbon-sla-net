"""Tests for env/renewable_model.py."""

import numpy as np
import pytest

from env.renewable_model import RenewableModel, MAX_RE_PER_NODE

T = 8
_CLOUD_IDS = {7, 8, 9, 10}
_EDGE_IDS = {0, 1, 2, 3}


@pytest.fixture(scope="module")
def sunny_ep0():
    return RenewableModel(condition="sunny", T=T, seed=0).sample_episode(0)


# 1. sample_episode returns dict with 11 keys
def test_episode_has_11_nodes(sunny_ep0):
    assert len(sunny_ep0) == 11


# 2. Each value is a numpy array of shape (T,)
def test_array_shapes(sunny_ep0):
    for node_id, arr in sunny_ep0.items():
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (T,), f"Node {node_id}: shape {arr.shape}"


# 3. Cloud nodes always return 0.0 for all conditions
@pytest.mark.parametrize("condition", ["sunny", "cloudy", "no_re"])
def test_cloud_nodes_zero(condition):
    model = RenewableModel(condition=condition, T=T, seed=0)
    ep = model.sample_episode(0)
    for node_id in _CLOUD_IDS:
        assert np.all(ep[node_id] == 0.0), (
            f"Cloud node {node_id} should be 0 for condition={condition}"
        )


# 4. no_re: all values are 0.0
def test_no_re_all_zero():
    model = RenewableModel(condition="no_re", T=T, seed=0)
    ep = model.sample_episode(0)
    for node_id, arr in ep.items():
        assert np.all(arr == 0.0), f"no_re: node {node_id} has non-zero values"


# 5. sunny: non-cloud nodes in [85.0, 100.0]
def test_sunny_range():
    model = RenewableModel(condition="sunny", T=T, seed=0)
    ep = model.sample_episode(0)
    for node_id in range(11):
        if node_id not in _CLOUD_IDS:
            arr = ep[node_id]
            lo = 0.85 * MAX_RE_PER_NODE
            hi = 1.00 * MAX_RE_PER_NODE
            assert np.all(arr >= lo) and np.all(arr <= hi), (
                f"sunny node {node_id}: values {arr} out of [{lo}, {hi}]"
            )


# 6. cloudy: non-cloud nodes in [40.0, 70.0]
def test_cloudy_range():
    model = RenewableModel(condition="cloudy", T=T, seed=0)
    ep = model.sample_episode(0)
    for node_id in range(11):
        if node_id not in _CLOUD_IDS:
            arr = ep[node_id]
            lo = 0.40 * MAX_RE_PER_NODE
            hi = 0.70 * MAX_RE_PER_NODE
            assert np.all(arr >= lo) and np.all(arr <= hi), (
                f"cloudy node {node_id}: values {arr} out of [{lo}, {hi}]"
            )


# 7. Same episode_idx always produces the same output
def test_determinism():
    model = RenewableModel(condition="sunny", T=T, seed=7)
    ep_a = model.sample_episode(5)
    ep_b = model.sample_episode(5)
    for node_id in range(11):
        np.testing.assert_array_equal(ep_a[node_id], ep_b[node_id])


# 8. Different episode_idx produces different output for non-zero conditions
def test_different_episodes_differ():
    model = RenewableModel(condition="sunny", T=T, seed=0)
    ep0 = model.sample_episode(0)
    ep1 = model.sample_episode(1)
    differ = any(
        not np.array_equal(ep0[nid], ep1[nid])
        for nid in range(11)
        if nid not in _CLOUD_IDS
    )
    assert differ, "Episodes 0 and 1 returned identical RE for every non-cloud node"
