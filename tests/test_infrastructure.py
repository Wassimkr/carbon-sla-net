"""Tests for env/infrastructure.py."""

import pytest

from env.infrastructure import (
    Node,
    build_infrastructure,
    get_node_by_id,
    get_nodes_by_tier,
    node_summary,
)


@pytest.fixture(scope="module")
def nodes():
    return build_infrastructure()


# 1. Exactly 11 nodes
def test_total_node_count(nodes):
    assert len(nodes) == 11


# 2. Correct tier counts
def test_tier_counts(nodes):
    tiers = [n.tier for n in nodes]
    assert tiers.count("edge") == 4
    assert tiers.count("far_edge") == 3
    assert tiers.count("cloud") == 4


# 3. Node ids are unique and form {0..10}
def test_node_ids_unique_and_complete(nodes):
    ids = {n.node_id for n in nodes}
    assert ids == set(range(11))


# 4. Battery: edge nodes > 0, cloud nodes == 0
def test_battery_capacity_by_tier(nodes):
    for n in nodes:
        if n.tier == "edge":
            assert n.battery_capacity_wh > 0, f"Edge node {n.node_id} should have battery"
        if n.tier == "cloud":
            assert n.battery_capacity_wh == 0.0, f"Cloud node {n.node_id} should have no battery"


# 5. All emission factors > 0
def test_emission_factors_positive(nodes):
    for n in nodes:
        assert n.emission_factor > 0, f"Node {n.node_id} emission_factor must be > 0"


# 6. All capacity values > 0
def test_capacities_positive(nodes):
    for n in nodes:
        assert n.cpu_capacity > 0, f"Node {n.node_id} cpu_capacity must be > 0"
        assert n.mem_capacity > 0, f"Node {n.node_id} mem_capacity must be > 0"


# 7. get_nodes_by_tier returns exactly 4 edge nodes
def test_get_nodes_by_tier_edge(nodes):
    edge_nodes = get_nodes_by_tier(nodes, "edge")
    assert len(edge_nodes) == 4
    assert all(n.tier == "edge" for n in edge_nodes)


# 8. get_node_by_id returns node with correct id
def test_get_node_by_id(nodes):
    node = get_node_by_id(nodes, 5)
    assert node.node_id == 5


# 9. build_infrastructure is deterministic
def test_determinism():
    nodes_a = build_infrastructure()
    nodes_b = build_infrastructure()
    assert len(nodes_a) == len(nodes_b)
    for a, b in zip(nodes_a, nodes_b):
        assert a.node_id == b.node_id
        assert a.tier == b.tier
        assert a.cpu_capacity == b.cpu_capacity
        assert a.emission_factor == b.emission_factor


# 10. No node has base_latency_ms <= 0
def test_latency_positive(nodes):
    for n in nodes:
        assert n.base_latency_ms > 0, f"Node {n.node_id} latency must be > 0"
