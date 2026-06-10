"""Infrastructure definition for Carbon-SLA-Net: fixed 11-node federated cloud-edge topology.

Topology is taken directly from Paper B's MILP model and must not be randomized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Node:
    """A single compute node in the federated cloud-edge continuum.

    Captures all per-node parameters used by Paper B's energy, carbon,
    latency, and reliability models.
    """

    node_id: int
    tier: str              # 'edge' | 'far_edge' | 'cloud'
    cpu_capacity: float    # normalized CPU units
    mem_capacity: float    # normalized memory units
    baseline_power_w: float   # idle power draw (W)
    max_power_w: float        # peak power draw (W)
    emission_factor: float    # kgCO2 per kWh (static, geographic)
    battery_capacity_wh: float  # S_max_j (0.0 for cloud nodes)
    base_latency_ms: float    # tier-representative round-trip latency (ms)
    contention_coeff: float   # alpha_j: latency growth under load
    reliability_base: float   # R_base_ij
    reliability_degrad: float # beta_j: reliability drop rate above threshold
    utilization_thresh: float # theta_j: threshold above which reliability degrades


def build_infrastructure() -> List[Node]:
    """Return the fixed 11-node topology matching Paper B's infrastructure.

    Returns a deterministic list: 4 edge nodes (ids 0–3),
    3 far-edge nodes (ids 4–6), 4 cloud nodes (ids 7–10).
    """
    nodes: List[Node] = []

    # --- Edge nodes (ids 0–3) ---
    _edge_latencies = [12.0, 13.0, 14.0, 15.0]
    for i in range(4):
        nodes.append(Node(
            node_id=i,
            tier="edge",
            cpu_capacity=25.0,
            mem_capacity=32.0,
            baseline_power_w=50.0,
            max_power_w=120.0,
            emission_factor=0.05,
            battery_capacity_wh=500.0,
            base_latency_ms=_edge_latencies[i],
            contention_coeff=0.3,
            reliability_base=0.99,
            reliability_degrad=0.05,
            utilization_thresh=0.70,
        ))

    # --- Far-edge nodes (ids 4–6) ---
    _far_latencies = [33.0, 36.0, 39.0]
    for i in range(3):
        nodes.append(Node(
            node_id=4 + i,
            tier="far_edge",
            cpu_capacity=60.0,
            mem_capacity=64.0,
            baseline_power_w=100.0,
            max_power_w=250.0,
            emission_factor=0.20,
            battery_capacity_wh=200.0,
            base_latency_ms=_far_latencies[i],
            contention_coeff=0.2,
            reliability_base=0.97,
            reliability_degrad=0.03,
            utilization_thresh=0.75,
        ))

    # --- Cloud nodes (ids 7–10) ---
    _cloud_emission = [0.15, 0.30, 0.45, 0.60]
    _cloud_latencies = [85.0, 90.0, 100.0, 110.0]
    for i in range(4):
        nodes.append(Node(
            node_id=7 + i,
            tier="cloud",
            cpu_capacity=150.0,
            mem_capacity=256.0,
            baseline_power_w=200.0,
            max_power_w=600.0,
            emission_factor=_cloud_emission[i],
            battery_capacity_wh=0.0,
            base_latency_ms=_cloud_latencies[i],
            contention_coeff=0.1,
            reliability_base=0.999,
            reliability_degrad=0.01,
            utilization_thresh=0.85,
        ))

    return nodes


def get_node_by_id(nodes: List[Node], node_id: int) -> Node:
    """Return the Node whose node_id matches the given value.

    Raises ValueError if no matching node is found.
    """
    for node in nodes:
        if node.node_id == node_id:
            return node
    raise ValueError(f"No node with node_id={node_id}")


def get_nodes_by_tier(nodes: List[Node], tier: str) -> List[Node]:
    """Return all nodes whose tier matches the given string."""
    return [n for n in nodes if n.tier == tier]


def node_summary(node: Node) -> str:
    """Return a single-line debug string describing the node."""
    return (
        f"Node(id={node.node_id:2d}, tier={node.tier:8s}, "
        f"cpu={node.cpu_capacity:5.1f}, mem={node.mem_capacity:5.1f}, "
        f"batt={node.battery_capacity_wh:5.1f}Wh, "
        f"lat={node.base_latency_ms:6.1f}ms, "
        f"ef={node.emission_factor:.2f})"
    )
