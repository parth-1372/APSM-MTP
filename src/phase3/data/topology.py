import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd

from phase3.config import TopologyConfig


@dataclass
class EdgeTopology:
    """Deterministic undirected sparse edge mesh with link RTT & bandwidth attributes."""

    n_nodes: int
    k_edge_connectivity: int
    neighbors: List[List[int]]  # Per-node neighbor list sorted ascending by RTT, tiebreak ID
    adj_matrix: np.ndarray  # (N, N) bool
    rtt_s: np.ndarray  # (N, N) float32 seconds (0 on diagonal, inf if not connected)
    bw_mbps: np.ndarray  # (N, N) float32 Mbps (0 if not connected)
    coords_lon_lat: np.ndarray  # (N, 2) float64
    max_degree: int

    def neighbor_slots(self, node_id: int) -> List[int]:
        return self.neighbors[node_id]


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r_km = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r_km * math.asin(min(1.0, math.sqrt(max(0.0, a))))


def build_edge_topology(porto_cells_parquet: Path, cfg: TopologyConfig) -> EdgeTopology:
    """
    Generate an undirected k-edge-connected edge mesh from real tower coordinates
    (`porto_cells.parquet`) with realistic edge propagation RTTs (`[5, 40] ms`)
    and heterogeneous bandwidths (`[50, 200] Mbps`).
    """
    rng = np.random.default_rng(cfg.topology_seed)
    cells = pd.read_parquet(porto_cells_parquet)
    n = cfg.n_nodes
    k = min(cfg.k_edge_connectivity, max(1, n - 1))

    sample_idx = rng.choice(len(cells), size=n, replace=False)
    coords = cells.iloc[sample_idx][["lon", "lat"]].to_numpy(dtype=np.float64)

    dist_mat = np.zeros((n, n), dtype=np.float64)
    avail_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            d = _haversine_km(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
            dist_mat[i, j] = dist_mat[j, i] = d
            avail_edges.append((i, j, d))

    g = nx.Graph()
    g.add_nodes_from(range(n))
    edges = nx.k_edge_augmentation(g, k, avail=avail_edges)
    g.add_edges_from(edges)

    edge_dists = [dist_mat[u, v] for u, v in g.edges()]
    d_min = min(edge_dists) if edge_dists else 0.0
    d_max = max(edge_dists) if edge_dists else 1.0
    d_span = max(1e-6, d_max - d_min)

    adj = np.zeros((n, n), dtype=bool)
    rtt = np.full((n, n), np.inf, dtype=np.float32)
    bw = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(rtt, 0.0)

    for u, v in g.edges():
        adj[u, v] = adj[v, u] = True
        frac = (dist_mat[u, v] - d_min) / d_span
        link_rtt = cfg.rtt_min_s + frac * (cfg.rtt_max_s - cfg.rtt_min_s)
        # Heterogeneous edge link bandwidth inversely slightly skewed vs distance + jitter
        link_bw = float(rng.uniform(cfg.bw_min_mbps, cfg.bw_max_mbps))
        rtt[u, v] = rtt[v, u] = float(link_rtt)
        bw[u, v] = bw[v, u] = link_bw

    neighbors: List[List[int]] = []
    for i in range(n):
        nbrs = list(g.neighbors(i))
        nbrs.sort(key=lambda j: (float(rtt[i, j]), j))
        neighbors.append(nbrs)

    max_deg = max(len(x) for x in neighbors) if neighbors else 0
    return EdgeTopology(
        n_nodes=n,
        k_edge_connectivity=k,
        neighbors=neighbors,
        adj_matrix=adj,
        rtt_s=rtt,
        bw_mbps=bw,
        coords_lon_lat=coords,
        max_degree=max_deg,
    )


def summarize_topology(topo: EdgeTopology) -> Dict[str, object]:
    degrees = [len(n) for n in topo.neighbors]
    connected_rtts = topo.rtt_s[topo.adj_matrix]
    connected_bws = topo.bw_mbps[topo.adj_matrix]
    return {
        "n_nodes": topo.n_nodes,
        "k_edge_connectivity": topo.k_edge_connectivity,
        "n_undirected_edges": int(topo.adj_matrix.sum() // 2),
        "deg_min_mean_max": [min(degrees), float(np.mean(degrees)), max(degrees)],
        "is_fully_connected": bool(all(d == topo.n_nodes - 1 for d in degrees)),
        "rtt_ms_min_max": [
            float(connected_rtts.min() * 1000.0),
            float(connected_rtts.max() * 1000.0),
        ],
        "bw_mbps_min_max": [float(connected_bws.min()), float(connected_bws.max())],
    }

