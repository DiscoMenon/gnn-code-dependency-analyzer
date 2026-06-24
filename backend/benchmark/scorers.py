import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from gnn.model import CodeGNN
from gnn.scorer import score_nodes

def degree_heuristic_scores(G: nx.DiGraph, node_list: list[str]) -> list[float]:
    """Baseline 1: normalized in-degree as anomaly score."""
    import_graph = nx.DiGraph()
    for u, v, d in G.edges(data=True):
        if d.get("type") == "imports":
            import_graph.add_edge(u, v)

    max_in = max((int(import_graph.in_degree(n)) if import_graph.has_node(n) else 0 for n in node_list), default=1)
    if max_in == 0:
        max_in = 1

    return [(int(import_graph.in_degree(n)) if import_graph.has_node(n) else 0) / max_in for n in node_list]


def pagerank_scores(G: nx.DiGraph, node_list: list[str]) -> list[float]:
    """Baseline 2: PageRank score."""
    import_graph = nx.DiGraph()
    for u, v, d in G.edges(data=True):
        if d.get("type") == "imports":
            import_graph.add_edge(u, v)

    pr = nx.pagerank(import_graph, alpha=0.85) if import_graph.number_of_nodes() > 0 else {}
    max_pr = max(pr.values(), default=1.0)
    if max_pr == 0:
        max_pr = 1.0

    return [pr.get(n, 0.0) / max_pr for n in node_list]


def gnn_scores(
    model: CodeGNN,
    data: Data,
    node_list: list[str]
) -> list[float]:
    """Your GNN anomaly scorer."""
    scored = score_nodes(model, data, node_list)
    score_map = {s["node"]: s["anomaly_score"] for s in scored}
    return [score_map.get(n, 0.0) for n in node_list]

def combined_heuristic_scores(G: nx.DiGraph, node_list: list[str]) -> list[float]:
    deg = degree_heuristic_scores(G, node_list)
    pr = pagerank_scores(G, node_list)
    return [0.6 * d + 0.4 * p for d, p in zip(deg, pr)]