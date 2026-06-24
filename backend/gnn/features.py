import networkx as nx
import torch
import numpy as np

def compute_node_features(G: nx.DiGraph) -> dict:
    """
    For each node, compute a feature vector:
    [in_degree, out_degree, pagerank, in_cycle, is_file, is_function, is_class]
    """
    # Compute graph-level metrics
    pagerank = nx.pagerank(G, alpha=0.85)
    
    # Detect which nodes are in cycles
    cycles = []
    try:
        gen = nx.simple_cycles(G)
        for _ in range(1000):  # cap at 1000 cycles max
            try:
                cycles.append(next(gen))
            except StopIteration:
                break
    except Exception:
        pass
    nodes_in_cycles = set()
    for cycle in cycles:
        nodes_in_cycles.update(cycle)

    # Normalize degrees
    max_in = max(dict(G.in_degree()).values(), default=1)
    max_out = max(dict(G.out_degree()).values(), default=1)

    features = {}
    for node, data in G.nodes(data=True):
        node_type = data.get("type", "file")
        features[node] = [
            G.in_degree(node) / max_in,           # normalized in-degree
            G.out_degree(node) / max_out,          # normalized out-degree
            pagerank.get(node, 0.0),               # pagerank score
            1.0 if node in nodes_in_cycles else 0.0,  # in a cycle?
            1.0 if node_type == "file" else 0.0,
            1.0 if node_type == "function" else 0.0,
            1.0 if node_type == "class" else 0.0,
        ]

    return features