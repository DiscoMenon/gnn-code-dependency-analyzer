import torch
import numpy as np
from backend.gnn.model import CodeGNN
from torch_geometric.data import Data

def score_nodes(
    model: CodeGNN,
    data: Data,
    node_list: list[str]
) -> list[dict]:
    """
    Returns nodes ranked by anomaly score with metadata.
    Higher score = more structurally unusual.
    """
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)

    embeddings_np = embeddings.numpy()

    # Centroid of all embeddings
    centroid = embeddings_np.mean(axis=0)

    # Distance from centroid = anomaly score
    distances = np.linalg.norm(embeddings_np - centroid, axis=1)

    # Normalize to 0-1
    min_d, max_d = distances.min(), distances.max()
    if max_d - min_d > 0:
        normalized = (distances - min_d) / (max_d - min_d)
    else:
        normalized = distances

    results = []
    for i, node in enumerate(node_list):
        node_data = data.x[i].tolist()
        results.append({
            "node": node,
            "anomaly_score": float(normalized[i]),
            "in_degree_norm": node_data[0],
            "out_degree_norm": node_data[1],
            "pagerank": node_data[2],
            "in_cycle": bool(node_data[3] > 0.5),
            "is_file": bool(node_data[4] > 0.5),
            "is_function": bool(node_data[5] > 0.5),
            "is_class": bool(node_data[6] > 0.5),
        })

    return sorted(results, key=lambda x: x["anomaly_score"], reverse=True)  