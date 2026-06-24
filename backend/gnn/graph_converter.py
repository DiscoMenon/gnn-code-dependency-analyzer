import torch
import networkx as nx
from torch_geometric.data import Data
from backend.gnn.features import compute_node_features

def networkx_to_pyg(G: nx.DiGraph) -> tuple[Data, list]:
    """
    Converts a NetworkX DiGraph to a PyTorch Geometric Data object.
    Returns (pyg_data, node_list) where node_list[i] = node name at index i
    """
    node_list = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}

    # Build feature matrix
    node_features = compute_node_features(G)
    feature_dim = 7
    x = torch.zeros((len(node_list), feature_dim), dtype=torch.float)
    
    for node, feat in node_features.items():
        idx = node_to_idx[node]
        x[idx] = torch.tensor(feat, dtype=torch.float)

    # Build edge index (COO format — PyG standard)
    edge_index = []
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            edge_index.append([node_to_idx[u], node_to_idx[v]])

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    return data, node_list