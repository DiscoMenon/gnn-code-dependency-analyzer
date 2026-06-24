import os
import torch
from pathlib import Path
from torch_geometric.data import Dataset, Data
from backend.graph.graph_builder import CodeGraphBuilder
from backend.gnn.graph_converter import networkx_to_pyg

class RepoGraphDataset(Dataset):
    def __init__(self, repos_dir: str = "data/repos"):
        self.repos_dir = Path(repos_dir)
        self.repo_paths = [
            p for p in self.repos_dir.iterdir()
            if p.is_dir()
        ]
        self.graphs = []
        self.node_lists = []
        self.repo_names = []
        self._build_all()

    def _build_all(self):
        print(f"Building graphs for {len(self.repo_paths)} repos...")
        for repo_path in self.repo_paths:
            try:
                print(f"  Parsing {repo_path.name}...")
                builder = CodeGraphBuilder(str(repo_path))
                G = builder.build()

                if G.number_of_nodes() < 5:
                    print(f"  Skipping {repo_path.name} — too small")
                    continue

                data, node_list = networkx_to_pyg(G)
                self.graphs.append(data)
                self.node_lists.append(node_list)
                self.repo_names.append(repo_path.name)
                print(f"  {repo_path.name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

            except Exception as e:
                print(f"  Failed {repo_path.name}: {e}")

        print(f"\nSuccessfully built {len(self.graphs)} repo graphs")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.node_lists[idx], self.repo_names[idx]