import subprocess
from pathlib import Path
from backend.graph.graph_builder import CodeGraphBuilder
from backend.gnn.dataset import RepoGraphDataset
from backend.gnn.trainer import train_multi_repo, load_model
from backend.gnn.graph_converter import networkx_to_pyg
from backend.gnn.scorer import score_nodes

MODEL_PATH = "data/model.pt"

def run_pipeline(github_url: str):
    # Clone target repo
    repo_name = github_url.split("/")[-1]
    clone_path = f"data/repos/{repo_name}"
    
    if not Path(clone_path).exists():
        print(f"Cloning {github_url}...")
        subprocess.run(["git", "clone", "--depth=1", github_url, clone_path])

    # Build graph
    print(f"Building graph for {repo_name}...")
    builder = CodeGraphBuilder(clone_path)
    G = builder.build()
    data, node_list = networkx_to_pyg(G)

    # Load or train model
    if Path(MODEL_PATH).exists():
        print("Loading saved model...")
        model = load_model(MODEL_PATH, input_dim=data.x.shape[1])
    else:
        print("No saved model found, training from scratch...")
        dataset = RepoGraphDataset("data/repos")
        model = train_multi_repo(dataset, save_path=MODEL_PATH)

    # Score nodes
    scores = score_nodes(model, data, node_list)

    print(f"\n=== Top 10 Anomalous Nodes in {repo_name} ===")
    for item in scores[:10]:
        flags = []
        if item["in_cycle"]:
            flags.append("CIRCULAR DEP")
        if item["in_degree_norm"] > 0.7:
            flags.append("GOD FILE")
        if item["out_degree_norm"] == 0 and item["in_degree_norm"] == 0:
            flags.append("DEAD CODE")

        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        print(f"  {item['anomaly_score']:.3f}  {item['node']}{flag_str}")

if __name__ == "__main__":
    run_pipeline("https://github.com/psf/requests")