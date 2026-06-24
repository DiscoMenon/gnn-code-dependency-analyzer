import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path
from backend.graph.graph_builder import CodeGraphBuilder
from backend.gnn.graph_converter import networkx_to_pyg
from backend.gnn.trainer import load_model
from backend.benchmark.ground_truth import get_ground_truth
from backend.benchmark.scorers import degree_heuristic_scores, pagerank_scores, gnn_scores, combined_heuristic_scores
from backend.benchmark.metrics import ndcg_at_k, precision_at_k, average_precision

MODEL_PATH = "backend/data/model.pt"
REPOS_DIR = "backend/data/repos"
K = 10

def run_benchmark():
    repo_paths = [p for p in Path(REPOS_DIR).iterdir() if p.is_dir()]
    print(f"Benchmarking on {len(repo_paths)} repos\n")

    results = []
    model = None

    for repo_path in repo_paths:
        try:
            builder = CodeGraphBuilder(str(repo_path))
            G = builder.build()

            if G.number_of_nodes() < 10:
                continue

            data, node_list = networkx_to_pyg(G)

            if model is None:
                model = load_model(MODEL_PATH, input_dim=data.x.shape[1])

            truth = get_ground_truth(G)
            problematic = truth["all_problematic"]

            file_nodes = [n for n in node_list if "::" not in n]
            if len(file_nodes) < 5:
                continue

            labels = [1 if n in problematic else 0 for n in file_nodes]

            if sum(labels) == 0:
                print(f"  {repo_path.name}: no problematic nodes found, skipping")
                continue

            deg_scores = degree_heuristic_scores(G, file_nodes)
            pr_scores = pagerank_scores(G, file_nodes)
            combined_scores = combined_heuristic_scores(G, file_nodes)

            all_gnn = gnn_scores(model, data, node_list)
            gnn_score_map = dict(zip(node_list, all_gnn))
            gnn_file_scores = [gnn_score_map.get(n, 0.0) for n in file_nodes]

            repo_result = {
                "repo": repo_path.name,
                "n_files": len(file_nodes),
                "n_problematic": sum(labels),
                "degree": {
                    "ndcg@10": ndcg_at_k(deg_scores, labels, K),
                    "p@10": precision_at_k(deg_scores, labels, K),
                    "ap": average_precision(deg_scores, labels),
                },
                "pagerank": {
                    "ndcg@10": ndcg_at_k(pr_scores, labels, K),
                    "p@10": precision_at_k(pr_scores, labels, K),
                    "ap": average_precision(pr_scores, labels),
                },
                "combined": {
                    "ndcg@10": ndcg_at_k(combined_scores, labels, K),
                    "p@10": precision_at_k(combined_scores, labels, K),
                    "ap": average_precision(combined_scores, labels),
                },
                "gnn": {
                    "ndcg@10": ndcg_at_k(gnn_file_scores, labels, K),
                    "p@10": precision_at_k(gnn_file_scores, labels, K),
                    "ap": average_precision(gnn_file_scores, labels),
                },
            }
            results.append(repo_result)

            print(f"  {repo_path.name:20s} | "
                  f"Degree: {repo_result['degree']['ndcg@10']:.3f} | "
                  f"PageRank: {repo_result['pagerank']['ndcg@10']:.3f} | "
                  f"Combined: {repo_result['combined']['ndcg@10']:.3f} | "
                  f"GNN: {repo_result['gnn']['ndcg@10']:.3f}")

        except Exception as e:
            print(f"  Failed {repo_path.name}: {e}")

    if not results:
        print("No results collected")
        return

    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS (averaged over {len(results)} repos)")
    print(f"{'='*60}")

    for method in ["degree", "pagerank", "combined", "gnn"]:
        avg_ndcg = np.mean([r[method]["ndcg@10"] for r in results])
        avg_p = np.mean([r[method]["p@10"] for r in results])
        avg_ap = np.mean([r[method]["ap"] for r in results])
        print(f"\n{method.upper():10s}")
        print(f"  NDCG@10:  {avg_ndcg:.4f}")
        print(f"  P@10:     {avg_p:.4f}")
        print(f"  MAP:      {avg_ap:.4f}")

    with open("backend/data/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to backend/data/benchmark_results.json")

    print(f"\n{'='*60}")
    print("GNN vs COMBINED (per repo, NDCG@10)")
    print(f"{'='*60}")
    gnn_wins = 0
    for r in results:
        gnn_better = r["gnn"]["ndcg@10"] > r["combined"]["ndcg@10"]
        marker = "✓ GNN" if gnn_better else "✗ CMB"
        if gnn_better:
            gnn_wins += 1
        print(f"  {marker}  {r['repo']:20s}  GNN={r['gnn']['ndcg@10']:.3f}  Combined={r['combined']['ndcg@10']:.3f}")

    print(f"\nGNN wins {gnn_wins}/{len(results)} repos")

if __name__ == "__main__":
    run_benchmark()