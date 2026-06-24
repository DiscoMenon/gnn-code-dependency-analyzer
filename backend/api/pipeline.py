import subprocess
import networkx as nx
from pathlib import Path
from backend.graph.graph_builder import CodeGraphBuilder
from backend.gnn.graph_converter import networkx_to_pyg
from backend.gnn.trainer import load_model, train_multi_repo
from backend.gnn.dataset import RepoGraphDataset
from backend.gnn.scorer import score_nodes
from backend.api.schemas import (
    AnalyzeResponse, NodeSchema, EdgeSchema,
    IssueSchema, GraphStats
)

MODEL_PATH = "data/model.pt"
REPOS_DIR = "data/repos"

def get_model(input_dim: int = 7):
    if Path(MODEL_PATH).exists():
        return load_model(MODEL_PATH, input_dim=input_dim)
    
    print("No model found, training from scratch...")
    dataset = RepoGraphDataset(REPOS_DIR)
    model = train_multi_repo(dataset, save_path=MODEL_PATH)
    return model


def clone_repo(github_url: str) -> str:
    repo_name = github_url.rstrip("/").split("/")[-1]
    clone_path = f"{REPOS_DIR}/{repo_name}"

    if not Path(clone_path).exists():
        result = subprocess.run(
            ["git", "clone", "--depth=1", github_url, clone_path],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            raise ValueError(f"Git clone failed: {result.stderr[:200]}")

    return clone_path, repo_name


def detect_issues(G: nx.DiGraph, scored_nodes: list[dict]) -> list[IssueSchema]:
    issues = []

    # Import-only subgraph for cycle detection
    import_graph = nx.DiGraph()
    for u, v, d in G.edges(data=True):
        if d.get("type") == "imports":
            import_graph.add_edge(u, v)

    # Circular dependencies
    
    cycles = []
    try:
        gen = nx.simple_cycles(import_graph)
        for _ in range(1000):
            try:
                cycles.append(next(gen))
            except StopIteration:
                break
    except Exception:
        pass

    for cycle in cycles:
        issues.append(IssueSchema(
            type="circular_dep",
            severity="high",
            nodes=cycle,
            description=f"Circular dependency: {' → '.join(cycle)} → {cycle[0]}"
        ))

    # God files (top 5% by in-degree)
    file_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "file"]
    in_degrees = {n: int(import_graph.in_degree(n)) if import_graph.has_node(n) else 0 for n in file_nodes}
    if in_degrees:
        threshold = max(in_degrees.values()) * 0.5
        god_files = [n for n, deg in in_degrees.items() if deg >= threshold and deg > 2]
        for gf in god_files:
            issues.append(IssueSchema(
                type="god_file",
                severity="high",
                nodes=[gf],
                description=f"{gf} has {in_degrees[gf]} incoming imports — single point of failure"
            ))

    # Dead code clusters (isolated file nodes)
    dead = [
        n for n in file_nodes
        if (int(import_graph.in_degree(n)) if import_graph.has_node(n) else 0) == 0
        and (int(import_graph.out_degree(n)) if import_graph.has_node(n) else 0) == 0
        and int(G.out_degree(n)) <= 1
    ]
    if dead:
        issues.append(IssueSchema(
            type="dead_code",
            severity="low",
            nodes=dead,
            description=f"{len(dead)} files have no import connections — likely dead code"
        ))

    return issues


def run_analysis(github_url: str) -> AnalyzeResponse:
    try:
        # 1. Clone
        clone_path, repo_name = clone_repo(github_url)

        # 2. Build graph
        builder = CodeGraphBuilder(clone_path)
        G = builder.build()

        if G.number_of_nodes() == 0:
            raise ValueError("No Python files found in repository")

        # 3. Convert to PyG
        data, node_list = networkx_to_pyg(G)

        # 4. Load model + score
        model = get_model(input_dim=data.x.shape[1])
        scored = score_nodes(model, data, node_list)

        # 5. Detect issues
        issues = detect_issues(G, scored)

        # 6. Build node schemas
        nodes_in_cycles = set()
        for issue in issues:
            if issue.type == "circular_dep":
                nodes_in_cycles.update(issue.nodes)

        god_file_nodes = set()
        for issue in issues:
            if issue.type == "god_file":
                god_file_nodes.update(issue.nodes)

        dead_nodes = set()
        for issue in issues:
            if issue.type == "dead_code":
                dead_nodes.update(issue.nodes)

        node_schemas = []
        for item in scored:
            node_issues = []
            if item["node"] in nodes_in_cycles:
                node_issues.append("CIRCULAR DEP")
            if item["node"] in god_file_nodes:
                node_issues.append("GOD FILE")
            if item["node"] in dead_nodes:
                node_issues.append("DEAD CODE")

            node_schemas.append(NodeSchema(
                id=item["node"],
                type="file" if item["is_file"] else "function" if item["is_function"] else "class",
                anomaly_score=item["anomaly_score"],
                in_degree=int(G.in_degree(item["node"])),
                out_degree=int(G.out_degree(item["node"])),
                pagerank=item["pagerank"],
                in_cycle=item["in_cycle"],
                issues=node_issues
            ))

        # 7. Build edge schemas
        edge_schemas = [
            EdgeSchema(source=u, target=v, type=d.get("type", "unknown"))
            for u, v, d in G.edges(data=True)
        ]

        # 8. Stats
        file_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "file"]
        stats = GraphStats(
            total_nodes=G.number_of_nodes(),
            total_edges=G.number_of_edges(),
            file_count=len(file_nodes),
            function_count=len([n for n, d in G.nodes(data=True) if d.get("type") == "function"]),
            class_count=len([n for n, d in G.nodes(data=True) if d.get("type") == "class"]),
            circular_dep_count=len([i for i in issues if i.type == "circular_dep"]),
            god_file_count=len([i for i in issues if i.type == "god_file"]),
            dead_code_count=len([i for i in issues if i.type == "dead_code"]),
        )

        return AnalyzeResponse(
            repo_name=repo_name,
            status="success",
            stats=stats,
            nodes=node_schemas,
            edges=edge_schemas,
            issues=issues
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return AnalyzeResponse(
            repo_name="",
            status="error",
            stats=GraphStats(
                total_nodes=0, total_edges=0, file_count=0,
                function_count=0, class_count=0,
                circular_dep_count=0, god_file_count=0, dead_code_count=0
            ),
            nodes=[], edges=[], issues=[],
            error=str(e)
        )