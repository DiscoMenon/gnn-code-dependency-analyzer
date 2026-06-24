import networkx as nx

def get_ground_truth(G: nx.DiGraph) -> dict[str, set[str]]:
    import_graph = nx.DiGraph()
    for u, v, d in G.edges(data=True):
        if d.get("type") == "imports":
            import_graph.add_edge(u, v)

    file_nodes = {n for n, d in G.nodes(data=True) if d.get("type") == "file"}

    # Circular dependency participants
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
    circular_nodes = set()
    for cycle in cycles:
        circular_nodes.update(cycle)

    # Betweenness centrality — structural bridges
    if import_graph.number_of_nodes() > 0:
        betweenness = nx.betweenness_centrality(import_graph, normalized=True)
        if betweenness:
            sorted_bc = sorted(betweenness.values(), reverse=True)
            threshold_idx = max(1, len(sorted_bc) // 5)
            threshold = sorted_bc[threshold_idx]
            high_betweenness = {
                n for n, bc in betweenness.items()
                if bc >= threshold and bc > 0 and n in file_nodes
            }
        else:
            high_betweenness = set()
    else:
        high_betweenness = set()

    # Dead code
    dead_code = {
        n for n in file_nodes
        if (int(import_graph.in_degree(n)) if import_graph.has_node(n) else 0) == 0
        and (int(import_graph.out_degree(n)) if import_graph.has_node(n) else 0) == 0
    }

    return {
        "circular": circular_nodes,
        "high_betweenness": high_betweenness,
        "dead_code": dead_code,
        "all_problematic": circular_nodes | high_betweenness | dead_code
    }