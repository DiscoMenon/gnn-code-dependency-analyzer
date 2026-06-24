import numpy as np

def ndcg_at_k(scores: list[float], labels: list[int], k: int = 10) -> float:
    """
    NDCG@K — how well does your ranking surface the truly problematic nodes?
    scores: predicted anomaly score per node (higher = more anomalous)
    labels: 1 if node is truly problematic, 0 otherwise
    k: evaluate top-K ranked nodes
    """
    if sum(labels) == 0:
        return 0.0  # no positive examples

    # Rank nodes by score descending
    ranked_indices = np.argsort(scores)[::-1][:k]
    ranked_labels = [labels[i] for i in ranked_indices]

    # DCG: relevance weighted by log position
    dcg = sum(
        rel / np.log2(i + 2)
        for i, rel in enumerate(ranked_labels)
    )

    # Ideal DCG: best possible ranking
    ideal_labels = sorted(labels, reverse=True)[:k]
    idcg = sum(
        rel / np.log2(i + 2)
        for i, rel in enumerate(ideal_labels)
    )

    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(scores: list[float], labels: list[int], k: int = 10) -> float:
    """What fraction of top-K nodes are actually problematic?"""
    ranked_indices = np.argsort(scores)[::-1][:k]
    return sum(labels[i] for i in ranked_indices) / k


def average_precision(scores: list[float], labels: list[int]) -> float:
    """AP — area under precision-recall curve."""
    if sum(labels) == 0:
        return 0.0

    ranked_indices = np.argsort(scores)[::-1]
    precisions = []
    hits = 0

    for i, idx in enumerate(ranked_indices):
        if labels[idx] == 1:
            hits += 1
            precisions.append(hits / (i + 1))

    return np.mean(precisions) if precisions else 0.0