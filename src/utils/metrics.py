"""
Shared metric utilities for VetVision-LM evaluation.

Provides:
    - recall_at_k
    - mean_reciprocal_rank
    - zero_shot_classification_metrics
    - compute_retrieval_metrics
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    similarity_matrix: np.ndarray,
    k: int,
    query_axis: int = 0,
) -> float:
    """
    Compute Recall@K for a similarity matrix.

    Args:
        similarity_matrix: (N_query, N_gallery) float array.
        k:                 K value.
        query_axis:        0 → image→text retrieval, 1 → text→image.

    Returns:
        Recall@K as a float in [0, 1].
    """
    if query_axis == 1:
        similarity_matrix = similarity_matrix.T

    n_queries = similarity_matrix.shape[0]
    hits = 0

    for i in range(n_queries):
        row = similarity_matrix[i]
        top_k_idx = np.argsort(-row)[:k]
        if i in top_k_idx:
            hits += 1

    return hits / n_queries


def mean_reciprocal_rank(
    similarity_matrix: np.ndarray,
    query_axis: int = 0,
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    Args:
        similarity_matrix: (N_query, N_gallery) float array.
        query_axis:        0 → image→text, 1 → text→image.

    Returns:
        MRR as float in [0, 1].
    """
    if query_axis == 1:
        similarity_matrix = similarity_matrix.T

    n_queries = similarity_matrix.shape[0]
    rr_sum = 0.0

    for i in range(n_queries):
        row = similarity_matrix[i]
        ranked_idx = np.argsort(-row)
        rank = np.where(ranked_idx == i)[0]
        if len(rank) > 0:
            rr_sum += 1.0 / (rank[0] + 1)

    return rr_sum / n_queries


def compute_retrieval_metrics(
    similarity_matrix: np.ndarray,
    k_values: List[int] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Compute all image→text and text→image retrieval metrics.

    Args:
        similarity_matrix: (N, N) cosine similarity matrix.
        k_values:          List of K values for Recall@K.

    Returns:
        Dict with keys like ``i2t_R@1``, ``t2i_R@5``, ``i2t_MRR``, etc.
    """
    metrics: Dict[str, float] = {}

    for direction, axis in [("i2t", 0), ("t2i", 1)]:
        for k in k_values:
            metrics[f"{direction}_R@{k}"] = recall_at_k(
                similarity_matrix, k=k, query_axis=axis
            )
        metrics[f"{direction}_MRR"] = mean_reciprocal_rank(
            similarity_matrix, query_axis=axis
        )

    return metrics


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute accuracy, AUC-ROC, and F1 score.

    Args:
        y_true:      (N,) integer ground-truth labels.
        y_pred:      (N,) integer predicted labels.
        y_score:     (N,) or (N, C) probability scores.
        class_names: Optional list of class name strings.

    Returns:
        Dict of metric values.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        roc_auc_score,
    )

    metrics: Dict[str, float] = {}

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_weighted"] = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )

    # Per-class F1
    classes = class_names or [str(c) for c in sorted(set(y_true.tolist()))]
    per_class_f1 = f1_score(
        y_true, y_pred, average=None, labels=list(range(len(classes))), zero_division=0
    )
    for c, f1 in zip(classes, per_class_f1):
        metrics[f"f1_{c}"] = float(f1)

    # AUC-ROC (binary or multiclass)
    if y_score is not None:
        try:
            n_classes = len(classes)
            if n_classes == 2:
                scores = y_score[:, 1] if y_score.ndim == 2 else y_score
                metrics["auc_roc"] = float(roc_auc_score(y_true, scores))
            else:
                metrics["auc_roc"] = float(
                    roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
                )
        except Exception:
            metrics["auc_roc"] = float("nan")

    return metrics


# ---------------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_similarity_matrix(
    vision_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
) -> np.ndarray:
    """
    Compute cosine similarity matrix between vision and text embeddings.

    Assumes both tensors are already L2-normalised.

    Args:
        vision_embeds: (N, D)
        text_embeds:   (N, D)

    Returns:
        (N, N) numpy array of cosine similarities.
    """
    sim = vision_embeds @ text_embeds.T  # (N, N)
    return sim.cpu().numpy()
