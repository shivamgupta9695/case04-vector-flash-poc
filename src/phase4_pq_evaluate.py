"""
Case 04 — Phase 4: Recall, Latency, and Search Quality Evaluation

Benchmarking and Quality Analysis across 4 Search Methods:
    1. Exact FP32 (FAISS IndexFlatL2) — The Phase 1 Exact Oracle
    2. Reconstructed PQ (Full vector reconstruction + exact L2 distance)
    3. Custom ADC (Asymmetric Distance Computation via LUT)
    4. FAISS IndexPQ (C++ native ADC baseline)

Evaluated Metrics:
    - Recall@1, Recall@5, Recall@10 against frozen Phase 1 ground truth
    - Candidate Recall@10-in-100 (Phase 1 top-10 GT captured in top-100 candidates)
    - Exact Recall@100 (against in-memory exact FP32 top-100 oracle)
    - Search Latency: Mean, P50, P95, P99
    - Throughput (QPS)
    - Comprehensive Publication-Quality Visualization Plots
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Ensure workspace root in path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

try:
    import numpy as np
except ImportError as err:
    print(f"CRITICAL ERROR: Failed to import NumPy: {err}", file=sys.stderr)
    sys.exit(1)

try:
    import faiss
except ImportError as err:
    print(f"CRITICAL ERROR: Failed to import FAISS: {err}", file=sys.stderr)
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from src.phase1_2_evaluation import calculate_recall_at_k, calculate_per_query_recall
from src.phase4_pq_adc import ADCSearcher, compute_adc_lut_single_query, compute_adc_distances
from src.phase4_pq_reconstruct import reconstruct_vectors_numpy


def calculate_candidate_recall_10_in_k(
    ground_truth_top10: np.ndarray,
    predicted_top_k: np.ndarray,
    k_candidate: int = 100,
) -> float:
    """
    Measures the fraction of the true top-10 ground truth neighbors
    that are retrieved within the top-k_candidate predicted neighbors.
    Essential for evaluating PQ candidate generation for the downstream
    PPT 'Candidate Merge + Exact Re-Rank' stage.
    """
    num_queries = ground_truth_top10.shape[0]
    recalls = np.zeros(num_queries, dtype=np.float64)

    for i in range(num_queries):
        gt_set = set(ground_truth_top10[i, :10].tolist())
        pred_cand_set = set(predicted_top_k[i, :k_candidate].tolist())
        hits = len(gt_set.intersection(pred_cand_set))
        recalls[i] = hits / 10.0

    return float(np.mean(recalls))


class ReconstructedPQSearcher:
    """
    Searcher that evaluates distance against explicitly reconstructed PQ vectors.
    Used to demonstrate the equivalence and computational cost vs ADC.
    """

    def __init__(self, reconstructed_vectors: np.ndarray):
        self.vectors = reconstructed_vectors.astype(np.float32)
        self.num_vectors, self.dimension = self.vectors.shape

    def search_batch(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        num_queries = queries.shape[0]
        all_indices = np.empty((num_queries, k), dtype=np.int32)
        all_dists = np.empty((num_queries, k), dtype=np.float32)

        for q_idx in range(num_queries):
            q = queries[q_idx]
            dists = np.sum((q - self.vectors) ** 2, axis=1)
            part = np.argpartition(dists, k)[:k]
            top_k = part[np.argsort(dists[part])]
            all_indices[q_idx] = top_k
            all_dists[q_idx] = dists[top_k]

        return all_indices, all_dists


def measure_search_latency_and_recall(
    search_fn,
    queries: np.ndarray,
    ground_truth_10: np.ndarray,
    ground_truth_100: Optional[np.ndarray] = None,
    k: int = 100,
    warmup_queries: int = 100,
) -> Dict[str, Any]:
    """
    Rigorously benchmarks a search function:
    1. Warm-up phase to eliminate cache and JIT cold starts.
    2. Per-query latency profiling (P50, P95, P99, mean).
    3. Multi-k recall calculation against frozen Phase 1 ground truth.
    """
    num_queries = queries.shape[0]

    # Warm-up phase
    if warmup_queries > 0:
        warmup_subset = queries[:min(warmup_queries, num_queries)]
        _ = search_fn(warmup_subset, min(k, 10))

    # Individual per-query latency profiling
    query_latencies_ms = np.zeros(num_queries, dtype=np.float64)
    all_predictions = np.empty((num_queries, k), dtype=np.int32)
    all_distances = np.empty((num_queries, k), dtype=np.float32)

    t0_total = time.perf_counter()
    for i in range(num_queries):
        t_start = time.perf_counter()
        idxs, dsts = search_fn(queries[i:i+1], k)
        t_elapsed = (time.perf_counter() - t_start) * 1000.0  # ms
        query_latencies_ms[i] = t_elapsed
        all_predictions[i] = idxs[0]
        all_distances[i] = dsts[0]
    total_time_seconds = time.perf_counter() - t0_total

    # Accuracy calculations
    recall_at_1 = calculate_recall_at_k(ground_truth_10, all_predictions, k=1)
    recall_at_5 = calculate_recall_at_k(ground_truth_10, all_predictions, k=5)
    recall_at_10 = calculate_recall_at_k(ground_truth_10, all_predictions, k=10)

    # Candidate recall: true top 10 captured in top 100 candidates
    candidate_recall_10_in_100 = calculate_candidate_recall_10_in_k(
        ground_truth_10, all_predictions, k_candidate=k
    )

    # If ground_truth_100 is provided, calculate exact recall@100
    recall_at_100 = None
    if ground_truth_100 is not None and ground_truth_100.shape[1] >= 100 and k >= 100:
        recall_at_100 = calculate_recall_at_k(ground_truth_100, all_predictions, k=100)

    avg_latency = float(np.mean(query_latencies_ms))
    p50_latency = float(np.percentile(query_latencies_ms, 50))
    p95_latency = float(np.percentile(query_latencies_ms, 95))
    p99_latency = float(np.percentile(query_latencies_ms, 99))
    qps = float(num_queries / max(total_time_seconds, 1e-9))

    return {
        "num_queries": num_queries,
        "k": k,
        "total_time_seconds": round(total_time_seconds, 4),
        "avg_latency_ms": round(avg_latency, 4),
        "p50_latency_ms": round(p50_latency, 4),
        "p95_latency_ms": round(p95_latency, 4),
        "p99_latency_ms": round(p99_latency, 4),
        "qps": round(qps, 2),
        "recall_at_1": round(recall_at_1, 6),
        "recall_at_5": round(recall_at_5, 6),
        "recall_at_10": round(recall_at_10, 6),
        "candidate_recall_10_in_100": round(candidate_recall_10_in_100, 6),
        "recall_at_100": round(recall_at_100, 6) if recall_at_100 is not None else None,
        "sample_predictions_q0": all_predictions[0, :10].tolist(),
    }


def generate_evaluation_plots(
    results_summary: Dict[str, Any],
    reconstruction_errors_l2: np.ndarray,
    output_dir: Path | str,
) -> List[Path]:
    """Generates publication-quality evaluation charts and dashboard."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    methods = ["Exact FP32", "Reconstructed PQ", "ADC (NumPy)", "FAISS IndexPQ"]
    keys = ["exact_fp32", "reconstructed_pq", "adc_numpy", "faiss_index_pq"]
    active_keys = [k for k in keys if k in results_summary]
    active_labels = [results_summary[k]["method_name"] for k in active_keys]

    colors = ["#2b5c8f", "#d95f02", "#1b9e77", "#7570b3"]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

    # Plot 1: Recall Comparison Bar Chart
    fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
    x = np.arange(len(active_labels))
    width = 0.2

    r1_vals = [results_summary[k]["recall_at_1"] for k in active_keys]
    r5_vals = [results_summary[k]["recall_at_5"] for k in active_keys]
    r10_vals = [results_summary[k]["recall_at_10"] for k in active_keys]
    r10_in_100_vals = [results_summary[k]["candidate_recall_10_in_100"] for k in active_keys]

    ax1.bar(x - 1.5 * width, r1_vals, width, label="Recall@1", color="#386cb0")
    ax1.bar(x - 0.5 * width, r5_vals, width, label="Recall@5", color="#7fc97f")
    ax1.bar(x + 0.5 * width, r10_vals, width, label="Recall@10", color="#beaed4")
    ax1.bar(x + 1.5 * width, r10_in_100_vals, width, label="Candidate R@10@100", color="#fdc086")

    ax1.set_ylabel("Recall Score")
    ax1.set_title("Search Recall Comparison Across Methods (vs Frozen Phase 1 GT)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(active_labels)
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    p1 = out_dir / "recall_comparison.png"
    fig1.savefig(p1)
    plt.close(fig1)
    generated.append(p1)

    # Plot 2: Latency Percentiles Comparison
    fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=300)
    p50_vals = [results_summary[k]["p50_latency_ms"] for k in active_keys]
    p95_vals = [results_summary[k]["p95_latency_ms"] for k in active_keys]
    p99_vals = [results_summary[k]["p99_latency_ms"] for k in active_keys]

    ax2.bar(x - width, p50_vals, width, label="P50 Latency", color="#4daf4a")
    ax2.bar(x, p95_vals, width, label="P95 Latency", color="#377eb8")
    ax2.bar(x + width, p99_vals, width, label="P99 Latency", color="#e41a1c")

    ax2.set_ylabel("Latency (ms / query)")
    ax2.set_title("Per-Query Search Latency Percentiles (P50, P95, P99)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(active_labels)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.5)

    p2 = out_dir / "latency_comparison.png"
    fig2.savefig(p2)
    plt.close(fig2)
    generated.append(p2)

    # Plot 3: Reconstruction Error Histogram
    fig3, ax3 = plt.subplots(figsize=(8, 5), dpi=300)
    ax3.hist(reconstruction_errors_l2, bins=60, color="#2b5c8f", edgecolor="black", alpha=0.75)
    mean_err = np.mean(reconstruction_errors_l2)
    median_err = np.median(reconstruction_errors_l2)
    ax3.axvline(mean_err, color="red", linestyle="--", label=f"Mean L2 Error = {mean_err:.3f}")
    ax3.axvline(median_err, color="orange", linestyle=":", label=f"Median L2 Error = {median_err:.3f}")
    ax3.set_xlabel("Reconstruction L2 Error ||x - x_hat||_2")
    ax3.set_ylabel("Number of Vectors (100K Total)")
    ax3.set_title("Product Quantization Reconstruction Error Distribution (M=16, Ks=256)")
    ax3.legend(loc="upper right")
    ax3.grid(True, linestyle="--", alpha=0.5)

    p3 = out_dir / "reconstruction_error_hist.png"
    fig3.savefig(p3)
    plt.close(fig3)
    generated.append(p3)

    # Plot 4: Comprehensive 4-Panel Dashboard
    fig4, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # Top-Left: Recall
    ax = axes[0, 0]
    ax.bar(x - 1.5 * width, r1_vals, width, label="R@1", color="#386cb0")
    ax.bar(x - 0.5 * width, r5_vals, width, label="R@5", color="#7fc97f")
    ax.bar(x + 0.5 * width, r10_vals, width, label="R@10", color="#beaed4")
    ax.bar(x + 1.5 * width, r10_in_100_vals, width, label="R@10@100", color="#fdc086")
    ax.set_title("A. Recall@K Comparison", fontweight="bold")
    ax.set_ylabel("Recall")
    ax.set_xticks(x)
    ax.set_xticklabels(active_labels, fontsize=8.5)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Top-Right: Latency
    ax = axes[0, 1]
    ax.bar(x - width, p50_vals, width, label="P50", color="#4daf4a")
    ax.bar(x, p95_vals, width, label="P95", color="#377eb8")
    ax.bar(x + width, p99_vals, width, label="P99", color="#e41a1c")
    ax.set_title("B. Search Latency Percentiles (ms)", fontweight="bold")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(active_labels, fontsize=8.5)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Bottom-Left: Throughput QPS
    ax = axes[1, 0]
    qps_vals = [results_summary[k]["qps"] for k in active_keys]
    ax.set_xticks(x)
    bars = ax.bar(x, qps_vals, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"], width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 20, f"{int(h):,} QPS", ha="center", va="bottom", fontsize=8.5)
    ax.set_title("C. Query Throughput (QPS)", fontweight="bold")
    ax.set_ylabel("QPS")
    ax.set_xticklabels(active_labels, fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Bottom-Right: Reconstruction Error Histogram
    ax = axes[1, 1]
    ax.hist(reconstruction_errors_l2, bins=50, color="#377eb8", edgecolor="black", alpha=0.7)
    ax.axvline(mean_err, color="red", linestyle="--", label=f"Mean = {mean_err:.2f}")
    ax.set_title("D. PQ Reconstruction L2 Error (100K Base Vectors)", fontweight="bold")
    ax.set_xlabel("||x - x_hat||_2")
    ax.set_ylabel("Vectors")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig4.suptitle("Case 04 — Phase 4: Product Quantization & ADC Search Quality Dashboard", fontsize=14, fontweight="bold")
    p4 = out_dir / "phase4_dashboard.png"
    fig4.savefig(p4)
    plt.close(fig4)
    generated.append(p4)

    return generated
