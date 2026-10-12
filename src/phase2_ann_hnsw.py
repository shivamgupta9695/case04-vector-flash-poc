"""
Case 04 — Phase 2: ANN Index + Baseline Approximate Vector Search
Graph-Based ANN Baseline using FAISS IndexHNSWFlat

Evaluates FAISS IndexHNSWFlat across multiple graph configurations (M=16, M=32)
and search depths (efSearch=10, 20, 40, 80, 160, 320) against the frozen
Phase 1.1 exact ground truth on 100K x 128 vectors.
"""

import sys
import time
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Verify dependencies
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

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
except ImportError as err:
    print(f"CRITICAL ERROR: Failed to import Matplotlib: {err}", file=sys.stderr)
    sys.exit(1)

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

# Import Phase 1 evaluation framework utilities
try:
    from src.phase1_2_evaluation import (
        load_fvecs,
        load_ivecs,
        compute_sha256,
        EXPECTED_CHECKSUMS,
        calculate_recall_at_k,
        calculate_per_query_recall,
        validate_predictions,
        measure_batch_search,
        measure_individual_query_latencies,
        FaissIndexFlatL2Searcher,
        VectorSearcher,
    )
except ImportError as err:
    print(f"CRITICAL ERROR: Failed to import Phase 1.2 framework: {err}", file=sys.stderr)
    sys.exit(1)


# ==============================================================================
# HNSW Searcher Wrapper
# ==============================================================================
class FaissHNSWSearcher:
    """
    HNSW ANN search implementation wrapping faiss.IndexHNSWFlat.
    Implements the VectorSearcher protocol for seamless benchmarking.
    """

    def __init__(self, dimension: int, m: int = 32, ef_construction: int = 40):
        self.dimension = dimension
        self.m = m
        self.ef_construction = ef_construction
        self.index = faiss.IndexHNSWFlat(dimension, m, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = ef_construction
        self.build_time_seconds = 0.0
        self.index_size_bytes = 0
        self.index_size_mb = 0.0
        self.current_ef_search = 16  # FAISS default

    @property
    def algorithm_name(self) -> str:
        return f"FAISS IndexHNSWFlat (M={self.m}, efC={self.ef_construction}, efS={self.current_ef_search})"

    def add(self, base_vectors: np.ndarray) -> float:
        """Adds base vectors and builds the HNSW graph index while measuring build time."""
        if base_vectors.dtype != np.float32:
            base_vectors = base_vectors.astype(np.float32)

        t_start = time.perf_counter()
        self.index.add(base_vectors)
        t_end = time.perf_counter()

        self.build_time_seconds = t_end - t_start

        # Measure serialized index size
        serialized_chunk = faiss.serialize_index(self.index)
        self.index_size_bytes = len(serialized_chunk)
        self.index_size_mb = self.index_size_bytes / (1024 * 1024)

        return self.build_time_seconds

    def set_ef_search(self, ef_search: int) -> None:
        """Dynamically adjusts the search beam width (efSearch)."""
        self.index.hnsw.efSearch = ef_search
        self.current_ef_search = ef_search

    def search(self, query_vectors: np.ndarray, k: int) -> np.ndarray:
        """Executes approximate nearest neighbor search."""
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)
        _, raw_labels = self.index.search(query_vectors, k)
        return raw_labels.astype(np.int32)


# ==============================================================================
# Benchmarking Engine
# ==============================================================================
def benchmark_configuration(
    searcher: VectorSearcher,
    query_vectors: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 10,
    repetitions: int = 5,
    sample_queries: int = 200,
    num_base_vectors: int = 100_000,
) -> Dict[str, Any]:
    """
    Runs a rigorous benchmark for a single searcher configuration:
    1. Steady-state batch search measurement (QPS, average latency)
    2. Prediction integrity validation (ID bounds, duplicates)
    3. Multi-K recall calculation (Recall@1, Recall@5, Recall@10)
    4. Individual query latency percentiles (P50, P95, P99)
    """
    # 1. Batch performance measurement
    batch_metrics = measure_batch_search(
        searcher=searcher,
        query_vectors=query_vectors,
        k=k,
        num_warmup_queries=100,
        repetitions=repetitions,
    )

    predictions = batch_metrics["last_predictions"]

    # 2. Prediction validation
    val_report = validate_predictions(
        predictions=predictions,
        k=k,
        num_base_vectors=num_base_vectors,
    )

    # 3. Accuracy metrics
    recall_1 = calculate_recall_at_k(ground_truth, predictions, k=1)
    recall_5 = calculate_recall_at_k(ground_truth, predictions, k=5)
    recall_10 = calculate_recall_at_k(ground_truth, predictions, k=10)

    # 4. Individual per-query latency percentiles
    percentiles = measure_individual_query_latencies(
        searcher=searcher,
        query_vectors=query_vectors,
        k=k,
        sample_queries=sample_queries,
    )

    return {
        "k": k,
        "recall_at_1": round(recall_1, 6),
        "recall_at_5": round(recall_5, 6),
        "recall_at_10": round(recall_10, 6),
        "batch_avg_latency_ms": batch_metrics["avg_latency_ms"],
        "batch_mean_time_sec": batch_metrics["mean_time_seconds"],
        "batch_min_time_sec": batch_metrics["min_time_seconds"],
        "batch_max_time_sec": batch_metrics["max_time_seconds"],
        "batch_runs": batch_metrics["run_times_seconds"],
        "qps": batch_metrics["qps"],
        "p50_latency_ms": percentiles["p50_ms"],
        "p95_latency_ms": percentiles["p95_ms"],
        "p99_latency_ms": percentiles["p99_ms"],
        "sample_size_individual": percentiles["sample_size"],
        "validation": val_report,
    }


# ==============================================================================
# Plotting Engine (Matplotlib Publication-Grade Visualizations)
# ==============================================================================
def generate_phase2_plots(
    results: List[Dict[str, Any]],
    output_dir: Path,
) -> List[Path]:
    """
    Generates publication-quality benchmark plots:
    1. Recall@10 vs efSearch
    2. Latency vs efSearch (batch and P95)
    3. QPS vs efSearch
    4. Recall@10 vs Latency (Trade-off Pareto frontier)
    5. Combined 4-panel comprehensive dashboard
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files: List[Path] = []

    # Separate exact reference from HNSW runs
    exact_res = next((r for r in results if r["method"] == "Exact FAISS"), None)
    hnsw_m16 = [r for r in results if r.get("m") == 16]
    hnsw_m32 = [r for r in results if r.get("m") == 32]

    color_m16 = "#1f77b4"  # Distinct Blue
    color_m32 = "#2ca02c"  # Distinct Green
    color_exact = "#d62728"  # Distinct Red

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 15,
        "lines.linewidth": 2.2,
        "lines.markersize": 7,
    })

    # --------------------------------------------------------------------------
    # Graph 1: Recall@10 vs efSearch
    # --------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 5.5), dpi=300)
    if hnsw_m16:
        ax1.plot(
            [r["ef_search"] for r in hnsw_m16],
            [r["recall_at_10"] for r in hnsw_m16],
            marker="o",
            label="HNSW (M=16)",
            color=color_m16,
        )
    if hnsw_m32:
        ax1.plot(
            [r["ef_search"] for r in hnsw_m32],
            [r["recall_at_10"] for r in hnsw_m32],
            marker="s",
            label="HNSW (M=32)",
            color=color_m32,
        )
    if exact_res:
        ax1.axhline(
            y=exact_res["recall_at_10"],
            color=color_exact,
            linestyle="--",
            label="Exact FAISS (1.000)",
        )

    ax1.set_xlabel("efSearch (Beam Search Depth)")
    ax1.set_ylabel("Recall@10")
    ax1.set_title("Graph 1: Recall@10 vs. efSearch")
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_xticks([10, 20, 40, 80, 160, 320])
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)
    fig1.tight_layout()
    p1 = output_dir / "recall_vs_efsearch.png"
    fig1.savefig(p1)
    plt.close(fig1)
    generated_files.append(p1)

    # --------------------------------------------------------------------------
    # Graph 2: Latency vs efSearch
    # --------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 5.5), dpi=300)
    if hnsw_m16:
        ax2.plot(
            [r["ef_search"] for r in hnsw_m16],
            [r["batch_avg_latency_ms"] for r in hnsw_m16],
            marker="o",
            label="HNSW M=16 (Batch Avg Latency)",
            color=color_m16,
        )
        ax2.plot(
            [r["ef_search"] for r in hnsw_m16],
            [r["p95_latency_ms"] for r in hnsw_m16],
            marker="^",
            linestyle=":",
            label="HNSW M=16 (P95 Per-Query)",
            color=color_m16,
            alpha=0.75,
        )
    if hnsw_m32:
        ax2.plot(
            [r["ef_search"] for r in hnsw_m32],
            [r["batch_avg_latency_ms"] for r in hnsw_m32],
            marker="s",
            label="HNSW M=32 (Batch Avg Latency)",
            color=color_m32,
        )
        ax2.plot(
            [r["ef_search"] for r in hnsw_m32],
            [r["p95_latency_ms"] for r in hnsw_m32],
            marker="v",
            linestyle=":",
            label="HNSW M=32 (P95 Per-Query)",
            color=color_m32,
            alpha=0.75,
        )
    if exact_res:
        ax2.axhline(
            y=exact_res["batch_avg_latency_ms"],
            color=color_exact,
            linestyle="--",
            label=f"Exact FAISS Batch Avg ({exact_res['batch_avg_latency_ms']:.3f} ms)",
        )
        ax2.axhline(
            y=exact_res["p95_latency_ms"],
            color=color_exact,
            linestyle=":",
            label=f"Exact FAISS P95 ({exact_res['p95_latency_ms']:.2f} ms)",
        )

    ax2.set_xlabel("efSearch (Beam Search Depth)")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Graph 2: Search Latency vs. efSearch")
    ax2.set_xticks([10, 20, 40, 80, 160, 320])
    ax2.legend(loc="upper left", frameon=True, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.6)
    fig2.tight_layout()
    p2 = output_dir / "latency_vs_efsearch.png"
    fig2.savefig(p2)
    plt.close(fig2)
    generated_files.append(p2)

    # --------------------------------------------------------------------------
    # Graph 3: QPS vs efSearch
    # --------------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(8, 5.5), dpi=300)
    if hnsw_m16:
        ax3.plot(
            [r["ef_search"] for r in hnsw_m16],
            [r["qps"] for r in hnsw_m16],
            marker="o",
            label="HNSW (M=16)",
            color=color_m16,
        )
    if hnsw_m32:
        ax3.plot(
            [r["ef_search"] for r in hnsw_m32],
            [r["qps"] for r in hnsw_m32],
            marker="s",
            label="HNSW (M=32)",
            color=color_m32,
        )
    if exact_res:
        ax3.axhline(
            y=exact_res["qps"],
            color=color_exact,
            linestyle="--",
            label=f"Exact FAISS ({exact_res['qps']:.0f} QPS)",
        )

    ax3.set_xlabel("efSearch (Beam Search Depth)")
    ax3.set_ylabel("Throughput (QPS - Queries Per Second)")
    ax3.set_title("Graph 3: Search Throughput (QPS) vs. efSearch")
    ax3.set_xticks([10, 20, 40, 80, 160, 320])
    ax3.legend(loc="upper right", frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.6)
    fig3.tight_layout()
    p3 = output_dir / "qps_vs_efsearch.png"
    fig3.savefig(p3)
    plt.close(fig3)
    generated_files.append(p3)

    # --------------------------------------------------------------------------
    # Graph 4: Recall@10 vs Latency (Pareto Frontier Trade-Off)
    # --------------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(8.5, 6), dpi=300)
    if hnsw_m16:
        ax4.plot(
            [r["batch_avg_latency_ms"] for r in hnsw_m16],
            [r["recall_at_10"] for r in hnsw_m16],
            marker="o",
            label="HNSW (M=16)",
            color=color_m16,
        )
        for r in hnsw_m16:
            ax4.annotate(
                f"ef={r['ef_search']}",
                (r["batch_avg_latency_ms"], r["recall_at_10"]),
                textcoords="offset points",
                xytext=(-10, 8),
                fontsize=8,
                color=color_m16,
            )

    if hnsw_m32:
        ax4.plot(
            [r["batch_avg_latency_ms"] for r in hnsw_m32],
            [r["recall_at_10"] for r in hnsw_m32],
            marker="s",
            label="HNSW (M=32)",
            color=color_m32,
        )
        for r in hnsw_m32:
            ax4.annotate(
                f"ef={r['ef_search']}",
                (r["batch_avg_latency_ms"], r["recall_at_10"]),
                textcoords="offset points",
                xytext=(8, -10),
                fontsize=8,
                color=color_m32,
            )

    if exact_res:
        ax4.scatter(
            [exact_res["batch_avg_latency_ms"]],
            [exact_res["recall_at_10"]],
            marker="*",
            s=220,
            color=color_exact,
            label="Exact FAISS (Reference Oracle)",
            zorder=5,
        )
        ax4.annotate(
            "Exact FAISS\n(100% Recall)",
            (exact_res["batch_avg_latency_ms"], exact_res["recall_at_10"]),
            textcoords="offset points",
            xytext=(-45, -28),
            fontsize=9,
            fontweight="bold",
            color=color_exact,
        )

    ax4.set_xlabel("Batch Average Latency (ms/query)")
    ax4.set_ylabel("Recall@10")
    ax4.set_title("Graph 4: Recall@10 vs. Latency Trade-Off (Pareto Frontier)")
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(loc="lower right", frameon=True)
    ax4.grid(True, linestyle="--", alpha=0.6)
    fig4.tight_layout()
    p4 = output_dir / "recall_vs_latency_tradeoff.png"
    fig4.savefig(p4)
    plt.close(fig4)
    generated_files.append(p4)

    # --------------------------------------------------------------------------
    # Graph 5: Combined 4-Panel Dashboard
    # --------------------------------------------------------------------------
    fig_dash, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    fig_dash.suptitle("Case 04 Phase 2: HNSW ANN Baseline vs. Exact FAISS Benchmark", fontsize=16, y=0.98)

    # Top-Left: Recall@10 vs efSearch
    ax = axes[0, 0]
    if hnsw_m16:
        ax.plot([r["ef_search"] for r in hnsw_m16], [r["recall_at_10"] for r in hnsw_m16], "o-", color=color_m16, label="HNSW (M=16)")
    if hnsw_m32:
        ax.plot([r["ef_search"] for r in hnsw_m32], [r["recall_at_10"] for r in hnsw_m32], "s-", color=color_m32, label="HNSW (M=32)")
    if exact_res:
        ax.axhline(y=exact_res["recall_at_10"], color=color_exact, linestyle="--", label="Exact FAISS (1.00)")
    ax.set_title("Recall@10 vs. efSearch", fontweight="bold")
    ax.set_xlabel("efSearch")
    ax.set_ylabel("Recall@10")
    ax.set_xticks([10, 20, 40, 80, 160, 320])
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Top-Right: Latency vs efSearch
    ax = axes[0, 1]
    if hnsw_m16:
        ax.plot([r["ef_search"] for r in hnsw_m16], [r["batch_avg_latency_ms"] for r in hnsw_m16], "o-", color=color_m16, label="HNSW M=16 (Avg)")
        ax.plot([r["ef_search"] for r in hnsw_m16], [r["p95_latency_ms"] for r in hnsw_m16], "^:", color=color_m16, alpha=0.7, label="HNSW M=16 (P95)")
    if hnsw_m32:
        ax.plot([r["ef_search"] for r in hnsw_m32], [r["batch_avg_latency_ms"] for r in hnsw_m32], "s-", color=color_m32, label="HNSW M=32 (Avg)")
        ax.plot([r["ef_search"] for r in hnsw_m32], [r["p95_latency_ms"] for r in hnsw_m32], "v:", color=color_m32, alpha=0.7, label="HNSW M=32 (P95)")
    if exact_res:
        ax.axhline(y=exact_res["batch_avg_latency_ms"], color=color_exact, linestyle="--", label=f"Exact Avg ({exact_res['batch_avg_latency_ms']:.2f}ms)")
    ax.set_title("Search Latency vs. efSearch", fontweight="bold")
    ax.set_xlabel("efSearch")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks([10, 20, 40, 80, 160, 320])
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Bottom-Left: QPS vs efSearch
    ax = axes[1, 0]
    if hnsw_m16:
        ax.plot([r["ef_search"] for r in hnsw_m16], [r["qps"] for r in hnsw_m16], "o-", color=color_m16, label="HNSW (M=16)")
    if hnsw_m32:
        ax.plot([r["ef_search"] for r in hnsw_m32], [r["qps"] for r in hnsw_m32], "s-", color=color_m32, label="HNSW (M=32)")
    if exact_res:
        ax.axhline(y=exact_res["qps"], color=color_exact, linestyle="--", label=f"Exact FAISS ({exact_res['qps']:.0f} QPS)")
    ax.set_title("Throughput (QPS) vs. efSearch", fontweight="bold")
    ax.set_xlabel("efSearch")
    ax.set_ylabel("QPS (Queries / Sec)")
    ax.set_xticks([10, 20, 40, 80, 160, 320])
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Bottom-Right: Recall@10 vs Latency (Pareto Frontier)
    ax = axes[1, 1]
    if hnsw_m16:
        ax.plot([r["batch_avg_latency_ms"] for r in hnsw_m16], [r["recall_at_10"] for r in hnsw_m16], "o-", color=color_m16, label="HNSW (M=16)")
    if hnsw_m32:
        ax.plot([r["batch_avg_latency_ms"] for r in hnsw_m32], [r["recall_at_10"] for r in hnsw_m32], "s-", color=color_m32, label="HNSW (M=32)")
    if exact_res:
        ax.scatter([exact_res["batch_avg_latency_ms"]], [exact_res["recall_at_10"]], marker="*", s=220, color=color_exact, label="Exact FAISS", zorder=5)
    ax.set_title("Recall@10 vs. Latency Trade-Off", fontweight="bold")
    ax.set_xlabel("Batch Avg Latency (ms/query)")
    ax.set_ylabel("Recall@10")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig_dash.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdash = output_dir / "combined_hnsw_dashboard.png"
    fig_dash.savefig(pdash)
    plt.close(fig_dash)
    generated_files.append(pdash)

    return generated_files


# ==============================================================================
# Main Orchestrator
# ==============================================================================
def main() -> int:
    workspace_root = Path(__file__).resolve().parent.parent
    data_dir = workspace_root / "data"
    results_dir = workspace_root / "results" / "phase2"
    raw_dir = results_dir / "raw"
    summaries_dir = results_dir / "summaries"
    plots_dir = results_dir / "plots"

    for d in [raw_dir, summaries_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("CASE 04 — PHASE 2: ANN INDEX + BASELINE APPROXIMATE VECTOR SEARCH")
    print("FAISS IndexHNSWFlat Benchmark vs. Exact FAISS Ground Truth")
    print("================================================================================")

    # 1. Dataset Verification & Invariant Guard
    print("\n[Step 1/6] Verifying Phase 1 Dataset Invariants & Cryptographic Checksums...")
    base_path = data_dir / "base_vectors.fvecs"
    query_path = data_dir / "query_vectors.fvecs"
    gt_path = data_dir / "ground_truth.ivecs"
    gt_dist_path = data_dir / "ground_truth_distances.fvecs"

    target_files = [base_path, query_path, gt_path, gt_dist_path]
    for p in target_files:
        if not p.exists():
            print(f"CRITICAL ERROR: Required dataset file missing: {p}", file=sys.stderr)
            return 1

    actual_checksums = {p.name: compute_sha256(p) for p in target_files}
    checksum_failures = 0
    for fname, expected_hash in EXPECTED_CHECKSUMS.items():
        actual_hash = actual_checksums.get(fname, "")
        if actual_hash != expected_hash:
            print(f"FAILED CHECKSUM for {fname}!\n  Expected: {expected_hash}\n  Actual:   {actual_hash}")
            checksum_failures += 1
        else:
            print(f"  [PASS] {fname}: SHA-256 verified.")

    if checksum_failures > 0:
        print("CRITICAL ERROR: Phase 1 dataset corruption detected. Halting Phase 2.", file=sys.stderr)
        return 1

    # 2. Load Binary Assets
    print("\n[Step 2/6] Loading Base Vectors, Query Vectors, and Exact Ground Truth...")
    t0_load = time.perf_counter()
    base_vectors = load_fvecs(base_path)
    query_vectors = load_fvecs(query_path)
    ground_truth = load_ivecs(gt_path)
    t1_load = time.perf_counter()

    n_base, d_base = base_vectors.shape
    n_query, d_query = query_vectors.shape
    n_gt, k_gt = ground_truth.shape

    print(f"  Base vectors : {n_base:,} × {d_base} (float32)")
    print(f"  Queries      : {n_query:,} × {d_query} (float32)")
    print(f"  Ground truth : {n_gt:,} × {k_gt} (int32)")
    print(f"  Load time    : {t1_load - t0_load:.3f} sec")

    if n_base != 100_000 or d_base != 128 or n_query != 1_000 or d_query != 128 or k_gt != 10:
        print("CRITICAL ERROR: Dataset dimensions do not match benchmark specification!", file=sys.stderr)
        return 1

    results_table: List[Dict[str, Any]] = []

    # 3. Reference Exact FAISS Oracle
    print("\n[Step 3/6] Running Exact FAISS Reference Baseline (IndexFlatL2)...")
    exact_searcher = FaissIndexFlatL2Searcher(dimension=d_base)
    t0_exact_build = time.perf_counter()
    exact_searcher.add(base_vectors)
    exact_build_time = time.perf_counter() - t0_exact_build
    exact_size_bytes = n_base * d_base * 4  # Raw flat vectors
    exact_size_mb = exact_size_bytes / (1024 * 1024)

    exact_res = benchmark_configuration(
        searcher=exact_searcher,
        query_vectors=query_vectors,
        ground_truth=ground_truth,
        k=10,
        repetitions=5,
        sample_queries=200,
        num_base_vectors=n_base,
    )

    exact_record = {
        "method": "Exact FAISS",
        "index_type": "IndexFlatL2",
        "m": None,
        "ef_construction": None,
        "ef_search": None,
        "build_time_sec": round(exact_build_time, 4),
        "index_size_mb": round(exact_size_mb, 2),
        "recall_at_1": exact_res["recall_at_1"],
        "recall_at_5": exact_res["recall_at_5"],
        "recall_at_10": exact_res["recall_at_10"],
        "batch_avg_latency_ms": exact_res["batch_avg_latency_ms"],
        "p50_latency_ms": exact_res["p50_latency_ms"],
        "p95_latency_ms": exact_res["p95_latency_ms"],
        "p99_latency_ms": exact_res["p99_latency_ms"],
        "qps": exact_res["qps"],
        "validation_pass": exact_res["validation"]["is_valid"] and (exact_res["recall_at_10"] == 1.0),
    }
    results_table.append(exact_record)

    with open(raw_dir / "exact_faiss.json", "w", encoding="utf-8") as f:
        json.dump({**exact_record, "raw_details": exact_res}, f, indent=2)

    print(f"  Exact FAISS: Recall@10 = {exact_record['recall_at_10']:.4f} | "
          f"Avg Latency = {exact_record['batch_avg_latency_ms']:.4f} ms | "
          f"P95 = {exact_record['p95_latency_ms']:.4f} ms | "
          f"QPS = {exact_record['qps']:.1f}")

    # 4. HNSW Search Experiments
    m_values = [16, 32]
    ef_search_values = [10, 20, 40, 80, 160, 320]
    ef_construction_default = 40  # Standard FAISS HNSW default

    print("\n[Step 4/6] Executing HNSW Multi-Configuration Benchmarks...")

    for m in m_values:
        print(f"\n--- Constructing HNSW Graph (M={m}, efConstruction={ef_construction_default}) ---")
        hnsw_searcher = FaissHNSWSearcher(
            dimension=d_base,
            m=m,
            ef_construction=ef_construction_default,
        )
        build_time = hnsw_searcher.add(base_vectors)
        print(f"  Graph Build Time: {build_time:.3f} sec | Index Size: {hnsw_searcher.index_size_mb:.2f} MB")

        for ef in ef_search_values:
            hnsw_searcher.set_ef_search(ef)
            res = benchmark_configuration(
                searcher=hnsw_searcher,
                query_vectors=query_vectors,
                ground_truth=ground_truth,
                k=10,
                repetitions=5,
                sample_queries=200,
                num_base_vectors=n_base,
            )

            record = {
                "method": f"HNSW (M={m}, ef={ef})",
                "index_type": "IndexHNSWFlat",
                "m": m,
                "ef_construction": ef_construction_default,
                "ef_search": ef,
                "build_time_sec": round(build_time, 4),
                "index_size_mb": round(hnsw_searcher.index_size_mb, 2),
                "recall_at_1": res["recall_at_1"],
                "recall_at_5": res["recall_at_5"],
                "recall_at_10": res["recall_at_10"],
                "batch_avg_latency_ms": res["batch_avg_latency_ms"],
                "p50_latency_ms": res["p50_latency_ms"],
                "p95_latency_ms": res["p95_latency_ms"],
                "p99_latency_ms": res["p99_latency_ms"],
                "qps": res["qps"],
                "validation_pass": res["validation"]["is_valid"],
            }
            results_table.append(record)

            raw_file = raw_dir / f"hnsw_m{m}_ef{ef}.json"
            with open(raw_file, "w", encoding="utf-8") as f:
                json.dump({**record, "raw_details": res}, f, indent=2)

            print(f"  M={m:2d} | ef={ef:3d} => "
                  f"Recall@1={record['recall_at_1']:.4f} | "
                  f"Recall@5={record['recall_at_5']:.4f} | "
                  f"Recall@10={record['recall_at_10']:.4f} | "
                  f"Avg Lat={record['batch_avg_latency_ms']:.4f} ms | "
                  f"P95={record['p95_latency_ms']:.4f} ms | "
                  f"QPS={record['qps']:7.1f} | "
                  f"Valid={record['validation_pass']}")

    # 5. Save Summaries & Tables
    print("\n[Step 5/6] Serializing Summaries and Tabular Comparison Assets...")

    # CSV Summary
    csv_path = summaries_dir / "comparison_table.csv"
    csv_columns = [
        "method", "index_type", "m", "ef_construction", "ef_search",
        "recall_at_1", "recall_at_5", "recall_at_10",
        "batch_avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
        "qps", "build_time_sec", "index_size_mb", "validation_pass"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for r in results_table:
            writer.writerow({k: r[k] for k in csv_columns})
    print(f"  Saved CSV table: {csv_path}")

    # JSON Comprehensive Summary
    summary_path = summaries_dir / "hnsw_benchmark_summary.json"
    summary_json = {
        "phase": "Phase 2",
        "objective": "ANN Index + Baseline Approximate Vector Search",
        "dataset": {
            "num_base_vectors": n_base,
            "num_queries": n_query,
            "dimension": d_base,
            "k": 10,
        },
        "exact_reference": exact_record,
        "configurations": results_table,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)
    print(f"  Saved JSON summary: {summary_path}")

    # 6. Generate Publication Graphs
    print("\n[Step 6/6] Generating Publication-Grade Plots...")
    generated_plots = generate_phase2_plots(results_table, plots_dir)
    for p in generated_plots:
        print(f"  Generated Plot: {p}")

    # Print Formatted Markdown Table for Report
    print("\n" + "=" * 95)
    print("PHASE 2 BENCHMARK RESULTS TABLE")
    print("=" * 95)
    header = f"| {'Method':<20} | {'Recall@10':<9} | {'Recall@5':<8} | {'Recall@1':<8} | {'Avg Lat (ms)':<12} | {'P95 (ms)':<9} | {'QPS':<8} | {'Build (s)':<9} | {'Size (MB)':<9} |"
    sep = f"|{'-'*22}|{'-'*11}|{'-'*10}|{'-'*10}|{'-'*14}|{'-'*11}|{'-'*10}|{'-'*11}|{'-'*11}|"
    print(header)
    print(sep)
    for r in results_table:
        m_str = r["method"]
        r10 = f"{r['recall_at_10']:.4f}"
        r5 = f"{r['recall_at_5']:.4f}"
        r1 = f"{r['recall_at_1']:.4f}"
        avg_lat = f"{r['batch_avg_latency_ms']:.4f}"
        p95 = f"{r['p95_latency_ms']:.4f}"
        qps = f"{r['qps']:.1f}"
        btime = f"{r['build_time_sec']:.2f}"
        sz = f"{r['index_size_mb']:.2f}"
        print(f"| {m_str:<20} | {r10:>9} | {r5:>8} | {r1:>8} | {avg_lat:>12} | {p95:>9} | {qps:>8} | {btime:>9} | {sz:>9} |")
    print("=" * 95)

    all_valid = all(r["validation_pass"] for r in results_table)
    if all_valid:
        print("\nPHASE 2 EXECUTION: SUCCESS (All validation criteria satisfied)")
        return 0
    else:
        print("\nPHASE 2 EXECUTION: FAILED validation checks!", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
