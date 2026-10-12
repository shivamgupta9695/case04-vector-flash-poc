"""
Case 04 — Phase 3 Hardening Diagnostic
Investigates the 10% DRAM + Uniform latency distribution and anomaly.
Generates per-query CSV, latency histogram, and latency CDF.
"""

import sys
import time
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from src.phase1_2_evaluation import load_fvecs, load_ivecs
from src.phase3_ssd_store import SSDVectorStore
from src.phase3_hotness import AccessTracker
from src.phase3_tiering import TieredVectorManager
from src.phase3_workload import WorkloadGenerator
import faiss


def run_diagnostic():
    data_dir = workspace_root / "data"
    storage_dir = workspace_root / "storage"
    diag_dir = workspace_root / "results" / "phase3_hardened" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    print("[Diagnostic] Loading datasets...")
    base_vectors = load_fvecs(data_dir / "base_vectors.fvecs")
    query_vectors = load_fvecs(data_dir / "query_vectors.fvecs")
    ground_truth = load_ivecs(data_dir / "ground_truth.ivecs")

    ssd_store = SSDVectorStore(storage_dir, num_vectors=len(base_vectors), dimension=base_vectors.shape[1])
    ssd_store.open()

    print("[Diagnostic] Building shared HNSW candidate index (M=32, ef=40)...")
    index = faiss.IndexHNSWFlat(base_vectors.shape[1], 32, faiss.METRIC_L2)
    index.hnsw.efSearch = 40
    index.add(base_vectors)

    tracker = AccessTracker(num_vectors=len(base_vectors))
    tier_manager = TieredVectorManager(
        ssd_store=ssd_store,
        access_tracker=tracker,
        dram_capacity_ratio=0.10,  # 10% DRAM
        dimension=base_vectors.shape[1],
    )
    tier_manager.populate_initial_dram(list(range(10_000)))

    workload_gen = WorkloadGenerator(query_vectors, seed=42)
    uniform_stream = workload_gen.generate_uniform_stream(stream_length=1000)

    csv_path = diag_dir / "10pct_uniform_latencies.csv"
    records = []

    print("[Diagnostic] Executing 1,000 queries with microsecond timing...")
    current_sim_time = 0.0
    warmup_count = 50

    for i, q_idx in enumerate(uniform_stream):
        current_sim_time += 0.001
        q_vec = query_vectors[q_idx].reshape(1, -1).astype(np.float32)

        t_start_ns = time.perf_counter_ns()

        # Step 1: HNSW search
        t_hnsw_0 = time.perf_counter_ns()
        _, raw_cand_ids = index.search(q_vec, 40)
        candidates = raw_cand_ids[0].tolist()
        hnsw_ns = time.perf_counter_ns() - t_hnsw_0

        # Step 2: Tiered candidate vector retrieval
        t_tier_0 = time.perf_counter_ns()
        prom_before = tier_manager.promotion_count
        dem_before = tier_manager.demotion_count
        ssd_reads_before = ssd_store.total_reads
        ssd_time_before = ssd_store.total_read_time_seconds

        cand_vectors, dram_hits, ssd_misses = tier_manager.get_vectors_for_candidates(
            candidates, current_sim_time
        )
        tier_ns = time.perf_counter_ns() - t_tier_0

        prom_delta = tier_manager.promotion_count - prom_before
        dem_delta = tier_manager.demotion_count - dem_before
        ssd_reads_delta = ssd_store.total_reads - ssd_reads_before
        ssd_time_delta_us = (ssd_store.total_read_time_seconds - ssd_time_before) * 1e6

        # Step 3: Exact candidate reranking
        t_rerank_0 = time.perf_counter_ns()
        diff = cand_vectors - q_vec
        dists = np.sum(diff * diff, axis=1)
        top_k_indices = np.argsort(dists)[:10]
        rerank_ns = time.perf_counter_ns() - t_rerank_0

        t_end_ns = time.perf_counter_ns()
        total_latency_us = (t_end_ns - t_start_ns) / 1000.0

        if i > 0 and i % 200 == 0:
            tracker.decay_recent_counts(decay_rate=0.88)

        record = {
            "query_id": i,
            "stream_query_idx": q_idx,
            "is_warmup": i < warmup_count,
            "latency_us": round(total_latency_us, 2),
            "latency_ms": round(total_latency_us / 1000.0, 4),
            "hnsw_latency_us": round(hnsw_ns / 1000.0, 2),
            "tier_lookup_latency_us": round(tier_ns / 1000.0, 2),
            "rerank_latency_us": round(rerank_ns / 1000.0, 2),
            "dram_hit": dram_hits,
            "ssd_read": ssd_reads_delta,
            "ssd_read_latency_us": round(ssd_time_delta_us, 2),
            "promotion": prom_delta,
            "demotion": dem_delta,
        }
        records.append(record)

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "query_id", "stream_query_idx", "is_warmup", "latency_us", "latency_ms",
            "hnsw_latency_us", "tier_lookup_latency_us", "rerank_latency_us",
            "dram_hit", "ssd_read", "ssd_read_latency_us", "promotion", "demotion"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"[Diagnostic] Saved CSV to {csv_path}")

    # Analyze samples (post warmup)
    measured = [r for r in records if not r["is_warmup"]]
    lats_us = np.array([r["latency_us"] for r in measured], dtype=np.float64)
    lats_ms = lats_us / 1000.0

    min_ms = float(np.min(lats_ms))
    p50_ms = float(np.percentile(lats_ms, 50))
    p90_ms = float(np.percentile(lats_ms, 90))
    p95_ms = float(np.percentile(lats_ms, 95))
    p99_ms = float(np.percentile(lats_ms, 99))
    max_ms = float(np.max(lats_ms))
    mean_ms = float(np.mean(lats_ms))
    std_ms = float(np.std(lats_ms))
    ratio = mean_ms / p95_ms if p95_ms > 0 else 0.0

    stats = {
        "sample_count": len(measured),
        "min_ms": round(min_ms, 4),
        "p50_ms": round(p50_ms, 4),
        "p90_ms": round(p90_ms, 4),
        "p95_ms": round(p95_ms, 4),
        "p99_ms": round(p99_ms, 4),
        "max_ms": round(max_ms, 4),
        "mean_ms": round(mean_ms, 4),
        "stddev_ms": round(std_ms, 4),
        "mean_over_p95_ratio": round(ratio, 4),
        "total_promotions": sum(r["promotion"] for r in measured),
        "total_demotions": sum(r["demotion"] for r in measured),
        "total_ssd_reads": sum(r["ssd_read"] for r in measured),
        "total_dram_hits": sum(r["dram_hit"] for r in measured),
    }

    stats_path = diag_dir / "10pct_uniform_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[Diagnostic] Stats: {json.dumps(stats, indent=2)}")

    # Plot 1: Latency Histogram
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    ax.hist(lats_ms, bins=50, color="#1f77b4", edgecolor="black", alpha=0.75)
    ax.axvline(mean_ms, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_ms:.2f} ms")
    ax.axvline(p50_ms, color="green", linestyle="-.", linewidth=2, label=f"P50: {p50_ms:.2f} ms")
    ax.axvline(p95_ms, color="purple", linestyle=":", linewidth=2, label=f"P95: {p95_ms:.2f} ms")
    ax.axvline(p99_ms, color="orange", linestyle="-", linewidth=1.5, label=f"P99: {p99_ms:.2f} ms")
    ax.set_xlabel("Query Latency (ms)")
    ax.set_ylabel("Query Count")
    ax.set_title("10% DRAM + Uniform: Latency Distribution Histogram (N=950)")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    hist_path = diag_dir / "10pct_uniform_histogram.png"
    fig.savefig(hist_path)
    plt.close(fig)
    print(f"[Diagnostic] Saved histogram to {hist_path}")

    # Plot 2: Latency CDF
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    sorted_lats = np.sort(lats_ms)
    cdf = np.arange(1, len(sorted_lats) + 1) / len(sorted_lats)
    ax.plot(sorted_lats, cdf, color="#1f77b4", linewidth=2.5, label="Empirical Latency CDF")
    ax.axvline(p50_ms, color="green", linestyle="-.", label=f"P50: {p50_ms:.2f} ms")
    ax.axvline(p95_ms, color="purple", linestyle=":", label=f"P95: {p95_ms:.2f} ms")
    ax.axvline(p99_ms, color="orange", linestyle="-", label=f"P99: {p99_ms:.2f} ms")
    ax.axvline(mean_ms, color="red", linestyle="--", label=f"Mean: {mean_ms:.2f} ms")
    ax.set_xlabel("Query Latency (ms)")
    ax.set_ylabel("Cumulative Probability P(X <= x)")
    ax.set_title("10% DRAM + Uniform: Cumulative Distribution Function (CDF)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    cdf_path = diag_dir / "10pct_uniform_cdf.png"
    fig.savefig(cdf_path)
    plt.close(fig)
    print(f"[Diagnostic] Saved CDF to {cdf_path}")


if __name__ == "__main__":
    run_diagnostic()
