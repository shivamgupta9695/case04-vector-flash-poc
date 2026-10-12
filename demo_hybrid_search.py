"""
Case 04 — Mentor-Facing Live Hybrid Vector Search Demo
Executes a single genuine query (q_0) through the existing two-tier hybrid
vector retrieval pipeline (FAISS HNSW in DRAM + Microsoft DiskANN on NVMe SSD)
with exact FP32 squared-L2 reranking.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Optional

# 1. Resolve project directory relative to this script
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import numpy as np
except ImportError as err:
    print(f"[FATAL] Failed to import NumPy: {err}", file=sys.stderr)
    sys.exit(1)

try:
    from src.phase1_2_evaluation import load_fvecs
    from src.phase5_diskann_search import ColdDiskANNIndex
    from src.phase6_hybrid_search import (
        HotHNSWIndex,
        TieredVectorStore,
        HybridVectorPipeline,
        HOT_VECTOR_COUNT,
        COLD_VECTOR_COUNT,
        TOTAL_VECTORS,
        DIMENSION,
    )
except ImportError as err:
    print(f"[FATAL] Failed to import project modules: {err}", file=sys.stderr)
    sys.exit(1)


def run_demo() -> int:
    """Executes a single live hybrid search query with rigorous validation."""
    print("=" * 78)
    print(" CASE 04: HYBRID TWO-TIER VECTOR SEARCH — LIVE MENTOR DEMO")
    print("=" * 78)

    # Required artifact paths
    data_dir = PROJECT_ROOT / "data"
    storage_dir = PROJECT_ROOT / "storage"
    query_file = data_dir / "query_vectors.fvecs"
    base_file = data_dir / "base_vectors.fvecs"
    hot_index_file = PROJECT_ROOT / "results/phase6_hybrid/index/hot_hnsw.index"
    cold_prefix = PROJECT_ROOT / "results/phase5_diskann/index/cold_disk_index_tuned"

    # Verify prerequisite files exist
    required_files = [
        (query_file, "Query vectors dataset"),
        (base_file, "Base vectors dataset"),
        (hot_index_file, "HOT HNSW index"),
        (Path(str(cold_prefix) + "_disk.index"), "COLD DiskANN disk index"),
        (Path(str(cold_prefix) + "_pq_compressed.bin"), "COLD DiskANN PQ table"),
        (Path(str(cold_prefix) + "_pq_pivots.bin"), "COLD DiskANN PQ pivots"),
        (storage_dir / "ssd_vectors/vectors.bin", "Physical SSD vector binary"),
    ]

    for path, desc in required_files:
        if not path.is_file():
            print(f"[ERROR] Required artifact missing: {desc} at {path}", file=sys.stderr)
            return 2

    # Step 1: Load and validate first query vector
    print(f"\n[1/5] Loading query vector from {query_file.name}...")
    queries = load_fvecs(query_file)
    if queries.shape[1] != DIMENSION:
        print(f"[ERROR] Expected dimension {DIMENSION}, got {queries.shape[1]}", file=sys.stderr)
        return 3

    query_idx = 0
    q0 = queries[query_idx : query_idx + 1]
    if q0.dtype != np.float32:
        q0 = q0.astype(np.float32)

    if not np.all(np.isfinite(q0)):
        print("[ERROR] Query vector contains non-finite values (NaN/Inf).", file=sys.stderr)
        return 4

    print(f"      Query Index: {query_idx}")
    print(f"      Dimension:   {q0.shape[1]}")
    print(f"      Data Type:   {q0.dtype}")
    print(f"      Query Norm:  ||q||_2 = {float(np.linalg.norm(q0)):.4f}")

    # Step 2: Initialize engine components
    print("\n[2/5] Initializing Hybrid Vector Engine components...")
    vector_store: Optional[TieredVectorStore] = None
    cold_index: Optional[ColdDiskANNIndex] = None

    try:
        # 2a. Base vectors for HOT tier initialization
        base_vectors = load_fvecs(base_file)

        # 2b. HOT tier index (18k vectors in DRAM via FAISS HNSW)
        hot_index = HotHNSWIndex(index_path=hot_index_file)
        hot_index.build_or_load(base_vectors[:HOT_VECTOR_COUNT])
        print(f"      HOT Tier:  FAISS HNSW ({HOT_VECTOR_COUNT:,} vectors in DRAM, M=32, efConstruction=40)")

        # 2c. TieredVectorStore (DRAM cache for HOT + unbuffered NVMe file for COLD)
        vector_store = TieredVectorStore(
            storage_dir=storage_dir,
            base_vectors_path=base_file,
        )
        vector_store.open()
        print(f"      Storage:   TieredVectorStore online (18k DRAM hit buffer + 82k NVMe direct read)")

        # 2d. COLD tier index (82k vectors on NVMe via Microsoft DiskANN3)
        cold_index = ColdDiskANNIndex(cold_prefix)
        print(f"      COLD Tier: Microsoft DiskANN3 ({COLD_VECTOR_COUNT:,} vectors on NVMe, L=100, beam=4)")

        # 2e. Hybrid Vector Pipeline Coordinator
        pipeline = HybridVectorPipeline(
            hot_index=hot_index,
            cold_index=cold_index,
            vector_store=vector_store,
        )

        # Step 3: Execute single query search
        k = 10
        k_hot = 50
        k_cold = 50
        print(f"\n[3/5] Executing live query through Hybrid Vector Pipeline...")
        print(f"      Target Top-K:       {k}")
        print(f"      Routing Policy:     ALL_TIERS (Broadcast to HOT DRAM + COLD NVMe)")
        print(f"      Candidate Probes:   k_hot={k_hot} (FAISS DRAM), k_cold={k_cold} (DiskANN SSD)")
        print(f"      Reranking Strategy: Exact FP32 squared-L2 recomputation from storage")

        t_start_ns = time.perf_counter_ns()
        result = pipeline.search(
            queries=q0,
            k=k,
            k_hot=k_hot,
            k_cold=k_cold,
            ef_search_hot=64,
            l_search_cold=100,
            beam_width_cold=4,
        )
        total_latency_ms = (time.perf_counter_ns() - t_start_ns) / 1_000_000.0

        # Step 4: Extract and validate results
        print("\n[4/5] Search Results & Top-10 Ranking:")
        print("-" * 78)
        print(f" {'Rank':<6} {'Global ID':<12} {'Tier Residency':<18} {'Squared-L2 Distance (||q - x||^2)':<35}")
        print("-" * 78)

        ret_ids = result.global_ids[0]
        ret_dists = result.distances[0]

        # Validation assertions
        if len(ret_ids) != k:
            print(f"[ERROR] Expected {k} returned IDs, got {len(ret_ids)}", file=sys.stderr)
            return 5

        if len(set(ret_ids.tolist())) != k:
            print("[ERROR] Returned IDs contain duplicate candidates.", file=sys.stderr)
            return 6

        if not np.all(np.isfinite(ret_dists)):
            print("[ERROR] Returned distances contain non-finite values.", file=sys.stderr)
            return 7

        if not np.all(ret_dists >= 0.0):
            print("[ERROR] Returned distances contain negative values.", file=sys.stderr)
            return 8

        # Monotonic non-decreasing check
        diffs = np.diff(ret_dists)
        if np.any(diffs < -1e-5):
            print("[ERROR] Distances are not monotonically sorted.", file=sys.stderr)
            return 9

        for rank, (gid, dist) in enumerate(zip(ret_ids, ret_dists), start=1):
            if gid < 0 or gid >= TOTAL_VECTORS:
                print(f"[ERROR] Global ID {gid} out of bounds [0, {TOTAL_VECTORS - 1}]", file=sys.stderr)
                return 10

            tier_name = "HOT (DRAM)" if gid < HOT_VECTOR_COUNT else "COLD (NVMe SSD)"
            print(f" #{rank:<5} {gid:<12} {tier_name:<18} {dist:<35.4f}")

        print("-" * 78)

        # Step 5: Report actual measured statistics
        print("\n[5/5] Execution & Telemetry Summary:")
        print(f"      End-to-End Latency:      {total_latency_ms:.2f} ms")
        print(f"      HOT Search Time (DRAM):  {result.hot_search_ms:.2f} ms")
        print(f"      COLD Search Time (NVMe): {result.cold_search_ms:.2f} ms")
        print(f"      Exact FP32 Rerank Time:  {result.rerank_ms:.2f} ms")
        print(f"      Raw Candidates Merged:   {result.raw_candidate_counts[0]} ({k_hot} HOT + {k_cold} COLD)")
        print(f"      Deduplicated Candidates: {result.dedup_candidate_counts[0]}")
        print(f"      Top-10 Tier Composition: HOT: {result.hot_contribution_pct:.1f}% | COLD: {result.cold_contribution_pct:.1f}%")

        # I/O Telemetry
        io_stats = result.io_stats
        dram_reads = io_stats.get("dram_reads", 0)
        ssd_reads = io_stats.get("ssd_reads", 0)
        print(f"      Application Vector Reads: {dram_reads} from DRAM cache | {ssd_reads} from unbuffered SSD file")

        print("\n" + "=" * 78)
        print(" [PASS] DEMO EXECUTION COMPLETE — ALL VALIDATION GATES PASSED")
        print("=" * 78)
        return 0

    except Exception as exc:
        print(f"\n[FATAL ERROR] Hybrid search execution failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Resource cleanup
        cleaned = []
        if vector_store is not None:
            try:
                vector_store.close()
                cleaned.append("TieredVectorStore")
            except Exception as e:
                print(f"[WARNING] Failed to close TieredVectorStore: {e}", file=sys.stderr)

        if cold_index is not None:
            try:
                cold_index.close()
                cleaned.append("ColdDiskANNIndex")
            except Exception as e:
                print(f"[WARNING] Failed to close ColdDiskANNIndex: {e}", file=sys.stderr)

        if cleaned:
            print(f"\n[Cleanup] Released handles: {', '.join(cleaned)}.")
        else:
            print("\n[Cleanup] No persistent handles required release.")


if __name__ == "__main__":
    sys.exit(run_demo())
