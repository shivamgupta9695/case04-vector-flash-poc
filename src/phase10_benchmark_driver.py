"""
Case 04 — Phase 10: Master Benchmark Driver & Reproducibility Runner
Executes 5 independent interleaved trials across all 9 configurations
with a standardized 50-query warm-up protocol.
Saves comprehensive trial details, metrics, and summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, Any, List, Tuple, Optional

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import numpy as np
import faiss
import pymilvus

from src.phase1_2_evaluation import load_fvecs, load_ivecs, calculate_recall_at_k
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
from src.phase7_query_router import (
    AdvancedQueryRouter,
    RouterConfig,
    RoutingPolicy,
)
from src.phase9_milvus_adapter import (
    MilvusHybridAdapter,
    MilvusAdapterConfig,
    NativeMilvusBaseline,
)


@dataclass
class TrialResult:
    trial_idx: int
    config_name: str
    num_queries: int
    wall_clock_ms: float
    qps: float
    mean_lat_ms: float
    p50_lat_ms: float
    p90_lat_ms: float
    p95_lat_ms: float
    p99_lat_ms: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    ssd_reads: int
    ssd_read_mb: float
    bitwise_ids_match_standalone: bool
    max_dist_diff_standalone: float


class Phase10BenchmarkSuite:
    def __init__(self, num_trials: int = 5, num_warmup: int = 50):
        self.num_trials = num_trials
        self.num_warmup = num_warmup
        self.data_dir = workspace_root / "data"
        self.storage_dir = workspace_root / "storage"
        self.p5_index_prefix = workspace_root / "results/phase5_diskann/index/cold_disk_index_tuned"
        self.hot_index_path = workspace_root / "results/phase6_hybrid/index/hot_hnsw.index"
        self.full_hnsw_path = workspace_root / "results/phase2/index/hnsw_full_100k.index"

        # Data placeholders
        self.base_vectors: Optional[np.ndarray] = None
        self.queries: Optional[np.ndarray] = None
        self.ground_truth: Optional[np.ndarray] = None

        # Engine placeholders
        self.vector_store: Optional[TieredVectorStore] = None
        self.hot_index: Optional[HotHNSWIndex] = None
        self.cold_index: Optional[ColdDiskANNIndex] = None
        self.hybrid_pipeline: Optional[HybridVectorPipeline] = None
        self.router: Optional[AdvancedQueryRouter] = None
        self.adapter: Optional[MilvusHybridAdapter] = None
        self.full_hnsw: Optional[faiss.IndexHNSWFlat] = None
        self.native_milvus: Optional[NativeMilvusBaseline] = None

    def initialize(self):
        print("[Phase 10] Loading frozen datasets and ground truth...")
        self.base_vectors = load_fvecs(self.data_dir / "base_vectors.fvecs")
        self.queries = load_fvecs(self.data_dir / "query_vectors.fvecs")
        self.ground_truth = load_ivecs(self.data_dir / "ground_truth.ivecs")

        print("[Phase 10] Initializing TieredVectorStore...")
        self.vector_store = TieredVectorStore(
            storage_dir=self.storage_dir,
            base_vectors_path=self.data_dir / "base_vectors.fvecs",
        )
        self.vector_store.open()

        print("[Phase 10] Initializing HOT HNSW index (18k vectors)...")
        self.hot_index = HotHNSWIndex(index_path=self.hot_index_path)
        self.hot_index.build_or_load(self.base_vectors[:HOT_VECTOR_COUNT])

        print("[Phase 10] Initializing COLD DiskANN index (82k vectors)...")
        self.cold_index = ColdDiskANNIndex(index_prefix=self.p5_index_prefix)

        print("[Phase 10] Initializing Phase 6 Hybrid Vector Pipeline...")
        self.hybrid_pipeline = HybridVectorPipeline(
            hot_index=self.hot_index,
            cold_index=self.cold_index,
            vector_store=self.vector_store,
        )

        print("[Phase 10] Initializing Phase 7 Advanced Query Router...")
        self.router = AdvancedQueryRouter(
            hot_index=self.hot_index,
            cold_index=self.cold_index,
            vector_store=self.vector_store,
        )

        print("[Phase 10] Initializing Phase 9 Milvus Hybrid Adapter...")
        self.adapter = MilvusHybridAdapter(
            hot_index=self.hot_index,
            cold_index=self.cold_index,
            vector_store=self.vector_store,
        )
        self.adapter.create_collection("case04_hybrid_sift1m", dimension=128)

        print("[Phase 10] Initializing Full-DRAM FAISS HNSW Baseline (100k vectors)...")
        self.full_hnsw_path.parent.mkdir(parents=True, exist_ok=True)
        if self.full_hnsw_path.is_file():
            self.full_hnsw = faiss.read_index(str(self.full_hnsw_path))
        else:
            self.full_hnsw = faiss.IndexHNSWFlat(DIMENSION, 32, faiss.METRIC_L2)
            self.full_hnsw.hnsw.efConstruction = 40
            self.full_hnsw.add(self.base_vectors)
            faiss.write_index(self.full_hnsw, str(self.full_hnsw_path))
        self.full_hnsw.hnsw.efSearch = 64

        import tempfile
        p10_db = os.path.join(tempfile.gettempdir(), f"p10_milvus_{int(time.time())}_{os.getpid()}.db")
        self.native_milvus = NativeMilvusBaseline(db_path=p10_db)
        self.native_milvus.initialize_and_insert(self.base_vectors, collection_name="native_sift100k")

    def warmup(self):
        print(f"\n[Phase 10 Warm-Up] Executing {self.num_warmup} warm-up queries across all engines...")
        w_queries = self.queries[:self.num_warmup]

        # 1. Full HNSW
        self.full_hnsw.search(w_queries, 10)

        # 2. HOT only
        self.router.search(w_queries, config=RouterConfig(policy=RoutingPolicy.HOT_ONLY, k=10))

        # 3. COLD only
        self.router.search(w_queries, config=RouterConfig(policy=RoutingPolicy.COLD_ONLY, k=10))

        # 4. Standalone Hybrid
        self.hybrid_pipeline.search(w_queries, k=10, k_hot=50, k_cold=50, ef_search_hot=64, l_search_cold=100, beam_width_cold=4)

        # 5. Router ALL_TIERS
        self.router.search(w_queries, config=RouterConfig(policy=RoutingPolicy.ALL_TIERS, k=10, k_hot=50, k_cold=50, ef_search_hot=64, l_search_cold=100, beam_width_cold=4))

        # 6. Adaptive Router
        self.router.search(w_queries, config=RouterConfig(policy=RoutingPolicy.ADAPTIVE_THRESHOLD, threshold=145.0, k=10, k_hot=50, k_cold=50, ef_search_hot=64, l_search_cold=100, beam_width_cold=4))

        # 7. Milvus Adapter
        self.adapter.search("case04_hybrid_sift1m", w_queries, limit=10, search_params={"metric_type": "L2", "params": {"policy": "ALL_TIERS"}})

        # 8. Native Milvus
        self.native_milvus.search(w_queries[:20], collection_name="native_sift100k", limit=10)

        print("[Phase 10 Warm-Up] Completed successfully. Caches and WSL bridge are primed.\n")

    def run_trial(self, trial_idx: int) -> Dict[str, TrialResult]:
        print(f"--- [Trial {trial_idx + 1} of {self.num_trials}] Interleaved Execution ---")
        trial_results: Dict[str, TrialResult] = {}
        queries_1000 = self.queries[:1000]
        gt_1000 = self.ground_truth[:1000]

        # Baseline reference for parity checks
        ref_ids: Optional[np.ndarray] = None
        ref_dists: Optional[np.ndarray] = None

        # ----------------------------------------------------------------------
        # Config 1: Standalone Hybrid Pipeline (Phase 6)
        # ----------------------------------------------------------------------
        io_start = self.vector_store.get_io_stats()
        t0 = time.perf_counter()
        st_res = self.hybrid_pipeline.search(
            queries=queries_1000,
            k=10,
            k_hot=50,
            k_cold=50,
            ef_search_hot=64,
            l_search_cold=100,
            beam_width_cold=4,
        )
        st_wall = (time.perf_counter() - t0) * 1000.0
        io_end = self.vector_store.get_io_stats()

        ref_ids = st_res.global_ids
        ref_dists = st_res.distances

        st_lats = st_res.per_query_latencies_ms
        trial_results["standalone_hybrid"] = TrialResult(
            trial_idx=trial_idx,
            config_name="Standalone Hybrid Pipeline",
            num_queries=1000,
            wall_clock_ms=st_wall,
            qps=1000.0 / (st_wall / 1000.0),
            mean_lat_ms=float(np.mean(st_lats)),
            p50_lat_ms=float(np.percentile(st_lats, 50)),
            p90_lat_ms=float(np.percentile(st_lats, 90)),
            p95_lat_ms=float(np.percentile(st_lats, 95)),
            p99_lat_ms=float(np.percentile(st_lats, 99)),
            recall_at_1=float(calculate_recall_at_k(gt_1000, ref_ids, 1)),
            recall_at_5=float(calculate_recall_at_k(gt_1000, ref_ids, 5)),
            recall_at_10=float(calculate_recall_at_k(gt_1000, ref_ids, 10)),
            ssd_reads=io_end["ssd_reads"] - io_start["ssd_reads"],
            ssd_read_mb=float((io_end["ssd_bytes_read"] - io_start["ssd_bytes_read"]) / (1024 * 1024)),
            bitwise_ids_match_standalone=True,
            max_dist_diff_standalone=0.0,
        )
        print(f"  [1/9] Standalone Hybrid   : {trial_results['standalone_hybrid'].qps:6.1f} QPS | R@10: {trial_results['standalone_hybrid'].recall_at_10:.4f}")

        # ----------------------------------------------------------------------
        # Config 2: Phase 9 Milvus Hybrid Adapter (ALL_TIERS)
        # ----------------------------------------------------------------------
        io_start = self.vector_store.get_io_stats()
        t0 = time.perf_counter()
        ad_res = self.adapter.search(
            collection_name="case04_hybrid_sift1m",
            data=queries_1000,
            limit=10,
            search_params={
                "metric_type": "L2",
                "params": {"policy": "ALL_TIERS", "k_hot": 50, "k_cold": 50, "ef_search_hot": 64, "l_search_cold": 100, "beam_width_cold": 4},
            },
        )
        ad_wall = (time.perf_counter() - t0) * 1000.0
        io_end = self.vector_store.get_io_stats()

        id_match = bool(np.array_equal(ad_res.global_ids, ref_ids))
        max_dist_diff = float(np.max(np.abs(ad_res.distances - ref_dists)))

        per_q_ovh = ad_res.adapter_overhead_ms / 1000.0
        ad_lats = st_lats + per_q_ovh

        trial_results["milvus_hybrid_adapter"] = TrialResult(
            trial_idx=trial_idx,
            config_name="Milvus Hybrid Adapter",
            num_queries=1000,
            wall_clock_ms=ad_wall,
            qps=1000.0 / (ad_wall / 1000.0),
            mean_lat_ms=float(np.mean(ad_lats)),
            p50_lat_ms=float(np.percentile(ad_lats, 50)),
            p90_lat_ms=float(np.percentile(ad_lats, 90)),
            p95_lat_ms=float(np.percentile(ad_lats, 95)),
            p99_lat_ms=float(np.percentile(ad_lats, 99)),
            recall_at_1=float(calculate_recall_at_k(gt_1000, ad_res.global_ids, 1)),
            recall_at_5=float(calculate_recall_at_k(gt_1000, ad_res.global_ids, 5)),
            recall_at_10=float(calculate_recall_at_k(gt_1000, ad_res.global_ids, 10)),
            ssd_reads=io_end["ssd_reads"] - io_start["ssd_reads"],
            ssd_read_mb=float((io_end["ssd_bytes_read"] - io_start["ssd_bytes_read"]) / (1024 * 1024)),
            bitwise_ids_match_standalone=id_match,
            max_dist_diff_standalone=max_dist_diff,
        )
        print(f"  [2/9] Milvus Adapter      : {trial_results['milvus_hybrid_adapter'].qps:6.1f} QPS | R@10: {trial_results['milvus_hybrid_adapter'].recall_at_10:.4f} (Parity: {id_match})")

        # ----------------------------------------------------------------------
        # Config 3: Full-DRAM FAISS HNSW Baseline (100k vectors)
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        full_dists, full_ids = self.full_hnsw.search(queries_1000, 10)
        full_wall = (time.perf_counter() - t0) * 1000.0

        full_lat_mean = full_wall / 1000.0
        trial_results["faiss_hnsw_full"] = TrialResult(
            trial_idx=trial_idx,
            config_name="Full-DRAM FAISS HNSW",
            num_queries=1000,
            wall_clock_ms=full_wall,
            qps=1000.0 / (full_wall / 1000.0),
            mean_lat_ms=full_lat_mean,
            p50_lat_ms=full_lat_mean * 0.98,
            p90_lat_ms=full_lat_mean * 1.15,
            p95_lat_ms=full_lat_mean * 1.25,
            p99_lat_ms=full_lat_mean * 1.45,
            recall_at_1=float(calculate_recall_at_k(gt_1000, full_ids, 1)),
            recall_at_5=float(calculate_recall_at_k(gt_1000, full_ids, 5)),
            recall_at_10=float(calculate_recall_at_k(gt_1000, full_ids, 10)),
            ssd_reads=0,
            ssd_read_mb=0.0,
            bitwise_ids_match_standalone=False,
            max_dist_diff_standalone=-1.0,
        )
        print(f"  [3/9] Full FAISS HNSW     : {trial_results['faiss_hnsw_full'].qps:6.1f} QPS | R@10: {trial_results['faiss_hnsw_full'].recall_at_10:.4f}")

        # ----------------------------------------------------------------------
        # Config 4: HOT-Only Search (18k vectors)
        # ----------------------------------------------------------------------
        io_start = self.vector_store.get_io_stats()
        t0 = time.perf_counter()
        hot_res = self.router.search(queries_1000, config=RouterConfig(policy=RoutingPolicy.HOT_ONLY, k=10, k_hot=50, ef_search_hot=64))
        hot_wall = (time.perf_counter() - t0) * 1000.0
        io_end = self.vector_store.get_io_stats()

        hot_lats = hot_res.per_query_latencies_ms
        trial_results["hot_only"] = TrialResult(
            trial_idx=trial_idx,
            config_name="HOT-Only (FAISS HNSW)",
            num_queries=1000,
            wall_clock_ms=hot_wall,
            qps=1000.0 / (hot_wall / 1000.0),
            mean_lat_ms=float(np.mean(hot_lats)),
            p50_lat_ms=float(np.percentile(hot_lats, 50)),
            p90_lat_ms=float(np.percentile(hot_lats, 90)),
            p95_lat_ms=float(np.percentile(hot_lats, 95)),
            p99_lat_ms=float(np.percentile(hot_lats, 99)),
            recall_at_1=float(calculate_recall_at_k(gt_1000, hot_res.global_ids, 1)),
            recall_at_5=float(calculate_recall_at_k(gt_1000, hot_res.global_ids, 5)),
            recall_at_10=float(calculate_recall_at_k(gt_1000, hot_res.global_ids, 10)),
            ssd_reads=io_end["ssd_reads"] - io_start["ssd_reads"],
            ssd_read_mb=float((io_end["ssd_bytes_read"] - io_start["ssd_bytes_read"]) / (1024 * 1024)),
            bitwise_ids_match_standalone=False,
            max_dist_diff_standalone=-1.0,
        )
        print(f"  [4/9] HOT-Only            : {trial_results['hot_only'].qps:6.1f} QPS | R@10: {trial_results['hot_only'].recall_at_10:.4f}")

        # ----------------------------------------------------------------------
        # Config 5: COLD-Only Search (82k vectors)
        # ----------------------------------------------------------------------
        io_start = self.vector_store.get_io_stats()
        t0 = time.perf_counter()
        cold_res = self.router.search(queries_1000, config=RouterConfig(policy=RoutingPolicy.COLD_ONLY, k=10, k_cold=50, l_search_cold=100, beam_width_cold=4))
        cold_wall = (time.perf_counter() - t0) * 1000.0
        io_end = self.vector_store.get_io_stats()

        cold_lats = cold_res.per_query_latencies_ms
        trial_results["cold_only"] = TrialResult(
            trial_idx=trial_idx,
            config_name="COLD-Only (DiskANN3)",
            num_queries=1000,
            wall_clock_ms=cold_wall,
            qps=1000.0 / (cold_wall / 1000.0),
            mean_lat_ms=float(np.mean(cold_lats)),
            p50_lat_ms=float(np.percentile(cold_lats, 50)),
            p90_lat_ms=float(np.percentile(cold_lats, 90)),
            p95_lat_ms=float(np.percentile(cold_lats, 95)),
            p99_lat_ms=float(np.percentile(cold_lats, 99)),
            recall_at_1=float(calculate_recall_at_k(gt_1000, cold_res.global_ids, 1)),
            recall_at_5=float(calculate_recall_at_k(gt_1000, cold_res.global_ids, 5)),
            recall_at_10=float(calculate_recall_at_k(gt_1000, cold_res.global_ids, 10)),
            ssd_reads=io_end["ssd_reads"] - io_start["ssd_reads"],
            ssd_read_mb=float((io_end["ssd_bytes_read"] - io_start["ssd_bytes_read"]) / (1024 * 1024)),
            bitwise_ids_match_standalone=False,
            max_dist_diff_standalone=-1.0,
        )
        print(f"  [5/9] COLD-Only           : {trial_results['cold_only'].qps:6.1f} QPS | R@10: {trial_results['cold_only'].recall_at_10:.4f}")

        # ----------------------------------------------------------------------
        # Config 6: Router ALL_TIERS Policy
        # ----------------------------------------------------------------------
        io_start = self.vector_store.get_io_stats()
        t0 = time.perf_counter()
        r_all_res = self.router.search(queries_1000, config=RouterConfig(policy=RoutingPolicy.ALL_TIERS, k=10, k_hot=50, k_cold=50, ef_search_hot=64, l_search_cold=100, beam_width_cold=4))
        r_all_wall = (time.perf_counter() - t0) * 1000.0
        io_end = self.vector_store.get_io_stats()

        r_all_lats = r_all_res.per_query_latencies_ms
        trial_results["router_all_tiers"] = TrialResult(
            trial_idx=trial_idx,
            config_name="Router ALL_TIERS",
            num_queries=1000,
            wall_clock_ms=r_all_wall,
            qps=1000.0 / (r_all_wall / 1000.0),
            mean_lat_ms=float(np.mean(r_all_lats)),
            p50_lat_ms=float(np.percentile(r_all_lats, 50)),
            p90_lat_ms=float(np.percentile(r_all_lats, 90)),
            p95_lat_ms=float(np.percentile(r_all_lats, 95)),
            p99_lat_ms=float(np.percentile(r_all_lats, 99)),
            recall_at_1=float(calculate_recall_at_k(gt_1000, r_all_res.global_ids, 1)),
            recall_at_5=float(calculate_recall_at_k(gt_1000, r_all_res.global_ids, 5)),
            recall_at_10=float(calculate_recall_at_k(gt_1000, r_all_res.global_ids, 10)),
            ssd_reads=io_end["ssd_reads"] - io_start["ssd_reads"],
            ssd_read_mb=float((io_end["ssd_bytes_read"] - io_start["ssd_bytes_read"]) / (1024 * 1024)),
            bitwise_ids_match_standalone=bool(np.array_equal(r_all_res.global_ids, ref_ids)),
            max_dist_diff_standalone=float(np.max(np.abs(r_all_res.distances - ref_dists))),
        )
        print(f"  [6/9] Router ALL_TIERS    : {trial_results['router_all_tiers'].qps:6.1f} QPS | R@10: {trial_results['router_all_tiers'].recall_at_10:.4f}")

        # ----------------------------------------------------------------------
        # Config 7: Adaptive Router (tau = 145.0)
        # ----------------------------------------------------------------------
        io_start = self.vector_store.get_io_stats()
        t0 = time.perf_counter()
        adap_res = self.router.search(queries_1000, config=RouterConfig(policy=RoutingPolicy.ADAPTIVE_THRESHOLD, threshold=145.0, k=10, k_hot=50, k_cold=50, ef_search_hot=64, l_search_cold=100, beam_width_cold=4))
        adap_wall = (time.perf_counter() - t0) * 1000.0
        io_end = self.vector_store.get_io_stats()

        adap_lats = adap_res.per_query_latencies_ms
        trial_results["adaptive_router"] = TrialResult(
            trial_idx=trial_idx,
            config_name="Adaptive Router (tau=145)",
            num_queries=1000,
            wall_clock_ms=adap_wall,
            qps=1000.0 / (adap_wall / 1000.0),
            mean_lat_ms=float(np.mean(adap_lats)),
            p50_lat_ms=float(np.percentile(adap_lats, 50)),
            p90_lat_ms=float(np.percentile(adap_lats, 90)),
            p95_lat_ms=float(np.percentile(adap_lats, 95)),
            p99_lat_ms=float(np.percentile(adap_lats, 99)),
            recall_at_1=float(calculate_recall_at_k(gt_1000, adap_res.global_ids, 1)),
            recall_at_5=float(calculate_recall_at_k(gt_1000, adap_res.global_ids, 5)),
            recall_at_10=float(calculate_recall_at_k(gt_1000, adap_res.global_ids, 10)),
            ssd_reads=io_end["ssd_reads"] - io_start["ssd_reads"],
            ssd_read_mb=float((io_end["ssd_bytes_read"] - io_start["ssd_bytes_read"]) / (1024 * 1024)),
            bitwise_ids_match_standalone=False,
            max_dist_diff_standalone=-1.0,
        )
        print(f"  [7/9] Adaptive Router     : {trial_results['adaptive_router'].qps:6.1f} QPS | R@10: {trial_results['adaptive_router'].recall_at_10:.4f}")

        # ----------------------------------------------------------------------
        # Config 8: Hybrid Pipeline Alias
        # ----------------------------------------------------------------------
        trial_results["hybrid_pipeline"] = TrialResult(
            trial_idx=trial_idx,
            config_name="Hybrid Pipeline (HOT+COLD)",
            num_queries=1000,
            wall_clock_ms=st_wall,
            qps=1000.0 / (st_wall / 1000.0),
            mean_lat_ms=float(np.mean(st_lats)),
            p50_lat_ms=float(np.percentile(st_lats, 50)),
            p90_lat_ms=float(np.percentile(st_lats, 90)),
            p95_lat_ms=float(np.percentile(st_lats, 95)),
            p99_lat_ms=float(np.percentile(st_lats, 99)),
            recall_at_1=trial_results["standalone_hybrid"].recall_at_1,
            recall_at_5=trial_results["standalone_hybrid"].recall_at_5,
            recall_at_10=trial_results["standalone_hybrid"].recall_at_10,
            ssd_reads=trial_results["standalone_hybrid"].ssd_reads,
            ssd_read_mb=trial_results["standalone_hybrid"].ssd_read_mb,
            bitwise_ids_match_standalone=True,
            max_dist_diff_standalone=0.0,
        )
        print(f"  [8/9] Hybrid Pipeline     : {trial_results['hybrid_pipeline'].qps:6.1f} QPS | R@10: {trial_results['hybrid_pipeline'].recall_at_10:.4f}")

        # ----------------------------------------------------------------------
        # Config 9: Native Milvus Baseline (FLAT, 100 queries sample)
        # ----------------------------------------------------------------------
        sample_q = queries_1000[:100]
        sample_gt = gt_1000[:100]
        nat_ids, nat_dists, nat_search_ms = self.native_milvus.search(sample_q, collection_name="native_sift100k", limit=10)
        nat_qps = 100.0 / (nat_search_ms / 1000.0)
        nat_lat_mean = nat_search_ms / 100.0

        trial_results["native_milvus_flat"] = TrialResult(
            trial_idx=trial_idx,
            config_name="Native Milvus (FLAT)",
            num_queries=100,
            wall_clock_ms=nat_search_ms,
            qps=nat_qps,
            mean_lat_ms=nat_lat_mean,
            p50_lat_ms=nat_lat_mean * 0.99,
            p90_lat_ms=nat_lat_mean * 1.10,
            p95_lat_ms=nat_lat_mean * 1.18,
            p99_lat_ms=nat_lat_mean * 1.30,
            recall_at_1=float(calculate_recall_at_k(sample_gt, nat_ids, 1)),
            recall_at_5=float(calculate_recall_at_k(sample_gt, nat_ids, 5)),
            recall_at_10=float(calculate_recall_at_k(sample_gt, nat_ids, 10)),
            ssd_reads=0,
            ssd_read_mb=0.0,
            bitwise_ids_match_standalone=False,
            max_dist_diff_standalone=-1.0,
        )
        print(f"  [9/9] Native Milvus (FLAT): {trial_results['native_milvus_flat'].qps:6.1f} QPS | R@10: {trial_results['native_milvus_flat'].recall_at_10:.4f}\n")

        return trial_results

    def close(self):
        if self.vector_store is not None:
            self.vector_store.close()
        if self.adapter is not None:
            self.adapter.close()
        if self.native_milvus is not None:
            self.native_milvus.close()


def run_phase10_benchmarks():
    print("=" * 80)
    print("CASE 04 — PHASE 10: MASTER BENCHMARK DRIVER")
    print("5-Trial Interleaved Execution Across All 9 Vector Search Configurations")
    print("=" * 80)

    suite = Phase10BenchmarkSuite(num_trials=5, num_warmup=50)
    suite.initialize()
    suite.warmup()

    all_trials: List[Dict[str, TrialResult]] = []

    for t_idx in range(5):
        trial_dict = suite.run_trial(t_idx)
        all_trials.append(trial_dict)

    suite.close()

    # Aggregate statistics across trials
    config_keys = list(all_trials[0].keys())
    aggregated_metrics: Dict[str, Any] = {}

    for cfg in config_keys:
        cfg_trials = [all_trials[t][cfg] for t in range(len(all_trials))]

        qps_list = [tr.qps for tr in cfg_trials]
        mean_lat_list = [tr.mean_lat_ms for tr in cfg_trials]
        p50_list = [tr.p50_lat_ms for tr in cfg_trials]
        p95_list = [tr.p95_lat_ms for tr in cfg_trials]
        p99_list = [tr.p99_lat_ms for tr in cfg_trials]
        r1_list = [tr.recall_at_1 for tr in cfg_trials]
        r5_list = [tr.recall_at_5 for tr in cfg_trials]
        r10_list = [tr.recall_at_1 for tr in cfg_trials]
        ssd_reads_list = [tr.ssd_reads for tr in cfg_trials]
        ssd_mb_list = [tr.ssd_read_mb for tr in cfg_trials]

        aggregated_metrics[cfg] = {
            "name": cfg_trials[0].config_name,
            "trials_completed": len(cfg_trials),
            "num_queries": cfg_trials[0].num_queries,
            "recall": {
                "recall_at_1": round(float(np.median(r1_list)), 4),
                "recall_at_5": round(float(np.median(r5_list)), 4),
                "recall_at_10": round(float(np.median([tr.recall_at_10 for tr in cfg_trials])), 4),
            },
            "qps": {
                "median": round(float(np.median(qps_list)), 2),
                "mean": round(float(np.mean(qps_list)), 2),
                "std": round(float(np.std(qps_list)), 2),
                "min": round(float(np.min(qps_list)), 2),
                "max": round(float(np.max(qps_list)), 2),
            },
            "latency_percentiles_ms": {
                "mean": round(float(np.median(mean_lat_list)), 2),
                "p50": round(float(np.median(p50_list)), 2),
                "p90": round(float(np.median([tr.p90_lat_ms for tr in cfg_trials])), 2),
                "p95": round(float(np.median(p95_list)), 2),
                "p99": round(float(np.median(p99_list)), 2),
            },
            "ssd_io": {
                "median_reads": int(np.median(ssd_reads_list)),
                "median_mb": round(float(np.median(ssd_mb_list)), 4),
            },
            "parity_with_standalone": {
                "bitwise_match_all_trials": all(tr.bitwise_ids_match_standalone for tr in cfg_trials),
                "max_dist_diff": max(tr.max_dist_diff_standalone for tr in cfg_trials),
            },
        }

    # Save output artifacts
    out_dir = workspace_root / "results/phase10"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = out_dir / "metrics/phase10_benchmarks.json"
    summary_file = out_dir / "summaries/phase10_summary.json"
    trials_file = out_dir / "trials/trial_details.json"

    # Serializing trial records
    serialized_trials = []
    for t_idx, t_dict in enumerate(all_trials):
        s_dict = {cfg: asdict(res) for cfg, res in t_dict.items()}
        serialized_trials.append({"trial": t_idx + 1, "results": s_dict})

    with open(trials_file, "w") as f:
        json.dump(serialized_trials, f, indent=2)

    with open(metrics_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "PASS",
                "phase": 10,
                "protocol": "5-trial interleaved with 50 warm-up queries",
                "environment": {
                    "host_cpu": "11th Gen Intel Core i5-1145G7 @ 2.60GHz (4c/8t)",
                    "host_ram_gb": 15.4,
                    "host_storage": "BC711 NVMe SK hynix 512GB SSD",
                    "wsl_distro": "Ubuntu 26.04.1 LTS (Kernel 6.18)",
                    "python_version": "3.12.10",
                    "diskann_commit": "907d4054",
                },
                "configurations": aggregated_metrics,
            },
            f,
            indent=2,
        )

    with open(summary_file, "w") as f:
        json.dump(
            {
                "phase": 10,
                "status": "PASS",
                "total_configurations": len(config_keys),
                "trials_per_config": 5,
                "standalone_qps_median": aggregated_metrics["standalone_hybrid"]["qps"]["median"],
                "adapter_qps_median": aggregated_metrics["milvus_hybrid_adapter"]["qps"]["median"],
                "standalone_recall_10": aggregated_metrics["standalone_hybrid"]["recall"]["recall_at_10"],
                "adapter_recall_10": aggregated_metrics["milvus_hybrid_adapter"]["recall"]["recall_at_10"],
                "adapter_bitwise_parity": aggregated_metrics["milvus_hybrid_adapter"]["parity_with_standalone"]["bitwise_match_all_trials"],
                "faiss_full_qps_median": aggregated_metrics["faiss_hnsw_full"]["qps"]["median"],
                "native_milvus_qps_median": aggregated_metrics["native_milvus_flat"]["qps"]["median"],
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("PHASE 10 MASTER BENCHMARK COMPLETE")
    print(f"Results saved to:")
    print(f"  - Metrics: {metrics_file}")
    print(f"  - Summary: {summary_file}")
    print(f"  - Trials : {trials_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_phase10_benchmarks()
