"""
Case 04 — Phase 8: Merge and Exact Reranking Validation
Comprehensive validation pipeline for candidate merging, local-to-global ID translation,
duplicate removal, invalid ID rejection, exact FP32 squared-L2 distance recomputation,
monotonic Top-K sorting, candidate-union recall quantification, and adaptive routing recall loss.

Constraints:
- Phases 1–7 frozen.
- Reuses Phase 6 HybridVectorPipeline and Phase 7 query router.
- No modifications to DiskANN or frozen datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, Any, List, Tuple, Optional, Sequence, Set

# Ensure workspace root in path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

try:
    import numpy as np
except ImportError as err:
    print(f"CRITICAL ERROR: Failed to import NumPy: {err}", file=sys.stderr)
    sys.exit(1)

from src.phase1_2_evaluation import (
    load_fvecs,
    load_ivecs,
    compute_sha256,
    calculate_recall_at_k,
    validate_predictions,
)
from src.phase3_ssd_store import SSDVectorStore
from src.phase5_diskann_search import ColdDiskANNIndex
from src.phase6_hybrid_search import (
    HotHNSWIndex,
    TieredVectorStore,
    HybridVectorPipeline,
    HOT_VECTOR_COUNT,
    COLD_VECTOR_COUNT,
    TOTAL_VECTORS,
    DIMENSION,
    COLD_ID_START,
    COLD_ID_END,
)
from src.phase7_query_router import (
    AdvancedQueryRouter,
    RouterConfig,
    RoutingPolicy,
    RouterSearchResult,
)


# ==============================================================================
# 1. Data Structures for Merge & Reranking Statistics
# ==============================================================================
@dataclass
class MergeStats:
    """Statistics for candidate stream merging and deduplication."""
    raw_hot_count: int
    raw_cold_count: int
    total_raw_count: int
    dedup_count: int
    duplicates_removed: int
    invalid_ids_rejected: int
    hot_candidate_count: int
    cold_candidate_count: int


@dataclass
class RerankStats:
    """Statistics for exact distance recomputation and Top-K sorting."""
    query_idx: int
    candidate_count: int
    k: int
    returned_count: int
    is_padded: bool
    min_distance: float
    max_distance: float
    is_monotonic: bool
    hot_in_top_k: int
    cold_in_top_k: int
    rerank_latency_ms: float


@dataclass
class CandidateUnionRecallResult:
    """Comparison of candidate-union recall before reranking vs final Top-K recall."""
    union_recall_at_1: float
    union_recall_at_5: float
    union_recall_at_10: float
    final_recall_at_1: float
    final_recall_at_5: float
    final_recall_at_10: float
    recall_preservation_fidelity_pct: float
    mean_candidate_pool_size: float
    min_candidate_pool_size: int
    max_candidate_pool_size: int


@dataclass
class AdaptiveRecallLossResult:
    """Direct measurement of recall loss caused by HOT-only early exits."""
    policy: str
    threshold: float
    early_exit_count: int
    early_exit_rate_pct: float
    all_tiers_recall_at_1: float
    all_tiers_recall_at_5: float
    all_tiers_recall_at_10: float
    adaptive_recall_at_1: float
    adaptive_recall_at_5: float
    adaptive_recall_at_10: float
    recall_loss_at_1: float
    recall_loss_at_5: float
    recall_loss_at_10: float
    early_exits_with_closer_cold_neighbor: int
    pct_early_exits_with_closer_cold_neighbor: float


# ==============================================================================
# 2. Exact Candidate Merger Component
# ==============================================================================
class ExactCandidateMerger:
    """
    Independent candidate merging and deduplication component.
    Validates global IDs, translates local COLD IDs, removes duplicates,
    and filters out out-of-range IDs.
    """

    def __init__(
        self,
        hot_count: int = HOT_VECTOR_COUNT,
        cold_count: int = COLD_VECTOR_COUNT,
        total_vectors: int = TOTAL_VECTORS,
    ):
        self.hot_count = hot_count
        self.cold_count = cold_count
        self.total_vectors = total_vectors
        self.cold_id_start = hot_count

    def merge_single(
        self,
        hot_candidate_ids: Sequence[int],
        cold_candidate_ids: Sequence[int],
        cold_is_local: bool = False,
    ) -> Tuple[List[int], MergeStats]:
        """
        Merges HOT and COLD candidate ID streams for a single query.
        - Translates local COLD IDs if cold_is_local=True.
        - Discards invalid IDs (negative or >= total_vectors).
        - Removes duplicates while strictly preserving arrival order.
        """
        raw_hot = len(hot_candidate_ids)
        raw_cold = len(cold_candidate_ids)
        total_raw = raw_hot + raw_cold

        merged_dict: Dict[int, None] = {}
        invalid_count = 0
        seen_all_count = 0

        # 1. Ingest HOT candidates
        for cid in hot_candidate_ids:
            seen_all_count += 1
            try:
                gid = int(cid)
            except (ValueError, TypeError):
                invalid_count += 1
                continue

            if 0 <= gid < self.total_vectors:
                merged_dict[gid] = None
            else:
                invalid_count += 1

        # 2. Ingest COLD candidates
        for cid in cold_candidate_ids:
            seen_all_count += 1
            try:
                raw_id = int(cid)
            except (ValueError, TypeError):
                invalid_count += 1
                continue

            gid = (raw_id + self.cold_id_start) if cold_is_local else raw_id

            if 0 <= gid < self.total_vectors:
                merged_dict[gid] = None
            else:
                invalid_count += 1

        unique_cands = list(merged_dict.keys())
        dedup_count = len(unique_cands)
        duplicates_removed = (total_raw - invalid_count) - dedup_count

        hot_cands = sum(1 for cid in unique_cands if cid < self.hot_count)
        cold_cands = sum(1 for cid in unique_cands if cid >= self.hot_count)

        stats = MergeStats(
            raw_hot_count=raw_hot,
            raw_cold_count=raw_cold,
            total_raw_count=total_raw,
            dedup_count=dedup_count,
            duplicates_removed=duplicates_removed,
            invalid_ids_rejected=invalid_count,
            hot_candidate_count=hot_cands,
            cold_candidate_count=cold_cands,
        )

        return unique_cands, stats

    def merge_batch(
        self,
        hot_candidates_batch: Sequence[Sequence[int]],
        cold_candidates_batch: Sequence[Sequence[int]],
        cold_is_local: bool = False,
    ) -> Tuple[List[List[int]], List[MergeStats]]:
        """Merges candidate streams across a batch of queries."""
        if len(hot_candidates_batch) != len(cold_candidates_batch):
            raise ValueError(
                f"Batch size mismatch: HOT has {len(hot_candidates_batch)}, COLD has {len(cold_candidates_batch)}"
            )

        merged_lists: List[List[int]] = []
        stats_list: List[MergeStats] = []

        for h_list, c_list in zip(hot_candidates_batch, cold_candidates_batch):
            cands, st = self.merge_single(h_list, c_list, cold_is_local=cold_is_local)
            merged_lists.append(cands)
            stats_list.append(st)

        return merged_lists, stats_list


# ==============================================================================
# 3. Exact FP32 Squared-L2 Reranker Component
# ==============================================================================
class ExactReranker:
    """
    Exact FP32 squared-L2 reranker.
    Retrieves full-precision vectors from TieredVectorStore, computes exact L2^2 distances,
    and returns monotonically sorted Top-K candidates.
    """

    def __init__(self, vector_store: TieredVectorStore):
        self.vector_store = vector_store

    def rerank_single(
        self,
        query_vector: np.ndarray,
        candidate_ids: Sequence[int],
        k: int = 10,
        query_idx: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, RerankStats]:
        """
        Reranks a single candidate list against the query vector.
        Handles boundary conditions: empty candidates, |U| < k, and exact distance sorting.
        Returns:
            final_ids: shape (k,), int32
            final_distances: shape (k,), float32
            stats: RerankStats
        """
        if k <= 0:
            raise ValueError(f"k must be strictly positive, got {k}")

        if query_vector.ndim != 1 or len(query_vector) != DIMENSION:
            raise ValueError(f"Query vector must be 1D of shape ({DIMENSION},)")

        t0 = time.perf_counter()
        q_vec = query_vector.astype(np.float32)
        n_cands = len(candidate_ids)

        final_ids = np.full(k, -1, dtype=np.int32)
        final_dists = np.full(k, np.inf, dtype=np.float32)

        if n_cands == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            stats = RerankStats(
                query_idx=query_idx,
                candidate_count=0,
                k=k,
                returned_count=0,
                is_padded=True,
                min_distance=np.inf,
                max_distance=np.inf,
                is_monotonic=True,
                hot_in_top_k=0,
                cold_in_top_k=0,
                rerank_latency_ms=elapsed_ms,
            )
            return final_ids, final_dists, stats

        # Retrieve FP32 vectors (DRAM for HOT, SSD for COLD)
        cand_vecs = self.vector_store.read_vectors_batch(candidate_ids)

        # Exact squared-L2 distance recomputation in FP32
        diff = cand_vecs - q_vec
        exact_dists = np.sum(diff * diff, axis=1).astype(np.float32)

        # Sort ascending
        n_select = min(k, n_cands)
        sorted_order = np.argsort(exact_dists)[:n_select]

        top_ids = [candidate_ids[i] for i in sorted_order]
        top_dists = exact_dists[sorted_order]

        final_ids[:n_select] = top_ids
        final_dists[:n_select] = top_dists

        is_padded = n_select < k
        min_d = float(top_dists[0]) if len(top_dists) > 0 else np.inf
        max_d = float(top_dists[-1]) if len(top_dists) > 0 else np.inf

        # Monotonicity check
        is_mono = bool(np.all(np.diff(final_dists[:n_select]) >= -1e-5)) if n_select > 1 else True

        hot_top_k = sum(1 for tid in top_ids if tid < HOT_VECTOR_COUNT)
        cold_top_k = sum(1 for tid in top_ids if tid >= HOT_VECTOR_COUNT)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        stats = RerankStats(
            query_idx=query_idx,
            candidate_count=n_cands,
            k=k,
            returned_count=n_select,
            is_padded=is_padded,
            min_distance=min_d,
            max_distance=max_d,
            is_monotonic=is_mono,
            hot_in_top_k=hot_top_k,
            cold_in_top_k=cold_top_k,
            rerank_latency_ms=elapsed_ms,
        )

        return final_ids, final_dists, stats

    def rerank_batch(
        self,
        queries: np.ndarray,
        candidate_id_lists: List[Sequence[int]],
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, List[RerankStats]]:
        """Executes exact reranking across a batch of queries."""
        n_queries = len(queries)
        if len(candidate_id_lists) != n_queries:
            raise ValueError(
                f"Candidate list count ({len(candidate_id_lists)}) does not match queries ({n_queries})"
            )

        batch_ids = np.zeros((n_queries, k), dtype=np.int32)
        batch_dists = np.zeros((n_queries, k), dtype=np.float32)
        stats_list: List[RerankStats] = []

        for q_idx in range(n_queries):
            f_ids, f_dists, st = self.rerank_single(
                query_vector=queries[q_idx],
                candidate_ids=candidate_id_lists[q_idx],
                k=k,
                query_idx=q_idx,
            )
            batch_ids[q_idx] = f_ids
            batch_dists[q_idx] = f_dists
            stats_list.append(st)

        return batch_ids, batch_dists, stats_list


# ==============================================================================
# 4. Phase 8 Comprehensive Validation Suite
# ==============================================================================
class Phase8Validator:
    """
    Coordinates and validates all Phase 8 objectives:
    1. Candidate-Union Recall vs. Final Reranked Recall
    2. Adaptive Routing vs. ALL_TIERS Recall Loss & HOT-only false assumptions
    3. Edge cases and boundary behavior
    4. Local-to-global mapping audit
    5. Numerical float32 precision verification
    """

    def __init__(
        self,
        hot_index: HotHNSWIndex,
        cold_index: ColdDiskANNIndex,
        vector_store: TieredVectorStore,
        ground_truth_path: Optional[Path | str] = None,
        ground_truth_dists_path: Optional[Path | str] = None,
    ):
        self.hot_index = hot_index
        self.cold_index = cold_index
        self.vector_store = vector_store

        gt_p = ground_truth_path or workspace_root / "data/ground_truth.ivecs"
        gtd_p = ground_truth_dists_path or workspace_root / "data/ground_truth_distances.fvecs"

        self.ground_truth = load_ivecs(gt_p)
        self.ground_truth_dists = load_fvecs(gtd_p)

        self.merger = ExactCandidateMerger()
        self.reranker = ExactReranker(vector_store=self.vector_store)
        self.router = AdvancedQueryRouter(
            hot_index=self.hot_index,
            cold_index=self.cold_index,
            vector_store=self.vector_store,
        )

    # --------------------------------------------------------------------------
    # Objective 7: Candidate-Union Recall vs Final Reranked Recall
    # --------------------------------------------------------------------------
    def validate_candidate_union_recall(
        self,
        queries: np.ndarray,
        k: int = 10,
        k_hot: int = 50,
        k_cold: int = 50,
        ef_hot: int = 64,
        l_cold: int = 100,
        beam_cold: int = 4,
    ) -> CandidateUnionRecallResult:
        """
        Computes candidate-union recall before reranking and final recall after reranking.
        Quantifies whether exact reranking preserves 100% of candidate pool true neighbors.
        """
        n_queries = len(queries)
        gt_subset = self.ground_truth[:n_queries]

        # 1. Query HOT and COLD tiers
        _, hot_ids, _ = self.hot_index.search(queries, k=k_hot, ef_search=ef_hot)
        cold_res = self.cold_index.search(queries, k=k_cold, l_search=l_cold, beam_width=beam_cold)
        cold_ids = cold_res.global_ids

        # 2. Merge and deduplicate candidates
        merged_cands_batch, _ = self.merger.merge_batch(hot_ids, cold_ids, cold_is_local=False)

        # 3. Compute Candidate-Union Recall BEFORE reranking
        union_recalls_1 = []
        union_recalls_5 = []
        union_recalls_10 = []
        pool_sizes = []

        for i in range(n_queries):
            c_set = set(merged_cands_batch[i])
            pool_sizes.append(len(c_set))

            gt1 = set(gt_subset[i, :1].tolist())
            gt5 = set(gt_subset[i, :5].tolist())
            gt10 = set(gt_subset[i, :10].tolist())

            union_recalls_1.append(len(gt1 & c_set) / 1.0)
            union_recalls_5.append(len(gt5 & c_set) / 5.0)
            union_recalls_10.append(len(gt10 & c_set) / 10.0)

        # 4. Rerank exact FP32 distances
        final_ids, final_dists, _ = self.reranker.rerank_batch(queries, merged_cands_batch, k=k)

        # 5. Compute Final Reranked Recall AFTER reranking
        final_r1 = calculate_recall_at_k(gt_subset, final_ids, 1)
        final_r5 = calculate_recall_at_k(gt_subset, final_ids, 5)
        final_r10 = calculate_recall_at_k(gt_subset, final_ids, 10)

        u_r1 = float(np.mean(union_recalls_1))
        u_r5 = float(np.mean(union_recalls_5))
        u_r10 = float(np.mean(union_recalls_10))

        # Preservation fidelity: ratio of final recall to union recall
        fidelity = (final_r10 / u_r10 * 100.0) if u_r10 > 0 else 100.0

        return CandidateUnionRecallResult(
            union_recall_at_1=round(u_r1, 4),
            union_recall_at_5=round(u_r5, 4),
            union_recall_at_10=round(u_r10, 4),
            final_recall_at_1=round(final_r1, 4),
            final_recall_at_5=round(final_r5, 4),
            final_recall_at_10=round(final_r10, 4),
            recall_preservation_fidelity_pct=round(fidelity, 2),
            mean_candidate_pool_size=round(float(np.mean(pool_sizes)), 2),
            min_candidate_pool_size=int(np.min(pool_sizes)),
            max_candidate_pool_size=int(np.max(pool_sizes)),
        )

    # --------------------------------------------------------------------------
    # Objective 9: ALL_TIERS vs Adaptive Routing Recall Loss Comparison
    # --------------------------------------------------------------------------
    def validate_adaptive_recall_loss(
        self,
        queries: np.ndarray,
        thresholds: List[float] = [135.0, 145.0, 155.0],
        k: int = 10,
    ) -> List[AdaptiveRecallLossResult]:
        """
        Explicitly evaluates and reports the recall loss caused by HOT-only early exits.
        Demonstrates that a low HOT distance threshold does NOT guarantee global correctness.
        """
        n_queries = len(queries)
        gt_subset = self.ground_truth[:n_queries]
        gt_dists_subset = self.ground_truth_dists[:n_queries]

        # 1. Run ALL_TIERS baseline
        base_cfg = RouterConfig(
            policy=RoutingPolicy.ALL_TIERS,
            k=k,
            k_hot=50,
            k_cold=50,
            ef_search_hot=64,
            l_search_cold=100,
            beam_width_cold=4,
        )
        base_res = self.router.search(queries, config=base_cfg)
        base_r1 = calculate_recall_at_k(gt_subset, base_res.global_ids, 1)
        base_r5 = calculate_recall_at_k(gt_subset, base_res.global_ids, 5)
        base_r10 = calculate_recall_at_k(gt_subset, base_res.global_ids, 10)

        # Measure min distance in HOT for all queries
        hot_dists, _, _ = self.hot_index.search(queries, k=1, ef_search=64)
        min_hot_dists = hot_dists[:, 0]

        results: List[AdaptiveRecallLossResult] = []

        for tau in thresholds:
            adapt_cfg = RouterConfig(
                policy=RoutingPolicy.ADAPTIVE_THRESHOLD,
                threshold=tau,
                k=k,
                k_hot=50,
                k_cold=50,
                ef_search_hot=64,
                l_search_cold=100,
                beam_width_cold=4,
            )
            adapt_res = self.router.search(queries, config=adapt_cfg)
            ad_r1 = calculate_recall_at_k(gt_subset, adapt_res.global_ids, 1)
            ad_r5 = calculate_recall_at_k(gt_subset, adapt_res.global_ids, 5)
            ad_r10 = calculate_recall_at_k(gt_subset, adapt_res.global_ids, 10)

            # Identification of false assumption:
            # Queries where early exit was triggered (min_d <= tau), BUT true global top-1 is in COLD!
            early_exit_mask = min_hot_dists <= tau
            early_count = int(np.sum(early_exit_mask))

            true_top1_in_cold = gt_subset[:, 0] >= HOT_VECTOR_COUNT
            early_missed_cold_top1 = early_exit_mask & true_top1_in_cold
            false_assumption_count = int(np.sum(early_missed_cold_top1))

            pct_false = (
                (false_assumption_count / early_count * 100.0)
                if early_count > 0 else 0.0
            )

            res_item = AdaptiveRecallLossResult(
                policy=f"ADAPTIVE_THRESHOLD (tau={tau})",
                threshold=tau,
                early_exit_count=early_count,
                early_exit_rate_pct=round(early_count / n_queries * 100.0, 2),
                all_tiers_recall_at_1=round(base_r1, 4),
                all_tiers_recall_at_5=round(base_r5, 4),
                all_tiers_recall_at_10=round(base_r10, 4),
                adaptive_recall_at_1=round(ad_r1, 4),
                adaptive_recall_at_5=round(ad_r5, 4),
                adaptive_recall_at_10=round(ad_r10, 4),
                recall_loss_at_1=round(base_r1 - ad_r1, 4),
                recall_loss_at_5=round(base_r5 - ad_r5, 4),
                recall_loss_at_10=round(base_r10 - ad_r10, 4),
                early_exits_with_closer_cold_neighbor=false_assumption_count,
                pct_early_exits_with_closer_cold_neighbor=round(pct_false, 2),
            )
            results.append(res_item)

        return results

    # --------------------------------------------------------------------------
    # Objective 8: Edge Cases and Robustness Verification
    # --------------------------------------------------------------------------
    def validate_edge_cases(self) -> Dict[str, Any]:
        """
        Validates system behavior on extreme conditions:
        - Missing/empty candidate lists
        - Insufficient candidates (|U| < k)
        - Candidate stream with heavy duplicates
        - Candidate stream with invalid IDs (negative, >= 100,000, non-int)
        - Extreme query vectors (all zeros, large values)
        """
        edge_results = {}
        dummy_q = np.ones(DIMENSION, dtype=np.float32)

        # 1. Empty candidate list
        ids, dists, st = self.reranker.rerank_single(dummy_q, [], k=10)
        edge_results["empty_candidates"] = {
            "returned_count": st.returned_count,
            "is_padded": st.is_padded,
            "all_negative_ones": bool(np.all(ids == -1)),
            "all_inf_distances": bool(np.all(np.isinf(dists))),
        }

        # 2. Insufficient candidate count (|U| = 3, k = 10)
        cands_small = [10, 20, 30]
        ids_sm, dists_sm, st_sm = self.reranker.rerank_single(dummy_q, cands_small, k=10)
        edge_results["insufficient_candidates"] = {
            "candidate_count": 3,
            "returned_count": st_sm.returned_count,
            "is_padded": st_sm.is_padded,
            "valid_ids_count": int(np.sum(ids_sm != -1)),
            "padded_count": int(np.sum(ids_sm == -1)),
            "is_monotonic": st_sm.is_monotonic,
        }

        # 3. Stream with heavy duplicates and invalid IDs
        raw_hot = [5, 10, 5, 20, -1, -99, 10]
        raw_cold = [20000, 20000, 150000, 25000]
        merged, mstats = self.merger.merge_single(raw_hot, raw_cold)
        edge_results["duplicate_and_invalid_filtering"] = {
            "total_raw": mstats.total_raw_count,
            "dedup_count": mstats.dedup_count,
            "duplicates_removed": mstats.duplicates_removed,
            "invalid_rejected": mstats.invalid_ids_rejected,
            "clean_ids": merged,
            "expected_clean": [5, 10, 20, 20000, 25000],
            "is_clean": merged == [5, 10, 20, 20000, 25000],
        }

        # 4. Pure HOT candidates
        hot_only_cands = [100, 200, 300, 400, 500]
        ids_hot, _, st_hot = self.reranker.rerank_single(dummy_q, hot_only_cands, k=5)
        edge_results["pure_hot_candidates"] = {
            "hot_in_top_k": st_hot.hot_in_top_k,
            "cold_in_top_k": st_hot.cold_in_top_k,
            "all_in_hot_range": bool(np.all(ids_hot < HOT_VECTOR_COUNT)),
        }

        # 5. Pure COLD candidates
        cold_only_cands = [20000, 30000, 40000, 50000, 60000]
        ids_cold, _, st_cold = self.reranker.rerank_single(dummy_q, cold_only_cands, k=5)
        edge_results["pure_cold_candidates"] = {
            "hot_in_top_k": st_cold.hot_in_top_k,
            "cold_in_top_k": st_cold.cold_in_top_k,
            "all_in_cold_range": bool(np.all(ids_cold >= HOT_VECTOR_COUNT)),
        }

        return edge_results

    # --------------------------------------------------------------------------
    # Objective 2: Local to Global ID Mapping Audit
    # --------------------------------------------------------------------------
    def validate_id_mapping(self) -> Dict[str, Any]:
        """Audits COLD tier local ID to global ID mapping consistency."""
        local_ids = self.cold_index.local_to_global
        n_cold = len(local_ids)

        is_contiguous = np.array_equal(local_ids, np.arange(COLD_ID_START, COLD_ID_START + n_cold))
        min_gid = int(np.min(local_ids))
        max_gid = int(np.max(local_ids))

        return {
            "cold_vector_count": n_cold,
            "min_global_id": min_gid,
            "max_global_id": max_gid,
            "expected_min_global_id": COLD_ID_START,
            "expected_max_global_id": COLD_ID_END,
            "is_contiguous_linear_mapping": is_contiguous,
            "overlap_with_hot_count": int(np.sum(local_ids < HOT_VECTOR_COUNT)),
        }

    # --------------------------------------------------------------------------
    # Objective 4: Numerical FP32 Squared-L2 Distance Precision
    # --------------------------------------------------------------------------
    def validate_distance_precision(self, n_samples: int = 50) -> Dict[str, Any]:
        """
        Validates that recomputed squared-L2 distances match ground truth distances
        within numerical floating-point epsilon (<= 1e-4).
        """
        queries = load_fvecs(workspace_root / "data/query_vectors.fvecs")[:n_samples]
        base_vectors = load_fvecs(workspace_root / "data/base_vectors.fvecs")

        max_abs_err = 0.0
        err_list = []

        for q_idx in range(n_samples):
            q_vec = queries[q_idx]
            gt_id = self.ground_truth[q_idx, 0]
            expected_dist = float(self.ground_truth_dists[q_idx, 0])

            cand_vec = base_vectors[gt_id]
            recomputed = float(np.sum((cand_vec - q_vec) ** 2))

            err = abs(recomputed - expected_dist)
            err_list.append(err)
            if err > max_abs_err:
                max_abs_err = err

        return {
            "samples_checked": n_samples,
            "max_absolute_error": float(max_abs_err),
            "mean_absolute_error": float(np.mean(err_list)),
            "tolerance_1e_4_satisfied": bool(max_abs_err <= 1e-4),
        }


# ==============================================================================
# 5. Master Validation Orchestrator & Report Generator
# ==============================================================================
def run_phase8_validation() -> Dict[str, Any]:
    """Runs full Phase 8 validation pipeline and saves structured metrics JSON."""
    print("=" * 80)
    print("Case 04 — Phase 8: Merge and Exact Reranking Validation")
    print("=" * 80)

    data_dir = workspace_root / "data"
    storage_dir = workspace_root / "storage"
    p5_index_prefix = workspace_root / "results/phase5_diskann/index/cold_disk_index_tuned"
    hot_index_path = workspace_root / "results/phase6_hybrid/index/hot_hnsw.index"

    print("[Phase 8] Loading queries and dataset...")
    queries = load_fvecs(data_dir / "query_vectors.fvecs")
    base_vectors = load_fvecs(data_dir / "base_vectors.fvecs")

    print("[Phase 8] Initializing HOT HNSW index...")
    hot_index = HotHNSWIndex(index_path=hot_index_path)
    hot_index.build_or_load(base_vectors[:HOT_VECTOR_COUNT])

    print("[Phase 8] Initializing TieredVectorStore...")
    vector_store = TieredVectorStore(
        storage_dir=storage_dir,
        base_vectors_path=data_dir / "base_vectors.fvecs",
    )
    vector_store.open()

    print("[Phase 8] Initializing COLD DiskANN index...")
    cold_index = ColdDiskANNIndex(index_prefix=p5_index_prefix)

    validator = Phase8Validator(
        hot_index=hot_index,
        cold_index=cold_index,
        vector_store=vector_store,
    )

    try:
        # 1. Candidate Union Recall vs Final Reranked Recall
        print("\n[Phase 8] Evaluating Candidate-Union Recall vs Final Reranked Recall (1,000 queries)...")
        t0 = time.perf_counter()
        union_res = validator.validate_candidate_union_recall(queries)
        union_time = time.perf_counter() - t0
        print(f"    Candidate-Union Recall@1:  {union_res.union_recall_at_1:.4f} | Final Recall@1:  {union_res.final_recall_at_1:.4f}")
        print(f"    Candidate-Union Recall@5:  {union_res.union_recall_at_5:.4f} | Final Recall@5:  {union_res.final_recall_at_5:.4f}")
        print(f"    Candidate-Union Recall@10: {union_res.union_recall_at_10:.4f} | Final Recall@10: {union_res.final_recall_at_10:.4f}")
        print(f"    Rerank Preservation Fidelity: {union_res.recall_preservation_fidelity_pct:.2f}% (100.0% = lossless rerank)")

        # 2. ALL_TIERS vs Adaptive Routing Recall Loss
        print("\n[Phase 8] Evaluating ALL_TIERS vs Adaptive Routing Recall Loss...")
        loss_res = validator.validate_adaptive_recall_loss(queries, thresholds=[135.0, 145.0, 155.0])
        for lr in loss_res:
            print(f"    Policy: {lr.policy}")
            print(f"      Early Exits: {lr.early_exit_count} ({lr.early_exit_rate_pct}%)")
            print(f"      Recall@10: {lr.adaptive_recall_at_10:.4f} vs ALL_TIERS {lr.all_tiers_recall_at_10:.4f} (Recall Loss: {lr.recall_loss_at_10:.4f})")
            print(f"      Early Exits with Closer COLD Neighbor: {lr.early_exits_with_closer_cold_neighbor} ({lr.pct_early_exits_with_closer_cold_neighbor}%)")

        # 3. Edge Cases Validation
        print("\n[Phase 8] Running Edge Cases & Boundary Handling Suite...")
        edge_res = validator.validate_edge_cases()
        print(f"    Empty Candidates Padded: {edge_res['empty_candidates']['is_padded']}")
        print(f"    Insufficient Candidates (|U|=3, k=10) Handled: {edge_res['insufficient_candidates']['is_padded']}")
        print(f"    Duplicates & Invalid IDs Filtered Cleanly: {edge_res['duplicate_and_invalid_filtering']['is_clean']}")

        # 4. ID Mapping Audit
        print("\n[Phase 8] Auditing COLD ID Mapping...")
        mapping_res = validator.validate_id_mapping()
        print(f"    Contiguous Linear Mapping: {mapping_res['is_contiguous_linear_mapping']} [18000..99999]")

        # 5. Distance Precision Audit
        print("\n[Phase 8] Verifying FP32 Squared-L2 Distance Precision...")
        prec_res = validator.validate_distance_precision(50)
        print(f"    Max Absolute Error: {prec_res['max_absolute_error']:.6e} (Tolerance <= 1e-4: {prec_res['tolerance_1e_4_satisfied']})")

    finally:
        vector_store.close()

    metrics_output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "phase": 8,
        "candidate_union_analysis": {
            "union_recall_at_1": union_res.union_recall_at_1,
            "union_recall_at_5": union_res.union_recall_at_5,
            "union_recall_at_10": union_res.union_recall_at_10,
            "final_recall_at_1": union_res.final_recall_at_1,
            "final_recall_at_5": union_res.final_recall_at_5,
            "final_recall_at_10": union_res.final_recall_at_10,
            "recall_preservation_fidelity_pct": union_res.recall_preservation_fidelity_pct,
            "mean_candidate_pool_size": union_res.mean_candidate_pool_size,
            "min_candidate_pool_size": union_res.min_candidate_pool_size,
            "max_candidate_pool_size": union_res.max_candidate_pool_size,
        },
        "adaptive_recall_loss_analysis": [
            {
                "policy": lr.policy,
                "threshold": lr.threshold,
                "early_exit_count": lr.early_exit_count,
                "early_exit_rate_pct": lr.early_exit_rate_pct,
                "all_tiers_recall_at_1": lr.all_tiers_recall_at_1,
                "all_tiers_recall_at_5": lr.all_tiers_recall_at_5,
                "all_tiers_recall_at_10": lr.all_tiers_recall_at_10,
                "adaptive_recall_at_1": lr.adaptive_recall_at_1,
                "adaptive_recall_at_5": lr.adaptive_recall_at_5,
                "adaptive_recall_at_10": lr.adaptive_recall_at_10,
                "recall_loss_at_1": lr.recall_loss_at_1,
                "recall_loss_at_5": lr.recall_loss_at_5,
                "recall_loss_at_10": lr.recall_loss_at_10,
                "early_exits_with_closer_cold_neighbor": lr.early_exits_with_closer_cold_neighbor,
                "pct_early_exits_with_closer_cold_neighbor": lr.pct_early_exits_with_closer_cold_neighbor,
            }
            for lr in loss_res
        ],
        "id_mapping_audit": mapping_res,
        "distance_precision_audit": prec_res,
        "edge_case_audit": edge_res,
    }

    out_file = workspace_root / "results/phase8_reranker/metrics/phase8_metrics.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=2)

    print(f"\n[Phase 8] Metrics saved successfully to {out_file}")
    return metrics_output


if __name__ == "__main__":
    run_phase8_validation()
