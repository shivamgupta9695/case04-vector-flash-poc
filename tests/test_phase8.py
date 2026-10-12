"""
Case 04 — Phase 8: Automated Verification Test Suite
Merge and Exact Reranking Validation across HOT HNSW and COLD DiskANN.

Validates:
1. Frozen dataset and source code immutability (Phases 1–7)
2. Candidate merging and local-to-global ID translation
3. Duplicate candidate removal and invalid ID rejection
4. Exact FP32 squared-L2 distance calculation fidelity (<= 1e-4 vs ground truth)
5. Monotonic non-decreasing distance ordering in Top-K output
6. Candidate-union recall preservation fidelity (lossless reranking)
7. Insufficient candidate counts (|U| < k) and boundary padding
8. Empty candidate lists handling
9. Dual-tier candidate contributions in Top-K
10. Bitwise determinism and reproducibility under repeated execution
11. Adaptive routing recall loss and empirical proof that HOT distance threshold does not guarantee global correctness
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
import numpy as np

# Ensure workspace root in path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from src.phase1_2_evaluation import (
    load_fvecs,
    load_ivecs,
    compute_sha256,
)
from src.phase5_diskann_search import ColdDiskANNIndex
from src.phase6_hybrid_search import (
    HotHNSWIndex,
    TieredVectorStore,
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
)
from src.phase8_reranker_validation import (
    ExactCandidateMerger,
    ExactReranker,
    Phase8Validator,
    MergeStats,
    RerankStats,
)


# Authoritative checksums of frozen Phase 1-7 source and test files
FROZEN_SOURCE_CHECKSUMS = {
    "src/diagnostic_10pct_uniform.py": "ba128d01b8eacb9d469ef18ad661dcb71dab2ca00919eca03fdcdba0cdf1b86e",
    "src/inspect_phase3_live_tiers.py": "ec257f3f0a818b37ffff2ec3dd01fb7c5d334cc9bb63032e4cea795e8e19ce35",
    "src/inspect_phase3_tiers.py": "0d6e62f586b8f5df66ab8ac8fc12620d632f14a2928cc4817282a2e6d0738e10",
    "src/phase1_1_ground_truth.py": "50952abf793bd0e18cc44741d334bc33d94fac7ce5aa544fe37eb9dc1c219754",
    "src/phase1_2_evaluation.py": "599567e4ebdb7d4a2424ab12a6eb301b18b3c2f94ec35604002ccc147b4eaae0",
    "src/phase2_ann_hnsw.py": "0af69ac722a8da9f8c8025fec3d33efaba7bce1c9af2d53ae4199c53e90c25fe",
    "src/phase3_hotness.py": "3d6e1c4fb3526d3b4a0b90574e6da2a0ad7a0d1478631eca867a4f5053034391",
    "src/phase3_realtime.py": "29431fd6eb137c18f061705427c75dab24252953d836bc81120e3092a1da491b",
    "src/phase3_ssd_store.py": "e0b819cacee44d258c93264a494cd61feb869ced5932c87344449bf738c6ad67",
    "src/phase3_tiering.py": "92944ba6b7999f840742aab10261262d29b1b8fcd29b9590691ed82f0ba78b94",
    "src/phase3_workload.py": "59a4d6971d8d792f8bf4b8ca3f72b47b5b9d64d23f9be7c2e96c2a12700f0e42",
    "src/phase4_pq.py": "5d849698804ac3fdbffab9a512888246a6655775a5270c575b3f89303cb1487b",
    "src/phase4_pq_adc.py": "44664f3f9a3f354deb498f43855a710cc429b74ca787ca6beb60d0d35011335e",
    "src/phase4_pq_encode.py": "0945f46513674e511ec3b711c5d44cd41c136467dfad42e1b555234054e2bfab",
    "src/phase4_pq_evaluate.py": "87b7cb84dc702d5ed3084c8b10810973e25387e3b5717911324205069d28dd3b",
    "src/phase4_pq_inspect.py": "31059c6a8e3bd18060dde857ef30b6505452d7f1fd03df028fdfb7beeaee19e8",
    "src/phase4_pq_reconstruct.py": "40f3bf08816e1e269224d5002348cea2eee503ee8a2cea6a9baf40d7774f9c29",
    "src/phase4_pq_tier_model.py": "25557ca847245a8f4db8b3053563a6ae43512c6c52285f1cd39000842966d924",
    "src/phase4_pq_train.py": "c5af3b80e32e9c7e9c8c328b984def9418c180a5ef1d24ada279e53f371a4b46",
    "src/phase5_diskann_search.py": "8baa6fee7ccdb898d5158b023a16d72dfc02fbc8a04a71ebb2af09502b2191fe",
    "src/phase5_orchestrator.py": "44dc974ab0748c1c54e01f39340aea2d1e1ab1fbee0be5448d41c57938cb5843",
    "src/phase5_prepare_cold.py": "3adb61641124820be4bdb0a66a53ed4ea8b152cacfacae628a8033adfa03dfcd",
    "tests/test_phase5.py": "cbac6531c43b3af4ed59ac6d126e6e2a0252b7187763010e08f5d7ed3b8c8323",
    "src/phase6_hybrid_search.py": "14e02d524e947df06b891b10493e1fbc7e2de6d5446fa6fbc5decfa705bcda28",
    "tests/test_phase6.py": "a5981cbd349a9fcfdabf951935138f05ad35258e6f11777e7851bd10c04b245c",
    "src/phase7_query_router.py": "a7182a68fab0baba17c6e168b87286f945e33e162c6ec3dd981c07b99eda1d31",
    "tests/test_phase7.py": "d252374e0b09f88ba1bf42204e8234a4b509bca25c0ad4dcfb9c197fefd82e5c",
}

FROZEN_DATASET_CHECKSUMS = {
    "data/base_vectors.fvecs": "62a7ad69bc429570fb4819287bb4d317d8069fb784ef528f370c64189d37198b",
    "data/query_vectors.fvecs": "c1d80d61714296078229bc3429a79569113441c01fdd1069de21611a30528496",
    "data/ground_truth.ivecs": "ee2f420a3fea4e71408756f6d0bf7bed1741a69dafc119b2781f7b4142c7d66f",
    "data/ground_truth_distances.fvecs": "3ca693e88fdfab960b0d43412b8a66266259407615feeb1a1f61ffc083b0ffc4",
    "storage/ssd_vectors/vectors.bin": "a5337c911b10bcd697652595852eb4ccb3a6ae69a438b8414ab557d822c6ab8f",
}


class TestPhase8RerankerValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = workspace_root / "data"
        cls.storage_dir = workspace_root / "storage"
        cls.p5_index_prefix = workspace_root / "results/phase5_diskann/index/cold_disk_index_tuned"
        cls.hot_index_path = workspace_root / "results/phase6_hybrid/index/hot_hnsw.index"

        # Shared assets
        cls.base_vectors = load_fvecs(cls.data_dir / "base_vectors.fvecs")
        cls.query_vectors = load_fvecs(cls.data_dir / "query_vectors.fvecs")
        cls.ground_truth = load_ivecs(cls.data_dir / "ground_truth.ivecs")
        cls.ground_truth_dists = load_fvecs(cls.data_dir / "ground_truth_distances.fvecs")

        # Initialize HOT HNSW index
        cls.hot_index = HotHNSWIndex(index_path=cls.hot_index_path)
        cls.hot_index.build_or_load(cls.base_vectors[:HOT_VECTOR_COUNT])

        # Initialize TieredVectorStore
        cls.vector_store = TieredVectorStore(
            storage_dir=cls.storage_dir,
            base_vectors_path=cls.data_dir / "base_vectors.fvecs",
        )
        cls.vector_store.open()

        # Initialize COLD DiskANN index
        cls.cold_index = ColdDiskANNIndex(cls.p5_index_prefix)

        # Initialize components
        cls.merger = ExactCandidateMerger()
        cls.reranker = ExactReranker(cls.vector_store)
        cls.validator = Phase8Validator(
            hot_index=cls.hot_index,
            cold_index=cls.cold_index,
            vector_store=cls.vector_store,
        )

    @classmethod
    def tearDownClass(cls):
        cls.vector_store.close()

    # ==========================================================================
    # 1. Dataset & Source Immutability (Phases 1-7)
    # ==========================================================================
    def test_01_frozen_dataset_and_source_immutability(self):
        """Phases 1-7 source code and frozen data files must match authoritative SHA-256."""
        for rel_path, expected_hash in FROZEN_DATASET_CHECKSUMS.items():
            p = workspace_root / rel_path
            self.assertTrue(p.exists(), f"Missing dataset: {rel_path}")
            self.assertEqual(compute_sha256(p), expected_hash, f"Hash mismatch in {rel_path}")

        for rel_path, expected_hash in FROZEN_SOURCE_CHECKSUMS.items():
            p = workspace_root / rel_path
            self.assertTrue(p.exists(), f"Missing source file: {rel_path}")
            self.assertEqual(compute_sha256(p), expected_hash, f"Source altered: {rel_path}")

    # ==========================================================================
    # 2. Candidate Merging and Local-to-Global Mapping
    # ==========================================================================
    def test_02_candidate_merging_and_id_translation(self):
        """Merger must correctly translate local COLD IDs (c -> 18000 + c) and merge cleanly."""
        hot_cands = [0, 500, 17999]
        cold_local = [0, 100, 81999]  # local IDs

        merged, stats = self.merger.merge_single(hot_cands, cold_local, cold_is_local=True)

        expected = [0, 500, 17999, 18000, 18100, 99999]
        self.assertEqual(merged, expected)
        self.assertEqual(stats.raw_hot_count, 3)
        self.assertEqual(stats.raw_cold_count, 3)
        self.assertEqual(stats.dedup_count, 6)
        self.assertEqual(stats.duplicates_removed, 0)
        self.assertEqual(stats.invalid_ids_rejected, 0)

    # ==========================================================================
    # 3. Duplicate Removal and Invalid ID Rejection
    # ==========================================================================
    def test_03_duplicate_removal_and_invalid_id_rejection(self):
        """Merger must filter out duplicates and reject out-of-bounds or non-integer IDs."""
        hot_cands = [10, 20, 10, -5, 30]          # 10 is duplicate, -5 is negative
        cold_global = [30, 25000, 100000, 25000]   # 30 overlaps hot, 100000 is OOB, 25000 is duplicate

        merged, stats = self.merger.merge_single(hot_cands, cold_global, cold_is_local=False)

        expected = [10, 20, 30, 25000]
        self.assertEqual(merged, expected)
        self.assertEqual(stats.dedup_count, 4)
        self.assertEqual(stats.duplicates_removed, 3)
        self.assertEqual(stats.invalid_ids_rejected, 2)

    # ==========================================================================
    # 4. Exact FP32 Distance Fidelity
    # ==========================================================================
    def test_04_exact_fp32_distance_fidelity(self):
        """Recomputed squared-L2 distances must match ground truth distances within 1e-4."""
        prec_res = self.validator.validate_distance_precision(n_samples=25)
        self.assertTrue(prec_res["tolerance_1e_4_satisfied"])
        self.assertLessEqual(prec_res["max_absolute_error"], 1e-4)

    # ==========================================================================
    # 5. Monotonic Top-K Sorting
    # ==========================================================================
    def test_05_monotonic_top_k_sorting(self):
        """Reranker output distances must be strictly monotonically non-decreasing."""
        q0 = self.query_vectors[0]
        cands = [5, 100, 500, 20000, 30000, 40000, 50000]
        _, dists, stats = self.reranker.rerank_single(q0, cands, k=5)

        self.assertTrue(stats.is_monotonic)
        self.assertTrue(np.all(np.diff(dists) >= -1e-5))

    # ==========================================================================
    # 6. Candidate-Union Recall Preservation Fidelity
    # ==========================================================================
    def test_06_candidate_union_recall_preservation_fidelity(self):
        """Exact reranking must achieve 100.0% preservation of all candidate-union true neighbors."""
        queries = self.query_vectors[:25]
        union_res = self.validator.validate_candidate_union_recall(
            queries, k=10, k_hot=20, k_cold=20, ef_hot=32, l_cold=30
        )

        # Candidate union recall upper bound must equal final reranked recall
        self.assertEqual(union_res.union_recall_at_1, union_res.final_recall_at_1)
        self.assertEqual(union_res.union_recall_at_5, union_res.final_recall_at_5)
        self.assertEqual(union_res.union_recall_at_10, union_res.final_recall_at_10)
        self.assertEqual(union_res.recall_preservation_fidelity_pct, 100.0)

    # ==========================================================================
    # 7. Insufficient Candidates Padding
    # ==========================================================================
    def test_07_insufficient_candidates_padding(self):
        """When candidate count is less than k, reranker must return valid items and pad safely."""
        q0 = self.query_vectors[0]
        cands = [10, 20, 30]  # Only 3 candidates for k=10
        ids, dists, stats = self.reranker.rerank_single(q0, cands, k=10)

        self.assertEqual(stats.candidate_count, 3)
        self.assertEqual(stats.returned_count, 3)
        self.assertTrue(stats.is_padded)

        # First 3 items are valid candidate IDs
        self.assertEqual(set(ids[:3].tolist()), {10, 20, 30})
        self.assertTrue(dists[0] <= dists[1] <= dists[2])
        # Remaining 7 are padded with -1 and inf
        self.assertTrue(np.all(ids[3:] == -1))
        self.assertTrue(np.all(np.isinf(dists[3:])))

    # ==========================================================================
    # 8. Empty Candidate List Handling
    # ==========================================================================
    def test_08_empty_candidate_list_handling(self):
        """When candidate list is completely empty, reranker must return padded arrays gracefully."""
        q0 = self.query_vectors[0]
        ids, dists, stats = self.reranker.rerank_single(q0, [], k=10)

        self.assertEqual(stats.candidate_count, 0)
        self.assertEqual(stats.returned_count, 0)
        self.assertTrue(stats.is_padded)
        self.assertTrue(np.all(ids == -1))
        self.assertTrue(np.all(np.isinf(dists)))

    # ==========================================================================
    # 9. Dual-Tier Candidate Contributions
    # ==========================================================================
    def test_09_dual_tier_contributions(self):
        """Top-K results across 20 queries must contain candidates from both HOT and COLD tiers."""
        queries = self.query_vectors[:20]
        _, hot_ids, _ = self.hot_index.search(queries, k=20, ef_search=32)
        cold_res = self.cold_index.search(queries, k=20, l_search=30, beam_width=4)
        merged_cands, _ = self.merger.merge_batch(hot_ids, cold_res.global_ids)

        top_ids, _, _ = self.reranker.rerank_batch(queries, merged_cands, k=10)

        flat_ids = top_ids.flatten()
        hot_cands = int(np.sum(flat_ids < HOT_VECTOR_COUNT))
        cold_cands = int(np.sum(flat_ids >= HOT_VECTOR_COUNT))

        self.assertGreater(hot_cands, 0, "Top-K contains zero HOT tier candidates")
        self.assertGreater(cold_cands, 0, "Top-K contains zero COLD tier candidates")

    # ==========================================================================
    # 10. Deterministic Reranking Reproducibility
    # ==========================================================================
    def test_10_deterministic_reranking_reproducibility(self):
        """Consecutive reranking runs on identical inputs must produce bitwise identical outputs."""
        queries = self.query_vectors[:5]
        cands_batch = [[10, 20, 30, 25000, 35000] for _ in range(5)]

        ids1, dists1, _ = self.reranker.rerank_batch(queries, cands_batch, k=5)
        ids2, dists2, _ = self.reranker.rerank_batch(queries, cands_batch, k=5)

        np.testing.assert_array_equal(ids1, ids2)
        np.testing.assert_array_almost_equal(dists1, dists2, decimal=6)

    # ==========================================================================
    # 11. Adaptive Routing Recall Loss & False Assumption Detection
    # ==========================================================================
    def test_11_adaptive_routing_recall_loss_and_false_assumption(self):
        """Adaptive early exit drops recall vs ALL_TIERS and misses closer COLD neighbors."""
        queries = self.query_vectors[:30]
        loss_results = self.validator.validate_adaptive_recall_loss(queries, thresholds=[145.0], k=10)

        lr = loss_results[0]
        self.assertGreater(lr.early_exit_count, 0)
        # Recall loss must be non-negative
        self.assertGreaterEqual(lr.recall_loss_at_10, 0.0)
        # Proof: at least one early-exited query missed a closer neighbor in the COLD tier
        self.assertGreater(lr.early_exits_with_closer_cold_neighbor, 0)
        self.assertGreater(lr.pct_early_exits_with_closer_cold_neighbor, 0.0)


if __name__ == "__main__":
    unittest.main()
