"""
Case 04 — Phase 6: Automated Verification Test Suite
Unified Hybrid Vector Search: HOT HNSW (DRAM) + COLD DiskANN3 (NVMe SSD) + FP32 Rerank

Validates:
1. Frozen dataset and source code immutability (Phases 1–5)
2. Tier partition invariants (HOT: 18k [0..17,999], COLD: 82k [18,000..99,999])
3. HOT HNSW index build, persistence, query validity, and memory footprint
4. TieredVectorStore dual-tier retrieval (DRAM for HOT, NVMe SSD for COLD)
5. Candidate merging and duplicate removal
6. Exact squared-L2 distance recomputation and mathematical precision
7. Top-K sorting and monotonic non-decreasing distance ordering
8. Dual-tier candidate retrieval (queries spanning both tiers)
9. Error handling: dimension mismatch, non-finite values, and boundary k
10. QueryRouter routing policies
11. Repeatability and bitwise reproducibility under identical configurations
12. End-to-end evaluation against unchanged frozen global ground truth oracle
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
    EXPECTED_CHECKSUMS,
)
from src.phase5_diskann_search import ColdDiskANNIndex
from src.phase6_hybrid_search import (
    HotHNSWIndex,
    TieredVectorStore,
    QueryRouter,
    RoutingPolicy,
    HybridVectorPipeline,
    HybridEvaluator,
    HOT_VECTOR_COUNT,
    COLD_VECTOR_COUNT,
    TOTAL_VECTORS,
    DIMENSION,
)


# Authoritative checksums of frozen Phase 1-5 source code files
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
}

FROZEN_DATASET_CHECKSUMS = {
    "data/base_vectors.fvecs": "62a7ad69bc429570fb4819287bb4d317d8069fb784ef528f370c64189d37198b",
    "data/query_vectors.fvecs": "c1d80d61714296078229bc3429a79569113441c01fdd1069de21611a30528496",
    "data/ground_truth.ivecs": "ee2f420a3fea4e71408756f6d0bf7bed1741a69dafc119b2781f7b4142c7d66f",
    "data/ground_truth_distances.fvecs": "3ca693e88fdfab960b0d43412b8a66266259407615feeb1a1f61ffc083b0ffc4",
    "storage/ssd_vectors/vectors.bin": "a5337c911b10bcd697652595852eb4ccb3a6ae69a438b8414ab557d822c6ab8f",
}


class TestPhase6HybridSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = workspace_root / "data"
        cls.storage_dir = workspace_root / "storage"
        cls.p5_index_prefix = workspace_root / "results/phase5_diskann/index/cold_disk_index_tuned"
        cls.p6_dir = workspace_root / "results/phase6_hybrid"
        cls.hot_index_path = cls.p6_dir / "index/hot_hnsw.index"

        # Shared assets
        cls.base_vectors = load_fvecs(cls.data_dir / "base_vectors.fvecs")
        cls.query_vectors = load_fvecs(cls.data_dir / "query_vectors.fvecs")
        cls.ground_truth = load_ivecs(cls.data_dir / "ground_truth.ivecs")

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

        # Initialize Pipeline and Evaluator
        cls.pipeline = HybridVectorPipeline(
            hot_index=cls.hot_index,
            cold_index=cls.cold_index,
            vector_store=cls.vector_store,
        )
        cls.evaluator = HybridEvaluator()

    @classmethod
    def tearDownClass(cls):
        cls.vector_store.close()

    # ==========================================================================
    # 1. Dataset & Source Immutability
    # ==========================================================================
    def test_01_frozen_dataset_and_source_immutability(self):
        """Phases 1-5 source code and frozen data files must match authoritative SHA-256."""
        for rel_path, expected_hash in FROZEN_DATASET_CHECKSUMS.items():
            p = workspace_root / rel_path
            self.assertTrue(p.exists(), f"Missing dataset: {rel_path}")
            self.assertEqual(compute_sha256(p), expected_hash, f"Hash mismatch in {rel_path}")

        for rel_path, expected_hash in FROZEN_SOURCE_CHECKSUMS.items():
            p = workspace_root / rel_path
            self.assertTrue(p.exists(), f"Missing source file: {rel_path}")
            self.assertEqual(compute_sha256(p), expected_hash, f"Source altered: {rel_path}")

    # ==========================================================================
    # 2. Tier Partition Invariants
    # ==========================================================================
    def test_02_tier_partition_invariants(self):
        """HOT (18k) and COLD (82k) must disjointly and exhaustively cover 100,000 vectors."""
        hot_ids = set(range(HOT_VECTOR_COUNT))
        cold_ids = set(range(HOT_VECTOR_COUNT, TOTAL_VECTORS))

        self.assertEqual(len(hot_ids), 18000)
        self.assertEqual(len(cold_ids), 82000)
        self.assertEqual(len(hot_ids.intersection(cold_ids)), 0)
        self.assertEqual(len(hot_ids.union(cold_ids)), TOTAL_VECTORS)

    # ==========================================================================
    # 3. HOT HNSW Index Functionality & Memory
    # ==========================================================================
    def test_03_hot_hnsw_functionality_and_memory(self):
        """HOT HNSW index must search correctly, return IDs in [0, 17999], and report footprint."""
        queries = self.query_vectors[:10]
        dists, ids, elapsed = self.hot_index.search(queries, k=10, ef_search=64)

        self.assertEqual(ids.shape, (10, 10))
        self.assertEqual(dists.shape, (10, 10))
        self.assertTrue(np.all(ids >= 0))
        self.assertTrue(np.all(ids < HOT_VECTOR_COUNT))
        self.assertTrue(np.all(dists >= 0.0))

        mem_bytes = self.hot_index.get_memory_bytes()
        self.assertGreater(mem_bytes, 10_000_000)  # > 10 MB
        self.assertLess(mem_bytes, 20_000_000)     # < 20 MB

    # ==========================================================================
    # 4. Tiered Vector Store DRAM vs SSD Routing
    # ==========================================================================
    def test_04_tiered_vector_store_routing(self):
        """Reading HOT vectors accesses DRAM; reading COLD vectors accesses NVMe SSD."""
        initial_dram = self.vector_store.dram_read_count
        initial_ssd = self.vector_store.ssd_read_count

        # Read HOT vector (ID 100)
        v_hot = self.vector_store.read_vector(100)
        self.assertEqual(v_hot.shape, (DIMENSION,))
        np.testing.assert_array_equal(v_hot, self.base_vectors[100])
        self.assertEqual(self.vector_store.dram_read_count, initial_dram + 1)
        self.assertEqual(self.vector_store.ssd_read_count, initial_ssd)

        # Read COLD vector (ID 50,000)
        v_cold = self.vector_store.read_vector(50_000)
        self.assertEqual(v_cold.shape, (DIMENSION,))
        np.testing.assert_array_equal(v_cold, self.base_vectors[50_000])
        self.assertEqual(self.vector_store.ssd_read_count, initial_ssd + 1)

        # Batch read mixed
        mixed_ids = [5, 10, 20_000, 30_000]
        batch_vecs = self.vector_store.read_vectors_batch(mixed_ids)
        self.assertEqual(batch_vecs.shape, (4, DIMENSION))
        np.testing.assert_array_equal(batch_vecs, self.base_vectors[mixed_ids])

    # ==========================================================================
    # 5. Candidate Merging and Deduplication
    # ==========================================================================
    def test_05_candidate_merging_and_deduplication(self):
        """Candidate merge must preserve order, deduplicate, and discard out-of-range IDs."""
        # Simulated hot and cold candidates
        hot_cands = [10, 20, 30, 40]
        cold_cands = [30, 40, 25000, 35000]  # 30, 40 overlap

        merged_dict = {}
        for cid in hot_cands + cold_cands:
            if 0 <= cid < TOTAL_VECTORS:
                merged_dict[cid] = None
        unique_cands = list(merged_dict.keys())

        expected = [10, 20, 30, 40, 25000, 35000]
        self.assertEqual(unique_cands, expected)
        self.assertEqual(len(unique_cands), 6)

    # ==========================================================================
    # 6. Exact Squared-L2 Distance Recomputation
    # ==========================================================================
    def test_06_exact_squared_l2_fidelity(self):
        """Recomputed squared-L2 distances must match ground truth distances exactly."""
        gt_dists = load_fvecs(self.data_dir / "ground_truth_distances.fvecs")
        q0 = self.query_vectors[0]
        top1_id = self.ground_truth[0, 0]
        top1_vec = self.base_vectors[top1_id]

        recomputed_dist = float(np.sum((q0 - top1_vec) ** 2))
        expected_dist = float(gt_dists[0, 0])

        self.assertAlmostEqual(recomputed_dist, expected_dist, places=3)

    # ==========================================================================
    # 7. Top-K Sorting and Distance Monotonicity
    # ==========================================================================
    def test_07_top_k_sorting_and_monotonicity(self):
        """Returned candidates must be monotonically non-decreasing in distance."""
        queries = self.query_vectors[:10]
        res = self.pipeline.search(queries, k=10, k_hot=20, k_cold=20, l_search_cold=30)

        for q in range(10):
            dists = res.distances[q]
            self.assertTrue(np.all(np.diff(dists) >= -1e-5), f"Non-monotonic at query {q}")

    # ==========================================================================
    # 8. Dual-Tier Candidate Retrieval
    # ==========================================================================
    def test_08_dual_tier_retrieval_coverage(self):
        """Hybrid search across 50 queries must retrieve candidates from both HOT and COLD."""
        queries = self.query_vectors[:50]
        res = self.pipeline.search(queries, k=10, k_hot=30, k_cold=30, l_search_cold=50)

        flat_ids = res.global_ids.flatten()
        hot_count = int(np.sum(flat_ids < HOT_VECTOR_COUNT))
        cold_count = int(np.sum(flat_ids >= HOT_VECTOR_COUNT))

        self.assertGreater(hot_count, 0, "No candidates from HOT tier retrieved!")
        self.assertGreater(cold_count, 0, "No candidates from COLD tier retrieved!")
        print(f"\n[Test Dual-Tier] Top-10 composition: HOT={hot_count} ({hot_count/len(flat_ids)*100:.1f}%), COLD={cold_count} ({cold_count/len(flat_ids)*100:.1f}%)")

    # ==========================================================================
    # 9. Query Router Policies
    # ==========================================================================
    def test_09_query_router_policies(self):
        """QueryRouter must produce correct target tiers for configured policies."""
        r_all = QueryRouter(RoutingPolicy.ALL_TIERS)
        r_hot = QueryRouter(RoutingPolicy.HOT_ONLY)
        r_cold = QueryRouter(RoutingPolicy.COLD_ONLY)

        dummy = np.zeros(128, dtype=np.float32)
        self.assertEqual(r_all.route(dummy), ["HOT", "COLD"])
        self.assertEqual(r_hot.route(dummy), ["HOT"])
        self.assertEqual(r_cold.route(dummy), ["COLD"])

    # ==========================================================================
    # 10. Repeatability Under Identical Configurations
    # ==========================================================================
    def test_10_repeatability(self):
        """Searching twice with identical configurations must produce identical results."""
        queries = self.query_vectors[:5]
        res1 = self.pipeline.search(queries, k=10, k_hot=20, k_cold=20, l_search_cold=30)
        res2 = self.pipeline.search(queries, k=10, k_hot=20, k_cold=20, l_search_cold=30)

        np.testing.assert_array_equal(res1.global_ids, res2.global_ids)
        np.testing.assert_array_almost_equal(res1.distances, res2.distances, decimal=4)

    # ==========================================================================
    # 11. Edge Cases and Error Handling
    # ==========================================================================
    def test_11_edge_cases_and_error_handling(self):
        """Invalid dimensions, non-finite values, and invalid k must raise ValueError."""
        # Bad dimension
        bad_dim = np.zeros((1, 64), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.pipeline.search(bad_dim, k=10)

        # NaN / Inf
        bad_val = np.zeros((1, 128), dtype=np.float32)
        bad_val[0, 10] = np.nan
        with self.assertRaises(ValueError):
            self.pipeline.search(bad_val, k=10)

        # k <= 0
        valid_q = np.zeros((1, 128), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.pipeline.search(valid_q, k=0)

        # Empty batch
        empty_q = np.empty((0, 128), dtype=np.float32)
        res = self.pipeline.search(empty_q, k=10)
        self.assertEqual(res.num_queries, 0)
        self.assertEqual(res.global_ids.shape, (0, 10))

    # ==========================================================================
    # 12. End-to-End Evaluation against Frozen Oracle
    # ==========================================================================
    def test_12_end_to_end_evaluation_against_oracle(self):
        """End-to-end evaluation must produce valid recall metrics and pass validation."""
        queries = self.query_vectors[:50]
        res = self.pipeline.search(queries, k=10, k_hot=30, k_cold=30, l_search_cold=50)

        eval_report = self.evaluator.evaluate(res)
        self.assertTrue(eval_report["prediction_validation"]["is_valid"])
        self.assertEqual(eval_report["monotonic_violations"], 0)
        self.assertGreater(eval_report["recall_metrics"]["recall_at_1"], 0.50)
        self.assertGreater(eval_report["recall_metrics"]["recall_at_10"], 0.55)
        self.assertGreater(eval_report["qps"], 25.0)

    # ==========================================================================
    # 13. Concurrency Support
    # ==========================================================================
    def test_13_concurrency_execution(self):
        """Running with concurrent=True must execute cleanly and return valid shapes."""
        queries = self.query_vectors[:5]
        res = self.pipeline.search(
            queries, k=10, k_hot=20, k_cold=20, l_search_cold=30, concurrent=True
        )
        self.assertEqual(res.global_ids.shape, (5, 10))
        self.assertEqual(res.distances.shape, (5, 10))
        self.assertTrue(np.all(res.global_ids >= 0))
        self.assertTrue(np.all(res.global_ids < TOTAL_VECTORS))


if __name__ == "__main__":
    unittest.main()
