"""
Case 04 — Phase 7: Automated Verification Test Suite
Hot/Cold Query Router Implementation on top of Phase 6 Hybrid Vector Pipeline.

Validates:
1. Frozen dataset and source code immutability (Phases 1–6)
2. RouterConfig parameter validation and error handling
3. ALL_TIERS routing policy bitwise parity with Phase 6 baseline
4. HOT_ONLY routing policy integrity: valid DRAM-resident IDs [0..17,999] and 0 SSD reads
5. COLD_ONLY routing policy integrity: valid SSD-resident IDs [18,000..99,999]
6. ADAPTIVE_THRESHOLD cascaded probe early exit vs. fallback behavior
7. ADAPTIVE_THRESHOLD parameter sweep monotonicity (early exit rate increases with tau)
8. Bitwise reproducibility and repeatability under identical configurations
9. Single-query route_decision API correctness across all policies
10. Edge cases: empty queries, dimension mismatch, non-finite values
11. End-to-end evaluation metrics and prediction validation
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
    HybridVectorPipeline,
    HOT_VECTOR_COUNT,
    COLD_VECTOR_COUNT,
    TOTAL_VECTORS,
    DIMENSION,
)
from src.phase7_query_router import (
    RoutingPolicy,
    RouterConfig,
    RoutingDecision,
    RouterSearchResult,
    AdvancedQueryRouter,
    Phase7Evaluator,
)


# Authoritative checksums of frozen Phase 1-6 source and test files
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
}

FROZEN_DATASET_CHECKSUMS = {
    "data/base_vectors.fvecs": "62a7ad69bc429570fb4819287bb4d317d8069fb784ef528f370c64189d37198b",
    "data/query_vectors.fvecs": "c1d80d61714296078229bc3429a79569113441c01fdd1069de21611a30528496",
    "data/ground_truth.ivecs": "ee2f420a3fea4e71408756f6d0bf7bed1741a69dafc119b2781f7b4142c7d66f",
    "data/ground_truth_distances.fvecs": "3ca693e88fdfab960b0d43412b8a66266259407615feeb1a1f61ffc083b0ffc4",
    "storage/ssd_vectors/vectors.bin": "a5337c911b10bcd697652595852eb4ccb3a6ae69a438b8414ab557d822c6ab8f",
}


class TestPhase7QueryRouter(unittest.TestCase):
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

        # Initialize Router and Evaluator
        cls.router = AdvancedQueryRouter(
            hot_index=cls.hot_index,
            cold_index=cls.cold_index,
            vector_store=cls.vector_store,
        )
        cls.evaluator = Phase7Evaluator()

    @classmethod
    def tearDownClass(cls):
        cls.vector_store.close()

    # ==========================================================================
    # 1. Dataset & Source Immutability (Phases 1-6)
    # ==========================================================================
    def test_01_frozen_dataset_and_source_immutability(self):
        """Phases 1-6 source code and frozen data files must match authoritative SHA-256."""
        for rel_path, expected_hash in FROZEN_DATASET_CHECKSUMS.items():
            p = workspace_root / rel_path
            self.assertTrue(p.exists(), f"Missing dataset: {rel_path}")
            self.assertEqual(compute_sha256(p), expected_hash, f"Hash mismatch in {rel_path}")

        for rel_path, expected_hash in FROZEN_SOURCE_CHECKSUMS.items():
            p = workspace_root / rel_path
            self.assertTrue(p.exists(), f"Missing source file: {rel_path}")
            self.assertEqual(compute_sha256(p), expected_hash, f"Source altered: {rel_path}")

    # ==========================================================================
    # 2. Router Configuration Validation
    # ==========================================================================
    def test_02_router_config_validation(self):
        """RouterConfig must validate all constraints and reject invalid configurations."""
        # Valid config
        valid_cfg = RouterConfig(policy=RoutingPolicy.ALL_TIERS, k=10, k_hot=20, k_cold=20, l_search_cold=30, ef_search_hot=32)
        valid_cfg.validate()

        # Invalid policy
        with self.assertRaises(ValueError):
            RouterConfig(policy="INVALID_POLICY").validate()

        # Invalid k <= 0
        with self.assertRaises(ValueError):
            RouterConfig(k=0).validate()

        # k_hot < k
        with self.assertRaises(ValueError):
            RouterConfig(k=10, k_hot=5, k_cold=20).validate()

        # k_cold < k
        with self.assertRaises(ValueError):
            RouterConfig(k=10, k_hot=20, k_cold=5).validate()

        # l_search_cold < k_cold
        with self.assertRaises(ValueError):
            RouterConfig(k=10, k_hot=20, k_cold=20, l_search_cold=15).validate()

        # Invalid threshold for adaptive policy
        with self.assertRaises(ValueError):
            RouterConfig(policy=RoutingPolicy.ADAPTIVE_THRESHOLD, threshold=-1.0).validate()
        with self.assertRaises(ValueError):
            RouterConfig(policy=RoutingPolicy.ADAPTIVE_THRESHOLD, threshold=float("nan")).validate()

    # ==========================================================================
    # 3. ALL_TIERS Baseline Parity with Phase 6
    # ==========================================================================
    def test_03_all_tiers_baseline_parity(self):
        """ALL_TIERS policy must match Phase 6 HybridVectorPipeline bitwise under identical settings."""
        queries = self.query_vectors[:10]
        cfg = RouterConfig(
            policy=RoutingPolicy.ALL_TIERS,
            k=10,
            k_hot=20,
            k_cold=20,
            ef_search_hot=32,
            l_search_cold=30,
            beam_width_cold=4,
        )

        router_res = self.router.search(queries, config=cfg)
        hybrid_res = self.router.baseline_pipeline.search(
            queries, k=10, k_hot=20, k_cold=20, ef_search_hot=32, l_search_cold=30, beam_width_cold=4
        )

        np.testing.assert_array_equal(router_res.global_ids, hybrid_res.global_ids)
        np.testing.assert_array_almost_equal(router_res.distances, hybrid_res.distances, decimal=4)
        self.assertEqual(router_res.hot_request_count, 10)
        self.assertEqual(router_res.cold_request_count, 10)
        self.assertEqual(router_res.early_exit_count, 0)

        # Check routing decisions
        self.assertEqual(len(router_res.routing_decisions), 10)
        for dec in router_res.routing_decisions:
            self.assertEqual(dec.target_tiers, ["HOT", "COLD"])
            self.assertTrue(dec.hot_probed)
            self.assertTrue(dec.cold_probed)
            self.assertFalse(dec.early_exited)

    # ==========================================================================
    # 4. HOT_ONLY Routing Policy Integrity
    # ==========================================================================
    def test_04_hot_only_policy_integrity(self):
        """HOT_ONLY policy must query DRAM only, return IDs in [0, 17999], and perform 0 SSD reads."""
        queries = self.query_vectors[:15]
        cfg = RouterConfig(
            policy=RoutingPolicy.HOT_ONLY,
            k=10,
            k_hot=30,
            ef_search_hot=32,
        )

        res = self.router.search(queries, config=cfg)

        self.assertEqual(res.global_ids.shape, (15, 10))
        self.assertTrue(np.all(res.global_ids >= 0))
        self.assertTrue(np.all(res.global_ids < HOT_VECTOR_COUNT))
        self.assertEqual(res.hot_contribution_pct, 100.0)
        self.assertEqual(res.cold_contribution_pct, 0.0)
        self.assertEqual(res.cold_request_count, 0)
        self.assertEqual(res.hot_request_count, 15)
        self.assertEqual(res.io_stats.get("ssd_reads", 0), 0)

        for dec in res.routing_decisions:
            self.assertEqual(dec.target_tiers, ["HOT"])
            self.assertTrue(dec.hot_probed)
            self.assertFalse(dec.cold_probed)

    # ==========================================================================
    # 5. COLD_ONLY Routing Policy Integrity
    # ==========================================================================
    def test_05_cold_only_policy_integrity(self):
        """COLD_ONLY policy must query SSD only, return IDs in [18000, 99999], and probe 0 HOT requests."""
        queries = self.query_vectors[:15]
        cfg = RouterConfig(
            policy=RoutingPolicy.COLD_ONLY,
            k=10,
            k_cold=30,
            l_search_cold=40,
            beam_width_cold=4,
        )

        res = self.router.search(queries, config=cfg)

        self.assertEqual(res.global_ids.shape, (15, 10))
        self.assertTrue(np.all(res.global_ids >= HOT_VECTOR_COUNT))
        self.assertTrue(np.all(res.global_ids < TOTAL_VECTORS))
        self.assertEqual(res.hot_contribution_pct, 0.0)
        self.assertEqual(res.cold_contribution_pct, 100.0)
        self.assertEqual(res.hot_request_count, 0)
        self.assertEqual(res.cold_request_count, 15)

        for dec in res.routing_decisions:
            self.assertEqual(dec.target_tiers, ["COLD"])
            self.assertFalse(dec.hot_probed)
            self.assertTrue(dec.cold_probed)

    # ==========================================================================
    # 6. ADAPTIVE_THRESHOLD Cascaded Probe Early Exit vs. Fallback
    # ==========================================================================
    def test_06_adaptive_threshold_behavior(self):
        """ADAPTIVE_THRESHOLD must exit early when min_d <= tau and cascade to COLD when min_d > tau."""
        queries = self.query_vectors[:30]

        # First measure exact min hot distances
        hot_dists, _, _ = self.hot_index.search(queries, k=1, ef_search=64)
        min_dists = hot_dists[:, 0]
        median_tau = float(np.median(min_dists))

        cfg = RouterConfig(
            policy=RoutingPolicy.ADAPTIVE_THRESHOLD,
            threshold=median_tau,
            k=10,
            k_hot=20,
            k_cold=20,
            ef_search_hot=64,
            l_search_cold=30,
        )

        res = self.router.search(queries, config=cfg)

        expected_early = int(np.sum(min_dists <= median_tau))
        expected_cold = len(queries) - expected_early

        self.assertEqual(res.early_exit_count, expected_early)
        self.assertEqual(res.cold_request_count, expected_cold)

        for i, dec in enumerate(res.routing_decisions):
            d_val = min_dists[i]
            if d_val <= median_tau:
                self.assertTrue(dec.early_exited)
                self.assertEqual(dec.target_tiers, ["HOT"])
                self.assertFalse(dec.cold_probed)
                # Early exit candidate IDs must all be in HOT tier
                self.assertTrue(np.all(res.global_ids[i] < HOT_VECTOR_COUNT))
            else:
                self.assertFalse(dec.early_exited)
                self.assertEqual(dec.target_tiers, ["HOT", "COLD"])
                self.assertTrue(dec.cold_probed)

    # ==========================================================================
    # 7. ADAPTIVE_THRESHOLD Parameter Sweep Monotonicity
    # ==========================================================================
    def test_07_adaptive_threshold_sweep_monotonicity(self):
        """Higher threshold tau must yield monotonically non-decreasing early exit counts."""
        queries = self.query_vectors[:50]
        thresholds = [135.0, 145.0, 155.0]
        early_counts = []

        for tau in thresholds:
            cfg = RouterConfig(
                policy=RoutingPolicy.ADAPTIVE_THRESHOLD,
                threshold=tau,
                k=10,
                k_hot=20,
                k_cold=20,
                ef_search_hot=32,
                l_search_cold=30,
            )
            res = self.router.search(queries, config=cfg)
            early_counts.append(res.early_exit_count)

        self.assertLessEqual(early_counts[0], early_counts[1])
        self.assertLessEqual(early_counts[1], early_counts[2])

    # ==========================================================================
    # 8. Bitwise Reproducibility Under Identical Configurations
    # ==========================================================================
    def test_08_reproducibility(self):
        """Searching twice with identical configuration must yield identical results."""
        queries = self.query_vectors[:10]
        cfg = RouterConfig(
            policy=RoutingPolicy.ADAPTIVE_THRESHOLD,
            threshold=145.0,
            k=10,
            k_hot=20,
            k_cold=20,
            ef_search_hot=32,
            l_search_cold=30,
        )

        res1 = self.router.search(queries, config=cfg)
        res2 = self.router.search(queries, config=cfg)

        np.testing.assert_array_equal(res1.global_ids, res2.global_ids)
        np.testing.assert_array_almost_equal(res1.distances, res2.distances, decimal=4)
        self.assertEqual(res1.early_exit_count, res2.early_exit_count)

    # ==========================================================================
    # 9. Single Query route_decision API
    # ==========================================================================
    def test_09_single_query_route_decision_api(self):
        """route_decision must correctly classify single queries for all policies."""
        q0 = self.query_vectors[0]

        d_all = self.router.route_decision(q0, RouterConfig(policy=RoutingPolicy.ALL_TIERS))
        self.assertEqual(d_all.target_tiers, ["HOT", "COLD"])
        self.assertFalse(d_all.early_exited)

        d_hot = self.router.route_decision(q0, RouterConfig(policy=RoutingPolicy.HOT_ONLY))
        self.assertEqual(d_hot.target_tiers, ["HOT"])

        d_cold = self.router.route_decision(q0, RouterConfig(policy=RoutingPolicy.COLD_ONLY))
        self.assertEqual(d_cold.target_tiers, ["COLD"])

        d_adapt_low = self.router.route_decision(
            q0, RouterConfig(policy=RoutingPolicy.ADAPTIVE_THRESHOLD, threshold=1.0)
        )
        self.assertEqual(d_adapt_low.target_tiers, ["HOT", "COLD"])
        self.assertFalse(d_adapt_low.early_exited)

        d_adapt_high = self.router.route_decision(
            q0, RouterConfig(policy=RoutingPolicy.ADAPTIVE_THRESHOLD, threshold=10000.0)
        )
        self.assertEqual(d_adapt_high.target_tiers, ["HOT"])
        self.assertTrue(d_adapt_high.early_exited)

    # ==========================================================================
    # 10. Edge Cases and Error Handling
    # ==========================================================================
    def test_10_edge_cases_and_error_handling(self):
        """Invalid dimensions, non-finite values, and empty queries must be handled correctly."""
        # Bad dimension
        bad_dim = np.zeros((2, 64), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.router.search(bad_dim)

        # NaN / Inf
        bad_val = np.zeros((2, 128), dtype=np.float32)
        bad_val[0, 5] = np.nan
        with self.assertRaises(ValueError):
            self.router.search(bad_val)

        # Empty batch
        empty_q = np.empty((0, 128), dtype=np.float32)
        res = self.router.search(empty_q)
        self.assertEqual(res.num_queries, 0)
        self.assertEqual(res.global_ids.shape, (0, 10))

    # ==========================================================================
    # 11. End-to-End Evaluation Report Validation
    # ==========================================================================
    def test_11_end_to_end_evaluation_validation(self):
        """Phase7Evaluator must compute valid metrics, valid percentiles, and pass validation."""
        queries = self.query_vectors[:20]
        cfg = RouterConfig(
            policy=RoutingPolicy.ALL_TIERS,
            k=10,
            k_hot=20,
            k_cold=20,
            ef_search_hot=32,
            l_search_cold=30,
        )
        res = self.router.search(queries, config=cfg)
        eval_report = self.evaluator.evaluate(res)

        self.assertTrue(eval_report["prediction_validation"]["is_valid"])
        self.assertEqual(eval_report["prediction_validation"]["duplicate_rows"], 0)
        self.assertFalse(eval_report["prediction_validation"]["has_out_of_bounds_ids"])
        self.assertEqual(eval_report["prediction_validation"]["monotonic_violations"], 0)
        self.assertGreater(eval_report["recall_metrics"]["recall_at_1"], 0.40)
        self.assertGreater(eval_report["recall_metrics"]["recall_at_10"], 0.45)
        self.assertGreater(eval_report["throughput"]["qps"], 10.0)


if __name__ == "__main__":
    unittest.main()
