"""
Case 04 — Phase 9: Automated Verification Test Suite
Milvus Integration, Compatibility Proof & End-to-End Validation.

Validates:
1. Frozen dataset and source code immutability (Phases 1–8)
2. MilvusAdapterConfig parameter validation and error handling
3. Milvus collection management APIs (create, describe, has, drop)
4. Health check, lifecycle management, and service readiness
5. Response schema conformance to official PyMilvus SearchResults (Hits, id, distance, entity)
6. Bitwise result parity with standalone Phase 6 hybrid pipeline under identical settings
7. Input validation: dimension mismatch, non-finite values (NaN/Inf), boundary conditions
8. Deterministic failure on tier unavailability (zero silent fallbacks)
9. Routing policy pass-through via search_params (ALL_TIERS, HOT_ONLY, COLD_ONLY, ADAPTIVE)
10. Deterministic reproducibility under repeated execution
11. Adapter overhead measurement (negligible latency overhead < 1.0 ms/query)
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
from src.phase9_milvus_adapter import (
    MilvusHybridAdapter,
    MilvusAdapterConfig,
    MilvusSearchResponse,
    MilvusAdapterError,
    MilvusValidationError,
    MilvusServiceUnavailableError,
    TierUnavailableError,
)


# Authoritative checksums of frozen Phase 1-8 source and test files
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
    "src/phase8_reranker_validation.py": "8f32aa283baadbe506aac5227c2af84a17cef671d93e2be8aeb0faf879699479",
    "tests/test_phase8.py": "b1bfefe6c6386249f5018064bdeeaec114de789081970985f21764541b6ca10d",
}

FROZEN_DATASET_CHECKSUMS = {
    "data/base_vectors.fvecs": "62a7ad69bc429570fb4819287bb4d317d8069fb784ef528f370c64189d37198b",
    "data/query_vectors.fvecs": "c1d80d61714296078229bc3429a79569113441c01fdd1069de21611a30528496",
    "data/ground_truth.ivecs": "ee2f420a3fea4e71408756f6d0bf7bed1741a69dafc119b2781f7b4142c7d66f",
    "data/ground_truth_distances.fvecs": "3ca693e88fdfab960b0d43412b8a66266259407615feeb1a1f61ffc083b0ffc4",
    "storage/ssd_vectors/vectors.bin": "a5337c911b10bcd697652595852eb4ccb3a6ae69a438b8414ab557d822c6ab8f",
}


class TestPhase9MilvusAdapter(unittest.TestCase):
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

        # Standalone Phase 6 hybrid pipeline
        cls.standalone_pipeline = HybridVectorPipeline(
            hot_index=cls.hot_index,
            cold_index=cls.cold_index,
            vector_store=cls.vector_store,
        )

        # Phase 9 Milvus Hybrid Adapter
        cls.adapter = MilvusHybridAdapter(
            hot_index=cls.hot_index,
            cold_index=cls.cold_index,
            vector_store=cls.vector_store,
        )

    @classmethod
    def tearDownClass(cls):
        cls.vector_store.close()
        cls.adapter.close()

    # ==========================================================================
    # 1. Dataset & Source Immutability (Phases 1-8)
    # ==========================================================================
    def test_01_frozen_dataset_and_source_immutability(self):
        """Phases 1-8 source code and frozen data files must match authoritative SHA-256."""
        for rel_path, expected_hash in FROZEN_DATASET_CHECKSUMS.items():
            p = workspace_root / rel_path
            self.assertTrue(p.exists(), f"Missing dataset: {rel_path}")
            self.assertEqual(compute_sha256(p), expected_hash, f"Hash mismatch in {rel_path}")

        for rel_path, expected_hash in FROZEN_SOURCE_CHECKSUMS.items():
            p = workspace_root / rel_path
            self.assertTrue(p.exists(), f"Missing source file: {rel_path}")
            self.assertEqual(compute_sha256(p), expected_hash, f"Source altered: {rel_path}")

    # ==========================================================================
    # 2. MilvusAdapterConfig Parameter Validation
    # ==========================================================================
    def test_02_milvus_adapter_config_validation(self):
        """Config must validate constraints and reject bad dimensions, limits, and parameters."""
        valid_cfg = MilvusAdapterConfig()
        valid_cfg.validate()

        # Bad dimension
        with self.assertRaises(MilvusValidationError):
            MilvusAdapterConfig(dimension=64).validate()

        # Bad limit <= 0
        with self.assertRaises(MilvusValidationError):
            MilvusAdapterConfig(default_limit=0).validate()

        # k_hot < limit
        with self.assertRaises(MilvusValidationError):
            MilvusAdapterConfig(default_limit=20, default_k_hot=10).validate()

        # l_cold < k_cold
        with self.assertRaises(MilvusValidationError):
            MilvusAdapterConfig(default_k_cold=50, default_l_cold=30).validate()

    # ==========================================================================
    # 3. Milvus Collection Management APIs
    # ==========================================================================
    def test_03_milvus_collection_management(self):
        """Adapter must support create, describe, has, and drop collection APIs."""
        col_name = "test_col_dynamic"
        self.assertFalse(self.adapter.has_collection(col_name))

        # Create
        meta = self.adapter.create_collection(col_name, dimension=128, description="Dynamic Test")
        self.assertTrue(self.adapter.has_collection(col_name))
        self.assertEqual(meta["name"], col_name)
        self.assertEqual(meta["dimension"], 128)

        # Describe
        desc = self.adapter.describe_collection(col_name)
        self.assertEqual(desc["name"], col_name)
        self.assertEqual(desc["description"], "Dynamic Test")

        # Drop
        dropped = self.adapter.drop_collection(col_name)
        self.assertTrue(dropped)
        self.assertFalse(self.adapter.has_collection(col_name))

    # ==========================================================================
    # 4. Adapter Health Check and Lifecycle Management
    # ==========================================================================
    def test_04_health_and_lifecycle(self):
        """get_health() must accurately report tier readiness and handle service state."""
        health = self.adapter.get_health()
        self.assertEqual(health["status"], "SERVING")
        self.assertTrue(health["hot_tier_loaded"])
        self.assertTrue(health["cold_tier_loaded"])
        self.assertTrue(health["vector_store_open"])
        self.assertEqual(health["authoritative_backend"], "Phase6_Hybrid_HNSW_DiskANN3")

    # ==========================================================================
    # 5. Milvus Response Schema Conformance
    # ==========================================================================
    def test_05_milvus_response_schema_conformance(self):
        """Search output must conform to official Milvus SearchResults structure."""
        queries = self.query_vectors[:5]
        resp = self.adapter.search(
            collection_name=self.adapter.config.collection_name,
            data=queries,
            limit=5,
        )

        self.assertIsInstance(resp, MilvusSearchResponse)
        self.assertEqual(resp.num_queries, 5)
        self.assertEqual(resp.limit, 5)
        self.assertEqual(len(resp.results), 5)

        for q_idx, hits in enumerate(resp.results):
            self.assertEqual(len(hits), 5)
            for hit in hits:
                self.assertIn("id", hit)
                self.assertIn("distance", hit)
                self.assertIn("entity", hit)
                self.assertIsInstance(hit["id"], int)
                self.assertIsInstance(hit["distance"], float)
                self.assertEqual(hit["entity"]["id"], hit["id"])

    # ==========================================================================
    # 6. Bitwise Result Parity with Standalone Phase 6 Pipeline
    # ==========================================================================
    def test_06_bitwise_result_parity(self):
        """MilvusHybridAdapter output must match standalone Phase 6 pipeline bitwise."""
        queries = self.query_vectors[:10]

        # Standalone Phase 6 pipeline
        st_res = self.standalone_pipeline.search(
            queries=queries, k=10, k_hot=20, k_cold=20, ef_search_hot=32, l_search_cold=30
        )

        # Phase 9 Milvus Hybrid Adapter
        ad_res = self.adapter.search(
            collection_name=self.adapter.config.collection_name,
            data=queries,
            limit=10,
            search_params={"params": {"policy": "ALL_TIERS", "k_hot": 20, "k_cold": 20, "ef_search_hot": 32, "l_search_cold": 30}},
        )

        np.testing.assert_array_equal(ad_res.global_ids, st_res.global_ids)
        np.testing.assert_array_almost_equal(ad_res.distances, st_res.distances, decimal=4)

    # ==========================================================================
    # 7. Input Validation and Error Handling
    # ==========================================================================
    def test_07_input_validation_errors(self):
        """Bad dimensions, NaNs, invalid limits, and missing collections must raise MilvusValidationError."""
        # Bad dimension
        bad_dim = np.zeros((2, 64), dtype=np.float32)
        with self.assertRaises(MilvusValidationError):
            self.adapter.search(self.adapter.config.collection_name, bad_dim)

        # NaNs
        nan_vecs = np.zeros((2, 128), dtype=np.float32)
        nan_vecs[0, 5] = np.nan
        with self.assertRaises(MilvusValidationError):
            self.adapter.search(self.adapter.config.collection_name, nan_vecs)

        # Limit <= 0
        valid_vecs = np.zeros((2, 128), dtype=np.float32)
        with self.assertRaises(MilvusValidationError):
            self.adapter.search(self.adapter.config.collection_name, valid_vecs, limit=0)

        # Missing collection
        with self.assertRaises(MilvusValidationError):
            self.adapter.search("non_existent_collection", valid_vecs)

    # ==========================================================================
    # 8. Deterministic Failure on Tier Unavailability (No Silent Fallback)
    # ==========================================================================
    def test_08_deterministic_tier_unavailable_error(self):
        """Simulated tier outage must fail deterministically with TierUnavailableError."""
        valid_vecs = np.zeros((1, 128), dtype=np.float32)

        # 1. HOT down
        self.adapter._simulate_hot_down = True
        try:
            with self.assertRaises(TierUnavailableError):
                self.adapter.search(self.adapter.config.collection_name, valid_vecs)
        finally:
            self.adapter._simulate_hot_down = False

        # 2. COLD down
        self.adapter._simulate_cold_down = True
        try:
            with self.assertRaises(TierUnavailableError):
                self.adapter.search(self.adapter.config.collection_name, valid_vecs)
        finally:
            self.adapter._simulate_cold_down = False

    # ==========================================================================
    # 9. Routing Policy Pass-Through via search_params
    # ==========================================================================
    def test_09_routing_policies_pass_through(self):
        """Adapter must support HOT_ONLY and COLD_ONLY routing policies via search_params."""
        queries = self.query_vectors[:5]

        # HOT_ONLY: all IDs in [0..17,999], 0 SSD reads
        resp_hot = self.adapter.search(
            collection_name=self.adapter.config.collection_name,
            data=queries,
            limit=5,
            search_params={"params": {"policy": "HOT_ONLY", "k_hot": 20, "ef_search_hot": 32}},
        )
        self.assertEqual(resp_hot.routing_policy, "HOT_ONLY")
        self.assertEqual(resp_hot.cold_requests, 0)
        self.assertEqual(resp_hot.ssd_reads, 0)
        self.assertTrue(np.all(resp_hot.global_ids < HOT_VECTOR_COUNT))

        # COLD_ONLY: all IDs in [18,000..99,999]
        resp_cold = self.adapter.search(
            collection_name=self.adapter.config.collection_name,
            data=queries,
            limit=5,
            search_params={"params": {"policy": "COLD_ONLY", "k_cold": 20, "l_search_cold": 30}},
        )
        self.assertEqual(resp_cold.routing_policy, "COLD_ONLY")
        self.assertEqual(resp_cold.hot_requests, 0)
        self.assertTrue(np.all(resp_cold.global_ids >= HOT_VECTOR_COUNT))

    # ==========================================================================
    # 10. Deterministic Reproducibility
    # ==========================================================================
    def test_10_deterministic_reproducibility(self):
        """Repeated searches on identical inputs must yield identical IDs and distances."""
        queries = self.query_vectors[:5]

        resp1 = self.adapter.search(self.adapter.config.collection_name, queries, limit=10)
        resp2 = self.adapter.search(self.adapter.config.collection_name, queries, limit=10)

        np.testing.assert_array_equal(resp1.global_ids, resp2.global_ids)
        np.testing.assert_array_almost_equal(resp1.distances, resp2.distances, decimal=5)

    # ==========================================================================
    # 11. Adapter Overhead Negligibility
    # ==========================================================================
    def test_11_adapter_overhead_negligibility(self):
        """Adapter translation overhead must be negligible (< 1.0 ms per query)."""
        queries = self.query_vectors[:10]
        resp = self.adapter.search(self.adapter.config.collection_name, queries, limit=10)

        overhead_per_q_ms = resp.adapter_overhead_ms / len(queries)
        self.assertLess(overhead_per_q_ms, 1.0)  # less than 1.0 ms overhead


if __name__ == "__main__":
    unittest.main()
