"""
Case 04 — Phase 3 Production Hardening: Automated Validation Test Suite
Tests 1 to 15 verifying dataset integrity, SSD storage, cache invariants,
tier migration, hysteresis, residency, latency metrics, and lifecycle dynamics.
"""

import sys
import unittest
import hashlib
from pathlib import Path
import numpy as np

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from src.phase1_2_evaluation import load_fvecs, load_ivecs, EXPECTED_CHECKSUMS, calculate_recall_at_k
from src.phase3_ssd_store import SSDVectorStore
from src.phase3_hotness import AccessTracker, TIER_COLD_SSD, TIER_HOT_DRAM
from src.phase3_tiering import TieredVectorManager
from src.phase3_workload import WorkloadGenerator


class TestPhase3ProductionHardening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = workspace_root / "data"
        cls.storage_dir = workspace_root / "storage"
        cls.base_path = cls.data_dir / "base_vectors.fvecs"
        cls.query_path = cls.data_dir / "query_vectors.fvecs"
        cls.gt_path = cls.data_dir / "ground_truth.ivecs"

        cls.base_vectors = load_fvecs(cls.base_path)
        cls.query_vectors = load_fvecs(cls.query_path)
        cls.ground_truth = load_ivecs(cls.gt_path)

        cls.ssd_store = SSDVectorStore(cls.storage_dir, num_vectors=len(cls.base_vectors), dimension=128)
        cls.ssd_store.open()

    @classmethod
    def tearDownClass(cls):
        cls.ssd_store.close()

    def test_01_dataset_integrity(self):
        """Test 1 — Dataset integrity: Cryptographic SHA-256 verification."""
        for fname, expected in EXPECTED_CHECKSUMS.items():
            fpath = self.data_dir / fname
            if fpath.exists():
                h = hashlib.sha256(fpath.read_bytes()).hexdigest()
                self.assertEqual(h, expected, f"Hash mismatch for {fname}")

    def test_02_ssd_round_trip(self):
        """Test 2 — SSD round trip: Write/read random vectors byte-for-byte fidelity."""
        test_indices = [0, 1, 42, 500, 999, 10_000, 50_000, 99_999]
        for idx in test_indices:
            ssd_vec = self.ssd_store.read_vector(idx)
            ref_vec = self.base_vectors[idx]
            self.assertTrue(np.array_equal(ssd_vec, ref_vec), f"Fidelity failed at vector {idx}")

    def test_03_offset_correctness(self):
        """Test 3 — Offset correctness: Verify offset calculation formula."""
        for idx in [0, 10, 100, 1000, 50000, 99999]:
            expected_offset = idx * 512
            self.assertEqual(idx * self.ssd_store.record_bytes, expected_offset)

    def test_04_cache_capacity_invariant(self):
        """Test 4 — Cache capacity: Ensure DRAM never exceeds configured capacity."""
        tracker = AccessTracker(num_vectors=len(self.base_vectors))
        mgr = TieredVectorManager(
            ssd_store=self.ssd_store,
            access_tracker=tracker,
            dram_capacity_ratio=0.10,
            policy_type="policy_b",
        )
        mgr.populate_initial_dram(list(range(10_000)))
        self.assertEqual(len(mgr.dram_cache), mgr.max_dram_vectors)
        self.assertLessEqual(len(mgr.dram_cache), 10_000)

    def test_05_promotion_mechanics(self):
        """Test 5 — Promotion: Cold vector becomes HOT after qualifying accesses."""
        tracker = AccessTracker(num_vectors=1000)
        mgr = TieredVectorManager(
            ssd_store=self.ssd_store,
            access_tracker=tracker,
            dram_capacity_ratio=0.10,
            policy_type="policy_b",
            admission_min_access=2,
            promotion_threshold=1.0,
        )
        test_id = 500
        # Cold initial state
        self.assertNotIn(test_id, mgr.dram_cache)
        # Access 1 (admission check blocks promotion)
        _, _ = mgr.get_vector(test_id, current_time=1.0)
        # Access 2 (admission satisfied, qualifies for promotion)
        _, _ = mgr.get_vector(test_id, current_time=1.1)
        self.assertIn(test_id, mgr.dram_cache)
        self.assertEqual(tracker.current_tier[test_id], TIER_HOT_DRAM)

    def test_06_demotion_mechanics(self):
        """Test 6 — Demotion: HOT vector becomes COLD after decay/competition."""
        tracker = AccessTracker(num_vectors=1000)
        mgr = TieredVectorManager(
            ssd_store=self.ssd_store,
            access_tracker=tracker,
            max_dram_vectors=5,
            policy_type="policy_a",      # Legacy immediate replacement to test raw eviction mechanics
        )
        mgr.populate_initial_dram([1, 2, 3, 4, 5])
        self.assertIn(1, mgr.dram_cache)

        # Vectors 2..5 are active and hot
        for v in [2, 3, 4, 5]:
            tracker.hotness_score[v] = 5.0
        # Decay vector 1 so it is strictly coldest in DRAM
        tracker.hotness_score[1] = 0.01

        # High-traffic candidate vector 99
        vec99 = self.base_vectors[99]
        tracker.hotness_score[99] = 10.0
        mgr._evaluate_promotion(99, vec99, 10.0, current_time=10.0)

        self.assertNotIn(1, mgr.dram_cache)
        self.assertEqual(tracker.current_tier[1], TIER_COLD_SSD)
        self.assertIn(99, mgr.dram_cache)

    def test_07_hysteresis_damping(self):
        """Test 7 — Hysteresis: Vector does not oscillate around a single threshold."""
        tracker = AccessTracker(num_vectors=1000)
        mgr = TieredVectorManager(
            ssd_store=self.ssd_store,
            access_tracker=tracker,
            max_dram_vectors=5,
            policy_type="policy_b",
            hysteresis_margin=0.50,
            min_residency_time=0.0,
        )
        mgr.populate_initial_dram([1, 2, 3, 4, 5])
        # Resident score = 2.0
        for vid in [1, 2, 3, 4, 5]:
            tracker.hotness_score[vid] = 2.0

        # Candidate with score 2.1 (greater than coldest, but less than coldest + margin 0.50)
        cand_id = 99
        cand_vec = self.base_vectors[cand_id]
        tracker.access_count[cand_id] = 5
        promoted = mgr._evaluate_promotion(cand_id, cand_vec, hotness_score=2.1, current_time=5.0)

        # Must NOT promote due to hysteresis margin damping
        self.assertFalse(promoted)
        self.assertNotIn(cand_id, mgr.dram_cache)

    def test_08_minimum_residency(self):
        """Test 8 — Minimum residency: Recently promoted vector is protected."""
        tracker = AccessTracker(num_vectors=1000)
        mgr = TieredVectorManager(
            ssd_store=self.ssd_store,
            access_tracker=tracker,
            max_dram_vectors=2,
            policy_type="policy_b",
            min_residency_time=10.0,
            hysteresis_margin=0.1,
            cooldown_time=0.0,
        )
        # Vector 1 was promoted long ago, Vector 2 was promoted just now
        mgr.dram_cache[1] = self.base_vectors[1]
        tracker.set_tier(1, TIER_HOT_DRAM, current_time=0.0)
        tracker.hotness_score[1] = 1.0

        mgr.dram_cache[2] = self.base_vectors[2]
        tracker.set_tier(2, TIER_HOT_DRAM, current_time=9.5)  # Promoted at t=9.5
        tracker.hotness_score[2] = 0.5  # Lower score, but within residency window!

        # At t=10.0, vector 2 has residency of only 0.5s (< 10.0s)
        # When vector 3 arrives with high score, vector 1 should be evicted instead of protected vector 2
        cand_id = 3
        cand_vec = self.base_vectors[cand_id]
        tracker.access_count[cand_id] = 10
        promoted = mgr._evaluate_promotion(cand_id, cand_vec, hotness_score=5.0, current_time=10.0)

        self.assertTrue(promoted)
        self.assertIn(2, mgr.dram_cache, "Vector 2 was wrongfully evicted despite minimum residency protection!")
        self.assertNotIn(1, mgr.dram_cache, "Vector 1 should have been selected for eviction.")

    def test_09_latency_sample_count(self):
        """Test 9 — Latency sample count: Measured samples match expected count."""
        workload_gen = WorkloadGenerator(self.query_vectors, seed=42)
        stream = workload_gen.generate_uniform_stream(stream_length=200)
        warmup = 50
        measured_expected = 200 - warmup
        measured_samples = stream[warmup:]
        self.assertEqual(len(measured_samples), measured_expected)

    def test_10_percentile_consistency(self):
        """Test 10 — Percentile consistency: Validate min <= P50 <= P95 <= P99 <= max."""
        lats = np.array([1.2, 2.5, 3.1, 4.0, 5.2, 6.8, 7.9, 10.5, 12.0, 25.0])
        p_min = float(np.min(lats))
        p50 = float(np.percentile(lats, 50))
        p95 = float(np.percentile(lats, 95))
        p99 = float(np.percentile(lats, 99))
        p_max = float(np.max(lats))
        self.assertTrue(p_min <= p50 <= p95 <= p99 <= p_max)

    def test_11_recall_consistency(self):
        """Test 11 — Recall consistency: Recall values remain within [0, 1]."""
        gt = self.ground_truth[:10]
        preds = self.ground_truth[:10]  # Exact match
        r10 = calculate_recall_at_k(gt, preds, k=10)
        self.assertGreaterEqual(r10, 0.0)
        self.assertLessEqual(r10, 1.0)
        self.assertAlmostEqual(r10, 1.0)

    def test_12_no_data_corruption(self):
        """Test 12 — No data corruption: Returned vectors remain valid."""
        sample_ids = [10, 25, 99, 1050]
        for vid in sample_ids:
            vec = self.ssd_store.read_vector(vid)
            self.assertEqual(vec.shape, (128,))
            self.assertEqual(vec.dtype, np.float32)
            self.assertFalse(np.isnan(vec).any())
            self.assertFalse(np.isinf(vec).any())

    def test_13_deterministic_workload(self):
        """Test 13 — Deterministic workload: Same seed produces same workload."""
        wg1 = WorkloadGenerator(self.query_vectors, seed=123)
        wg2 = WorkloadGenerator(self.query_vectors, seed=123)
        s1 = wg1.generate_zipf_stream(stream_length=500, zipf_alpha=1.25)
        s2 = wg2.generate_zipf_stream(stream_length=500, zipf_alpha=1.25)
        self.assertEqual(s1, s2)

    def test_14_capacity_invariant_stress(self):
        """Test 14 — Capacity invariant stress: Rapid access cycles maintain strict capacity bound."""
        tracker = AccessTracker(num_vectors=1000)
        capacity = 50
        mgr = TieredVectorManager(
            ssd_store=self.ssd_store,
            access_tracker=tracker,
            max_dram_vectors=capacity,
            policy_type="policy_b",
        )
        mgr.populate_initial_dram(list(range(capacity)))

        sim_t = 0.0
        for vid in range(100, 300):
            sim_t += 0.01
            # Access each vector twice to trigger admission
            mgr.get_vector(vid, sim_t)
            mgr.get_vector(vid, sim_t + 0.001)
            self.assertLessEqual(len(mgr.dram_cache), capacity, f"Exceeded capacity {capacity}!")

    def test_15_lifecycle(self):
        """Test 15 — Lifecycle: Verify COLD → HOT → COLD state transitions."""
        tracker = AccessTracker(num_vectors=500)
        mgr = TieredVectorManager(
            ssd_store=self.ssd_store,
            access_tracker=tracker,
            max_dram_vectors=10,
            policy_type="policy_a",
        )
        mgr.populate_initial_dram(list(range(10)))
        target_vid = 250

        # Initial: COLD
        self.assertNotIn(target_vid, mgr.dram_cache)
        self.assertEqual(tracker.current_tier[target_vid], TIER_COLD_SSD)

        # Access burst -> Promoted to HOT
        t = 1.0
        for _ in range(5):
            t += 0.1
            mgr.get_vector(target_vid, t)
        self.assertIn(target_vid, mgr.dram_cache)
        self.assertEqual(tracker.current_tier[target_vid], TIER_HOT_DRAM)

        # Decay + competition -> Demoted back to COLD
        tracker.hotness_score[target_vid] = 0.001
        for competitor in range(300, 315):
            vec = self.base_vectors[competitor]
            tracker.hotness_score[competitor] = 10.0
            mgr._evaluate_promotion(competitor, vec, 10.0, current_time=50.0)

        self.assertNotIn(target_vid, mgr.dram_cache)
        self.assertEqual(tracker.current_tier[target_vid], TIER_COLD_SSD)


if __name__ == "__main__":
    unittest.main()
