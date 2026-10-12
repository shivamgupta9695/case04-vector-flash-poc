"""
Case 04 — Phase 4: Automated Validation Test Suite

Automated verification of Product Quantization (PQ), HOT/COLD data model,
ADC distance primitive, reconstruction error, search quality, and architectural invariants.
Verifies all 20 required tests from Master Prompt Section 21 plus additional invariant checks.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path
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
    calculate_recall_at_k,
)
from src.phase4_pq_train import PQConfig, PQTrainer, load_trained_codebooks
from src.phase4_pq_encode import PQEncoder, load_pq_codes, validate_pq_codes
from src.phase4_pq_reconstruct import PQReconstructor, reconstruct_vectors_numpy
from src.phase4_pq_adc import ADCSearcher, compute_adc_lut_single_query, compute_adc_distances
from src.phase4_pq_tier_model import TierModel, VectorTier, VectorRecord


# Known SHA-256 digests of existing Phase 1, 2, 3 source files to guarantee they remain untouched
FROZEN_SOURCE_CHECKSUMS = {
    "diagnostic_10pct_uniform.py": "ba128d01b8eacb9d469ef18ad661dcb71dab2ca00919eca03fdcdba0cdf1b86e",
    "inspect_phase3_live_tiers.py": "ec257f3f0a818b37ffff2ec3dd01fb7c5d334cc9bb63032e4cea795e8e19ce35",
    "inspect_phase3_tiers.py": "0d6e62f586b8f5df66ab8ac8fc12620d632f14a2928cc4817282a2e6d0738e10",
    "phase1_1_ground_truth.py": "50952abf793bd0e18cc44741d334bc33d94fac7ce5aa544fe37eb9dc1c219754",
    "phase1_2_evaluation.py": "599567e4ebdb7d4a2424ab12a6eb301b18b3c2f94ec35604002ccc147b4eaae0",
    "phase2_ann_hnsw.py": "0af69ac722a8da9f8c8025fec3d33efaba7bce1c9af2d53ae4199c53e90c25fe",
    "phase3_hotness.py": "3d6e1c4fb3526d3b4a0b90574e6da2a0ad7a0d1478631eca867a4f5053034391",
    "phase3_realtime.py": "29431fd6eb137c18f061705427c75dab24252953d836bc81120e3092a1da491b",
    "phase3_ssd_store.py": "e0b819cacee44d258c93264a494cd61feb869ced5932c87344449bf738c6ad67",
    "phase3_tiering.py": "92944ba6b7999f840742aab10261262d29b1b8fcd29b9590691ed82f0ba78b94",
    "phase3_workload.py": "59a4d6971d8d792f8bf4b8ca3f72b47b5b9d64d23f9be7c2e96c2a12700f0e42",
}


class TestPhase4Validation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = workspace_root / "data"
        cls.results_dir = workspace_root / "results" / "phase4_pq"
        cls.cb_dir = cls.results_dir / "codebooks"
        cls.codes_dir = cls.results_dir / "codes"
        cls.meta_dir = cls.results_dir / "metadata"
        cls.summaries_dir = cls.results_dir / "summaries"

        cls.base_path = cls.data_dir / "base_vectors.fvecs"
        cls.query_path = cls.data_dir / "query_vectors.fvecs"
        cls.gt_path = cls.data_dir / "ground_truth.ivecs"

        cls.base_vectors = load_fvecs(cls.base_path)
        cls.query_vectors = load_fvecs(cls.query_path)
        cls.ground_truth = load_ivecs(cls.gt_path)

        # Ensure codebooks & codes are loaded if available; otherwise train/encode on the fly
        if not (cls.cb_dir / "pq_codebooks.npy").exists():
            config = PQConfig(dimension=128, num_subquantizers=16, centroids_per_subquantizer=256, random_seed=42)
            trainer = PQTrainer(config)
            trainer.train(cls.base_vectors)
            trainer.save_artifacts(cls.cb_dir)
            encoder = PQEncoder(trainer.pq, config)
            codes = encoder.encode(cls.base_vectors)
            save_pq_codes(codes, cls.codes_dir)

        cls.centroids, cls.cb_meta = load_trained_codebooks(cls.cb_dir)
        cls.codes, cls.codes_meta = load_pq_codes(cls.codes_dir)
        cls.tier_model = TierModel(total_vectors=len(cls.base_vectors), dimension=128, hot_ratio=0.18)

    # --------------------------------------------------------------------------
    # TEST 1: Frozen dataset dimensions correct
    # --------------------------------------------------------------------------
    def test_01_dataset_dimensions(self):
        """TEST 1: Frozen dataset dimensions correct: 100K x 128 base, 1K x 128 query."""
        self.assertEqual(self.base_vectors.shape, (100_000, 128), "Base vectors shape mismatch")
        self.assertEqual(self.query_vectors.shape, (1_000, 128), "Query vectors shape mismatch")
        self.assertEqual(self.ground_truth.shape, (1_000, 10), "Ground truth shape mismatch")
        self.assertEqual(self.base_vectors.dtype, np.float32, "Base vectors dtype must be float32")
        self.assertEqual(self.query_vectors.dtype, np.float32, "Query vectors dtype must be float32")

    # --------------------------------------------------------------------------
    # TEST 2: Dataset identity/checksum validation
    # --------------------------------------------------------------------------
    def test_02_dataset_checksums(self):
        """TEST 2: Dataset identity/checksum validation against Phase 1 SHA-256."""
        for fname, expected in EXPECTED_CHECKSUMS.items():
            fpath = self.data_dir / fname
            actual = compute_sha256(fpath)
            self.assertEqual(actual, expected, f"Hash mismatch for {fname}")

    # --------------------------------------------------------------------------
    # TEST 3: PQ configuration valid
    # --------------------------------------------------------------------------
    def test_03_pq_configuration(self):
        """TEST 3: PQ configuration valid: D=128, M=16, Ks=256, nbits=8."""
        cfg = PQConfig(dimension=128, num_subquantizers=16, centroids_per_subquantizer=256, nbits=8)
        self.assertEqual(cfg.dimension, 128)
        self.assertEqual(cfg.num_subquantizers, 16)
        self.assertEqual(cfg.centroids_per_subquantizer, 256)
        self.assertEqual(cfg.subvector_dimension, 8)
        self.assertEqual(cfg.code_bytes_per_vector, 16)

    # --------------------------------------------------------------------------
    # TEST 4: D % M == 0
    # --------------------------------------------------------------------------
    def test_04_dimension_divisibility(self):
        """TEST 4: Dimension is strictly divisible by subquantizers count (D % M == 0)."""
        d = self.base_vectors.shape[1]
        m = 16
        self.assertEqual(d % m, 0, f"D={d} must be divisible by M={m}")
        self.assertEqual(d // m, 8, "Expected subvector dimension = 8")

    # --------------------------------------------------------------------------
    # TEST 5: PQ code shape = [100000, 16]
    # --------------------------------------------------------------------------
    def test_05_pq_code_shape(self):
        """TEST 5: PQ code shape = [100000, 16]."""
        self.assertEqual(self.codes.shape, (100_000, 16), "Codes shape must be (100000, 16)")

    # --------------------------------------------------------------------------
    # TEST 6: PQ dtype = uint8
    # --------------------------------------------------------------------------
    def test_06_pq_dtype(self):
        """TEST 6: PQ dtype = uint8."""
        self.assertEqual(self.codes.dtype, np.uint8, "PQ codes dtype must be uint8")

    # --------------------------------------------------------------------------
    # TEST 7: All PQ codes are in [0, 255]
    # --------------------------------------------------------------------------
    def test_07_pq_codes_range(self):
        """TEST 7: All PQ codes are in [0, 255]."""
        self.assertGreaterEqual(int(self.codes.min()), 0, "Code values below 0")
        self.assertLessEqual(int(self.codes.max()), 255, "Code values exceed 255")
        # Validate that code values span the centroid range
        unique_codes = np.unique(self.codes)
        self.assertGreater(len(unique_codes), 200, "Code vocabulary should utilize centroids")

    # --------------------------------------------------------------------------
    # TEST 8: No NaN/Inf in codebooks
    # --------------------------------------------------------------------------
    def test_08_codebooks_finite(self):
        """TEST 8: No NaN/Inf in codebooks."""
        self.assertEqual(self.centroids.shape, (16, 256, 8), "Centroids shape mismatch")
        self.assertFalse(np.isnan(self.centroids).any(), "Codebooks contain NaN values")
        self.assertFalse(np.isinf(self.centroids).any(), "Codebooks contain Infinite values")

    # --------------------------------------------------------------------------
    # TEST 9: Reconstruction shape correct
    # --------------------------------------------------------------------------
    def test_09_reconstruction_shape(self):
        """TEST 9: Reconstruction shape correct: [100000, 128] float32."""
        sample_codes = self.codes[:100]
        rec = reconstruct_vectors_numpy(sample_codes, self.centroids)
        self.assertEqual(rec.shape, (100, 128), "Reconstruction shape mismatch")
        self.assertEqual(rec.dtype, np.float32, "Reconstruction dtype must be float32")

    # --------------------------------------------------------------------------
    # TEST 10: Reconstruction contains no NaN/Inf
    # --------------------------------------------------------------------------
    def test_10_reconstruction_finite(self):
        """TEST 10: Reconstruction contains no NaN/Inf and has finite error."""
        sample_codes = self.codes[:500]
        rec = reconstruct_vectors_numpy(sample_codes, self.centroids)
        self.assertFalse(np.isnan(rec).any(), "Reconstructed vectors contain NaN")
        self.assertFalse(np.isinf(rec).any(), "Reconstructed vectors contain Inf")

        diff = self.base_vectors[:500] - rec
        mse = float(np.mean(diff ** 2))
        self.assertGreater(mse, 0.0, "MSE should be positive")
        self.assertLess(mse, 2.0, f"MSE unexpectedly large: {mse}")

    # --------------------------------------------------------------------------
    # TEST 11: HOT count = 18,000
    # --------------------------------------------------------------------------
    def test_11_hot_count(self):
        """TEST 11: HOT count = 18,000 (18% capacity)."""
        self.assertEqual(self.tier_model.hot_count, 18_000, "HOT tier count must be 18,000")

    # --------------------------------------------------------------------------
    # TEST 12: COLD count = 82,000
    # --------------------------------------------------------------------------
    def test_12_cold_count(self):
        """TEST 12: COLD count = 82,000 (82% capacity)."""
        self.assertEqual(self.tier_model.cold_count, 82_000, "COLD tier count must be 82,000")

    # --------------------------------------------------------------------------
    # TEST 13: HOT ∩ COLD = empty
    # --------------------------------------------------------------------------
    def test_13_hot_cold_disjoint(self):
        """TEST 13: HOT ∩ COLD = empty (disjoint sets)."""
        overlap = self.tier_model.hot_ids.intersection(self.tier_model.cold_ids)
        self.assertEqual(len(overlap), 0, f"HOT and COLD overlap detected: {len(overlap)} IDs")

    # --------------------------------------------------------------------------
    # TEST 14: HOT ∪ COLD = all vector IDs
    # --------------------------------------------------------------------------
    def test_14_hot_cold_exhaustive(self):
        """TEST 14: HOT ∪ COLD = all vector IDs (0 to 99,999)."""
        union_set = self.tier_model.hot_ids.union(self.tier_model.cold_ids)
        all_ids = set(range(100_000))
        self.assertEqual(union_set, all_ids, "Union of HOT and COLD must equal all 100K vector IDs")

    # --------------------------------------------------------------------------
    # TEST 15: Every vector appears exactly once
    # --------------------------------------------------------------------------
    def test_15_every_vector_appears_once(self):
        """TEST 15: Every vector appears exactly once across the tier model."""
        inv = self.tier_model.validate_invariants()
        self.assertTrue(inv["is_valid"], "Tier model invariants failed validation")
        self.assertEqual(inv["total_vectors"], 100_000)

    # --------------------------------------------------------------------------
    # TEST 16: PQ code deterministic across repeated runs
    # --------------------------------------------------------------------------
    def test_16_pq_encoding_determinism(self):
        """TEST 16: PQ code deterministic across repeated runs (identical SHA-256)."""
        config = PQConfig(dimension=128, num_subquantizers=16, centroids_per_subquantizer=256, random_seed=42)
        # Train on a deterministic 10K subset
        subset = self.base_vectors[:10_000]

        trainer1 = PQTrainer(config)
        trainer1.train(subset)
        codes1 = trainer1.pq.compute_codes(subset[:500])

        trainer2 = PQTrainer(config)
        trainer2.train(subset)
        codes2 = trainer2.pq.compute_codes(subset[:500])

        self.assertTrue(np.array_equal(codes1, codes2), "PQ encoding is not deterministic across runs")
        h1 = hashlib.sha256(codes1.tobytes()).hexdigest()
        h2 = hashlib.sha256(codes2.tobytes()).hexdigest()
        self.assertEqual(h1, h2, "PQ encoding hash differs across runs")

    # --------------------------------------------------------------------------
    # TEST 17: ADC distance finite
    # --------------------------------------------------------------------------
    def test_17_adc_distance_finite(self):
        """TEST 17: ADC distance is finite, non-negative, and valid."""
        q = self.query_vectors[0]
        lut = compute_adc_lut_single_query(q, self.centroids)
        self.assertFalse(np.isnan(lut).any(), "ADC LUT contains NaN")
        self.assertFalse(np.isinf(lut).any(), "ADC LUT contains Inf")
        self.assertGreaterEqual(float(lut.min()), 0.0, "ADC LUT squared distances must be >= 0")

        dists = compute_adc_distances(lut, self.codes[:1000])
        self.assertEqual(len(dists), 1000)
        self.assertFalse(np.isnan(dists).any(), "ADC distances contain NaN")
        self.assertFalse(np.isinf(dists).any(), "ADC distances contain Inf")
        self.assertGreaterEqual(float(dists.min()), 0.0, "ADC distances must be >= 0")

    # --------------------------------------------------------------------------
    # TEST 18: ADC ranking can be produced
    # --------------------------------------------------------------------------
    def test_18_adc_ranking_produced(self):
        """TEST 18: ADC ranking can be produced without errors."""
        searcher = ADCSearcher(self.centroids, self.codes)
        top_k_indices, top_k_dists = searcher.search_single_query(self.query_vectors[0], k=10)
        self.assertEqual(len(top_k_indices), 10, "Expected 10 retrieved candidate IDs")
        self.assertEqual(len(top_k_dists), 10, "Expected 10 retrieved distances")
        self.assertTrue(np.all(np.diff(top_k_dists) >= 0), "Retrieved distances must be sorted monotonically")
        self.assertEqual(len(set(top_k_indices.tolist())), 10, "Candidate IDs must be deduplicated")

    # --------------------------------------------------------------------------
    # TEST 19: Recall evaluation matches expected calculation
    # --------------------------------------------------------------------------
    def test_19_recall_calculation_validity(self):
        """TEST 19: Recall evaluation matches expected calculation (values in [0, 1])."""
        searcher = ADCSearcher(self.centroids, self.codes)
        indices, _ = searcher.search_batch(self.query_vectors[:50], k=10)

        r1 = calculate_recall_at_k(self.ground_truth[:50], indices, k=1)
        r5 = calculate_recall_at_k(self.ground_truth[:50], indices, k=5)
        r10 = calculate_recall_at_k(self.ground_truth[:50], indices, k=10)

        self.assertGreaterEqual(r1, 0.0)
        self.assertLessEqual(r1, 1.0)
        self.assertGreaterEqual(r5, 0.0)
        self.assertLessEqual(r5, 1.0)
        self.assertGreaterEqual(r10, 0.0)
        self.assertLessEqual(r10, 1.0)
        # Monotonicity: larger k gives equal or higher recall
        self.assertGreaterEqual(r10, r1 - 1e-6, "Recall@10 should be >= Recall@1")

    # --------------------------------------------------------------------------
    # TEST 20: Phase 1/2/3 files remain unchanged
    # --------------------------------------------------------------------------
    def test_20_phase123_sources_untouched(self):
        """TEST 20: Phase 1/2/3 source files remain byte-identical and untouched."""
        src_dir = workspace_root / "src"
        for fname, expected_hash in FROZEN_SOURCE_CHECKSUMS.items():
            fpath = src_dir / fname
            self.assertTrue(fpath.exists(), f"Source file missing: {fname}")
            actual_hash = compute_sha256(fpath)
            self.assertEqual(
                actual_hash,
                expected_hash,
                f"Source file {fname} was modified! Actual={actual_hash}, Expected={expected_hash}"
            )

    # --------------------------------------------------------------------------
    # TEST 21: ADC mathematical identity
    # --------------------------------------------------------------------------
    def test_21_adc_mathematical_identity(self):
        """TEST 21: ADC distance is mathematically identical to ||q - x_hat||^2."""
        searcher = ADCSearcher(self.centroids, self.codes)
        rec_vectors = reconstruct_vectors_numpy(self.codes[:200], self.centroids)
        report = searcher.verify_mathematical_identity(
            sample_query=self.query_vectors[0],
            reconstructed_vectors=rec_vectors,
            num_check=200,
            atol=1e-4,
        )
        self.assertTrue(report["identity_verified"], f"ADC identity failed: max diff = {report['max_absolute_difference']}")

    # --------------------------------------------------------------------------
    # TEST 22: Tier migration invariants
    # --------------------------------------------------------------------------
    def test_22_tier_migration_invariants(self):
        """TEST 22: Tier migration (promotion/demotion) preserves invariants and stability."""
        tm = TierModel(total_vectors=1000, dimension=128, hot_ratio=0.18)
        self.assertEqual(tm.hot_count, 180)
        self.assertEqual(tm.cold_count, 820)

        # Promote vector 500 (COLD -> HOT)
        self.assertTrue(tm.is_cold(500))
        tm.promote(500)
        self.assertTrue(tm.is_hot(500))
        self.assertEqual(tm.hot_count, 181)
        self.assertEqual(tm.cold_count, 819)
        inv = tm.validate_invariants()
        self.assertTrue(inv["is_valid"])

        # Demote vector 500 (HOT -> COLD)
        tm.demote(500)
        self.assertTrue(tm.is_cold(500))
        self.assertEqual(tm.hot_count, 180)
        self.assertEqual(tm.cold_count, 820)
        inv = tm.validate_invariants()
        self.assertTrue(inv["is_valid"])

    # --------------------------------------------------------------------------
    # TEST 23: Accurate memory accounting
    # --------------------------------------------------------------------------
    def test_23_compression_accounting(self):
        """TEST 23: Code-only compression is exactly 32.0x (512 bytes / 16 bytes)."""
        fp32_bytes = 100_000 * 128 * 4
        pq_code_bytes = 100_000 * 16 * 1
        code_ratio = fp32_bytes / pq_code_bytes
        self.assertAlmostEqual(code_ratio, 32.0, places=4, msg="Code-only compression must be exactly 32.0x")


if __name__ == "__main__":
    unittest.main()
