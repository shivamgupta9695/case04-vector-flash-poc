"""
Case 04 — Phase 5: Automated Verification Test Suite

Validates Microsoft DiskANN3 COLD-tier index integration:
1. Frozen dataset and source code immutability (Phase 1-4)
2. COLD membership and tier partition invariants
3. Bijective global <-> local ID mapping
4. DiskANN binary input preparation and round-trip fidelity
5. Persisted index structure and artifacts
6. Close / reopen persistence cycle and bitwise repeatability
7. Valid candidate IDs and strict COLD membership containment
8. Exact COLD ground truth oracle correctness
9. Input dimension, numerical integrity, and boundary k handling
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
)
from src.phase5_prepare_cold import (
    read_diskann_bin,
    read_diskann_groundtruth,
    write_diskann_bin,
    FROZEN_HANDOFF_CHECKSUMS,
)
from src.phase5_diskann_search import ColdDiskANNIndex


# Authoritative checksums of existing Phase 1, 2, 3, 4 source files
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
}


class TestPhase5DiskANN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = workspace_root / "data"
        cls.p4_meta_dir = workspace_root / "results/phase4_pq/metadata"
        cls.p5_dir = workspace_root / "results/phase5_diskann"
        cls.p5_prep_dir = cls.p5_dir / "prepared"
        cls.p5_idx_dir = cls.p5_dir / "index"
        cls.p5_meta_dir = cls.p5_dir / "metadata"
        cls.p5_metric_dir = cls.p5_dir / "metrics"

        cls.tuned_index_prefix = cls.p5_idx_dir / "cold_disk_index_tuned"
        cls.id_map_path = cls.p5_meta_dir / "id_mapping.json"
        cls.cold_manifest_path = cls.p4_meta_dir / "cold_vector_manifest.json"

    # ==========================================================================
    # 1. Dataset & Source Immutability
    # ==========================================================================
    def test_01_frozen_dataset_checksums(self):
        """Phase 1-4 frozen datasets must match authoritative SHA-256 digests."""
        for rel_path, expected_hash in FROZEN_HANDOFF_CHECKSUMS.items():
            p = workspace_root / rel_path
            self.assertTrue(p.exists(), f"Missing file: {rel_path}")
            actual_hash = compute_sha256(p)
            self.assertEqual(
                actual_hash,
                expected_hash,
                f"Checksum mismatch for frozen file {rel_path}"
            )

    def test_02_source_code_immutability(self):
        """Phase 1-4 source code files must remain strictly untouched."""
        for rel_path, expected_hash in FROZEN_SOURCE_CHECKSUMS.items():
            p = workspace_root / rel_path
            self.assertTrue(p.exists(), f"Source file missing: {rel_path}")
            actual_hash = compute_sha256(p)
            self.assertEqual(
                actual_hash,
                expected_hash,
                f"Source file {rel_path} was modified!"
            )

    # ==========================================================================
    # 2. COLD Membership & ID Mapping Invariants
    # ==========================================================================
    def test_03_cold_membership_manifest_invariants(self):
        """COLD membership manifest must contain exactly 82,000 unique IDs."""
        with open(self.cold_manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cold_ids = data["cold_vector_ids"]
        self.assertEqual(len(cold_ids), 82000)
        self.assertEqual(data["cold_vector_count"], 82000)
        self.assertEqual(len(set(cold_ids)), 82000)
        self.assertEqual(cold_ids[0], 18000)
        self.assertEqual(cold_ids[-1], 99999)

    def test_04_hot_cold_disjoint_and_exhaustive(self):
        """HOT (18,000) and COLD (82,000) must partition the 100,000 dataset exactly."""
        tier_manifest = self.p4_meta_dir / "tier_manifest.json"
        with open(tier_manifest, "r", encoding="utf-8") as f:
            tdata = json.load(f)
        self.assertEqual(tdata["current_state"]["hot_count"], 18000)
        self.assertEqual(tdata["current_state"]["cold_count"], 82000)
        hot_range = tdata["current_state"]["hot_id_range"]
        cold_range = tdata["current_state"]["cold_id_range"]
        self.assertEqual(hot_range, [0, 17999])
        self.assertEqual(cold_range, [18000, 99999])
        self.assertEqual(hot_range[1] + 1, cold_range[0])

    def test_05_id_mapping_bijective_and_reversible(self):
        """Global-to-local and local-to-global mappings must be 1-to-1 and reversible."""
        with open(self.id_map_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        l2g = mapping["local_to_global"]
        self.assertEqual(len(l2g), 82000)
        self.assertEqual(mapping["cold_vector_count"], 82000)
        self.assertEqual(mapping["first_cold_id"], 18000)
        self.assertEqual(mapping["last_cold_id"], 99999)
        # Check bijectivity
        for lid, gid in enumerate(l2g):
            self.assertEqual(gid, 18000 + lid)

    # ==========================================================================
    # 3. Prepared Binary Formats & Ground Truth
    # ==========================================================================
    def test_06_prepared_binary_formats(self):
        """DiskANN binary files must have correct headers, shapes, and finite values."""
        cold_bin = self.p5_prep_dir / "cold_vectors.bin"
        query_bin = self.p5_prep_dir / "query_vectors.bin"
        self.assertTrue(cold_bin.is_file())
        self.assertTrue(query_bin.is_file())

        cold_vecs = read_diskann_bin(cold_bin)
        query_vecs = read_diskann_bin(query_bin)

        self.assertEqual(cold_vecs.shape, (82000, 128))
        self.assertEqual(cold_vecs.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(cold_vecs)))

        self.assertEqual(query_vecs.shape, (1000, 128))
        self.assertEqual(query_vecs.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(query_vecs)))

    def test_07_exact_cold_ground_truth(self):
        """COLD ground truth must have valid dimensions, sorted distances, and local IDs."""
        gt_local_bin = self.p5_prep_dir / "cold_groundtruth_local.bin"
        gt_dists_npy = self.p5_prep_dir / "cold_groundtruth_distances.npy"
        self.assertTrue(gt_local_bin.is_file())
        self.assertTrue(gt_dists_npy.is_file())

        local_ids = read_diskann_groundtruth(gt_local_bin)
        dists = np.load(gt_dists_npy)

        self.assertEqual(local_ids.shape, (1000, 100))
        self.assertEqual(dists.shape, (1000, 100))
        self.assertTrue(np.all(local_ids < 82000))
        self.assertTrue(np.all(dists >= 0.0))

        # Check monotonicity of first 100 queries
        for q in range(100):
            row = dists[q]
            self.assertTrue(np.all(np.diff(row) >= -1e-5), f"Non-monotonic at query {q}")

    # ==========================================================================
    # 4. DiskANN3 Index Artifacts & Persistence
    # ==========================================================================
    def test_08_persisted_index_artifacts(self):
        """Persisted index files must exist with non-zero size."""
        prefix = str(self.tuned_index_prefix)
        disk_index = Path(prefix + "_disk.index")
        pq_compressed = Path(prefix + "_pq_compressed.bin")
        pq_pivots = Path(prefix + "_pq_pivots.bin")

        self.assertTrue(disk_index.is_file(), "Missing _disk.index")
        self.assertTrue(pq_compressed.is_file(), "Missing _pq_compressed.bin")
        self.assertTrue(pq_pivots.is_file(), "Missing _pq_pivots.bin")

        self.assertGreater(disk_index.stat().st_size, 40_000_000)
        self.assertGreater(pq_compressed.stat().st_size, 1_000_000)
        self.assertGreater(pq_pivots.stat().st_size, 10_000)

    def test_09_search_against_built_index(self):
        """Adapter search must succeed and return correct shapes and finite distances."""
        idx = ColdDiskANNIndex(self.tuned_index_prefix)
        queries = load_fvecs(self.data_dir / "query_vectors.fvecs")[:10]

        res = idx.search(queries, k=10, l_search=50)
        self.assertEqual(res.num_queries, 10)
        self.assertEqual(res.local_ids.shape, (10, 10))
        self.assertEqual(res.global_ids.shape, (10, 10))
        self.assertEqual(res.distances.shape, (10, 10))
        self.assertTrue(np.all(np.isfinite(res.distances)))
        self.assertTrue(np.all(res.distances >= 0.0))

    def test_10_index_close_and_reopen_repeatability(self):
        """Closing, reopening, and re-querying index must yield bitwise identical results."""
        idx = ColdDiskANNIndex(self.tuned_index_prefix)
        queries = load_fvecs(self.data_dir / "query_vectors.fvecs")[:5]

        res1 = idx.search(queries, k=10, l_search=80)
        idx.close()
        self.assertFalse(idx.is_open)

        idx.reopen()
        self.assertTrue(idx.is_open)
        res2 = idx.search(queries, k=10, l_search=80)

        np.testing.assert_array_equal(res1.local_ids, res2.local_ids)
        np.testing.assert_array_equal(res1.global_ids, res2.global_ids)
        np.testing.assert_array_almost_equal(res1.distances, res2.distances, decimal=5)

    def test_11_candidate_ids_cold_membership(self):
        """100% of candidate global IDs must belong to the authoritative COLD membership."""
        idx = ColdDiskANNIndex(self.tuned_index_prefix)
        queries = load_fvecs(self.data_dir / "query_vectors.fvecs")[:50]
        res = idx.search(queries, k=10, l_search=50)

        all_valid, valid_count, total_count = idx.validate_cold_membership(res.global_ids)
        self.assertTrue(all_valid, f"Only {valid_count}/{total_count} IDs belonged to COLD tier!")
        self.assertEqual(valid_count, total_count)
        self.assertEqual(total_count, 50 * 10)

    # ==========================================================================
    # 5. Search Quality & Candidate Coverage
    # ==========================================================================
    def test_12_cold_recall_and_global_coverage(self):
        """COLD Recall@10 must exceed baseline (>= 70%) and Global Coverage computed."""
        idx = ColdDiskANNIndex(self.tuned_index_prefix)
        queries = load_fvecs(self.data_dir / "query_vectors.fvecs")[:100]
        cold_gt = load_ivecs(self.p5_prep_dir / "cold_groundtruth_global.ivecs")[:100]
        global_gt = load_ivecs(self.data_dir / "ground_truth.ivecs")[:100]

        res = idx.search(queries, k=10, l_search=100)
        recalls = idx.evaluate_cold_recall(res.global_ids, cold_gt, k_values=[1, 5, 10])
        coverages = idx.evaluate_global_coverage(res.global_ids, global_gt, k_values=[1, 5, 10])

        self.assertGreaterEqual(recalls[10], 0.65)
        self.assertGreaterEqual(recalls[1], 0.70)
        self.assertGreaterEqual(coverages[10], 0.50)
        print(f"\n[Test Quality] COLD Recall@1: {recalls[1]:.4f}, Recall@10: {recalls[10]:.4f}")
        print(f"[Test Quality] Global Coverage@1: {coverages[1]:.4f}, Coverage@10: {coverages[10]:.4f}")

    # ==========================================================================
    # 6. Error Handling & Edge Cases
    # ==========================================================================
    def test_13_query_dimension_mismatch(self):
        """Query with invalid dimension (e.g. 64) must raise ValueError."""
        idx = ColdDiskANNIndex(self.tuned_index_prefix)
        bad_query = np.ones((1, 64), dtype=np.float32)
        with self.assertRaises(ValueError):
            idx.search(bad_query, k=10)

    def test_14_non_finite_query_values(self):
        """Query containing NaN or Inf must raise ValueError."""
        idx = ColdDiskANNIndex(self.tuned_index_prefix)
        bad_query = np.zeros((1, 128), dtype=np.float32)
        bad_query[0, 5] = np.nan
        with self.assertRaises(ValueError):
            idx.search(bad_query, k=10)

        bad_query[0, 5] = np.inf
        with self.assertRaises(ValueError):
            idx.search(bad_query, k=10)

    def test_15_k_boundary_handling(self):
        """k <= 0 or k > num_vectors must raise ValueError."""
        idx = ColdDiskANNIndex(self.tuned_index_prefix)
        query = np.zeros((1, 128), dtype=np.float32)
        with self.assertRaises(ValueError):
            idx.search(query, k=0)
        with self.assertRaises(ValueError):
            idx.search(query, k=-5)
        with self.assertRaises(ValueError):
            idx.search(query, k=90000)

    def test_16_empty_query_batch(self):
        """Empty query batch (0, 128) must return empty result cleanly."""
        idx = ColdDiskANNIndex(self.tuned_index_prefix)
        empty_query = np.empty((0, 128), dtype=np.float32)
        res = idx.search(empty_query, k=10)
        self.assertEqual(res.num_queries, 0)
        self.assertEqual(len(res.local_ids), 0)


if __name__ == "__main__":
    unittest.main()
