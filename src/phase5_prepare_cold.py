"""
Case 04 — Phase 5: COLD Tier Dataset Preparation and Ground Truth Generator

Extracts full-precision COLD vectors according to Phase 4 tier manifests,
produces DiskANN3 compatible binary inputs (.bin), creates bidirectional
local <-> global ID mappings, and computes exact COLD-only ground truth
using FAISS IndexFlatL2 as oracle.
"""

from __future__ import annotations

import json
import hashlib
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Ensure workspace root in path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

try:
    import numpy as np
except ImportError as err:
    print(f"CRITICAL ERROR: Failed to import NumPy: {err}", file=sys.stderr)
    sys.exit(1)

try:
    import faiss
except ImportError as err:
    print(f"CRITICAL ERROR: Failed to import FAISS: {err}", file=sys.stderr)
    sys.exit(1)

from src.phase1_2_evaluation import load_fvecs, load_ivecs, compute_sha256, EXPECTED_CHECKSUMS


# ==============================================================================
# Expected Checksums & Invariants
# ==============================================================================
FROZEN_HANDOFF_CHECKSUMS = {
    "data/base_vectors.fvecs": "62a7ad69bc429570fb4819287bb4d317d8069fb784ef528f370c64189d37198b",
    "data/query_vectors.fvecs": "c1d80d61714296078229bc3429a79569113441c01fdd1069de21611a30528496",
    "data/ground_truth.ivecs": "ee2f420a3fea4e71408756f6d0bf7bed1741a69dafc119b2781f7b4142c7d66f",
    "results/phase4_pq/metadata/phase5_handoff_manifest.json": "c0a9e6423504bfa89a44e17b7f94c46e4db1ca78febb7507d909bc5e8de2baa6",
    "results/phase4_pq/metadata/cold_vector_manifest.json": "f1893f346f996875590578c5d9fba11cf0824e0d94e84249f5c0ae1c987f17f8",
    "results/phase4_pq/metadata/tier_manifest.json": "7361fa27c413f8f3457c0f16e108d933da1c4f354fda22282ebfce3966f935fc",
}


# ==============================================================================
# DiskANN Binary Format I/O
# ==============================================================================
def write_diskann_bin(filepath: Path | str, data: np.ndarray, dtype=np.float32) -> int:
    """
    Writes a 2D array to disk in DiskANN binary format:
      - 4 bytes little-endian u32: number of points (nrows)
      - 4 bytes little-endian u32: dimensionality (ncols)
      - nrows * ncols * sizeof(dtype) contiguous payload
    Returns total bytes written.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array for DiskANN bin, got ndim={data.ndim}")
    
    npoints, ndims = data.shape
    arr = np.ascontiguousarray(data, dtype=dtype)
    header = np.array([npoints, ndims], dtype="<u4").tobytes()
    
    with open(filepath, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())
        
    return 8 + arr.nbytes


def read_diskann_bin(filepath: Path | str, dtype=np.float32) -> np.ndarray:
    """
    Reads a 2D array from a DiskANN binary file.
    Header: 8 bytes (npoints: u32, ndims: u32).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"DiskANN binary file not found: {filepath}")
        
    file_size = filepath.stat().st_size
    if file_size < 8:
        raise ValueError(f"File {filepath} too small for DiskANN header: {file_size} bytes")
        
    with open(filepath, "rb") as f:
        header_bytes = f.read(8)
        npoints, ndims = np.frombuffer(header_bytes, dtype="<u4")
        expected_bytes = 8 + int(npoints) * int(ndims) * np.dtype(dtype).itemsize
        if file_size != expected_bytes:
            raise ValueError(
                f"File size mismatch for {filepath}: expected {expected_bytes} bytes "
                f"({npoints} x {ndims} x {np.dtype(dtype).itemsize} + 8), found {file_size} bytes"
            )
        data = np.fromfile(f, dtype=dtype, count=int(npoints) * int(ndims))
        return data.reshape((int(npoints), int(ndims)))


def write_diskann_groundtruth(filepath: Path | str, groundtruth_ids: np.ndarray) -> int:
    """
    Writes groundtruth nearest-neighbor IDs in DiskANN binary format:
      - 4 bytes little-endian u32: number of query points
      - 4 bytes little-endian u32: k (neighbors per query)
      - num_queries * k * 4 bytes: contiguous uint32 IDs
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if groundtruth_ids.ndim != 2:
        raise ValueError(f"Expected 2D array for ground truth, got ndim={groundtruth_ids.ndim}")
        
    num_queries, k = groundtruth_ids.shape
    arr = np.ascontiguousarray(groundtruth_ids, dtype="<u4")
    header = np.array([num_queries, k], dtype="<u4").tobytes()
    
    with open(filepath, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())
        
    return 8 + arr.nbytes


def read_diskann_groundtruth(filepath: Path | str) -> np.ndarray:
    """Reads groundtruth nearest-neighbor IDs from DiskANN binary format."""
    return read_diskann_bin(filepath, dtype=np.uint32)


# ==============================================================================
# Standard IVECS Writer (for evaluation compatibility)
# ==============================================================================
def write_ivecs(filepath: Path | str, data: np.ndarray) -> None:
    """Writes a 2D int32 array in standard .ivecs binary format."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array for ivecs, got ndim={data.ndim}")
    n, d = data.shape
    dt = np.dtype([("dim", "<i4"), ("vec", "<i4", (d,))])
    records = np.empty(n, dtype=dt)
    records["dim"] = d
    records["vec"] = data.astype(np.int32)
    records.tofile(filepath)


# ==============================================================================
# Pipeline Execution
# ==============================================================================
def validate_inputs_and_manifests(root: Path) -> Dict[str, Any]:
    """Validates frozen files against authoritative checksums and manifests."""
    print("[Phase 5 Prepare] Step 1: Validating frozen inputs and manifests...")
    
    manifest_path = root / "results/phase4_pq/metadata/phase5_handoff_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        handoff_data = json.load(f)
        
    cold_manifest_path = root / "results/phase4_pq/metadata/cold_vector_manifest.json"
    with open(cold_manifest_path, "r", encoding="utf-8") as f:
        cold_data = json.load(f)
        
    tier_manifest_path = root / "results/phase4_pq/metadata/tier_manifest.json"
    with open(tier_manifest_path, "r", encoding="utf-8") as f:
        tier_data = json.load(f)
        
    # Checksum validation
    for rel_path, expected_hash in FROZEN_HANDOFF_CHECKSUMS.items():
        abs_p = root / rel_path
        if not abs_p.exists():
            raise FileNotFoundError(f"Missing required frozen file: {abs_p}")
        actual_hash = compute_sha256(abs_p)
        if actual_hash != expected_hash:
            raise ValueError(f"Checksum mismatch for {rel_path}!\nExpected: {expected_hash}\nActual:   {actual_hash}")
        print(f"  [OK] {rel_path} checksum verified.")

    # Manifest invariant verification
    num_cold = cold_data["cold_vector_count"]
    cold_ids = cold_data["cold_vector_ids"]
    if num_cold != 82000:
        raise ValueError(f"Expected 82,000 cold vectors, manifest has {num_cold}")
    if len(cold_ids) != num_cold:
        raise ValueError(f"Cold IDs length mismatch: {len(cold_ids)} vs {num_cold}")
    if cold_ids[0] != 18000 or cold_ids[-1] != 99999:
        raise ValueError(f"Unexpected cold ID boundaries: [{cold_ids[0]}, {cold_ids[-1]}]")
    if len(set(cold_ids)) != num_cold:
        raise ValueError("Cold vector IDs contain duplicates")
        
    print(f"  [OK] Manifest invariants verified: 82,000 COLD vectors (IDs {cold_ids[0]}..{cold_ids[-1]}).")
    return {
        "handoff": handoff_data,
        "cold_manifest": cold_data,
        "tier_manifest": tier_data,
    }


def extract_cold_vectors(root: Path, cold_ids: List[int]) -> np.ndarray:
    """
    Extracts full-precision COLD vectors using both vectors.bin and base_vectors.fvecs
    and confirms byte-level equivalence.
    """
    print("[Phase 5 Prepare] Step 2: Extracting full-precision COLD vectors...")
    
    # Read from base_vectors.fvecs
    base_fvecs_path = root / "data/base_vectors.fvecs"
    all_vectors = load_fvecs(base_fvecs_path)
    if all_vectors.shape != (100000, 128):
        raise ValueError(f"Unexpected base vectors shape: {all_vectors.shape}")
        
    cold_ids_arr = np.array(cold_ids, dtype=np.int64)
    cold_vectors = all_vectors[cold_ids_arr].copy()
    
    # Cross-verify against storage/ssd_vectors/vectors.bin
    ssd_bin_path = root / "storage/ssd_vectors/vectors.bin"
    with open(ssd_bin_path, "rb") as f:
        ssd_all_vectors = np.fromfile(f, dtype=np.float32).reshape(-1, 128)
    ssd_cold_vectors = ssd_all_vectors[cold_ids_arr]
    
    if not np.array_equal(cold_vectors, ssd_cold_vectors):
        raise ValueError("Discrepancy detected between fvecs and ssd_vectors.bin COLD vectors!")
        
    if not np.all(np.isfinite(cold_vectors)):
        raise ValueError("COLD vectors contain NaN or Inf values!")
        
    print(f"  [OK] Extracted {len(cold_vectors):,} COLD vectors of dimension {cold_vectors.shape[1]}.")
    print(f"  [OK] Cross-verification between base_vectors.fvecs and ssd_vectors.bin passed.")
    return cold_vectors


def build_id_mapping(cold_ids: List[int]) -> Dict[str, Any]:
    """Constructs deterministic local <-> global ID mappings."""
    print("[Phase 5 Prepare] Step 3: Constructing local <-> global ID mapping...")
    local_to_global = [int(gid) for gid in cold_ids]
    global_to_local = {int(gid): int(lid) for lid, gid in enumerate(cold_ids)}
    
    # Invariant checks
    assert len(local_to_global) == 82000
    assert len(global_to_local) == 82000
    for lid, gid in enumerate(local_to_global):
        assert global_to_local[gid] == lid
        
    print("  [OK] Mapping constructed: 82,000 reversible entries.")
    return {
        "cold_count": len(cold_ids),
        "min_global_id": min(cold_ids),
        "max_global_id": max(cold_ids),
        "local_to_global": local_to_global,
        "global_to_local": global_to_local,
    }


def compute_cold_ground_truth(
    cold_vectors: np.ndarray,
    query_vectors: np.ndarray,
    local_to_global: List[int],
    top_k: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes exact ground truth over COLD vectors using FAISS IndexFlatL2 (Squared L2).
    Returns (local_gt_ids, global_gt_ids, gt_distances).
    """
    print(f"[Phase 5 Prepare] Step 4: Computing exact COLD ground truth (top-{top_k} nearest neighbors)...")
    dim = cold_vectors.shape[1]
    
    # Initialize FAISS exact oracle (IndexFlatL2 computes squared L2)
    index = faiss.IndexFlatL2(dim)
    index.add(cold_vectors)
    
    start_t = time.perf_counter()
    distances, local_ids = index.search(query_vectors, top_k)
    elapsed = time.perf_counter() - start_t
    
    local_ids = local_ids.astype(np.uint32)
    distances = distances.astype(np.float32)
    
    # Map to global IDs
    local_to_global_arr = np.array(local_to_global, dtype=np.int32)
    global_ids = local_to_global_arr[local_ids]
    
    # Numerical and invariant validation
    if not np.all(np.isfinite(distances)):
        raise ValueError("Ground truth distances contain non-finite values!")
    if np.any(distances < 0):
        raise ValueError("Ground truth distances contain negative values!")
        
    # Check monotonicity of distances per query
    for q_idx in range(len(query_vectors)):
        row = distances[q_idx]
        if not np.all(np.diff(row) >= -1e-5):
            raise ValueError(f"Ground truth distances not monotonic for query {q_idx}")
            
    print(f"  [OK] Exact COLD ground truth computed in {elapsed:.3f}s.")
    print(f"  [OK] Verified monotonicity and non-negativity across all {len(query_vectors)} queries.")
    return local_ids, global_ids, distances


def prepare_all(root: Path | None = None) -> Dict[str, Any]:
    """Executes complete Phase 5 COLD preparation pipeline."""
    if root is None:
        root = workspace_root
        
    start_total = time.perf_counter()
    print("=" * 78)
    print("PHASE 5: COLD TIER PREPARATION PIPELINE")
    print("=" * 78)
    
    # Step 1: Validation
    manifest_info = validate_inputs_and_manifests(root)
    cold_ids = manifest_info["cold_manifest"]["cold_vector_ids"]
    
    # Step 2: Extract COLD vectors
    cold_vectors = extract_cold_vectors(root, cold_ids)
    
    # Step 3: Load queries
    query_vectors = load_fvecs(root / "data/query_vectors.fvecs")
    if query_vectors.shape != (1000, 128):
        raise ValueError(f"Unexpected query shape: {query_vectors.shape}")
        
    # Step 4: Build mapping
    mapping = build_id_mapping(cold_ids)
    
    # Step 5: Compute COLD ground truth
    local_gt_ids, global_gt_ids, gt_dists = compute_cold_ground_truth(
        cold_vectors, query_vectors, mapping["local_to_global"], top_k=100
    )
    
    # Step 6: Write prepared files
    print("[Phase 5 Prepare] Step 5: Writing DiskANN inputs and metadata...")
    prep_dir = root / "results/phase5_diskann/prepared"
    meta_dir = root / "results/phase5_diskann/metadata"
    prep_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    cold_bin = prep_dir / "cold_vectors.bin"
    query_bin = prep_dir / "query_vectors.bin"
    cold_gt_local_bin = prep_dir / "cold_groundtruth_local.bin"
    cold_gt_global_ivecs = prep_dir / "cold_groundtruth_global.ivecs"
    cold_gt_dists_npy = prep_dir / "cold_groundtruth_distances.npy"
    id_map_path = meta_dir / "id_mapping.json"
    report_path = meta_dir / "preparation_report.json"
    
    bytes_cold = write_diskann_bin(cold_bin, cold_vectors)
    bytes_query = write_diskann_bin(query_bin, query_vectors)
    bytes_gt_local = write_diskann_groundtruth(cold_gt_local_bin, local_gt_ids)
    write_ivecs(cold_gt_global_ivecs, global_gt_ids)
    np.save(cold_gt_dists_npy, gt_dists)
    
    # Save ID mapping JSON
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump({
            "cold_vector_count": len(cold_ids),
            "first_cold_id": cold_ids[0],
            "last_cold_id": cold_ids[-1],
            "local_to_global": mapping["local_to_global"],
        }, f, indent=2)
        
    # Cross-check round-trip reading
    print("[Phase 5 Prepare] Step 6: Verifying round-trip readability of prepared files...")
    rt_cold = read_diskann_bin(cold_bin)
    assert np.array_equal(rt_cold, cold_vectors), "Round-trip cold vector mismatch!"
    
    rt_query = read_diskann_bin(query_bin)
    assert np.array_equal(rt_query, query_vectors), "Round-trip query vector mismatch!"
    
    rt_gt_local = read_diskann_groundtruth(cold_gt_local_bin)
    assert np.array_equal(rt_gt_local, local_gt_ids), "Round-trip ground truth mismatch!"
    print("  [OK] All binary files verified with round-trip bitwise exact matches.")
    
    # Calculate output hashes
    output_hashes = {
        "cold_vectors_bin": compute_sha256(cold_bin),
        "query_vectors_bin": compute_sha256(query_bin),
        "cold_groundtruth_local_bin": compute_sha256(cold_gt_local_bin),
        "cold_groundtruth_global_ivecs": compute_sha256(cold_gt_global_ivecs),
        "cold_groundtruth_distances_npy": compute_sha256(cold_gt_dists_npy),
        "id_mapping_json": compute_sha256(id_map_path),
    }
    
    total_time = time.perf_counter() - start_total
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "PASS",
        "duration_seconds": round(total_time, 3),
        "counts": {
            "cold_vectors": int(len(cold_vectors)),
            "query_vectors": int(len(query_vectors)),
            "dimension": int(cold_vectors.shape[1]),
            "ground_truth_top_k": int(local_gt_ids.shape[1]),
        },
        "file_sizes": {
            "cold_vectors_bin_bytes": bytes_cold,
            "query_vectors_bin_bytes": bytes_query,
            "cold_groundtruth_local_bin_bytes": bytes_gt_local,
            "cold_groundtruth_global_ivecs_bytes": cold_gt_global_ivecs.stat().st_size,
        },
        "hashes": output_hashes,
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"\n[Phase 5 Prepare] Complete! Summary saved to {report_path.relative_to(root)}")
    print(f"Total time: {total_time:.2f}s")
    print("=" * 78)
    return report


if __name__ == "__main__":
    prepare_all()
