"""
Case 04 — Phase 4: Artifact and Representation Inspector

Provides CLI inspection of Phase 4 artifacts:
    - PQ codebooks (centroids shape, distribution, statistics)
    - PQ codes (uint8 byte codes, sample vector IDs)
    - HOT/COLD tier model (membership, capacity invariants, storage mapping)
    - Reconstruction fidelity (original vs reconstructed comparison)
    - ADC distance verification (LUT inspection, exact vs ADC comparison)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure workspace root in path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

try:
    import numpy as np
except ImportError as err:
    print(f"CRITICAL ERROR: Failed to import NumPy: {err}", file=sys.stderr)
    sys.exit(1)

from src.phase1_2_evaluation import load_fvecs, load_ivecs
from src.phase4_pq_train import load_trained_codebooks
from src.phase4_pq_encode import load_pq_codes
from src.phase4_pq_reconstruct import reconstruct_vectors_numpy
from src.phase4_pq_adc import compute_adc_lut_single_query, compute_adc_distances
from src.phase4_pq_tier_model import TierModel


def inspect_codebooks(base_dir: Path) -> None:
    codebooks_dir = base_dir / "results" / "phase4_pq" / "codebooks"
    print("\n" + "=" * 70)
    print(" INSPECTING PQ CODEBOOKS")
    print("=" * 70)
    if not (codebooks_dir / "pq_codebooks.npy").exists():
        print(f"Codebooks not found at {codebooks_dir}. Please run Phase 4 training first.")
        return

    centroids, meta = load_trained_codebooks(codebooks_dir)
    m, ks, dsub = centroids.shape
    print(f"File             : {codebooks_dir / 'pq_codebooks.npy'}")
    print(f"Shape            : {centroids.shape} (M={m}, Ks={ks}, dsub={dsub})")
    print(f"Dtype            : {centroids.dtype}")
    print(f"Total Centroids  : {m * ks:,}")
    print(f"Memory Size      : {centroids.nbytes:,} bytes ({centroids.nbytes / 1024:.2f} KB)")
    print(f"Stats            : Min={centroids.min():.4f}, Max={centroids.max():.4f}, "
          f"Mean={centroids.mean():.4f}, Std={centroids.std():.4f}")
    print(f"SHA-256 Checksum : {meta['artifacts']['pq_codebooks_npy']['sha256']}")
    print(f"Training Time    : {meta['training_stats']['training_time_seconds']} s")

    print("\nSample Centroid Vector (Subquantizer 0, Centroid 0):")
    print(f"  {centroids[0, 0, :].round(4)}")


def inspect_codes(base_dir: Path) -> None:
    codes_dir = base_dir / "results" / "phase4_pq" / "codes"
    print("\n" + "=" * 70)
    print(" INSPECTING PQ CODES")
    print("=" * 70)
    if not (codes_dir / "base_pq_codes.npy").exists():
        print(f"PQ codes not found at {codes_dir}. Please run Phase 4 encoding first.")
        return

    codes, meta = load_pq_codes(codes_dir)
    n, m = codes.shape
    print(f"File             : {codes_dir / 'base_pq_codes.npy'}")
    print(f"Shape            : {codes.shape} (N={n:,}, M={m})")
    print(f"Dtype            : {codes.dtype}")
    print(f"Code Size/Vector : {m} bytes")
    print(f"Total Bytes      : {codes.nbytes:,} bytes ({codes.nbytes / (1024*1024):.2f} MB)")
    print(f"Value Range      : [{codes.min()}, {codes.max()}]")
    print(f"SHA-256 Checksum : {meta['artifacts']['base_pq_codes_npy']['sha256']}")

    print("\nSample PQ Codes (Vector IDs):")
    sample_ids = [0, 1, 17_999, 18_000, 50_000, 99_999]
    for vid in sample_ids:
        print(f"  Vector ID {vid:6d}: {codes[vid].tolist()}")


def inspect_tiers(base_dir: Path) -> None:
    meta_dir = base_dir / "results" / "phase4_pq" / "metadata"
    print("\n" + "=" * 70)
    print(" INSPECTING HOT/COLD TIER MODEL")
    print("=" * 70)
    manifest_path = meta_dir / "tier_manifest.json"
    if not manifest_path.exists():
        print(f"Tier manifest not found at {manifest_path}. Generating on the fly...")
        tm = TierModel()
    else:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        tm = TierModel(
            total_vectors=manifest["capacity_config"]["total_vectors"],
            dimension=manifest["capacity_config"]["dimension"],
            hot_ratio=manifest["capacity_config"]["hot_ratio"],
        )

    val = tm.validate_invariants()
    print(f"Total Vectors    : {tm.total_vectors:,}")
    print(f"HOT Count (DRAM) : {tm.hot_count:,} ({tm.hot_count / tm.total_vectors * 100:.1f}%)")
    print(f"COLD Count (SSD) : {tm.cold_count:,} ({tm.cold_count / tm.total_vectors * 100:.1f}%)")
    print(f"Disjoint Set     : {val['is_disjoint']} (Overlap = {val['overlap_count']})")
    print(f"Exhaustive Union : {val['is_exhaustive']}")
    print(f"Invariants Check : {'PASS' if val['is_valid'] else 'FAIL'}")

    print("\nSample Vector Records:")
    sample_ids = [0, 17_999, 18_000, 99_999]
    for vid in sample_ids:
        rec = tm.get_record(vid)
        print(f"  ID {rec.vector_id:6d} -> Tier: {rec.tier:4s} | "
              f"NVMe Offset: {rec.byte_offset:10,d} bytes | "
              f"DRAM Code Index: {rec.pq_code_index:6d}")


def inspect_reconstruction(base_dir: Path, vector_id: int = 42) -> None:
    print("\n" + "=" * 70)
    print(f" INSPECTING RECONSTRUCTION FIDELITY (Vector ID {vector_id})")
    print("=" * 70)

    cb_dir = base_dir / "results" / "phase4_pq" / "codebooks"
    codes_dir = base_dir / "results" / "phase4_pq" / "codes"
    data_dir = base_dir / "data"

    if not (cb_dir / "pq_codebooks.npy").exists() or not (codes_dir / "base_pq_codes.npy").exists():
        print("Required artifacts missing. Run Phase 4 training/encoding first.")
        return

    centroids, _ = load_trained_codebooks(cb_dir)
    codes, _ = load_pq_codes(codes_dir)
    base_vectors = load_fvecs(data_dir / "base_vectors.fvecs")

    orig_vec = base_vectors[vector_id]
    code = codes[vector_id:vector_id+1]
    rec_vec = reconstruct_vectors_numpy(code, centroids)[0]

    diff = orig_vec - rec_vec
    l2_err = float(np.linalg.norm(diff))
    rel_err = float(l2_err / np.linalg.norm(orig_vec))
    mse = float(np.mean(diff ** 2))

    print(f"Original Vector Norm      : {np.linalg.norm(orig_vec):.4f}")
    print(f"Reconstructed Vector Norm : {np.linalg.norm(rec_vec):.4f}")
    print(f"Reconstruction MSE        : {mse:.6f}")
    print(f"Reconstruction L2 Error   : {l2_err:.4f}")
    print(f"Relative L2 Error         : {rel_err:.4f} ({rel_err * 100:.2f}%)")

    print("\nFirst 8 Dimensions (Subvector 0):")
    print(f"  Original      : {orig_vec[:8].round(4)}")
    print(f"  Reconstructed : {rec_vec[:8].round(4)}")
    print(f"  Difference    : {diff[:8].round(4)}")


def inspect_adc(base_dir: Path, query_idx: int = 0) -> None:
    print("\n" + "=" * 70)
    print(f" INSPECTING ASYMMETRIC DISTANCE COMPUTATION (Query {query_idx})")
    print("=" * 70)

    cb_dir = base_dir / "results" / "phase4_pq" / "codebooks"
    codes_dir = base_dir / "results" / "phase4_pq" / "codes"
    data_dir = base_dir / "data"

    if not (cb_dir / "pq_codebooks.npy").exists() or not (codes_dir / "base_pq_codes.npy").exists():
        print("Required artifacts missing. Run Phase 4 training/encoding first.")
        return

    centroids, _ = load_trained_codebooks(cb_dir)
    codes, _ = load_pq_codes(codes_dir)
    query_vectors = load_fvecs(data_dir / "query_vectors.fvecs")
    base_vectors = load_fvecs(data_dir / "base_vectors.fvecs")
    ground_truth = load_ivecs(data_dir / "ground_truth.ivecs")

    q = query_vectors[query_idx]
    lut = compute_adc_lut_single_query(q, centroids)
    print(f"ADC LUT Shape : {lut.shape} (M={lut.shape[0]}, Ks={lut.shape[1]})")
    print(f"LUT Stats     : Min={lut.min():.4f}, Max={lut.max():.4f}, Mean={lut.mean():.4f}")

    dists = compute_adc_distances(lut, codes)
    top_10 = np.argpartition(dists, 10)[:10]
    top_10_sorted = top_10[np.argsort(dists[top_10])]

    gt_10 = ground_truth[query_idx]
    exact_dists_for_gt = np.sum((q - base_vectors[gt_10]) ** 2, axis=1)

    print(f"\nGround Truth Top 10 IDs (Exact Phase 1 Oracle):")
    print(f"  IDs       : {gt_10.tolist()}")
    print(f"  Exact L2  : {exact_dists_for_gt.round(2).tolist()}")

    print(f"\nADC Top 10 Predicted Candidates:")
    print(f"  IDs       : {top_10_sorted.tolist()}")
    print(f"  ADC Dist  : {dists[top_10_sorted].round(2).tolist()}")

    overlap = set(gt_10.tolist()).intersection(set(top_10_sorted.tolist()))
    print(f"\nTop-10 Overlap: {len(overlap)}/10 IDs: {sorted(list(overlap))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Phase 4 PQ artifacts and representations")
    parser.add_argument("--all", action="store_true", help="Inspect all artifacts")
    parser.add_argument("--codebooks", action="store_true", help="Inspect PQ codebooks")
    parser.add_argument("--codes", action="store_true", help="Inspect PQ codes")
    parser.add_argument("--tiers", action="store_true", help="Inspect HOT/COLD tier model")
    parser.add_argument("--reconstruction", action="store_true", help="Inspect reconstruction fidelity")
    parser.add_argument("--adc", action="store_true", help="Inspect ADC Look-Up Table and search")
    parser.add_argument("--vector-id", type=int, default=42, help="Vector ID to inspect")
    parser.add_argument("--query-id", type=int, default=0, help="Query ID to inspect")

    args = parser.parse_args()
    root = workspace_root

    if not any([args.all, args.codebooks, args.codes, args.tiers, args.reconstruction, args.adc]):
        args.all = True

    if args.all or args.codebooks:
        inspect_codebooks(root)
    if args.all or args.codes:
        inspect_codes(root)
    if args.all or args.tiers:
        inspect_tiers(root)
    if args.all or args.reconstruction:
        inspect_reconstruction(root, args.vector_id)
    if args.all or args.adc:
        inspect_adc(root, args.query_id)


if __name__ == "__main__":
    main()
