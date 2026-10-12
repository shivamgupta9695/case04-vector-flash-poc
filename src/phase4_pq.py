"""
Case 04 — Phase 4: Master Pipeline Orchestrator
Product Quantization + HOT/COLD Data Model + ADC Evaluation

Primary Architectural Mental Model (PPT Target Architecture):
                    QUERY
                      |
                HOT/COLD ROUTER
                  /         \\
                 /           \\
        HOT TIER              COLD TIER
        DRAM                   NVMe SSD
        HNSW (Phase 6)         DiskANN (Phase 5)
        HOT vectors            COLD vectors
        PQ codes/metadata      PQ-assisted search
                 \\             /
                  \\           /
                 MERGE + EXACT
                    RE-RANK
                       |
                     TOP-K

Phase 4 bridges raw FP32 vectors to the production-ready compressed representation:
1. Verifies frozen dataset & ground truth integrity.
2. Deterministically trains PQ codebooks (M=16, Ks=256, dsub=8, seed=42).
3. Encodes all 100,000 base vectors into uint8 PQ codes (16 bytes/vector).
4. Reconstructs vectors and evaluates MSE, RMSE, and relative L2 error.
5. Establishes the logical HOT/COLD tier model (18K HOT / 82K COLD) and NVMe storage mapping.
6. Implements and validates Asymmetric Distance Computation (ADC).
7. Evaluates recall (R@1, R@5, R@10, Candidate R@10@100, R@100) and query latency across methods.
8. Computes actual storage/memory accounting and compression ratios.
9. Exports Phase 5 handoff manifests and comprehensive evaluation reports.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

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

from src.phase1_2_evaluation import (
    load_fvecs,
    load_ivecs,
    compute_sha256,
    EXPECTED_CHECKSUMS,
    FaissIndexFlatL2Searcher,
)
from src.phase4_pq_train import PQConfig, PQTrainer, load_trained_codebooks
from src.phase4_pq_encode import PQEncoder, save_pq_codes, load_pq_codes, validate_pq_codes
from src.phase4_pq_reconstruct import PQReconstructor, save_reconstruction_metrics
from src.phase4_pq_adc import ADCSearcher, compute_adc_lut_single_query, compute_adc_distances
from src.phase4_pq_tier_model import TierModel, VectorTier
from src.phase4_pq_evaluate import (
    measure_search_latency_and_recall,
    ReconstructedPQSearcher,
    generate_evaluation_plots,
)


def run_phase4_pipeline(
    workspace: Path,
    random_seed: int = 42,
    hot_ratio: float = 0.18,
    pq_m: int = 16,
    pq_ks: int = 256,
) -> Dict[str, Any]:
    """Executes the full, end-to-end Phase 4 pipeline."""
    print("=" * 80)
    print(" CASE 04 — PHASE 4: PRODUCT QUANTIZATION & HOT/COLD TIER MODEL")
    print("=" * 80)

    data_dir = workspace / "data"
    results_dir = workspace / "results" / "phase4_pq"
    cb_dir = results_dir / "codebooks"
    codes_dir = results_dir / "codes"
    meta_dir = results_dir / "metadata"
    summaries_dir = results_dir / "summaries"
    diagnostics_dir = results_dir / "diagnostics"
    plots_dir = results_dir / "plots"

    for d in [cb_dir, codes_dir, meta_dir, summaries_dir, diagnostics_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    t0_pipeline = time.perf_counter()

    # --------------------------------------------------------------------------
    # Step 1: Verify Frozen Dataset Integrity
    # --------------------------------------------------------------------------
    print("\n[Step 1/8] Verifying Frozen Phase 1 Dataset & Ground Truth Integrity...")
    for fname, expected_hash in EXPECTED_CHECKSUMS.items():
        fpath = data_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing required Phase 1 dataset file: {fpath}")
        actual_hash = compute_sha256(fpath)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Dataset hash mismatch for {fname}!\n"
                f"  Expected: {expected_hash}\n"
                f"  Actual:   {actual_hash}"
            )
        print(f"  [PASS] {fname:28s} SHA-256: {actual_hash[:16]}...")

    base_vectors = load_fvecs(data_dir / "base_vectors.fvecs")
    query_vectors = load_fvecs(data_dir / "query_vectors.fvecs")
    ground_truth = load_ivecs(data_dir / "ground_truth.ivecs")

    n_base, d_base = base_vectors.shape
    n_query, d_query = query_vectors.shape
    n_gt, k_gt = ground_truth.shape

    print(f"  Base vectors : {n_base:,} × {d_base} (float32)")
    print(f"  Queries      : {n_query:,} × {d_query} (float32)")
    print(f"  Ground truth : {n_gt:,} × {k_gt} (int32)")

    if n_base != 100_000 or d_base != 128 or n_query != 1_000 or d_query != 128:
        raise ValueError(f"Unexpected dataset dimensions: base={base_vectors.shape}, query={query_vectors.shape}")

    # --------------------------------------------------------------------------
    # Step 2: Deterministic PQ Training
    # --------------------------------------------------------------------------
    print(f"\n[Step 2/8] Training Product Quantizer Codebooks (M={pq_m}, Ks={pq_ks}, seed={random_seed})...")
    config = PQConfig(
        dimension=d_base,
        num_subquantizers=pq_m,
        centroids_per_subquantizer=pq_ks,
        nbits=8,
        random_seed=random_seed,
        training_sample_count=n_base,
    )

    trainer = PQTrainer(config)
    centroids = trainer.train(base_vectors)
    cb_artifacts = trainer.save_artifacts(cb_dir)
    print(f"  [PASS] Training complete in {trainer.training_time_seconds:.2f} s")
    print(f"  Centroids shape : {centroids.shape} float32")
    print(f"  Codebooks saved : {cb_artifacts['codebooks_npy'].name}")

    # --------------------------------------------------------------------------
    # Step 3: Vector Encoding (All 100,000 Base Vectors)
    # --------------------------------------------------------------------------
    print("\n[Step 3/8] Encoding 100,000 Base Vectors into PQ uint8 Codes...")
    encoder = PQEncoder(trainer.pq, config)
    t0_enc = time.perf_counter()
    codes = encoder.encode(base_vectors)
    enc_time = time.perf_counter() - t0_enc

    codes_artifacts = save_pq_codes(codes, codes_dir, {"encoding_time_seconds": enc_time})
    print(f"  [PASS] Encoding complete in {enc_time:.3f} s")
    print(f"  PQ Codes shape  : {codes.shape} uint8 ({codes.nbytes:,} bytes)")
    print(f"  Codes saved to  : {codes_artifacts['codes_bin'].name} & {codes_artifacts['codes_npy'].name}")

    # --------------------------------------------------------------------------
    # Step 4: Reconstruction & Error Measurement
    # --------------------------------------------------------------------------
    print("\n[Step 4/8] Evaluating PQ Vector Reconstruction & Error Fidelity (Full 100K)...")
    reconstructor = PQReconstructor(centroids)
    reconstructed_vectors = reconstructor.reconstruct(codes)
    rec_metrics = reconstructor.evaluate_reconstruction_error(base_vectors, codes)
    rec_metrics_path = summaries_dir / "reconstruction_metrics.json"
    save_reconstruction_metrics(rec_metrics, rec_metrics_path)

    print(f"  [PASS] Reconstruction throughput : {rec_metrics['reconstruction_throughput_vps']:,} vectors/sec")
    print(f"  Mean Squared Error (MSE)   : {rec_metrics['aggregate_metrics']['mse']:.6f}")
    print(f"  Root Mean Sq Error (RMSE)  : {rec_metrics['aggregate_metrics']['rmse']:.6f}")
    print(f"  Relative L2 Error          : {rec_metrics['aggregate_metrics']['relative_l2_error']:.6f} "
          f"({rec_metrics['aggregate_metrics']['relative_l2_error'] * 100:.2f}%)")

    # --------------------------------------------------------------------------
    # Step 5: Logical HOT/COLD Tier Model & NVMe Storage Mapping
    # --------------------------------------------------------------------------
    print(f"\n[Step 5/8] Constructing HOT/COLD Tier Model (Target {hot_ratio*100:.0f}% DRAM / {(1-hot_ratio)*100:.0f}% NVMe)...")
    tier_model = TierModel(
        total_vectors=n_base,
        dimension=d_base,
        hot_ratio=hot_ratio,
        nvme_store_path="storage/ssd_vectors/vectors.bin",
        dram_pq_codes_path="results/phase4_pq/codes/base_pq_codes.bin",
    )
    tier_artifacts = tier_model.save_artifacts(meta_dir)
    invariants = tier_model.validate_invariants()

    print(f"  [PASS] Tier Invariants Validated:")
    print(f"    HOT Vectors  (DRAM) : {tier_model.hot_count:,} ({tier_model.hot_count / n_base * 100:.1f}%)")
    print(f"    COLD Vectors (NVMe) : {tier_model.cold_count:,} ({tier_model.cold_count / n_base * 100:.1f}%)")
    print(f"    Disjoint Partition  : {invariants['is_disjoint']} (Overlap = {invariants['overlap_count']})")
    print(f"    Exhaustive Union    : {invariants['is_exhaustive']}")
    print(f"  Tier Manifests saved  : {tier_artifacts['tier_manifest'].name}")

    # --------------------------------------------------------------------------
    # Step 6: ADC Mathematical Identity & Verification
    # --------------------------------------------------------------------------
    print("\n[Step 6/8] Validating Asymmetric Distance Computation (ADC) Primitive...")
    adc_searcher = ADCSearcher(centroids, codes)
    identity_check = adc_searcher.verify_mathematical_identity(
        sample_query=query_vectors[0],
        reconstructed_vectors=reconstructed_vectors,
        num_check=500,
        atol=1e-4,
    )
    if not identity_check["identity_verified"]:
        raise ValueError(
            f"ADC identity check failed! Max difference: {identity_check['max_absolute_difference']}"
        )
    print(f"  [PASS] Mathematical Identity D_ADC(q, code_i) == ||q - x_hat_i||^2 verified "
          f"(Max diff = {identity_check['max_absolute_difference']:.2e} < 1e-4)")

    # --------------------------------------------------------------------------
    # Step 7: Search Quality & Latency Benchmark Across Methods
    # --------------------------------------------------------------------------
    print("\n[Step 7/8] Benchmarking Search Quality & Latency (1,000 queries, k=100)...")

    # Generate reference top-100 ground truth using exact IndexFlatL2 in-memory
    # (Without modifying or touching the frozen Phase 1 ground_truth.ivecs file)
    exact_oracle = FaissIndexFlatL2Searcher(dimension=d_base)
    exact_oracle.add(base_vectors)
    print("  Generating in-memory exact top-100 reference for Recall@100 evaluation...")
    _, exact_top100_gt = exact_oracle.index.search(query_vectors, 100)

    # Method 1: Exact FP32
    print("  1. Evaluating Exact FP32 (FAISS IndexFlatL2 Oracle)...")
    res_exact = measure_search_latency_and_recall(
        search_fn=lambda q, k: (exact_oracle.search(q, k), np.zeros((q.shape[0], k), dtype=np.float32)),
        queries=query_vectors,
        ground_truth_10=ground_truth,
        ground_truth_100=exact_top100_gt,
        k=100,
        warmup_queries=100,
    )
    res_exact["method_name"] = "Exact FP32"

    # Method 2: Custom ADC (LUT-based)
    print("  2. Evaluating Custom ADC (Asymmetric Distance Computation)...")
    res_adc = measure_search_latency_and_recall(
        search_fn=lambda q, k: adc_searcher.search_batch(q, k),
        queries=query_vectors,
        ground_truth_10=ground_truth,
        ground_truth_100=exact_top100_gt,
        k=100,
        warmup_queries=100,
    )
    res_adc["method_name"] = "Custom ADC"

    # Method 3: Reconstructed PQ
    print("  3. Evaluating Reconstructed PQ (Exact L2 on reconstructed vectors)...")
    rec_searcher = ReconstructedPQSearcher(reconstructed_vectors)
    res_recon = measure_search_latency_and_recall(
        search_fn=lambda q, k: rec_searcher.search_batch(q, k),
        queries=query_vectors,
        ground_truth_10=ground_truth,
        ground_truth_100=exact_top100_gt,
        k=100,
        warmup_queries=50,
    )
    res_recon["method_name"] = "Reconstructed PQ"

    # Method 4: FAISS IndexPQ (C++ native ADC baseline)
    print("  4. Evaluating FAISS IndexPQ (C++ Native ADC Baseline)...")
    faiss_index_pq = faiss.IndexPQ(d_base, pq_m, 8)
    faiss_index_pq.pq = trainer.pq
    faiss_index_pq.is_trained = True
    faiss_index_pq.add(base_vectors)
    res_faiss_pq = measure_search_latency_and_recall(
        search_fn=lambda q, k: (faiss_index_pq.search(q, k)[1], faiss_index_pq.search(q, k)[0]),
        queries=query_vectors,
        ground_truth_10=ground_truth,
        ground_truth_100=exact_top100_gt,
        k=100,
        warmup_queries=100,
    )
    res_faiss_pq["method_name"] = "FAISS IndexPQ"

    search_summary = {
        "exact_fp32": res_exact,
        "adc_numpy": res_adc,
        "reconstructed_pq": res_recon,
        "faiss_index_pq": res_faiss_pq,
    }
    with open(summaries_dir / "search_comparison.json", "w", encoding="utf-8") as f:
        json.dump(search_summary, f, indent=2)

    print("\n  ==========================================================================")
    print(f"  {'Method':<18s} | {'R@1':<7s} | {'R@5':<7s} | {'R@10':<7s} | {'Cand R@10@100':<13s} | {'Avg Lat':<8s} | {'QPS':<8s}")
    print("  --------------------------------------------------------------------------")
    for key in ["exact_fp32", "adc_numpy", "reconstructed_pq", "faiss_index_pq"]:
        rec = search_summary[key]
        print(f"  {rec['method_name']:<18s} | "
              f"{rec['recall_at_1']:.4f}  | "
              f"{rec['recall_at_5']:.4f}  | "
              f"{rec['recall_at_10']:.4f}  | "
              f"{rec['candidate_recall_10_in_100']:.4f}        | "
              f"{rec['avg_latency_ms']:.3f} ms | "
              f"{rec['qps']:>6.0f}")
    print("  ==========================================================================")

    # Generate plots
    print("\n  Generating publication-quality comparison charts...")
    diff_l2 = np.linalg.norm(base_vectors - reconstructed_vectors, axis=1)
    gen_plots = generate_evaluation_plots(search_summary, diff_l2, plots_dir)
    for p in gen_plots:
        print(f"    Saved: {p.name}")

    # --------------------------------------------------------------------------
    # Step 8: Memory & Storage Accounting
    # --------------------------------------------------------------------------
    print("\n[Step 8/8] Computing Accurate Memory & Storage Accounting...")
    fp32_vector_bytes = n_base * d_base * 4  # 51,200,000 bytes
    pq_code_bytes = n_base * pq_m * 1       # 1,600,000 bytes
    codebook_bytes = pq_m * pq_ks * (d_base // pq_m) * 4  # 131,072 bytes
    metadata_bytes = sum(f.stat().st_size for f in meta_dir.glob("*.json"))

    code_only_compression_ratio = fp32_vector_bytes / pq_code_bytes
    total_pq_representation_bytes = pq_code_bytes + codebook_bytes + metadata_bytes
    effective_compression_ratio = fp32_vector_bytes / total_pq_representation_bytes

    print(f"  Raw FP32 Base Vectors    : {fp32_vector_bytes:,} bytes ({fp32_vector_bytes / (1024*1024):.2f} MB)")
    print(f"  PQ Codes Array           : {pq_code_bytes:,} bytes ({pq_code_bytes / (1024*1024):.2f} MB)")
    print(f"  Codebooks (Centroids)    : {codebook_bytes:,} bytes ({codebook_bytes / 1024:.2f} KB)")
    print(f"  Tier & Catalog Metadata  : {metadata_bytes:,} bytes ({metadata_bytes / 1024:.2f} KB)")
    print(f"  Total PQ DRAM Footprint  : {total_pq_representation_bytes:,} bytes ({total_pq_representation_bytes / (1024*1024):.2f} MB)")
    print(f"  Code-Only Compression    : {code_only_compression_ratio:.2f}x (512 bytes -> 16 bytes/vector)")
    print(f"  Effective Compression    : {effective_compression_ratio:.2f}x (including codebooks and metadata)")

    # --------------------------------------------------------------------------
    # Export Phase 5 Handoff Manifest
    # --------------------------------------------------------------------------
    phase5_manifest = {
        "phase": "Phase 4 -> Phase 5 Handoff",
        "description": "Artifacts and data structures for Phase 5 (DiskANN + NVMe COLD vectors)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_dataset": {
            "base_vectors_file": "data/base_vectors.fvecs",
            "base_sha256": EXPECTED_CHECKSUMS["base_vectors.fvecs"],
            "query_vectors_file": "data/query_vectors.fvecs",
            "query_sha256": EXPECTED_CHECKSUMS["query_vectors.fvecs"],
            "ground_truth_file": "data/ground_truth.ivecs",
            "gt_sha256": EXPECTED_CHECKSUMS["ground_truth.ivecs"],
            "num_base_vectors": n_base,
            "num_queries": n_query,
            "dimension": d_base,
            "dtype": "float32",
        },
        "pq_artifacts": {
            "codebooks_file": "results/phase4_pq/codebooks/pq_codebooks.npy",
            "codebooks_sha256": compute_sha256(cb_artifacts["codebooks_npy"]),
            "codes_bin_file": "results/phase4_pq/codes/base_pq_codes.bin",
            "codes_bin_sha256": compute_sha256(codes_artifacts["codes_bin"]),
            "codes_npy_file": "results/phase4_pq/codes/base_pq_codes.npy",
            "codes_npy_sha256": compute_sha256(codes_artifacts["codes_npy"]),
            "m": pq_m,
            "ks": pq_ks,
            "dsub": d_base // pq_m,
            "code_bytes_per_vector": pq_m,
        },
        "cold_tier_for_phase5_diskann": {
            "cold_manifest_file": "results/phase4_pq/metadata/cold_vector_manifest.json",
            "num_cold_vectors": tier_model.cold_count,
            "nvme_store_file": tier_model.nvme_store_path,
            "nvme_offset_rule": "byte_offset = vector_id * 128 * 4",
            "first_cold_id": min(tier_model.cold_ids),
            "last_cold_id": max(tier_model.cold_ids),
        },
        "search_quality_baseline": {
            "adc_recall_at_1": res_adc["recall_at_1"],
            "adc_recall_at_5": res_adc["recall_at_5"],
            "adc_recall_at_10": res_adc["recall_at_10"],
            "adc_candidate_recall_10_in_100": res_adc["candidate_recall_10_in_100"],
            "reconstruction_mse": rec_metrics["aggregate_metrics"]["mse"],
            "reconstruction_rmse": rec_metrics["aggregate_metrics"]["rmse"],
            "reconstruction_relative_l2_error": rec_metrics["aggregate_metrics"]["relative_l2_error"],
        },
    }
    phase5_manifest_path = meta_dir / "phase5_handoff_manifest.json"
    with open(phase5_manifest_path, "w", encoding="utf-8") as f:
        json.dump(phase5_manifest, f, indent=2)
    print(f"\n  [PASS] Phase 5 Handoff Manifest exported: {phase5_manifest_path.name}")

    total_pipeline_time = time.perf_counter() - t0_pipeline

    # --------------------------------------------------------------------------
    # Master Summary Report
    # --------------------------------------------------------------------------
    summary = {
        "phase": "Phase 4",
        "title": "Product Quantization & HOT/COLD Tier Data Model",
        "status": "PASS",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_execution_seconds": round(total_pipeline_time, 2),
        "dataset": {
            "num_base_vectors": n_base,
            "num_queries": n_query,
            "dimension": d_base,
            "dtype": "float32",
        },
        "pq_configuration": config.to_dict(),
        "storage_accounting": {
            "fp32_vector_bytes": fp32_vector_bytes,
            "pq_code_bytes": pq_code_bytes,
            "codebook_bytes": codebook_bytes,
            "metadata_bytes": metadata_bytes,
            "total_pq_representation_bytes": total_pq_representation_bytes,
            "code_only_compression_ratio": round(code_only_compression_ratio, 2),
            "effective_compression_ratio": round(effective_compression_ratio, 2),
        },
        "reconstruction_error": rec_metrics["aggregate_metrics"],
        "tier_model": {
            "total_vectors": tier_model.total_vectors,
            "hot_count": tier_model.hot_count,
            "cold_count": tier_model.cold_count,
            "hot_ratio": round(tier_model.hot_count / tier_model.total_vectors, 4),
            "cold_ratio": round(tier_model.cold_count / tier_model.total_vectors, 4),
            "invariants_pass": invariants["is_valid"],
        },
        "search_benchmark": search_summary,
        "phase5_readiness": "READY",
    }

    summary_path = summaries_dir / "phase4_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DONE] Master summary report generated: {summary_path}")
    print(f"Total pipeline execution time: {total_pipeline_time:.2f} s")
    print("=" * 80)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 Master Pipeline Orchestrator")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--hot-ratio", type=float, default=0.18, help="Initial HOT tier capacity ratio")
    parser.add_argument("--m", type=int, default=16, help="Number of subquantizers")
    parser.add_argument("--ks", type=int, default=256, help="Centroids per subquantizer")
    args = parser.parse_args()

    run_phase4_pipeline(
        workspace=workspace_root,
        random_seed=args.seed,
        hot_ratio=args.hot_ratio,
        pq_m=args.m,
        pq_ks=args.ks,
    )


if __name__ == "__main__":
    main()
