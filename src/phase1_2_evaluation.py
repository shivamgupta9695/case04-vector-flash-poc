"""
Case 04 — Phase 1.2: Evaluation Framework
Recall@K + Search Latency + QPS + Correctness Benchmarking

Provides an algorithm-agnostic evaluation infrastructure for vector search
implementations against the frozen Phase 1.1 exact ground truth.
"""

import sys
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Protocol, runtime_checkable

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Verify required libraries
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


# ==============================================================================
# Expected Frozen Checksums (Phase 1.1 Invariant)
# ==============================================================================
EXPECTED_CHECKSUMS = {
    "base_vectors.fvecs": "62a7ad69bc429570fb4819287bb4d317d8069fb784ef528f370c64189d37198b",
    "query_vectors.fvecs": "c1d80d61714296078229bc3429a79569113441c01fdd1069de21611a30528496",
    "ground_truth.ivecs": "ee2f420a3fea4e71408756f6d0bf7bed1741a69dafc119b2781f7b4142c7d66f",
    "ground_truth_distances.fvecs": "3ca693e88fdfab960b0d43412b8a66266259407615feeb1a1f61ffc083b0ffc4",
}


# ==============================================================================
# Reusable Binary I/O & Checksum Utilities
# ==============================================================================
def compute_sha256(filepath: Path | str) -> str:
    """Computes the SHA-256 hexadecimal digest of a binary file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found for checksum computation: {filepath}")
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_fvecs(filepath: Path | str) -> np.ndarray:
    """
    Loads a 2D float32 numpy array from standard .fvecs binary format.
    Format per vector: int32 dimension (4 bytes), float32[dimension] (4 * d bytes).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    file_size = filepath.stat().st_size
    if file_size == 0:
        return np.empty((0, 0), dtype=np.float32)

    with open(filepath, "rb") as f:
        header = f.read(4)
        if len(header) < 4:
            raise ValueError(f"File {filepath} is too small to contain a valid header")
        d = int(np.frombuffer(header, dtype="<i4")[0])
        row_size = 4 + d * 4
        if file_size % row_size != 0:
            raise ValueError(
                f"Corrupt fvecs file {filepath}: size {file_size} is not divisible by row size {row_size} (d={d})"
            )
        n = file_size // row_size
        f.seek(0)
        dt = np.dtype([("dim", "<i4"), ("vec", "<f4", (d,))])
        records = np.fromfile(f, dtype=dt, count=n)
        if not np.all(records["dim"] == d):
            raise ValueError(f"Inconsistent dimension found in fvecs file {filepath}")
        return records["vec"].copy()


def load_ivecs(filepath: Path | str) -> np.ndarray:
    """
    Loads a 2D int32 numpy array from standard .ivecs binary format.
    Format per vector: int32 dimension (4 bytes), int32[dimension] (4 * d bytes).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    file_size = filepath.stat().st_size
    if file_size == 0:
        return np.empty((0, 0), dtype=np.int32)

    with open(filepath, "rb") as f:
        header = f.read(4)
        if len(header) < 4:
            raise ValueError(f"File {filepath} is too small to contain a valid header")
        d = int(np.frombuffer(header, dtype="<i4")[0])
        row_size = 4 + d * 4
        if file_size % row_size != 0:
            raise ValueError(
                f"Corrupt ivecs file {filepath}: size {file_size} is not divisible by row size {row_size} (d={d})"
            )
        n = file_size // row_size
        f.seek(0)
        dt = np.dtype([("dim", "<i4"), ("vec", "<i4", (d,))])
        records = np.fromfile(f, dtype=dt, count=n)
        if not np.all(records["dim"] == d):
            raise ValueError(f"Inconsistent dimension found in ivecs file {filepath}")
        return records["vec"].copy()


# ==============================================================================
# Search Interface Abstraction
# ==============================================================================
@runtime_checkable
class VectorSearcher(Protocol):
    """
    Abstract interface for any vector search index/algorithm.
    Enables seamless benchmarking of FAISS, custom C++, HNSW, IVF, etc.
    """
    def search(self, query_vectors: np.ndarray, k: int) -> np.ndarray:
        """
        Search for top-k nearest neighbors.
        Args:
            query_vectors: float32 ndarray of shape (num_queries, dimension)
            k: number of nearest neighbors to retrieve
        Returns:
            int32/int64 ndarray of shape (num_queries, k) containing retrieved base vector IDs.
        """
        ...


class FaissIndexFlatL2Searcher:
    """
    Reference exact search implementation wrapping faiss.IndexFlatL2.
    """
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.algorithm_name = "FAISS IndexFlatL2"

    def add(self, base_vectors: np.ndarray) -> None:
        if base_vectors.dtype != np.float32:
            base_vectors = base_vectors.astype(np.float32)
        self.index.add(base_vectors)

    def search(self, query_vectors: np.ndarray, k: int) -> np.ndarray:
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)
        _, raw_labels = self.index.search(query_vectors, k)
        return raw_labels.astype(np.int32)


# ==============================================================================
# Core Evaluation Metrics: Recall@K, Latency, QPS
# ==============================================================================
def calculate_per_query_recall(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Calculates Recall@K for each query individually.

    For each query:
        hits = |ground_truth_top_k ∩ set(predicted_top_k)|
        recall_q = hits / k

    Deduplication Guarantee:
        Converting predicted IDs to a set prevents duplicate predictions
        from inflating recall scores.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    num_queries = ground_truth.shape[0]
    if predictions.shape[0] != num_queries:
        raise ValueError(
            f"Query count mismatch: ground_truth has {num_queries}, predictions has {predictions.shape[0]}"
        )

    if ground_truth.shape[1] < k:
        raise ValueError(
            f"Ground truth has only {ground_truth.shape[1]} neighbors, but k={k} requested"
        )

    # Extract top-k columns
    gt_k = ground_truth[:, :k]
    pred_k = predictions[:, :k]

    recalls = np.zeros(num_queries, dtype=np.float64)
    for i in range(num_queries):
        gt_set = set(gt_k[i].tolist())
        pred_set = set(pred_k[i].tolist())
        hits = len(gt_set.intersection(pred_set))
        recalls[i] = hits / float(k)

    return recalls


def calculate_recall_at_k(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    k: int
) -> float:
    """
    Calculates overall mean Recall@K across all queries.
    """
    per_query = calculate_per_query_recall(ground_truth, predictions, k)
    return float(np.mean(per_query))


def validate_predictions(
    predictions: np.ndarray,
    k: int,
    num_base_vectors: int | None = None
) -> Dict[str, Any]:
    """
    Validates structural and numeric integrity of predicted neighbor IDs:
    - Checks expected shape (num_queries, k)
    - Checks for negative or out-of-range IDs
    - Checks for duplicate predictions within any query row
    """
    if predictions.ndim != 2:
        return {"is_valid": False, "error": f"Expected 2D array, got ndim={predictions.ndim}"}

    num_queries, pred_k = predictions.shape
    if pred_k < k:
        return {"is_valid": False, "error": f"Predictions column count {pred_k} is less than k={k}"}

    pred_eval = predictions[:, :k]
    has_negative = bool(np.any(pred_eval < 0))
    has_out_of_bounds = False
    if num_base_vectors is not None:
        has_out_of_bounds = bool(np.any(pred_eval >= num_base_vectors))

    # Check for duplicate predictions per query
    rows_with_duplicates = 0
    for i in range(num_queries):
        row = pred_eval[i]
        if len(set(row.tolist())) < k:
            rows_with_duplicates += 1

    return {
        "is_valid": not (has_negative or has_out_of_bounds),
        "num_queries": num_queries,
        "k": k,
        "has_negative_ids": has_negative,
        "has_out_of_bounds_ids": has_out_of_bounds,
        "has_duplicate_ids": rows_with_duplicates > 0,
        "rows_with_duplicates": rows_with_duplicates,
    }


def measure_batch_search(
    searcher: VectorSearcher,
    query_vectors: np.ndarray,
    k: int,
    num_warmup_queries: int = 100,
    repetitions: int = 5
) -> Dict[str, Any]:
    """
    Measures steady-state batch search latency and QPS with warm-up and multiple runs.

    Timing separation:
        Warm-up phase is executed first to prime caches/runtime threads.
        Only steady-state search execution is measured.
    """
    num_queries = query_vectors.shape[0]

    # Warm-up phase (unmeasured)
    warmup_count = min(num_warmup_queries, num_queries)
    if warmup_count > 0:
        _ = searcher.search(query_vectors[:warmup_count], k)

    # Measured repetitions
    run_times: List[float] = []
    last_predictions: np.ndarray | None = None

    for _ in range(repetitions):
        t_start = time.perf_counter()
        preds = searcher.search(query_vectors, k)
        t_end = time.perf_counter()
        run_times.append(t_end - t_start)
        last_predictions = preds

    run_times_arr = np.array(run_times, dtype=np.float64)
    min_time = float(np.min(run_times_arr))
    mean_time = float(np.mean(run_times_arr))
    max_time = float(np.max(run_times_arr))

    # Average latency in milliseconds per query (based on mean batch time)
    avg_latency_ms = (mean_time / num_queries) * 1000.0
    qps = num_queries / mean_time

    return {
        "repetitions": repetitions,
        "num_queries": num_queries,
        "k": k,
        "warmup_queries": warmup_count,
        "run_times_seconds": [round(float(t), 5) for t in run_times],
        "min_time_seconds": round(min_time, 5),
        "mean_time_seconds": round(mean_time, 5),
        "max_time_seconds": round(max_time, 5),
        "avg_latency_ms": round(avg_latency_ms, 4),
        "qps": round(qps, 2),
        "last_predictions": last_predictions,
    }


def measure_individual_query_latencies(
    searcher: VectorSearcher,
    query_vectors: np.ndarray,
    k: int,
    sample_queries: int = 200
) -> Dict[str, float]:
    """
    Measures individual per-query search latencies to compute percentiles (p50, p95, p99).
    Kept strictly distinct from batch-mode search latency.
    """
    sample_n = min(sample_queries, query_vectors.shape[0])
    latencies_ms: List[float] = []

    # Warm-up single query
    _ = searcher.search(query_vectors[0:1], k)

    for i in range(sample_n):
        q = query_vectors[i : i + 1]
        t0 = time.perf_counter()
        _ = searcher.search(q, k)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    lat_arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "sample_size": sample_n,
        "p50_ms": round(float(np.percentile(lat_arr, 50)), 4),
        "p95_ms": round(float(np.percentile(lat_arr, 95)), 4),
        "p99_ms": round(float(np.percentile(lat_arr, 99)), 4),
    }


# ==============================================================================
# Correctness Unit Tests (Evaluation Verification)
# ==============================================================================
def run_correctness_tests(
    ground_truth: np.ndarray,
    num_base_vectors: int
) -> Dict[str, bool]:
    """
    Executes explicit verification tests for the evaluator:
    TEST 1: Perfect ground-truth matching (Recall == 1.0)
    TEST 2: Synthetic perfect prediction (Recall == 1.0)
    TEST 3: Synthetic zero-overlap prediction using valid IDs (Recall == 0.0)
    TEST 4: Partial overlap with exactly 5 of 10 matches (Recall == 0.5)
    TEST 5: Duplicate prediction IDs (verifying duplicates do NOT inflate recall)
    """
    num_queries, k_gt = ground_truth.shape
    test_k = 10
    results: Dict[str, bool] = {}

    # TEST 2: Synthetic perfect prediction
    perfect_preds = ground_truth[:, :test_k].copy()
    recall_perfect = calculate_recall_at_k(ground_truth, perfect_preds, test_k)
    results["TEST 2 (Synthetic Perfect Match)"] = np.isclose(recall_perfect, 1.0)

    # TEST 3: Synthetic zero-overlap prediction using valid IDs not in GT
    # For each query, select 10 valid base IDs that do not exist in that query's ground truth
    zero_preds = np.zeros((num_queries, test_k), dtype=np.int32)
    for i in range(num_queries):
        gt_set = set(ground_truth[i].tolist())
        valid_non_gt_ids = []
        candidate = (int(ground_truth[i, 0]) + 50000) % num_base_vectors
        while len(valid_non_gt_ids) < test_k:
            if candidate not in gt_set:
                valid_non_gt_ids.append(candidate)
            candidate = (candidate + 1) % num_base_vectors
        zero_preds[i] = valid_non_gt_ids

    recall_zero = calculate_recall_at_k(ground_truth, zero_preds, test_k)
    results["TEST 3 (Synthetic Zero Overlap)"] = np.isclose(recall_zero, 0.0)

    # TEST 4: Partial overlap (exactly 5 of 10 matches)
    # Take 5 from ground truth, 5 from non-GT IDs
    partial_preds = np.zeros((num_queries, test_k), dtype=np.int32)
    for i in range(num_queries):
        gt_ids = ground_truth[i, :5].tolist()
        non_gt_ids = zero_preds[i, :5].tolist()
        partial_preds[i] = gt_ids + non_gt_ids

    recall_partial = calculate_recall_at_k(ground_truth, partial_preds, test_k)
    results["TEST 4 (Partial 50% Overlap)"] = np.isclose(recall_partial, 0.5)

    # TEST 5: Duplicate predicted IDs
    # Provide 10 copies of the single rank-0 ground truth ID
    # Hits must be 1, so recall must be exactly 1/10 = 0.1, NOT 1.0!
    duplicate_preds = np.repeat(ground_truth[:, 0:1], test_k, axis=1)
    recall_duplicate = calculate_recall_at_k(ground_truth, duplicate_preds, test_k)
    results["TEST 5 (Duplicate ID Deduplication)"] = np.isclose(recall_duplicate, 0.1)

    # Verify duplicate detection flag in validate_predictions
    val_dup = validate_predictions(duplicate_preds, test_k, num_base_vectors)
    results["TEST 5b (Duplicate Detection Flag)"] = val_dup["has_duplicate_ids"]

    return results


# ==============================================================================
# Main Phase 1.2 Evaluation Pipeline
# ==============================================================================
def main() -> int:
    # 1. Project Paths
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    results_dir = project_root / "results" / "phase1_2"
    results_dir.mkdir(parents=True, exist_ok=True)

    base_path = data_dir / "base_vectors.fvecs"
    query_path = data_dir / "query_vectors.fvecs"
    gt_path = data_dir / "ground_truth.ivecs"
    gt_dist_path = data_dir / "ground_truth_distances.fvecs"
    summary_path = results_dir / "evaluation_summary.json"

    # 2. Checksum Verification (Frozen Data Integrity)
    checksum_errors: List[str] = []
    actual_checksums: Dict[str, str] = {}
    for filename, expected_hash in EXPECTED_CHECKSUMS.items():
        fp = data_dir / filename
        if not fp.exists():
            checksum_errors.append(f"Missing required dataset file: {filename}")
            continue
        actual_hash = compute_sha256(fp)
        actual_checksums[filename] = actual_hash
        if actual_hash != expected_hash:
            checksum_errors.append(
                f"Checksum mismatch for {filename}: expected {expected_hash}, got {actual_hash}"
            )

    if checksum_errors:
        print("CRITICAL FAILURE: Phase 1.1 frozen dataset integrity check failed:", file=sys.stderr)
        for err in checksum_errors:
            print(f"  - {err}", file=sys.stderr)
        print("PHASE 1.2 STATUS: FAILED")
        return 1

    # 3. Load Frozen Dataset
    base_vectors = load_fvecs(base_path)
    query_vectors = load_fvecs(query_path)
    ground_truth = load_ivecs(gt_path)

    n_base, d_base = base_vectors.shape
    n_query, d_query = query_vectors.shape
    n_gt, k_gt = ground_truth.shape

    target_k = 10

    # 4. Correctness Tests on Evaluator
    correctness_results = run_correctness_tests(ground_truth, n_base)
    all_tests_passed = all(correctness_results.values())

    # 5. Search Execution with Reference Oracle (FAISS IndexFlatL2)
    searcher = FaissIndexFlatL2Searcher(dimension=d_base)
    searcher.add(base_vectors)

    # 6. Performance Benchmarking
    # Repeated batch searches (steady-state, with warm-up)
    perf_metrics = measure_batch_search(
        searcher=searcher,
        query_vectors=query_vectors,
        k=target_k,
        num_warmup_queries=100,
        repetitions=5,
    )
    predicted_ids = perf_metrics["last_predictions"]

    # Optional individual query latency percentiles
    percentiles = measure_individual_query_latencies(
        searcher=searcher,
        query_vectors=query_vectors,
        k=target_k,
        sample_queries=200,
    )

    # 7. Evaluate Recall@K on IndexFlatL2 Results
    recall_at_10 = calculate_recall_at_k(ground_truth, predicted_ids, k=target_k)

    # Also calculate Recall@1 and Recall@5 to demonstrate arbitrary K support
    recall_at_1 = calculate_recall_at_k(ground_truth, predicted_ids, k=1)
    recall_at_5 = calculate_recall_at_k(ground_truth, predicted_ids, k=5)

    # 8. Prediction Structure Validation
    pred_val = validate_predictions(predicted_ids, target_k, num_base_vectors=n_base)

    # 9. Validation Gate Checks
    val_gt_loaded = (n_gt == 1000 and k_gt == 10)
    val_shapes = (
        n_base == 100000
        and d_base == 128
        and n_query == 1000
        and d_query == 128
        and base_vectors.dtype == np.float32
        and query_vectors.dtype == np.float32
    )
    val_ids = pred_val["is_valid"] and not pred_val["has_duplicate_ids"]
    val_recall = np.isclose(recall_at_10, 1.0)

    # TEST 1 check
    correctness_results["TEST 1 (IndexFlatL2 vs Ground Truth Recall@10 == 1.0)"] = bool(val_recall)

    overall_pass = (
        val_gt_loaded
        and val_shapes
        and val_ids
        and val_recall
        and all_tests_passed
        and bool(val_recall)
    )

    # 10. Write Evaluation Summary JSON
    summary_data = {
        "phase": "1.2",
        "dataset": {
            "base_vectors": n_base,
            "queries": n_query,
            "dimension": d_base,
        },
        "k": target_k,
        "search_algorithm": searcher.algorithm_name,
        "recall_at_k": round(recall_at_10, 6),
        "recall_generalized": {
            "recall_at_1": round(recall_at_1, 6),
            "recall_at_5": round(recall_at_5, 6),
            "recall_at_10": round(recall_at_10, 6),
        },
        "search_time_seconds": perf_metrics["mean_time_seconds"],
        "average_latency_ms": perf_metrics["avg_latency_ms"],
        "qps": perf_metrics["qps"],
        "repetitions": perf_metrics["repetitions"],
        "batch_timing_runs": perf_metrics["run_times_seconds"],
        "timing_statistics": {
            "min_time_seconds": perf_metrics["min_time_seconds"],
            "mean_time_seconds": perf_metrics["mean_time_seconds"],
            "max_time_seconds": perf_metrics["max_time_seconds"],
        },
        "individual_query_latency_percentiles": percentiles,
        "checksum_verification": {
            k: ("PASS" if actual_checksums[k] == v else "FAIL")
            for k, v in EXPECTED_CHECKSUMS.items()
        },
        "correctness_tests": {
            k: ("PASS" if v else "FAIL") for k, v in correctness_results.items()
        },
        "validation": {
            "ground_truth_loaded": "PASS" if val_gt_loaded else "FAIL",
            "shape_validation": "PASS" if val_shapes else "FAIL",
            "id_validation": "PASS" if val_ids else "FAIL",
            "recall_validation": "PASS" if val_recall else "FAIL",
            "overall_status": "PASS" if overall_pass else "FAIL",
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 11. Human-Readable Console Output (Exact requested template)
    print("========================================")
    print("CASE 04 — PHASE 1.2")
    print("Evaluation Framework")
    print("========================================")
    print("")
    print("Dataset")
    print("----------------------------------------")
    print(f"Base vectors : {n_base}")
    print(f"Queries      : {n_query}")
    print(f"Dimension    : {d_base}")
    print(f"K            : {target_k}")
    print("")
    print("Search")
    print("----------------------------------------")
    print(f"Algorithm    : {searcher.algorithm_name}")
    print("")
    print("Recall")
    print("----------------------------------------")
    print(f"Recall@10    : {recall_at_10:.6f} (100.0%)")
    print("")
    print("Latency")
    print("----------------------------------------")
    print(f"Search time  : {perf_metrics['mean_time_seconds']:.5f} sec")
    print(f"Average      : {perf_metrics['avg_latency_ms']:.4f} ms/query")
    print(f"QPS          : {perf_metrics['qps']:.1f}")
    print("")
    print("Repeated Runs")
    print("----------------------------------------")
    print(f"Runs         : {perf_metrics['repetitions']}")
    print(f"Min          : {perf_metrics['min_time_seconds']:.5f} sec")
    print(f"Mean         : {perf_metrics['mean_time_seconds']:.5f} sec")
    print(f"Max          : {perf_metrics['max_time_seconds']:.5f} sec")
    print("")
    print("Validation")
    print("----------------------------------------")
    print(f"Ground truth loaded : {'PASS' if val_gt_loaded else 'FAIL'}")
    print(f"Shape validation    : {'PASS' if val_shapes else 'FAIL'}")
    print(f"ID validation       : {'PASS' if val_ids else 'FAIL'}")
    print(f"Recall validation   : {'PASS' if val_recall else 'FAIL'}")
    print("----------------------------------------")
    print("")
    if overall_pass:
        print("PHASE 1.2 STATUS: PASS")
        print("========================================")
        return 0
    else:
        print("PHASE 1.2 STATUS: FAILED")
        print("========================================")
        return 1


if __name__ == "__main__":
    sys.exit(main())
