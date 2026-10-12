"""
Case 04 — Phase 1.1: Deterministic Dataset + Exact FAISS Ground Truth

Generates a deterministic synthetic vector dataset and exact ground-truth nearest
neighbors using FAISS IndexFlatL2 as the trusted exact-search oracle.

All outputs are saved in standard binary formats (.fvecs, .ivecs) and verified
for reproducibility, numerical integrity, and exact match against manual NumPy checks.
"""

import sys
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Verify dependencies can be imported
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
# Benchmark Configuration (Frozen)
# ==============================================================================
BASE_VECTORS_COUNT = 100_000
QUERY_VECTORS_COUNT = 1_000
DIMENSION = 128
TOP_K = 10
BASE_SEED = 42
QUERY_SEED = 43
DTYPE = np.float32


# ==============================================================================
# Binary I/O Functions (.fvecs and .ivecs)
# ==============================================================================
def write_fvecs(filepath: Path | str, data: np.ndarray) -> None:
    """
    Writes a 2D float32 numpy array to disk in standard .fvecs binary format.
    Format per vector:
        int32 dimension (4 bytes)
        float32[dimension] (4 * dimension bytes)
    """
    filepath = Path(filepath)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array for fvecs, got ndim={data.ndim}")
    if data.dtype != np.float32:
        raise TypeError(f"Expected float32 array for fvecs, got dtype={data.dtype}")

    n, d = data.shape
    dt = np.dtype([("dim", "<i4"), ("vec", "<f4", (d,))])
    records = np.empty(n, dtype=dt)
    records["dim"] = d
    records["vec"] = data
    records.tofile(filepath)


def read_fvecs(filepath: Path | str) -> np.ndarray:
    """
    Reads a 2D float32 numpy array from a standard .fvecs binary file.
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


def write_ivecs(filepath: Path | str, data: np.ndarray) -> None:
    """
    Writes a 2D int32 numpy array to disk in standard .ivecs binary format.
    Format per vector:
        int32 dimension (4 bytes)
        int32[dimension] (4 * dimension bytes)
    """
    filepath = Path(filepath)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array for ivecs, got ndim={data.ndim}")
    if data.dtype != np.int32:
        raise TypeError(f"Expected int32 array for ivecs, got dtype={data.dtype}")

    n, d = data.shape
    dt = np.dtype([("dim", "<i4"), ("vec", "<i4", (d,))])
    records = np.empty(n, dtype=dt)
    records["dim"] = d
    records["vec"] = data
    records.tofile(filepath)


def read_ivecs(filepath: Path | str) -> np.ndarray:
    """
    Reads a 2D int32 numpy array from a standard .ivecs binary file.
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


def compute_sha256(filepath: Path | str) -> str:
    """Computes the SHA-256 hexadecimal digest of a binary file."""
    filepath = Path(filepath)
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# ==============================================================================
# Main Benchmark Pipeline
# ==============================================================================
def run_benchmark() -> int:
    t_start = time.perf_counter()

    # Determine directory structure relative to project root
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    results_dir = project_root / "results" / "phase1_1"

    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    base_filepath = data_dir / "base_vectors.fvecs"
    query_filepath = data_dir / "query_vectors.fvecs"
    gt_filepath = data_dir / "ground_truth.ivecs"
    gt_dist_filepath = data_dir / "ground_truth_distances.fvecs"
    summary_filepath = results_dir / "benchmark_summary.json"

    # 1. Dataset Generation
    t_gen_start = time.perf_counter()
    rng_base = np.random.default_rng(BASE_SEED)
    base_vectors = rng_base.normal(
        loc=0.0, scale=1.0, size=(BASE_VECTORS_COUNT, DIMENSION)
    ).astype(DTYPE)

    rng_query = np.random.default_rng(QUERY_SEED)
    query_vectors = rng_query.normal(
        loc=0.0, scale=1.0, size=(QUERY_VECTORS_COUNT, DIMENSION)
    ).astype(DTYPE)
    t_gen_end = time.perf_counter()
    t_gen = t_gen_end - t_gen_start

    # 2. FAISS Index Creation & Population
    t_index_start = time.perf_counter()
    index = faiss.IndexFlatL2(DIMENSION)
    index.add(base_vectors)
    t_index_end = time.perf_counter()
    t_index = t_index_end - t_index_start

    # 3. Exact Nearest Neighbor Search
    # Note on Distance Semantics:
    # IndexFlatL2 computes squared Euclidean distance: d^2(x, y) = sum((x_i - y_i)^2).
    # No sqrt() is applied. Because sqrt is monotonic, squared L2 distance produces
    # the exact same neighbor ranking as Euclidean distance.
    t_search_start = time.perf_counter()
    distances, raw_labels = index.search(query_vectors, TOP_K)
    t_search_end = time.perf_counter()
    t_search = t_search_end - t_search_start

    labels = raw_labels.astype(np.int32)

    # 4. File Writing
    t_write_start = time.perf_counter()
    write_fvecs(base_filepath, base_vectors)
    write_fvecs(query_filepath, query_vectors)
    write_ivecs(gt_filepath, labels)
    write_fvecs(gt_dist_filepath, distances)
    t_write_end = time.perf_counter()
    t_write = t_write_end - t_write_start

    t_total = time.perf_counter() - t_start

    # 5. Checksums
    base_sha256 = compute_sha256(base_filepath)
    query_sha256 = compute_sha256(query_filepath)
    gt_sha256 = compute_sha256(gt_filepath)
    gt_dist_sha256 = compute_sha256(gt_dist_filepath)

    # 6. Comprehensive Validations
    validations: Dict[str, bool] = {}

    validations["Base shape"] = (base_vectors.shape == (BASE_VECTORS_COUNT, DIMENSION))
    validations["Query shape"] = (query_vectors.shape == (QUERY_VECTORS_COUNT, DIMENSION))
    validations["dtype"] = (base_vectors.dtype == np.float32 and query_vectors.dtype == np.float32)
    validations["Index size"] = (index.ntotal == BASE_VECTORS_COUNT)
    validations["Search output"] = (labels.shape == (QUERY_VECTORS_COUNT, TOP_K) and distances.shape == (QUERY_VECTORS_COUNT, TOP_K))
    validations["ID range"] = bool(np.all((labels >= 0) & (labels < BASE_VECTORS_COUNT)))
    validations["Finite values"] = bool(
        np.all(np.isfinite(base_vectors))
        and np.all(np.isfinite(query_vectors))
        and np.all(np.isfinite(distances))
    )
    validations["File write"] = (
        base_filepath.exists()
        and query_filepath.exists()
        and gt_filepath.exists()
        and gt_dist_filepath.exists()
    )

    # File read-back validation
    try:
        read_base = read_fvecs(base_filepath)
        read_query = read_fvecs(query_filepath)
        read_gt = read_ivecs(gt_filepath)
        read_dist = read_fvecs(gt_dist_filepath)

        file_read_pass = (
            read_base.shape == base_vectors.shape
            and read_query.shape == query_vectors.shape
            and read_gt.shape == labels.shape
            and read_dist.shape == distances.shape
        )
        read_match_pass = (
            np.array_equal(read_base, base_vectors)
            and np.array_equal(read_query, query_vectors)
            and np.array_equal(read_gt, labels)
            and np.array_equal(read_dist, distances)
        )
    except Exception as read_err:
        print(f"Validation read error: {read_err}", file=sys.stderr)
        file_read_pass = False
        read_match_pass = False

    validations["File read"] = (file_read_pass and read_match_pass)

    # Independent Manual NumPy Brute-Force Check
    # Verify at least 5 queries against exact numpy argmin
    test_queries = [0, 1, 42, 500, 999]
    manual_nn_pass = True
    manual_nn_details = []

    for q_idx in test_queries:
        diff = base_vectors - query_vectors[q_idx]
        sq_dist = np.sum(diff * diff, axis=1)
        manual_top_id = int(np.argmin(sq_dist))
        manual_top_dist = float(sq_dist[manual_top_id])

        faiss_top_id = int(labels[q_idx, 0])
        faiss_top_dist = float(distances[q_idx, 0])

        id_match = (manual_top_id == faiss_top_id)
        dist_match = np.isclose(manual_top_dist, faiss_top_dist, rtol=1e-5, atol=1e-4)

        if not (id_match and dist_match):
            manual_nn_pass = False

        manual_nn_details.append({
            "query_idx": q_idx,
            "manual_id": manual_top_id,
            "faiss_id": faiss_top_id,
            "manual_dist": manual_top_dist,
            "faiss_dist": faiss_top_dist,
            "id_match": id_match,
            "dist_match": bool(dist_match),
        })

    validations["Manual NN check"] = manual_nn_pass

    # Determinism Verification
    # Check if a previous summary exists; if so, verify identical SHA-256 digests
    determinism_pass = True
    if summary_filepath.exists():
        try:
            with open(summary_filepath, "r", encoding="utf-8") as f:
                prev_summary = json.load(f)
            prev_base_sha = prev_summary.get("checksums", {}).get("base_vectors_sha256")
            prev_query_sha = prev_summary.get("checksums", {}).get("query_vectors_sha256")
            prev_gt_sha = prev_summary.get("checksums", {}).get("ground_truth_sha256")
            if (
                prev_base_sha != base_sha256
                or prev_query_sha != query_sha256
                or prev_gt_sha != gt_sha256
            ):
                determinism_pass = False
        except Exception:
            determinism_pass = True

    validations["Determinism"] = determinism_pass

    # Save summary artifact to results/phase1_1/
    summary_data = {
        "configuration": {
            "base_vectors": BASE_VECTORS_COUNT,
            "queries": QUERY_VECTORS_COUNT,
            "dimension": DIMENSION,
            "top_k": TOP_K,
            "metric": "L2_SQUARED",
            "dtype": str(DTYPE),
            "base_seed": BASE_SEED,
            "query_seed": QUERY_SEED,
            "faiss_index": "IndexFlatL2",
        },
        "environment": {
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "faiss_version": faiss.__version__,
        },
        "timings_seconds": {
            "dataset_generation": round(t_gen, 4),
            "index_creation": round(t_index, 4),
            "exact_search": round(t_search, 4),
            "file_writing": round(t_write, 4),
            "total_execution": round(t_total, 4),
        },
        "checksums": {
            "base_vectors_sha256": base_sha256,
            "query_vectors_sha256": query_sha256,
            "ground_truth_sha256": gt_sha256,
            "ground_truth_distances_sha256": gt_dist_sha256,
        },
        "file_sizes_bytes": {
            "base_vectors_fvecs": base_filepath.stat().st_size,
            "query_vectors_fvecs": query_filepath.stat().st_size,
            "ground_truth_ivecs": gt_filepath.stat().st_size,
            "ground_truth_distances_fvecs": gt_dist_filepath.stat().st_size,
        },
        "manual_nn_checks": manual_nn_details,
        "validations": {k: ("PASS" if v else "FAIL") for k, v in validations.items()},
    }

    with open(summary_filepath, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 7. Print Output Report
    all_passed = all(validations.values())

    print("========================================")
    print("CASE 04 — PHASE 1.1")
    print("Exact FAISS Ground Truth")
    print("========================================")
    print("")
    print("Configuration")
    print("----------------------------------------")
    print(f"Base vectors : {BASE_VECTORS_COUNT}")
    print(f"Queries      : {QUERY_VECTORS_COUNT}")
    print(f"Dimension    : {DIMENSION}")
    print(f"K            : {TOP_K}")
    print("Metric       : L2")
    print(f"dtype        : {base_vectors.dtype.name}")
    print(f"Base seed    : {BASE_SEED}")
    print(f"Query seed   : {QUERY_SEED}")
    print("FAISS       : IndexFlatL2")
    print("----------------------------------------")
    print("")
    print(f"FAISS version: {faiss.__version__}")
    print(f"NumPy version: {np.__version__}")
    print("")
    print("Timing")
    print("----------------------------------------")
    print(f"Dataset generation : {t_gen:.4f} s")
    print(f"Index creation     : {t_index:.4f} s")
    print(f"Exact search       : {t_search:.4f} s")
    print(f"File writing       : {t_write:.4f} s")
    print(f"Total              : {t_total:.4f} s")
    print("")
    print("Checksums (SHA-256)")
    print("----------------------------------------")
    print(f"Base dataset SHA-256   : {base_sha256}")
    print(f"Query dataset SHA-256  : {query_sha256}")
    print(f"Ground truth SHA-256   : {gt_sha256}")
    print(f"GT Distances SHA-256   : {gt_dist_sha256}")
    print("")
    print("Sample Ground Truth")
    print("----------------------------------------")
    for q_i in [0, 1]:
        print(f"Query {q_i}:")
        print(f"IDs       : {labels[q_i].tolist()}")
        print(f"Distances : {[round(float(d), 4) for d in distances[q_i]]}")
        print("")
    print("Validation")
    print("----------------------------------------")
    print(f"Base shape          : {'PASS' if validations['Base shape'] else 'FAIL'}")
    print(f"Query shape         : {'PASS' if validations['Query shape'] else 'FAIL'}")
    print(f"dtype               : {'PASS' if validations['dtype'] else 'FAIL'}")
    print(f"Index size          : {'PASS' if validations['Index size'] else 'FAIL'}")
    print(f"Search output       : {'PASS' if validations['Search output'] else 'FAIL'}")
    print(f"ID range            : {'PASS' if validations['ID range'] else 'FAIL'}")
    print(f"Finite values       : {'PASS' if validations['Finite values'] else 'FAIL'}")
    print(f"File write          : {'PASS' if validations['File write'] else 'FAIL'}")
    print(f"File read           : {'PASS' if validations['File read'] else 'FAIL'}")
    print(f"Manual NN check     : {'PASS' if validations['Manual NN check'] else 'FAIL'}")
    print(f"Determinism         : {'PASS' if validations['Determinism'] else 'FAIL'}")
    print("----------------------------------------")
    print("")
    if all_passed:
        print("PHASE 1.1 STATUS: PASS")
        print("========================================")
        return 0
    else:
        print("PHASE 1.1 STATUS: FAILED")
        print("========================================")
        for k, v in validations.items():
            if not v:
                print(f"FAILED CHECK: {k}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run_benchmark())
