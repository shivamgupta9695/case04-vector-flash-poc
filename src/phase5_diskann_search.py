"""
Case 04 — Phase 5: Official DiskANN3 Python Integration Adapter

Wraps the official Microsoft DiskANN3 disk index search engine. Provides query
execution, local <-> global ID translation, COLD membership validation,
COLD Recall@K evaluation against exact FAISS oracle, and Global Candidate Coverage.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Ensure workspace root in path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

try:
    import numpy as np
except ImportError as err:
    print(f"CRITICAL ERROR: Failed to import NumPy: {err}", file=sys.stderr)
    sys.exit(1)

from src.phase1_2_evaluation import load_fvecs, load_ivecs, compute_sha256
from src.phase5_prepare_cold import write_diskann_bin, read_diskann_bin


# ==============================================================================
# Environment Configuration & Path Resolution (Item 0)
# ==============================================================================

# Primary environment variable: DISKANN_QUERY_BIN
# Legacy alias: DISKANN_QUERY_BIN_WSL
# Repository path variables: DISKANN_REPO_PATH_WSL, DISKANN_REPO_WSL
def get_diskann_query_bin() -> str:
    """
    Resolves the DiskANN query executable path with documented precedence:
    1. DISKANN_QUERY_BIN (primary mentor-specified environment variable)
    2. DISKANN_QUERY_BIN_WSL (legacy backward-compatible alias)
    3. Derived from DISKANN_REPO_PATH_WSL or DISKANN_REPO_WSL environment variables
    4. Portable fallback default: ~/diskann-work/DiskANN/target/release/diskann_query
    
    Any personal paths (e.g. /home/<user>/...) are normalized to ~/
    to ensure portability across machines and user accounts.
    """
    raw_bin = (
        os.environ.get("DISKANN_QUERY_BIN")
        or os.environ.get("DISKANN_QUERY_BIN_WSL")
    )
    if raw_bin and raw_bin.strip():
        bin_path = raw_bin.strip()
    else:
        repo = os.environ.get("DISKANN_REPO_PATH_WSL") or os.environ.get("DISKANN_REPO_WSL")
        if repo and repo.strip():
            bin_path = f"{repo.strip().rstrip('/')}/target/release/diskann_query"
        else:
            bin_path = "~/diskann-work/DiskANN/target/release/diskann_query"

    # Normalize any /home/<user>/... path to portable ~/ representation
    if bin_path.startswith("/home/"):
        parts = bin_path.split("/", 3)
        if len(parts) >= 4:
            bin_path = f"~/{parts[3]}"

    return bin_path


DISKANN_QUERY_BIN = get_diskann_query_bin()
# Maintain backward-compatible aliases
DISKANN_QUERY_BIN_WSL = DISKANN_QUERY_BIN
DISKANN_REPO_PATH_WSL = os.environ.get("DISKANN_REPO_PATH_WSL", "~/diskann-work/DiskANN").rstrip("/")


def is_windows_native_executable(bin_path: str) -> bool:
    """Checks whether bin_path is a native Windows executable that exists and is runnable."""
    p = Path(bin_path)
    if p.is_file():
        if p.suffix.lower() in [".exe", ".bat", ".cmd"] or os.access(str(p), os.X_OK):
            return True
    return False


def format_wsl_command(bin_path: str, args: List[str]) -> List[str]:
    """
    Constructs a robust WSL command array via 'wsl bash -c':
    1. Guarantees ~/ tilde expansion in bash without leaking personal paths.
    2. Safely quotes all arguments using shlex to handle spaces and special characters.
    """
    if bin_path.startswith("~/"):
        sub_path = bin_path[2:]
        quoted_bin = f"~/{shlex.quote(sub_path)}"
    else:
        quoted_bin = shlex.quote(bin_path)

    quoted_args = " ".join(shlex.quote(a) for a in args)
    bash_cmd_str = f"{quoted_bin} {quoted_args}" if quoted_args else quoted_bin
    return ["wsl", "bash", "-c", bash_cmd_str]


def validate_diskann_executable(bin_path: str) -> Tuple[bool, bool, str]:
    """
    Validates that the configured DiskANN query binary can be found and launched.
    Returns (is_valid, is_windows_native, error_message).
    """
    if is_windows_native_executable(bin_path):
        try:
            res = subprocess.run([bin_path, "--help"], capture_output=True, text=True, check=False)
            if res.returncode == 0 or "--index-prefix" in res.stdout or "--index-prefix" in res.stderr:
                return True, True, ""
            return False, True, f"Native Windows binary '{bin_path}' failed --help check (code {res.returncode}): {res.stderr.strip()}"
        except Exception as e:
            return False, True, f"Failed to launch native Windows executable '{bin_path}': {e}"

    # WSL mode validation
    # 1. Check if executable exists and has execute permissions in WSL
    if bin_path.startswith("~/"):
        test_path = f"$HOME/{shlex.quote(bin_path[2:])}"
    else:
        test_path = shlex.quote(bin_path)

    cmd_test = ["wsl", "bash", "-c", f"test -x {test_path}"]
    res_test = subprocess.run(cmd_test, capture_output=True, text=True, check=False)
    if res_test.returncode != 0:
        return (
            False,
            False,
            f"DiskANN query executable not found or not executable in WSL at '{bin_path}'. "
            f"Please ensure DiskANN is built or configure DISKANN_QUERY_BIN to the valid path of diskann_query."
        )

    # 2. Check if it launches with --help
    cmd_run = format_wsl_command(bin_path, ["--help"])
    res_run = subprocess.run(cmd_run, capture_output=True, text=True, check=False)
    if res_run.returncode == 0 or "--index-prefix" in res_run.stdout:
        return True, False, ""

    return (
        False,
        False,
        f"DiskANN query executable in WSL failed launch check (code {res_run.returncode}): {res_run.stderr.strip()}"
    )


def resolve_diskann_index_prefix(index_prefix: Path | str | None = None) -> Path:
    """
    Resolves the DiskANN index prefix with documented precedence:
    1. Explicit index_prefix argument (if provided and not None)
    2. DISKANN_INDEX_PREFIX environment variable
    3. Project-relative default: workspace_root / "results/phase5_diskann/index/cold_disk_index_tuned"
    
    Returns a resolved absolute Path.
    """
    if index_prefix is not None:
        p = Path(index_prefix)
        if not p.is_absolute():
            p = workspace_root / p
    elif "DISKANN_INDEX_PREFIX" in os.environ and os.environ["DISKANN_INDEX_PREFIX"].strip():
        env_val = os.environ["DISKANN_INDEX_PREFIX"].strip()
        p = Path(env_val)
        if not p.is_absolute():
            p = workspace_root / p
    else:
        p = workspace_root / "results/phase5_diskann/index/cold_disk_index_tuned"

    return p.resolve()


DISKANN_INDEX_PREFIX = str(resolve_diskann_index_prefix())


@dataclass
class DiskANNSearchResult:
    """Encapsulates results from a DiskANN search query batch."""
    num_queries: int
    k: int
    l_search: int
    beam_width: int
    local_ids: np.ndarray      # shape (N, k), uint32
    global_ids: np.ndarray     # shape (N, k), int64
    distances: np.ndarray      # shape (N, k), float32
    latencies_micros: np.ndarray  # shape (N,), uint64
    total_elapsed_ms: float
    qps: float


class ColdDiskANNIndex:
    """
    Python integration adapter for Microsoft DiskANN3 disk-backed index.
    """

    def __init__(
        self,
        index_prefix: Path | str | None = None,
        id_map_path: Path | str | None = None,
        cold_manifest_path: Path | str | None = None,
        num_threads: int = 8,
        cache_nodes: int = 2000,
        query_bin: str | None = None,
    ):
        self.index_prefix = resolve_diskann_index_prefix(index_prefix)
        self.num_threads = num_threads
        self.cache_nodes = cache_nodes
        self.is_open = False
        self.last_command: List[str] | None = None
        self.last_elapsed_wall_ms: float = 0.0

        # Resolve and validate executable
        self.query_bin = query_bin or get_diskann_query_bin()
        is_valid, is_win, err_msg = validate_diskann_executable(self.query_bin)
        if not is_valid:
            raise FileNotFoundError(f"Cannot initialize ColdDiskANNIndex: {err_msg}")
        self.is_windows_native = is_win

        # Resolve ID mapping path
        if id_map_path is None:
            self.id_map_path = workspace_root / "results/phase5_diskann/metadata/id_mapping.json"
        else:
            self.id_map_path = Path(id_map_path).resolve()
            
        # Resolve COLD manifest path
        if cold_manifest_path is None:
            self.cold_manifest_path = workspace_root / "results/phase4_pq/metadata/cold_vector_manifest.json"
        else:
            self.cold_manifest_path = Path(cold_manifest_path).resolve()
            
        self.local_to_global: np.ndarray | None = None
        self.cold_membership_set: set[int] | None = None
        
        self.load()

    def _to_wsl_path(self, path: Path | str) -> str:
        """
        Converts a path for WSL execution.
        If the path is within workspace_root, returns a relative path starting with './'
        (e.g., './results/phase5_diskann/index/cold_disk_index_tuned').
        This ensures that personal paths (like /home/<user> or C:/Users/<user>) are not embedded
        in the executed search command.
        If outside workspace_root, converts to a standard WSL mount path (/mnt/<drive>/...).
        """
        resolved = Path(path).resolve()
        try:
            rel = resolved.relative_to(workspace_root)
            return f"./{rel.as_posix()}"
        except ValueError:
            p = str(resolved).replace("\\", "/")
            if len(p) >= 2 and p[1] == ":":
                drive = p[0].lower()
                return f"/mnt/{drive}{p[2:]}"
            return p

    def load(self) -> None:
        """Verifies index artifact files on disk and loads mapping metadata."""
        prefix_str = str(self.index_prefix)
        disk_index = Path(prefix_str + "_disk.index")
        pq_compressed = Path(prefix_str + "_pq_compressed.bin")
        pq_pivots = Path(prefix_str + "_pq_pivots.bin")

        for f, desc in [
            (disk_index, "disk index file"),
            (pq_compressed, "compressed pq file"),
            (pq_pivots, "pq pivots file"),
        ]:
            if not f.is_file():
                raise FileNotFoundError(
                    f"Missing required DiskANN artifact: {desc} at '{f}'. "
                    f"Prefix resolved to: '{self.index_prefix}'. "
                    f"Please verify DISKANN_INDEX_PREFIX or ensure Phase 5 index is built."
                )

        # Load ID mapping
        if not self.id_map_path.is_file():
            raise FileNotFoundError(f"ID mapping file not found at {self.id_map_path}")
            
        with open(self.id_map_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.local_to_global = np.array(data["local_to_global"], dtype=np.int64)

        # Load COLD membership
        if self.cold_manifest_path.is_file():
            with open(self.cold_manifest_path, "r", encoding="utf-8") as f:
                cdata = json.load(f)
                self.cold_membership_set = set(cdata["cold_vector_ids"])
        else:
            self.cold_membership_set = set(self.local_to_global)

        self.is_open = True

    def close(self) -> None:
        """Closes the index handle."""
        self.is_open = False

    def reopen(self) -> None:
        """Reopens and re-validates the index."""
        self.load()

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        l_search: int = 100,
        beam_width: int = 4,
    ) -> DiskANNSearchResult:
        """
        Executes approximate nearest neighbor search over query batch.
        queries: shape (N, 128), float32.
        """
        if not self.is_open:
            raise RuntimeError("Cannot search: ColdDiskANNIndex is closed.")

        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        if queries.ndim != 2:
            raise ValueError(f"Queries must be 2D array, got shape {queries.shape}")

        n_queries, dim = queries.shape
        if dim != 128:
            raise ValueError(f"Query dimension must be 128, got {dim}")

        if not np.all(np.isfinite(queries)):
            raise ValueError("Query array contains non-finite values (NaN or Inf)")

        if k <= 0:
            raise ValueError(f"k must be strictly positive, got {k}")

        if k > len(self.local_to_global):
            raise ValueError(f"k ({k}) exceeds total indexed vectors ({len(self.local_to_global)})")

        if l_search < k:
            raise ValueError(f"search_l ({l_search}) must be >= k ({k})")

        if n_queries == 0:
            return DiskANNSearchResult(
                num_queries=0,
                k=k,
                l_search=l_search,
                beam_width=beam_width,
                local_ids=np.empty((0, k), dtype=np.uint32),
                global_ids=np.empty((0, k), dtype=np.int64),
                distances=np.empty((0, k), dtype=np.float32),
                latencies_micros=np.empty((0,), dtype=np.uint64),
                total_elapsed_ms=0.0,
                qps=0.0,
            )

        # Temporary files for WSL execution
        temp_dir = workspace_root / "results/phase5_diskann/logs"
        temp_dir.mkdir(parents=True, exist_ok=True)
        t_stamp = int(time.time() * 1000)
        temp_queries_file = temp_dir / f"temp_queries_{t_stamp}.bin"
        temp_output_file = temp_dir / f"temp_search_out_{t_stamp}.json"

        try:
            write_diskann_bin(temp_queries_file, queries)

            wsl_index_prefix = self._to_wsl_path(self.index_prefix)
            wsl_queries = self._to_wsl_path(temp_queries_file)
            wsl_out = self._to_wsl_path(temp_output_file)
            wsl_id_map = self._to_wsl_path(self.id_map_path)

            if self.is_windows_native:
                cmd = [
                    self.query_bin,
                    "--index-prefix", str(self.index_prefix),
                    "--queries", str(temp_queries_file),
                    "-k", str(k),
                    "-l", str(l_search),
                    "--beam-width", str(beam_width),
                    "--num-threads", str(self.num_threads),
                    "--cache-nodes", str(self.cache_nodes),
                    "--id-map", str(self.id_map_path),
                    "--output", str(temp_output_file),
                ]
            else:
                wsl_index_prefix = self._to_wsl_path(self.index_prefix)
                wsl_queries = self._to_wsl_path(temp_queries_file)
                wsl_out = self._to_wsl_path(temp_output_file)
                wsl_id_map = self._to_wsl_path(self.id_map_path)

                search_args = [
                    "--index-prefix", wsl_index_prefix,
                    "--queries", wsl_queries,
                    "-k", str(k),
                    "-l", str(l_search),
                    "--beam-width", str(beam_width),
                    "--num-threads", str(self.num_threads),
                    "--cache-nodes", str(self.cache_nodes),
                    "--id-map", wsl_id_map,
                    "--output", wsl_out,
                ]
                cmd = format_wsl_command(self.query_bin, search_args)

            self.last_command = list(cmd)

            t0 = time.perf_counter()
            proc = subprocess.run(
                cmd,
                cwd=str(workspace_root),
                capture_output=True,
                text=True,
                check=False
            )
            self.last_elapsed_wall_ms = (time.perf_counter() - t0) * 1000.0

            if proc.returncode != 0:
                raise RuntimeError(
                    f"DiskANN query search failed (code {proc.returncode}):\nStdout: {proc.stdout}\nStderr: {proc.stderr}"
                )

            with open(temp_output_file, "r", encoding="utf-8") as f:
                report = json.load(f)

            # Parse results
            results_list = report["results"]
            local_ids_arr = np.zeros((n_queries, k), dtype=np.uint32)
            global_ids_arr = np.zeros((n_queries, k), dtype=np.int64)
            distances_arr = np.zeros((n_queries, k), dtype=np.float32)
            latencies_arr = np.zeros((n_queries,), dtype=np.uint64)

            for item in results_list:
                qi = item["query_idx"]
                l_ids = item["local_ids"]
                g_ids = item.get("global_ids", [])
                dists = item["distances"]
                n_res = min(k, len(l_ids))

                local_ids_arr[qi, :n_res] = l_ids[:n_res]
                distances_arr[qi, :n_res] = dists[:n_res]
                latencies_arr[qi] = item.get("duration_micros", 0)

                if g_ids:
                    global_ids_arr[qi, :n_res] = g_ids[:n_res]
                else:
                    global_ids_arr[qi, :n_res] = self.local_to_global[local_ids_arr[qi, :n_res]]

            return DiskANNSearchResult(
                num_queries=n_queries,
                k=k,
                l_search=l_search,
                beam_width=beam_width,
                local_ids=local_ids_arr,
                global_ids=global_ids_arr,
                distances=distances_arr,
                latencies_micros=latencies_arr,
                total_elapsed_ms=report["total_elapsed_ms"],
                qps=report["qps"],
            )
        finally:
            # Clean up temp files
            if temp_queries_file.exists():
                try:
                    temp_queries_file.unlink()
                except OSError:
                    pass
            if temp_output_file.exists():
                try:
                    temp_output_file.unlink()
                except OSError:
                    pass

    def validate_cold_membership(self, candidate_global_ids: np.ndarray) -> Tuple[bool, int, int]:
        """
        Validates that all candidate global IDs belong strictly to the authoritative COLD membership.
        Returns (all_valid, valid_count, total_count).
        """
        flat_ids = candidate_global_ids.flatten()
        total_count = len(flat_ids)
        valid_mask = [gid in self.cold_membership_set for gid in flat_ids]
        valid_count = sum(valid_mask)
        return (valid_count == total_count, valid_count, total_count)

    def evaluate_cold_recall(
        self,
        candidate_ids: np.ndarray,
        ground_truth_ids: np.ndarray,
        k_values: List[int] = [1, 5, 10],
    ) -> Dict[int, float]:
        """
        Calculates exact COLD Recall@K between candidate IDs and COLD ground truth IDs.
        Both arrays should share the same ID space (either both local or both global).
        """
        recalls = {}
        n_queries = candidate_ids.shape[0]

        for k in k_values:
            matches = 0
            total_possible = n_queries * k

            for q_idx in range(n_queries):
                retrieved_k = set(candidate_ids[q_idx, :k])
                truth_k = set(ground_truth_ids[q_idx, :k])
                matches += len(retrieved_k.intersection(truth_k))

            recalls[k] = matches / total_possible if total_possible > 0 else 0.0

        return recalls

    def evaluate_global_coverage(
        self,
        candidate_global_ids: np.ndarray,
        global_ground_truth_ids: np.ndarray,
        k_values: List[int] = [1, 5, 10],
    ) -> Dict[int, float]:
        """
        Calculates Global Candidate Coverage: fraction of true global top-K neighbors
        (from data/ground_truth.ivecs) captured in the candidate set retrieved by COLD search.
        """
        coverages = {}
        n_queries = candidate_global_ids.shape[0]

        for k in k_values:
            captured = 0
            total_global_k = n_queries * k

            for q_idx in range(n_queries):
                candidates = set(candidate_global_ids[q_idx])
                global_top_k = set(global_ground_truth_ids[q_idx, :k])
                captured += len(candidates.intersection(global_top_k))

            coverages[k] = captured / total_global_k if total_global_k > 0 else 0.0

        return coverages


if __name__ == "__main__":
    print("=" * 78)
    print("PHASE 5: DISKANN3 ADAPTER SMOKE TEST & LATENCY BASELINE (ITEM 0)")
    print("=" * 78)

    print(f"[Config] Resolved Executable: {get_diskann_query_bin()}")
    print(f"[Config] Resolved Index Prefix: {resolve_diskann_index_prefix()}")

    # Initialize adapter
    idx = ColdDiskANNIndex()
    print("[Config] ColdDiskANNIndex initialized successfully.")

    # Load 10 test queries
    queries_path = workspace_root / "data/query_vectors.fvecs"
    queries = load_fvecs(queries_path)[:10]
    print(f"[Search] Executing 10-query search with k=10, L=100, beam_width=4...")

    t0_wall = time.perf_counter()
    res = idx.search(queries, k=10, l_search=100, beam_width=4)
    t_wall_ms = (time.perf_counter() - t0_wall) * 1000.0

    print(f"[Result] Output shapes: local_ids={res.local_ids.shape}, global_ids={res.global_ids.shape}, distances={res.distances.shape}")
    print(f"[Result] Internal engine search time: {res.total_elapsed_ms:.3f} ms (QPS: {res.qps:.1f})")
    print(f"[Result] End-to-end Python wall-clock time: {t_wall_ms:.3f} ms (includes process startup & WSL overhead)")
    print(f"[Command] Executed command:\n  {' '.join(idx.last_command)}")

    # Check sanitation
    cmd_str = " ".join(idx.last_command)
    has_personal_home = "/home/shivamgupta" in cmd_str
    has_personal_win = "shivam.gupta" in cmd_str
    print(f"[Sanitation] Contains '/home/shivamgupta': {has_personal_home}")
    print(f"[Sanitation] Contains 'shivam.gupta': {has_personal_win}")
    assert not has_personal_home, "Sanitation violation: command contains /home/shivamgupta"
    assert not has_personal_win, "Sanitation violation: command contains shivam.gupta"

    # Validate output
    assert res.num_queries == 10
    assert res.global_ids.shape == (10, 10)
    assert res.distances.shape == (10, 10)
    assert np.all(np.isfinite(res.distances))
    for q_idx in range(10):
        assert len(set(res.global_ids[q_idx])) == 10, f"Query {q_idx} has duplicate IDs!"

    valid, v_cnt, t_cnt = idx.validate_cold_membership(res.global_ids)
    print(f"[Validation] COLD membership: {v_cnt}/{t_cnt} valid ({valid})")
    assert valid, "COLD membership validation failed!"
    print("=" * 78)
    print("ITEM 0 SMOKE TEST PASSED.")
    print("=" * 78)
