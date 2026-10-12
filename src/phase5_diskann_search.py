"""
Case 04 — Phase 5: Official DiskANN3 Python Integration Adapter

Wraps the official Microsoft DiskANN3 disk index search engine. Provides query
execution, local <-> global ID translation, COLD membership validation,
COLD Recall@K evaluation against exact FAISS oracle, and Global Candidate Coverage.
"""

from __future__ import annotations

import json
import os
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


DISKANN_REPO_PATH_WSL = "/home/shivamgupta/diskann-work/DiskANN"
DISKANN_QUERY_BIN_WSL = f"{DISKANN_REPO_PATH_WSL}/target/release/diskann_query"


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
    total_elapsed_ms: f64
    qps: f64


class ColdDiskANNIndex:
    """
    Python integration adapter for Microsoft DiskANN3 disk-backed index.
    """

    def __init__(
        self,
        index_prefix: Path | str,
        id_map_path: Path | str | None = None,
        cold_manifest_path: Path | str | None = None,
        num_threads: int = 8,
        cache_nodes: int = 2000,
    ):
        self.index_prefix = Path(index_prefix).resolve()
        self.num_threads = num_threads
        self.cache_nodes = cache_nodes
        self.is_open = False
        
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
        """Converts Windows Path to WSL path (/mnt/c/...)."""
        p = str(Path(path).resolve()).replace("\\", "/")
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
                raise FileNotFoundError(f"Missing required DiskANN artifact: {desc} at {f}")

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

            cmd = [
                "wsl",
                DISKANN_QUERY_BIN_WSL,
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

            t0 = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
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
    # Smoke test of Python adapter
    index_path = workspace_root / "results/phase5_diskann/index/cold_disk_index_tuned"
    print(f"[Adapter Test] Initializing ColdDiskANNIndex from {index_path}...")
    idx = ColdDiskANNIndex(index_path)
    
    # Load test queries
    queries = load_fvecs(workspace_root / "data/query_vectors.fvecs")[:5]
    print(f"[Adapter Test] Executing test search for 5 queries with k=10, L=100...")
    res = idx.search(queries, k=10, l_search=100)
    print(f"[Adapter Test] Returned {res.num_queries} queries in {res.total_elapsed_ms:.2f}ms (QPS: {res.qps:.1f})")
    print(f"[Adapter Test] Query 0 global IDs: {res.global_ids[0].tolist()}")
    print(f"[Adapter Test] Query 0 distances: {res.distances[0].tolist()}")
    
    valid, v_cnt, t_cnt = idx.validate_cold_membership(res.global_ids)
    print(f"[Adapter Test] COLD membership validation: {v_cnt}/{t_cnt} valid ({valid})")
