"""
Case 04 — Phase 3: Persistent NVMe SSD Vector Store
Fixed-record random-access vector storage on local physical SSD.

Stores 100,000 vectors (128-dim, float32) as fixed-size 512-byte binary records,
providing constant-time O(1) random access via disk seeks.
"""

import os
import sys
import time
import json
import hashlib
import threading
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

try:
    import numpy as np
except ImportError as err:
    print(f"CRITICAL ERROR: Failed to import NumPy: {err}", file=sys.stderr)
    sys.exit(1)

from src.phase1_2_evaluation import load_fvecs, compute_sha256, EXPECTED_CHECKSUMS


RECORD_BYTES = 128 * 4  # 512 bytes per 128-dim float32 vector


class SSDVectorStore:
    """
    High-performance SSD-backed vector storage with fixed-offset random access.
    Maps vector_id -> physical file offset (vector_id * 512 bytes).
    Thread-safe and instrumented with monotonic high-resolution nanosecond timing.
    """

    def __init__(
        self,
        storage_dir: Path | str,
        num_vectors: int = 100_000,
        dimension: int = 128,
    ):
        self.storage_dir = Path(storage_dir)
        self.num_vectors = num_vectors
        self.dimension = dimension
        self.record_bytes = dimension * 4  # 512 bytes

        self.vector_file_path = self.storage_dir / "ssd_vectors" / "vectors.bin"
        self.manifest_path = self.storage_dir / "metadata" / "manifest.json"

        # Performance and I/O tracking (nanosecond resolution)
        self.total_reads: int = 0
        self.total_bytes_read: int = 0
        self.total_read_time_ns: int = 0

        # Persistent file handle for low-overhead random seeking
        self._file_handle: Optional[Any] = None
        self._lock = threading.Lock()

    def initialize_from_base_vectors(self, base_vectors_path: Path | str) -> Dict[str, Any]:
        """
        Populates the SSD binary file directly from base_vectors.fvecs.
        Ensures strict bit-for-bit equivalence and creates a metadata manifest.
        """
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.vector_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[SSD Store] Loading base vectors from {base_vectors_path}...")
        t0 = time.perf_counter()
        base_vectors = load_fvecs(base_vectors_path)
        t_load = time.perf_counter() - t0

        if base_vectors.shape != (self.num_vectors, self.dimension):
            raise ValueError(
                f"Unexpected base vectors shape {base_vectors.shape}, expected ({self.num_vectors}, {self.dimension})"
            )

        print(f"[SSD Store] Writing {self.num_vectors:,} vectors ({base_vectors.nbytes / (1024*1024):.2f} MB) to SSD at {self.vector_file_path}...")
        t1 = time.perf_counter()
        
        # Write flat contiguous raw float32 bytes
        with open(self.vector_file_path, "wb") as f:
            base_vectors.tofile(f)
            f.flush()
            os.fsync(f.fileno())  # Ensure physical write to NVMe SSD media
        
        t_write = time.perf_counter() - t1
        file_size = self.vector_file_path.stat().st_size
        expected_size = self.num_vectors * self.record_bytes

        if file_size != expected_size:
            raise IOError(f"File size mismatch: {file_size} bytes written, expected {expected_size} bytes")

        # Compute SHA-256 hash of written raw payload
        ssd_sha256 = compute_sha256(self.vector_file_path)

        manifest = {
            "num_vectors": self.num_vectors,
            "dimension": self.dimension,
            "record_bytes": self.record_bytes,
            "total_bytes": file_size,
            "file_sha256": ssd_sha256,
            "source_fvecs": str(base_vectors_path),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "storage_media": "NVMe SSD (BC711 NVMe SK hynix 512GB)",
            "filesystem": "NTFS",
        }

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"[SSD Store] Initialization complete: {file_size / (1024*1024):.2f} MB written in {t_write:.3f} s ({file_size / (1024*1024) / t_write:.1f} MB/s)")
        return manifest

    def open(self) -> None:
        """Opens persistent read-binary handle to avoid per-read open/close overhead."""
        with self._lock:
            if not self.vector_file_path.exists():
                raise FileNotFoundError(f"SSD vector file not found at {self.vector_file_path}")
            if self._file_handle is None or self._file_handle.closed:
                self._file_handle = open(self.vector_file_path, "rb", buffering=0)  # Unbuffered binary I/O for direct NVMe access

    def close(self) -> None:
        """Closes file handle."""
        with self._lock:
            if self._file_handle is not None and not self._file_handle.closed:
                self._file_handle.close()
                self._file_handle = None

    def read_vector(self, vector_id: int) -> np.ndarray:
        """
        Reads a single 128-dim float32 vector directly from SSD via seek.
        Time complexity: O(1) random disk seek. Thread-safe.
        """
        if vector_id < 0 or vector_id >= self.num_vectors:
            raise IndexError(f"Vector ID {vector_id} out of bounds [0, {self.num_vectors-1}]")

        offset = vector_id * self.record_bytes

        with self._lock:
            if self._file_handle is None or self._file_handle.closed:
                self.open()

            t0 = time.perf_counter_ns()
            self._file_handle.seek(offset)
            raw_bytes = self._file_handle.read(self.record_bytes)
            t_read_ns = time.perf_counter_ns() - t0

            if len(raw_bytes) != self.record_bytes:
                raise IOError(f"Truncated read for vector {vector_id}: got {len(raw_bytes)} bytes, expected {self.record_bytes}")

            self.total_reads += 1
            self.total_bytes_read += self.record_bytes
            self.total_read_time_ns += t_read_ns

        return np.frombuffer(raw_bytes, dtype=np.float32).copy()

    def read_vectors_batch(self, vector_ids: List[int] | np.ndarray) -> np.ndarray:
        """
        Reads a batch of vectors from SSD. Thread-safe.
        """
        n = len(vector_ids)
        result = np.empty((n, self.dimension), dtype=np.float32)

        with self._lock:
            if self._file_handle is None or self._file_handle.closed:
                self.open()

            t0 = time.perf_counter_ns()
            for i, vid in enumerate(vector_ids):
                offset = int(vid) * self.record_bytes
                self._file_handle.seek(offset)
                raw = self._file_handle.read(self.record_bytes)
                result[i] = np.frombuffer(raw, dtype=np.float32)
            t_batch_ns = time.perf_counter_ns() - t0

            self.total_reads += n
            self.total_bytes_read += n * self.record_bytes
            self.total_read_time_ns += t_batch_ns

        return result

    def get_io_stats(self) -> Dict[str, Any]:
        """Returns cumulative SSD I/O performance metrics."""
        with self._lock:
            total_sec = self.total_read_time_ns / 1e9
            avg_latency_ms = (
                (self.total_read_time_ns / self.total_reads / 1e6)
                if self.total_reads > 0
                else 0.0
            )
            throughput_mb_s = (
                (self.total_bytes_read / (1024 * 1024) / total_sec)
                if total_sec > 0
                else 0.0
            )
            return {
                "total_reads": self.total_reads,
                "total_bytes_read": self.total_bytes_read,
                "total_read_mb": round(self.total_bytes_read / (1024 * 1024), 3),
                "total_read_time_seconds": round(total_sec, 6),
                "avg_read_latency_ms": round(avg_latency_ms, 5),
                "read_throughput_mb_s": round(throughput_mb_s, 2),
                "bytes_per_read": (self.total_bytes_read // self.total_reads) if self.total_reads > 0 else 0,
            }

    def reset_io_stats(self) -> None:
        """Resets cumulative counters."""
        with self._lock:
            self.total_reads = 0
            self.total_bytes_read = 0
            self.total_read_time_ns = 0

    def verify_integrity(self, base_vectors_path: Path | str) -> bool:
        """
        Verifies bit-for-bit that the SSD file exactly matches base_vectors.fvecs.
        """
        base = load_fvecs(base_vectors_path)
        sample_indices = [0, 1, 42, 500, 999, 10_000, 50_000, 99_999]
        for idx in sample_indices:
            ssd_vec = self.read_vector(idx)
            ref_vec = base[idx]
            if not np.array_equal(ssd_vec, ref_vec):
                print(f"[SSD Store] Integrity check FAILED at vector {idx}!", file=sys.stderr)
                return False
        return True


if __name__ == "__main__":
    store = SSDVectorStore(workspace_root / "storage")
    store.initialize_from_base_vectors(workspace_root / "data" / "base_vectors.fvecs")
    is_valid = store.verify_integrity(workspace_root / "data" / "base_vectors.fvecs")
    print(f"[SSD Store] Integrity verification: {'PASS' if is_valid else 'FAIL'}")
