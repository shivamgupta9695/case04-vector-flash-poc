"""
Case 04 — Phase 4: Product Quantization (PQ) Vector Encoding

Encodes all 100,000 frozen base vectors into uint8 PQ codes:
    Base Vectors: [100000, 128] float32 (512 bytes / vector)
    PQ Codes:     [100000, 16]  uint8   (16 bytes / vector)

Guarantees:
1. Strict 1:1 alignment: vector ID i corresponds exactly to code[i].
2. Every code is an integer in [0, 255].
3. Output format is documented and cryptographically verified.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

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

from src.phase1_2_evaluation import compute_sha256
from src.phase4_pq_train import PQConfig, load_trained_codebooks


class PQEncoder:
    """Encodes full-precision vectors into uint8 PQ codes."""

    def __init__(self, pq_quantizer: faiss.ProductQuantizer, config: Optional[PQConfig] = None):
        self.pq = pq_quantizer
        self.config = config or PQConfig()

    @classmethod
    def from_codebooks_dir(cls, codebooks_dir: Path | str) -> PQEncoder:
        cb_dir = Path(codebooks_dir)
        faiss_bin = cb_dir / "faiss_pq_quantizer.bin"
        if not faiss_bin.exists():
            raise FileNotFoundError(f"FAISS PQ quantizer not found: {faiss_bin}")
        pq = faiss.read_ProductQuantizer(str(faiss_bin))
        return cls(pq)

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encodes vectors into uint8 PQ codes of shape (N, M).
        """
        if not isinstance(vectors, np.ndarray):
            raise TypeError(f"Expected numpy ndarray, got {type(vectors)}")
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        n, d = vectors.shape
        if d != self.config.dimension:
            raise ValueError(f"Vector dimension {d} does not match config {self.config.dimension}")

        t0 = time.perf_counter()
        codes = self.pq.compute_codes(vectors)
        encoding_time = time.perf_counter() - t0

        # Validate shape and dtype
        expected_shape = (n, self.config.num_subquantizers)
        if codes.shape != expected_shape:
            raise ValueError(f"Codes shape {codes.shape} does not match expected {expected_shape}")
        if codes.dtype != np.uint8:
            raise ValueError(f"Codes dtype {codes.dtype} is not uint8")

        # Validate range [0, 255]
        if codes.min() < 0 or codes.max() > 255:
            raise ValueError(f"Code values out of bounds [0, 255]: min={codes.min()}, max={codes.max()}")

        return codes


def validate_pq_codes(codes: np.ndarray, expected_n: int = 100_000, expected_m: int = 16) -> Dict[str, Any]:
    """Validates structural and numeric integrity of PQ codes."""
    if not isinstance(codes, np.ndarray):
        return {"is_valid": False, "error": f"Expected numpy ndarray, got {type(codes)}"}

    if codes.shape != (expected_n, expected_m):
        return {
            "is_valid": False,
            "error": f"Shape mismatch: got {codes.shape}, expected ({expected_n}, {expected_m})",
        }

    if codes.dtype != np.uint8:
        return {"is_valid": False, "error": f"dtype mismatch: got {codes.dtype}, expected uint8"}

    min_val = int(codes.min())
    max_val = int(codes.max())

    is_range_valid = (0 <= min_val) and (max_val <= 255)
    if not is_range_valid:
        return {
            "is_valid": False,
            "error": f"Code values outside [0, 255]: min={min_val}, max={max_val}",
        }

    return {
        "is_valid": True,
        "num_vectors": expected_n,
        "num_subquantizers": expected_m,
        "dtype": "uint8",
        "min_code": min_val,
        "max_code": max_val,
        "total_bytes": int(codes.nbytes),
    }


def save_pq_codes(codes: np.ndarray, output_dir: Path | str, metadata_extra: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    """
    Saves PQ codes in both raw binary (.bin) and NumPy (.npy) formats along with metadata.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_report = validate_pq_codes(codes)
    if not val_report["is_valid"]:
        raise ValueError(f"Cannot save invalid PQ codes: {val_report.get('error')}")

    # 1. Save raw binary format (flat uint8 bytes: 100,000 * 16 = 1,600,000 bytes)
    bin_path = out_dir / "base_pq_codes.bin"
    with open(bin_path, "wb") as f:
        f.write(codes.tobytes())

    # 2. Save numpy format
    npy_path = out_dir / "base_pq_codes.npy"
    np.save(npy_path, codes)

    bin_hash = compute_sha256(bin_path)
    npy_hash = compute_sha256(npy_path)

    # 3. Save metadata JSON
    meta = {
        "component": "Phase 4 Base Vectors PQ Codes",
        "shape": list(codes.shape),
        "dtype": str(codes.dtype),
        "code_bytes_per_vector": codes.shape[1],
        "total_vectors": codes.shape[0],
        "total_code_bytes": int(codes.nbytes),
        "artifacts": {
            "base_pq_codes_bin": {
                "file": bin_path.name,
                "sha256": bin_hash,
                "bytes": bin_path.stat().st_size,
                "format": "Raw uint8 flat array, shape (N, M), row-major (C-contiguous)",
            },
            "base_pq_codes_npy": {
                "file": npy_path.name,
                "sha256": npy_hash,
                "bytes": npy_path.stat().st_size,
            },
        },
        "validation": val_report,
    }
    if metadata_extra:
        meta["extra"] = metadata_extra

    meta_path = out_dir / "encoding_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "codes_bin": bin_path,
        "codes_npy": npy_path,
        "metadata_json": meta_path,
    }


def load_pq_codes(codes_dir: Path | str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Loads PQ codes and metadata from disk with checksum verification."""
    c_dir = Path(codes_dir)
    npy_path = c_dir / "base_pq_codes.npy"
    meta_path = c_dir / "encoding_metadata.json"

    if not npy_path.exists():
        raise FileNotFoundError(f"Codes file not found: {npy_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    codes = np.load(npy_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    expected_hash = meta["artifacts"]["base_pq_codes_npy"]["sha256"]
    actual_hash = compute_sha256(npy_path)
    if actual_hash != expected_hash:
        raise ValueError(f"Codes hash mismatch! Expected {expected_hash}, got {actual_hash}")

    return codes, meta
