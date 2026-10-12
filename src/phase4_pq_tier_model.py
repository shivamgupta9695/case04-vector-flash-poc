"""
Case 04 — Phase 4: HOT/COLD Logical Data Model & Storage Layout

Prepares the vector identity and tiered representation required by the target PPT architecture:
    HOT TIER  (DRAM): HNSW, HOT vectors, PQ codes, PQ codebooks, Tier metadata
    COLD TIER (NVMe): DiskANN (Phase 5), COLD full vectors, PQ-assisted search

Key Architectural Invariants:
1. Vector ID is immutable and stable across all tier migrations.
2. HOT / COLD membership is represented as logical metadata, NOT hardcoded permanent ID ranges.
3. Every vector belongs to exactly one tier at any given moment (disjoint and exhaustive partition).
4. PQ representation resides in DRAM for all vectors (compressed search representation).
5. Full-precision vectors are preserved on NVMe SSD for exact reranking.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple
import hashlib

# Ensure workspace root in path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))


class VectorTier(str, Enum):
    HOT = "HOT"
    COLD = "COLD"


@dataclass
class VectorRecord:
    """
    Conceptual vector identity record.
    Decouples logical vector identity from physical storage tier.
    """
    vector_id: int
    tier: str  # "HOT" or "COLD"
    full_vector_uri: str  # Storage URI / file path
    byte_offset: int  # Byte offset in NVMe binary store
    pq_code_index: int  # Index into DRAM PQ-code array (0-based)
    hotness: float = 0.0  # Dynamic access frequency / hotness score
    graph_metadata: Optional[Dict[str, Any]] = field(default_factory=dict)  # Placeholder for DiskANN/HNSW node info

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TierModel:
    """
    Manages the logical HOT/COLD tier partitioning, membership tracking,
    and storage layout metadata.

    Capacity model:
        Total vectors = 100,000
        Initial HOT ratio = 0.18 (18,000 vectors)
        Initial COLD ratio = 0.82 (82,000 vectors)
    """

    def __init__(
        self,
        total_vectors: int = 100_000,
        dimension: int = 128,
        hot_ratio: float = 0.18,
        nvme_store_path: Optional[str | Path] = None,
        dram_pq_codes_path: Optional[str | Path] = None,
    ):
        self.total_vectors = total_vectors
        self.dimension = dimension
        self.hot_ratio = hot_ratio
        self.expected_hot_count = int(round(total_vectors * hot_ratio))
        self.expected_cold_count = total_vectors - self.expected_hot_count

        self.nvme_store_path = str(nvme_store_path) if nvme_store_path else "storage/ssd_vectors/vectors.bin"
        self.dram_pq_codes_path = str(dram_pq_codes_path) if dram_pq_codes_path else "results/phase4_pq/codes/base_pq_codes.bin"

        # Explicit membership sets
        self._hot_ids: Set[int] = set()
        self._cold_ids: Set[int] = set()
        self._vector_tiers: Dict[int, VectorTier] = {}

        self._initialize_default_partition()

    def _initialize_default_partition(self) -> None:
        """
        Initializes the baseline deterministic partition:
        First 18,000 vectors initialized as HOT (matching Phase 3 baseline),
        Remaining 82,000 vectors initialized as COLD.
        Note: This is an INITIAL state; tier membership is dynamic.
        """
        self._hot_ids = set(range(self.expected_hot_count))
        self._cold_ids = set(range(self.expected_hot_count, self.total_vectors))
        self._vector_tiers = {}

        for vid in self._hot_ids:
            self._vector_tiers[vid] = VectorTier.HOT
        for vid in self._cold_ids:
            self._vector_tiers[vid] = VectorTier.COLD

        self.validate_invariants()

    @property
    def hot_count(self) -> int:
        return len(self._hot_ids)

    @property
    def cold_count(self) -> int:
        return len(self._cold_ids)

    @property
    def hot_ids(self) -> Set[int]:
        return set(self._hot_ids)

    @property
    def cold_ids(self) -> Set[int]:
        return set(self._cold_ids)

    def get_tier(self, vector_id: int) -> VectorTier:
        if vector_id not in self._vector_tiers:
            raise KeyError(f"Vector ID {vector_id} not found in tier model (valid 0..{self.total_vectors-1})")
        return self._vector_tiers[vector_id]

    def is_hot(self, vector_id: int) -> bool:
        return self.get_tier(vector_id) == VectorTier.HOT

    def is_cold(self, vector_id: int) -> bool:
        return self.get_tier(vector_id) == VectorTier.COLD

    def get_full_vector_offset(self, vector_id: int) -> int:
        """
        Computes byte offset for full-precision FP32 vector in NVMe storage.
        Each vector is 128 floats * 4 bytes = 512 bytes.
        """
        if not (0 <= vector_id < self.total_vectors):
            raise IndexError(f"Vector ID {vector_id} out of bounds [0, {self.total_vectors})")
        return vector_id * self.dimension * 4

    def get_record(self, vector_id: int, hotness: float = 0.0) -> VectorRecord:
        """Creates a VectorRecord for the given vector_id."""
        tier = self.get_tier(vector_id)
        return VectorRecord(
            vector_id=vector_id,
            tier=tier.value,
            full_vector_uri=self.nvme_store_path,
            byte_offset=self.get_full_vector_offset(vector_id),
            pq_code_index=vector_id,  # 1:1 mapped in DRAM PQ codes array
            hotness=hotness,
            graph_metadata={"phase5_diskann_node": None if tier == VectorTier.HOT else True},
        )

    def promote(self, vector_id: int) -> None:
        """Promotes vector from COLD to HOT tier."""
        if vector_id not in self._cold_ids:
            if vector_id in self._hot_ids:
                return  # Already hot, idempotent
            raise KeyError(f"Vector ID {vector_id} not in cold tier")

        self._cold_ids.remove(vector_id)
        self._hot_ids.add(vector_id)
        self._vector_tiers[vector_id] = VectorTier.HOT

    def demote(self, vector_id: int) -> None:
        """Demotes vector from HOT to COLD tier."""
        if vector_id not in self._hot_ids:
            if vector_id in self._cold_ids:
                return  # Already cold, idempotent
            raise KeyError(f"Vector ID {vector_id} not in hot tier")

        self._hot_ids.remove(vector_id)
        self._cold_ids.add(vector_id)
        self._vector_tiers[vector_id] = VectorTier.COLD

    def validate_invariants(self) -> Dict[str, Any]:
        """
        Validates all architectural invariants:
        1. HOT count + COLD count == total_vectors
        2. HOT and COLD are disjoint (HOT ∩ COLD = ∅)
        3. HOT ∪ COLD = {0..total_vectors-1}
        4. Every vector appears exactly once
        5. IDs are valid integers within [0, total_vectors-1]
        """
        hot_len = len(self._hot_ids)
        cold_len = len(self._cold_ids)
        total_len = hot_len + cold_len

        overlap = self._hot_ids.intersection(self._cold_ids)
        is_disjoint = len(overlap) == 0

        all_ids = set(range(self.total_vectors))
        union_ids = self._hot_ids.union(self._cold_ids)
        is_exhaustive = union_ids == all_ids

        all_valid_ranges = (
            min(union_ids) == 0 and
            max(union_ids) == self.total_vectors - 1 and
            len(union_ids) == self.total_vectors
        )

        invariants_pass = (
            total_len == self.total_vectors and
            is_disjoint and
            is_exhaustive and
            all_valid_ranges
        )

        if not invariants_pass:
            raise ValueError(
                f"Tier invariants violated! Total={total_len}/{self.total_vectors}, "
                f"Disjoint={is_disjoint} (overlap={len(overlap)}), "
                f"Exhaustive={is_exhaustive}"
            )

        return {
            "is_valid": invariants_pass,
            "total_vectors": self.total_vectors,
            "hot_count": hot_len,
            "cold_count": cold_len,
            "hot_ratio": hot_len / self.total_vectors,
            "cold_ratio": cold_len / self.total_vectors,
            "is_disjoint": is_disjoint,
            "is_exhaustive": is_exhaustive,
            "overlap_count": len(overlap),
        }

    def export_cold_vector_manifest(self) -> Dict[str, Any]:
        """
        Generates the Phase 5 handoff manifest for the COLD tier.
        Phase 5 (DiskANN) needs:
            - Exact list of COLD vector IDs
            - Byte offsets in NVMe full-vector binary storage
            - Storage format & dimensions
        """
        cold_ids_sorted = sorted(list(self._cold_ids))
        return {
            "cold_vector_count": len(cold_ids_sorted),
            "storage_media": "NVMe SSD",
            "nvme_file": self.nvme_store_path,
            "vector_dimension": self.dimension,
            "dtype": "float32",
            "vector_bytes": self.dimension * 4,
            "cold_vector_ids": cold_ids_sorted,
            "first_cold_id": cold_ids_sorted[0] if cold_ids_sorted else None,
            "last_cold_id": cold_ids_sorted[-1] if cold_ids_sorted else None,
            "storage_offset_rule": "byte_offset = vector_id * 128 * 4",
        }

    def export_manifest(self) -> Dict[str, Any]:
        """Exports complete tier model metadata."""
        validation = self.validate_invariants()
        hot_ids_sorted = sorted(list(self._hot_ids))
        cold_ids_sorted = sorted(list(self._cold_ids))

        # Compute deterministic hashes of membership for reproducibility checks
        hot_hash = hashlib.sha256(",".join(map(str, hot_ids_sorted)).encode("utf-8")).hexdigest()
        cold_hash = hashlib.sha256(",".join(map(str, cold_ids_sorted)).encode("utf-8")).hexdigest()

        return {
            "model_name": "Phase 4 Hot/Cold Logical Tier Model",
            "capacity_config": {
                "total_vectors": self.total_vectors,
                "dimension": self.dimension,
                "hot_ratio": self.hot_ratio,
                "expected_hot_count": self.expected_hot_count,
                "expected_cold_count": self.expected_cold_count,
            },
            "current_state": {
                "hot_count": len(hot_ids_sorted),
                "cold_count": len(cold_ids_sorted),
                "hot_membership_sha256": hot_hash,
                "cold_membership_sha256": cold_hash,
                "hot_id_range": [hot_ids_sorted[0], hot_ids_sorted[-1]] if hot_ids_sorted else None,
                "cold_id_range": [cold_ids_sorted[0], cold_ids_sorted[-1]] if cold_ids_sorted else None,
            },
            "storage_layout": {
                "dram_resident": {
                    "pq_codes": {
                        "path": self.dram_pq_codes_path,
                        "vector_count": self.total_vectors,
                        "size_bytes": self.total_vectors * 16,
                        "role": "Search-tier compressed representation for both HOT and COLD vectors",
                    },
                    "pq_codebooks": {
                        "path": "results/phase4_pq/codebooks/pq_codebooks.npy",
                        "size_bytes": 16 * 256 * 8 * 4,
                        "role": "In-memory LUT generation for ADC queries",
                    },
                    "hot_full_vectors": {
                        "count": len(hot_ids_sorted),
                        "role": "DRAM-resident full-precision vectors for HOT tier",
                    },
                },
                "nvme_resident": {
                    "cold_full_vectors": {
                        "path": self.nvme_store_path,
                        "count": len(cold_ids_sorted),
                        "size_bytes": len(cold_ids_sorted) * self.dimension * 4,
                        "role": "Full-precision vectors on NVMe for exact reranking",
                    },
                    "future_phase5_diskann": {
                        "role": "DiskANN graph structure to be created in Phase 5 on NVMe",
                    },
                },
            },
            "validation": validation,
        }

    def save_artifacts(self, metadata_dir: Path | str) -> Dict[str, Path]:
        """Saves tier manifests to disk."""
        meta_dir = Path(metadata_dir)
        meta_dir.mkdir(parents=True, exist_ok=True)

        manifest = self.export_manifest()
        manifest_path = meta_dir / "tier_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        cold_manifest = self.export_cold_vector_manifest()
        cold_manifest_path = meta_dir / "cold_vector_manifest.json"
        with open(cold_manifest_path, "w", encoding="utf-8") as f:
            json.dump(cold_manifest, f, indent=2)

        # Sample catalog records for inspection
        sample_ids = [0, 1, 17_999, 18_000, 50_000, 99_999]
        sample_catalog = [self.get_record(vid).to_dict() for vid in sample_ids]
        sample_catalog_path = meta_dir / "vector_catalog_sample.json"
        with open(sample_catalog_path, "w", encoding="utf-8") as f:
            json.dump(sample_catalog, f, indent=2)

        return {
            "tier_manifest": manifest_path,
            "cold_vector_manifest": cold_manifest_path,
            "vector_catalog_sample": sample_catalog_path,
        }
