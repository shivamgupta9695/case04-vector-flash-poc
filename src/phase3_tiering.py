"""
Case 04 — Phase 3: Tiered Vector Manager & DRAM Cache
Dynamic DRAM/SSD Tier Management, Promotion, Demotion, and Eviction Policies.

Supports Policy A (legacy baseline) and Policy B (production-hardened with
admission control, hysteresis, minimum residency, cooldown, and rate limiting).
Enforces hard capacity invariants and thread-safe operations.
"""

import sys
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

try:
    import numpy as np
except ImportError as err:
    print(f"CRITICAL ERROR: Failed to import NumPy: {err}", file=sys.stderr)
    sys.exit(1)

from src.phase3_ssd_store import SSDVectorStore
from src.phase3_hotness import AccessTracker, TIER_COLD_SSD, TIER_HOT_DRAM


class TieredVectorManager:
    """
    Manages hybrid DRAM cache and NVMe SSD vector retrieval with dynamic hot/cold migration.
    Thread-safe and supports Policy A vs Policy B execution.
    """

    def __init__(
        self,
        ssd_store: SSDVectorStore,
        access_tracker: AccessTracker,
        dram_capacity_ratio: float = 0.18,  # Default target 18% DRAM
        dimension: int = 128,
        policy_type: str = "policy_b",      # "policy_a" (legacy) or "policy_b" (hardened)
        max_dram_vectors: Optional[int] = None,
        # Policy B hardening parameters:
        admission_min_access: int = 2,
        admission_min_hotness: float = 1.0,
        promotion_threshold: float = 1.1,
        demotion_threshold: float = 0.5,
        hysteresis_margin: float = 0.20,
        min_residency_time: float = 0.05,
        cooldown_time: float = 0.05,
        max_migrations_per_batch: int = 10,
    ):
        self.ssd_store = ssd_store
        self.tracker = access_tracker
        self.dram_capacity_ratio = dram_capacity_ratio
        self.dimension = dimension
        self.num_vectors = ssd_store.num_vectors
        self.policy_type = policy_type.lower()

        # Policy B hardening hyperparameters
        self.admission_min_access = admission_min_access
        self.admission_min_hotness = admission_min_hotness
        self.promotion_threshold = promotion_threshold
        self.demotion_threshold = demotion_threshold
        self.hysteresis_margin = hysteresis_margin
        self.min_residency_time = min_residency_time
        self.cooldown_time = cooldown_time
        self.max_migrations_per_batch = max_migrations_per_batch

        # Capacity calculation: strict hard limit
        if max_dram_vectors is not None:
            self.max_dram_vectors = max(1, int(max_dram_vectors))
        else:
            self.max_dram_vectors = max(1, int(self.num_vectors * dram_capacity_ratio))

        # DRAM Cache: vector_id -> float32 vector array
        self.dram_cache: Dict[int, np.ndarray] = {}
        self._lock = threading.RLock()

        # Performance & I/O Counters (nanosecond resolution)
        self.total_queries_served: int = 0
        self.dram_hits: int = 0
        self.ssd_reads: int = 0
        self.promotion_count: int = 0
        self.demotion_count: int = 0
        self.promotion_time_ns: int = 0
        self.demotion_time_ns: int = 0
        self.admission_rejections: int = 0
        self.cooldown_rejections: int = 0
        self.rate_limit_triggered_count: int = 0

    @property
    def dram_bytes_used(self) -> int:
        """Memory footprint of vector payloads currently resident in DRAM."""
        with self._lock:
            return len(self.dram_cache) * self.dimension * 4

    @property
    def ssd_bytes_used(self) -> int:
        """Physical footprint of vector payloads stored on SSD."""
        return self.num_vectors * self.dimension * 4

    def populate_initial_dram(self, initial_vector_ids: List[int] | np.ndarray) -> None:
        """
        Populates initial DRAM tier up to capacity limit.
        Initializes metadata in AccessTracker to eliminate cold-start asymmetry.
        """
        with self._lock:
            self.dram_cache.clear()
            initial_set = list(initial_vector_ids)[: self.max_dram_vectors]
            print(f"[Tier Manager] Initializing DRAM cache with {len(initial_set):,} vectors ({len(initial_set) * self.dimension * 4 / (1024*1024):.2f} MB)...")
            
            vectors = self.ssd_store.read_vectors_batch(initial_set)
            for vid, vec in zip(initial_set, vectors):
                vid = int(vid)
                self.dram_cache[vid] = vec
                self.tracker.set_tier(vid, TIER_HOT_DRAM, current_time=0.0)

            # Assert strict invariant
            assert len(self.dram_cache) <= self.max_dram_vectors, (
                f"Capacity violation: {len(self.dram_cache)} > {self.max_dram_vectors}"
            )

    def get_vector(self, vector_id: int, current_time: float) -> Tuple[np.ndarray, bool]:
        """
        Retrieves vector payload from either DRAM (hit) or SSD (miss).
        Returns: (vector_array, is_dram_hit).
        """
        vid = int(vector_id)
        with self._lock:
            if vid in self.dram_cache:
                self.dram_hits += 1
                self.tracker.record_access(vid, current_time)
                return self.dram_cache[vid], True

            # SSD Read (Cache Miss)
            self.ssd_reads += 1
            vec = self.ssd_store.read_vector(vid)
            new_score = self.tracker.record_access(vid, current_time)

            # Evaluate Dynamic Promotion
            self._evaluate_promotion(vid, vec, new_score, current_time)

            return vec, False

    def get_vectors_for_candidates(
        self, candidate_ids: List[int] | np.ndarray, current_time: float
    ) -> Tuple[np.ndarray, int, int]:
        """
        Retrieves vector representations for an ANN candidate set.
        Batches SSD reads for cold candidates and executes tier management.
        Returns: (vectors_array, dram_hits, ssd_reads).
        """
        candidate_ids = [int(x) for x in candidate_ids]
        n = len(candidate_ids)
        result = np.empty((n, self.dimension), dtype=np.float32)

        dram_hit_count = 0
        ssd_miss_ids: List[int] = []
        ssd_miss_indices: List[int] = []

        with self._lock:
            # Step 1: Check DRAM residency
            for i, vid in enumerate(candidate_ids):
                if vid in self.dram_cache:
                    result[i] = self.dram_cache[vid]
                    dram_hit_count += 1
                else:
                    ssd_miss_ids.append(vid)
                    ssd_miss_indices.append(i)

            # Step 2: Batch-read cold vectors from physical NVMe SSD
            if ssd_miss_ids:
                cold_vectors = self.ssd_store.read_vectors_batch(ssd_miss_ids)
                for idx, vid, vec in zip(ssd_miss_indices, ssd_miss_ids, cold_vectors):
                    result[idx] = vec

            self.dram_hits += dram_hit_count
            self.ssd_reads += len(ssd_miss_ids)

            # Step 3: Record accesses
            self.tracker.record_access_batch(candidate_ids, current_time)

            # Step 4: Evaluate promotion candidates among cold vectors accessed
            migrations_this_batch = 0
            for vid, idx in zip(ssd_miss_ids, ssd_miss_indices):
                if self.policy_type == "policy_b" and migrations_this_batch >= self.max_migrations_per_batch:
                    self.rate_limit_triggered_count += 1
                    break

                vec = result[idx]
                score = float(self.tracker.hotness_score[vid])
                promoted = self._evaluate_promotion(vid, vec, score, current_time)
                if promoted:
                    migrations_this_batch += 1

            # Invariant check
            assert len(self.dram_cache) <= self.max_dram_vectors, (
                f"FATAL: DRAM capacity violated! {len(self.dram_cache)} > {self.max_dram_vectors}"
            )

        return result, dram_hit_count, len(ssd_miss_ids)

    def _evaluate_promotion(
        self, vector_id: int, vector_data: np.ndarray, hotness_score: float, current_time: float
    ) -> bool:
        """
        Evaluates whether an SSD vector should be promoted to DRAM cache.
        Dispatches to Policy A (legacy) or Policy B (hardened).
        """
        if vector_id in self.dram_cache:
            return False

        if self.policy_type == "policy_a":
            return self._evaluate_promotion_policy_a(vector_id, vector_data, hotness_score, current_time)
        else:
            return self._evaluate_promotion_policy_b(vector_id, vector_data, hotness_score, current_time)

    def _evaluate_promotion_policy_a(
        self, vector_id: int, vector_data: np.ndarray, hotness_score: float, current_time: float
    ) -> bool:
        """
        Policy A: Immediate promotion if hotness > coldest in DRAM.
        No admission control, no hysteresis, no residency protection, no cooldown.
        """
        # Case 1: Spare capacity
        if len(self.dram_cache) < self.max_dram_vectors:
            t0 = time.perf_counter_ns()
            self.dram_cache[vector_id] = vector_data.copy()
            self.tracker.set_tier(vector_id, TIER_HOT_DRAM, current_time)
            t_prom_ns = time.perf_counter_ns() - t0
            self.promotion_count += 1
            self.promotion_time_ns += t_prom_ns
            return True

        # Case 2: DRAM is full -> Compare hotness with coldest DRAM entry
        coldest_vid = self.tracker.get_coldest_in_dram(self.dram_cache.keys())
        coldest_score = float(self.tracker.hotness_score[coldest_vid])

        if hotness_score > coldest_score:
            t0 = time.perf_counter_ns()
            # Demote coldest vector to SSD
            del self.dram_cache[coldest_vid]
            self.tracker.set_tier(coldest_vid, TIER_COLD_SSD, current_time)
            self.demotion_count += 1
            t_dem_ns = time.perf_counter_ns() - t0
            self.demotion_time_ns += t_dem_ns

            # Promote candidate vector to DRAM
            t1 = time.perf_counter_ns()
            self.dram_cache[vector_id] = vector_data.copy()
            self.tracker.set_tier(vector_id, TIER_HOT_DRAM, current_time)
            t_prom_ns = time.perf_counter_ns() - t1
            self.promotion_count += 1
            self.promotion_time_ns += t_prom_ns
            return True

        return False

    def _evaluate_promotion_policy_b(
        self, vector_id: int, vector_data: np.ndarray, hotness_score: float, current_time: float
    ) -> bool:
        """
        Policy B (Production Hardened):
        1. Admission Control: Must meet access count or minimum hotness.
        2. Cooldown: Cannot promote if recently demoted within cooldown window.
        3. Promotion Threshold & Hysteresis: Must exceed threshold AND be hotter than coldest by margin.
        4. Minimum Residency Protection: Evicts only vectors whose residency exceeds window.
        """
        # 1. Admission Control
        access_cnt = int(self.tracker.access_count[vector_id])
        if access_cnt < self.admission_min_access and hotness_score < self.admission_min_hotness:
            self.admission_rejections += 1
            return False

        # 2. Cooldown enforcement
        if not self.tracker.can_promote(vector_id, current_time, self.cooldown_time):
            self.cooldown_rejections += 1
            return False

        # 3. Promotion threshold
        if hotness_score < self.promotion_threshold:
            return False

        # Case 1: Spare capacity
        if len(self.dram_cache) < self.max_dram_vectors:
            t0 = time.perf_counter_ns()
            self.dram_cache[vector_id] = vector_data.copy()
            self.tracker.set_tier(vector_id, TIER_HOT_DRAM, current_time)
            t_prom_ns = time.perf_counter_ns() - t0
            self.promotion_count += 1
            self.promotion_time_ns += t_prom_ns
            return True

        # Case 2: DRAM is full -> Select coldest eligible vector (respecting residency window)
        coldest_vid = self.tracker.get_coldest_in_dram_eligible(
            self.dram_cache.keys(), current_time, self.min_residency_time
        )
        coldest_score = float(self.tracker.hotness_score[coldest_vid])

        # 4. Hysteresis margin check: candidate must be hotter by margin
        if hotness_score > (coldest_score + self.hysteresis_margin):
            t0 = time.perf_counter_ns()
            # Demote coldest vector to SSD
            del self.dram_cache[coldest_vid]
            self.tracker.set_tier(coldest_vid, TIER_COLD_SSD, current_time)
            self.demotion_count += 1
            t_dem_ns = time.perf_counter_ns() - t0
            self.demotion_time_ns += t_dem_ns

            # Promote candidate vector to DRAM
            t1 = time.perf_counter_ns()
            self.dram_cache[vector_id] = vector_data.copy()
            self.tracker.set_tier(vector_id, TIER_HOT_DRAM, current_time)
            t_prom_ns = time.perf_counter_ns() - t1
            self.promotion_count += 1
            self.promotion_time_ns += t_prom_ns
            return True

        return False

    def promote_vector_explicit(self, vector_id: int, current_time: float) -> bool:
        """Explicitly forces promotion of a vector (e.g. for lifecycle test)."""
        vid = int(vector_id)
        with self._lock:
            if vid in self.dram_cache:
                return False
            vec = self.ssd_store.read_vector(vid)
            # Temporary bypass for deterministic testing
            old_pol = self.policy_type
            self.policy_type = "policy_a"
            res = self._evaluate_promotion_policy_a(vid, vec, float("inf"), current_time)
            self.policy_type = old_pol
            return res

    def demote_vector_explicit(self, vector_id: int, current_time: float = 0.0) -> bool:
        """Explicitly demotes a vector from DRAM to SSD."""
        vid = int(vector_id)
        with self._lock:
            if vid not in self.dram_cache:
                return False
            del self.dram_cache[vid]
            self.tracker.set_tier(vid, TIER_COLD_SSD, current_time)
            self.demotion_count += 1
            return True

    def compute_working_set_overlap(self, previous_dram_set: Set[int]) -> float:
        """
        Calculates working set overlap: |DRAM_t ∩ DRAM_(t-1)| / |DRAM_t|.
        Returns float in [0.0, 1.0].
        """
        with self._lock:
            if not self.dram_cache or not previous_dram_set:
                return 1.0 if not self.dram_cache and not previous_dram_set else 0.0
            current_set = set(self.dram_cache.keys())
            intersection = current_set.intersection(previous_dram_set)
            return len(intersection) / len(current_set)

    def get_tiering_metrics(self) -> Dict[str, Any]:
        """Calculates current tiering efficiency, hit rates, and migration metrics."""
        with self._lock:
            total_accesses = self.dram_hits + self.ssd_reads
            hit_rate = (self.dram_hits / total_accesses) if total_accesses > 0 else 0.0
            ssd_rate = (self.ssd_reads / total_accesses) if total_accesses > 0 else 0.0

            avg_prom_ms = (
                (self.promotion_time_ns / self.promotion_count / 1e6)
                if self.promotion_count > 0
                else 0.0
            )
            avg_dem_ms = (
                (self.demotion_time_ns / self.demotion_count / 1e6)
                if self.demotion_count > 0
                else 0.0
            )

            tier_sum = self.tracker.get_tier_summary()

            return {
                "policy_type": self.policy_type,
                "dram_capacity_ratio": self.dram_capacity_ratio,
                "dram_capacity_percentage": round(self.dram_capacity_ratio * 100.0, 2),
                "max_dram_vectors": self.max_dram_vectors,
                "current_dram_vectors": len(self.dram_cache),
                "current_ssd_vectors": self.num_vectors - len(self.dram_cache),
                "dram_bytes_used": self.dram_bytes_used,
                "dram_mb_used": round(self.dram_bytes_used / (1024 * 1024), 2),
                "ssd_bytes_used": self.ssd_bytes_used,
                "ssd_mb_used": round(self.ssd_bytes_used / (1024 * 1024), 2),
                "dram_hits": self.dram_hits,
                "ssd_reads": self.ssd_reads,
                "total_accesses": total_accesses,
                "dram_hit_rate": round(hit_rate, 4),
                "ssd_read_rate": round(ssd_rate, 4),
                "promotion_count": self.promotion_count,
                "demotion_count": self.demotion_count,
                "total_migrations": self.promotion_count + self.demotion_count,
                "avg_promotion_latency_ms": round(avg_prom_ms, 5),
                "avg_demotion_latency_ms": round(avg_dem_ms, 5),
                "admission_rejections": self.admission_rejections,
                "cooldown_rejections": self.cooldown_rejections,
                "rate_limit_triggered_count": self.rate_limit_triggered_count,
                "unique_vectors_promoted": tier_sum["unique_vectors_promoted"],
                "unique_vectors_demoted": tier_sum["unique_vectors_demoted"],
                "repeat_promotions": tier_sum["repeat_promotions"],
                "repeat_demotions": tier_sum["repeat_demotions"],
                "promotion_after_previous_demotion_count": tier_sum["promotion_after_previous_demotion_count"],
                "demotion_after_previous_promotion_count": tier_sum["demotion_after_previous_promotion_count"],
                "ssd_io_stats": self.ssd_store.get_io_stats(),
            }

    def reset_metrics(self) -> None:
        """Resets query and I/O counters while preserving tier placement."""
        with self._lock:
            self.total_queries_served = 0
            self.dram_hits = 0
            self.ssd_reads = 0
            self.promotion_count = 0
            self.demotion_count = 0
            self.promotion_time_ns = 0
            self.demotion_time_ns = 0
            self.admission_rejections = 0
            self.cooldown_rejections = 0
            self.rate_limit_triggered_count = 0
            self.ssd_store.reset_io_stats()
