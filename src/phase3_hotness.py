"""
Case 04 — Phase 3: Access Tracking & Dynamic Hotness Engine
Time-decayed access frequency, recency, and hotness scoring with production hardening.

Tracks access history, frequency, recency, residency durations, and tier assignments across all 100,000 vectors
with O(1) per-query updates, thread safety, and hysteresis/cooldown tracking.
"""

import sys
import time
import math
import threading
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional, Iterable

try:
    import numpy as np
except ImportError as err:
    print(f"CRITICAL ERROR: Failed to import NumPy: {err}", file=sys.stderr)
    sys.exit(1)


# Tier constants
TIER_COLD_SSD = 0
TIER_HOT_DRAM = 1


class AccessTracker:
    """
    Manages vector access statistics and evaluates dynamic hotness scores.
    Uses dense NumPy arrays for instant O(1) indexed reads and updates.
    Thread-safe and supports residency/cooldown tracking.
    """

    def __init__(
        self,
        num_vectors: int = 100_000,
        alpha: float = 0.5,      # Weight for recent access frequency
        beta: float = 0.2,       # Weight for cumulative historical frequency
        gamma: float = 0.3,      # Weight for recency score
        time_decay_lambda: float = 0.05,  # Exponential recency decay constant
    ):
        self.num_vectors = num_vectors
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.time_decay_lambda = time_decay_lambda

        self._lock = threading.Lock()

        # Per-vector tracking arrays
        self.access_count = np.zeros(num_vectors, dtype=np.int32)
        self.recent_access_count = np.zeros(num_vectors, dtype=np.float32)
        self.last_access_time = np.zeros(num_vectors, dtype=np.float64)
        self.hotness_score = np.zeros(num_vectors, dtype=np.float32)
        self.current_tier = np.zeros(num_vectors, dtype=np.int8)  # Default all COLD (SSD)
        self.promotion_count = np.zeros(num_vectors, dtype=np.int32)
        self.demotion_count = np.zeros(num_vectors, dtype=np.int32)

        # Hardening: residency and cooldown tracking
        self.last_promotion_time = np.zeros(num_vectors, dtype=np.float64)
        self.last_demotion_time = np.zeros(num_vectors, dtype=np.float64)

        self.total_access_events: int = 0
        self.last_global_time: float = 0.0

    def record_access(self, vector_id: int, current_time: float) -> float:
        """
        Records a single vector access event and immediately updates its hotness score.
        Time complexity: O(1).
        """
        vid = int(vector_id)
        if vid < 0 or vid >= self.num_vectors:
            raise IndexError(f"Vector ID {vid} out of bounds")

        with self._lock:
            self.access_count[vid] += 1
            self.recent_access_count[vid] += 1.0

            prev_time = self.last_access_time[vid]
            if prev_time > 0.0:
                delta_t = max(0.0, current_time - prev_time)
                recency = math.exp(-self.time_decay_lambda * delta_t)
            else:
                recency = 1.0  # First access is considered fully recent

            self.last_access_time[vid] = current_time
            self.last_global_time = max(self.last_global_time, current_time)
            self.total_access_events += 1

            # Hotness score formulation:
            score = (
                self.alpha * self.recent_access_count[vid]
                + self.beta * math.log1p(float(self.access_count[vid]))
                + self.gamma * recency
            )
            self.hotness_score[vid] = score
            return score

    def record_access_batch(self, vector_ids: List[int] | np.ndarray, current_time: float) -> None:
        """
        Vectorized batch access recording for query candidate sets.
        """
        unique_ids, counts = np.unique(vector_ids, return_counts=True)
        with self._lock:
            for vid, c in zip(unique_ids, counts):
                vid = int(vid)
                self.access_count[vid] += int(c)
                self.recent_access_count[vid] += float(c)

                prev_time = self.last_access_time[vid]
                delta_t = max(0.0, current_time - prev_time) if prev_time > 0.0 else 0.0
                recency = math.exp(-self.time_decay_lambda * delta_t) if prev_time > 0.0 else 1.0
                self.last_access_time[vid] = current_time

                score = (
                    self.alpha * self.recent_access_count[vid]
                    + self.beta * math.log1p(float(self.access_count[vid]))
                    + self.gamma * recency
                )
                self.hotness_score[vid] = score

            self.total_access_events += len(vector_ids)
            self.last_global_time = max(self.last_global_time, current_time)

    def decay_recent_counts(self, decay_rate: float = 0.85) -> None:
        """
        Decays recent access counters periodically across active vectors.
        Prevents historical inertia from permanently locking vectors in the hot tier.
        """
        with self._lock:
            active_mask = (self.recent_access_count > 0.0001) | (self.current_tier == TIER_HOT_DRAM)
            self.recent_access_count[active_mask] *= decay_rate
            dt = np.maximum(0.0, self.last_global_time - self.last_access_time[active_mask])
            has_accessed = self.last_access_time[active_mask] > 0.0
            recency = np.where(has_accessed, np.exp(-self.time_decay_lambda * dt), 0.0)
            self.hotness_score[active_mask] = (
                self.alpha * self.recent_access_count[active_mask]
                + self.beta * np.log1p(self.access_count[active_mask])
                + self.gamma * recency
            )

    def set_tier(self, vector_id: int, tier: int, current_time: float = 0.0) -> None:
        """Updates tier assignment, tracks transitions, and logs transition timestamps."""
        vid = int(vector_id)
        with self._lock:
            old_tier = self.current_tier[vid]
            if old_tier != tier:
                self.current_tier[vid] = tier
                if old_tier == TIER_COLD_SSD and tier == TIER_HOT_DRAM:
                    self.promotion_count[vid] += 1
                    self.last_promotion_time[vid] = current_time
                elif old_tier == TIER_HOT_DRAM and tier == TIER_COLD_SSD:
                    self.demotion_count[vid] += 1
                    self.last_demotion_time[vid] = current_time

    def get_coldest_in_dram(self, dram_vector_ids: Iterable[int]) -> int:
        """Finds the vector in DRAM with the lowest hotness score for eviction."""
        ids_list = list(dram_vector_ids)
        if not ids_list:
            raise ValueError("DRAM set is empty")
        with self._lock:
            scores = self.hotness_score[ids_list]
            min_idx = int(np.argmin(scores))
            return ids_list[min_idx]

    def get_coldest_in_dram_eligible(
        self,
        dram_vector_ids: Iterable[int],
        current_time: float,
        min_residency_time: float = 0.0,
    ) -> int:
        """
        Finds the coldest vector in DRAM eligible for eviction considering minimum residency time.
        If all resident vectors are within the residency window, falls back to the absolute coldest
        to prevent violating the hard DRAM capacity invariant.
        """
        ids_list = list(dram_vector_ids)
        if not ids_list:
            raise ValueError("DRAM set is empty")

        with self._lock:
            if min_residency_time <= 0.0:
                scores = self.hotness_score[ids_list]
                return ids_list[int(np.argmin(scores))]

            # Filter candidates that satisfy minimum residency duration
            eligible = [
                vid for vid in ids_list
                if (current_time - self.last_promotion_time[vid]) >= min_residency_time
            ]

            if eligible:
                scores = self.hotness_score[eligible]
                return eligible[int(np.argmin(scores))]

            # Fallback to absolute coldest if all are protected (preserves capacity invariant)
            scores = self.hotness_score[ids_list]
            return ids_list[int(np.argmin(scores))]

    def can_promote(self, vector_id: int, current_time: float, cooldown: float) -> bool:
        """Checks if a vector is eligible for promotion after demotion cooldown."""
        vid = int(vector_id)
        if cooldown <= 0.0:
            return True
        last_dem = self.last_demotion_time[vid]
        if last_dem <= 0.0:
            return True
        return (current_time - last_dem) >= cooldown

    def get_vector_status(self, vector_id: int) -> Dict[str, Any]:
        """Returns complete tracking diagnostics for a vector."""
        vid = int(vector_id)
        with self._lock:
            return {
                "vector_id": vid,
                "access_count": int(self.access_count[vid]),
                "recent_access_count": round(float(self.recent_access_count[vid]), 4),
                "last_access_time": round(float(self.last_access_time[vid]), 4),
                "last_promotion_time": round(float(self.last_promotion_time[vid]), 4),
                "last_demotion_time": round(float(self.last_demotion_time[vid]), 4),
                "hotness_score": round(float(self.hotness_score[vid]), 4),
                "current_tier": "DRAM" if self.current_tier[vid] == TIER_HOT_DRAM else "SSD",
                "promotion_count": int(self.promotion_count[vid]),
                "demotion_count": int(self.demotion_count[vid]),
            }

    def get_tier_summary(self) -> Dict[str, Any]:
        """Summarizes current hotness distribution and tier populations with churn metrics."""
        with self._lock:
            dram_count = int(np.sum(self.current_tier == TIER_HOT_DRAM))
            ssd_count = int(np.sum(self.current_tier == TIER_COLD_SSD))
            tot_prom = int(np.sum(self.promotion_count))
            tot_dem = int(np.sum(self.demotion_count))

            unique_prom = int(np.sum(self.promotion_count > 0))
            unique_dem = int(np.sum(self.demotion_count > 0))
            repeat_prom = int(np.sum(self.promotion_count > 1))
            repeat_dem = int(np.sum(self.demotion_count > 1))

            # Ping-pong churn metrics
            prom_after_dem = int(np.sum((self.promotion_count > 1) & (self.demotion_count > 0)))
            dem_after_prom = int(np.sum((self.demotion_count > 0) & (self.promotion_count > 0)))

            return {
                "total_vectors": self.num_vectors,
                "dram_hot_vectors": dram_count,
                "ssd_cold_vectors": ssd_count,
                "dram_percentage": round(dram_count / self.num_vectors * 100.0, 2),
                "ssd_percentage": round(ssd_count / self.num_vectors * 100.0, 2),
                "total_promotions": tot_prom,
                "total_demotions": tot_dem,
                "unique_vectors_promoted": unique_prom,
                "unique_vectors_demoted": unique_dem,
                "repeat_promotions": repeat_prom,
                "repeat_demotions": repeat_dem,
                "promotion_after_previous_demotion_count": prom_after_dem,
                "demotion_after_previous_promotion_count": dem_after_prom,
                "total_access_events": self.total_access_events,
            }
