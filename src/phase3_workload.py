"""
Case 04 — Phase 3: Real-Time Query Workload Generator
Configurable access distributions: Uniform, Zipf/Skewed, Hotspot (80/20), and Targeted Bursts.

Simulates real-time streaming queries with configurable skew to evaluate
DRAM cache hit rates and SSD I/O offloading.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

try:
    import numpy as np
except ImportError as err:
    print(f"CRITICAL ERROR: Failed to import NumPy: {err}", file=sys.stderr)
    sys.exit(1)


class WorkloadGenerator:
    """
    Generates deterministic, parameterized query streams over the 1,000 queries.
    """

    def __init__(self, query_vectors: np.ndarray, seed: int = 100):
        self.query_vectors = query_vectors
        self.num_queries = len(query_vectors)
        self.rng = np.random.default_rng(seed)

    def generate_uniform_stream(self, stream_length: int = 2_000) -> List[int]:
        """
        Uniform workload: all query vectors have equal probability (1 / N_queries).
        """
        return self.rng.integers(0, self.num_queries, size=stream_length).tolist()

    def generate_zipf_stream(self, stream_length: int = 2_000, zipf_alpha: float = 1.25) -> List[int]:
        """
        Skewed Zipfian workload: query frequency follows a power-law distribution.
        A small set of queries dominates traffic.
        """
        # Generate ranks from 1 to num_queries
        ranks = np.arange(1, self.num_queries + 1, dtype=np.float64)
        probabilities = 1.0 / (ranks ** zipf_alpha)
        probabilities /= np.sum(probabilities)

        # Shuffle query IDs once deterministically so hot queries are distributed
        permuted_ids = self.rng.permutation(self.num_queries)
        sampled_ranks = self.rng.choice(len(permuted_ids), size=stream_length, p=probabilities)
        return permuted_ids[sampled_ranks].tolist()

    def generate_hotspot_stream(
        self,
        stream_length: int = 2_000,
        hot_fraction: float = 0.10,   # Top 10%
        hot_probability: float = 0.80 # Receives 80% of traffic
    ) -> List[int]:
        """
        Hotspot workload (Pareto 80/20):
        Top 10% of queries generate 80% of all access traffic;
        Remaining 90% generate 20%.
        """
        num_hot = max(1, int(self.num_queries * hot_fraction))
        hot_ids = np.arange(0, num_hot)
        cold_ids = np.arange(num_hot, self.num_queries)

        choices = self.rng.random(stream_length)
        stream: List[int] = []
        for r in choices:
            if r < hot_probability:
                stream.append(int(self.rng.choice(hot_ids)))
            else:
                stream.append(int(self.rng.choice(cold_ids)))
        return stream

    def generate_targeted_burst(
        self,
        target_query_id: int,
        burst_length: int = 100,
        background_length: int = 200,
    ) -> List[int]:
        """
        Targeted burst: generates a background stream followed by a high-frequency
        burst directed at a specific query/neighborhood, then returns to background.
        Used to demonstrate cold -> hot dynamic promotion and decay.
        """
        background_pre = self.rng.integers(0, self.num_queries, size=background_length // 2).tolist()
        burst = [int(target_query_id)] * burst_length
        background_post = self.rng.integers(0, self.num_queries, size=background_length // 2).tolist()
        return background_pre + burst + background_post
