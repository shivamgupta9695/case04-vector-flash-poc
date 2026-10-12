# CASE 04 — PHASE 3 PRODUCTION HARDENING: COMPREHENSIVE CODE & ARCHITECTURE AUDIT REPORT

**Date:** October 2026  
**Auditor:** Senior Systems & Storage Engineer, ANN Researcher  
**Target:** Case 04 — Phase 3 Real-Time DRAM + NVMe SSD Tiered Vector Storage  
**Status:** AUDIT COMPLETE — ISSUES IDENTIFIED — HARDENING ROADMAP ESTABLISHED  

---

## 1. Executive Summary

An exhaustive code audit of the Case 04 Phase 3 implementation was conducted across all core modules:
- `src/phase3_ssd_store.py` (NVMe SSD storage layer)
- `src/phase3_hotness.py` (Access tracking & dynamic hotness scoring)
- `src/phase3_tiering.py` (Tiered vector manager & DRAM cache)
- `src/phase3_workload.py` (Parameterized query workload generator)
- `src/phase3_realtime.py` (Search engine orchestrator & benchmarking harness)

The audit confirmed the functional viability of the hybrid DRAM/SSD tiering architecture, but uncovered **two critical production defects** and **three major architectural performance bottlenecks**:

1. **Latency Statistics Anomaly (10% DRAM + Uniform):**
   - Reported: Mean = 753.17 ms, P50 = 15.75 ms, P95 = 58.39 ms, P99 = 93.05 ms.
   - P95 is less than one-twelfth of the arithmetic mean. This mathematical discrepancy arises from extreme positive outliers in the latency distribution caused by severe unamortized eviction overhead, potential Windows OS background stalls, and unbuffered I/O contention during catastrophic cache thrashing.
2. **Pathological Cache Thrashing & Migration Churn:**
   - At 18% DRAM (Uniform), 8,653 promotions and 8,653 demotions occurred across 1,000 queries. Every single SSD miss resulted in an eviction and a promotion (100% churn ratio).
   - This occurs because the policy lacks **admission control**, **hysteresis**, **minimum residency**, and **cooldown**.
3. **$O(N_{\text{DRAM}})$ Algorithmic Bottleneck on the Query Critical Path:**
   - Eviction evaluation calls `get_coldest_in_dram(set(self.dram_cache.keys()))`, converting the dict keys to a Python set, then to a Python list, slicing a NumPy array of size $N_{\text{DRAM}}$, and running `np.argmin`. For 30+ misses per query, this executes dozens of times synchronously within the search latency window.
4. **Cold-Start Metadata Asymmetry:**
   - Initial DRAM priming loads vectors into `dram_cache` but never records accesses in `AccessTracker`. All initial DRAM vectors have `hotness = 0.0`, rendering them instantly vulnerable to eviction by the first random access to any cold SSD vector.
5. **Concurrency & Thread Safety Deficiencies:**
   - Neither `TieredVectorManager`, `AccessTracker`, nor `SSDVectorStore` are thread-safe. Concurrent multi-worker execution will corrupt dictionary and NumPy tracking state.

---

## 2. Current Architecture

```text
                             Query Stream
                                  │
                                  ▼
                        FAISS IndexHNSWFlat
                      (M=32, efConstruction=40)
                       efSearch=40 Candidates
                                  │
                                  ▼
                     TieredVectorManager
                       Candidate Routing
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
            DRAM Cache                         NVMe SSD
           (Dict[int, np.ndarray])          (vectors.bin, 512B)
            HOT Vectors                       COLD Vectors
                 │                                 │
                 └────────────────┬────────────────┘
                                  ▼
                       Exact Distance Rerank
                          Top-K Selection
                                  │
                                  ▼
                         Query Latency End
                                  │
                     (Blocking / Synchronous)
                                  ▼
                            AccessTracker
                     (Dense NumPy Arrays 100K)
                                  │
                                  ▼
                           Hotness Engine
                                  │
                                  ▼
                         _evaluate_promotion
                   (Coldest DRAM Linear Search)
                      Promote SSD -> DRAM
                      Demote DRAM -> SSD
```

---

## 3. Current Timing Model

### Timer Implementation
- Measured using `time.perf_counter()` (floating-point seconds).
- Calculated as: `latency_ms = (t_end - t_start) * 1000.0`.

### Timing Scope
- The measured query latency window encloses:
  1. HNSW candidate generation: `self.index.search(q_arr, num_candidates)`
  2. Full vector payload retrieval: `self.tier_manager.get_vectors_for_candidates(candidates, current_time)`
     - DRAM cache lookup
     - SSD batch read: `self.ssd_store.read_vectors_batch(ssd_miss_ids)`
     - Batch access tracking: `self.tracker.record_access_batch(candidate_ids, current_time)`
     - **Synchronous promotion & eviction loop:** `self._evaluate_promotion` for every miss
  3. Exact distance reranking: Euclidean L2 calculation and `np.argsort`
- **Defects in Timing Model:**
  - Component latencies (search, tier lookup, SSD I/O, migration) are not recorded on a per-query basis.
  - Timer precision is standard `time.perf_counter()` rather than monotonic high-resolution integer nanoseconds (`time.perf_counter_ns()`).
  - Warm-up queries (first 50 queries) are discarded from the latency array, but migration and I/O counters accumulate across both warm-up and measurement intervals.

---

## 4. Current Cache Policy

### Capacity Management
- Configured budget: `max_dram_vectors = int(num_vectors * dram_capacity_ratio)`.
- Storage: Python `dict` mapping `vector_id -> np.ndarray (128-dim float32)`.

### Eviction & Promotion Logic
```python
if len(self.dram_cache) < self.max_dram_vectors:
    # Spare capacity: insert directly
    self.dram_cache[vector_id] = vector_data.copy()
    self.tracker.set_tier(vector_id, TIER_HOT_DRAM)
else:
    # Full: compare with coldest DRAM entry
    coldest_vid = self.tracker.get_coldest_in_dram(set(self.dram_cache.keys()))
    coldest_score = self.tracker.hotness_score[coldest_vid]
    if hotness_score > coldest_score:
        del self.dram_cache[coldest_vid]
        self.tracker.set_tier(coldest_vid, TIER_COLD_SSD)
        self.dram_cache[vector_id] = vector_data.copy()
        self.tracker.set_tier(vector_id, TIER_HOT_DRAM)
```

### Flaws in Cache Policy
1. **Zero Hysteresis:** The eviction threshold equals the promotion threshold. Any infinitesimal advantage causes immediate swap.
2. **Zero Admission Control:** A single access to an unaccessed cold SSD vector grants a hotness score of $\approx 0.9386$ ($0.5 \times 1.0 + 0.2 \times \ln(2) + 0.3 \times 1.0$). This score exceeds any initial DRAM entry ($0.0$) and decayed entries, immediately evicting resident vectors.
3. **Zero Minimum Residency:** A newly promoted vector can be evicted immediately on the next miss.
4. **Zero Cooldown:** A vector can be promoted and demoted endlessly within the same batch.
5. **Algorithmic Inefficiency:** Eviction scans all DRAM keys via linear NumPy array indexing on every candidate miss.

---

## 5. Current Hotness Model

### Scoring Formula
$$\text{Hotness}(v) = \alpha \cdot f_{\text{recent}}(v) + \beta \cdot \ln(1 + C_{\text{historical}}(v)) + \gamma \cdot \exp(-\lambda \Delta t)$$

Default weights: $\alpha = 0.5$, $\beta = 0.2$, $\gamma = 0.3$, $\lambda = 0.05$.

### Decay Policy
- Triggered every 200 queries:
  $$\text{recent\_access\_count} \leftarrow \text{recent\_access\_count} \times 0.88$$
- Recalculates scores for active vectors and resident DRAM vectors.
- Does not protect newly admitted vectors from rapid eviction.

---

## 6. Current Migration Model

- **Coupling:** Migration is tightly coupled to query execution. The query thread blocks while demoting vectors from DRAM and inserting new vectors.
- **Migration Cost:**
  - Dict deletion: $O(1)$
  - Dict insertion: $O(1)$
  - Coldest search: $O(N_{\text{DRAM}})$ — heavily taxing Python interpreter when $N_{\text{DRAM}} \ge 10,000$.

---

## 7. Current Workload Model

- Three synthetic distributions based on the 1,000 query vectors:
  1. **Uniform:** Equal probability across all queries ($1 / 1,000$).
  2. **Zipf (Skewed):** Power law distribution with $\alpha = 1.25$.
  3. **Hotspot (80/20):** Top 10% queries generate 80% traffic; remaining 90% generate 20%.
- Workload generation is deterministic given a fixed random seed.

---

## 8. Identified Correctness Risks

| Risk ID | Component | Description | Impact |
|:---|:---|:---|:---|
| **CR-01** | Latency Aggregation | P95 < Mean anomaly indicates severe outlier skew or sample corruption. | Misleading tail latency reporting. |
| **CR-02** | Metadata Asymmetry | Initial DRAM vectors have zero hotness score upon startup. | Immediate eviction of primed vectors on cold start. |
| **CR-03** | Denominator Inconsistency | SSD reads per query divides by total queries (including warmup) while latency uses measured queries. | Distorted I/O rates. |
| **CR-04** | Lack of Monotonic Ints | Float `time.perf_counter()` subject to precision drift over long runs. | Sub-microsecond timing inaccuracy. |
| **CR-05** | Thread Safety | Global mutable state in tracker and tier manager without locking. | High risk of race conditions in multi-threaded serving. |

---

## 9. Identified Performance Risks

| Risk ID | Component | Description | Impact |
|:---|:---|:---|:---|
| **PR-01** | Eviction Search | $O(N_{\text{DRAM}})$ linear scan through all DRAM keys inside `get_coldest_in_dram`. | 10–50 ms CPU overhead per query on high-miss workloads. |
| **PR-02** | Cache Thrashing | Complete 1:1 churn between SSD misses and promotions under Uniform workload. | Pathological migration traffic (up to 9,455 migrations / 1,000 queries). |
| **PR-03** | Synchronous Eviction | Query thread blocks on migration accounting and data structures. | Degrades query throughput (QPS drops to 1.3). |
| **PR-04** | Garbage Collection | Repeated creation of large sets and lists inside `_evaluate_promotion`. | Triggers Python GC pauses, creating multi-hundred ms tail latency spikes. |

---

## 10. Recommended Fixes

1. **High-Resolution Per-Query Latency Instrumentation:**
   - Use `time.perf_counter_ns()`.
   - Record per-query structured telemetry: `query_id`, `start_ns`, `end_ns`, `total_latency_us`, `dram_hit`, `ssd_reads`, `ssd_read_latency_us`, `promotion_occurred`, `promotion_latency_us`, `demotion_occurred`, `search_latency_us`, `tier_lookup_latency_us`.
2. **Hardened Cache Policy (Policy B):**
   - **Admission Control:** Vector must achieve minimum access threshold ($C_{\text{admission}} \ge 2$) or minimum hotness score before qualifying for promotion.
   - **Hysteresis:** Separate promotion threshold ($\theta_{\text{prom}}$) and demotion threshold ($\theta_{\text{dem}}$) where $\theta_{\text{prom}} > \theta_{\text{dem}}$.
   - **Minimum Residency Time:** Vector cannot be evicted within $T_{\text{residency}}$ seconds of promotion.
   - **Cooldown Windows:** Prevent repeated promotion/demotion oscillation within $T_{\text{cooldown}}$.
   - **Optimized Eviction:** Maintain a bounded min-heap / sample-based eviction candidate pool or indexed search to eliminate the $O(N_{\text{DRAM}})$ linear scan.
3. **Decoupled Asynchronous / Buffered Migration:**
   - Allow query thread to return results immediately while queueing eviction/promotion events or executing fast bounded updates.
4. **Latency Sanity Checks:**
   - Automated assertions: $\min \le P_{50} \le P_{90} \le P_{95} \le P_{99} \le \max$.
   - Sample count validation: `len(samples) == num_measurement_queries`.
   - Outlier diagnostic logging whenever $\text{mean} / P_{95} > 1.5$.
5. **Thread-Safe Architecture:**
   - Thread-safe synchronization (reentrant lock / fine-grained locking) for multi-worker benchmark validation.

---

## 11. Required Tests Before Benchmark Reruns

The following 15 tests must be implemented and passing in `tests/test_phase3.py`:
1. `test_dataset_integrity`: Cryptographic SHA-256 validation of frozen Phase 1 files.
2. `test_ssd_round_trip`: Byte-for-byte fidelity of random vector write/read.
3. `test_offset_correctness`: Exact record offset calculation (`vector_id * 512`).
4. `test_cache_capacity_invariant`: DRAM vector count never exceeds `max_dram_vectors`.
5. `test_promotion_mechanics`: Cold vector promotes to DRAM upon exceeding threshold.
6. `test_demotion_mechanics`: Hot vector demotes to SSD after decay and competition.
7. `test_hysteresis_damping`: Vector does not oscillate between tiers around a single threshold.
8. `test_minimum_residency`: Recently promoted vector cannot be evicted within residency window.
9. `test_latency_sample_count`: Measured sample count matches expected query count.
10. `test_percentile_monotonicity`: $\min \le P_{50} \le P_{95} \le P_{99} \le \max$ verified.
11. `test_recall_consistency`: Recall@1, 5, 10 remain strictly within $[0.0, 1.0]$.
12. `test_data_corruption_free`: Retrieved vectors match ground truth exactly.
13. `test_deterministic_workloads`: Identical seeds generate identical query streams.
14. `test_capacity_invariant_stress`: Stress testing with 5,000 rapid access cycles maintains strict capacity bound.
15. `test_v9000_lifecycle`: Full COLD $\to$ HOT $\to$ COLD state transition verified.

---
*Report approved for Phase 3 Hardening execution.*
