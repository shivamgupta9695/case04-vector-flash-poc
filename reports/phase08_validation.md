# Case 04 — Phase 8: Merge and Exact Reranking Validation Report

**Author:** Senior Software Development Engineer (Antigravity Agent)  
**Date:** October 11, 2026  
**System:** AMD Ryzen 9 7940HS (8 Cores / 16 Threads, 4.0 GHz Base), 32 GB DDR5-5600, WD PC SN740 NVMe PCIe 4.0 SSD, Windows 11 Enterprise + WSL2 Ubuntu 24.04  
**Project:** Case04_Vector_POC — Phase 8 Merge and Exact Reranking Validation  
**Verdict:** **PASS** (100% Automated Tests Passing, 0 Regressions, Frozen Checksums Verified)

---

## 1. Executive Summary

Phase 8 executes an independent, rigorous validation of the candidate merging, local-to-global ID translation, duplicate removal, invalid ID rejection, exact FP32 squared-L2 distance recomputation, and monotonic Top-K sorting layers on top of the frozen Phase 6 `HybridVectorPipeline` and Phase 7 `AdvancedQueryRouter`.

All validation objectives were evaluated on the full 1,000 query vectors (`data/query_vectors.fvecs`) against the frozen global ground truth oracle (`data/ground_truth.ivecs`) and distance oracle (`data/ground_truth_distances.fvecs`):
1. **Candidate-Union Recall vs. Final Reranked Recall:**
   - **Candidate-Union Recall@1 = 0.8270** (82.70%) $\rightarrow$ **Final Reranked Recall@1 = 0.8270** (82.70%).
   - **Candidate-Union Recall@5 = 0.7726** (77.26%) $\rightarrow$ **Final Reranked Recall@5 = 0.7726** (77.26%).
   - **Candidate-Union Recall@10 = 0.7397** (73.97%) $\rightarrow$ **Final Reranked Recall@10 = 0.7397** (73.97%).
   - **Reranking Preservation Fidelity:** **100.00%**. Exact FP32 squared-L2 reranking introduces **zero recall loss** relative to the candidate pool; every ground-truth neighbor present in the merged candidate union is preserved and sorted into the final Top-K.
2. **ALL_TIERS vs. Adaptive Routing Recall Loss:**
   - Evaluated across thresholds $\tau \in [135.0, 145.0, 155.0]$.
   - At $\tau = 135.0$: 25 early exits (2.5%), Recall@10 drops from 0.7397 to 0.7242 ($\Delta = 0.0155$). In **36.0%** of early exits (9 queries), an even closer neighbor existed in the COLD tier and was missed.
   - At $\tau = 145.0$: 159 early exits (15.9%), Recall@10 drops to 0.6479 ($\Delta = 0.0918$). In **56.6%** of early exits (90 queries), a closer neighbor existed in the COLD tier and was missed.
   - At $\tau = 155.0$: 429 early exits (42.9%), Recall@10 drops to 0.4932 ($\Delta = 0.2465$). In **67.4%** of early exits (289 queries), a closer neighbor existed in the COLD tier and was missed.
   - **Definitive Conclusion:** A HOT distance threshold does **NOT** guarantee global nearest-neighbor correctness. While $d_1 \le \tau$ confirms proximity to a DRAM vector, it cannot guarantee that an even closer vector does not reside on the SSD among the 82,000 COLD vectors.
3. **Local-to-Global Mapping & Precision Audit:**
   - Validated that all 82,000 COLD local IDs $c \in [0..81,999]$ map linearly and contiguously to global IDs $18,000 + c \in [18,000..99,999]$ with zero overlap with the HOT partition $[0..17,999]$.
   - Verified that recomputed float32 squared-L2 distances match ground truth distances to a maximum absolute error of $3.05 \times 10^{-5}$, well within the required $1 \times 10^{-4}$ tolerance.
4. **Boundary and Edge Case Resilience:**
   - Empty candidate lists return safely with padded $-1$ IDs and $\infty$ distances.
   - Insufficient candidate pools ($|U| < k$) return available candidates in monotonic order and pad remaining slots with $-1$ / $\infty$.
   - Streams with negative IDs, out-of-range IDs ($\ge 100,000$), non-integers, and heavy duplicates are filtered cleanly.
5. **Full Regression Suite:**
   - **89 out of 89 automated tests passed** across Phases 3–8 in 16.49s.
   - All 27 frozen source/test files and 5 dataset files matched their authoritative SHA-256 hashes bitwise.

---

## 2. Candidate Merging and Local-to-Global Translation

The coordinator receives two streams of candidate IDs:
- **HOT Tier:** Local IDs $[0..17,999]$ from FAISS IndexHNSWFlat, which map 1:1 to global IDs $[0..17,999]$.
- **COLD Tier:** Local IDs $[0..81,999]$ from Microsoft DiskANN3, which map to global IDs $[18,000..99,999]$ via $g = c + 18,000$.

### Table 1: ID Partition Invariants and Audit Results

| Parameter | HOT Partition (DRAM) | COLD Partition (NVMe SSD) | Combined Dataset |
| :--- | :---: | :---: | :---: |
| **Vector Count** | 18,000 | 82,000 | 100,000 |
| **Local Index ID Range** | $[0, 17,999]$ | $[0, 81,999]$ | — |
| **Global ID Range** | $[0, 17,999]$ | $[18,000, 99,999]$ | $[0, 99,999]$ |
| **Index Engine** | FAISS IndexHNSWFlat | Microsoft DiskANN3 (WSL2) | Unified Hybrid Coordinator |
| **Physical Storage** | DRAM FP32 Array | `storage/ssd_vectors/vectors.bin` | Dual-Tiered Store |
| **Mapping Invariant** | Identity ($g = l$) | Linear Shift ($g = l + 18,000$) | Disjoint & Exhaustive |
| **Audit Status** | **VERIFIED** | **VERIFIED** | **PASS (0 Overlap)** |

### Deduplication and Filtering Logic
When merging candidate streams:
1. Candidate IDs from both tiers are ingested sequentially to preserve rank ordering.
2. Any ID $x < 0$ or $x \ge 100,000$ (or non-integer) is rejected and counted under `invalid_ids_rejected`.
3. An ordered hash set preserves the first occurrence of each valid global ID and drops subsequent duplicates.
4. Duplicate removal rate across benchmark runs is recorded, ensuring candidate pools are deduplicated before issuing disk I/O reads.

---

## 3. Candidate-Union Recall vs. Final Reranked Recall

### 3.1 Mathematical Theory of Reranking Fidelity
Let $G_k(q)$ be the ground-truth top-$k$ nearest neighbors for query $q$ across the entire 100,000-vector dataset. Let $U(q) = C_{\text{hot}}(q) \cup C_{\text{cold}}(q)$ be the deduplicated candidate union pool retrieved by the approximate search indexes.

- **Candidate-Union Recall@K:**
  $$\text{Recall}_{\text{union}}@K = \frac{|G_k(q) \cap U(q)|}{k}$$
- **Final Reranked Recall@K:**
  $$\text{Recall}_{\text{final}}@K = \frac{|G_k(q) \cap \text{Top-K}_{\text{reranked}}(q)|}{k}$$

Because ground truth is defined by exact squared-L2 distance, any vector $g \in G_k(q)$ is strictly closer to $q$ than any non-ground-truth vector $v \notin G_k(q)$. Therefore, when exact FP32 squared-L2 reranking is applied to $U(q)$, any true neighbor $g \in G_k(q) \cap U(q)$ must sort ahead of non-ground-truth candidates. Consequently:
$$\text{Recall}_{\text{final}}@K \equiv \text{Recall}_{\text{union}}@K$$

### Table 2: Candidate-Union Recall vs. Final Reranked Recall (1,000 Queries)

| Metric | Candidate-Union Pool (Before Rerank) | Final Output (After FP32 Rerank) | Preservation Fidelity | Rerank Impact |
| :--- | :---: | :---: | :---: | :---: |
| **Recall@1** | **0.8270** (82.70%) | **0.8270** (82.70%) | **100.00%** | Zero Degradation |
| **Recall@5** | **0.7726** (77.26%) | **0.7726** (77.26%) | **100.00%** | Zero Degradation |
| **Recall@10** | **0.7397** (73.97%) | **0.7397** (73.97%) | **100.00%** | Zero Degradation |
| **Candidate Pool Size** | Mean: 100.0 (Min: 100, Max: 100) | Final Top-K: 10 | — | Deduplicated |

*Measured across all 1,000 queries in `data/query_vectors.fvecs` with $k_{\text{hot}}=50, k_{\text{cold}}=50, k=10$.*

**Key Architectural Finding:** The candidate merging and exact reranking stage achieves **100.00% preservation fidelity**. The ~26% recall gap from theoretical 100% is entirely due to candidate generation (ANN graph exploration parameters $ef_{\text{hot}}$ and $l_{\text{cold}}$), never due to reranking truncation or sorting defects.

---

## 4. ALL_TIERS vs. Adaptive Routing: Analysis of Recall Loss

### 4.1 The False Assumption of HOT Distance Thresholds
A critical question in tiered vector databases is whether a low distance in the in-memory tier guarantees that the global nearest neighbor has been found.

The empirical data proves this assumption is **false**:
$$\min_{v \in \text{HOT}} \|q - v\|_2^2 \le \tau \quad \not\Longrightarrow \quad \text{Global Nearest Neighbor} \in \text{HOT}$$

Even when a query vector has an extremely close neighbor in the 18,000 HOT vectors, there is a substantial probability that an even closer vector exists among the 82,000 COLD vectors on SSD.

### Table 3: Adaptive Early-Exit Recall Loss and Missed COLD Neighbors (1,000 Queries)

| Policy / Threshold | Early Exits (%) | Final Recall@1 | Final Recall@5 | Final Recall@10 | Recall Loss @ 10 vs ALL_TIERS | Early Exits with Closer COLD Neighbor | % of Early Exits Missing True Top-1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ALL_TIERS` (Baseline)** | 0 (0.0%) | **0.8270** | **0.7726** | **0.7397** | **0.0000** | 0 | 0.0% |
| **`ADAPTIVE` ($\tau=135.0$)** | 25 (2.5%) | 0.8200 | 0.7572 | 0.7242 | **0.0155** | **9 queries** | **36.0%** |
| **`ADAPTIVE` ($\tau=145.0$)** | 159 (15.9%) | 0.7530 | 0.6832 | 0.6479 | **0.0918** | **90 queries** | **56.6%** |
| **`ADAPTIVE` ($\tau=155.0$)** | 429 (42.9%) | 0.5890 | 0.5206 | 0.4932 | **0.2465** | **289 queries** | **67.4%** |

### 4.2 Key Insights on Recall Loss
1. **At $\tau = 135.0$ (Conservative):**
   - 25 queries (2.5%) exit early in DRAM.
   - Recall@10 drops by only 1.55% (from 73.97% to 72.42%).
   - However, even under this strict threshold, **9 of the 25 early exits (36.0%)** missed their true global Top-1 neighbor because an even closer vector resided in the COLD tier.
2. **At $\tau = 145.0$ (Balanced):**
   - 159 queries (15.9%) exit early in DRAM.
   - Recall@10 drops by 9.18% (from 73.97% to 64.79%).
   - For **56.6% of early-exited queries** (90 out of 159), the true global nearest neighbor was on SSD in the COLD partition.
3. **At $\tau = 155.0$ (Aggressive):**
   - 429 queries (42.9%) exit early in DRAM.
   - Recall@10 drops by 24.65% (from 73.97% to 49.32%).
   - For **67.4% of early-exited queries** (289 out of 429), an even closer neighbor was in COLD.

**Engineering Principle:** An adaptive routing threshold is an I/O and latency optimization tool, **not** an accuracy guarantee. Applications requiring high global recall must use `ALL_TIERS` or set a conservative threshold ($\tau \le 135.0$).

---

## 5. Edge Cases and Boundary Behavior Validation

The reranking engine was subjected to edge-case stress tests to verify stability under abnormal operational conditions:

### Table 4: Edge Case Validation Suite

| Scenario | Input Condition | Expected Behavior | Measured Outcome | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Empty Candidate List** | $U = []$ | Return shape $(k,)$, all $-1$ IDs, $\infty$ dists | `ids = [-1]*10`, `dists = [inf]*10`, `is_padded = True` | **PASS** |
| **Insufficient Candidates** | $|U| = 3$, requested $k = 10$ | Return 3 sorted IDs, pad remaining 7 with $-1$ / $\infty$ | 3 valid IDs monotonic, 7 padded with $-1$ / $\infty$ | **PASS** |
| **Duplicate Candidate Streams** | $[10, 20, 10]$ + $[20, 30, 20]$ | Remove duplicates, preserve order $[10, 20, 30]$ | Clean deduplicated output, duplicate count tracked | **PASS** |
| **Out-of-Bounds & Negative IDs** | $[-1, -99, 150000]$ in stream | Filter out invalid IDs, do not crash | Invalid IDs rejected, clean valid candidates retained | **PASS** |
| **Pure HOT Candidates** | All candidate IDs $< 18,000$ | DRAM-only retrieval, 0 SSD reads | 100% DRAM reads, valid Top-K | **PASS** |
| **Pure COLD Candidates** | All candidate IDs $\ge 18,000$ | SSD-only retrieval | 100% SSD reads, valid Top-K | **PASS** |
| **Mixed Dual-Tier Candidates** | Candidates spanning $[0..99,999]$ | Correct multi-tier fetch, rerank, and sort | Top-K composition verified across both tiers | **PASS** |
| **Extreme Vector Inputs** | Identical coordinates, zeros | Valid distance calculation, stable sort | Distance monotonicity preserved, zero NaNs | **PASS** |

---

## 6. Full Multi-Phase Regression Suite (Phases 3–8)

The complete regression test suite was executed across all phases:

```
Ran 89 tests in 16.494s
OK
```

### Table 5: Multi-Phase Automated Test Breakdown

| Phase | Test Suite File | Test Count | Status | Scope |
| :--- | :--- | :---: | :---: | :--- |
| **Phase 3** | `tests/test_phase3.py` | 18 | **PASSED** | SSD store, access skew, LRU cache, latency monitoring |
| **Phase 4** | `tests/test_phase4.py` | 20 | **PASSED** | Product quantization, ADC, codebook train, reconstruction |
| **Phase 5** | `tests/test_phase5.py` | 16 | **PASSED** | DiskANN build, cold search, ID mapping, WSL bridge |
| **Phase 6** | `tests/test_phase6.py` | 13 | **PASSED** | Hybrid HNSW+DiskANN, FP32 reranking, bitwise reproducibility |
| **Phase 7** | `tests/test_phase7.py` | 11 | **PASSED** | Query router policies, config validation, early exit, parity |
| **Phase 8** | `tests/test_phase8.py` | 11 | **PASSED** | Candidate merging, local-to-global mapping, union recall, edge cases |
| **Total** | **All Phases Combined** | **89** | **PASSED** | **100% Pass Rate (0 Failures, 0 Errors)** |

### Table 6: Checksum Immutability Verification (Phases 1–7)

All 27 frozen source and test files and all 5 dataset files were verified against their authoritative SHA-256 hashes:

| File Path | Expected SHA-256 Checksum | Verified Status |
| :--- | :--- | :---: |
| `data/base_vectors.fvecs` | `62a7ad69bc429570fb4819287bb4d317d8069fb784ef528f370c64189d37198b` | **MATCH** |
| `data/query_vectors.fvecs` | `c1d80d61714296078229bc3429a79569113441c01fdd1069de21611a30528496` | **MATCH** |
| `data/ground_truth.ivecs` | `ee2f420a3fea4e71408756f6d0bf7bed1741a69dafc119b2781f7b4142c7d66f` | **MATCH** |
| `data/ground_truth_distances.fvecs` | `3ca693e88fdfab960b0d43412b8a66266259407615feeb1a1f61ffc083b0ffc4` | **MATCH** |
| `storage/ssd_vectors/vectors.bin` | `a5337c911b10bcd697652595852eb4ccb3a6ae69a438b8414ab557d822c6ab8f` | **MATCH** |
| `src/phase1_1_ground_truth.py` | `50952abf793bd0e18cc44741d334bc33d94fac7ce5aa544fe37eb9dc1c219754` | **MATCH** |
| `src/phase1_2_evaluation.py` | `599567e4ebdb7d4a2424ab12a6eb301b18b3c2f94ec35604002ccc147b4eaae0` | **MATCH** |
| `src/phase2_ann_hnsw.py` | `0af69ac722a8da9f8c8025fec3d33efaba7bce1c9af2d53ae4199c53e90c25fe` | **MATCH** |
| `src/phase3_ssd_store.py` | `e0b819cacee44d258c93264a494cd61feb869ced5932c87344449bf738c6ad67` | **MATCH** |
| `src/phase4_pq.py` | `5d849698804ac3fdbffab9a512888246a6655775a5270c575b3f89303cb1487b` | **MATCH** |
| `src/phase5_diskann_search.py` | `8baa6fee7ccdb898d5158b023a16d72dfc02fbc8a04a71ebb2af09502b2191fe` | **MATCH** |
| `src/phase6_hybrid_search.py` | `14e02d524e947df06b891b10493e1fbc7e2de6d5446fa6fbc5decfa705bcda28` | **MATCH** |
| `src/phase7_query_router.py` | `a7182a68fab0baba17c6e168b87286f945e33e162c6ec3dd981c07b99eda1d31` | **MATCH** |
| `tests/test_phase7.py` | `d252374e0b09f88ba1bf42204e8234a4b509bca25c0ad4dcfb9c197fefd82e5c` | **MATCH** |

---

## 7. Deliverables Summary

1. **`src/phase8_reranker_validation.py`**:
   - Production-grade candidate merging (`ExactCandidateMerger`), exact reranking (`ExactReranker`), union recall validation, and adaptive recall loss quantification.
2. **`tests/test_phase8.py`**:
   - 11 comprehensive automated tests verifying candidate merging, ID translation, duplicate filtering, FP32 precision, union recall fidelity, edge-case padding, and reproducibility.
3. **`results/phase8_reranker/metrics/phase8_metrics.json`**:
   - Structured JSON metrics output covering candidate union analysis, adaptive recall loss analysis, ID mapping audit, precision audit, and edge-case results.
4. **`reports/phase08_validation.md`**:
   - Authoritative technical report (this document).

---

## 8. Final Verdict

| Objective | Criterion | Measured Outcome | Status |
| :--- | :--- | :--- | :---: |
| **Candidate Merging** | Merge HOT & COLD streams | Deduplicated cleanly, order preserved | **PASS** |
| **Local-to-Global Mapping** | $c \rightarrow 18,000 + c$ | Contiguous linear mapping across 82k IDs verified | **PASS** |
| **Duplicate & Invalid Filtering** | Reject $<0, \ge 100k$, dedup | Clean filtering verified across all streams | **PASS** |
| **Exact FP32 Distance Fidelity** | Error $\le 10^{-4}$ vs oracle | Max absolute error $3.05 \times 10^{-5} \le 10^{-4}$ | **PASS** |
| **Monotonic Top-K Sorting** | Distances $d_i \le d_{i+1}$ | 100% monotonic non-decreasing verified | **PASS** |
| **Union Recall Fidelity** | Exact rerank preservation | 100.00% fidelity (lossless reranking) | **PASS** |
| **Adaptive Recall Loss Analysis** | Quantify loss, disprove claim | Proven: 36–67% of early exits miss closer COLD neighbor | **PASS** |
| **Edge Case Resilience** | Empty, $|U| < k$, invalid IDs | Padded safely, zero crashes, robust handling | **PASS** |
| **Regression Suite (Phases 3–8)** | All tests pass, 0 regressions | 89 / 89 tests passed in 16.49s | **PASS** |
| **Frozen Code Immutability** | Phases 1–7 SHA-256 unchanged | 27 source/test files + 5 data files verified | **PASS** |

**Final Assessment: PASS**
