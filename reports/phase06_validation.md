# Case 04 — Phase 6 Validation Report: Unified Hybrid Vector-Search Pipeline

## Executive Summary

This report documents the implementation, architectural verification, and metrics reconciliation of **Phase 6: Unified Hybrid Vector-Search Pipeline** for Case 04.

The hybrid pipeline unifies:
1. **HOT Tier (DRAM)**: 18,000 vectors indexed in FAISS HNSW (`IndexHNSWFlat`, $M=32, efConstruction=40$).
2. **COLD Tier (NVMe SSD)**: 82,000 vectors indexed in Microsoft DiskANN3 ($R=48, L_{build}=80, 32\text{ PQ chunks}$).
3. **Hybrid Coordinator**: Merges per-tier candidate global IDs, eliminates duplicates, retrieves exact full-precision FP32 vectors via `TieredVectorStore`, and performs squared-$L_2$ reranking to return the final global Top-$K$.
4. **Global Evaluation Oracle**: Evaluated against the frozen Phase 1.1 exact ground truth (`data/ground_truth.ivecs`, 1,000 queries).

### Key Validation Verdict: **PASS**

- **Correct Partitioning & Invariants**: HOT IDs $[0, 17999]$ and COLD IDs $[18000, 99999]$ partition the 100,000 dataset disjointly and exhaustively.
- **Dual-Tier Retrieval**: Queries retrieve true nearest neighbors originating from both tiers simultaneously (85.9% of queries have ground truth neighbors in both tiers).
- **High Recall**: Global Recall@10 reaches **82.17%** (Recall@1 = **88.30%** for $K_{cand}=100, L=150$), outperforming the COLD-only baseline (which is bounded at ~58.4% global coverage at $L=100$) and Phase 2 HNSW ($efS=160$ at 75.66%).
- **Reconciled DRAM Footprint**: Total hybrid DRAM footprint is **26.54 MiB** (13.46 MiB HOT index + 8.79 MiB HOT vector buffer + 2.63 MiB DiskANN PQ + 1.66 MiB Phase 4 PQ) compared to **74.79 MiB** for full DRAM HNSW — achieving a **64.5% DRAM reduction** (or **76.3% reduction** considering unique search indices alone).

---

## Architecture & Storage Tier Placement

### 1. DRAM Allocation Breakdown

| Component | Tier Location | Media / Host Layer | Size (MiB) | Architectural Role |
| :--- | :--- | :--- | :---: | :--- |
| **HOT HNSW Index** | DRAM | Host System Memory | 13.46 | Rapid graph exploration for 18k active vectors |
| **HOT FP32 Vector Buffer** | DRAM | Host System Memory | 8.79 | Zero-latency vector retrieval for exact reranking |
| **COLD DiskANN PQ Tables** | DRAM | Host System Memory | 2.63 | Distance estimation and graph routing in DiskANN |
| **Phase 4 PQ Codebooks** | DRAM | Host System Memory | 1.66 | Search-tier compressed representation |
| **Total Hybrid DRAM Footprint** | **DRAM** | **Host System Memory** | **26.54** | **64.5% savings vs. 100k DRAM HNSW (74.79 MiB)** |
| *Unique Search Index Structures* | DRAM | Host System Memory | *17.75* | *Excludes NumPy adapter buffer (76.3% savings)* |

### 2. NVMe Storage Allocation Breakdown

| Persisted NVMe Artifact | Physical File Path | Physical Size | Logical COLD Slice | Resident At Runtime |
| :--- | :--- | :---: | :---: | :---: |
| **COLD Vamana Graph** | `cold_disk_index_tuned_disk.index` | 64.07 MiB | 64.07 MiB | NVMe SSD (sector seeks) |
| **COLD PQ Codes** | `cold_disk_index_tuned_pq_compressed.bin` | 2.50 MiB | 2.50 MiB | Persisted on NVMe, Loaded into DRAM |
| **COLD PQ Centroids** | `cold_disk_index_tuned_pq_pivots.bin` | 0.13 MiB | 0.13 MiB | Persisted on NVMe, Loaded into DRAM |
| **Raw Full Vectors** | `storage/ssd_vectors/vectors.bin` | 48.83 MiB (100k vecs) | 40.04 MiB (82k vecs) | NVMe SSD (512 B seeks for COLD) |
| **Total Logical COLD NVMe** | *COLD Tier Allocation* | — | **104.11 MiB** | **Logical COLD storage budget** |
| **Total Physical NVMe Files** | *All Disk Files* | **115.53 MiB** | — | **Actual physical disk utilization** |

---

## Hybrid Parameter Sweep Results (1,000 Queries, K=10)

| Configuration | $K_{hot}$ | $K_{cold}$ | $L_{cold}$ | $efS_{hot}$ | Recall@1 | Recall@5 | Recall@10 | QPS | Mean Lat (ms) | P95 Lat (ms) | Contribution (HOT / COLD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Low Latency** | 20 | 20 | 30 | 32 | 0.6250 | 0.5534 | **0.5095** | 375.1 | 13.97 | 19.45 | 28.8% / 71.2% |
| **Balanced Low** | 30 | 30 | 50 | 40 | 0.7210 | 0.6486 | **0.6119** | 268.4 | 21.63 | 27.80 | 24.4% / 75.5% |
| **Balanced High** | 50 | 50 | 80 | 64 | 0.7920 | 0.7376 | **0.7044** | 133.8 | 46.02 | 59.57 | 22.7% / 77.3% |
| **High Recall** | 60 | 60 | 100 | 100 | 0.8330 | 0.7806 | **0.7488** | 69.2 | 99.55 | 260.63 | 22.5% / 77.5% |
| **Max Recall** | 100 | 100 | 150 | 160 | 0.8830 | 0.8478 | **0.8217** | 69.7 | 93.88 | 108.42 | 21.3% / 78.7% |

---

## Comparative Baseline Analysis

| Architecture / Strategy | Target Vectors | Recall@1 | Recall@10 | QPS | P50 (ms) | P95 (ms) | DRAM Size | SSD Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 2 HNSW (efS=80)** | 100k (Full DRAM) | 0.7070 | 0.6148 | 3678.4 | 0.46 | 0.58 | 74.8 MB | 0 MB |
| **Phase 2 HNSW (efS=160)** | 100k (Full DRAM) | 0.8190 | 0.7566 | 1974.7 | 0.88 | 1.31 | 74.8 MB | 0 MB |
| **Phase 5 DiskANN3 (L=100)** | 82k (COLD Only) | 0.7009 (COLD) | 0.5843 (Global) | 135.2 | 64.19 | 99.96 | 2.6 MB | 66.7 MB |
| **Phase 6 Hybrid (Balanced High)** | 100k (HOT+COLD) | 0.7920 | **0.7044** | 133.8 | 47.07 | 59.57 | **26.5 MB** | **104.1 MB** |
| **Phase 6 Hybrid (Max Recall)** | 100k (HOT+COLD) | 0.8830 | **0.8217** | 69.7 | 93.07 | 108.42 | **26.5 MB** | **104.1 MB** |
| **Exact FAISS FlatL2** | 100k (Oracle Ref) | 1.0000 | 1.0000 | 4570.8 | 5.23 | 6.62 | 48.8 MB | 0 MB |

---

## Performance Analysis: QPS Throughput vs. Per-Query Latency

The evaluation methodology measures two complementary metrics across the batch of $N = 1,000$ queries:
1. **Aggregate Throughput (QPS)**: Computed from total wall-clock time $\text{QPS} = \frac{1000}{T_{\text{wall\_clock}}}$. For instance, Configuration 1 takes $2.67$ seconds total, yielding **375.1 QPS** (equivalent to $2.67$ ms per query aggregate completion time).
2. **Service Latency Distribution**: Computed from individual query durations $\text{latency}_q = \text{hot\_time}_q + \text{cold\_time}_q + \text{rerank\_time}_q$. Mean latency is **13.97 ms** (P50: 13.61 ms, P95: 19.45 ms).

**Reconciliation of Apparent Difference**:
- Both Microsoft DiskANN3 and FAISS execute using **8 parallel worker threads**.
- In a multi-threaded parallel batch execution, total throughput exceeds single-thread inverted latency by a factor proportional to concurrency ($\approx 8\times$).
- Wall-clock batch duration includes inter-process communication overhead over the WSL2 bridge and file I/O coordination.

---

## Candidate Deduplication and Reranking Correctness

- **Deduplication Rate**: Because HOT IDs ($[0, 17999]$) and COLD IDs ($[18000, 99999]$) are mutually disjoint, the candidate set union is guaranteed to contain 0 cross-tier duplicate IDs.
- **Within-Tier Integrity**: Neither FAISS HNSW nor Microsoft DiskANN returns duplicate IDs within a single query.
- **Monotonicity**: Across all 1,000 queries in all parameter configurations, 100% of the returned Top-K distances are strictly ascending ($0$ monotonic violations).
- **Exact FP32 Distance Fidelity**: Recomputed squared-$L_2$ distances match the exact mathematical definition $\sum_{d=0}^{127} (q_d - v_d)^2$ and match `data/ground_truth_distances.fvecs` to within $10^{-4}$.

---

## Hardware and Runtime Details

- **Host Platform**: Windows 11 Enterprise (Build 26100), Intel Core i7-11850H @ 2.50GHz, 32GB RAM.
- **Storage**: BC711 NVMe SK hynix 512GB SSD (NTFS C: drive).
- **Execution Bridge**: WSL2 Ubuntu 24.04 (Linux kernel 6.6.x) via `/mnt/c/` `drvfs` translation layer for Microsoft DiskANN3; Windows Python 3.12 for FAISS and coordination.

## Final Acceptance Conclusion

Phase 6 meets all implementation, mathematical, persistence, and evaluation criteria specified for Case 04. Status: **PASS**.
