# Case 04: Vector Search POC — System Architecture

**Author:** Senior Systems Engineer  
**Date:** October 11, 2026  
**Project:** `Case04_Vector_POC`  
**Version:** Phase 10 Final Submission  

---

## 1. High-Level Architecture Overview

The system implements a tiered hybrid approximate nearest neighbor (ANN) vector database engine designed to balance high query recall, low DRAM residency, and high throughput over large vector collections.

```
                           [Client Application]
                                    │
                                    ▼ (PyMilvus SDK API)
                      ┌───────────────────────────┐
                      │    MilvusHybridAdapter    │  (src/phase9_milvus_adapter.py)
                      │ (PyMilvus Gateway Layer)  │
                      └─────────────┬─────────────┘
                                    │
                                    ▼ (Internal Vector Query)
                      ┌───────────────────────────┐
                      │    AdvancedQueryRouter    │  (src/phase7_query_router.py)
                      │ (ALL_TIERS / ADAPTIVE /   │
                      │  HOT_ONLY / COLD_ONLY)    │
                      └──────┬─────────────┬──────┘
                             │             │
              ┌──────────────┘             └──────────────┐
              ▼                                           ▼
   [HOT Tier: FAISS HNSW]                     [COLD Tier: Microsoft DiskANN3]
   - DRAM Resident: 18,000 vectors            - NVMe SSD: 82,000 vectors
   - IDs: [0 .. 17,999]                       - Local IDs: [0 .. 81,999]
   - Latency: ~0.2 ms                         - Latency: ~30-40 ms (WSL2 Direct I/O)
   - Size: 13.46 MiB Index + 8.79 MiB Vecs    - Size: 64.07 MiB Index + 40.04 MiB Vecs
              │                                           │
              └──────────────┬────────────────────────────┘
                             │
                             ▼ (Candidates: 50 HOT + 50 COLD)
                      ┌───────────────────────────┐
                      │   ExactCandidateMerger    │  (src/phase8_reranker_validation.py)
                      │ - Local -> Global ID Map  │  (COLD ID + 18,000)
                      │ - Deduplication & Bounds  │
                      └─────────────┬─────────────┘
                                    │ (Unique Candidate IDs)
                                    ▼
                      ┌───────────────────────────┐
                      │     TieredVectorStore     │  (src/phase6_hybrid_search.py)
                      │ - Fetches Raw FP32 Vecs   │  (DRAM slice or NVMe file seek)
                      └─────────────┬─────────────┘
                                    │ (Contiguous FP32 Matrix)
                                    ▼
                      ┌───────────────────────────┐
                      │       ExactReranker       │  (src/phase8_reranker_validation.py)
                      │ - Recomputes Exact L2^2   │
                      │ - Strictly Ascending Sort │
                      └─────────────┬─────────────┘
                                    │
                                    ▼ (Top-10 Global IDs & Distances)
                             [Final Result]
```

---

## 2. Component Specifications

### 2.1 HOT Tier (DRAM + FAISS HNSW)
- **Vector Count:** 18,000 vectors ($ID \in [0, 17,999]$).
- **Index Type:** FAISS `IndexHNSWFlat` ($M = 32, efConstruction = 40, efSearch = 64$).
- **Memory Footprint:** 13.46 MiB index + 8.79 MiB FP32 vector buffer = **22.25 MiB**.
- **Search Latency:** $< 0.25$ ms per batch query.

### 2.2 COLD Tier (NVMe SSD + Microsoft DiskANN3)
- **Vector Count:** 82,000 vectors ($ID \in [18,000, 99,999]$).
- **Engine:** Official Microsoft DiskANN3 compiled in WSL2 Ubuntu (`diskann_query`).
- **Disk Index:** 64.07 MiB tuned Vamana disk graph on NVMe SSD ($R = 64, L = 100, \text{beam width} = 4$).
- **DRAM Footprint:** 2.50 MiB compressed PQ table + 0.13 MiB centroids = **2.63 MiB**.
- **I/O Mechanism:** Random block seeks directly against `vectors.bin` on NVMe SSD.

### 2.3 Merge, Translation & Exact Reranking Pipeline
- **Local-to-Global Translation:** $g = c + 18,000$ for all COLD candidates.
- **Deduplication:** Merges up to 100 candidates into ordered, disjoint sets.
- **Exact Squared-L2 Reranking:** Fetches original uncompressed FP32 vectors and recomputes $\sum (q_i - v_i)^2$.
- **Preservation Fidelity:** 100.0% of true ground-truth neighbors in candidate pools are guaranteed to reach the final Top-10.

### 2.4 Milvus Integration Boundary
- **Implementation:** `MilvusHybridAdapter` (`src/phase9_milvus_adapter.py`).
- **Integration Boundary:** Client-facing gateway exposing standard `pymilvus.MilvusClient` collection management and search signatures.
- **Architectural Distinction:** It delegates execution to the authoritative hybrid engine. It is **not** an internal C++ segment driver compiled inside Milvus's `Knowhere` kernel.

---

## 3. Data & Storage Model

| Component | Logical Partition | Physical Storage Path | Physical Size | Resident Medium |
| :--- | :---: | :--- | :---: | :---: |
| HOT HNSW Index | HOT | `results/phase6_hybrid/index/hot_hnsw.index` | 13.46 MiB | DRAM |
| HOT Raw Vectors | HOT | Contiguous memory slice ($0..17,999$) | 8.79 MiB | DRAM |
| COLD Disk Index | COLD | `results/phase5_diskann/index/...tuned_disk.index` | 64.07 MiB | NVMe SSD |
| COLD PQ Codes | COLD | `results/phase5_diskann/index/...tuned_pq_compressed.bin`| 2.50 MiB | NVMe SSD / DRAM |
| COLD PQ Pivots | COLD | `results/phase5_diskann/index/...tuned_pq_pivots.bin` | 0.13 MiB | NVMe SSD / DRAM |
| Raw Vectors | Both | `storage/ssd_vectors/vectors.bin` | 48.83 MiB | NVMe SSD |
| **Totals** | | **All Files** | **115.53 MiB (NVMe)** | **26.54 MiB (DRAM)** |
