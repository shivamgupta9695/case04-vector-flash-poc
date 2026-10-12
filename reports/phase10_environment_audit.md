# Case 04 — Phase 10: Environment and Repository Audit Report

**Author:** Senior Systems Engineer & Independent Validation Auditor  
**Date:** October 11, 2026  
**Project:** `Case04_Vector_POC` — Final Phase 10 Audit  
**Status:** **GATE 10.1 PASSED** (Environment verified, all prerequisites and dependencies intact)

---

## 1. Repository Structure Summary

The repository `C:\Users\shivam.gupta\Desktop\Case04_Vector_POC` consists of the following core structural directories:

- `src/`: Core implementation modules covering dataset utilities, tiering, caching, FAISS HNSW, Microsoft DiskANN integration, hybrid coordination, query routing, reranking, and Milvus adapter.
- `tests/`: Multi-phase unit and integration test suite (`test_phase3.py` through `test_phase9.py`).
- `data/`: Frozen base vectors, query vectors, and ground truth files.
- `storage/`: Persisted tiered vector store binaries (`storage/ssd_vectors/vectors.bin`) and metadata.
- `results/`: Historical phase result directories (`phase1_1` through `phase9_milvus`), index binaries, and plots.
- `reports/`: Detailed technical validation and audit reports for Phases 5 through 9.
- `.venv/`: Preserved Windows Python 3.12 virtual environment.
- WSL Subsystem (`/mnt/c/...` and `/home/shivamgupta/diskann-work/DiskANN`): Official Microsoft DiskANN repository and compiled search binary.

---

## 2. Actual Execution Environment and Verified Hardware

A live inspection of system hardware was conducted using PowerShell CIM queries. 

> [!IMPORTANT]
> **Historical Hardware Discrepancy Resolved:**  
> Several historical reports (Phases 5, 8, and early Phase 9 drafts) cited an *"AMD Ryzen 9 7940HS (8 Cores / 16 Threads), 32 GB DDR5, WD PC SN740 NVMe SSD"*.  
> Live hardware interrogation confirms that the actual host machine is an **Intel Core i5-1145G7 with 16 GB RAM and an SK hynix BC711 NVMe SSD**. All Phase 10 benchmarks must be explicitly grounded in this actual machine profile.

### Verified Hardware Specifications:
- **Processor:** 11th Gen Intel(R) Core(TM) i5-1145G7 @ 2.60GHz (4 physical cores, 8 logical processors)
- **Host Memory:** 15.4 GB total physical RAM (~16 GB DDR4/LPDDR4x)
- **Storage Device:** BC711 NVMe SK hynix 512GB SSD (NTFS filesystem, C: drive)
- **Host OS:** Windows 11 Enterprise (Build 26200)

### Verified Subsystem Environment:
- **WSL Distribution:** Ubuntu 26.04.1 LTS
- **WSL Linux Kernel:** 6.18.33.2-microsoft-standard-WSL2 (SMP PREEMPT_DYNAMIC x86_64)
- **WSL File Bridge:** `/mnt/c/Users/shivam.gupta/Desktop/Case04_Vector_POC` mapped via `drvfs`
- **Official Microsoft DiskANN Repository:**
  - Path: `/home/shivamgupta/diskann-work/DiskANN`
  - Git Branch: `main`
  - Verified Git Commit: `907d4054 [diskann-quantization] Teach PQ training to return a BasicTable. (#1467)`
  - Query Binary: `/home/shivamgupta/diskann-work/DiskANN/target/release/diskann_query` (16,116,400 bytes, release build)

---

## 3. Detected Python Dependencies and Environment

- **Python Runtime:** Python 3.12.10 (MSC v.1943 64-bit AMD64)
- **Virtual Environment:** `C:\Users\shivam.gupta\Desktop\Case04_Vector_POC\.venv`

### Core Package Manifest:
| Package | Verified Installed Version | Purpose |
| :--- | :---: | :--- |
| `faiss-cpu` | **1.15.1** | HOT DRAM HNSW index and exact oracle ground-truth evaluation |
| `numpy` | **2.5.3** | Vector buffers, array transformations, distance computation |
| `pymilvus` | **3.0.2** | Official Milvus client SDK interface |
| `milvus-lite` | **3.2.1** | Embedded C++ Milvus search engine baseline |
| `pandas` | **3.0.6** | Tabular metrics reporting and analysis |
| `matplotlib` | **3.11.2** | Result plotting and dashboard rendering |
| `pyarrow` | **26.0.0** | High-performance serialization |

No package updates or environment re-creations are required.

---

## 4. Subsystem Entry Points

| Subsystem | Entry Point Module | Key Classes / Functions | Invocation Target |
| :--- | :--- | :--- | :--- |
| **Dataset & Ground Truth** | `src/phase1_2_evaluation.py` | `load_fvecs`, `load_ivecs`, `calculate_recall_at_k` | `data/base_vectors.fvecs`, `data/ground_truth.ivecs` |
| **HOT Tier HNSW** | `src/phase6_hybrid_search.py` | `HotHNSWIndex` | `results/phase6_hybrid/index/hot_hnsw.index` |
| **COLD Tier DiskANN** | `src/phase5_diskann_search.py` | `ColdDiskANNIndex` | `results/phase5_diskann/index/cold_disk_index_tuned` via WSL2 binary |
| **Tiered Store** | `src/phase6_hybrid_search.py` | `TieredVectorStore` | `storage/ssd_vectors/vectors.bin` |
| **Hybrid Pipeline** | `src/phase6_hybrid_search.py` | `HybridVectorPipeline` | Unifies HOT HNSW + COLD DiskANN + FP32 Reranking |
| **Query Router** | `src/phase7_query_router.py` | `AdvancedQueryRouter`, `RouterConfig` | Policy dispatcher (`ALL_TIERS`, `HOT_ONLY`, `COLD_ONLY`, `ADAPTIVE`) |
| **Merge & Rerank** | `src/phase8_reranker_validation.py` | `ExactCandidateMerger`, `ExactReranker` | ID translation ($c \rightarrow c + 18000$), exact squared-L2 distances |
| **Milvus Adapter** | `src/phase9_milvus_adapter.py` | `MilvusHybridAdapter`, `MilvusAdapterConfig` | PyMilvus-compatible collection & search gateway |

---

## 5. Summary of Historical Conflicting Claims and Reconciliations

1. **Host Hardware Specifications:**
   - *Conflict:* Prior text cited AMD Ryzen 9 7940HS 32GB; live interrogation proves Intel Core i5-1145G7 16GB with SK hynix BC711 NVMe SSD.
   - *Action for Phase 10:* Ground all reports and performance context strictly in the Intel i5-1145G7 host.
2. **Phase 9 Test Count Breakdown:**
   - *Conflict:* Phase 9 draft listed an incomplete breakdown summing to 82 tests, while claiming 100 tests.
   - *Resolution:* Reconciled to the actual test suite: 15 (P3) + 23 (P4) + 16 (P5) + 13 (P6) + 11 (P7) + 11 (P8) + 11 (P9) = 100 tests.
3. **Storage Accounting ("11.42 MB compressed table"):**
   - *Conflict:* Historical report listed "11.42 MB compressed code table", which did not match the 2.50 MB `pq_compressed.bin` file.
   - *Resolution:* Reconciled: 11.42 MB is the exact mathematical difference between the physical disk total (115.53 MB) and the logical COLD allocation (104.11 MB), representing the 8.79 MB HOT slice inside `vectors.bin` plus the 2.63 MB PQ code/centroid files.
4. **Throughput Variance (184.94 QPS vs. 156.02 QPS):**
   - *Observation:* Standalone pipeline executed at 184.94 QPS in trial 1, while adapter executed at 156.02 QPS in trial 2, despite only 10.57 $\mu$s adapter overhead.
   - *Action for Phase 10:* Implement 5 interleaved trials with a standard warm-up policy to measure steady-state variance and report medians across trials.

---

## 6. Sequential Phase 10 Execution Plan

- [x] **Step 10.1 — Repository & Environment Audit:** Generate `reports/phase10_environment_audit.md` (Gate 10.1 Passed).
- [ ] **Step 10.2 — Frozen Artifact & Dataset Integrity:** Cryptographically verify all 34 frozen files, shapes, ID mappings, and vector values; write `reports/phase10_integrity_report.md`.
- [ ] **Step 10.3 — Existing Regression Test Suite Execution:** Run all 100 tests across Phases 3–9; write `reports/phase10_regression_test_report.md`.
- [ ] **Step 10.4 — Benchmark Methodology & Protocol:** Define 5-trial interleaved protocol, warm-up rules, and timer boundaries; write `reports/phase10_benchmark_methodology.md`.
- [ ] **Step 10.5 — Master Accuracy and Performance Evaluation:** Execute multi-trial benchmark across all 9 configurations, saving raw outputs to `results/phase10/`.
- [ ] **Step 10.6 — Memory, Storage & I/O Accounting:** Reconcile logical vs. physical sizes and measured process memory; write `reports/phase10_resource_accounting.md`.
- [ ] **Step 10.7 — Scalability & Workload Validation:** Evaluate concurrency and access skew (Uniform, Zipf, Hotspot); write `reports/phase10_scalability_report.md`.
- [ ] **Step 10.8 — Operational Readiness & Documentation:** Create production runbook, architecture guide, and operational entry points.
- [ ] **Step 10.9 — Final Comprehensive Report Generation:** Compile `reports/CASE04_FINAL_REPORT.md`, `reports/phase10_validation.md`, and `CASE04_POC_STATUS.md`.
- [ ] **Step 10.10 — Independent Final Audit:** Perform final clean verification and issue the final verdict.
