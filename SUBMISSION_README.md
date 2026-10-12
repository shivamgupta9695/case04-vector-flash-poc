# Case 04: Tiered Vector Search Proof-of-Concept (POC)
## Official Submission & Reproduction Guide

---

### 1. Executive Summary & Project Objective

**Case 04** is an end-to-end Systems Engineering and Vector Database Proof-of-Concept demonstrating a high-performance **Hybrid Two-Tier Vector Retrieval Architecture**. The system addresses the severe memory-cost bottleneck of large-scale vector databases by combining:
1. **HOT Tier (DRAM)**: In-memory **FAISS HNSW** (`IndexHNSWFlat`, $M=32, efSearch=40$) storing the most active working set (18,000 vectors / 18.0% capacity).
2. **COLD Tier (NVMe SSD)**: Persistent **Microsoft DiskANN** graph index storing the remaining vectors (82,000 vectors / 82.0% capacity) with Product Quantization (PQ $M=16, K_s=256$) for in-memory graph routing and direct, unbuffered 512-byte sector reads on NVMe storage for candidate vector retrieval.
3. **Unified Candidate Merge & Exact FP32 Reranker**: Translates tier-local identifiers into a unified global namespace ($[0, 99999]$), removes duplicates, reads exact FP32 representations, computes exact squared Euclidean distances ($||q - x||^2$), and ranks the true global Top-$K$ candidates.
4. **Hot/Cold Query Router**: Policy-driven routing layer supporting `ALL_TIERS` (full correctness baseline), `HOT_ONLY` (sub-millisecond DRAM retrieval), `COLD_ONLY` (SSD evaluation), and `ADAPTIVE_THRESHOLD` (early exit on distance margin $\tau$).
5. **Milvus Integration Adapter (`MilvusHybridAdapter`)**: A gateway adapter conforming to official `pymilvus` collection management and search schemas, proving drop-in compatibility with enterprise vector database workflows.

---

### 2. Architecture & Data Flow

```text
                           CLIENT QUERY STREAM (q in R^128)
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │      MilvusHybridAdapter        │
                         │   (pymilvus Schema Conformance) │
                         └────────────────┬────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │       QueryRouter (Phase 7)     │
                         │ (ALL_TIERS / HOT_ONLY / ADAPT)  │
                         └────────┬───────────────┬────────┘
                                  │               │
                  HOT Route       │               │       COLD Route
                  (DRAM Path)     │               │       (NVMe Path)
                                  ▼               ▼
                      ┌──────────────────┐  ┌──────────────────┐
                      │    HOT HNSW      │  │  DiskANN Vamana  │
                      │  (FAISS DRAM)    │  │ (PQ Graph in RAM)│
                      │  18,000 vectors  │  │  82,000 vectors  │
                      └────────┬─────────┘  └────────┬─────────┘
                               │                     │
                               │                     ▼
                               │            ┌──────────────────┐
                               │            │  Direct SSD I/O  │
                               │            │ (512-byte reads) │
                               │            └────────┬─────────┘
                               │                     │
                               ▼                     ▼
                      ┌────────────────────────────────────────┐
                      │ Candidate Merge & Deduplication (Ph. 8)│
                      │       Global ID Range: [0, 99999]      │
                      └──────────────────┬─────────────────────┘
                                         │
                                         ▼
                      ┌────────────────────────────────────────┐
                      │   Exact FP32 Squared-L2 Reranker       │
                      │  Recomputes ||q - x||^2 from Storage   │
                      └──────────────────┬─────────────────────┘
                                         │
                                         ▼
                      ┌────────────────────────────────────────┐
                      │   Top-K Results (Bitwise Identical)    │
                      └────────────────────────────────────────┘
```

---

### 3. Environment Prerequisites & Hardware Specification

The POC was evaluated and benchmarked under the following hardware and software environment:

- **Operating System**: Windows 11 Enterprise (Build 26100) / WSL2 (Ubuntu 24.04/26.04 LTS kernel 6.6+)
- **CPU**: 11th Gen Intel(R) Core(TM) i5-1145G7 @ 2.60GHz (4 physical cores, 8 logical execution threads, AVX-512 capable)
- **Host DRAM**: 15.4 GB physical RAM
- **NVMe Storage**: BC711 NVMe SK hynix 512GB SSD
- **Python Runtime**: Python 3.12.10 (64-bit)
- **Primary Dependencies**:
  - `numpy==2.2.6`
  - `faiss-cpu==1.10.0`
  - `pymilvus==2.5.4`
  - `psutil==7.0.0`
  - `scipy==1.15.2`

---

### 4. Setup Instructions

#### Step 1: Clone or Extract Repository
Ensure you are in the repository root directory:
```powershell
cd C:\Users\shivam.gupta\Desktop\Case04_Vector_POC
```

#### Step 2: Initialize Python Virtual Environment
```powershell
# Create virtual environment using Python 3.12
python -m venv .venv

# Activate environment (PowerShell)
.\.venv\Scripts\Activate.ps1
```

#### Step 3: Install Required Dependencies
```powershell
.\.venv\Scripts\pip.exe install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
```

---

### 5. Exact Reproduction Commands

The repository provides a unified orchestrator CLI (`run_poc.py`):

#### 1. Cryptographic Checksum Verification (34 Frozen Files)
Verifies that all frozen datasets, source modules, and regression test suites match their authoritative SHA-256 digests bit-for-bit:
```powershell
.\.venv\Scripts\python.exe run_poc.py --checksum
```
*Expected Output*: `[PASS] All 34 frozen source, test, and dataset files bitwise match 100.0%.` (Exit code: `0`)

#### 2. Multi-Phase Regression Test Suite (100 Tests)
Executes all unit and integration tests across Phases 3 through 9:
```powershell
.\.venv\Scripts\python.exe run_poc.py --test
```
*Expected Output*: `Ran 100 tests in ~21s ... OK` (Exit code: `0`, 0 failures, 0 errors, 0 skips).

#### 3. Standard Test Runner (unittest discovery)
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests -v
```

#### 4. Phase 10 Multi-Trial Benchmark Execution
Executes 5 independent warm trials across all 9 engine configurations with automated statistical aggregation:
```powershell
.\.venv\Scripts\python.exe run_poc.py --bench
```
*Output Artifacts*: Saved to `results/phase10/trials/trial_details.json` and `results/phase10/metrics/phase10_benchmarks.json`.

#### 5. Full End-to-End POC Pipeline
Executes checksum verification, the 100-test regression suite, and the 5-trial benchmark suite sequentially:
```powershell
.\.venv\Scripts\python.exe run_poc.py --all
```

---

### 6. Benchmark Methodology & Scope

- **Evaluation Dataset**: 100,000 base vectors ($D=128$, FP32, Gaussian $\mathcal{N}(0, 1)$), 1,000 query vectors.
- **Top-$K$ Ground Truth**: Computed using exhaustive exhaustive Euclidean search (`faiss.IndexFlatL2`).
- **Benchmark Protocol**:
  - 5 independent warm trials per configuration.
  - 1,000 queries per trial (total: 5,000 queries evaluated per configuration).
  - Per-query latency captured via high-resolution monotonic nanosecond counters (`time.perf_counter_ns()`).
  - Strict sample-matched latency percentiles: $\min \le P_{50} \le P_{95} \le P_{99} \le \max$.
  - Unbuffered SSD read counters tracked via direct software probes on `io.FileIO` (`buffering=0`).

---

### 7. Verified Results (Authoritative Phase 10 Baseline)

The following metrics represent the **independently verified 5-trial aggregated results** traced directly to `results/phase10/trials/trial_details.json`:

| Configuration | Median QPS | Mean Latency (ms) | P50 Latency (ms) | P95 Latency (ms) | Recall@1 | Recall@10 | Median SSD Reads |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Standalone Hybrid Pipeline** | **81.43** | 82.50 | 80.50 | 112.82 | 0.8270 | 0.7397 | 50,000 |
| **Milvus Hybrid Adapter** | **81.35** | 82.51 | 80.51 | 112.91 | 0.8270 | 0.7397 | 50,000 |
| **FAISS HNSW (Full DRAM Baseline)** | 4,303.67 | 0.23 | 0.23 | 0.29 | 0.6540 | 0.5639 | 0 |
| **Router: HOT Tier Only** | 5,461.16 | 0.18 | 0.18 | 0.22 | 0.1780 | 0.1554 | 0 |
| **Router: COLD Tier Only** | 81.48 | 82.58 | 79.67 | 100.62 | 0.6490 | 0.5843 | 50,000 |
| **Router: ALL TIERS** | 84.54 | 77.36 | 76.33 | 94.21 | 0.8270 | 0.7397 | 50,000 |
| **Adaptive Threshold Router** | 94.28 | 66.23 | 74.97 | 97.25 | 0.7530 | 0.6479 | 42,050 |
| **Native Milvus Flat (Brute Force)** | 10.43 | 95.91 | 94.95 | 113.17 | 1.0000 | 1.0000 | 0 |

#### Key Performance & Resource Findings:
1. **Milvus Adapter Parity**: 100.0% bitwise candidate ID match with Standalone Hybrid Pipeline. Maximum distance difference is exactly `0.0` across all 50,000 evaluated candidate slots (1,000 queries $\times K=10 \times 5$ trials).
2. **Adapter Translation Overhead**: Measured at $10.57\ \mu\text{s}$ ($0.01057\text{ ms}$) in unit microbenchmarks, contributing $< 0.02\%$ latency overhead in end-to-end execution.
3. **Memory Footprint & DRAM Reduction**:
   - Total hybrid active component DRAM: **26.54 MiB = 27.83 MB** (HOT HNSW: 13.46 MiB, Vector Store Cache: 8.79 MiB, DiskANN PQ Table: 2.63 MiB, Metadata: 1.66 MiB).
   - **63.6% DRAM Reduction** compared to the theoretical in-memory FP32 + HNSW index floor (76.50 MB).
   - **88.7% DRAM Reduction** compared to measured FAISS in-memory process working set delta (245.43 MB).
4. **SSD I/O Characteristics**: 50,000 application-level 512-byte vector reads per 1,000 queries on an unbuffered file handle, delivering 24.41 MiB of payload data across NVMe storage.

#### Reconciliation Against Historical Claims:
- **Phase 6 Preliminary QPS (184.94)**: Measured under earlier unbuffered I/O instrumentation and single-trial runs; Phase 10 establishes the authoritative 5-trial median at 81.43 QPS.
- **Phase 9 Preliminary QPS (156.02)**: Attributable to cold-start execution differences; Phase 10 confirms steady-state median throughput of 81.35 QPS.

---

### 8. Explicitly Disclosed Limitations

In accordance with strict scientific integrity principles, the following operational and architectural boundaries are explicitly disclosed:

1. **WSL2 `/mnt/c` Filesystem Bridge Overhead**:
   The Microsoft DiskANN C++ shared library is compiled under Linux and accessed through the WSL2 `/mnt/c` cross-filesystem translation layer (9P/drvfs). File I/O operations traverse this inter-OS boundary, introducing non-negligible latency overhead compared to bare-metal Linux with direct asynchronous I/O (`io_uring` / `O_DIRECT`) on raw NVMe block devices.

2. **Client-Tier Milvus Gateway Architecture**:
   The `MilvusHybridAdapter` operates as a client-side gateway implementing official `pymilvus` collection and search APIs. It routes queries to the underlying hybrid vector engine rather than functioning as an in-process C++ Knowhere index plugin inside the Milvus server binary.

3. **100,000-Vector Evaluation Scope**:
   All benchmarks were performed on a 100,000-vector dataset ($D=128$, 51.2 MB base vector storage). While mathematically sound for establishing tiering invariants, cache migration stability, and bitwise recall accuracy, multi-terabyte NVMe queue depth saturation requires scaling to 10M–1B vectors.

4. **Phase 9 Source Manifest Coverage Boundary**:
   The authoritative 34-file SHA-256 integrity manifest in `tests/test_phase9.py` covers 5 data files, 25 Phase 1–8 source files, and 4 test files. `src/phase9_milvus_adapter.py` was created during Phase 9 and lacks a pre-Phase-10 birth hash in historical manifests. Rather than generating a retrospective hash and claiming historical baseline verification, this is formally disclosed as an unmanifested boundary (Sub-Gate 10.2B BLOCKED).

---

### 9. Document Navigation & Core Reports

- **Master Final Report**: [`reports/CASE04_FINAL_REPORT.md`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/reports/CASE04_FINAL_REPORT.md)
- **Phase 10 Release Audit**: [`reports/phase10_final_release_audit.md`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/reports/phase10_final_release_audit.md)
- **Benchmark Reconciliation**: [`reports/phase10_benchmark_reconciliation.md`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/reports/phase10_benchmark_reconciliation.md)
- **Resource Accounting Report**: [`reports/phase10_resource_accounting.md`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/reports/phase10_resource_accounting.md)
- **Cryptographic Integrity Report**: [`reports/phase10_integrity_report.md`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/reports/phase10_integrity_report.md)
- **Production Architecture Guide**: [`docs/ARCHITECTURE.md`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/docs/ARCHITECTURE.md)
- **Submission Manifest**: [`SUBMISSION_MANIFEST.md`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/SUBMISSION_MANIFEST.md)
