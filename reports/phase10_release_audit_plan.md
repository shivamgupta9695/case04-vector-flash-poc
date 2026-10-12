# Case 04 — Phase 10: Release Audit Plan & Discrepancy Resolution Protocol

**Author:** Senior Systems Engineer, Vector Database Engineer, QA Lead & Independent Release Auditor  
**Date:** October 11, 2026  
**Project:** `Case04_Vector_POC` — Final Release Audit  
**Current Status:** **PROVISIONAL PASS WITH LIMITATIONS** $\rightarrow$ Transitioning to **FINAL VERIFIED RELEASE**

---

## 1. Executive Context & Objective

Phase 10 of `Case04_Vector_POC` achieved provisional validation by establishing:
1. Complete regression test execution (100/100 tests passing across Phases 3–9).
2. Cryptographic immutability verification of frozen code and dataset artifacts.
3. 5-trial interleaved benchmark execution across all 9 search configurations.
4. Operational runbook, architecture documentation, and unified CLI entry points (`run_poc.py`).

However, before final submission, an independent release audit identified several numerical, terminological, and documentation discrepancies that require systematic forensic resolution. The objective of this release audit plan is to investigate each discrepancy down to the byte and system timer level, correct documentation, verify parity independently, and deliver a mathematically sound, technically defensible release package.

---

## 2. Artifacts Inspected in Initial Forensic Pass

The following repository artifacts were inspected:
1. **Core Final Reports:**
   - `reports/CASE04_FINAL_REPORT.md` (Master 18-section capstone report)
   - `reports/phase10_validation.md` (Phase 10 validation and gate verification report)
   - `reports/phase10_environment_audit.md` (Host hardware and software environment audit)
   - `reports/phase10_integrity_report.md` (Dataset and source checksum verification)
   - `reports/phase10_regression_test_report.md` (100-test regression suite execution log)
   - `reports/phase10_benchmark_methodology.md` (Benchmark protocol and timing definitions)
   - `reports/phase10_resource_accounting.md` (DRAM, storage, and SSD I/O byte reconciliation)
   - `reports/phase10_scalability_report.md` (Workload skew and concurrency analysis)
   - `CASE04_POC_STATUS.md` (Root-level executive status dashboard)
2. **Benchmark Drivers & Output Files:**
   - `src/phase10_benchmark_driver.py` (Authoritative 5-trial benchmark runner)
   - `results/phase10/metrics/phase10_benchmarks.json` (Machine-readable benchmark metrics)
   - `results/phase10/summaries/phase10_summary.json` (Summary metrics)
   - `results/phase10/trials/trial_details.json` (45 individual trial raw measurements)
3. **Frozen Source Code & Checksum Manifests:**
   - `tests/test_phase9.py` (`FROZEN_SOURCE_CHECKSUMS` and `FROZEN_DATASET_CHECKSUMS`)
   - `tests/test_phase8.py`, `tests/test_phase7.py`, `tests/test_phase6.py` (Frozen checksum assertions)
   - `src/phase6_hybrid_search.py` (`TieredVectorStore` and `HybridVectorPipeline`)
   - `src/phase5_diskann_search.py` (`ColdDiskANNIndex` and `DiskANNSearchResult`)
   - `src/phase3_ssd_store.py` (`SSDVectorStore`)
   - `run_poc.py` (Master unified CLI driver)

---

## 3. Confirmed Discrepancies & Root-Cause Matrix

| ID | Domain | Inconsistency Observed | Root Cause | Planned Correction |
| :---: | :--- | :--- | :--- | :--- |
| **D1** | Checksum Integrity | `CASE04_FINAL_REPORT.md` lists placeholder hashes (`a351eb8d...`) for dataset files, while `test_phase9.py` and `phase10_integrity_report.md` assert `62a7ad69...`. | Copy-paste error from an external draft in Section 17.1 of `CASE04_FINAL_REPORT.md`. The actual files on disk have never changed and match `62a7ad69...` 100%. | Correct Section 17.1 in `CASE04_FINAL_REPORT.md` to display the verified, authoritative hashes matching `test_phase9.py` and `run_poc.py`. |
| **D2** | Storage Pathing | Reports reference `storage/ssd_vectors/disk.index`, but the actual physical file is located at `results/phase5_diskann/index/cold_disk_index_tuned_disk.index`. | Path abbreviation in historical reporting. `storage/ssd_vectors/vectors.bin` holds raw vectors, while the tuned Vamana index resides under `results/phase5_diskann/index/`. | Correct exact file paths and byte sizes across all reports; confirm byte sum equals 121,138,340 bytes (115.53 MiB). |
| **D3** | Throughput vs Latency | Standalone throughput is 81.43 QPS while mean latency is 82.50 ms. $1 / 0.0825\text{ s} = 12.12\text{ QPS}$, creating an apparent throughput/latency contradiction. | DiskANN3 executes with 8 parallel worker threads in WSL2 (`num_threads=8`). By Little's Law ($L = \lambda W$), multi-threaded concurrency $L \approx 6.72$ queries in flight yields $81.43\text{ QPS}$ wall-clock throughput despite an 82.50 ms individual query service time. | Document Little's Law relationship explicitly in `phase10_benchmark_methodology.md`, `phase10_benchmark_reconciliation.md`, and capstone report. |
| **D4** | 9th Benchmark Config | Summary lists 9 configurations, but only 8 distinct architectures are commonly cited. | In `phase10_benchmark_driver.py`, Config 8 is defined as `hybrid_pipeline`, which acts as a direct alias / pass-through for `standalone_hybrid` (Config 1). | Explicitly identify and document all 9 configurations, identifying `hybrid_pipeline` as the pipeline alias. |
| **D5** | Memory Accounting | Hybrid DRAM is reported as 26.54 MiB, but the relationship between FAISS `IndexHNSWFlat` vector storage and `TieredVectorStore` FP32 vectors was ambiguous. | FAISS `IndexHNSWFlat` encapsulates the 18,000 HOT vectors (8.79 MiB) internally inside its graph storage (13.46 MiB total). `TieredVectorStore` also opens the vectors in user space for reranking. | Provide distinct accounting: (1) Serialized structural index footprint (13.46 MiB), (2) Heap FP32 vector buffers (8.79 MiB), (3) PQ tables (2.63 MiB), (4) Cache/working memory (1.66 MiB), and contrast structural footprint with OS process working set. |
| **D6** | SSD I/O Description | Reports cited "50,000 SSD reads and 24.41 MiB" without stating whether these were logical requests or physical NVMe flash blocks. | `TieredVectorStore` invokes `read(512)` on an unbuffered (`buffering=0`) file descriptor 50 times per query. This represents 50,000 application-level OS read calls, not raw NAND flash page reads. | Update `phase10_resource_accounting.md` to precisely distinguish application unbuffered read system calls from physical NAND block transfers. |

---

## 4. Permitted Changes vs. Frozen Boundaries

### 4.1 Strictly Frozen (No Modifications Permitted)
- **Phases 1–9 Source Code:** `src/phase1_*.py` through `src/phase9_*.py`.
- **Phases 3–9 Test Files:** `tests/test_phase3.py` through `tests/test_phase9.py`.
- **Dataset Artifacts:** `data/base_vectors.fvecs`, `data/query_vectors.fvecs`, `data/ground_truth.ivecs`, `data/ground_truth_distances.fvecs`.
- **Storage & Index Files:** `storage/ssd_vectors/vectors.bin`, `results/phase5_diskann/index/*`.

### 4.2 Permitted Modifications & Extensions
- **Phase 10 Documentation & Reports:** All files in `reports/` and `docs/`.
- **Root Status Dashboard:** `CASE04_POC_STATUS.md`.
- **Phase 10 Drivers & Scripts:** `src/phase10_benchmark_driver.py`, `run_poc.py`, `run_poc.sh`.
- **Phase 10 Results & Evidence Files:** In `results/phase10/`.

---

## 5. Sequential Execution & Release Acceptance Gates

1. **Gate R1 (Integrity Provenance):** Verify authoritative SHA-256 hashes of all 34 frozen files, resolve the Section 17.1 discrepancy in `CASE04_FINAL_REPORT.md`, and produce `reports/phase10_integrity_report.md`.
2. **Gate R2 (Benchmark & Little's Law Reconciliation):** Reconcile 9 benchmark configurations, explain QPS vs latency mathematically via Little's Law, and generate `reports/phase10_benchmark_reconciliation.md`.
3. **Gate R3 (Adapter Bitwise Parity Audit):** Verify candidate ID parity and distance equality with raw trial outputs, and produce `reports/phase10_adapter_parity_report.md`.
4. **Gate R4 (Resource & I/O Accounting):** Reconcile memory structures, file paths, byte sizes, and unbuffered OS reads in `reports/phase10_resource_accounting.md`.
5. **Gate R5 (Regression & CLI Verification):** Run `run_poc.py --checksum`, `run_poc.py --test`, and generate `reports/phase10_regression_test_report.md`.
6. **Gate R6 (Master Capstone Regeneration):** Harmonize and update `reports/CASE04_FINAL_REPORT.md`, `reports/phase10_validation.md`, and `CASE04_POC_STATUS.md`.
7. **Gate R7 (Final Release Audit):** Produce `reports/phase10_final_release_audit.md` with final release recommendation.
