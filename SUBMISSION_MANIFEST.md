# Case 04: Project Submission Manifest
## Complete Repository Inventory & Packaging Classification

This manifest enumerates the core repository assets of `Case04_Vector_POC`, specifying the file purpose, artifact classification, requirement for full system reproduction, and safe archival status.

---

### 1. Root Orchestration & Project Configuration

| Relative Path | File Purpose | Category | Required for Reproduction | Safe to Exclude from Archive |
|:---|:---|:---:|:---:|:---:|
| `run_poc.py` | Unified Master CLI orchestrator (`--checksum`, `--test`, `--bench`, `--all`) | Source / Orchestrator | **YES** | NO |
| `run_poc.sh` | Bash wrapper script for Unix/WSL environments | Source / Orchestrator | **YES** | NO |
| `requirements.txt` | Python runtime dependencies specification | Configuration | **YES** | NO |
| `.gitignore` | Git exclusion rules for artifacts and virtual environments | Configuration | **YES** | NO |
| `SUBMISSION_README.md` | Authoritative submission guide, reproduction steps, verified metrics | Documentation | **YES** | NO |
| `SUBMISSION_MANIFEST.md` | Comprehensive file-by-file repository manifest and classification | Documentation | **YES** | NO |
| `README.md` | Foundational Phase 1–3 architecture documentation | Documentation | **YES** | NO |
| `CASE04_POC_STATUS.md` | Current status, verified milestones, and execution gate ledger | Report | Optional | NO |
| `PHASE3_AUDIT_REPORT.md` | Historical Phase 3 cache thrashing and latency audit | Report | Optional | Safe to Exclude |
| `PHASE4_IMPLEMENTATION_REPORT.md` | Historical Phase 4 PQ compression and tier partition report | Report | Optional | Safe to Exclude |
| `phase3_hardened_implementation_report.md` | Historical Phase 3.1 hardening and hysteresis damping report | Report | Optional | Safe to Exclude |

---

### 2. Core Python Source Code (`src/`)

All files in `src/` are required for full reproduction of Phases 1 through 10:

| Relative Path | File Purpose | Category | Required for Reproduction | Safe to Exclude from Archive |
|:---|:---|:---:|:---:|:---:|
| `src/phase1_1_ground_truth.py` | Ground truth generator using FAISS FlatL2 brute-force search | Source | **YES** | NO |
| `src/phase1_2_evaluation.py` | Standalone recall and evaluation metric calculations | Source | **YES** | NO |
| `src/phase2_ann_hnsw.py` | FAISS HNSW baseline benchmark implementation | Source | **YES** | NO |
| `src/phase3_hotness.py` | Frequency-decay hotness scoring and tracking model | Source | **YES** | NO |
| `src/phase3_realtime.py` | Dynamic query streaming, cache admission, and migration engine | Source | **YES** | NO |
| `src/phase3_ssd_store.py` | Unbuffered direct file-backed SSD vector storage engine | Source | **YES** | NO |
| `src/phase3_tiering.py` | Tier manager orchestrating DRAM cache and SSD store | Source | **YES** | NO |
| `src/phase3_workload.py` | Deterministic synthetic workload generator (Uniform, Zipf, Hotspot) | Source | **YES** | NO |
| `src/inspect_phase3_tiers.py` | Inspection utility for Phase 3 memory residency | Source | Optional | Safe to Exclude |
| `src/inspect_phase3_live_tiers.py`| Live tier inspection and verification utility | Source | Optional | Safe to Exclude |
| `src/diagnostic_10pct_uniform.py`| Diagnostic script for low-cache uniform workload behavior | Source | Optional | Safe to Exclude |
| `src/diskann_query.rs` | Experimental Rust DiskANN query binding prototype | Source / Experimental | Optional | Safe to Exclude |
| `src/phase4_pq.py` | Faiss Product Quantizer training and encoding coordinator | Source | **YES** | NO |
| `src/phase4_pq_adc.py` | Asymmetric Distance Computation (ADC) search routines | Source | **YES** | NO |
| `src/phase4_pq_encode.py` | Standalone PQ vector compression encoder | Source | **YES** | NO |
| `src/phase4_pq_evaluate.py` | Standalone PQ reconstruction distortion evaluator | Source | **YES** | NO |
| `src/phase4_pq_inspect.py` | PQ code inspection and validation utility | Source | Optional | Safe to Exclude |
| `src/phase4_pq_reconstruct.py` | PQ vector decompression and reconstruction engine | Source | **YES** | NO |
| `src/phase4_pq_tier_model.py` | Static 18/82 Hot/Cold tier partitioner | Source | **YES** | NO |
| `src/phase4_pq_train.py` | K-means subquantizer codebook training | Source | **YES** | NO |
| `src/phase5_prepare_cold.py` | DiskANN binary data and local ground truth preparation | Source | **YES** | NO |
| `src/phase5_orchestrator.py` | End-to-end DiskANN index build and search orchestrator | Source | **YES** | NO |
| `src/phase5_diskann_search.py` | DiskANN C++ shared library wrapper and search adapter | Source | **YES** | NO |
| `src/phase6_hybrid_search.py` | Hybrid Vector Pipeline (HOT FAISS + COLD DiskANN + Reranker) | Source | **YES** | NO |
| `src/phase7_query_router.py` | Policy-based Hot/Cold Query Router | Source | **YES** | NO |
| `src/phase8_reranker_validation.py`| Candidate union merging and exact FP32 reranker engine | Source | **YES** | NO |
| `src/phase9_milvus_adapter.py` | Milvus-compatible Hybrid Adapter (`MilvusHybridAdapter`) | Source | **YES** | NO |
| `src/phase10_benchmark_driver.py`| Multi-trial benchmark runner and statistical aggregator | Source | **YES** | NO |

---

### 3. Automated Regression Test Suite (`tests/`)

| Relative Path | File Purpose | Category | Required for Reproduction | Safe to Exclude from Archive |
|:---|:---|:---:|:---:|:---:|
| `tests/test_phase3.py` | Phase 3 production hardening and cache invariant test suite (15 tests) | Test | **YES** | NO |
| `tests/test_phase4.py` | Phase 4 PQ compression and tier partition test suite (23 tests) | Test | **YES** | NO |
| `tests/test_phase5.py` | Phase 5 DiskANN index and global ID mapping test suite (16 tests) | Test | **YES** | NO |
| `tests/test_phase6.py` | Phase 6 hybrid search and candidate merge test suite (13 tests) | Test | **YES** | NO |
| `tests/test_phase7.py` | Phase 7 query router policy and threshold test suite (11 tests) | Test | **YES** | NO |
| `tests/test_phase8.py` | Phase 8 candidate union and exact reranker test suite (11 tests) | Test | **YES** | NO |
| `tests/test_phase9.py` | Phase 9 Milvus adapter schema and parity test suite (11 tests) | Test | **YES** | NO |

---

### 4. Technical Documentation (`docs/`)

| Relative Path | File Purpose | Category | Required for Reproduction | Safe to Exclude from Archive |
|:---|:---|:---:|:---:|:---:|
| `docs/ARCHITECTURE.md` | System design, tier interactions, and data structures | Documentation | **YES** | NO |
| `docs/PRODUCTION_RUNBOOK.md` | Deployment procedures, operations, and recovery workflows | Documentation | **YES** | NO |
| `docs/TROUBLESHOOTING.md` | Diagnostic guides for cache and benchmark issues | Documentation | Optional | NO |

---

### 5. Final Audit & Validation Reports (`reports/`)

| Relative Path | File Purpose | Category | Required for Reproduction | Safe to Exclude from Archive |
|:---|:---|:---:|:---:|:---:|
| `reports/CASE04_FINAL_REPORT.md` | Master capstone engineering report for Case 04 | Report | **YES** | NO |
| `reports/phase10_final_release_audit.md` | Independent Phase 10 release audit and gate ledger | Report | **YES** | NO |
| `reports/phase10_benchmark_reconciliation.md` | Benchmark reconciliation against raw trial evidence | Report | **YES** | NO |
| `reports/phase10_resource_accounting.md` | Rigorous DRAM and NVMe storage footprint audit | Report | **YES** | NO |
| `reports/phase10_integrity_report.md` | Cryptographic SHA-256 verification and boundary report | Report | **YES** | NO |
| `reports/phase10_validation.md` | Phase 10 milestone validation summary | Report | **YES** | NO |
| `reports/phase10_adapter_parity_report.md` | Milvus adapter bitwise parity and overhead proof | Report | **YES** | NO |
| `reports/phase10_benchmark_methodology.md` | Statistical benchmarking protocol specification | Report | **YES** | NO |
| `reports/phase10_environment_audit.md` | Hardware, OS, and compiler environment audit | Report | **YES** | NO |
| `reports/phase10_regression_test_report.md` | 100-test multi-phase regression audit | Report | **YES** | NO |
| `reports/phase10_release_audit_plan.md` | Audit framework and verification procedure plan | Report | Optional | Safe to Exclude |
| `reports/phase10_scalability_report.md` | Extrapolation analysis to billion-scale datasets | Report | Optional | Safe to Exclude |
| `reports/phase09_validation.md` | Phase 9 Milvus integration validation report | Report | Optional | Safe to Exclude |
| `reports/phase08_validation.md` | Phase 8 candidate union reranking report | Report | Optional | Safe to Exclude |
| `reports/phase07_validation.md` | Phase 7 query router policy evaluation report | Report | Optional | Safe to Exclude |
| `reports/phase06_validation.md` | Phase 6 hybrid search pipeline report | Report | Optional | Safe to Exclude |
| `reports/phase05_validation.md` | Phase 5 DiskANN integration validation report | Report | Optional | Safe to Exclude |
| `reports/phase05_final_audit_report.md` | Phase 5 DiskANN index audit report | Report | Optional | Safe to Exclude |

---

### 6. Frozen Datasets & Base Storage (`data/`, `storage/`)

| Relative Path | File Purpose | Category | Required for Reproduction | Safe to Exclude from Archive |
|:---|:---|:---:|:---:|:---:|
| `data/base_vectors.fvecs` | 100,000 base vectors ($D=128$, float32, 51.6 MB) | Dataset | **YES** | NO |
| `data/query_vectors.fvecs` | 1,000 query vectors ($D=128$, float32, 516 KB) | Dataset | **YES** | NO |
| `data/ground_truth.ivecs` | Top-10 exact ground-truth IDs ($1000 \times 10$, int32, 44 KB) | Dataset | **YES** | NO |
| `data/ground_truth_distances.fvecs` | Top-10 exact ground-truth squared-L2 distances (44 KB) | Dataset | **YES** | NO |
| `storage/ssd_vectors/vectors.bin` | Direct binary vector store for Phase 3/6/8/10 (51.2 MB) | Storage / Dataset | **YES** | NO |
| `storage/metadata/manifest.json` | Offset and size mapping manifest for `vectors.bin` | Configuration | **YES** | NO |

---

### 7. DiskANN Index & Prepared Artifacts (`results/phase5_diskann/`)

These persisted binary index artifacts are required for immediate execution of COLD-tier queries without re-indexing:

| Relative Path | File Purpose | Category | Required for Reproduction | Safe to Exclude from Archive |
|:---|:---|:---:|:---:|:---:|
| `results/phase5_diskann/index/cold_disk_index_disk.index` | Built DiskANN Vamana graph index on SSD (55.98 MB) | Index Artifact | **YES** | NO |
| `results/phase5_diskann/index/cold_disk_index_pq_compressed.bin`| PQ compressed vector table for DiskANN (1.31 MB) | Index Artifact | **YES** | NO |
| `results/phase5_diskann/index/cold_disk_index_pq_pivots.bin` | PQ centroid pivots for DiskANN (135 KB) | Index Artifact | **YES** | NO |
| `results/phase5_diskann/index/cold_disk_index_tuned_disk.index` | Tuned parameter DiskANN graph index (67.18 MB) | Index Artifact | Optional | Safe to Exclude |
| `results/phase5_diskann/index/cold_disk_index_tuned_pq_compressed.bin`| Tuned PQ compressed vectors (2.62 MB) | Index Artifact | Optional | Safe to Exclude |
| `results/phase5_diskann/index/cold_disk_index_tuned_pq_pivots.bin`| Tuned PQ pivots (135 KB) | Index Artifact | Optional | Safe to Exclude |
| `results/phase5_diskann/prepared/cold_vectors.bin` | Prepared 82,000 COLD vector binary (41.98 MB) | Prepared Dataset | **YES** | NO |
| `results/phase5_diskann/prepared/query_vectors.bin` | Prepared 1,000 query vector binary (512 KB) | Prepared Dataset | **YES** | NO |
| `results/phase5_diskann/prepared/cold_groundtruth_local.bin` | Local COLD Top-100 ground truth (400 KB) | Prepared Ground Truth | **YES** | NO |
| `results/phase5_diskann/prepared/cold_groundtruth_global.ivecs`| Global COLD Top-100 ground truth (404 KB) | Prepared Ground Truth | Optional | Safe to Exclude |
| `results/phase5_diskann/metadata/id_mapping.json` | Bijective global <-> local ID mapping table (984 KB) | Configuration | **YES** | NO |
| `results/phase5_diskann/metadata/disk_index_config.json` | DiskANN build parameters configuration | Configuration | **YES** | NO |

---

### 8. Phase 10 Benchmark Evidence (`results/phase10/`)

| Relative Path | File Purpose | Category | Required for Reproduction | Safe to Exclude from Archive |
|:---|:---|:---:|:---:|:---:|
| `results/phase10/trials/trial_details.json` | Raw measurements for all 45 individual benchmark trials | Benchmark Evidence | **YES** | NO |
| `results/phase10/metrics/phase10_benchmarks.json` | Aggregated statistical metrics (QPS, percentiles, recall) | Benchmark Evidence | **YES** | NO |
| `results/phase10/summaries/phase10_summary.json` | High-level executive benchmark summary | Benchmark Evidence | **YES** | NO |

---

### 9. Product Quantization Artifacts (`results/phase4_pq/`)

| Relative Path | File Purpose | Category | Required for Reproduction | Safe to Exclude from Archive |
|:---|:---|:---:|:---:|:---:|
| `results/phase4_pq/codebooks/pq_codebook_M16_Ks256.npy` | Trained subquantizer centroid codebook | Index / ML Model | **YES** | NO |
| `results/phase4_pq/codes/pq_codes_M16_Ks256.npy` | Quantized uint8 codes for all 100k vectors | Index Artifact | **YES** | NO |
| `results/phase4_pq/metadata/tier_partition_18_82.json` | Authoritative HOT (18k) / COLD (82k) partition list | Configuration | **YES** | NO |
| `results/phase4_pq/diagnostics/*` | PQ distortion and error evaluation diagnostics | Evidence | Optional | Safe to Exclude |
| `results/phase4_pq/plots/*` | High-resolution publication plots for PQ distortion | Evidence | Optional | Safe to Exclude |

---

### 10. Ephemeral / Runtime Directories (Safe to Exclude from Submission Archive)

| Relative Path | File Purpose | Category | Action for Archive |
|:---|:---|:---:|:---:|
| `.venv/` | Python local virtual environment binaries and site-packages | Runtime Environment | **EXCLUDE** (Recreate via `pip install -r requirements.txt`) |
| `__pycache__/` | Python compiled bytecode (`*.pyc`) | Cache | **EXCLUDE** |
| `test_milvus_lite.db/` | Ephemeral SQLite database generated during Milvus Lite test runs | Ephemeral State | **EXCLUDE** (Automatically regenerated at test time) |
