# Case 04: Vector Search POC — Production Runbook

**Author:** Senior Systems Engineer  
**Date:** October 11, 2026  
**Project:** `Case04_Vector_POC`  
**Version:** Phase 10 Final Submission  

---

## 1. Prerequisites and Setup Verification

### 1.1 System Requirements:
- Host OS: Windows 11 64-bit with WSL2 enabled.
- WSL Subsystem: Ubuntu 24.04 or 26.04 LTS.
- Python: Version 3.12 in `.venv`.
- Official Microsoft DiskANN binary compiled at `/home/shivamgupta/diskann-work/DiskANN/target/release/diskann_query`.

### 1.2 Verification Command:
```powershell
# Verify Python environment
.\.venv\Scripts\python.exe -c "import faiss, pymilvus, numpy; print('Core dependencies OK')"

# Verify WSL DiskANN binary
wsl bash -c "ls -lh /home/shivamgupta/diskann-work/DiskANN/target/release/diskann_query"
```

---

## 2. Standard Operational Workflows

### Workflow A: Run Full Multi-Phase Regression Suite (100 Tests)
To execute all 100 automated unit and integration tests across Phases 3 through 9:
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests/ -v
```
- **Expected Duration:** ~20–25 seconds.
- **Expected Outcome:** `Ran 100 tests ... OK` with 0 failures and 0 errors.

### Workflow B: Run Phase 10 Master Benchmark Driver (5 Trials, 9 Configurations)
To execute the multi-trial benchmark:
```powershell
.\.venv\Scripts\python.exe src/phase10_benchmark_driver.py
```
- **Expected Duration:** ~5–6 minutes.
- **Outputs Produced:**
  - `results/phase10/metrics/phase10_benchmarks.json`
  - `results/phase10/summaries/phase10_summary.json`
  - `results/phase10/trials/trial_details.json`

### Workflow C: Single Unified POC Entry Point
```powershell
# Run complete verification and benchmark suite
.\.venv\Scripts\python.exe run_poc.py --all

# Run tests only
.\.venv\Scripts\python.exe run_poc.py --test

# Run benchmarks only
.\.venv\Scripts\python.exe run_poc.py --bench
```

---

## 3. Operational Monitoring Design

> [!NOTE]
> **Monitoring Status Notice:**  
> A live Prometheus/Grafana monitoring server is not deployed in this POC environment to avoid invasive infrastructure dependencies. The metrics below represent the production telemetry contract.

### Key Performance Indicators (KPIs):
1. **Query Latency (Histogram):** Per-query latency percentiles (P50, P95, P99).
2. **Throughput (Counter):** Aggregate QPS per collection.
3. **Tier Routing Distribution (Counter):** Queries routed to `HOT`, `COLD`, or `BOTH`.
4. **SSD Seek Rate (Counter):** SSD vector candidate reads per second.
5. **DRAM Cache Hit Rate (Gauge):** Percentage of candidate vector reads satisfied from DRAM cache.
6. **Error Rate (Counter):** Count of `MilvusValidationError` and `TierUnavailableError`.
