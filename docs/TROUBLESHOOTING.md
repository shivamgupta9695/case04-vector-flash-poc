# Case 04: Vector Search POC — Troubleshooting Guide

**Author:** Senior Systems Engineer  
**Date:** October 11, 2026  
**Project:** `Case04_Vector_POC`  

---

## 1. Common Operational Issues and Resolutions

### Issue 1: WSL2 DiskANN Binary Not Found
- **Symptom:** `FileNotFoundError: Missing required DiskANN artifact` or subprocess error pointing to `diskann_query`.
- **Root Cause:** WSL2 Ubuntu is not started, or the binary has not been compiled in the user profile.
- **Resolution:**
  1. Verify WSL status: `wsl --status`
  2. Verify binary existence:
     ```bash
     wsl bash -c "ls -la /home/shivamgupta/diskann-work/DiskANN/target/release/diskann_query"
     ```
  3. If missing, compile using Rust toolchain:
     ```bash
     wsl bash -c "cd /home/shivamgupta/diskann-work/DiskANN && cargo build --release --bin diskann_query"
     ```

---

### Issue 2: Milvus Collection Already Exists / Temp DB Lock
- **Symptom:** `MilvusException: collection '...' already exists` or `PermissionError: [WinError 5] Access is denied`.
- **Root Cause:** A prior process holding open file handles on `native_milvus_eval.db` inside Windows `%TEMP%` prevents file deletion.
- **Resolution:**
  - Close any dangling Python worker processes (`Get-Process python | Stop-Process`).
  - Use unique per-session SQLite database paths (implemented in Phase 10 driver via `p10_milvus_{timestamp}_{pid}.db`).

---

### Issue 3: Frozen Checksum Verification Failure
- **Symptom:** `test_01_frozen_dataset_and_source_immutability` fails.
- **Root Cause:** Unintended whitespace or code modification inside frozen Phases 1–8 files.
- **Resolution:**
  - Check which file failed in the test traceback.
  - Revert the affected file to its authoritative version.
  - Do not edit any file in `src/phase1_*.py` through `src/phase8_*.py` or `tests/test_phase3.py` through `tests/test_phase8.py`.

---

### Issue 4: Out-of-Memory (OOM) during Native Milvus Runs
- **Symptom:** High RAM consumption causing paging or process exit during native Milvus FLAT insertion.
- **Root Cause:** Native Milvus loads all vectors into full DRAM buffers (measured WorkingSet delta = 245 MB for 100k vectors).
- **Resolution:**
  - Ensure system has at least 4 GB available physical RAM.
  - In constrained environments, use the hybrid pipeline (`MilvusHybridAdapter`), which consumes only **26.54 MiB DRAM** (a 65.3% to 89.2% memory reduction).
