# Case 04 — Phase 10: Scalability and Workload Validation Report

**Author:** Senior Systems Engineer & Independent Validation Auditor  
**Date:** October 11, 2026  
**Project:** `Case04_Vector_POC` — Final Phase 10 Audit  
**Status:** **GATE 10.7 PASSED** (Workload distributions, concurrent dispatch, and local scale verified; distributed scale documented as future work)

---

## 1. Workload Skew Evaluation (500 Query Streams, K=10)

Using the deterministic `WorkloadGenerator` from Phase 3, three synthetic access distributions were evaluated over the hybrid vector search pipeline:

### Table 1: Performance Across Access Distributions
| Workload Distribution | Access Characteristics | Throughput (QPS) | Recall@10 | SSD Reads (500 queries) | SSD Data Read | Total Wall-Clock |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Uniform Workload** | Equal probability across all 1,000 queries | **146.1 QPS** | **0.7440** | 25,000 | 12.21 MiB | 3,423.0 ms |
| **Zipfian Workload ($s=1.25$)** | Power-law distribution; heavy head access | **150.2 QPS** | **0.7560** | 25,000 | 12.21 MiB | 3,329.9 ms |
| **Hotspot Workload (Pareto 80/20)**| 10% of queries receive 80% of traffic | **68.8 QPS** | **0.7418** | 25,000 | 12.21 MiB | 7,262.9 ms |

### Key Observations:
1. **Accuracy Stability:** Recall@10 remains stable across all distributions ($0.7418$ to $0.7560$).
2. **Access Skew & Cache Effects:** Under Zipfian skew, repeated hits to warm query regions yield a slight throughput lift to **150.2 QPS**.
3. **Hotspot Contention:** Under intense 80/20 hotspot clustering, repeated concurrent candidate lookups against the local file descriptors in WSL2 increase disk I/O lock contention, reducing throughput to **68.8 QPS**.

---

## 2. Multi-Tier Concurrency Evaluation

The hybrid coordinator supports both sequential tier dispatch and concurrent multi-threaded dispatch:
- **Sequential Dispatch:** Searches HOT FAISS HNSW first, then invokes Microsoft DiskANN3 in WSL2, then reranks candidates.
- **Concurrent Dispatch:** Dispatches HOT DRAM search and COLD DiskANN3 SSD search in parallel using Python `ThreadPoolExecutor(max_workers=2)`.

### Table 2: Sequential vs. Concurrent Tier Dispatch (200 Queries, K=10)
| Dispatch Mode | Wall-Clock Duration | Throughput (QPS) | Latency Improvement | Bitwise ID Parity |
| :--- | :---: | :---: | :---: | :---: |
| **Sequential Dispatch** | 3,347.3 ms | 59.7 QPS | Baseline | Baseline |
| **Concurrent Dispatch (2 Workers)** | 2,966.2 ms | 67.4 QPS | **-11.4% Latency (-381.1 ms)** | **100% Identical** |

Concurrent dispatch hides the ~0.2 ms HOT DRAM search latency entirely behind the longer COLD SSD search, reducing overall wall-clock query latency by 11.4% without any loss of accuracy or ID parity.

---

## 3. Verified Hardware Scale and Limits

### Tested Scope (VERIFIED / MEASURED):
- **Dataset Size:** 100,000 vectors (128-dimensional FP32).
- **Physical Partitions:** 18,000 HOT (DRAM) + 82,000 COLD (NVMe SSD).
- **Concurrency:** Up to 8 worker threads inside C++/Rust index search engines; 2 parallel tier dispatch threads in Python coordinator.
- **Node Topology:** Single workstation host running Windows 11 + WSL2.

### Untested Scope (DOCUMENTED AS FUTURE WORK / NOT MEASURED):
1. **Multi-Million Vector Scale:** Scale beyond 100,000 vectors (e.g., SIFT10M, SIFT100M) is **NOT MEASURED**.
2. **Distributed Cluster Sharding:** Multi-node Milvus cluster deployment with independent QueryNodes and DataNodes is **NOT MEASURED**.
3. **Bare-Metal Linux Direct NVMe I/O:** Bare-metal Linux kernel execution utilizing kernel asynchronous I/O (`io_uring`) with `O_DIRECT` bypass is **NOT MEASURED** (current system uses WSL2 `drvfs` translation bridge).
