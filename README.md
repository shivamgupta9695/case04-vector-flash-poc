# Case 04: Vector Search Proof of Concept (POC)

## Phase 1.1 — Deterministic Dataset & Exact FAISS Ground Truth

### 1. Purpose of Case 04
Case 04 is a comprehensive investigation and implementation proof-of-concept for high-performance Vector Search and Nearest Neighbor retrieval. The project evaluates vector indexing architectures, exact brute-force baselines, and approximate nearest neighbor (ANN) graph/quantization structures (such as HNSW, IVF, PQ) in terms of recall, query latency, index construction time, and memory footprint.

### 2. Purpose of Phase 1.1
Phase 1.1 establishes the foundational, deterministic testbed and exact ground-truth nearest-neighbor baseline against which all subsequent ANN algorithms and custom C++ search implementations will be evaluated. Without a verified, frozen ground truth, precision and recall metrics cannot be scientifically measured.

### 3. Exact Search vs. Approximate Search
- **Exact Search (Exhaustive k-NN):** Evaluates distances between each query vector and all $N$ base vectors ($O(N \cdot D)$ time complexity per query). It guarantees 100% recall and finds the true closest neighbors according to the chosen distance metric.
- **Approximate Search (ANN):** Trades off theoretical recall guarantees for sub-linear query time ($O(\log N)$ or $O(1)$ amortized) using clustering (IVF), graph navigation (HNSW), or vector compression (PQ).
Phase 1.1 strictly implements exact exhaustive search to produce the golden reference dataset.

### 4. Why `IndexFlatL2` is Used
`faiss.IndexFlatL2` stores raw vector representations without quantization, compression, or graph approximations. Search operations execute continuous batch GEMM / BLAS matrix multiplications computing exhaustive pairwise L2 distances with full SIMD acceleration. It serves as the trusted reference oracle for exact ground truth.

### 5. Dataset Configuration
- **Base Vectors ($N_{base}$):** 100,000
- **Query Vectors ($N_{query}$):** 1,000
- **Dimension ($D$):** 128
- **Top-$K$ ($K$):** 10
- **Data Type:** 32-bit floating point (`float32` / IEEE 754 single precision)
- **Distribution:** Standard Gaussian $\mathcal{N}(0, 1)$

### 6. Random Seeds
To ensure complete cross-platform reproducibility without system entropy or clock-dependent variance:
- **`BASE_SEED = 42`**: Drives NumPy's modern Generator (`np.random.default_rng(42)`) for base vector generation.
- **`QUERY_SEED = 43`**: Independently seeds query vector generation (`np.random.default_rng(43)`).

### 7. L2 Squared Distance Semantics
`faiss.IndexFlatL2` reports **squared Euclidean distance**:
$$d^2(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{D} (x_i - y_i)^2$$
- No square root is applied.
- Because $f(z) = \sqrt{z}$ is strictly monotonic for $z \ge 0$, squared Euclidean distance preserves identical topological sorting and identical neighbor ranking as Euclidean distance:
  $$\operatorname{argsort}(d^2(\mathbf{x}, \mathbf{y}_j)) \equiv \operatorname{argsort}(d(\mathbf{x}, \mathbf{y}_j))$$
- Avoiding square-root operations reduces floating-point roundoff errors and computational overhead.

### 8. Ground-Truth Meaning
The ground-truth file (`ground_truth.ivecs`) contains the exact indices of the top-10 nearest base vectors for each of the 1,000 queries (shape: $1000 \times 10$). These IDs represent the absolute topological truth. When evaluating future ANN algorithms, recall@K is computed directly by checking the fraction of returned IDs present in this matrix.

### 9. `.fvecs` Binary File Format
Standard binary format used by benchmark datasets (e.g., SIFT, GIST, Deep1B):
For each vector $i \in [0, N-1]$:
```
[int32: dimension d] [float32 * d: vector components]
```
- Total bytes per vector: $4 + (4 \times D)$ bytes.
- For 100,000 vectors of $D=128$: $516 \times 100,000 = 51,600,000$ bytes (~51.6 MB).
- For 1,000 vectors of $D=128$: $516 \times 1,000 = 516,000$ bytes (~516 KB).

### 10. `.ivecs` Binary File Format
Binary format for integer index matrices:
For each vector $i \in [0, N-1]$:
```
[int32: dimension d] [int32 * d: neighbor IDs]
```
- Total bytes per vector: $4 + (4 \times K)$ bytes.
- For 1,000 queries of $K=10$: $44 \times 1,000 = 44,000$ bytes (~44 KB).

### 11. How to Activate `.venv`
On Windows (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```
Or use the direct python executable:
```powershell
.\.venv\Scripts\python.exe
```

### 12. How to Run Phase 1.1
From the project root (`Case04_Vector_POC`):
```powershell
.\.venv\Scripts\python.exe src/phase1_1_ground_truth.py
```

### 13. Validation Methodology
The runner enforces 15 automated validation gates on every execution:
1. **Base Shape:** Verified strictly equal to `(100000, 128)`.
2. **Query Shape:** Verified strictly equal to `(1000, 128)`.
3. **Data Type:** Verified strictly equal to `float32`.
4. **Top-$K$:** Verified equal to `10`.
5. **Index Size:** Verified `index.ntotal == 100000`.
6. **Search Output Dimensions:** Verified labels shape `(1000, 10)` and distances shape `(1000, 10)`.
7. **ID Range Integrity:** Verified every label $l \in [0, 99999]$.
8. **Numerical Finiteness:** Verified zero `NaN` and zero `Inf` values across base vectors, queries, and distances.
9. **Disk Persistence:** Verified physical existence of all generated files.
10. **File Read-Back Integrity:** Binary files are read back from disk and verified to match in-memory data bit-for-bit.
11. **Independent Brute-Force Check:** For queries 0, 1, 42, 500, and 999, independent NumPy exhaustive distance computations (`np.sum((base - query)**2, axis=1)`) verify that `np.argmin` matches FAISS rank-0 ID and distance within floating-point tolerance ($10^{-5}$).
12. **Determinism:** Verifies that SHA-256 binary digests match previous runs.

### 14. Reproducibility Methodology & Verified Checksums
Every run computes SHA-256 cryptographic hashes on the generated binary files:
- **`data/base_vectors.fvecs`**: `62a7ad69bc429570fb4819287bb4d317d8069fb784ef528f370c64189d37198b`
- **`data/query_vectors.fvecs`**: `c1d80d61714296078229bc3429a79569113441c01fdd1069de21611a30528496`
- **`data/ground_truth.ivecs`**: `ee2f420a3fea4e71408756f6d0bf7bed1741a69dafc119b2781f7b4142c7d66f`
- **`data/ground_truth_distances.fvecs`**: `3ca693e88fdfab960b0d43412b8a66266259407615feeb1a1f61ffc083b0ffc4`

### 15. Why the Generated Dataset Must be Frozen for Future Phases
A scientific benchmark requires a stationary ground truth. If base or query vectors change across experiments:
- Recall comparisons between custom implementations (e.g. C++ brute-force, HNSW, IVF) become invalid.
- Cache behaviors and index topology change unpredictably.
- Reproducibility across multiple machines and execution runs is compromised.
The generated files in `data/` are frozen and will serve as the invariant benchmark assets across all subsequent phases.

---

## Phase 1.2 — Evaluation Framework: Recall@K + Latency + QPS

### 1. Purpose of the Evaluation Framework
The evaluation framework establishes an algorithm-agnostic, standardized measurement harness. By decoupling the evaluation metrics (Recall@K, latency, QPS, data validation) from specific index internals, any vector search implementation (custom C++ brute-force, HNSW, IVF, ScaNN, DiskANN) can be tested and compared directly against identical criteria.

### 2. Definition of Recall@K
Recall@K measures the proportion of true nearest neighbors retrieved by a search algorithm within its top-$K$ candidate set relative to the exact ground-truth nearest neighbors:
$$\text{Recall@K} = \frac{|\text{GroundTruth}_{1..K} \cap \text{Retrieved}_{1..K}|}{K}$$
A score of 1.0 (100%) indicates perfect overlap with the exact exhaustive nearest neighbors.

### 3. Recall Calculation & Deduplication
For a benchmark across $N_{query}$ queries:
1. For query $i$, retrieve predicted candidate IDs $P_i$ and ground-truth IDs $GT_i$.
2. Deduplicate predicted candidates using unique set intersection: $\text{Hits}_i = |GT_i[0..K-1] \cap \operatorname{set}(P_i[0..K-1])|$. This prevents algorithms with duplicate/corrupt predictions from artificially inflating recall scores.
3. Calculate query recall: $R_i = \frac{\text{Hits}_i}{K}$.
4. Overall Recall@K is the arithmetic mean across all queries:
   $$\text{Recall@K} = \frac{1}{N_{query}} \sum_{i=0}^{N_{query}-1} R_i = \frac{\sum_{i=0}^{N_{query}-1} \text{Hits}_i}{N_{query} \cdot K}$$
The implementation supports arbitrary $K$ values ($K=1, 5, 10$).

### 4. Search Latency
Search latency measures the wall-clock execution time taken by the search routine, strictly isolated from:
- Dataset disk loading
- In-memory index building / graph construction
- Output file serialization
- Post-hoc metric calculation
High-resolution timing is captured using Python's `time.perf_counter()`.

### 5. Queries Per Second (QPS)
QPS quantifies retrieval throughput under steady-state conditions:
$$\text{QPS} = \frac{N_{query}}{T_{\text{search}}}$$
Where $T_{\text{search}}$ is the measured batch search execution time in seconds.

### 6. Batch vs. Per-Query Latency
- **Batch Search Latency (Throughput):** Queries are issued simultaneously in a single matrix call ($1,000 \times 128$). This utilizes multi-threaded BLAS/SIMD parallelism across CPU cores. Average amortized per-query latency is:
  $$\text{Latency}_{\text{avg}} = \frac{T_{\text{batch}}}{N_{query}} \times 1000 \quad (\text{ms})$$
- **Individual Query Latency (Responsiveness):** Queries are dispatched individually ($1 \times 128$) to measure tail latencies without batch amortization. This reports percentile metrics: **p50**, **p95**, and **p99**.

### 7. Why `IndexFlatL2` Produces Recall@10 = 100%
Because `faiss.IndexFlatL2` computes exact squared Euclidean distance across all 100,000 base vectors without approximations, its retrieved candidate ranking is identical to the frozen ground truth (`ground_truth.ivecs`). Testing `IndexFlatL2` against the ground truth confirms evaluator correctness: any result below 100.0% would indicate a bug in the evaluation logic.

### 8. Preservation of Frozen Phase 1.1 Dataset
The evaluation harness strictly verifies that the Phase 1.1 binary files have not been modified, overwritten, or re-generated. Prior to benchmarking, cryptographic SHA-256 digests are validated against the expected reference signatures:
- `base_vectors.fvecs`: `62a7ad69bc429570fb4819287bb4d317d8069fb784ef528f370c64189d37198b`
- `query_vectors.fvecs`: `c1d80d61714296078229bc3429a79569113441c01fdd1069de21611a30528496`
- `ground_truth.ivecs`: `ee2f420a3fea4e71408756f6d0bf7bed1741a69dafc119b2781f7b4142c7d66f`
- `ground_truth_distances.fvecs`: `3ca693e88fdfab960b0d43412b8a66266259407615feeb1a1f61ffc083b0ffc4`

### 9. How Future ANN Algorithms Will Plug into the Evaluator
The evaluator defines an abstract `VectorSearcher` protocol:
```python
class VectorSearcher(Protocol):
    def search(self, query_vectors: np.ndarray, k: int) -> np.ndarray:
        ...
```
Any custom index (e.g. C++ shared library wrapper, HNSW graph index, IVF inverted index) simply implements `search(queries, k)` returning an integer array of neighbor IDs of shape `(num_queries, k)`. The evaluator handles warm-up, repetition benchmarking, latency percentile computation, duplicate detection, and Recall@K calculation uniformly.

### 10. How to Run Phase 1.2
```powershell
.\.venv\Scripts\python.exe src/phase1_2_evaluation.py
```
Output summary is saved to [`results/phase1_2/evaluation_summary.json`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase1_2/evaluation_summary.json).

---

## Phase 2 — ANN Index + Baseline Approximate Vector Search (FAISS IndexHNSWFlat)

### 1. Purpose of Phase 2
Phase 2 establishes the empirical Approximate Nearest Neighbor (ANN) search baseline over the exact same 100,000 base vectors and 1,000 query vectors ($D=128$, float32) evaluated in Phase 1. By benchmarking a graph-based ANN index (`faiss.IndexHNSWFlat`) directly against the frozen exact ground truth (`ground_truth.ivecs`), Phase 2 measures the fundamental Pareto trade-off:
$$\text{Search Latency} \longleftrightarrow \text{Recall@K} \longleftrightarrow \text{QPS}$$
This baseline establishes the performance bar before introducing disk-backed indices (SSD), product quantization (PQ), or LLM KV-cache architectures.

### 2. How HNSW Works
Hierarchical Navigable Small World (HNSW) is a multi-layer graph data structure inspired by probabilistic skip-lists:
- **Hierarchical Layers:** Vectors are inserted into multi-layer proximity graphs. The top layers contain sparse networks with long-range edges for coarse, logarithmic navigation ($O(\log N)$). Lower layers contain increasingly dense networks for fine-grained local neighborhood traversal.
- **Layer 0:** Ground layer containing all $N=100,000$ base vectors with up to $2M$ bidirectional links per node.
- **Greedy Routing:** During query execution, search starts at an entry point in the highest layer, greedily navigates to the local minimum, and descends layer-by-layer.
- **Beam Search (`efSearch`):** At Layer 0, the search maintains a priority queue of size `efSearch` candidates. Higher `efSearch` values broaden the exploration beam, visiting more candidate vertices and finding true nearest neighbors at the cost of more distance evaluations.

### 3. Evaluated Parameters
- **Metric:** Squared Euclidean Distance ($L2$) — strictly matching Phase 1.
- **$M$ (Graph Degree):** $M=16$ and $M=32$ (maximum outgoing edges per node on higher layers; $2M$ on Layer 0).
- **`efConstruction`:** 40 (FAISS standard default).
- **`efSearch`:** 10, 20, 40, 80, 160, 320.
- **Top-$K$:** $K=10$ (evaluating Recall@1, Recall@5, and Recall@10).

### 4. Benchmark Execution Results

| Method | Recall@10 | Recall@5 | Recall@1 | Batch Avg Latency (ms) | P95 Latency (ms) | QPS (Queries/s) | Build Time (s) | Index Size (MB) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Exact FAISS** | **1.0000** | **1.0000** | **1.0000** | 0.2188 | 6.6244 | 4,570.9 | 0.02 | 48.83 |
| **HNSW (M=16, ef=10)** | 0.1355 | 0.1450 | 0.1700 | **0.0188** | **0.0792** | **53,277.8** | 5.01 | 62.59 |
| **HNSW (M=16, ef=20)** | 0.2129 | 0.2288 | 0.2620 | 0.0357 | 0.1276 | 28,020.8 | 5.01 | 62.59 |
| **HNSW (M=16, ef=40)** | 0.3106 | 0.3312 | 0.3780 | 0.0616 | 0.2104 | 16,232.1 | 5.01 | 62.59 |
| **HNSW (M=16, ef=80)** | 0.4261 | 0.4526 | 0.5000 | 0.1208 | 0.3852 | 8,278.9 | 5.01 | 62.59 |
| **HNSW (M=16, ef=160)** | 0.5622 | 0.5940 | 0.6440 | 0.2247 | 0.6495 | 4,450.1 | 5.01 | 62.59 |
| **HNSW (M=16, ef=320)** | 0.6945 | 0.7218 | 0.7620 | 0.4243 | 1.2920 | 2,356.8 | 5.01 | 62.59 |
| **HNSW (M=32, ef=10)** | 0.2120 | 0.2244 | 0.2580 | 0.0493 | 0.1315 | 20,295.1 | 12.39 | 74.79 |
| **HNSW (M=32, ef=20)** | 0.3278 | 0.3512 | 0.3890 | 0.0880 | 0.3658 | 11,364.4 | 12.39 | 74.79 |
| **HNSW (M=32, ef=40)** | 0.4553 | 0.4830 | 0.5450 | 0.1520 | 0.3684 | 6,580.4 | 12.39 | 74.79 |
| **HNSW (M=32, ef=80)** | 0.6148 | 0.6414 | 0.7070 | 0.2719 | 0.5784 | 3,678.4 | 12.39 | 74.79 |
| **HNSW (M=32, ef=160)** | 0.7566 | 0.7782 | 0.8190 | 0.5064 | 1.3121 | 1,974.7 | 12.39 | 74.79 |
| **HNSW (M=32, ef=320)** | **0.8663** | **0.8828** | **0.9160** | 0.8945 | 2.1781 | 1,117.9 | 12.39 | 74.79 |

### 5. Key Empirical Observations
1. **Per-Query Responsiveness:**
   - Under individual single-query dispatch, `IndexFlatL2` exhibits a P95 latency of **6.62 ms** due to the requirement of scanning all 100,000 vectors.
   - In contrast, HNSW (M=16, ef=10) delivers a P95 latency of **0.079 ms** (~84× lower tail latency) and peak throughput of **53,278 QPS** (~11.6× higher throughput than exact search).
2. **Recall Scaling with efSearch:**
   - For $M=32$, increasing `efSearch` from 10 to 320 scales Recall@10 from **21.2% to 86.6%** (and Recall@1 to **91.6%**).
   - Higher graph connectivity ($M=32$) provides superior routing accuracy across all beam depths compared to $M=16$, with a modest increase in index build time (12.39s vs 5.01s) and memory footprint (74.79 MB vs 62.59 MB).
3. **Memory Footprint & Graph Overhead:**
   - Raw vector payload: $100,000 \times 128 \times 4\text{ bytes} = 48.83\text{ MB}$.
   - $M=16$ index size: $62.59\text{ MB}$ ($+13.76\text{ MB}$ or $+28.2\%$ graph link overhead).
   - $M=32$ index size: $74.79\text{ MB}$ ($+25.96\text{ MB}$ or $+53.2\%$ graph link overhead).
   - This in-memory overhead demonstrates why DRAM-only graph indices become cost-prohibitive at multi-million scale, necessitating SSD-backed tiered storage architectures.

### 6. Generated Visualizations
All publication plots are generated at 300 DPI and saved to `results/phase2/plots/`:
- [`recall_vs_efsearch.png`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase2/plots/recall_vs_efsearch.png): Graph 1 — Recall@10 vs. efSearch curves for M=16, M=32, and Exact reference.
- [`latency_vs_efsearch.png`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase2/plots/latency_vs_efsearch.png): Graph 2 — Batch average and P95 per-query search latencies.
- [`qps_vs_efsearch.png`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase2/plots/qps_vs_efsearch.png): Graph 3 — Queries Per Second vs. search depth.
- [`recall_vs_latency_tradeoff.png`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase2/plots/recall_vs_latency_tradeoff.png): Graph 4 — Pareto frontier depicting the Recall@10 vs. latency trade-off.
- [`combined_hnsw_dashboard.png`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase2/plots/combined_hnsw_dashboard.png): 4-panel comprehensive dashboard.

### 7. How to Run Phase 2
```powershell
.\.venv\Scripts\python.exe src/phase2_ann_hnsw.py
```
Output files:
- CSV Table: [`results/phase2/summaries/comparison_table.csv`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase2/summaries/comparison_table.csv)
- JSON Summary: [`results/phase2/summaries/hnsw_benchmark_summary.json`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase2/summaries/hnsw_benchmark_summary.json)
- Per-configuration outputs: `results/phase2/raw/`

---

## Phase 3 — Real-Time DRAM + NVMe SSD Tiered Vector Storage with Dynamic HOT/COLD Data Management

### 1. Purpose & Objective of Phase 3
Phase 3 builds the first real-time tiered storage engine in Case 04, bridging fast in-memory execution and persistent solid-state storage:
- **DRAM Capacity Constraints:** Enforces a configurable hardware DRAM budget (target: **18% of vector dataset** $\approx 8.79\text{ MB}$, leaving 82% residing exclusively on SSD).
- **Physical NVMe SSD Persistence:** Persists all 100,000 vectors ($D=128$, float32) as fixed-offset 512-byte records on the local NVMe SSD (`BC711 NVMe SK hynix 512GB`), supporting single and batch vector random reads without file scans.
- **Dynamic Hotness Engine:** Tracks vector access frequency, recency, and exponential time decay $\exp(-\lambda \Delta t)$ under continuous query streams.
- **Dynamic Tiering Controller:** Autonomously promotes hot vectors from SSD to DRAM and demotes cold vectors from DRAM to SSD without interrupting query execution.
- **Access Skew Validation:** Experimentally demonstrates why real-world access skew (Zipfian and Hotspot distributions) makes an 18% DRAM tier deliver up to **89–98% cache hit rates**.

---

### 2. Physical Storage Environment & Hardware Diagnostics
The system identifies and operates directly against the local physical storage device:
- **Drive / Mount:** `C:\Users\shivam.gupta\Desktop\Case04_Vector_POC\storage\ssd_vectors\vectors.bin`
- **Physical Disk:** `BC711 NVMe SK hynix 512GB`
- **Bus Type:** `NVMe` (Non-Volatile Memory Express)
- **Media Type:** `SSD` (Solid State Disk)
- **Filesystem:** `NTFS` (Sector Size: 4,096 bytes)
- **Capacity:** ~476 GB total, ~182.7 GB free space available.
- **Data Integrity:** Bit-for-bit SHA-256 cryptographic digest verified against frozen Phase 1 dataset.

---

### 3. Tiered System Architecture

```text
                     QUERY STREAM (Uniform, Zipf, Hotspot)
                                     │
                                     ▼
                        ANN CANDIDATE GENERATION
                     (FAISS IndexHNSWFlat, M=32, ef=40)
                                     │
                               Candidate IDs
                                     │
                                     ▼
                           TIERED VECTOR MANAGER
                       ┌─────────────┴─────────────┐
                       ▼                           ▼
                 DRAM CACHE                   NVMe SSD
             (HOT, Max 18% Budget)        (Fixed Offset 512B)
                       │                           │
                   Cache Hit                   Cache Miss
                       │                           │
                       │                    Batch NVMe Read
                       │                    (0.045 ms/vec)
                       │                           │
                       └─────────────┬─────────────┘
                                     ▼
                           TOP-K RE-RANKING & SEARCH
                                     │
                               Top-K Vectors
                                     │
                                     ▼
                            DENSE ACCESS TRACKER
                     (Frequency, Recency, Time Decay)
                                     │
                                     ▼
                           DYNAMIC HOTNESS ENGINE
                 Hotness = α·Freq_rec + β·ln(1+Freq_tot) + γ·e^(-λ·Δt)
                                     │
                                     ▼
                           TIERING CONTROLLER
                       ┌─────────────┴─────────────┐
                       ▼                           ▼
                 PROMOTE (SSD→DRAM)          DEMOTE (DRAM→SSD)
             If Hotness > Coldest_DRAM     Lowest Hotness Evicted
```

---

### 4. Dynamic Hotness Model
Hotness is computed incrementally per access using a three-component formula:
$$\text{Hotness}(v) = \alpha \cdot \text{Freq}_{\text{recent}}(v) + \beta \cdot \ln(1 + \text{Freq}_{\text{total}}(v)) + \gamma \cdot \exp(-\lambda \cdot \Delta t)$$
- **$\alpha = 0.40$ (Recent Frequency):** Rewards short-term burstiness. Decayed periodically via $\text{Freq}_{\text{recent}} \leftarrow \text{Freq}_{\text{recent}} \times \delta$ ($\delta = 0.85$).
- **$\beta = 0.35$ (Historical Frequency):** Logarithmic scaling prevents high past frequency from permanently monopolizing cache lines (cache pollution resistance).
- **$\gamma = 0.25$ (Recency Score):** Exponential recency boost where $\Delta t = t_{\text{current}} - t_{\text{last}}$, with decay rate $\lambda = 0.05\text{ s}^{-1}$.

---

### 5. Benchmark Execution Results: DRAM Capacity & Workload Sweep

All measurements gathered via continuous query streaming across 1,000 queries per run on the frozen Phase 1 dataset ($N=100,000$, $D=128$, $K=10$):

| DRAM % | SSD % | DRAM MB | Workload | Recall@10 | Hit Rate | SSD Reads/Q | Avg Lat (ms) | P95 Lat (ms) | QPS | Promotions | Demotions |
|:---:|:---:|---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **100%** | 0% | 48.83 | **Uniform** | 0.4635 | **100.0%** | 0.00 | 1.082 | 1.63 | **924.1** | 0 | 0 |
| **100%** | 0% | 48.83 | **Zipf ($\alpha=1.25$)** | 0.4863 | **100.0%** | 0.00 | 0.987 | 1.43 | **1,013.1** | 0 | 0 |
| **100%** | 0% | 48.83 | **Hotspot (80/20)** | 0.4412 | **100.0%** | 0.00 | 1.064 | 1.66 | **939.7** | 0 | 0 |
| **50%** | 50% | 24.41 | Uniform | 0.4635 | 86.8% | 5.29 | 49.806 | 143.42 | 20.1 | 5,287 | 5,287 |
| **50%** | 50% | 24.41 | **Zipf** | 0.4863 | **93.3%** | 2.69 | 11.422 | 53.97 | **87.5** | 2,693 | 2,693 |
| **50%** | 50% | 24.41 | **Hotspot** | 0.4412 | **92.4%** | 3.05 | 9.769 | 48.08 | **102.4** | 3,052 | 3,052 |
| **25%** | 75% | 12.21 | Uniform | 0.4635 | 80.3% | 7.88 | 55.807 | 138.29 | 17.9 | 7,881 | 7,881 |
| **25%** | 75% | 12.21 | **Zipf** | 0.4863 | **90.0%** | 3.98 | 27.031 | 146.86 | **37.0** | 3,985 | 3,985 |
| **25%** | 75% | 12.21 | **Hotspot** | 0.4412 | **88.6%** | 4.55 | 27.551 | 130.63 | **36.3** | 4,554 | 4,554 |
| **18%** | **82%** | **8.79** | Uniform | 0.4635 | 78.4% | 8.65 | 45.020 | 107.83 | 22.2 | 8,653 | 8,653 |
| **18%** | **82%** | **8.79** | **Zipf** | 0.4863 | **89.1%** | **4.38** | **21.005** | **110.71** | **47.6** | 4,377 | 4,377 |
| **18%** | **82%** | **8.79** | **Hotspot** | 0.4412 | **87.5%** | **5.00** | **20.815** | **102.78** | **48.0** | 5,002 | 5,002 |
| **10%** | 90% | 4.88 | Uniform | 0.4635 | 76.4% | 9.45 | 753.171 | 58.39 | 1.3 | 9,455 | 9,455 |
| **10%** | 90% | 4.88 | **Zipf** | 0.4863 | **88.1%** | 4.77 | 14.560 | 71.74 | **68.7** | 4,772 | 4,772 |
| **10%** | 90% | 4.88 | **Hotspot** | 0.4412 | **86.3%** | 5.50 | 4.602 | 21.42 | **217.3** | 5,495 | 5,495 |

---

### 6. Real-Time Dynamic Adaptation Trace (18% DRAM Target Configuration)
Streaming 2,400 consecutive queries into the 18% DRAM tier system demonstrates continuous online convergence:

| Interval | Total Queries | DRAM Hit Rate | SSD Reads / Interval | Promotions | Demotions | Avg Latency (ms) | QPS |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 200 | 80.4% | 1,565 | 1,565 | 1,565 | 9.254 | 108.1 |
| **2** | 400 | 91.1% | 714 | 714 | 714 | 5.285 | 189.2 |
| **3** | 600 | 93.8% | 500 | 500 | 500 | 3.569 | 280.2 |
| **4** | 800 | 94.1% | 475 | 475 | 475 | 3.603 | 277.5 |
| **5** | 1,000 | 94.9% | 406 | 406 | 406 | 3.427 | 291.8 |
| **6** | 1,200 | 96.4% | 291 | 291 | 291 | 3.244 | 308.3 |
| **7** | 1,400 | 96.4% | 289 | 289 | 289 | 3.541 | 282.4 |
| **8** | 1,600 | 97.8% | 176 | 176 | 176 | 2.450 | 408.2 |
| **9** | 1,800 | 97.1% | 231 | 231 | 231 | 2.391 | 418.2 |
| **10** | 2,000 | 97.1% | 232 | 232 | 232 | 1.966 | 508.6 |
| **11** | 2,200 | 97.8% | 178 | 178 | 178 | 2.008 | 497.9 |
| **12** | 2,400 | **98.3%** | **134** | **134** | **134** | **1.421** | **703.7** |

**Observation:** As the working set of hot vectors migrates into DRAM, NVMe SSD reads drop by **91.4%** (from 1,565 to 134 reads/interval), search latency drops by **84.6%** (from 9.25 ms to 1.42 ms), and throughput increases by **6.5×** (from 108 to 704 QPS).

---

### 7. Dynamic Cold → Hot → Cold Lifecycle Test (Vector V9000 Verification)
A dedicated test validates dynamic bi-directional tier transitions:
1. **Initial State (Cold on SSD):** Vector V9000 resident in DRAM = `False`, Hotness Score = `0.0000` (COLD tier).
2. **Access Burst (Promoted to DRAM):** V9000 accessed 15 times $\implies$ Hotness Score surged to `8.3538`. System triggered promotion into DRAM cache (`resident_in_dram = True`, `tier = DRAM`).
3. **Time Decay & Competition (Demoted back to SSD):** Queries to V9000 ceased for 120 seconds while 105 competitor vectors received active query traffic. V9000 hotness decayed to `0.9295`, below the competitor threshold. V9000 was autonomously evicted back to SSD (`resident_in_dram = False`, `tier = SSD`).
- **Promotion Verified:** `True`
- **Demotion Verified:** `True`

---

### 8. Generated Phase 3 Plots (300 DPI)
All plots are located in [`results/phase3/plots/`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase3/plots/):
1. [`dram_capacity_vs_p95_latency.png`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase3/plots/dram_capacity_vs_p95_latency.png): Graph 1 — DRAM Capacity % vs. P95 Search Latency across Uniform, Zipf, and Hotspot.
2. [`dram_capacity_vs_hit_rate.png`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase3/plots/dram_capacity_vs_hit_rate.png): Graph 2 — DRAM Capacity % vs. DRAM Hit Rate across workloads.
3. [`dram_capacity_vs_ssd_reads.png`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase3/plots/dram_capacity_vs_ssd_reads.png): Graph 3 — DRAM Capacity % vs. NVMe SSD Reads per Query.
4. [`dram_capacity_vs_qps.png`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase3/plots/dram_capacity_vs_qps.png): Graph 4 — DRAM Capacity % vs. Queries Per Second throughput.
5. [`hotness_rank_vs_access_frequency.png`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase3/plots/hotness_rank_vs_access_frequency.png): Graph 5 — Vector Hotness Rank vs. Access Frequency (power-law distribution).
6. [`dram_18pct_access_distribution.png`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase3/plots/dram_18pct_access_distribution.png): Graph 6 — HOT vs. COLD tier access distribution for the 18% DRAM configuration.
7. [`promotions_demotions_over_time.png`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase3/plots/promotions_demotions_over_time.png): Graph 7 — Promotion/Demotion rates and hit rate convergence over time.
8. [`combined_phase3_dashboard.png`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase3/plots/combined_phase3_dashboard.png): 4-panel unified executive dashboard.

---

### 9. How to Run Phase 3
```powershell
.\.venv\Scripts\python.exe src/phase3_realtime.py
```
Output files:
- CSV Comparison Table: [`results/phase3/summaries/phase3_comparison_table.csv`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase3/summaries/phase3_comparison_table.csv)
- JSON Full Summary: [`results/phase3/summaries/phase3_summary.json`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase3/summaries/phase3_summary.json)
- Dynamic Adaptation Trace: [`results/phase3/traces/realtime_18pct_adaptation_trace.json`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase3/traces/realtime_18pct_adaptation_trace.json)
- V9000 Lifecycle Trace: [`results/phase3/traces/v9000_lifecycle_trace.json`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase3/traces/v9000_lifecycle_trace.json)

---

## Phase 3 Hardened — Production-Ready Tiered Vector Engine (v3.1)

### 1. Hardening Purpose & Key Upgrades
Phase 3 Hardened resolves two critical production defects uncovered during the initial Phase 3 benchmark:
1. **Latency-Statistics Inversion:** Diagnosed and corrected the arithmetic mean vs. P95 latency anomaly. With high-resolution nanosecond clocks (`time.perf_counter_ns()`), rigorous per-query telemetry, and identical sample populations, $\min \le P_{50} \le P_{95} \le P_{99} \le \max$ is strictly satisfied across all runs.
2. **Pathological Cache Thrashing:** Engineered **Policy B**, incorporating:
   - **Admission Control:** Requires $C_{\text{access}} \ge 2$ or $H \ge 1.0$ before considering an SSD vector for promotion.
   - **Hysteresis Damping:** Promotion threshold ($\theta_{\text{prom}} = 1.1$) and demotion threshold ($\theta_{\text{dem}} = 0.5$) with a $0.20$ score margin over the coldest resident vector.
   - **Minimum Residency Protection:** Protects promoted vectors for a minimum time window ($0.05$ s).
   - **Migration Cooldown:** Prevents rapid re-promotion oscillations for recently demoted vectors.
   - **Batch Rate Limiting:** Caps migrations to $\le 10$ per candidate batch.

### 2. Architecture Diagram

```text
              REAL-TIME QUERY STREAM
                       │
                       ▼
                  HNSW SEARCH (M=32, ef=40)
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
           DRAM                NVMe SSD
           HOT                  COLD
             │                   │
             └─────────┬─────────┘
                       ▼
                 TOP-K RERANKING
                       │
                       ▼
                 ACCESS TRACKING
                       │
                       ▼
                  HOTNESS MODEL
                       │
                       ▼
                ADMISSION CONTROL
                       │
              ┌────────┴────────┐
              ▼                 ▼
           PROMOTE            DEMOTE
              │                 │
              └───────┬────────┘
                      ▼
               STABLE WORKING SET
```

### 3. Key Results: Policy A vs. Policy B (18% DRAM Target)

| Metric | Policy A (Legacy) | Policy B (Production Hardened) | Delta / Impact |
|:---|:---:|:---:|:---:|
| **Uniform Migrations** | 17,306 | **10,918** | **-36.9% Churn** |
| **Zipf Migrations** | 8,754 | **4,284** | **-51.1% Churn** |
| **Hotspot Migrations** | 10,004 | **5,876** | **-41.3% Churn** |
| **Working-Set Overlap (Zipf)** | ~80% | **99.4%** | **Stable steady-state** |
| **Recall@10 (Uniform)** | 0.4635 | 0.4635 | 100% Preserved |
| **Recall@10 (Zipf)** | 0.4863 | 0.4863 | 100% Preserved |
| **Recall@10 (Hotspot)** | 0.4412 | 0.4412 | 100% Preserved |
| **Multi-Worker Peak QPS (8T)** | — | **1,946.2 QPS** | **Thread-safe scaling** |

### 4. How to Run Phase 3 Hardened

Run the full experimental suite:
```powershell
.\.venv\Scripts\python.exe src/phase3_realtime.py
```

Run the automated validation test suite (15/15 tests):
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests -v
```

Run the 10% Uniform diagnostic analysis:
```powershell
.\.venv\Scripts\python.exe src/diagnostic_10pct_uniform.py
```

### 5. Output Artifacts
All hardened results are preserved in `results/phase3_hardened/`:
- **Summaries:**
  - Full JSON Summary: [`results/phase3_hardened/summaries/phase3_hardened_summary.json`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase3_hardened/summaries/phase3_hardened_summary.json)
  - Policy Comparison Table: [`results/phase3_hardened/summaries/policy_comparison_table.csv`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase3_hardened/summaries/policy_comparison_table.csv)
  - Reproducibility Config: [`results/phase3_hardened/summaries/phase3_config.json`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase3_hardened/summaries/phase3_config.json)
- **Traces:**
  - 18% Dynamic Adaptation (2,400 queries): [`results/phase3_hardened/traces/realtime_18pct_adaptation_trace.json`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase3_hardened/traces/realtime_18pct_adaptation_trace.json)
  - V9000 Extended Lifecycle: [`results/phase3_hardened/traces/v9000_lifecycle_trace.json`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase3_hardened/traces/v9000_lifecycle_trace.json)
  - Multi-Worker Concurrency: [`results/phase3_hardened/traces/concurrency_benchmark_trace.json`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase3_hardened/traces/concurrency_benchmark_trace.json)
- **12 Publication Plots (300 DPI):**
  Located in [`results/phase3_hardened/plots/`](file:///c:/Users/shivam.gupta/Desktop/Case04_Vector_POC/results/phase3_hardened/plots/).



