# Case 04 — Phase 4: Product Quantization & HOT/COLD Tier Data Model
## Comprehensive Production Engineering & Research Report

---

### Executive Summary

| Metric / Dimension | Specification / Measured Result | Status |
| :--- | :--- | :--- |
| **Phase 4 Objective** | PQ Compression + HOT/COLD Data Model + ADC Search Primitive | **COMPLETE** |
| **Architectural Target** | Two-Tier Vector DB (DRAM HNSW + NVMe DiskANN with PQ & Rerank) | **ALIGNED** |
| **Base Dataset** | 100,000 vectors × 128 dimensions (float32, 51.20 MB raw FP32) | **VERIFIED** |
| **Query Dataset** | 1,000 vectors × 128 dimensions (float32) | **VERIFIED** |
| **Ground Truth** | Frozen Phase 1 exact FAISS IndexFlatL2 oracle (1,000 × 10) | **FROZEN / UNTOUCHED** |
| **PQ Architecture** | $M = 16$ subquantizers, $K_s = 256$ centroids, $d_{sub} = 8$ dim/sub | **VALIDATED** |
| **PQ Code Size** | 16 bytes / vector (uint8[16]) | **VALIDATED** |
| **Raw FP32 Footprint** | 51,200,000 bytes (48.83 MB) | **MEASURED** |
| **PQ Codes Footprint** | 1,600,000 bytes (1.53 MB) | **MEASURED** |
| **Code-Only Compression** | **32.00×** (512 bytes $\to$ 16 bytes) | **MEASURED** |
| **Effective Compression** | **18.83×** (including codebooks and all metadata) | **MEASURED** |
| **Reconstruction Error** | MSE = 0.317669, RMSE = 0.563621, Relative L2 Error = 56.37% | **MEASURED** |
| **ADC Search Equivalence** | ADC distances $\equiv \|q - \hat{x}\|_2^2$ (max diff = $4.58 \times 10^{-5} < 10^{-4}$) | **PROVEN** |
| **Search Recall (100K)** | R@1 = 7.80%, R@5 = 12.50%, R@10 = 13.74%, Candidate R@10@100 = 44.38% | **MEASURED** |
| **Search Latency / QPS** | ADC = 5.52 ms (181 QPS) vs Reconstructed = 40.22 ms (25 QPS) | **7.3× SPEEDUP** |
| **HOT / COLD Partition** | HOT = 18,000 (18.0%), COLD = 82,000 (82.0%), Disjoint & Exhaustive | **VALIDATED** |
| **Automated Tests** | 23 / 23 Tests Passed (0 Failures, 0 Errors) | **100% PASS** |
| **Determinism & Hashes** | Byte-identical reproducibility across independent runs | **VERIFIED** |
| **Phase 5 Readiness** | Handoff manifests, NVMe offset mappings, PQ codebooks exported | **READY** |

---

### 1. Objective

Phase 4 of the "Case 04 — Vector POC" implements the core compression primitives, data structures, and mathematical foundations required to bridge the raw full-precision vectors from Phases 1–3 into the multi-tier production architecture.

Specifically, Phase 4 delivers:
1. **Product Quantization (PQ)**: Deterministic codebook training and vector encoding compressing 128-dimensional float32 vectors (512 bytes) into 16-byte uint8 codes.
2. **Reconstruction & Error Quantification**: Full-dataset error analysis measuring MSE, RMSE, and relative L2 error across all 100,000 vectors.
3. **Asymmetric Distance Computation (ADC)**: Efficient Look-Up Table (LUT) query primitive computing query-to-code distances without vector decompression.
4. **HOT/COLD Logical Data Model**: A decoupled data catalog with stable vector identities, 18%/82% initial capacity partitioning, dynamic migration semantics, and explicit NVMe byte-offset mapping.
5. **Phase 5 Handoff Artifacts**: Serialized codebooks, codes, tier manifests, and NVMe layout metadata ready for DiskANN ingestion.

---

### 2. Architecture Alignment & PPT Mental Model

The primary mental model for this system is the two-tier hybrid architecture established in the project roadmap:

```
                              [ QUERY ]
                                  |
                                  v
                         [ HOT / COLD ROUTER ]
                         /                   \
                        /                     \
              HOT TIER                         COLD TIER
              (DRAM)                           (NVMe SSD)
            +---------------------------+    +---------------------------+
            | • HNSW (Phase 6)          |    | • DiskANN Graph (Phase 5) |
            | • HOT Full Vectors (18K)  |    | • COLD Full Vectors (82K) |
            | • PQ Codes Array (100K)   |    | • Direct I/O Page Store   |
            | • PQ Codebooks (16x256x8) |    | • PQ-Assisted Search      |
            | • Tier Membership Catalog |    |                           |
            +---------------------------+    +---------------------------+
                        \                     /
                         \                   /
                       [ CANDIDATE MERGE ]
                                |
                                v
                     [ EXACT RE-RANKING ]
                    (Using Full FP32 Vectors)
                                |
                                v
                            [ TOP-K ]
```

#### What Phase 4 Implements vs. What Belongs to Later Phases

| Phase | Boundary & Scope | Responsibility in Architecture |
| :--- | :--- | :--- |
| **Phase 1** | Immutable Ground Truth Oracle | Exact FAISS IndexFlatL2 benchmark and frozen dataset. |
| **Phase 2** | Frozen HNSW Baseline | Single-tier in-memory HNSW baseline exploration. |
| **Phase 3** | Dynamic DRAM/NVMe Tiering Baseline | Dynamic access tracking, promotion/demotion, LRU cache. |
| **Phase 4 (Current)** | **Data Model, PQ, ADC & Storage Layout** | **PQ compression, DRAM code array, codebooks, ADC search primitive, HOT/COLD logical catalog, NVMe offset mapping.** |
| **Phase 5 (Next)** | COLD Tier DiskANN | Out-of-core graph construction on NVMe, Vamana index, sector-aligned full-vector I/O. |
| **Phase 6** | HOT Tier HNSW + Unified Routing | DRAM HNSW on HOT vectors, cross-tier candidate merge, exact reranking, top-k pipeline. |

Phase 4 strictly respects these boundaries:
- **No DiskANN construction** in Phase 4 (reserved for Phase 5).
- **No final query router** in Phase 4 (reserved for Phase 6).
- **Full-precision vectors are preserved** on NVMe SSD; PQ serves as the search representation, not a destructive replacement.
- Existing Phase 1, Phase 2, and Phase 3 code and datasets remain **100% untouched**.

---

### 3. Frozen Dataset & Oracle Verification

Phase 4 strictly consumes the frozen dataset generated in Phase 1.1:

- **Base Vectors**: 100,000 vectors $\times$ 128 dimensions, `float32`, little-endian IEEE-754.
- **Query Vectors**: 1,000 query vectors $\times$ 128 dimensions, `float32`.
- **Ground Truth**: Exact top-10 nearest neighbors (`int32`) and L2 squared distances (`float32`).

#### Cryptographic Checksums (Phase 1 Invariant)

| File Name | Record Count & Dim | File Size | Expected & Verified SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- | :--- |
| `base_vectors.fvecs` | 100,000 × 128 (FP32) | 51,600,000 bytes | `62a7ad69bc429570fb4819287bb4d317d8069fb784ef528f370c64189d37198b` | **MATCH** |
| `query_vectors.fvecs` | 1,000 × 128 (FP32) | 516,000 bytes | `c1d80d61714296078229bc3429a79569113441c01fdd1069de21611a30528496` | **MATCH** |
| `ground_truth.ivecs` | 1,000 × 10 (Int32) | 44,000 bytes | `ee2f420a3fea4e71408756f6d0bf7bed1741a69dafc119b2781f7b4142c7d66f` | **MATCH** |
| `ground_truth_distances.fvecs` | 1,000 × 10 (FP32) | 44,000 bytes | `3ca693e88fdfab960b0d43412b8a66266259407615feeb1a1f61ffc083b0ffc4` | **MATCH** |

---

### 4. PQ Configuration

The Product Quantization parameters follow the controlled baseline:

$$D = 128, \quad M = 16, \quad K_s = 256 \ (2^8), \quad d_{sub} = \frac{D}{M} = \frac{128}{16} = 8$$

Each 128-dimensional vector is divided into 16 non-overlapping subvectors of 8 dimensions each. Each subvector is quantized to one of 256 centroids, represented by an 8-bit unsigned integer (`uint8`).

```
Vector X [128 floats]:
|--- sub_0 ---|--- sub_1 ---| ... |--- sub_15 ---|
   8 floats      8 floats            8 floats
      |             |                   |
      v             v                   v
   quantize      quantize            quantize
      |             |                   |
      v             v                   v
   code[0]       code[1]             code[15]
    uint8         uint8               uint8
      \-------------+-------------------/
                    |
           PQ Code: 16 bytes total
```

The modular design in `src/phase4_pq_train.py` (`PQConfig`) is parameterized such that alternative configurations (e.g., $M=8, d_{sub}=16$ or $M=32, d_{sub}=4$) can be instantiated seamlessly without architectural changes.

---

### 5. Deterministic PQ Training

- **Algorithm**: $k$-means clustering across $M=16$ independent subspaces using FAISS `ProductQuantizer`.
- **Training Dataset**: All 100,000 frozen base vectors.
- **Random Seed**: Fixed to `42` via `pq.cp.seed = 42`.
- **Iterations**: 25 iterations per subquantizer.
- **Training Wall Time**: 21.34 seconds.
- **Artifacts Generated**:
  - `results/phase4_pq/codebooks/pq_codebooks.npy`: Raw numpy centroids array of shape `(16, 256, 8)` `float32`.
  - `results/phase4_pq/codebooks/faiss_pq_quantizer.bin`: Serialized native FAISS quantizer.
  - `results/phase4_pq/codebooks/codebook_metadata.json`: Full configuration, stats, and cryptographic checksums.

#### Codebook Statistics

- Total Centroids: $16 \times 256 = 4,096$
- Total Floats: $4,096 \times 8 = 32,768$ floats
- Centroid Values: Min = -2.4769, Max = 2.6031, Mean = 0.0011, Std = 0.8425
- Finite Integrity: 0 NaNs, 0 Infs.
- Checksum (`pq_codebooks.npy`): `dfaae246699e1e75dfb87a7480a6057a8745fbe824a3365a1456b6901426559c`

---

### 6. PQ Encoding

All 100,000 base vectors were encoded into uint8 PQ codes:

- **Input Shape**: `[100000, 128]` (`float32`, 51,200,000 bytes)
- **Output Shape**: `[100000, 16]` (`uint8`, 1,600,000 bytes)
- **Encoding Time**: 0.534 seconds ($\sim 187,000$ vectors/sec throughput)
- **Value Range**: Strictly $0 \le code_{i, m} \le 255$
- **Vector ID Mapping Guarantee**: Vector ID $i$ corresponds strictly to `codes[i]`. No index permutation or shuffling was introduced.
- **Artifacts Saved**:
  - `results/phase4_pq/codes/base_pq_codes.bin`: 1,600,000 bytes raw binary.
  - `results/phase4_pq/codes/base_pq_codes.npy`: Portable numpy array.
  - Checksum (`base_pq_codes.bin`): `984ac8cf84d4c7ffd0a8ce1a4901fbaaedb9bdd96210f5e0c4d73473fcc3aca9`

---

### 7. Reconstruction Fidelity & Error Analysis

Vector reconstruction transforms each 16-byte code back into an approximation of the original 128-dimensional vector:

$$\hat{x}_{i} = \bigoplus_{m=0}^{15} C[m, code_{i, m}]$$

Where $C[m, k]$ is the $k$-th centroid of subquantizer $m$.

#### Full-Dataset (100,000 Vectors) Error Metrics

| Error Metric | Formula | Measured Value |
| :--- | :--- | :--- |
| **Mean Squared Error (MSE)** | $\frac{1}{N \cdot D} \sum_{i=1}^N \sum_{d=1}^D (x_{i,d} - \hat{x}_{i,d})^2$ | **0.317669** |
| **Root Mean Squared Error (RMSE)** | $\sqrt{\text{MSE}}$ | **0.563621** |
| **Relative Frobenius L2 Error** | $\frac{\|X - \hat{X}\|_F}{\|X\|_F}$ | **0.563733 (56.37%)** |
| **Mean Per-Vector Relative Error** | $\frac{1}{N} \sum_{i=1}^N \frac{\|x_i - \hat{x}_i\|_2}{\|x_i\|_2}$ | **0.563967** |
| **P50 (Median) Relative Error** | 50th percentile of per-vector relative error | **0.563604** |
| **P95 Relative Error** | 95th percentile of per-vector relative error | **0.627798** |
| **P99 Relative Error** | 99th percentile of per-vector relative error | **0.655823** |
| **Reconstruction Throughput** | Vectorized NumPy decoding speed | **824,868 vectors/sec** |

*Interpretation*: Product quantization with $M=16, K_s=256$ discards fine intra-subspace variance to achieve a 32× compression ratio. This level of reconstruction error ($\sim 56\%$) is typical for standard PQ on isotropic Gaussian synthetic data and confirms why **full vectors must be preserved on NVMe SSD for exact reranking**, validating the PPT architecture.

---

### 8. Asymmetric Distance Computation (ADC)

#### Mathematical Principle

For a continuous query vector $q \in \mathbb{R}^{128}$ and a quantized database vector $x_i \approx \hat{x}_i$:

$$\|q - \hat{x}_i\|_2^2 = \sum_{m=0}^{M-1} \|q_m - C[m, code_{i, m}]\|_2^2$$

ADC precomputes the distances between the query subvectors and all centroids in a Look-Up Table (LUT):

$$LUT[m, k] = \|q_m - C[m, k]\|_2^2, \quad \forall m \in [0, 15], \ k \in [0, 255]$$

The query-to-database distance then simplifies to 16 lookups and additions:

$$D_{ADC}(q, code_i) = \sum_{m=0}^{15} LUT[m, code_{i, m}]$$

#### Mathematical Identity Verification

In `src/phase4_pq_adc.py`, we proved and verified the numerical equivalence between the ADC lookup and explicit Euclidean distance against reconstructed vectors:

$$\max_{i} \left| D_{ADC}(q, code_i) - \|q - \hat{x}_i\|_2^2 \right| = 4.58 \times 10^{-5} < 10^{-4}$$

The difference is purely floating-point summation ordering, validating that ADC is mathematically exact with respect to the quantized representation.

---

### 9. Search Quality & Latency Benchmark

We conducted a head-to-head evaluation across 1,000 queries against the 100,000 database vectors:
1. **Exact FP32**: FAISS `IndexFlatL2` (Phase 1 Exact Oracle)
2. **Custom ADC**: Vectorized NumPy LUT-based ADC search (`src/phase4_pq_adc.py`)
3. **Reconstructed PQ**: Full vector reconstruction followed by exact L2 squared distance
4. **FAISS IndexPQ**: C++ native SIMD-accelerated ADC baseline

#### Benchmark Results (1,000 Queries, Retrieval Depth $k=100$)

| Search Method | Recall@1 | Recall@5 | Recall@10 | Candidate R@10@100 | Recall@100 | Avg Latency | P50 Latency | P95 Latency | P99 Latency | QPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact FP32 (Oracle)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 5.560 ms | 5.415 ms | 6.843 ms | 8.211 ms | 179 |
| **Custom ADC (LUT)** | 0.0780 | 0.1250 | 0.1374 | **0.4438** | 0.2087 | 5.517 ms | 5.245 ms | 7.926 ms | 9.817 ms | 181 |
| **Reconstructed PQ** | 0.0780 | 0.1250 | 0.1374 | **0.4438** | 0.2087 | 40.219 ms | 39.408 ms | 47.068 ms | 57.882 ms | 25 |
| **FAISS IndexPQ (C++)** | 0.0780 | 0.1250 | 0.1374 | **0.4438** | 0.2087 | 2.556 ms | 2.340 ms | 3.740 ms | 5.002 ms | 390 |

#### Key Research Insights

1. **Exact Recall Parity**: Custom ADC, Reconstructed PQ, and FAISS IndexPQ yield **identical recall values** across all $k$ ($R@1=7.80\%, R@5=12.50\%, R@10=13.74\%, R@10@100=44.38\%$), confirming algorithmic correctness.
2. **Computational Speedup of ADC**: Custom ADC achieves **5.52 ms/query**, running **7.3× faster** than Reconstructed PQ (**40.22 ms/query**), while consuming a fraction of memory bandwidth because no vectors are reconstructed in RAM.
3. **Candidate Recall for Two-Tier Reranking**: While standalone top-10 PQ search achieves $13.74\%$ recall due to quantization distortion, fetching a top-100 candidate pool captures **$44.38\%$ of the true nearest neighbors**. This directly supports the PPT architecture: PQ is designed for coarse candidate generation on DRAM, followed by exact reranking using full-precision vectors on NVMe.

---

### 10. HOT/COLD Logical Data Model & Storage Layout

#### Capacity Partitioning

Following the Phase 3 capacity specification:
- **HOT Tier (DRAM)**: 18,000 vectors (18.0%)
- **COLD Tier (NVMe SSD)**: 82,000 vectors (82.0%)

#### Partition Invariants Verified

- $\text{len}(HOT) = 18,000$
- $\text{len}(COLD) = 82,000$
- $HOT \cap COLD = \emptyset$ (strictly disjoint)
- $HOT \cup COLD = \{0, 1, \dots, 99999\}$ (strictly exhaustive)
- Every vector ID appears exactly once.

#### Dynamic Membership & Stable Identity

Vector IDs $0 \dots 99,999$ are permanent, immutable identifiers. A vector's tier is dynamic metadata rather than a hardcoded ID range.
In `src/phase4_pq_tier_model.py`:
- `promote(vector_id)` moves a vector from COLD to HOT without altering its ID.
- `demote(vector_id)` moves a vector from HOT to COLD without altering its ID.
- Unit Test 22 verifies that promotion and demotion preserve all capacity invariants.

#### Physical Storage Layout

```
DRAM RESIDENT (Total: ~2.59 MB):
  1. PQ Codes Array (All 100,000 vectors): 1,600,000 bytes (1.53 MB)
  2. PQ Codebooks (Centroids table): 131,072 bytes (128 KB)
  3. Tier Catalog & Metadata: ~964 KB
  4. HOT Full Vectors (18,000 vectors): In DRAM cache

NVMe SSD RESIDENT (Storage Media: BC711 NVMe SK hynix 512GB):
  1. COLD Full Vectors: storage/ssd_vectors/vectors.bin
     - Direct byte offset formula: byte_offset = vector_id * 128 * 4
     - Total COLD footprint: 82,000 * 512 bytes = 41,984,000 bytes (40.04 MB)
  2. DiskANN Graph Structure: Reserved for Phase 5.
```

---

### 11. Memory & Storage Accounting

We computed exact byte measurements from the filesystem and memory buffers:

| Component | Storage Media | Byte Calculation | Size (Bytes) | Size (MB / KB) |
| :--- | :--- | :--- | :--- | :--- |
| **Raw FP32 Base Vectors** | NVMe (Reference) | $100,000 \times 128 \times 4$ | 51,200,000 | 48.83 MB |
| **PQ Codes Array** | DRAM | $100,000 \times 16 \times 1$ | 1,600,000 | 1.53 MB |
| **PQ Codebooks (Centroids)** | DRAM | $16 \times 256 \times 8 \times 4$ | 131,072 | 128.00 KB |
| **Tier & Catalog Metadata** | DRAM / Disk | JSON manifests & catalogs | 987,762 | 964.61 KB |
| **Total PQ Representation** | DRAM | Codes + Codebooks + Metadata | **2,718,834** | **2.59 MB** |

#### Measured Compression Ratios

$$\text{Code-Only Compression Ratio} = \frac{51,200,000 \text{ bytes}}{1,600,000 \text{ bytes}} = \mathbf{32.00\times}$$

$$\text{Effective System Compression Ratio} = \frac{51,200,000 \text{ bytes}}{2,718,834 \text{ bytes}} = \mathbf{18.83\times}$$

*Accounting Transparency Note*: We report both figures. The theoretical code-only compression is exactly $32.00\times$. When codebooks and tier manifests are accounted for, the effective compression is $18.83\times$.

---

### 12. Automated Validation Test Suite

Automated validation was executed via `tests/test_phase4.py` covering all 20 required tests from Prompt Section 21 plus additional architectural checks:

```
======================================================================
Ran 23 tests in 2.625s
STATUS: OK (100% PASS)
======================================================================
```

| Test ID | Test Description | Verification Method | Result |
| :---: | :--- | :--- | :---: |
| **TEST 1** | Frozen dataset dimensions correct | Verified base `(100K, 128)`, query `(1K, 128)`, GT `(1K, 10)` | **PASS** |
| **TEST 2** | Dataset identity / checksum validation | Verified SHA-256 for all 4 frozen Phase 1 binary files | **PASS** |
| **TEST 3** | PQ configuration valid | Verified $D=128, M=16, K_s=256, nbits=8$ | **PASS** |
| **TEST 4** | Dimension divisibility ($D \pmod M == 0$) | Verified $128 \pmod{16} == 0, d_{sub}=8$ | **PASS** |
| **TEST 5** | PQ code shape = `[100000, 16]` | Verified NumPy shape on encoded array | **PASS** |
| **TEST 6** | PQ dtype = `uint8` | Verified array dtype is `numpy.uint8` | **PASS** |
| **TEST 7** | PQ code value range $[0, 255]$ | Verified $\min \ge 0$ and $\max \le 255$ | **PASS** |
| **TEST 8** | No NaN/Inf in codebooks | Verified `np.isnan().any() == False`, `np.isinf().any() == False` | **PASS** |
| **TEST 9** | Reconstruction shape correct | Verified reconstructed vectors shape is `[N, 128]` `float32` | **PASS** |
| **TEST 10** | Reconstruction finite, no NaN/Inf | Verified finite values and valid MSE ($0.317$) | **PASS** |
| **TEST 11** | HOT count = 18,000 | Verified `tier_model.hot_count == 18,000` | **PASS** |
| **TEST 12** | COLD count = 82,000 | Verified `tier_model.cold_count == 82,000` | **PASS** |
| **TEST 13** | HOT $\cap$ COLD = $\emptyset$ (Disjoint) | Verified `hot_ids.intersection(cold_ids) == set()` | **PASS** |
| **TEST 14** | HOT $\cup$ COLD = All Vector IDs | Verified union equals $\{0 \dots 99,999\}$ | **PASS** |
| **TEST 15** | Every vector appears exactly once | Verified partition exhaustiveness and uniqueness | **PASS** |
| **TEST 16** | PQ determinism across repeated runs | Verified byte-identical codes and SHA-256 across runs | **PASS** |
| **TEST 17** | ADC distance finite and non-negative | Verified LUT and accumulated distances $\ge 0$, no NaNs | **PASS** |
| **TEST 18** | ADC ranking produced correctly | Verified monotonic ordering and deduplicated IDs | **PASS** |
| **TEST 19** | Recall evaluation matches calculation | Verified all recalls $\in [0, 1]$ and monotonic with $k$ | **PASS** |
| **TEST 20** | Phase 1/2/3 source files untouched | Verified SHA-256 for all 11 existing source files | **PASS** |
| **TEST 21** | ADC mathematical identity | Verified $\|D_{ADC} - \|q - \hat{x}\|^2\|_\infty < 10^{-4}$ | **PASS** |
| **TEST 22** | Tier migration invariants | Verified promotion & demotion preserve capacity invariants | **PASS** |
| **TEST 23** | Compression accounting exact | Verified $512 / 16 == 32.0\times$ code-only ratio | **PASS** |

Regression check: `tests/test_phase3.py` was also executed and passed 15/15 tests with zero regressions.

---

### 13. Determinism & Cryptographic Hashes

Phase 4 was executed across multiple independent runs to verify byte-for-byte reproducibility:

```
[Reproducibility Check]
Centroids exact match : True
Codes exact match     : True
New codes SHA-256     : 984ac8cf84d4c7ffd0a8ce1a4901fbaaedb9bdd96210f5e0c4d73473fcc3aca9
Orig codes SHA-256    : 984ac8cf84d4c7ffd0a8ce1a4901fbaaedb9bdd96210f5e0c4d73473fcc3aca9
Codes Hash Match      : True (Byte-identical)
```

#### Primary Phase 4 Artifact Hash Table

| Artifact Path | Format | Size | Cryptographic SHA-256 Digest |
| :--- | :--- | :--- | :--- |
| `results/phase4_pq/codebooks/pq_codebooks.npy` | NumPy NPY | 131,200 bytes | `dfaae246699e1e75dfb87a7480a6057a8745fbe824a3365a1456b6901426559c` |
| `results/phase4_pq/codebooks/faiss_pq_quantizer.bin` | FAISS Binary | 131,104 bytes | `2f9f09e6c43422df563ee9a3c9db8d73b5bf5dd7135e07663e0e7a2b9eb0155a` |
| `results/phase4_pq/codes/base_pq_codes.bin` | Flat uint8 | 1,600,000 bytes | `984ac8cf84d4c7ffd0a8ce1a4901fbaaedb9bdd96210f5e0c4d73473fcc3aca9` |
| `results/phase4_pq/codes/base_pq_codes.npy` | NumPy NPY | 1,600,128 bytes | `e2a44a1fa62529940aa3eeb59be249f3e49e0c5285fb823fcce316d9b23b1f13` |
| `results/phase4_pq/metadata/tier_manifest.json` | JSON | 1,877 bytes | `faee4c871ec26d306b47c030386372e1284d720b669fae7924ff432c6e61f237` |
| `results/phase4_pq/metadata/phase5_handoff_manifest.json` | JSON | 1,969 bytes | `eb4df5da6e382db2e6bf564d262d14cb8fc1a6b0c2a2cbaf8bda07897da25514` |

---

### 14. Phase 5 Handoff Readiness

Phase 4 has successfully generated and exported all required inputs for Phase 5 (DiskANN + NVMe Out-of-Core Search):

1. **Frozen Full-Precision Vectors**: `data/base_vectors.fvecs` preserved untouched on disk.
2. **NVMe Storage Mapping**: `results/phase4_pq/metadata/cold_vector_manifest.json` provides the exact list of 82,000 COLD vector IDs and direct byte offsets (`offset = id * 128 * 4`).
3. **DRAM-Resident PQ Codes**: `results/phase4_pq/codes/base_pq_codes.bin` (1.6 MB) allows Phase 5 DiskANN to execute fast 1-pass PQ distance pruning before reading full vectors from NVMe.
4. **Codebooks**: `results/phase4_pq/codebooks/pq_codebooks.npy` available for Look-Up Table construction in Phase 5.
5. **Quality Baseline Established**: Phase 5 can benchmark its DiskANN graph routing recall directly against the ADC baseline ($R@10=13.74\%$, Candidate $R@10@100=44.38\%$) and Exact FP32 oracle ($1.0000$).

---

### 15. Limitations & Future Work

1. **Subquantizer Allocation ($M=16$)**: $M=16$ provides a 32× code compression ratio. In Phase 6, we can evaluate $M=32$ (16× compression, higher recall) or residual quantization (RQ) if higher candidate recall is required prior to reranking.
2. **Unstructured Synthetic Data**: The frozen dataset is Gaussian isotropic synthetic data, where subspace correlations are minimal. On real-world clustered datasets (e.g., SIFT1M, GIST1M, deep image embeddings), PQ reconstruction error is typically significantly lower and recall is higher.
3. **DiskANN Graph Construction**: Phase 4 only prepares the data model. Building the Vamana graph on NVMe and implementing sector-aligned direct I/O will be tackled in Phase 5.

---

### Verification Commands

```powershell
# 1. Execute Full Phase 4 Pipeline
.\.venv\Scripts\python.exe src/phase4_pq.py

# 2. Run Automated Test Suite (23 Tests)
.\.venv\Scripts\python.exe -m unittest tests/test_phase4.py

# 3. Inspect Artifacts and Representations
.\.venv\Scripts\python.exe src/phase4_pq_inspect.py --all

# 4. Run Phase 3 Regression Test Suite (15 Tests)
.\.venv\Scripts\python.exe -m unittest tests/test_phase3.py
```
