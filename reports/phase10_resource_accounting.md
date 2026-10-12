# Case 04 — Phase 10: Resource Accounting Report (Memory, Storage & SSD I/O)
## Mathematical Byte Reconciliation, Memory Boundaries, Physical Storage & Unbuffered I/O Analysis

**Author:** Senior Systems Engineer & Independent Validation Auditor  
**Date:** October 11, 2026  
**Project:** `Case04_Vector_POC` — Final Phase 10 Audit  
**Status:** **GATE R4 PASSED** (Strict distinction between physical files, logical allocations, theoretical models, measured process working sets, and unbuffered OS file reads).

---

## 1. Accounting Principles & Unit Taxonomy

To eliminate historical ambiguities and maintain scientific precision, Phase 10 adheres to strict unit definitions:
- **Binary Units (MiB, GiB):** Based on powers of 2 ($1\text{ KiB} = 1,024\text{ B}$, $1\text{ MiB} = 1,048,576\text{ B}$, $1\text{ GiB} = 1,073,741,824\text{ B}$).
- **Decimal Units (MB, GB):** Based on powers of 10 ($1\text{ KB} = 1,000\text{ B}$, $1\text{ MB} = 1,000,000\text{ B}$, $1\text{ GB} = 1,000,000,000\text{ B}$).
- **Logical Allocation:** Vector payload size assigned to a tier (e.g., the 82,000 COLD vector partition).
- **Physical Allocation:** Actual on-disk file size reported by the operating system filesystem.
- **In-Memory Structural Footprint:** The sum of memory-resident index structures and vector buffers required for execution.
- **Process Working Set:** Physical RAM allocated to the process by the OS memory manager (`WorkingSet64`).

---

## 2. Storage Accounting: Physical Files vs. Logical Partitions

A forensic scan of the storage filesystem confirms the exact byte sizes of all index and vector assets:

### Table 1: Physical File Accounting on NVMe Storage
| Storage Component | Exact Verified File Path | Physical Byte Size | Size in MiB | Size in MB | Description & Lifecycle |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **COLD Disk Index** | `results/phase5_diskann/index/cold_disk_index_tuned_disk.index` | 67,178,496 B | **64.07 MiB** | 67.18 MB | Tuned Vamana graph on NVMe ($R=64, L=100$) |
| **COLD Compressed PQ** | `results/phase5_diskann/index/cold_disk_index_tuned_pq_compressed.bin`| 2,624,008 B | **2.50 MiB** | 2.62 MB | 82k vectors $\times$ 32 B PQ code + 8-byte header |
| **COLD PQ Centroids** | `results/phase5_diskann/index/cold_disk_index_tuned_pq_pivots.bin` | 135,836 B | **0.13 MiB** | 0.14 MB | 256 centroids $\times$ 128 dim FP32 codebook |
| **Raw Base Vectors** | `storage/ssd_vectors/vectors.bin` | 51,200,000 B | **48.83 MiB** | 51.20 MB | All 100k raw 128-dim FP32 vectors on NVMe |
| **Total Physical Disk Files**| *Sum of all on-disk index & vector files* | **121,138,340 B** | **115.53 MiB** | **121.14 MB** | **Total persistent NVMe storage footprint** |

### Table 2: Logical Allocation Accounting
| Logical Partition | Component Files Included | Exact Byte Size | Size in MiB | Percentage of Dataset |
| :--- | :--- | :---: | :---: | :---: |
| **HOT Logical Partition** | 18,000 raw FP32 vectors ($18,000 \times 512\text{ B}$) | 9,216,000 B | **8.79 MiB** | 18.0% |
| **COLD Logical Partition** | 82,000 raw FP32 vectors ($82,000 \times 512\text{ B}$) + Disk Index | 109,162,496 B | **104.11 MiB** | 82.0% |
| **COLD In-Memory PQ Table**| DiskANN PQ compressed codes + pivots | 2,759,844 B | **2.63 MiB** | — |

> [!NOTE]
> **Resolution of Historical "11.42 MB" Figure:**  
> The 11.42 MiB figure reported in earlier summaries equals the exact mathematical difference between total physical storage (115.53 MiB) and the logical COLD budget (104.11 MiB):  
> $$115.53\text{ MiB} - 104.11\text{ MiB} = \mathbf{11.42\text{ MiB}}$$  
> This represents the 8.79 MiB HOT vector slice resident inside `vectors.bin` ($48.83 - 40.04 = 8.79\text{ MiB}$) plus the 2.63 MiB PQ compressed code table and pivot artifacts ($8.79 + 2.63 = 11.42\text{ MiB}$).

---

## 3. Memory Accounting: Hybrid Pipeline vs. Native Milvus

### 3.1 Hybrid Search Memory Residency (Calculated Structural Breakdown)

During query execution, the hybrid architecture holds the following data structures in DRAM:
1. **HOT FAISS HNSW Index (`hot_hnsw.index`):** **13.46 MiB** (14,114,082 bytes).
   - FAISS `IndexHNSWFlat` stores its vectors internally inside an `IndexFlatL2` storage object:
     $$\text{Internal Vector Storage} = 18,000 \times 128 \times 4 = 9,216,000\text{ bytes (8.79 MiB)}$$
     $$\text{Graph Adjacency Links} = 14,114,082 - 9,216,000 = 4,898,082\text{ bytes (4.67 MiB)}$$
2. **HOT FP32 Vector Buffer (`TieredVectorStore.hot_dram_vectors`):** **8.79 MiB** (9,216,000 bytes).
   - Retained as a contiguous NumPy array in Python user space to allow microsecond-level BLAS distance recomputation without serializing out of FAISS.
3. **COLD DiskANN PQ Tables:** **2.63 MiB** (2,759,844 bytes).
   - 2.50 MiB compressed 1-byte PQ codes + 0.13 MiB centroid pivot table loaded into memory by DiskANN to perform distance estimation during graph traversal.
4. **Working Buffers & Cache:** **1.66 MiB** (1,740,634 bytes).
   - Candidate merge dictionaries, deduplication hash sets, and query scratch buffers.
- **Total In-Memory Structural Footprint:** **26.54 MiB** (27.83 MB).

### 3.2 Native Milvus Memory: Theoretical Floor vs. Measured Working Set

To prevent invalid comparisons between an in-memory structural calculation and an entire OS process working set:

1. **Theoretical Floor Estimate (76.50 MB / 72.96 MiB):**
   - $100,000\text{ vectors} \times 128 \times 4\text{ bytes} = 51.20\text{ MB}$ raw vector buffer.
   - Minimal C++ Knowhere segment buffers and execution overhead $\approx 25.30\text{ MB}$.
   - Sum = **76.50 MB**.
2. **Actual Measured Process Working Set (Windows Kernel Counter):**
   - Interrogated live via Windows Process Memory Counters (`(Get-Process -Id $PID).WorkingSet64`):
     - Python process working set before vector insertion: **159.89 MB**.
     - Python process working set after inserting 100,000 vectors into Milvus-Lite: **405.33 MB**.
     - **Net Working Set Delta:** **245.43 MB** (234.06 MiB).
   - Explanation: Milvus-Lite embeds its C++ Knowhere search engine within the process. Upon inserting 100,000 vector records, Milvus creates in-memory segment chunks, index metadata, and RPC marshaling buffers.

### Table 3: Reconciled DRAM Comparison
| System / Model | Accounting Basis | DRAM Footprint | Comparison vs. Hybrid Pipeline |
| :--- | :--- | :---: | :---: |
| **Hybrid Pipeline (Standalone/Adapter)**| **Calculated Structural Footprint** | **26.54 MiB** (27.83 MB) | **Baseline (100%)** |
| **Native Milvus FLAT (Theoretical)** | **Minimal Segment Estimate** | **76.50 MB** (72.96 MiB) | **63.6% DRAM Savings** |
| **Native Milvus FLAT (Measured)** | **Measured Process WorkingSet Delta** | **245.43 MB** (234.06 MiB) | **88.7% DRAM Savings** |

> [!NOTE]
> **Unit Conversion & Arithmetic Reconciliation:**  
> To perform a mathematically valid comparison against decimal values (76.50 MB and 245.43 MB), the hybrid structural footprint of **26.54 MiB** must first be converted to decimal megabytes:  
> $$26.54\text{ MiB} = 26.54 \times 1,048,576\text{ bytes} = 27,829,207\text{ bytes} \approx \mathbf{27.83\text{ MB}}$$  
> Computing the percentage reduction using consistent decimal MB units yields:  
> - **Vs. Theoretical Floor (76.50 MB):**  
>   $$\frac{76.50\text{ MB} - 27.83\text{ MB}}{76.50\text{ MB}} = \frac{48.67}{76.50} = \mathbf{63.62\% \approx 63.6\%}$$  
>   *(Equivalently in binary units: $\frac{72.96\text{ MiB} - 26.54\text{ MiB}}{72.96\text{ MiB}} = \mathbf{63.62\%}$)*.  
> - **Vs. Measured Process Working Set Delta (245.43 MB):**  
>   $$\frac{245.43\text{ MB} - 27.83\text{ MB}}{245.43\text{ MB}} = \frac{217.60}{245.43} = \mathbf{88.66\% \approx 88.7\%}$$  
>   *(Equivalently in binary units: $\frac{234.06\text{ MiB} - 26.54\text{ MiB}}{234.06\text{ MiB}} = \mathbf{88.66\%}$)*.  
>  
> Earlier draft reports cited 65.3% and 89.2% because the binary quantity ($26.54\text{ MiB}$) was erroneously subtracted directly from the decimal values ($76.50\text{ MB}$ and $245.43\text{ MB}$): $\frac{76.50 - 26.54}{76.50} = 65.3\%$ and $\frac{245.43 - 26.54}{245.43} = 89.2\%$. Converting to consistent units resolves this discrepancy.
>  
> **Crucial Distinction in Accounting Bases:**  
> The calculated structural footprint (27.83 MB / 26.54 MiB) and the measured process working-set delta (245.43 MB) are **fundamentally different accounting bases**:  
> 1. Calculated structural footprint is a bottom-up summation of vector array payloads and index graph data structures.  
> 2. Measured working-set delta is an end-to-end OS kernel metric capturing physical RAM pages allocated to the process, including the Milvus C++ Knowhere shared libraries, thread stacks, memory arena pre-allocations, and internal runtime buffers.  
> They are presented side-by-side to provide both lower-bound structural and upper-bound empirical process perspectives, but are not interchangeable.

---

## 4. SSD I/O Activity and Environment Limitations

### 4.1 Implementation & Instrumentation Details (Per 1,000 Queries, K=10)
In [`src/phase3_ssd_store.py`](file:///C:/Users/shivam.gupta/Desktop/Case04_Vector_POC/src/phase3_ssd_store.py#L121-L187), `SSDVectorStore` manages direct random access to `storage/ssd_vectors/vectors.bin`:
1. **File Handle API:**
   ```python
   self._file_handle = open(self.vector_file_path, "rb", buffering=0)
   ```
   This instantiates an unbuffered binary stream (`io.FileIO`) in Python.
2. **Read Vector API:**
   During candidate reranking, `read_vectors_batch(vector_ids)` iterates over candidate IDs and calls:
   ```python
   offset = int(vid) * self.record_bytes  # record_bytes = 512
   self._file_handle.seek(offset)
   raw = self._file_handle.read(self.record_bytes)
   ```
   The exact API executed is **`self._file_handle.read(512)`** (Python unbuffered `FileIO.read(512)`), **not** low-level POSIX `os.read(fd, 512)`.
3. **Instrumentation Counter Source:**
   The reported metrics originate from internal Python software counters incremented in user space:
   ```python
   self.total_reads += n
   self.total_bytes_read += n * self.record_bytes
   ```
   For 1,000 queries with 50 COLD candidates each, this accumulates exactly:
   - **Application-Level Read Calls:** $50 \times 1,000 = \mathbf{50,000\text{ read(512) calls}}$.
   - **Requested Payload Bytes:** $50,000 \times 512 = 25,600,000\text{ bytes} = \mathbf{24.414\text{ MiB}}$ (25.60 MB).
   - **HOT-Only Mode:** Strictly **0 read calls (0.0 MB)**.
   - **Adaptive Router Mode ($\tau=145.0$):** **42,050 read calls (20.53 MiB)** (a 15.9% reduction in I/O requests due to early DRAM exits).

### 4.2 Application Read Calls vs. Physical Flash Device Access

> [!IMPORTANT]
> **Physical Device Clarification & No-Guarantee Principle:**  
> The reported 50,000 reads and 24.41 MiB represent **application-level unbuffered file read calls and requested byte payloads** issued to the operating system.  
> **Unbuffered Python I/O (`buffering=0`) does NOT guarantee physical NAND flash media access:**  
> 1. `buffering=0` in Python disables only the Python runtime's user-space C stream buffers (`FILE*` buffers).  
> 2. It does **not** bypass the Windows OS filesystem cache (NTFS cache manager), the Linux VFS page cache inside WSL2, or the hardware controller RAM cache on the BC711 NVMe SSD.  
> 3. Modern NVMe storage controllers read data in 4 KiB physical sector or flash page units, not 512-byte slices.  
> 4. Physical NAND bus transfers were not measured via hardware bus sniffers; therefore, the reported numbers reflect **application-level OS filesystem read requests**, not raw physical flash operations.

**Gate R4 Verdict:** **PASS (All component byte sizes reconciled; unit conversion verified; memory accounting bases separated; unbuffered OS read call semantics scientifically documented).**
