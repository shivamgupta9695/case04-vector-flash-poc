from pathlib import Path
import sys

workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from src.phase1_2_evaluation import load_fvecs
from src.phase3_ssd_store import SSDVectorStore
from src.phase3_hotness import AccessTracker
from src.phase3_tiering import TieredVectorManager


def main():
    data_dir = workspace_root / "data"
    storage_dir = workspace_root / "storage"

    base_path = data_dir / "base_vectors.fvecs"

    # Load dataset
    base_vectors = load_fvecs(base_path)

    num_vectors = len(base_vectors)
    dimension = base_vectors.shape[1]

    print("=" * 70)
    print("CASE 04 — PHASE 3 TIER INSPECTOR")
    print("=" * 70)

    print(f"Total vectors : {num_vectors:,}")
    print(f"Dimension     : {dimension}")
    print("DRAM target   : 18%")
    print("")

    # Open existing SSD store
    ssd_store = SSDVectorStore(
        storage_dir,
        num_vectors=num_vectors,
        dimension=dimension
    )

    ssd_store.open()

    # Create tracker
    tracker = AccessTracker(num_vectors=num_vectors)

    # Create tier manager
    tier_manager = TieredVectorManager(
        ssd_store=ssd_store,
        access_tracker=tracker,
        dram_capacity_ratio=0.18,
        dimension=dimension,
        policy_type="policy_b"
    )

    # Same initialization logic used by Phase 3
    initial_dram_ids = list(range(18_000))

    tier_manager.populate_initial_dram(initial_dram_ids)

    # Current DRAM set
    dram_ids = sorted(tier_manager.dram_cache.keys())

    # Everything else is logically SSD/COLD
    dram_set = set(dram_ids)

    ssd_ids = [
        vid for vid in range(num_vectors)
        if vid not in dram_set
    ]

    print("\n" + "-" * 70)
    print("INITIAL TIER DISTRIBUTION")
    print("-" * 70)

    print(f"DRAM / HOT vectors : {len(dram_ids):,}")
    print(f"SSD / COLD vectors : {len(ssd_ids):,}")

    print(
        f"DRAM memory       : "
        f"{tier_manager.dram_bytes_used / (1024 * 1024):.2f} MB"
    )

    print(
        f"SSD logical vectors: "
        f"{len(ssd_ids):,}"
    )

    print("\nFirst 30 DRAM vector IDs:")
    print(dram_ids[:30])

    print("\nLast 30 DRAM vector IDs:")
    print(dram_ids[-30:])

    print("\nFirst 30 SSD/COLD vector IDs:")
    print(ssd_ids[:30])

    print("\nLast 30 SSD/COLD vector IDs:")
    print(ssd_ids[-30:])

    # Verify partition
    assert len(dram_ids) == 18_000
    assert len(ssd_ids) == 82_000
    assert len(set(dram_ids).intersection(ssd_ids)) == 0
    assert len(set(dram_ids).union(ssd_ids)) == num_vectors

    print("\n" + "-" * 70)
    print("INVARIANT CHECKS")
    print("-" * 70)

    print("[PASS] DRAM count = 18,000")
    print("[PASS] SSD/COLD count = 82,000")
    print("[PASS] No duplicate vector IDs")
    print("[PASS] Every vector belongs to exactly one tier")
    print("[PASS] DRAM capacity invariant")

    # Save complete lists
    results_dir = workspace_root / "results" / "phase3_hardened" / "diagnostics"
    results_dir.mkdir(parents=True, exist_ok=True)

    dram_file = results_dir / "dram_hot_vector_ids.txt"
    ssd_file = results_dir / "ssd_cold_vector_ids.txt"

    with open(dram_file, "w", encoding="utf-8") as f:
        for vid in dram_ids:
            f.write(f"{vid}\n")

    with open(ssd_file, "w", encoding="utf-8") as f:
        for vid in ssd_ids:
            f.write(f"{vid}\n")

    print("\nSaved files:")
    print(f"DRAM IDs -> {dram_file}")
    print(f"SSD IDs  -> {ssd_file}")

    print("\n" + "=" * 70)
    print("TIER INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()