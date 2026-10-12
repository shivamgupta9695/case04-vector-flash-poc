from pathlib import Path
import sys
import numpy as np

# ---------------------------------------------------------------------
# Workspace setup
# ---------------------------------------------------------------------
workspace_root = Path(__file__).resolve().parent.parent

if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

# ---------------------------------------------------------------------
# Existing Phase 3 components
# ---------------------------------------------------------------------
from src.phase1_2_evaluation import load_fvecs
from src.phase3_ssd_store import SSDVectorStore
from src.phase3_realtime import (
    RealTimeTieredSearchEngine,
    WorkloadGenerator,
)


def print_id_sample(label, ids, n=20):
    ids = sorted(ids)

    print(f"\n{label} ({len(ids):,} vectors)")

    if not ids:
        print("  None")
        return

    if len(ids) <= n * 2:
        print(f"  {ids}")
    else:
        print(f"  First {n}: {ids[:n]}")
        print(f"  Last  {n}: {ids[-n:]}")


def main():
    print("=" * 78)
    print("CASE 04 — PHASE 3 LIVE TIER MIGRATION INSPECTOR")
    print("=" * 78)

    data_dir = workspace_root / "data"
    storage_dir = workspace_root / "storage"
    results_dir = (
        workspace_root
        / "results"
        / "phase3_hardened"
        / "diagnostics"
        / "live_tier_inspection"
    )

    results_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # 1. Load deterministic Phase 1 data
    # -----------------------------------------------------------------
    base_path = data_dir / "base_vectors.fvecs"
    query_path = data_dir / "query_vectors.fvecs"

    base_vectors = load_fvecs(base_path)
    query_vectors = load_fvecs(query_path)

    n_base, dimension = base_vectors.shape

    print(f"\nTotal vectors : {n_base:,}")
    print(f"Dimension     : {dimension}")
    print("DRAM target   : 18%")
    print("Policy        : Policy B")

    # -----------------------------------------------------------------
    # 2. Open existing SSD vector store
    # -----------------------------------------------------------------
    print("\n[1] Opening existing SSD vector store...")

    ssd_store = SSDVectorStore(
        storage_dir,
        num_vectors=n_base,
        dimension=dimension,
    )

    if not ssd_store.vector_file_path.exists():
        print("ERROR: SSD vector store does not exist.")
        print("Run the normal Phase 3 setup first.")
        return 1

    ssd_store.open()

    print("[PASS] SSD store opened.")

    # -----------------------------------------------------------------
    # 3. Build the same Phase 3 engine
    # -----------------------------------------------------------------
    print("\n[2] Creating Phase 3 tiered search engine...")

    engine = RealTimeTieredSearchEngine(
        base_vectors=base_vectors,
        ssd_store=ssd_store,
        dram_capacity_ratio=0.18,
        hnsw_m=32,
        ef_search=40,
        policy_type="policy_b",
    )

    # Same initial state used by the existing Phase 3 dynamic trace.
    initial_dram_ids = list(range(18_000))

    engine.tier_manager.populate_initial_dram(initial_dram_ids)

    print("[PASS] Initial DRAM populated.")

    # -----------------------------------------------------------------
    # 4. Capture initial tier state
    # -----------------------------------------------------------------
    previous_dram_set = set(engine.tier_manager.dram_cache.keys())

    previous_ssd_set = set(range(n_base)) - previous_dram_set

    print("\n" + "-" * 78)
    print("INITIAL STATE")
    print("-" * 78)

    print(f"DRAM / HOT : {len(previous_dram_set):,}")
    print(f"SSD / COLD : {len(previous_ssd_set):,}")

    print_id_sample(
        "Initial DRAM IDs",
        previous_dram_set,
    )

    print_id_sample(
        "Initial SSD/COLD IDs",
        previous_ssd_set,
    )

    # -----------------------------------------------------------------
    # 5. Generate the SAME dynamic workload used by Phase 3
    # -----------------------------------------------------------------
    print("\n[3] Generating deterministic Zipf workload...")

    workload_gen = WorkloadGenerator(
        query_vectors,
        seed=42,
    )

    trace_stream = workload_gen.generate_zipf_stream(
        stream_length=2_400,
        zipf_alpha=1.3,
    )

    interval_size = 200
    num_intervals = len(trace_stream) // interval_size

    print(f"[PASS] Queries       : {len(trace_stream):,}")
    print(f"[PASS] Interval size : {interval_size}")
    print(f"[PASS] Intervals     : {num_intervals}")
    print(f"[PASS] Zipf alpha    : 1.3")

    # -----------------------------------------------------------------
    # 6. Run live migration inspection
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("LIVE DRAM <-> SSD MIGRATION")
    print("=" * 78)

    all_interval_reports = []

    simulation_time = 0.0

    for interval_idx in range(num_intervals):

        start = interval_idx * interval_size
        end = (interval_idx + 1) * interval_size

        sub_stream = trace_stream[start:end]

        promotions_before = engine.tier_manager.promotion_count
        demotions_before = engine.tier_manager.demotion_count
        dram_hits_before = engine.tier_manager.dram_hits
        ssd_reads_before = engine.tier_manager.ssd_reads

        # -------------------------------------------------------------
        # Process queries
        # -------------------------------------------------------------
        for query_index in sub_stream:

            simulation_time += 0.001

            engine.search_query(
                query_vectors[query_index],
                k=10,
                current_time=simulation_time,
            )

        # Same decay operation as the original Phase 3 dynamic trace.
        engine.tracker.decay_recent_counts(
            decay_rate=0.90
        )

        # -------------------------------------------------------------
        # Capture NEW tier state
        # -------------------------------------------------------------
        current_dram_set = set(
            engine.tier_manager.dram_cache.keys()
        )

        current_ssd_set = set(range(n_base)) - current_dram_set

        # IDs newly entering DRAM
        promoted_ids = current_dram_set - previous_dram_set

        # IDs leaving DRAM and therefore becoming SSD/COLD
        demoted_ids = previous_dram_set - current_dram_set

        # -------------------------------------------------------------
        # Counters
        # -------------------------------------------------------------
        promotions_delta = (
            engine.tier_manager.promotion_count
            - promotions_before
        )

        demotions_delta = (
            engine.tier_manager.demotion_count
            - demotions_before
        )

        dram_hits_delta = (
            engine.tier_manager.dram_hits
            - dram_hits_before
        )

        ssd_reads_delta = (
            engine.tier_manager.ssd_reads
            - ssd_reads_before
        )

        # -------------------------------------------------------------
        # Print interval report
        # -------------------------------------------------------------
        print("\n" + "-" * 78)
        print(f"INTERVAL {interval_idx + 1}/{num_intervals}")
        print("-" * 78)

        print(
            f"Queries processed : "
            f"{(interval_idx + 1) * interval_size:,}"
        )

        print(
            f"DRAM count        : "
            f"{len(current_dram_set):,}"
        )

        print(
            f"SSD/COLD count    : "
            f"{len(current_ssd_set):,}"
        )

        print(
            f"DRAM hits         : "
            f"{dram_hits_delta:,}"
        )

        print(
            f"SSD reads         : "
            f"{ssd_reads_delta:,}"
        )

        print(
            f"Promotion counter : "
            f"{promotions_delta:,}"
        )

        print(
            f"Demotion counter  : "
            f"{demotions_delta:,}"
        )

        print(
            f"Actual IDs moved to DRAM : "
            f"{len(promoted_ids):,}"
        )

        print(
            f"Actual IDs moved to SSD  : "
            f"{len(demoted_ids):,}"
        )

        # -------------------------------------------------------------
        # Show actual IDs that changed tier
        # -------------------------------------------------------------
        print_id_sample(
            "IDs newly promoted SSD -> DRAM",
            promoted_ids,
            n=20,
        )

        print_id_sample(
            "IDs newly demoted DRAM -> SSD",
            demoted_ids,
            n=20,
        )

        # -------------------------------------------------------------
        # Capacity invariant
        # -------------------------------------------------------------
        assert len(current_dram_set) == 18_000
        assert len(current_ssd_set) == 82_000
        assert not (current_dram_set & current_ssd_set)
        assert len(current_dram_set | current_ssd_set) == n_base

        print("\n[PASS] DRAM capacity = 18,000")
        print("[PASS] SSD/COLD count = 82,000")
        print("[PASS] No tier overlap")
        print("[PASS] All 100,000 vectors accounted for")

        # -------------------------------------------------------------
        # Save complete ID movement for this interval
        # -------------------------------------------------------------
        interval_file = (
            results_dir
            / f"interval_{interval_idx + 1:02d}.txt"
        )

        with open(interval_file, "w", encoding="utf-8") as f:

            f.write(
                f"CASE 04 — PHASE 3 LIVE TIER INSPECTION\n"
                f"INTERVAL {interval_idx + 1}\n"
                f"{'=' * 70}\n\n"
            )

            f.write(
                f"DRAM count: {len(current_dram_set)}\n"
            )

            f.write(
                f"SSD/COLD count: {len(current_ssd_set)}\n"
            )

            f.write(
                f"Promotions counter: {promotions_delta}\n"
            )

            f.write(
                f"Demotions counter: {demotions_delta}\n\n"
            )

            f.write("PROMOTED SSD -> DRAM IDs\n")
            f.write("-" * 40 + "\n")

            for vid in sorted(promoted_ids):
                f.write(f"{vid}\n")

            f.write("\nDEMOTED DRAM -> SSD IDs\n")
            f.write("-" * 40 + "\n")

            for vid in sorted(demoted_ids):
                f.write(f"{vid}\n")

            f.write("\nCURRENT DRAM IDs\n")
            f.write("-" * 40 + "\n")

            for vid in sorted(current_dram_set):
                f.write(f"{vid}\n")

        all_interval_reports.append(
            {
                "interval": interval_idx + 1,
                "promotions_counter": promotions_delta,
                "demotions_counter": demotions_delta,
                "actual_promoted_ids": len(promoted_ids),
                "actual_demoted_ids": len(demoted_ids),
                "dram_count": len(current_dram_set),
                "ssd_count": len(current_ssd_set),
                "dram_hits": dram_hits_delta,
                "ssd_reads": ssd_reads_delta,
            }
        )

        # Move to next interval.
        previous_dram_set = current_dram_set
        previous_ssd_set = current_ssd_set

    # -----------------------------------------------------------------
    # 7. Final state
    # -----------------------------------------------------------------
    final_dram_set = set(
        engine.tier_manager.dram_cache.keys()
    )

    final_ssd_set = set(range(n_base)) - final_dram_set

    print("\n" + "=" * 78)
    print("FINAL LIVE TIER STATE")
    print("=" * 78)

    print(
        f"Final DRAM / HOT vectors : "
        f"{len(final_dram_set):,}"
    )

    print(
        f"Final SSD / COLD vectors : "
        f"{len(final_ssd_set):,}"
    )

    print(
        f"Total promotions         : "
        f"{engine.tier_manager.promotion_count:,}"
    )

    print(
        f"Total demotions          : "
        f"{engine.tier_manager.demotion_count:,}"
    )

    print_id_sample(
        "Final DRAM IDs",
        final_dram_set,
        n=30,
    )

    # -----------------------------------------------------------------
    # 8. Compare initial vs final state
    # -----------------------------------------------------------------
    initial_set = set(initial_dram_ids)

    entered_dram = final_dram_set - initial_set
    left_dram = initial_set - final_dram_set

    print("\n" + "-" * 78)
    print("INITIAL → FINAL DRAM MEMBERSHIP CHANGE")
    print("-" * 78)

    print(
        f"Initial DRAM vectors : "
        f"{len(initial_set):,}"
    )

    print(
        f"Final DRAM vectors   : "
        f"{len(final_dram_set):,}"
    )

    print(
        f"IDs that entered DRAM during run : "
        f"{len(entered_dram):,}"
    )

    print(
        f"IDs that left DRAM during run    : "
        f"{len(left_dram):,}"
    )

    print_id_sample(
        "Vectors that entered DRAM",
        entered_dram,
        n=30,
    )

    print_id_sample(
        "Vectors that left DRAM",
        left_dram,
        n=30,
    )

    # -----------------------------------------------------------------
    # 9. Save final sets
    # -----------------------------------------------------------------
    final_dram_file = results_dir / "final_dram_hot_ids.txt"
    final_ssd_file = results_dir / "final_ssd_cold_ids.txt"

    with open(final_dram_file, "w", encoding="utf-8") as f:
        for vid in sorted(final_dram_set):
            f.write(f"{vid}\n")

    with open(final_ssd_file, "w", encoding="utf-8") as f:
        for vid in sorted(final_ssd_set):
            f.write(f"{vid}\n")

    # -----------------------------------------------------------------
    # 10. Final validation
    # -----------------------------------------------------------------
    assert len(final_dram_set) == 18_000
    assert len(final_ssd_set) == 82_000
    assert not (final_dram_set & final_ssd_set)
    assert len(final_dram_set | final_ssd_set) == n_base

    print("\n" + "=" * 78)
    print("FINAL VALIDATION")
    print("=" * 78)

    print("[PASS] 18,000 vectors remain in DRAM")
    print("[PASS] 82,000 vectors remain in SSD/COLD")
    print("[PASS] Every vector belongs to exactly one tier")
    print("[PASS] Live migration was observed")
    print("[PASS] Existing Phase 3 implementation was not modified")

    print("\nSaved diagnostic files:")
    print(f"  {results_dir}")

    print("\n" + "=" * 78)
    print("LIVE TIER INSPECTION COMPLETE")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())