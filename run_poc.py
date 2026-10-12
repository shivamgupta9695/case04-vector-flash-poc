#!/usr/bin/env python3
"""
Case 04: Vector Search POC — Master Unified CLI Entry Point
Provides a clean, single-entry CLI to run regression tests, verify frozen checksums,
and execute the Phase 10 master benchmark driver.
"""

import argparse
import subprocess
import sys
from pathlib import Path

workspace_root = Path(__file__).resolve().parent


def verify_checksums() -> bool:
    print("\n" + "=" * 70)
    print("STEP 1: Cryptographic Checksum Verification of Frozen Assets (34 Files)")
    print("=" * 70)
    cmd = [
        sys.executable,
        "-c",
        """
import hashlib
import sys
from pathlib import Path
from tests.test_phase9 import FROZEN_SOURCE_CHECKSUMS, FROZEN_DATASET_CHECKSUMS

mismatches = 0
for rel, expected in {**FROZEN_SOURCE_CHECKSUMS, **FROZEN_DATASET_CHECKSUMS}.items():
    p = Path(rel)
    if not p.is_file():
        print(f'  [MISSING] {rel}')
        mismatches += 1
        continue
    act = hashlib.sha256(p.read_bytes()).hexdigest()
    if act != expected:
        print(f'  [MISMATCH] {rel}')
        mismatches += 1

if mismatches == 0:
    print('  [PASS] All 34 frozen source, test, and dataset files bitwise match 100.0%.')
    sys.exit(0)
else:
    print(f'  [FAIL] {mismatches} checksum mismatches found!')
    sys.exit(1)
""",
    ]
    res = subprocess.run(cmd, cwd=str(workspace_root))
    return res.returncode == 0


def run_tests() -> bool:
    print("\n" + "=" * 70)
    print("STEP 2: Full Multi-Phase Regression Test Suite (100 Tests)")
    print("=" * 70)
    cmd = [sys.executable, "-m", "unittest", "discover", "tests/", "-v"]
    res = subprocess.run(cmd, cwd=str(workspace_root))
    return res.returncode == 0


def run_benchmarks() -> bool:
    print("\n" + "=" * 70)
    print("STEP 3: Phase 10 Master Benchmark Driver (5 Interleaved Trials)")
    print("=" * 70)
    cmd = [sys.executable, "src/phase10_benchmark_driver.py"]
    res = subprocess.run(cmd, cwd=str(workspace_root))
    return res.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Case 04 Vector Search POC — Master Runner")
    parser.add_argument("--test", action="store_true", help="Run full 100-test regression suite")
    parser.add_argument("--bench", action="store_true", help="Run 5-trial benchmark suite")
    parser.add_argument("--checksum", action="store_true", help="Verify frozen SHA-256 checksums")
    parser.add_argument("--all", action="store_true", help="Run checksums, tests, and benchmarks")

    args = parser.parse_args()

    # Default to --all if no specific flag is given
    if not (args.test or args.bench or args.checksum or args.all):
        args.all = True

    success = True

    if args.checksum or args.all:
        if not verify_checksums():
            print("\n[ABORT] Checksum verification failed.")
            sys.exit(1)

    if args.test or args.all:
        if not run_tests():
            print("\n[ABORT] Regression test suite failed.")
            sys.exit(1)

    if args.bench or args.all:
        if not run_benchmarks():
            print("\n[ABORT] Benchmark suite failed.")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("POC EXECUTION COMPLETE: ALL REQUESTED STEPS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
