#!/usr/bin/env python3
"""
scripts/run_experiments.py
Automated evaluation harness for PM-MLFQ.
Runs all 5 scheduler modes across 6 benchmark experiments and 5 seeds (150 total trials).
Saves raw serial output to results/raw/<mode>_<exp>_<seed>.log.
"""

import os
import sys
import subprocess
import time

sys.path.append("/home/sai/OS Project")
from scripts.qemu_runner import run_qemu

MODES = [0, 1, 2, 3, 4]
MODE_NAMES = {
    0: "RR",
    1: "MLFQ",
    2: "ML_PROB",
    3: "ML_DET",
    4: "RAND_CTRL"
}

EXPERIMENTS = [1, 2, 3, 4, 5, 6]
SEEDS = [101, 102, 103, 104, 105]

XV6_DIR = "/home/sai/OS Project/xv6-riscv"
RESULTS_RAW_DIR = "/home/sai/OS Project/results/raw"

def build_kernel(mode):
    print(f"\n=======================================================")
    print(f"Building xv6 Kernel for Mode {mode} ({MODE_NAMES[mode]})...")
    print(f"=======================================================")
    cmd = f"cd '{XV6_DIR}' && rm -f kernel/*.o kernel/kernel && make SCHED_MODE={mode} SCHED_LOG=1 -j$(nproc)"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"ERROR: Build failed for mode {mode}:\n{res.stderr}")
        sys.exit(1)
    print(f"Build successful for Mode {mode}.")

def main():
    os.makedirs(RESULTS_RAW_DIR, exist_ok=True)
    total_trials = len(MODES) * len(EXPERIMENTS) * len(SEEDS)
    completed = 0
    failed = 0

    start_time = time.time()
    print(f"Starting PM-MLFQ Evaluation Suite: {total_trials} total trials.")

    for mode in MODES:
        build_kernel(mode)
        mode_str = f"mode{mode}"

        for exp in EXPERIMENTS:
            for seed in SEEDS:
                completed += 1
                log_file = os.path.join(RESULTS_RAW_DIR, f"{mode_str}_exp{exp}_seed{seed}.log")
                print(f"[{completed}/{total_trials}] Running {mode_str} ({MODE_NAMES[mode]}) | Exp {exp} | Seed {seed}...")

                cmd = [f"schedtest {exp} {seed}"]
                timeout = 16 if exp == 5 else 12

                out = run_qemu(cmd, timeout=timeout)

                # Save raw serial log
                with open(log_file, "w") as f:
                    f.write(out)

                # Verification check
                if "SCHEDTEST_DONE" in out:
                    # Count logged child processes
                    schedlog_count = out.count("SCHEDLOG pid=")
                    print(f"  -> SUCCESS ({schedlog_count} processes logged)")
                else:
                    print(f"  -> WARNING/TIMEOUT: SCHEDTEST_DONE not detected!")
                    failed += 1

    elapsed = time.time() - start_time
    print(f"\n=======================================================")
    print(f"Experiments Finished in {elapsed:.1f}s!")
    print(f"Total: {total_trials}, Completed: {completed}, Warnings/Failures: {failed}")
    print(f"Raw logs saved to {RESULTS_RAW_DIR}/")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
