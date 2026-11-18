#!/usr/bin/env python3
"""
scripts/collect_real_traces.py
Runs actual xv6 workloads in QEMU and extracts MLTRACE feature vectors to build ml/real_traces.csv.
"""

import re
import csv
import os
import sys

sys.path.append("/home/sai/OS Project")
from scripts.qemu_runner import run_qemu

WORKLOAD_MAP = {
    "cpuwork": 2,      # CPU-bound
    "iowork": 1,       # I/O-bound
    "interactive": 0,  # Interactive
    "mixedwork": 3     # Mixed
}

def collect():
    print("Collecting real xv6 execution traces...")
    traces = []
    
    # We will run several varied batches of workloads
    batches = [
        # Batch 1: Standard runs
        ["cpuwork 250000000", "iowork 30", "interactive 15", "mixedwork 10"],
        # Batch 2: Heavier I/O & Interactive
        ["iowork 40", "interactive 20", "cpuwork 300000000", "mixedwork 15"],
        # Batch 3: Shorter runs
        ["cpuwork 150000000", "iowork 20", "interactive 10", "mixedwork 8"],
        # Batch 4: Concurrent background workloads
        ["cpuwork 200000000 &", "interactive 15", "iowork 25 &", "mixedwork 12"]
    ]
    
    for idx, batch in enumerate(batches):
        print(f"Running trace collection batch {idx+1}/{len(batches)}: {batch}")
        out = run_qemu(batch, timeout=16)
        
        # Parse MLTRACE lines
        # Format: MLTRACE pid=%d name=%s feat=%d,%d,%d,%d,%d,%d,%d,%d,%d
        matches = re.findall(r"MLTRACE pid=\d+ name=(\w+) feat=([\d,-]+)", out)
        for name, feat_str in matches:
            if name in WORKLOAD_MAP:
                class_id = WORKLOAD_MAP[name]
                feats = [int(x) for x in feat_str.split(",")]
                if len(feats) == 9:
                    traces.append([name, class_id] + feats)
                    print(f"  Captured trace for {name} (Class {class_id}): {feats}")

    csv_path = "/home/sai/OS Project/ml/real_traces.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "workload_name", "class_id",
            "avg_burst", "max_burst", "burst_mad", "exhaust_ratio",
            "block_rate", "yield_rate", "sleep_ratio", "cpu_util", "burst_count"
        ])
        writer.writerows(traces)
        
    print(f"\nSuccessfully collected {len(traces)} real xv6 traces into {csv_path}")

if __name__ == "__main__":
    collect()
