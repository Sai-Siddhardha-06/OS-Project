#!/usr/bin/env python3
"""
ml/generate_data.py
Generates synthetic behavioral training data for PM-MLFQ kernel classifier.
Calibrated to match real xv6 tick quantization and kernel feature extraction.
"""

import json
import random
import csv
import os

SEED = 42
random.seed(SEED)

NUM_SAMPLES = 6000
OBS_MIN_BURSTS = 3
OBS_MAX_BURSTS = 10
OBS_MAX_WINDOW_TICKS = 50

# 20 Blueprints (5 per class)
# class 0: Interactive, class 1: IO, class 2: CPU, class 3: Mixed
BLUEPRINTS = [
    # Interactive (0-4): Sleep-dominated (sleep_ratio 700-1000), block_rate 1000-2000, zero cpu_util
    {"id": 0, "class": 0, "sleep_total_range": (2, 3), "cpu_prob": 0.0, "yield_prob": 0.0},
    {"id": 1, "class": 0, "sleep_total_range": (1, 3), "cpu_prob": 0.0, "yield_prob": 0.05},
    {"id": 2, "class": 0, "sleep_total_range": (2, 4), "cpu_prob": 0.0, "yield_prob": 0.0},
    {"id": 3, "class": 0, "sleep_total_range": (1, 2), "cpu_prob": 0.05, "yield_prob": 0.05},
    {"id": 4, "class": 0, "sleep_total_range": (2, 3), "cpu_prob": 0.05, "yield_prob": 0.10},

    # I/O-bound (10-14): Rapid sub-tick blocks in 1 tick (block_rate 2000-4000), zero sleep_ratio, zero cpu_util
    {"id": 10, "class": 1, "subtick_blocks": (3, 4), "sleep_prob": 0.0, "cpu_prob": 0.0},
    {"id": 11, "class": 1, "subtick_blocks": (2, 3), "sleep_prob": 0.05, "cpu_prob": 0.0},
    {"id": 12, "class": 1, "subtick_blocks": (3, 5), "sleep_prob": 0.0, "cpu_prob": 0.0},
    {"id": 13, "class": 1, "subtick_blocks": (2, 4), "sleep_prob": 0.05, "cpu_prob": 0.0},
    {"id": 14, "class": 1, "subtick_blocks": (3, 3), "sleep_prob": 0.0, "cpu_prob": 0.0},

    # CPU-bound (20-24): 100% CPU utilization (850-1000), zero sleep, zero block_rate, exhaust_ratio 333-666
    {"id": 20, "class": 2, "burst_range": (1, 2), "exhaust_prob": 0.33, "noise": 0.0},
    {"id": 21, "class": 2, "burst_range": (1, 2), "exhaust_prob": 0.40, "noise": 0.01},
    {"id": 22, "class": 2, "burst_range": (1, 3), "exhaust_prob": 0.50, "noise": 0.01},
    {"id": 23, "class": 2, "burst_range": (1, 2), "exhaust_prob": 0.33, "noise": 0.0},
    {"id": 24, "class": 2, "burst_range": (1, 2), "exhaust_prob": 0.45, "noise": 0.02},

    # Mixed (30-34): Moderate CPU (500-750), moderate sleep (250-500), moderate block_rate (300-1000)
    {"id": 30, "class": 3, "burst_range": (1, 2), "sleep_total_range": (1, 2), "exhaust_prob": 0.10, "yield_prob": 0.0},
    {"id": 31, "class": 3, "burst_range": (1, 2), "sleep_total_range": (1, 2), "exhaust_prob": 0.20, "yield_prob": 0.05},
    {"id": 32, "class": 3, "burst_range": (1, 2), "sleep_total_range": (1, 3), "exhaust_prob": 0.15, "yield_prob": 0.0},
    {"id": 33, "class": 3, "burst_range": (1, 3), "sleep_total_range": (1, 2), "exhaust_prob": 0.25, "yield_prob": 0.05},
    {"id": 34, "class": 3, "burst_range": (1, 2), "sleep_total_range": (1, 2), "exhaust_prob": 0.10, "yield_prob": 0.05},
]

def simulate_observation(bp):
    cls = bp["class"]
    burst_count = 3  # In xv6, observation finishes at OBS_MIN_BURSTS = 3
    if random.random() < 0.03:
        burst_count = random.choice([2, 4])

    if cls == 2:  # CPU-bound
        burst_history = []
        burst_exhausted = []
        for _ in range(burst_count):
            b = random.randint(bp["burst_range"][0], bp["burst_range"][1])
            ex = 1 if random.random() < bp["exhaust_prob"] else 0
            burst_history.append(b)
            burst_exhausted.append(ex)

        total_cpu = sum(burst_history)
        total_sleep = 0
        blocking_cnt = 0
        yield_cnt = 0
        window_ticks = max(1, total_cpu)
        if random.random() < bp["noise"]:
            total_sleep = 1
            window_ticks += 1

    elif cls == 1:  # I/O-bound
        burst_history = [1 if random.random() < bp["cpu_prob"] else 0 for _ in range(burst_count)]
        burst_exhausted = [0 for _ in range(burst_count)]
        total_cpu = sum(burst_history)
        total_sleep = 1 if random.random() < bp["sleep_prob"] else 0
        window_ticks = max(1, total_cpu + total_sleep)
        blocking_cnt = random.randint(bp["subtick_blocks"][0], bp["subtick_blocks"][1])
        yield_cnt = 0

    elif cls == 0:  # Interactive
        burst_history = [1 if random.random() < bp["cpu_prob"] else 0 for _ in range(burst_count)]
        burst_exhausted = [0 for _ in range(burst_count)]
        total_cpu = sum(burst_history)
        total_sleep = random.randint(bp["sleep_total_range"][0], bp["sleep_total_range"][1])
        window_ticks = max(1, total_cpu + total_sleep)
        blocking_cnt = burst_count
        yield_cnt = 1 if random.random() < bp["yield_prob"] else 0

    else:  # Mixed (3)
        burst_history = []
        burst_exhausted = []
        for _ in range(burst_count):
            b = random.randint(bp["burst_range"][0], bp["burst_range"][1])
            ex = 1 if random.random() < bp["exhaust_prob"] else 0
            burst_history.append(b)
            burst_exhausted.append(ex)
        total_cpu = sum(burst_history)
        total_sleep = random.randint(bp["sleep_total_range"][0], bp["sleep_total_range"][1])
        window_ticks = max(1, total_cpu + total_sleep)
        blocking_cnt = burst_count
        yield_cnt = 1 if random.random() < bp["yield_prob"] else 0

    n = max(1, len(burst_history))
    avg_burst = (sum(burst_history) * 1000) // n
    max_burst = max(burst_history) * 1000
    mad_sum = sum(abs((b * 1000) - avg_burst) for b in burst_history)
    burst_mad = mad_sum // n

    exhaust_ratio = (sum(burst_exhausted) * 1000) // n
    block_rate = (blocking_cnt * 1000) // window_ticks
    yield_rate = (yield_cnt * 1000) // window_ticks
    sleep_ratio = min(1000, (total_sleep * 1000) // window_ticks)
    cpu_util = min(1000, (total_cpu * 1000) // window_ticks)
    burst_count_feat = min(1000, (len(burst_history) * 1000) // OBS_MAX_BURSTS)

    return [
        avg_burst, max_burst, burst_mad, exhaust_ratio,
        block_rate, yield_rate, sleep_ratio, cpu_util, burst_count_feat
    ]

def main():
    os.makedirs("/home/sai/OS Project/ml", exist_ok=True)
    dataset_path = "/home/sai/OS Project/ml/dataset.csv"
    config_path = "/home/sai/OS Project/ml/generation_config.json"
    
    samples_per_bp = NUM_SAMPLES // len(BLUEPRINTS)
    rows = []
    
    for bp in BLUEPRINTS:
        for _ in range(samples_per_bp):
            feat = simulate_observation(bp)
            row = [bp["id"], bp["class"]] + feat
            rows.append(row)
            
    random.shuffle(rows)
    
    header = [
        "blueprint_id", "class_id",
        "avg_burst", "max_burst", "burst_mad", "exhaust_ratio",
        "block_rate", "yield_rate", "sleep_ratio", "cpu_util", "burst_count"
    ]
    
    with open(dataset_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
        
    config = {
        "seed": SEED,
        "num_samples": len(rows),
        "num_blueprints": len(BLUEPRINTS),
        "num_classes": 4,
        "classes": ["Interactive", "I/O-bound", "CPU-bound", "Mixed"],
        "num_features": 9,
        "features": header[2:]
    }
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        
    print(f"Generated {len(rows)} samples across {len(BLUEPRINTS)} blueprints into {dataset_path}")

if __name__ == "__main__":
    main()
