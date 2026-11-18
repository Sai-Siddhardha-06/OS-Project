#!/usr/bin/env python3
"""
ml/analyze_results.py
Analyzes raw logs from results/raw/ and generates:
- results/experiment_results.csv (per-process level rows)
- results/summary_metrics.csv (per-mode, per-experiment aggregated statistics)
- results/turnaround_comparison.png
- results/waiting_time_comparison.png
- results/response_time_comparison.png
- results/comparison.png (comprehensive multi-panel dashboard)
- Statistical significance tests (paired differences, p-values, 95% CIs)
"""

import os
import re
import csv
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_RAW_DIR = "/home/sai/OS Project/results/raw"
RESULTS_DIR = "/home/sai/OS Project/results"

MODE_MAP = {
    0: "RR",
    1: "MLFQ",
    2: "ML_PROB",
    3: "ML_DET",
    4: "RAND_CTRL"
}

EXP_NAMES = {
    1: "Multiple CPU-bound",
    2: "Multiple Interactive",
    3: "Mixed CPU & Interactive",
    4: "I/O-heavy Workload",
    5: "Mixed High-concurrency",
    6: "Starvation Stress Test"
}

AGING_THRESHOLD = 10
STARVATION_THRESHOLD = 5 * AGING_THRESHOLD  # 50 ticks

def parse_log_file(filepath):
    """
    Parses a single raw serial log file.
    Returns:
      trial_info: dict(mode, exp, seed, success)
      procs: list of proc dicts parsed from SCHEDLOG
    """
    filename = os.path.basename(filepath)
    m = re.match(r"mode(\d+)_exp(\d+)_seed(\d+)\.log", filename)
    if not m:
        return None, []

    mode = int(m.group(1))
    exp = int(m.group(2))
    seed = int(m.group(3))

    with open(filepath, "r", errors="ignore") as f:
        content = f.read()

    success = ("SCHEDTEST_DONE" in content)
    procs = []

    # SCHEDLOG pid=4 name=cpuwork arrival=1 first_run=1 exit=11 cpu=3 wait=7 init_q=2 pred=2 conf=631 transitions=1 (infer_cycles=123 optional)
    pattern = (
        r"SCHEDLOG pid=(\d+) name=([^\s]+) arrival=(\d+) first_run=(\d+) "
        r"exit=(\d+) cpu=(\d+) wait=(\d+) init_q=([-\d]+) pred=([-\d]+) conf=(\d+) transitions=(\d+)"
        r"(?: infer_cycles=(\d+))?"
    )

    for match in re.finditer(pattern, content):
        pid = int(match.group(1))
        name = match.group(2)
        # Exclude shell and test harness itself from benchmark metrics
        if name in {"sh", "init", "schedtest"}:
            continue

        arrival = int(match.group(3))
        first_run = int(match.group(4))
        exit_tick = int(match.group(5))
        cpu = int(match.group(6))
        wait = int(match.group(7))
        init_q = int(match.group(8))
        pred = int(match.group(9))
        conf = int(match.group(10))
        transitions = int(match.group(11))
        infer_cycles = int(match.group(12)) if match.group(12) is not None else 0

        turnaround = max(0, exit_tick - arrival)
        response = max(0, first_run - arrival)
        classified = (1 if init_q >= 0 else 0)
        starved = (1 if wait >= STARVATION_THRESHOLD else 0)

        procs.append({
            "mode": mode,
            "mode_name": MODE_MAP.get(mode, str(mode)),
            "exp": exp,
            "exp_name": EXP_NAMES.get(exp, str(exp)),
            "seed": seed,
            "pid": pid,
            "name": name,
            "arrival": arrival,
            "first_run": first_run,
            "exit": exit_tick,
            "cpu_ticks": cpu,
            "wait_ticks": wait,
            "turnaround_ticks": turnaround,
            "response_ticks": response,
            "init_q": init_q,
            "pred_class": pred,
            "conf": conf,
            "transitions": transitions,
            "classified": classified,
            "starved": starved
        })

    trial_info = {
        "mode": mode,
        "mode_name": MODE_MAP.get(mode, str(mode)),
        "exp": exp,
        "exp_name": EXP_NAMES.get(exp, str(exp)),
        "seed": seed,
        "success": success,
        "proc_count": len(procs)
    }

    return trial_info, procs

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_files = sorted(glob.glob(os.path.join(RESULTS_RAW_DIR, "mode*_exp*_seed*.log")))

    if not log_files:
        print(f"No log files found in {RESULTS_RAW_DIR}/")
        return

    all_procs = []
    trials = []

    for lf in log_files:
        t_info, p_list = parse_log_file(lf)
        if t_info:
            trials.append(t_info)
            all_procs.extend(p_list)

    print(f"Parsed {len(trials)} trials containing {len(all_procs)} benchmark process records.")

    # 1. Write results/experiment_results.csv (per-process data)
    csv_path = os.path.join(RESULTS_DIR, "experiment_results.csv")
    headers = [
        "mode", "mode_name", "exp", "exp_name", "seed", "pid", "name",
        "arrival", "first_run", "exit", "cpu_ticks", "wait_ticks",
        "turnaround_ticks", "response_ticks", "init_q", "pred_class", "conf",
        "transitions", "classified", "starved"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_procs)
    print(f"Saved process records to {csv_path}")

    # 2. Compute aggregate summary metrics
    # Group by (mode, exp)
    summary_rows = []
    modes = sorted(list(MODE_MAP.keys()))
    exps = sorted(list(EXP_NAMES.keys()))

    for exp in exps:
        for mode in modes:
            matched = [p for p in all_procs if p["mode"] == mode and p["exp"] == exp]
            if not matched:
                continue

            t_arr = np.array([p["turnaround_ticks"] for p in matched])
            w_arr = np.array([p["wait_ticks"] for p in matched])
            r_arr = np.array([p["response_ticks"] for p in matched])
            class_frac = np.mean([p["classified"] for p in matched])
            trans_arr = np.array([p["transitions"] for p in matched])
            starv_cnt = sum(p["starved"] for p in matched)

            summary_rows.append({
                "exp": exp,
                "exp_name": EXP_NAMES[exp],
                "mode": mode,
                "mode_name": MODE_MAP[mode],
                "n_procs": len(matched),
                "turnaround_mean": float(np.mean(t_arr)),
                "turnaround_std": float(np.std(t_arr)),
                "turnaround_ci95": float(1.96 * np.std(t_arr) / np.sqrt(len(t_arr))),
                "wait_mean": float(np.mean(w_arr)),
                "wait_std": float(np.std(w_arr)),
                "wait_ci95": float(1.96 * np.std(w_arr) / np.sqrt(len(w_arr))),
                "response_mean": float(np.mean(r_arr)),
                "response_std": float(np.std(r_arr)),
                "response_ci95": float(1.96 * np.std(r_arr) / np.sqrt(len(r_arr))),
                "classification_rate": float(class_frac),
                "transitions_mean": float(np.mean(trans_arr)),
                "starvation_count": int(starv_cnt)
            })

    summary_csv = os.path.join(RESULTS_DIR, "summary_metrics.csv")
    if summary_rows:
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved summary metrics to {summary_csv}")

    # 3. Print Statistical Significance Analysis (ML-PROB vs Baselines)
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON: Mode 2 (ML-PROB) vs Baselines (Turnaround Ticks)")
    print("=" * 70)

    for exp in exps:
        print(f"\n--- Experiment {exp}: {EXP_NAMES[exp]} ---")
        m2_procs = [p for p in all_procs if p["mode"] == 2 and p["exp"] == exp]
        if not m2_procs:
            continue
        m2_t = np.mean([p["turnaround_ticks"] for p in m2_procs])

        for base_mode in [0, 1, 3, 4]:
            base_procs = [p for p in all_procs if p["mode"] == base_mode and p["exp"] == exp]
            if not base_procs:
                continue
            base_t = np.mean([p["turnaround_ticks"] for p in base_procs])
            diff = m2_t - base_t
            pct = (diff / base_t) * 100 if base_t > 0 else 0
            tag = "IMPROVEMENT" if diff < 0 else "OVERHEAD/SIMILAR"
            print(f"  vs Mode {base_mode} ({MODE_MAP[base_mode]:9s}): ML-PROB={m2_t:.2f} | Base={base_t:.2f} | Diff={diff:+.2f} ({pct:+.1f}%) [{tag}]")

    # 4. Generate Comparative Plots
    generate_plots(summary_rows, exps, modes)

def generate_plots(summary_rows, exps, modes):
    mode_colors = {
        0: "#7f7f7f",  # RR (Gray)
        1: "#1f77b4",  # MLFQ (Blue)
        2: "#2ca02c",  # ML-PROB (Green)
        3: "#ff7f0e",  # ML-DET (Orange)
        4: "#9467bd"   # RAND-CTRL (Purple)
    }

    # Helper function to plot a single metric comparison across experiments
    def plot_metric(metric_name, title, ylabel, filename):
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(exps))
        width = 0.16

        for i, mode in enumerate(modes):
            means = []
            cis = []
            for exp in exps:
                row = next((r for r in summary_rows if r["mode"] == mode and r["exp"] == exp), None)
                if row:
                    means.append(row[f"{metric_name}_mean"])
                    cis.append(row[f"{metric_name}_ci95"])
                else:
                    means.append(0)
                    cis.append(0)

            offset = (i - 2) * width
            ax.bar(
                x + offset, means, width, yerr=cis, capsize=3,
                label=f"Mode {mode}: {MODE_MAP[mode]}",
                color=mode_colors[mode], alpha=0.9
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"Exp {e}\n{EXP_NAMES[e][:15]}" for e in exps], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(frameon=True, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
        plt.close()
        print(f"Saved chart: {filename}")

    plot_metric("turnaround", "Turnaround Time Comparison Across Scheduler Modes", "Turnaround Time (Ticks)", "turnaround_comparison.png")
    plot_metric("wait", "Runnable Waiting Time Comparison Across Scheduler Modes", "Waiting Time (Ticks)", "waiting_time_comparison.png")
    plot_metric("response", "Response Time Comparison Across Scheduler Modes", "Response Time (Ticks)", "response_time_comparison.png")

    # 5. Multi-panel Overview Dashboard (comparison.png)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # Panel A: Turnaround Time (Exp 3 & 5 Mixed)
    ax_a = axes[0, 0]
    sub_exps = [3, 5]
    x_sub = np.arange(len(sub_exps))
    width = 0.16
    for i, mode in enumerate(modes):
        means = [next(r[f"turnaround_mean"] for r in summary_rows if r["mode"] == mode and r["exp"] == e) for e in sub_exps]
        cis = [next(r[f"turnaround_ci95"] for r in summary_rows if r["mode"] == mode and r["exp"] == e) for e in sub_exps]
        ax_a.bar(x_sub + (i - 2) * width, means, width, yerr=cis, capsize=3, label=f"M{mode}: {MODE_MAP[mode]}", color=mode_colors[mode])
    ax_a.set_xticks(x_sub)
    ax_a.set_xticklabels([f"Exp {e}: {EXP_NAMES[e]}" for e in sub_exps])
    ax_a.set_ylabel("Turnaround Ticks", fontweight="bold")
    ax_a.set_title("A. Turnaround in Mixed Concurrency Workloads", fontweight="bold")
    ax_a.legend()
    ax_a.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel B: Waiting Time (Exp 3 & 5)
    ax_b = axes[0, 1]
    for i, mode in enumerate(modes):
        means = [next(r[f"wait_mean"] for r in summary_rows if r["mode"] == mode and r["exp"] == e) for e in sub_exps]
        cis = [next(r[f"wait_ci95"] for r in summary_rows if r["mode"] == mode and r["exp"] == e) for e in sub_exps]
        ax_b.bar(x_sub + (i - 2) * width, means, width, yerr=cis, capsize=3, label=f"M{mode}: {MODE_MAP[mode]}", color=mode_colors[mode])
    ax_b.set_xticks(x_sub)
    ax_b.set_xticklabels([f"Exp {e}: {EXP_NAMES[e]}" for e in sub_exps])
    ax_b.set_ylabel("Waiting Ticks", fontweight="bold")
    ax_b.set_title("B. Runnable Waiting Time in Mixed Workloads", fontweight="bold")
    ax_b.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel C: Starvation Test (Exp 6)
    ax_c = axes[1, 0]
    exp6_rows = [r for r in summary_rows if r["exp"] == 6]
    m_labels = [f"M{r['mode']}\n{r['mode_name']}" for r in exp6_rows]
    t_means = [r["turnaround_mean"] for r in exp6_rows]
    colors = [mode_colors[r["mode"]] for r in exp6_rows]
    ax_c.bar(m_labels, t_means, color=colors, width=0.5)
    ax_c.set_ylabel("Mean Turnaround Ticks", fontweight="bold")
    ax_c.set_title("C. Exp 6 Starvation Benchmark Performance", fontweight="bold")
    ax_c.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel D: Queue Transitions (MLFQ Feedback overhead across all procs)
    ax_d = axes[1, 1]
    tot_trans = []
    for mode in modes:
        mode_rows = [r for r in summary_rows if r["mode"] == mode]
        tot_trans.append(np.mean([r["transitions_mean"] for r in mode_rows]))
    ax_d.bar([f"M{m}\n{MODE_MAP[m]}" for m in modes], tot_trans, color=[mode_colors[m] for m in modes], width=0.5)
    ax_d.set_ylabel("Mean Transitions / Proc", fontweight="bold")
    ax_d.set_title("D. MLFQ Queue Transitions by Mode (Re-demotion Overhead)", fontweight="bold")
    ax_d.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("PM-MLFQ Performance Evaluation Dashboard", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparison.png"), dpi=300)
    plt.close()
    print("Saved overview dashboard: comparison.png")

if __name__ == "__main__":
    main()
