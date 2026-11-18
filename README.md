# PM-MLFQ: Probabilistic Machine Learning-Assisted MLFQ CPU Scheduling

PM-MLFQ is an enhanced CPU scheduler implemented in the 64-bit RISC-V MIT xv6 operating system. It observes process execution characteristics during early CPU bursts, evaluates a fixed-point multilayer perceptron (MLP) entirely in supervisor mode without hardware floating-point registers, and assigns processes to initial MLFQ priority queues probabilistically to avoid herd-effect convoying.

---

## 1. Quick Start & Prerequisites

### Prerequisites
- **Toolchain**: `riscv64-unknown-elf-gcc` (or `riscv64-linux-gnu-gcc`), `qemu-system-riscv64` (v8.0+)
- **Host Tools**: `python3` (3.10+), `make`

### Building & Running
All kernel builds and QEMU executions are controlled from `xv6-riscv/`:

```bash
cd xv6-riscv

# Build Mode 0 (Round-Robin Baseline)
make clean && make SCHED_MODE=0 CPUS=1

# Build Mode 1 (Standard 3-level MLFQ)
make clean && make SCHED_MODE=1 CPUS=1

# Build Mode 2 (PM-MLFQ: Probabilistic ML-assisted MLFQ)
make clean && make SCHED_MODE=2 CPUS=1

# Build Mode 3 (ML-DET: Deterministic ML Ablation)
make clean && make SCHED_MODE=3 CPUS=1

# Build Mode 4 (RAND-CTRL: Prior Random Marginal Ablation)
make clean && make SCHED_MODE=4 CPUS=1
```

To run QEMU interactively:
```bash
make qemu CPUS=1
```

Inside the xv6 shell:
```bash
$ schedmon             # Display live process table and scheduling stats
$ schedtest 1 101      # Run experiment 1 with seed 101
$ usertests -q         # Run standard xv6 validation suite
```

---

## 2. Architecture & Scheduling Modes

| Mode | Macro | Description |
|---|---|---|
| **Mode 0** | `SCHED_RR` | MIT xv6 default round-robin scheduler. |
| **Mode 1** | `SCHED_MLFQ` | Deterministic 3-level feedback queue ($Q_0=1, Q_1=2, Q_2=4$ ticks). All enter $Q_0$. |
| **Mode 2** | `SCHED_ML_PROB` | PM-MLFQ. Provisional $Q_1 \to$ in-kernel MLP inference $\to$ sample from $P(Q)$. |
| **Mode 3** | `SCHED_ML_DET` | Deterministic ML ablation. Provisional $Q_1 \to$ MLP $\to \arg\max_Q P(Q)$. |
| **Mode 4** | `SCHED_RAND_CTRL` | Random control ablation. Provisional $Q_1 \to$ sample from marginal $[333, 334, 333]$. |

---

## 3. Machine Learning Subsystem (`ml/`)

1. **Synthetic Data Generation**:
   ```bash
   python3 ml/generate_data.py
   ```
   Generates `ml/dataset.csv` (6,000 samples across 20 distinct workload blueprints).

2. **Model Training**:
   ```bash
   python3 ml/train_model.py
   ```
   Trains the $9 \to 16 \to 8 \to 4$ neural network with blueprint-level group holdout and evaluates against `ml/real_traces.csv`.

3. **Weight Export**:
   ```bash
   python3 ml/export_weights.py
   ```
   Exports Q16.16 fixed-point parameters and 24 self-test vectors into `xv6-riscv/kernel/model_weights.h`.

---

## 4. Reproducing Experiments (`scripts/` & `results/`)

To reproduce the complete 150-trial matrix (5 modes $\times$ 6 workloads $\times$ 5 seeds):

```bash
python3 scripts/run_experiments.py
python3 ml/analyze_results.py
```

Outputs:
- `results/raw/`: Raw QEMU console outputs with `SCHEDLOG` lines.
- `results/experiment_results.csv`: 725 parsed process execution records.
- `results/summary_metrics.csv`: Aggregated means, standard deviations, and 95% confidence intervals.
- `results/*.png`: Comparative visualization charts with error bars.

---

## 5. Documentation Directory (`docs/`)

- [project_report.md](file:///home/sai/OS%20Project/docs/project_report.md): Complete academic report with theoretical formulation, benchmark tables, and threats to validity.
- [architecture.md](file:///home/sai/OS%20Project/docs/architecture.md): Kernel subsystem diagrams, queue transitions, and data structures.
- [testing.md](file:///home/sai/OS%20Project/docs/testing.md): Gate verification log (`usertests -q`, boot self-tests).
- [AUDIT.md](file:///home/sai/OS%20Project/docs/AUDIT.md): Comprehensive itemized verification audit log.
- [DESIGN_NOTES.md](file:///home/sai/OS%20Project/DESIGN_NOTES.md): Locking hierarchy and fixed-point math rationale.
- [VIVA_PREP.md](file:///home/sai/OS%20Project/VIVA_PREP.md): Oral defense and technical interview guide.


---

## 6. Optional Diagnostic & Safety Features

### Scheduler Event Trace Flight Recorder (`SCHED_TRACE`)
Build with the scheduler flight recorder enabled to trace kernel events into a 128-entry in-memory ring buffer:
```bash
cd xv6-riscv
make clean
make SCHED_MODE=2 SCHED_TRACE=1 CPUS=1 -j4
make fs.img SCHED_MODE=2 SCHED_TRACE=1 CPUS=1
make qemu CPUS=1
```
Inside the xv6 shell:
```bash
$ schedtest 1 101
$ schedtrace
```
Output displays chronological `PROC_NEW`, `CLASSIFIED`, `DEMOTE`, `PROMOTE_AGE`, `BOOST`, and `PROC_EXIT` events.

### Per-Process Memory Cap (`PROC_MEM_LIMIT`)
Enforce a hard byte limit on process heap allocations (`sbrk()`):
```bash
cd xv6-riscv
make clean
make SCHED_MODE=2 PROC_MEM_LIMIT=4194304 CPUS=1 -j4
make fs.img SCHED_MODE=2 PROC_MEM_LIMIT=4194304 CPUS=1
make qemu CPUS=1
```
Inside the xv6 shell:
```bash
$ memhog
```
`memhog` grows by 4 KB pages until reaching the 4 MB limit, whereupon subsequent allocations fail immediately with `-1`.

> **Host Memory Tip (Linux/systemd)**: To restrict host RAM consumption during QEMU runs, execute:
> ```bash
> systemd-run --user --scope -p MemoryMax=2G make qemu
> ```
