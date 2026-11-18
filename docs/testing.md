# Phase 4 Testing & Gate Verification Log

## 1. Usertests Verification Summary

| Mode | Scheduler Name | Build Status | Test Command | Outcome | Log Path |
|---|---|---|---|---|---|
| Mode 0 | RR (Baseline) | PASS | `usertests -q` | **ALL TESTS PASSED** | `results/raw/baseline_usertests.log` |
| Mode 1 | MLFQ | PASS | `usertests -q` | **ALL TESTS PASSED** | Verified in Phase 1 Gate |
| Mode 2 | ML-PROB | PASS | `usertests -q` | **ALL TESTS PASSED** | `results/raw/mode2_usertests.log` |
| Mode 3 | ML-DET | PASS | Smoke / schedtest | **PASS (0 panics)** | Verified across 30 experiment trials |
| Mode 4 | RAND-CTRL | PASS | Smoke / schedtest | **PASS (0 panics)** | Verified across 30 experiment trials |

### Mode 2 Usertests Result Details
- Command: `usertests -q` under QEMU RISC-V 64-bit virt board (`-smp 1`, 128 MB RAM).
- Total tests executed: Full `usertests -q` suite (fork, sbrk, exec, pipe, file system tests, lazy allocation, mem stress).
- Result: `ALL TESTS PASSED`.
- Zero kernel panics, page faults, or deadlocks detected.
- Real-time logging: `SCHEDLOG` emitted on all child process terminations with valid CPU/wait accounting and inference cycle tracking (`infer_cycles`).

---

## 2. Boot-Time Self-Test Verification (Python-to-Kernel Fixed-Point Match)

At system boot, `main.c` invokes `ml_self_test()` before launching `userinit()`. If any test vector produces a discrepancy with the Python reference model, the kernel panics immediately.

- **Status**: PASSED (`ml_self_test: all 24 vectors passed bit-level check`).
- **Vectors Verified**: 24 distinct feature combinations spanning Interactive, I/O-bound, CPU-bound, and Mixed classes.
- **Precision**: 100% bit-level identical match on class logits, softmax permille probabilities, and predicted class labels.
- Saved Reference Artifact: `results/kernel_python_validation.json`.

---

## 3. Workload Suite Gate Verification

All six experimental workloads run deterministically using `schedtest <exp_id> <seed>`:
1. `schedtest 1 <seed>`: 4 CPU-bound processes (`cpuwork`).
2. `schedtest 2 <seed>`: 4 Interactive processes (`interactive`).
3. `schedtest 3 <seed>`: 2 CPU-bound + 2 Interactive processes.
4. `schedtest 4 <seed>`: 4 I/O-heavy processes (`iowork`).
5. `schedtest 5 <seed>`: 8 Mixed high-concurrency processes (2 CPU, 2 IO, 2 Interactive, 2 Mixed).
6. `schedtest 6 <seed>`: 1 long CPU hog + 4 staggered Interactive processes (starvation stress test).

### Multi-Trial Experiment Matrix Execution
- Total trials planned: 5 Modes × 6 Experiments × 5 Seeds (101-105) = **150 trials**.
- Total trials successfully completed: **150 / 150** (100% completion rate).
- Total bench process records captured: **725 records**.
- Zero timeouts or corrupt logs encountered.
