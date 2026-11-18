# PM-MLFQ System Architecture & Subsystem Guide

## 1. System Overview

PM-MLFQ (Probabilistic ML-assisted Multilevel Feedback Queue) is an enhanced CPU scheduler implemented within the RISC-V 64-bit `xv6` operating system. Traditional MLFQ schedulers assume all new processes are short/interactive and insert them into the highest priority queue (Queue 0), demoting them only after they exhaust their timeslices. 

PM-MLFQ introduces an observation phase where execution features are captured dynamically during early bursts, evaluated by an in-kernel fixed-point Multilayer Perceptron (MLP), and used to sample an optimal initial queue from a probabilistic class-to-queue distribution.

```
+-----------------------------------------------------------------------------------+
|                               xv6 Kernel Space                                    |
|                                                                                   |
|  +-------------------+       +---------------------+       +-------------------+  |
|  |  Process Creation | ----> |  Observation Window | ----> | Fixed-Point MLP   |  |
|  |  (allocproc/exec) |       |  (3-10 bursts /     |       | (9 -> 16 -> 8 -> 4|  |
|  +-------------------+       |   50 ticks max)     |       |  Q16.16 inference)|  |
|                              +---------------------+       +-------------------+  |
|                                                                      |            |
|                                                                      v            |
|                              +-------------------------------------------------+  |
|                              | Initial Queue Assignment                        |  |
|                              | Mode 0: Round Robin                             |  |
|                              | Mode 1: Standard MLFQ (starts Q0)               |  |
|                              | Mode 2: ML-PROB (Sample from P(Q))              |  |
|                              | Mode 3: ML-DET (Argmax P(Q))                    |  |
|                              | Mode 4: RAND-CTRL (Sample from prior marginal)  |  |
|                              +-------------------------------------------------+  |
|                                                      |                            |
|                                                      v                            |
|  +-----------------------------------------------------------------------------+  |
|  |                           3-Level MLFQ Dispatcher                           |  |
|  |  [Queue 0] Quantum = 1 tick  (High Priority / Short Bursts)                 |  |
|  |  [Queue 1] Quantum = 2 ticks (Medium Priority / Provisional)                |  |
|  |  [Queue 2] Quantum = 4 ticks (Low Priority / CPU-bound)                     |  |
|  |                                                                             |  |
|  |  * Starvation Safeguards: Aging (10 ticks wait) + Boost (every 100 ticks)   |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Kernel Subsystems

### 2.1 Scheduler Dispatcher (`kernel/proc.c`)
- Implements `scheduler()` loop executing on each hart.
- Multi-queue selection:
  - Scans Queue 0 first, then Queue 1, then Queue 2.
  - Maintains rotating scan pointer `last_idx[q]` to prevent starvation among equal-priority threads.
  - Enforces per-queue quanta: Q0 (1 tick), Q1 (2 ticks), Q2 (4 ticks).
  - Handles quantum demotion on timer expiration and priority preservation on voluntary I/O yield.

### 2.2 Feature Extractor & Accounting (`kernel/ml_scheduler.c`, `kernel/proc.c`)
- Extracts 9 integer features ($0..1000$ fixed-point scale):
  1. `avg_burst`: Mean CPU burst duration.
  2. `max_burst`: Maximum single CPU burst duration.
  3. `burst_mad`: Mean absolute deviation of bursts.
  4. `exhaust_ratio`: Fraction of bursts ending via quantum expiry.
  5. `block_rate`: Voluntary I/O block events per observed tick.
  6. `yield_rate`: Voluntary yield events per observed tick.
  7. `sleep_ratio`: Fraction of observation window spent blocked.
  8. `cpu_util`: Fraction of observation window running on CPU.
  9. `burst_count`: Scaled count of completed bursts.

### 2.3 Fixed-Point ML Inference Engine (`kernel/ml_scheduler.c`)
- Zero floating-point dependency.
- Layer pipeline:
  - Normalization: $z_i = (x_i - \mu_i) / \sigma_i$ in Q16.16.
  - Dense 1: 9 inputs $\to$ 16 hidden units + ReLU.
  - Dense 2: 16 units $\to$ 8 hidden units + ReLU.
  - Output: 8 units $\to$ 4 logits + exp-table softmax.
- Softmax outputs permille integer probabilities ($0..1000$) summing strictly to 1000.
- Cycle profiling: Instrumented via RISC-V `csrr time` instruction (`r_time()`), logging cycles directly into `p->infer_cycles`.

### 2.4 Diagnostic & Telemetry Framework
- System call `getschedstats(struct schedstat *buf, int max)` (SYS 24):
  - Captures atomic snapshot of active process queue levels, dispatches, transitions, CPU ticks, and predictions.
- User utility `schedmon`:
  - Formats snapshot into a real-time terminal dashboard.
- Telemetry logging:
  - Emits `SCHEDLOG` on process exit containing PID, arrival, first run, exit tick, CPU ticks, wait ticks, initial queue, predicted class, confidence, and inference cycles.
