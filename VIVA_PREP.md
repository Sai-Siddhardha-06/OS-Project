# PM-MLFQ Viva Defense & Technical Interview Preparation Guide

This guide provides targeted answers to anticipated technical questions from examiners and faculty during the PM-MLFQ project defense.

---

## 1. High-Level Concepts & Motivation

### Q1: Why use Machine Learning inside an OS scheduler instead of traditional heuristics?
**Answer**:  
Traditional heuristics like classic MLFQ make rigid assumptions: all processes start at the top priority queue ($Q_0$) assuming they are interactive. Long compute-bound jobs initially contend with true interactive jobs, causing response time spikes. Static priority heuristics (like `nice`) require manual tuning. PM-MLFQ automates initial classification by observing early execution bursts and placing processes in appropriate queues from the start, minimizing queue demotion thrashing.

### Q2: Why Probabilistic Queue Assignment instead of Deterministic Argmax?
**Answer**:  
Deterministic ML assignment ($\arg\max_Q P(Q)$) creates a dangerous **herd effect**. If 10 compute-bound tasks start simultaneously and are all classified as Class 2 with 85% confidence, a deterministic scheduler dumps all 10 into Queue 2 at once. They contend heavily for Queue 2 while Queue 1 sits underutilized. Probabilistic assignment samples initial placement according to the confidence distribution, spreading the load across queues proportionally while still penalizing compute tasks, avoiding artificial convoys.

### Q3: What is the purpose of the observation window, and why place processes in Q1 provisionally?
**Answer**:  
A process's future behavior cannot be predicted before it executes. Running in an observation window of 3 bursts (or up to 50 ticks) allows the kernel to measure actual execution metrics (burst lengths, voluntary yields, blocking frequency). We place them provisionally in **Queue 1** (quantum = 2 ticks) rather than Queue 0 (quantum = 1 tick) to avoid polluting the highest priority queue with unknown jobs while still granting adequate responsiveness.

---

## 2. In-Kernel Machine Learning & Numerical Precision

### Q4: How can an ML model run in supervisor mode without hardware floating-point support?
**Answer**:  
Floating-point instructions are forbidden or undesirable in kernel supervisor mode because saving/restoring the 32 FPU registers across traps or context switches causes unacceptable overhead. We solved this by implementing the entire Multilayer Perceptron in **Q16.16 fixed-point arithmetic**:
- Integer representations where $1.0 = 65536$ ($2^{16}$).
- Multiplications accumulate into 64-bit integers (`int64`) to prevent overflow: $\text{acc} = b + (x \cdot W) \gg 16$.
- Activations use integer ReLU: $\max(0, x)$.
- Softmax is calculated using a 256-entry lookup table for the piecewise exponential $e^x$ over $[-8.0, 0.0]$, stabilized by subtracting $\max(z_k)$.
- Floating-point vs kernel fixed-point validation achieved a **100% bit-exact prediction match**.

### Q5: What is the overhead of running ML inference inside the kernel?
**Answer**:  
Inference occurs **exactly once per process lifetime** (at the close of its observation window). In `kernel/ml_scheduler.c`, the delta is recorded via `r_time()`, which reads the RISC-V hardware `time` CSR. In QEMU virt mode, this timer runs at **10 MHz** (100 ns per tick):
- **Mean timer delta**: **32.27 timer ticks**.
- **Real wall-clock latency**: **3.227 microseconds** (~32–64 RISC-V instructions).
- **Memory footprint**: 4.2 KB of static read-only weights and lookup tables in `.rodata`.
- **Dynamic memory**: **0 bytes**. Zero heap memory allocations (`kalloc`) are performed in the inference path.

---

## 3. Kernel Synchronization, Locking, and Safety

### Q6: How do you prevent deadlocks when updating process scheduling statistics?
**Answer**:  
We maintain a strict locking hierarchy:
1. Lock order: $\text{wait\_lock} \to p\text{->lock}$.
2. **Never acquire `tickslock` while holding `p->lock`**.
In `trap.c`, timer interrupts acquire `tickslock` to increment `ticks` and wake sleeping processes. If a function held `p->lock` and then attempted to acquire `tickslock`, an interrupt on another CPU or a reentrant call would cause an unrecoverable circular deadlock. In `get_ticks()`, we check `holding(&tickslock)`; if the CPU already holds it, we read `ticks` directly without re-acquiring.

### Q7: How do you prevent double-counting CPU ticks across context switches and timer interrupts?
**Answer**:  
Ticks are measured via state transition deltas. When a process is dispatched, we record `p->last_sched_tick = get_ticks()`. When it yields or blocks, $\Delta = \text{now} - p\text{->last\_sched\_tick}$ is added to `total_cpu_ticks` and `cur_burst_ticks`. Timer interrupts do not independently increment `p->total_cpu_ticks`, eliminating race conditions and double-counting under heavy load.

### Q8: How does PM-MLFQ prevent process starvation?
**Answer**:  
Two complementary mechanisms prevent starvation:
1. **Aging**: Every clock tick, the wait time of all `RUNNABLE` processes is incremented. If `p->waiting_ticks >= AGING_THRESHOLD` (10 ticks), the process is promoted to a higher queue ($\text{priority} - 1$).
2. **Periodic Priority Boost**: Every `BOOST_INTERVAL` (100 ticks), all processes across all queues are reset to Queue 0. This guarantees that no process remains trapped in lower-priority queues indefinitely.

---

## 4. Experimental Results & Ablation Analysis

### Q9: What do your experiments prove about the 5 scheduling modes?
**Answer**:  
Across 150 trials (5 modes $\times$ 6 workloads $\times$ 5 seeds):
- **Mode 0 (RR)**: High turnaround and wait times for mixed workloads due to lack of prioritization.
- **Mode 1 (MLFQ)**: Strong performance for short jobs, but degrades under heavy concurrency due to all jobs starting in Q0.
- **Mode 2 (ML-PROB)**: Outperforms deterministic ML by 8.4% in turnaround time under high concurrency (Exp 5) by preventing queue convoying.
- **Mode 3 (ML-DET)**: Slower under high concurrency (25.68 vs 23.53 ticks) because of herd effects in Queue 2.
- **Mode 4 (RAND-CTRL)**: Pure random initial assignment performs worse than ML-PROB (24.40 vs 23.53 ticks), proving that the performance gain comes from the ML classifier rather than random dispersion.

### Q10: How was system stability verified?
**Answer**:  
We ran the complete `usertests -q` test suite under Mode 2 (`ML-PROB`). All tests passed cleanly without kernel panics, memory leaks, page faults, or deadlocks. In addition, `ml_self_test()` runs at boot time and validates 24 test vectors with 100% bit-exact accuracy before `userinit()` is called.
