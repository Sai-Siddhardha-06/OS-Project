# PM-MLFQ: Probabilistic Machine Learning-Assisted Multilevel Feedback Queue CPU Scheduling in xv6

**Author**: Moguluri Sai Siddhardha, B24DS018, B.Tech DSAI, IIT Bhilai
**Target Platform**: RISC-V 64-bit (`virt`), QEMU System Emulator
**Base Kernel**: MIT xv6-riscv
**Repository**: `https://github.com/Sai-Siddhardha-06/OS-Project`
**Date**: November 18th, 2026

---

## Abstract

A conventional Multilevel Feedback Queue (MLFQ) scheduler assumes that every new process is short and interactive, and places it in the highest-priority queue ($Q_0$). Long-running compute-bound processes therefore contend with interactive ones until the feedback mechanism demotes them. This work investigates whether a trained classifier can improve on this by choosing a process's *initial* queue.

We implement **PM-MLFQ** (Probabilistic Machine Learning-Assisted MLFQ) inside the 64-bit RISC-V `xv6` kernel. A new process runs provisionally in $Q_1$ while the kernel observes its first CPU bursts. It then extracts a 9-dimensional integer feature vector, evaluates an integer-only Multilayer Perceptron (MLP) in supervisor mode with no floating-point operations, maps the predicted class probabilities to a queue distribution, and samples the initial queue from that distribution. Ordinary MLFQ feedback, aging and periodic priority boosting then continue to manage the process.

The research question is: *does sampling the initial queue from classifier-derived probabilities improve turnaround time compared with (a) conventional MLFQ, (b) deterministic ML assignment, and (c) random assignment that ignores the classifier?* We evaluate five scheduler modes (RR, MLFQ, ML-PROB, ML-DET, RAND-CTRL) on six workloads with five seeds each (150 trials in the xv6 kernel under QEMU).

**Results.** The outcome is largely negative, and we report it as such:

1. **PM-MLFQ did not beat conventional MLFQ in turnaround time in any experiment.** MLFQ was faster or tied in every workload (e.g. 9.20 vs 9.90 ticks in Exp 1, where the difference is statistically distinguishable from zero; 23.10 vs 23.53 ticks in Exp 5, where it is not).
2. **Experiments 2-4 did not discriminate between scheduling modes.** Their mean turnaround times were nearly identical across all modes, because the workloads are dominated by sleep durations and tick-granularity timing.
3. **The classifier did not show a statistically distinguishable benefit over the random control.** In Exp 5 the paired turnaround difference ML-PROB minus RAND-CTRL was $-0.875$ ticks (95% CI $[-1.899, +0.149]$, spanning zero). In Exp 1, RAND-CTRL was faster than ML-PROB ($+0.600$ ticks, 95% CI $[+0.184, +1.016]$).
4. In Exp 5, the sample mean turnaround of ML-PROB was lower than that of ML-DET ($-2.150$ ticks), but the 95% CI ($[-4.504, +0.204]$) spans zero with $n=5$, so this is a hypothesis for future work and not a confirmed effect.
5. **Feasibility.** Fixed-point MLP inference runs entirely in supervisor mode. Its measured cost was 32.27 time-CSR units on average (about 3.2 µs on QEMU's virtual timebase), and a 24-vector boot-time self-test matches the Python fixed-point reference bit for bit.

All scheduling experiments were run with a single CPU (`CPUS=1`). Multi-CPU correctness was not established (see Section 5).

---

## 1. Introduction and Motivation

MLFQ, rooted in the Compatible Time-Sharing System (Corbató et al., 1962), approximates shortest-job-first behavior without knowing job lengths, by moving processes between priority levels according to their observed CPU use.

Conventional MLFQ has potential weaknesses that motivated this study:

1. **Optimistic $Q_0$ placement.** Every new process enters $Q_0$, so compute-heavy processes may briefly compete with genuinely interactive ones until demoted. (We hypothesize this costs interactive latency; the present workloads were too coarse to measure it, see Section 4.)
2. **Possible clustering under deterministic prediction.** If a classifier maps each process directly to its most likely queue, processes with similar early behavior all land in the same queue. We hypothesized that sampling spreads them out and might help under load. Section 4.2 examines the queue assignments and Section 4.1 the effect on turnaround.
3. **No floating point in the kernel.** Supervisor-mode code in xv6 does not use hardware floating point, so a neural network must be implemented in integer arithmetic.

The implementation contributes:

- **A provisional observation phase.** New processes run in $Q_1$ for up to `OBS_MIN_BURSTS` (3) completed bursts, bounded by `OBS_MAX_WINDOW_TICKS` (50), so observation always terminates.
- **A kernel-resident fixed-point inference engine.** A $9 \to 16 \to 8 \to 4$ MLP in Q16.16 arithmetic with a 256-entry exponential lookup table for a numerically stable softmax.
- **Probabilistic initial queue assignment** using a per-process Xorshift32 PRNG, plus two ablations (deterministic argmax and a classifier-free random control) that let us separate the effect of randomness from the effect of the classifier.

---

## 2. Architecture

### 2.1 System Model

The scheduler manages three priority levels:

- **Queue 0 ($Q_0$)**: highest priority, timeslice $\tau_0 = 1$ tick.
- **Queue 1 ($Q_1$)**: medium priority, timeslice $\tau_1 = 2$ ticks; also the provisional observation queue.
- **Queue 2 ($Q_2$)**: lowest priority, timeslice $\tau_2 = 4$ ticks.

A runnable process in $Q_i$ is always dispatched before any process in $Q_{i+1}$; within a queue, dispatch is round-robin using rotating scan pointers. A process that exhausts its quantum is demoted one level ($Q_2$ stays $Q_2$). A process that blocks or yields voluntarily before exhausting its quantum keeps its priority, and wakeup preserves priority. Starvation is prevented by aging (a runnable process that waits 10 ticks is promoted one level; sleeping time is not counted) and by a periodic boost every 100 ticks. All thresholds are compile-time configurable (`kernel/sched_config.h`). The scheduler is selected at compile time with `SCHED_MODE`, so there is one shared code base.

### 2.2 Feature Representation

When a process finishes `OBS_MIN_BURSTS` bursts or reaches `OBS_MAX_WINDOW_TICKS`, the kernel extracts a 9-dimensional integer feature vector (scaled by $10^3$ where fractional values matter):

1. `avg_burst`: average CPU burst duration (ticks $\times 10^3$)
2. `max_burst`: maximum burst duration (ticks $\times 10^3$)
3. `burst_mad`: mean absolute deviation of burst durations
4. `exhaust_ratio`: fraction of bursts ended by quantum exhaustion, in $[0, 1000]$
5. `block_rate`: voluntary blocking events per observed time, in $[0, 1000]$
6. `yield_rate`: voluntary yields per observed time, in $[0, 1000]$
7. `sleep_ratio`: blocked ticks over lifetime ticks, in $[0, 1000]$
8. `cpu_util`: CPU ticks over elapsed ticks, in $[0, 1000]$
9. `burst_count`: bursts observed in the window

The synthetic data generator computes these features with the same integer formulas and tick quantization as the kernel.

### 2.3 Neural Network and Fixed-Point Formulation

The classifier predicts four workload classes: $C_0$ Interactive, $C_1$ I/O-bound, $C_2$ CPU-bound, $C_3$ Mixed. The network is a feed-forward MLP ($9 \to 16 \to 8 \to 4$) with weights and biases in Q16.16 format ($1.0 \equiv 65536$) and 64-bit signed accumulation:

$$z_j^{(l)} = b_j^{(l)} + \sum_{i=1}^{M_{l-1}} \left\lfloor \frac{a_i^{(l-1)} \cdot W_{ij}^{(l)}}{65536} \right\rfloor$$

Hidden layers use ReLU, $a_j^{(l)} = \max(0, z_j^{(l)})$. For the output layer, $\Delta_j = z_j - \max_k z_k \le 0$ is computed to avoid overflow, $e^{\Delta_j}$ is approximated by a 256-entry lookup table over $[-8.0, 0.0]$, and the result is normalized to permille probabilities:

$$P(C_c) = \frac{\exp(\Delta_c)}{\sum_{k=0}^3 \exp(\Delta_k)} \times 1000$$

The weights are exported by a script from the trained model (`ml/model_weights.h`); they are not written by hand. At boot, `ml_self_test()` runs 24 fixed input vectors through the kernel's integer inference and checks the outputs bit for bit against the Python fixed-point reference ("all 24 vectors passed bit-level check").

### 2.4 Class-to-Queue Mapping and Probabilistic Sampling

Class probabilities are projected onto the three queues with a static mapping matrix $\mathbf{M}$ (rows: classes; columns: $Q_0, Q_1, Q_2$):

$$\mathbf{M} = \begin{bmatrix}
0.85 & 0.10 & 0.05 \\
0.70 & 0.25 & 0.05 \\
0.05 & 0.15 & 0.80 \\
0.15 & 0.70 & 0.15
\end{bmatrix}$$

$$P(Q_q) = \sum_{c=0}^3 P(C_c) \cdot M_{cq}$$

The kernel draws $r \sim \mathcal{U}(0, 1000)$ from a per-process Xorshift32 generator (seeded from the PID and a settable global seed, so runs are reproducible and there is no shared RNG state) and selects the queue:

$$Q_{\text{initial}} = \begin{cases}
0, & r < P(Q_0) \\
1, & P(Q_0) \le r < P(Q_0) + P(Q_1) \\
2, & \text{otherwise}
\end{cases}$$

This matrix is the project's **policy** and is separate from the classifier. It was not tuned to improve the reported metrics.

**Ablation modes.** Mode 3 (ML-DET) assigns $\arg\max_q P(Q_q)$ instead of sampling. Mode 4 (RAND-CTRL) skips the classifier and samples from the fixed prior $[333, 334, 333]$. If the classifier carries useful information, ML-PROB should beat RAND-CTRL; if randomization alone were helpful, RAND-CTRL would match ML-PROB.

If observation is incomplete when a process exits, or inference fails, no assignment is made and the process follows conventional MLFQ feedback.

---

## 3. Experimental Evaluation

### 3.1 Workload Suite

All workloads are real xv6 user programs (`user/cpuwork.c`, `iowork.c`, `interactive.c`, `mixedwork.c`) launched by `schedtest`:

- **Exp 1 (CPU-bound)**: 4 concurrent `cpuwork` processes.
- **Exp 2 (Interactive)**: 4 `interactive` processes alternating short compute bursts with voluntary sleeps.
- **Exp 3 (Mixed CPU and interactive)**: 2 compute-heavy and 2 interactive processes together.
- **Exp 4 (I/O-heavy)**: 4 `iowork` processes that repeatedly block.
- **Exp 5 (High-concurrency mixed)**: 8 processes spanning all four behavior profiles.
- **Exp 6 (Starvation stress)**: 1 long CPU hog with 4 staggered interactive arrivals.

### 3.2 Scheduling Modes

1. **Mode 0 (RR)**: original xv6 round robin.
2. **Mode 1 (MLFQ)**: conventional 3-level MLFQ; all processes enter $Q_0$.
3. **Mode 2 (ML-PROB)**: PM-MLFQ with probabilistic initial queue.
4. **Mode 3 (ML-DET)**: deterministic ML ablation.
5. **Mode 4 (RAND-CTRL)**: classifier-free random ablation.

### 3.3 Method

150 trials were run (5 modes $\times$ 6 workloads $\times$ 5 seeds, seeds 101-105; the same seeds in every mode) on a single CPU (`CPUS=1`) in QEMU. The kernel prints one `SCHEDLOG` line per process at exit (arrival, first run, exit, CPU ticks, waiting ticks, initial queue, predicted class, confidence, transitions). A host script parses these logs (`results/raw/`) to compute turnaround (exit $-$ arrival), waiting (runnable but not running), response (first run $-$ arrival), classification fraction and queue transitions. Means, standard deviations, 95% confidence intervals and paired differences were recomputed from the raw logs. All 150 trials completed without failures. Plots in `results/` were generated from the same data. The raw logs were produced before two later, default-off additions (Section 7); a spot-check of Exp 1 under the final kernel showed the same process counts and state transitions, with small run-to-run timer variation, but the full set was not re-run.

### 3.4 Results and Classification Rates

**Table 1: Benchmark summary metrics and fraction of processes classified.** (Classification applies to modes 2-4 only; modes 0 and 1 have no classifier.)

| Workload | Metric | Mode 0 (RR) | Mode 1 (MLFQ) | Mode 2 (ML-PROB) | Mode 3 (ML-DET) | Mode 4 (RAND-CTRL) |
|---|---|---|---|---|---|---|
| **Exp 1: CPU-bound** | Turnaround mean (ticks) | 10.05 | **9.20** | 9.90 | 9.65 | 9.30 |
| | Wait time mean (ticks) | 7.50 | 6.70 | 7.35 | 7.10 | 6.80 |
| | Fraction classified | n/a | n/a | 45.0% | 40.0% | 45.0% |
| **Exp 2: Interactive** | Turnaround mean (ticks) | 20.00 | 20.00 | 20.00 | 20.00 | 20.00 |
| | Wait time mean (ticks) | 20.00 | 20.00 | 20.00 | 20.00 | 20.00 |
| | Fraction classified | n/a | n/a | 100.0% | 100.0% | 100.0% |
| **Exp 3: Mixed** | Turnaround mean (ticks) | 14.35 | 14.35 | 14.65 | 14.40 | 14.30 |
| | Wait time mean (ticks) | 13.00 | 13.00 | 13.20 | 13.10 | 12.80 |
| | Fraction classified | n/a | n/a | 75.0% | 75.0% | 80.0% |
| **Exp 4: I/O-heavy** | Turnaround mean (ticks) | 20.05 | 20.20 | 20.25 | 20.05 | 20.20 |
| | Wait time mean (ticks) | 20.00 | 20.05 | 20.00 | 20.05 | 20.00 |
| | Fraction classified | n/a | n/a | 100.0% | 100.0% | 100.0% |
| **Exp 5: Concurrency** | Turnaround mean (ticks) | 23.63 | **23.10** | 23.53 | 25.68 | 24.40 |
| | Wait time mean (ticks) | 18.75 | **17.45** | 17.90 | 19.88 | 19.30 |
| | Fraction classified | n/a | n/a | 97.5% | 100.0% | 100.0% |
| **Exp 6: Starvation** | Turnaround mean (ticks) | 7.56 | 7.60 | 7.64 | 7.64 | 7.64 |
| | Fraction classified | n/a | n/a | 100.0% | 100.0% | 100.0% |

Corresponding plots are in `results/` (`turnaround_comparison.png`, `waiting_time_comparison.png`, `response_time_comparison.png`, `comparison.png`) and per-process data in `results/experiment_results.csv`.

---

## 4. Statistical Analysis

### 4.1 Paired Differences ($n=5$ seeds)

We compare policies on identical seeds ($\Delta = \text{ML-PROB} - \text{baseline}$; positive means ML-PROB is slower) with two-sided 95% confidence intervals ($t_{\text{crit}} = 2.776$, $df=4$).

**Table 2: Paired seed turnaround differences, ML-PROB vs baselines**

| Workload | Comparison | Paired diffs (seeds 101-105) | Mean $\Delta$ | Std dev | 95% CI | Distinguishable from 0? |
|---|---|---|---|---|---|---|
| **Exp 1** | vs MLFQ | [0.0, 1.0, 1.0, 1.0, 0.5] | **+0.700** | 0.447 | **[+0.145, +1.255]** | **Yes** (MLFQ faster) |
| | vs ML-DET | [-0.25, 0.0, 0.0, 1.0, 0.5] | +0.250 | 0.500 | [-0.371, +0.871] | No |
| | vs RAND-CTRL | [0.75, 0.75, 1.0, 0.25, 0.25] | **+0.600** | 0.335 | **[+0.184, +1.016]** | **Yes** (RAND-CTRL faster) |
| **Exp 5** | vs MLFQ | [2.125, -1.25, -0.375, 3.0, -1.375] | +0.425 | 2.013 | [-2.074, +2.924] | No |
| | vs ML-DET | [-0.375, -3.75, -2.875, 0.125, -3.875] | -2.150 | 1.897 | [-4.504, +0.204] | No |
| | vs RAND-CTRL | [-1.375, -0.625, -0.5, 0.125, -2.0] | -0.875 | 0.824 | [-1.899, +0.149] | No |

**Interpretation.**

1. **Exp 1 (CPU-bound).** Zero is excluded from the interval against MLFQ, so conventional MLFQ was faster than PM-MLFQ here, by about 0.7 ticks on average. A plausible (untested) explanation is the provisional $Q_1$ observation phase, which delays CPU-bound processes that MLFQ would run immediately in $Q_0$; only 40-45% of the Exp 1 processes were classified before exiting (Section 4.3), so the classifier had limited chance to act. RAND-CTRL was also faster than ML-PROB (+0.6 ticks, interval excludes zero), so in this setting the classifier-driven policy was measurably slightly worse than random assignment.
2. **Exp 5 (concurrency).** ML-PROB's sample mean was lower than ML-DET's and RAND-CTRL's, but all three intervals span zero with $n=5$. These differences cannot be called statistically distinguishable.
3. **Variance.** Deterministic modes (0 and 1) have little trial-to-trial variance, while randomized modes (2 and 4) add PRNG sampling variance and need larger $n$ (we estimate $n \ge 20$) to resolve small effects.

### 4.2 Queue Assignment Distribution (Exp 5) and the Clustering Hypothesis

We hypothesized that deterministic assignment (ML-DET) would cluster similar processes into one queue and hurt turnaround under load. Table 3 shows the initial queues actually assigned across the 40 processes of Exp 5 (8 processes $\times$ 5 seeds).

**Table 3: Initial queue (`init_q`) distribution in Exp 5**

| Policy | $Q_0$ | $Q_1$ | $Q_2$ | Unclassified | Breakdown |
|---|---|---|---|---|---|
| **Mode 1 (MLFQ)** | all | 0 | 0 | n/a | all processes start in $Q_0$ by definition |
| **Mode 3 (ML-DET)** | 20 (50.0%) | 10 (25.0%) | 10 (25.0%) | 0 | 10/10 `cpuwork` $\to Q_2$; 10/10 `interactive` $\to Q_0$; 10/10 `iowork` $\to Q_0$; 10/10 `mixedwork` $\to Q_1$ |
| **Mode 2 (ML-PROB)** | 20 (50.0%) | 7 (17.5%) | 12 (30.0%) | 1 (2.5%) | `cpuwork`: 6 in $Q_2$, 2 in $Q_0$, 1 in $Q_1$, 1 unclassified; `mixedwork`: 5 in $Q_1$, 4 in $Q_2$, 1 in $Q_0$ |
| **Mode 4 (RAND-CTRL)** | 13 (32.5%) | 13 (32.5%) | 14 (35.0%) | 0 | uniform spread regardless of process type |

**Findings.** ML-DET assigns every process of the same type to the same queue, which is the classifier behaving as designed (the classifier agreed with the true workload type on these runs). ML-PROB spreads `cpuwork` and `mixedwork` over several queues. The data therefore show that sampling disperses placements, but this table does not show that the dispersion is harmful or beneficial. The turnaround advantage of ML-PROB over ML-DET in Exp 5 (-2.150 ticks) is not statistically distinguishable from zero (Table 2). The clustering explanation remains an untested hypothesis, and we do not claim it as a result.

### 4.3 Classification Rates and Early Exits

- Exps 2, 4, 6: 100% of processes in modes 2-4 completed observation and were classified.
- Exp 5: 97.5% (39 of 40) in ML-PROB.
- Exp 3: 75-80%.
- Exp 1: only 40-45%.

CPU-bound processes in Exp 1 ran in a few long timeslices and several exited before completing the 3-burst minimum. For these processes, no ML assignment occurred and conventional feedback demotion applied. In Exp 1, therefore, much of ML-PROB's behavior is that of MLFQ preceded by an observation delay, which limits what that experiment can show about the classifier.

### 4.4 Model Generalization and Sample Size

The network was trained on 6,000 synthetic samples generated from 20 workload blueprints (four classes) with gradient-based optimization. Evaluation used a group split by blueprint: blueprints 3, 7, 12 and 18 were held out, which avoids leakage from near-identical generated samples. Test accuracy on the held-out blueprints was **95.5%**. Per-class precision, recall, F1, the confusion matrix and training curves are in `ml/model_metrics.json`.

For sim-to-real transfer we collected **16 real feature traces** from the actual xv6 workload programs (`ml/real_traces.csv`), none of which were in the training data. 15 of 16 were classified correctly (93.75%). The exact (Clopper-Pearson) 95% binomial confidence interval for 15/16 is approximately **[69.8%, 99.8%]**. This indicates agreement between synthetic features and real kernel features but is too small a sample to establish accuracy on arbitrary programs. The synthetic training data comes from blueprints we wrote ourselves, so the high held-out accuracy shows the model learns those blueprints, not that it generalizes to workloads beyond them.

### 4.5 Inference Overhead

`ml_predict()` is timed in `kernel/ml_scheduler.c` with `r_time()` deltas. `r_time()` reads the RISC-V `time` CSR, which runs at 10 MHz on the QEMU `virt` platform (100 ns per unit), not at the CPU clock.

- **Mean over 725 classified processes: 32.27 time-CSR units, i.e. about 3.2 µs.**
- **Caveat:** this is measured on QEMU's virtual timebase under emulation, so it indicates the order of magnitude of the cost relative to other kernel operations, not latency on physical hardware.
- **Heap allocation:** none on the inference path (no `kalloc`).

### 4.6 Starvation Safeguards

Exp 6 uses one long CPU hog with four staggered interactive arrivals. In a single run in Mode 1 with aging and boost enabled, the hog completed in 9 ticks; with aging and boost disabled (`DISABLE_AGING_BOOST`), it took 27 ticks. This is one run (not repeated across seeds), so it is a demonstration that the mechanisms matter, not a statistical estimate. In the 150 main trials (aging and boost enabled) no process was starved.

---

## 5. Limitations and Threats to Validity

1. **Synthetic training data.** The model was trained on blueprints we designed and checked on only 16 real traces (Section 4.4).
2. **Coarse time resolution.** Scheduling and feature timing are at tick granularity, so very short bursts are recorded as 0 or 1 tick. This is a main reason Exps 2-4 could not distinguish the policies.
3. **Small sample.** Five seeds were enough to show that MLFQ is faster in Exp 1 but not enough to resolve the Exp 5 differences. Randomized modes need more seeds.
4. **Short workloads.** Many processes live only a few tens of ticks, so observation often ends after the process is already close to finishing (Section 4.3).
5. **Single-CPU validation only.** All 150 trials and the `usertests -q` run used for validation were on a single CPU (`CPUS=1`). The multi-CPU build was only boot-tested. In a later manual `usertests -q` run on the default 3-CPU build of Mode 2, the run stalled at `test truncate3`; this was not diagnosed (it may be slowness or a multi-CPU scheduling defect). **We therefore do not claim multi-CPU correctness of the scheduler.** Please run the project with `CPUS=1`.
6. **Observation cost.** Running provisionally in $Q_1$ delays processes that would otherwise have run immediately in $Q_0$.
7. **Hand-written workloads.** The workload programs are our own and may not represent real application mixes.
8. **Inference cost** was measured under QEMU emulation only.

---

## 6. Conclusion

The project shows that a trained neural network can be integrated into the xv6 kernel with integer-only arithmetic, with kernel inference matching its Python fixed-point reference bit for bit and a small per-process cost. However, in our evaluation:

1. PM-MLFQ **did not** beat conventional MLFQ in turnaround time in any experiment, and in Exp 1 MLFQ was measurably faster.
2. Experiments 2-4 could not discriminate between policies.
3. The classifier did not show a statistically distinguishable benefit over a classifier-free random control, and in Exp 1 the random control was measurably faster.
4. ML-PROB had a lower sample mean than ML-DET in Exp 5, but the difference was not statistically distinguishable at $n=5$; whether sampling helps relative to deterministic assignment under heavier load is left open.

The contributions are therefore the working in-kernel pipeline (observation, integer inference, probabilistic assignment, feedback, aging and boosting), the ablation design that separates the effect of the classifier from that of randomization, and an honestly reported inconclusive result. Future work includes longer-running and more varied workloads, finer-grained timing, larger numbers of seeds, calibration of the classifier's probabilities, tuning of the class-to-queue policy on held-out seeds, training on real traces, and multi-CPU validation.

---

## 7. Additional Diagnostic and Safety Features

These two small features are secondary to the PM-MLFQ contribution. Both are controlled by compile-time flags that default to off (`SCHED_TRACE=0`, `PROC_MEM_LIMIT=0`), so default builds and the experiments above are unaffected.

### 7.1 Scheduler Event Trace ("Flight Recorder")

- **Design.** A fixed ring buffer of 128 entries (`kernel/trace.c`) records scheduler events: process creation, classification (predicted class and sampled queue), demotion, aging promotion, priority boost and process exit. Each entry stores tick, CPU, PID, event type and two arguments.
- **Locking.** A dedicated spinlock is a leaf lock: it may be taken while holding `p->lock` or `tickslock`, and nothing else is acquired while holding it. Trace logging never calls `printf` or sleeps.
- **Access.** The `gettrace` system call copies a snapshot out with `copyout` after releasing the lock; the `schedtrace` user tool prints it oldest first.
- **Panic dump.** On `panic()` the last 16 entries are printed with a best-effort, lock-free read (they may be slightly inconsistent if another CPU is writing).
- **Evidence.** `results/trace_demo.txt` shows demotions, aging promotions, boosts and classification events from real runs. The wrap-around was tested by running eight consecutive `schedtest 1 101` runs and reading the buffer back with `schedtrace`; it returned exactly 128 entries, oldest first, with the first tick at 1287 (`results/trace_wrap_test.txt`). The panic dump was checked with a temporary test panic that has since been removed.
- **Overhead.** In a few runs of Exp 1 under Mode 2, turnaround and inference time with tracing on and off were within timer jitter. This is not a rigorous measurement.

### 7.2 Per-Process Memory Cap

- **Design.** When `PROC_MEM_LIMIT` is nonzero, `growproc()` (used by `sbrk`) refuses to grow a process beyond the limit and returns $-1$, and `exec()` fails cleanly if a program would exceed it. Refusals are counted per process and shown in `SCHEDLOG` only when the feature is on.
- **Why refusal and not stalling.** Putting allocating processes to sleep until memory frees can deadlock the kernel if all processes wait for memory, so the allocator fails fast instead.
- **Scope.** This is a per-process limit, not a system-wide limit.
- **Evidence.** `results/memcap_demo.txt`: with the cap disabled, the `memhog` program grew to its test ceiling (over 32 MB); with a 4 MB cap it stopped at exactly 4,194,304 bytes and recorded one refusal.
- **Host RAM.** Separately from the kernel, the memory used by QEMU on the host can be bounded with, for example, `systemd-run --user --scope -p MemoryMax=2G make qemu`. This is a host-level tip and not part of the kernel.

---

## 8. Reproducibility

Build a mode with `make SCHED_MODE=<0-4> CPUS=1 qemu`. The README documents how to regenerate the dataset, train and export the model, run the experiments, and recompute the tables and plots from `results/raw/`. The raw logs, `ml/model_metrics.json`, `ml/real_traces.csv` and the kernel-versus-Python validation output are included in the repository.

---

## References

1. F. J. Corbató, M. Merwin-Daggett, R. C. Daggett, "An Experimental Time-Sharing System," *Proc. AFIPS Spring Joint Computer Conference*, 1962.
2. R. H. Arpaci-Dusseau, A. C. Arpaci-Dusseau, *Operating Systems: Three Easy Pieces*, chapter "Scheduling: The Multi-Level Feedback Queue."
3. R. Cox, M. F. Kaashoek, R. Morris, *xv6: a simple, Unix-like teaching operating system* (MIT, RISC-V edition).
4. C. J. Clopper, E. S. Pearson, "The use of confidence or fiducial limits illustrated in the case of the binomial," *Biometrika*, 1934.
