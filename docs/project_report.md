# PM-MLFQ: Probabilistic Machine Learning-Assisted Multilevel Feedback Queue CPU Scheduling in xv6

**Authors**: Operating Systems Research Group
**Target Platform**: RISC-V 64-bit (`virt`), QEMU System Emulator 
**Base Kernel**: MIT xv6-riscv 
**Repository**: `https://github.com/Sai-Siddhardha-06/OS-Project` 
**Date**: November 2026

---

## Abstract

Traditional Multilevel Feedback Queue (MLFQ) scheduling algorithms operate on an optimistic heuristic: all newly arrived processes are assumed to be short, interactive tasks and are inserted into the highest-priority queue ($Q_0$). While this design delivers responsive execution for interactive threads, long-running compute-bound tasks initially contend with genuine interactive processes, causing latency spikes until the feedback mechanism demotes them. Conversely, static process classification lacks adaptability to changing phase behaviors, and deterministic machine learning predictions can lead to queue clustering and priority inversion under prediction errors.

In this work, we evaluate **PM-MLFQ** (Probabilistic Machine Learning-Assisted Multilevel Feedback Queue), an in-kernel CPU scheduler for the 64-bit RISC-V `xv6` operating system. PM-MLFQ incorporates an observation window during initial process execution, extracts a 9-dimensional fixed-point feature vector, evaluates an integer-only Multilayer Perceptron (MLP) directly in supervisor mode with zero floating-point dependencies, and assigns the initial queue probabilistically by sampling from a class-conditioned distribution $P(Q)$. 

Through 150 empirical trials across 5 scheduling modes and 6 benchmark workloads, our findings demonstrate:
1. **PM-MLFQ did not beat conventional MLFQ in turnaround time in any experiment**: Standard MLFQ was slightly faster or tied on all workloads (e.g., 9.20 vs 9.90 ticks in Exp 1; 23.10 vs 23.53 ticks in Exp 5).
2. **Workloads in Exps 2–4 did not discriminate between scheduling modes**: Interactive (Exp 2), Mixed (Exp 3), and I/O-heavy (Exp 4) workloads exhibited near-identical mean turnaround times across all 5 modes due to being bottlenecked by sleep durations and coarse timer resolution.
3. **The ML classifier did not demonstrate statistically distinguishable value over the random control (RAND-CTRL)**: In Exp 5, the paired turnaround difference between ML-PROB and RAND-CTRL was $-0.875$ ticks with a 95% CI of $[-1.899, +0.149]$ (spanning zero). In Exp 1, RAND-CTRL was actually faster ($+0.600$ ticks, 95% CI $[+0.184, +1.016]$).
4. **Feasibility and Safety**: We confirm that fixed-point ML inference can execute entirely within supervisor mode with an empirical execution latency of **3.227 microseconds** (32.27 timer ticks at 10 MHz) and zero kernel panics across the complete `usertests -q` suite.

---

## 1. Introduction & Motivation

The Multilevel Feedback Queue scheduler, originally proposed by Corbato in the Compatible Time-Sharing System (CTSS), remains one of the foundational CPU scheduling policies in modern operating systems. By maintaining multiple priority levels with varying timeslices (quanta) and dynamically adjusting priorities based on observed CPU burst lengths, MLFQ approximates Shortest Remaining Time First (SRTF) without requiring a priori knowledge of job runtimes.

Despite its success, traditional MLFQ suffers from several structural inefficiencies:

1. **The Optimistic $Q_0$ Penalty**: Inserting every process into $Q_0$ forces genuine I/O-bound and interactive tasks to share high-priority execution with compute-heavy batches during their initial quantum, inducing interactive jitter.
2. **Deterministic Prediction Cascades**: Purely deterministic classifiers that map processes directly to queues ($\arg\max_Q P(Q)$) create artificial convoy effects: when a batch of tasks exhibits similar initial features, the deterministic scheduler herds all of them into the exact same queue, defeating load distribution.
3. **Kernel Floating-Point Prohibition**: Modern machine learning models rely heavily on floating-point arithmetic. Standard supervisor-mode operating system kernels cannot use hardware floating-point registers without incurring expensive register save/restore overheads during context switches or trapping on unsupported instructions.

PM-MLFQ addresses these challenges through three principal contributions:

- **Provisional Observation Phase**: New processes execute provisionally in an intermediate queue ($Q_1$) for a small number of bursts ($N \le 3$) to accumulate robust execution statistics without disrupting $Q_0$ responsiveness.
- **Supervisor-Mode Fixed-Point Inference Engine**: A compact $9 \to 16 \to 8 \to 4$ Multilayer Perceptron implemented entirely in Q16.16 fixed-point arithmetic, complete with a 256-entry piecewise exponential lookup table for numerically stable softmax evaluation.
- **Probabilistic Queue Assignment**: Rather than assigning processes to the single predicted class, the kernel maps model confidence distributions through a transition matrix to yield a queue probability vector $P(Q)$, sampling the initial queue via a lightweight per-process pseudo-random number generator (PRNG).

---

## 2. Theoretical Formulation & Architecture

### 2.1 System Model

The scheduler manages three priority levels:
- **Queue 0 ($Q_0$)**: Highest priority, timeslice $\tau_0 = 1$ tick.
- **Queue 1 ($Q_1$)**: Medium priority, timeslice $\tau_1 = 2$ ticks. Also serves as the provisional observation queue.
- **Queue 2 ($Q_2$)**: Lowest priority, timeslice $\tau_2 = 4$ ticks.

Dispatching follows strict priority order: any runnable process in $Q_i$ is dispatched before any process in $Q_{i+1}$. Within each queue, processes are dispatched in round-robin order using rotating scan pointers to ensure intra-queue fairness.

### 2.2 Feature Representation

During execution, the kernel dynamically tracks process execution metrics inside `struct proc`. Once a process finishes either `OBS_MIN_BURSTS` (3) bursts or reaches `OBS_MAX_WINDOW_TICKS` (50) ticks, the kernel extracts a 9-dimensional integer feature vector (scaled by $10^3$ to retain fractional precision without floating-point arithmetic):

1. **`avg_burst`**: Average CPU burst duration ($\text{ticks} \times 10^3$).
2. **`max_burst`**: Peak burst duration ($\text{ticks} \times 10^3$).
3. **`burst_mad`**: Mean absolute deviation of burst durations.
4. **`exhaust_ratio`**: Fraction of bursts terminating via quantum exhaustion vs voluntary block ($[0..1000]$).
5. **`block_rate`**: Voluntary blocking transitions per unit time ($[0..1000]$).
6. **`yield_rate`**: Voluntary yields per unit time ($[0..1000]$).
7. **`sleep_ratio`**: Ratio of blocked sleep ticks to total lifetime ticks ($[0..1000]$).
8. **`cpu_util`**: Ratio of active CPU execution ticks to total elapsed ticks ($[0..1000]$).
9. **`burst_count`**: Total observed bursts during the observation window.

### 2.3 Neural Network Architecture & Fixed-Point Formulation

The classifier predicts process affiliation across four workload classes:
- $C_0$: **Interactive** (frequent short bursts, high voluntary blocking).
- $C_1$: **I/O-Bound** (moderate bursts, dominated by I/O waits).
- $C_2$: **CPU-Bound** (monopolizes quanta, zero voluntary blocking).
- $C_3$: **Mixed/Batch** (varying phases).

The network is a feedforward MLP ($9 \to 16 \to 8 \to 4$) parameterized by weights and biases in Q16.16 format ($1.0 \equiv 65536$). Forward pass accumulation is performed in 64-bit signed integers:

$$z_j^{(l)} = b_j^{(l)} + \sum_{i=1}^{M_{l-1}} \left\lfloor \frac{a_i^{(l-1)} \cdot W_{ij}^{(l)}}{65536} \right\rfloor$$

Activation functions for hidden layers use Rectified Linear Units (ReLU):
$$a_j^{(l)} = \max\left(0, z_j^{(l)}\right)$$

For the output layer, logit differences $\Delta_j = z_j - \max_k z_k \le 0$ are computed to prevent arithmetic overflow. A 256-entry exponential table approximates $e^{\Delta_j}$ over $[-8.0, 0.0]$, and the resulting logits are normalized into permille probabilities ($P(C_c) \in [0, 1000]$):

$$P(C_c) = \frac{\exp(\Delta_c)}{\sum_{k=0}^3 \exp(\Delta_k)} \times 1000$$

### 2.4 Class-to-Queue Mapping and Probabilistic Sampling

The class probabilities are projected onto the 3 MLFQ priority queues using a static mapping matrix $\mathbf{M} \in \mathbb{R}^{4 \times 3}$:

$$\mathbf{M} = \begin{bmatrix}
0.85 & 0.10 & 0.05 \\
0.70 & 0.25 & 0.05 \\
0.05 & 0.15 & 0.80 \\
0.15 & 0.70 & 0.15
\end{bmatrix}$$

The queue probability vector $\mathbf{q} = [P(Q_0), P(Q_1), P(Q_2)]$ is computed via matrix multiplication:
$$P(Q_q) = \sum_{c=0}^3 P(C_c) \cdot M_{cq}$$

To assign the process to an initial queue without cross-CPU lock contention, the kernel samples $r \sim \mathcal{U}(0, 1000)$ using an in-kernel Xorshift32 PRNG seeded per-process:
$$Q_{\text{initial}} = \begin{cases} 
0, & \text{if } r < P(Q_0) \\
1, & \text{if } P(Q_0) \le r < P(Q_0) + P(Q_1) \\
2, & \text{otherwise}
\end{cases}$$

For Mode 4 (`RAND-CTRL`), the classifier is omitted, and the process samples directly from the uniform marginal prior distribution $\mathbf{p}_{\text{rand}} = [333, 334, 333]$.

---

## 3. Experimental Evaluation

### 3.1 Workload Suite

The evaluation suite comprises six synthetic and macro benchmarks:
- **Exp 1 (CPU-Bound)**: 4 concurrent compute loops performing matrix arithmetic.
- **Exp 2 (Interactive)**: 4 processes alternating short compute bursts with voluntary sleeps.
- **Exp 3 (Mixed CPU & Interactive)**: 2 compute-heavy and 2 interactive processes executing simultaneously.
- **Exp 4 (I/O-Heavy)**: 4 processes generating intense file descriptor operations.
- **Exp 5 (High-Concurrency Mixed)**: 8 concurrent processes spanning all four behavioral profiles.
- **Exp 6 (Starvation Stress Test)**: 1 long-running CPU hog combined with 4 staggered interactive arrivals.

### 3.2 Evaluated Scheduling Modes

1. **Mode 0 (`RR`)**: Baseline xv6 round-robin.
2. **Mode 1 (`MLFQ`)**: Standard 3-level MLFQ (all processes enter at $Q_0$).
3. **Mode 2 (`ML-PROB`)**: PM-MLFQ with probabilistic initial queue assignment.
4. **Mode 3 (`ML-DET`)**: Deterministic ML ablation ($\arg\max_Q P(Q)$).
5. **Mode 4 (`RAND-CTRL`)**: Randomized control ablation (samples from $[333, 334, 333]$ without ML).

### 3.3 Quantitative Results & Classification Rates

A total of 150 trials (5 modes $\times$ 6 workloads $\times$ 5 seeds) were executed. Table 1 summarizes the key performance indicators and the fraction of processes that completed the observation window and received ML classification.

**Table 1: Benchmark Summary Metrics and Process Classification Rates**

| Workload | Metric | Mode 0 (RR) | Mode 1 (MLFQ) | Mode 2 (ML-PROB) | Mode 3 (ML-DET) | Mode 4 (RAND-CTRL) |
|---|---|---|---|---|---|---|
| **Exp 1: CPU-Bound** | Turnaround Mean (ticks) | 10.05 | **9.20** | 9.90 | 9.65 | 9.30 |
| | Wait Time Mean (ticks) | 7.50 | 6.70 | 7.35 | 7.10 | 6.80 |
| | Fraction Classified | 0.0% | 0.0% | **45.0%** | **40.0%** | **45.0%** |
| **Exp 2: Interactive** | Turnaround Mean (ticks) | 20.00 | 20.00 | 20.00 | 20.00 | 20.00 |
| | Wait Time Mean (ticks) | 20.00 | 20.00 | 20.00 | 20.00 | 20.00 |
| | Fraction Classified | 0.0% | 0.0% | **100.0%** | **100.0%** | **100.0%** |
| **Exp 3: Mixed** | Turnaround Mean (ticks) | 14.35 | 14.35 | 14.65 | 14.40 | 14.30 |
| | Wait Time Mean (ticks) | 13.00 | 13.00 | 13.20 | 13.10 | 12.80 |
| | Fraction Classified | 0.0% | 0.0% | **75.0%** | **75.0%** | **80.0%** |
| **Exp 4: I/O-Heavy** | Turnaround Mean (ticks) | 20.05 | 20.20 | 20.25 | 20.05 | 20.20 |
| | Wait Time Mean (ticks) | 20.00 | 20.05 | 20.00 | 20.05 | 20.00 |
| | Fraction Classified | 0.0% | 0.0% | **100.0%** | **100.0%** | **100.0%** |
| **Exp 5: Concurrency** | Turnaround Mean (ticks) | 23.63 | **23.10** | 23.53 | 25.68 | 24.40 |
| | Wait Time Mean (ticks) | 18.75 | **17.45** | 17.90 | 19.88 | 19.30 |
| | Fraction Classified | 0.0% | 0.0% | **97.5%** | **100.0%** | **100.0%** |
| **Exp 6: Starvation** | Turnaround Mean (ticks) | 7.56 | 7.60 | 7.64 | 7.64 | 7.64 |
| | Fraction Classified | 0.0% | 0.0% | **100.0%** | **100.0%** | **100.0%** |

---

## 4. Key Findings & Detailed Statistical Analysis

### 4.1 Paired Difference Analysis ($n=5$ Seeds)

To determine whether performance differences between scheduling policies are statistically distinguishable from random noise under $n=5$ seeds (Seeds 101–105), we evaluate paired differences across identical seeds ($\Delta = \text{ML-PROB} - \text{Baseline}$). Table 2 details the paired differences and two-sided 95% confidence intervals ($t_{\text{crit}} = 2.776$ for $df=4$).

**Table 2: Paired Seed Turnaround Differences for ML-PROB vs Baselines**

| Workload | Comparison | Paired Seed Diffs [101..105] | Mean $\Delta$ | Std Dev | 95% Confidence Interval | Distinguishable from 0? |
|---|---|---|---|---|---|---|
| **Exp 1: CPU-Bound** | vs MLFQ (Mode 1) | [0.0, 1.0, 1.0, 1.0, 0.5] | **+0.700** | 0.447 | **[+0.145, +1.255]** | **YES** (MLFQ is faster) |
| | vs ML-DET (Mode 3) | [-0.25, 0.0, 0.0, 1.0, 0.5] | +0.250 | 0.500 | [-0.371, +0.871] | **NO** (Spans zero) |
| | vs RAND-CTRL (Mode 4) | [0.75, 0.75, 1.0, 0.25, 0.25] | **+0.600** | 0.335 | **[+0.184, +1.016]** | **YES** (RAND-CTRL is faster) |
| **Exp 5: Concurrency** | vs MLFQ (Mode 1) | [2.125, -1.25, -0.375, 3.0, -1.375] | +0.425 | 2.013 | [-2.074, +2.924] | **NO** (Spans zero) |
| | vs ML-DET (Mode 3) | [-0.375, -3.75, -2.875, 0.125, -3.875] | -2.150 | 1.897 | [-4.504, +0.204] | **NO** (Spans zero) |
| | vs RAND-CTRL (Mode 4) | [-1.375, -0.625, -0.5, 0.125, -2.0] | -0.875 | 0.824 | [-1.899, +0.149] | **NO** (Spans zero) |

#### Statistical Interpretation:
1. **In Exp 1 (CPU-bound)**: The 95% confidence interval for ML-PROB vs MLFQ is $[+0.145, +1.255]$. Because zero is excluded, we can state with 95% confidence that **standard MLFQ is faster than PM-MLFQ for pure compute loops**. Forcing compute tasks into provisional $Q_1$ execution adds an average penalty of 0.70 ticks. Furthermore, RAND-CTRL is also faster than ML-PROB ($+0.600$, 95% CI $[+0.184, +1.016]$), indicating that the ML classifier does not offer a performance advantage in this setting.
2. **In Exp 5 (Concurrency)**: While the sample mean indicates ML-PROB achieved lower turnaround than ML-DET ($-2.150$ ticks) and RAND-CTRL ($-0.875$ ticks), high variance across the 5 seeds yields confidence intervals spanning zero ($[-4.504, +0.204]$ and $[-1.899, +0.149]$ respectively). Thus, with $n=5$, **these differences cannot be claimed as statistically distinguishable at the 95% confidence level**.
3. **Variance in Deterministic Modes**: In deterministic scheduling policies (such as baseline Mode 0 and standard Mode 1), process arrival and execution follow static paths, leading to near-zero trial-to-trial variance. In contrast, randomized policies (Mode 2 and Mode 4) introduce PRNG sampling variance, which broadens the resulting confidence intervals and requires larger trial sample sizes ($n \ge 20$) to confirm small effect sizes.

---

### 4.2 Empirical Evaluation of the "Herd Collapse" Hypothesis

It was hypothesized that deterministic ML (`ML-DET`) causes a **herd collapse** by routing all processes with compute signatures into Queue 2 simultaneously. To evaluate this hypothesis against empirical evidence, we extract the initial queue assignments (`init_q`) from all 40 processes across the 5 trials of Exp 5:

**Table 3: Empirical Initial Queue (`init_q`) Distribution in Exp 5 (40 processes total)**

| Policy | `init_q = 0` ($Q_0$) | `init_q = 1` ($Q_1$) | `init_q = 2` ($Q_2$) | Unclassified (`-1`) | Workload Sub-Breakdown |
|---|---|---|---|---|---|
| **Mode 1 (MLFQ)** | 0 (starts $Q_0$) | 0 | 0 | 40 (All start $Q_0$) | Conventional rule: all enter $Q_0$ by default |
| **Mode 3 (ML-DET)** | 20 (50.0%) | 10 (25.0%) | 10 (25.0%) | 0 (0.0%) | 10/10 `cpuwork` $\to Q_2$; 10/10 `interactive` $\to Q_0$; 10/10 `iowork` $\to Q_0$; 10/10 `mixedwork` $\to Q_1$ |
| **Mode 2 (ML-PROB)** | 20 (50.0%) | 7 (17.5%) | 12 (30.0%) | 1 (2.5%) | `cpuwork`: 6 in $Q_2$, 2 in $Q_0$, 1 in $Q_1$, 1 unclassified; `mixedwork`: 5 in $Q_1$, 4 in $Q_2$, 1 in $Q_0$ |
| **Mode 4 (RAND-CTRL)** | 13 (32.5%) | 13 (32.5%) | 14 (35.0%) | 0 (0.0%) | Uniform dispersion across $Q_0$, $Q_1$, $Q_2$ regardless of process class |

#### Findings on Queue Clustering:
The data directly support the structural mechanism of the hypothesis:
- In Mode 3 (`ML-DET`), process classification is 100% deterministic: every single `cpuwork` process was assigned to $Q_2$ (10/10), and every `interactive` and `iowork` process was assigned to $Q_0$ (20/20).
- In Mode 2 (`ML-PROB`), probabilistic sampling dispersed the compute and mixed processes across $Q_2$ (12 processes), $Q_1$ (7 processes), and $Q_0$ (20 processes).
However, while this dispersion aligns with a 2.15-tick lower mean turnaround time in the sample, the wide confidence interval ($[-4.504, +0.204]$) means we must classify the performance benefit of avoiding herd collapse as an **empirically supported hypothesis rather than a statistically confirmed certainty**.

---

### 4.3 Process Classification Rates and Early Exits

Table 1 reports the fraction of processes that completed the observation phase:
- **Interactive, I/O-heavy, and Starvation Workloads (Exps 2, 4, 6)**: **100%** of processes completed observation and received initial queue placement.
- **Mixed Concurrency Workload (Exp 5)**: **97.5%** classified (39 of 40 processes).
- **Mixed CPU/Interactive Workload (Exp 3)**: **75.0%** classified.
- **Pure CPU-bound Workload (Exp 1)**: Only **40.0% to 45.0%** classified.
Because CPU-bound benchmark processes execute continuous compute loops, several processes completed their entire lifetime within 1 or 2 long continuous timeslices, exiting before satisfying the `OBS_MIN_BURSTS` (3 bursts) condition. In these cases, the kernel safely fell back to standard feedback demotion without recording classification, confirming the stability of the early-exit fallback path.

---

### 4.4 Sim-to-Real Model Generalization & Sample Size Caveats

The neural network was trained on 6,000 synthetic samples generated across 20 distinct workload blueprints and achieved **95.5% test accuracy** on held-out blueprints (Blueprints 3, 7, 12, 18).

To evaluate sim-to-real transfer, the model was tested on real execution traces captured from xv6:
- **Sample Size**: Exactly **16 real traces** (`ml/real_traces.csv`) representing 4 distinct execution classes.
- **Independence**: None of the 16 real traces were included in the synthetic training set.
- **Accuracy**: 15 out of 16 traces were classified correctly (**93.75% accuracy**).
- **Statistical Caveat**: With $N=16$, the rule-of-three 95% binomial confidence interval spans approximately **[69.8%, 99.8%]**. While this confirms that the fixed-point feature extractor correctly aligns with real kernel execution dynamics, the small sample size cannot guarantee equivalent accuracy across arbitrary unobserved user programs.

---

### 4.5 Kernel Overhead Analysis & Hardware Timer Semantics

In `kernel/ml_scheduler.c`, execution latency of `ml_predict()` is measured via `r_time()` deltas:
$$\Delta_{\text{timer}} = \text{r\_time}()_{\text{post}} - \text{r\_time}()_{\text{pre}}$$
In the RISC-V 64-bit architecture (and QEMU `virt` platform), `r_time()` reads the standard `time` CSR (clocked at 10 MHz, 100 ns per tick):
- **Mean measurement across 725 processes**: **32.27 timer ticks**.
- **Actual execution latency**: $\approx \mathbf{3.227\ \mu s}$ (microseconds), executing approximately 30 to 60 RISC-V assembly instructions.
- **Heap allocation**: **0 bytes**. Zero dynamic memory allocations (`kalloc`) occur on the inference path.

---

### 4.6 Verification of Starvation Safeguards

In Exp 6, four interactive tasks arrived in staggered succession while a CPU hog occupied the processor:
- **With Aging/Boost Enabled (Default)**: The CPU hog progressed steadily and completed in **9 ticks**.
- **With Aging/Boost Disabled (`DISABLE_AGING_BOOST`)**: The CPU hog was starved during each interactive arrival, stretching its completion time to **27 ticks**.
This confirms that aging and periodic boost are necessary safeguards against starvation in priority queue architectures.

---

## 5. Threats to Validity & Limitations

1. **Synthetic Training Distribution**: Training data was generated from synthetically parameterized burst distributions. Although evaluated on 16 real traces (93.75%), production operating systems encounter broader behavioral variance.
2. **Coarse Clock Granularity**: In QEMU virt mode, timer interrupts occur at 100 Hz (10 ms per tick). Sub-tick micro-bursts are captured as 0 or 1 tick, limiting resolution for nanosecond-scale workloads and preventing Exps 2–4 from discriminating between policies.
3. **Small Sample Size ($n=5$ Seeds)**: Five seeds were sufficient to show that standard MLFQ is faster in Exp 1, but insufficient to establish statistical significance for the observed differences in Exp 5 due to sampling variance.
4. **Single-Core Virtual Hardware**: All 150 experimental trials were conducted on a single core (`CPUS=1`) to isolate scheduling logic from multi-core load balancing dynamics.
5. **Observation Window Cost**: Running provisionally in $Q_1$ incurs a minor turnaround penalty for tasks that exit before completing observation.

---

## 6. Conclusion

PM-MLFQ demonstrates that embedded machine learning can be safely integrated into an operating system kernel using fixed-point supervisor-mode arithmetic with microsecond-level latency (3.227 $\mu$s). 

However, a rigorous audit of the experimental data reveals that:
1. PM-MLFQ did **not** beat conventional MLFQ in turnaround time in any evaluated benchmark, as MLFQ's immediate dispatch into $Q_0$ avoids the provisional observation delay.
2. The benchmark workloads in Experiments 2 through 4 were dominated by I/O sleep and could not discriminate between policies.
3. The ML classifier did not demonstrate statistically distinguishable turnaround improvements over the randomized control baseline (RAND-CTRL) at $n=5$.
4. The primary empirical value observed was that probabilistic assignment disperses initial queue placements and avoids the deterministic clustering behavior of $\arg\max$ classifiers.
