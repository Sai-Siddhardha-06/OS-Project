# PM-MLFQ: Execution Plan for the Autonomous Agent

Project: Probabilistic ML-assisted Multilevel Feedback Queue scheduler in xv6-riscv.
Repo: `~/xv6-riscv` (existing). OS: Ubuntu. Emulator: QEMU.
Read this whole file before starting. Execute phase by phase, in order. Do not ask questions; make sensible decisions and record them in `DESIGN_NOTES.md`.

---

## 0. Why this project exists (the research question)

This is a university OS project. It will be reviewed for being genuine, original, and innovative, then defended in a viva. Reports without real implementation are rejected.

The central research question is:

> Does choosing a new process's initial MLFQ queue by sampling from classifier-derived probabilities improve turnaround time, compared with (a) plain MLFQ, (b) deterministic ML assignment, and (c) random assignment that ignores the classifier?

The answer may be "no improvement" or "worse". That is acceptable. A rigorous, honestly reported null or negative result is better than a fabricated win. Never tune things only to make PM-MLFQ look good.

---

## 1. Non-negotiable rules

1. Do NOT run `git init` or create any git repo. The user uploads to GitHub manually.
2. Do not delete existing xv6 files. Keep the original scheduler buildable.
3. No Python simulation may substitute for kernel behavior. All scheduling results must come from real xv6 runs in QEMU.
4. Never fabricate or hand-edit results. Every number in the report must trace to a raw log in `results/raw/`.
5. Never claim a feature works unless you ran it and saw it work. Mark unfinished items as unfinished.
6. Clearly separate three kinds of evidence in all docs: (a) synthetic-data ML results, (b) real xv6 feature traces, (c) real xv6 scheduling experiments.
7. The user must be able to explain every kernel change in a viva. For every kernel modification, add an entry to `DESIGN_NOTES.md` stating what changed, why, which locks are held, and what could go wrong. Keep the kernel diff small and readable.
8. After each phase: compile, boot QEMU, run a test, fix errors, then update `PROJECT_PROGRESS.md`. Never accumulate untested kernel changes.
9. Use `CPUS=1` for all main experiments (reproducibility). Verify the kernel boots and runs the smoke tests with `CPUS=3` or 4 as well, and report SMP as "boot-and-smoke-tested only" unless you test it more.

---

## 2. Scheduler modes (compile-time, single code base)

Select with `make SCHED_MODE=<n>` (or equivalent). Do not duplicate the scheduler; share MLFQ machinery and branch only on initial-queue policy.

| Mode | Name | Behavior |
|---|---|---|
| 0 | RR | Original xv6 round robin (baseline, kept as close to the original as possible) |
| 1 | MLFQ | Conventional 3-queue MLFQ; every new process starts in Q0 |
| 2 | ML-PROB | Observation phase, ML inference, initial queue sampled from probabilities (the proposed method) |
| 3 | ML-DET | Same as ML-PROB, but initial queue = argmax of the combined queue distribution (deterministic ablation) |
| 4 | RAND-CTRL | Same observation phase, but queue sampled from a fixed prior distribution that ignores the classifier (control: shows whether the classifier information matters, or just randomization) |

The prior for mode 4 should equal the average queue distribution produced by the classifier over the training workload mix, so the marginal is comparable.

Shared settings, all configurable via `#define` in one header (`kernel/sched_config.h`):
- Queue quanta: Q0=1, Q1=2, Q2=4 ticks
- Aging threshold: 10 runnable-waiting ticks
- Boost interval: 100 ticks
- Observation: min 3 bursts, max observation window (ticks) and max bursts, so observation always ends
- Class-to-queue probability table (default values below)
- Base RNG seed

Default class-to-queue table:

| Class | P(Q0) | P(Q1) | P(Q2) |
|---|---:|---:|---:|
| Interactive | 0.75 | 0.20 | 0.05 |
| I/O-bound | 0.55 | 0.35 | 0.10 |
| CPU-bound | 0.05 | 0.25 | 0.70 |
| Mixed | 0.25 | 0.50 | 0.25 |

Combined queue distribution: `P(Qj) = sum_c P(class=c) * Table[c][j]`, then normalized.

---

## 3. Phase 1: Kernel foundation (highest priority)

First inspect the actual repo version: `proc.c`, `proc.h`, `trap.c`, `sysproc.c`, `syscall.*`, `defs.h`, `usys.pl`, `start.c`, `riscv.h`, `Makefile`. Note which syscalls exist (e.g. `sleep` vs `pause`), since "voluntary yield" must be built from what exists.

Tasks:
1. Verify baseline `make qemu` boots and `usertests` basics pass. Save baseline output.
2. Extend `struct proc` with scheduling metadata (all protected by `p->lock`): `priority`, `quantum_used`, `cur_burst_ticks`, `total_cpu_ticks`, `waiting_ticks` (runnable-waiting only, never sleeping), `dispatch_count`, `queue_transitions`, `quantum_exhaustions`, `blocking_count`, `voluntary_yields`, `completed_bursts`, `arrival_tick`, `first_run_tick`, `exit_tick`, `total_runnable_wait`, plus ML fields (`obs_done`, `pred_class`, `pred_conf`, `init_queue`, `infer_status`, `rng_state`).
3. Implement MLFQ in `scheduler()`: pick the highest-priority RUNNABLE process; round robin within a queue (use a rotating scan or an enqueue order stamp; document the choice); per-queue quantum; preemption on quantum expiry.
4. CPU accounting: prefer a robust method. Record the tick at dispatch, and on return from `swtch` in the scheduler compute elapsed ticks, rather than incrementing in several trap paths (avoids double counting). If you also increment in the timer path, make sure it is counted once.
5. Distinguish preemption from voluntary yielding. Set a flag when the timer interrupt causes `yield()`; voluntary blocking comes from `sleep()` and voluntary yield from a user-initiated sleep or yield path. Quantum exhaustion means demote one level (Q2 stays Q2). A voluntary block or yield before exhaustion keeps priority. A burst ends when the process stops running for any reason.
6. Wakeup preserves priority.
7. Aging: once per global tick, for each RUNNABLE process increment `waiting_ticks` (reset it when the process is dispatched); at the threshold promote one level and reset the counter. Sleeping time never counts.
8. Periodic boost: every `BOOST_INTERVAL` ticks, move RUNNABLE processes to Q0, reset waiting counters, and keep cumulative stats. Be careful with the currently running process (adjust it safely or on its next scheduling point).
9. Locking: follow xv6 rules (acquire `p->lock` before touching scheduling fields; never hold `tickslock` and `p->lock` in an order that can deadlock; document the order you use). Do not hold `p->lock` during `copyout`.
10. Timestamps: set `arrival_tick` in `allocproc`/`fork`, `first_run_tick` at the first dispatch, `exit_tick` in `exit`.
11. Add `SCHED_LOG` compile flag: at process exit, the kernel prints one machine-parseable line, e.g. `SCHEDLOG pid=.. name=.. arrival=.. first_run=.. exit=.. cpu=.. wait=.. init_q=.. pred=.. conf=.. transitions=..`. The host harness parses these from the QEMU serial output. This avoids losing stats when the process is reaped.
12. Timer resolution: xv6's default tick (~100 ms in QEMU) makes experiments slow and burst lengths very coarse. Allow a compile-time reduced timer interval (document it, e.g. 10 ms) in `start.c`. For feature measurement, also consider `r_time()` (the time CSR) for finer burst timing. Decide, document, and make the synthetic generator match whatever resolution the kernel actually uses (see Phase 2).

Syscalls and tools (also in this phase):
- `getschedstats(struct schedstat *buf, int max)` with a fixed-layout struct (PID, state, priority, predicted class, CPU ticks, waiting ticks, dispatches, transitions, quantum exhaustions, confidence). Validate user pointer and `max`, build the snapshot into a kernel buffer under locks, release locks, then `copyout`. No kernel pointers exposed. Register it correctly in `syscall.h`, `syscall.c`, `sysproc.c`, `user.h`, `usys.pl`.
- `setschedseed(int)` (or equivalent) so the test harness can control the RNG seed per trial.
- `user/schedmon.c`: readable table of current stats.

Gate: all modes 0 and 1 boot, run `usertests` (at least the quick/basic subset; note anything that fails and compare against baseline failures), and `schedmon` shows sensible demotion for a CPU hog. Write `docs/testing.md` entries as you go.

---

## 4. Phase 2: Feature definition and ML pipeline

### 4.1 Feature definition comes from the kernel, not the other way round

Define the feature set around what the kernel can compute with integers over the observation window, at the resolution that actually exists. Suggested features (adjust if something is unreliable):

1. Average CPU burst length
2. Maximum CPU burst length
3. Burst length variability (integer variance or mean absolute deviation)
4. Fraction of bursts that ended by quantum exhaustion
5. Blocking events per observed tick
6. Voluntary yield events per observed tick
7. Blocked/sleep time ratio
8. CPU utilization over the observation window
9. Number of completed bursts

Document each as: definition, unit, integer scaling used in the kernel. Any feature that cannot be measured reliably in the kernel is replaced with a simpler one.

### 4.2 Synthetic generator: `ml/generate_data.py`

- Four classes: Interactive, I/O-bound, CPU-bound, Mixed, each generated from a behavioral blueprint (burst/block/sleep sequences), not independent random columns.
- Simulate the burst/sleep sequence at the same time quantization and quantum rules the kernel uses, then compute the features exactly as the kernel will (same integer formulas, same scaling). This is the main defense against train/deploy mismatch.
- Vary per blueprint: burst durations, blocking counts, wait durations, yield frequency, number of bursts, observation window length.
- Each sample carries a `blueprint_id` / parameter-config ID. Add noise and deliberately borderline/overlapping samples so classes are not trivially separable.
- At least 5,000 samples. Save `ml/dataset.csv` and `ml/generation_config.json` (config + seed).

### 4.3 Real xv6 traces (key integrity feature)

Add a kernel debug hook (compile flag) that prints the feature vector when observation completes, plus the true workload type known by the test program (passed by name or arg). Run your actual `cpuwork`, `iowork`, `interactive`, `mixedwork` programs many times (varied parameters and seeds) and save `ml/real_traces.csv`. Use this to:
- Report how a synthetic-trained model performs on real xv6 traces (a separate, honest "sim-to-real" number).
- Optionally train a second variant on synthetic plus a portion of real traces, with the split by run so no leakage. If used, state clearly which model each experiment used. Keep the main model synthetic-trained unless the sim-to-real gap is bad; report both.

### 4.4 Training: `ml/train_model.py`, `ml/evaluate_model.py`

- Use NumPy (easy exact weight export) or PyTorch. Network: input, 16 ReLU, 8 ReLU, 4 softmax. Normalization with stored parameters. Adam or SGD, cross-entropy, validation monitoring, early stopping.
- Splits: train/val/test must hold out entire blueprint configurations or parameter ranges (group split by `blueprint_id`), not random rows. Report this explicitly.
- Compare against simple baselines for context (majority class, logistic regression or a small decision tree). If a simpler model performs equally well, say so and justify using the MLP (e.g. required for probabilistic outputs and the stated architecture).
- Report accuracy, per-class precision/recall/F1, confusion matrix, train/val loss curves, test performance, and calibration notes (are confidences meaningful? this matters because the probabilities drive sampling).
- Save `ml/model_metrics.json` and the training config.

### 4.5 Export: `ml/model_weights.h`

- Generated by script from the trained weights (never hand-written): fixed-point integers (e.g. Q16.16 or Q8.8, choose by checking accuracy loss), normalization constants as integers, layer shapes.
- Write a Python fixed-point reference implementation that mimics the kernel integer arithmetic exactly (same rounding, saturation, exp approximation). Report float-vs-fixed agreement on the test set.

---

## 5. Phase 3: Kernel inference and probabilistic assignment

1. `kernel/ml_scheduler.c` (+ declarations in `defs.h`): integer-only forward pass, ReLU, and softmax. For softmax: subtract the max logit, then use a small lookup table or piecewise approximation for exp over a clamped range; output probabilities in fixed point summing to ~1 (renormalize). Watch for overflow (use 64-bit accumulators, saturate, document ranges).
2. Observation phase: new processes start in Q1 (provisional). After `OBS_MIN_BURSTS` completed bursts (or the max window or max bursts, whichever comes first), compute features, run inference, build the combined queue distribution, then act per mode (sample / argmax / prior sample). Then continue with ordinary MLFQ feedback.
3. Short-lived processes: if the process exits before observation completes, nothing is assigned. Record this. Report the fraction of processes that were actually classified in each experiment, since it directly limits what the method can do.
4. Insufficient data or invalid inference: fall back to conventional MLFQ behavior (start in Q0 behavior or stay in Q1 as documented) and set `infer_status` accordingly.
5. RNG: small deterministic per-process generator (e.g. xorshift32) seeded from `pid ^ global_seed`. No shared global RNG state without a lock. Seed is settable for reproducibility.
6. Record per process: class probabilities (or at least argmax and confidence), the sampled queue, whether observation was sufficient, and inference status. Include these in `SCHEDLOG`.
7. Kernel-vs-Python validation: add a boot-time or syscall-triggered self-test that runs the integer inference on a fixed list of 20+ input vectors and prints outputs. A host script compares these to the Python fixed-point reference. Require an exact match (or document tiny, justified differences). Save the comparison under `results/`.

Gate: the validation passes; in modes 2/3/4, `schedmon` shows predicted class, confidence, and initial queue for running workloads; the kernel never panics across repeated runs.

---

## 6. Phase 4: Workloads and experiment harness

User programs (deterministic given parameters/seed, long enough to span many scheduling decisions): `cpuwork.c`, `iowork.c`, `interactive.c`, `mixedwork.c`, `schedtest.c`.
- `schedtest <experiment> <seed>` forks the workload mix, waits for all, and prints a summary. Per-process timestamps come from `SCHEDLOG`.
- Host script `scripts/run_experiments.sh` (or Python): for each mode, build with that `SCHED_MODE`, run QEMU headless (`-nographic`, with a timeout) over the chosen experiments and seeds, and save the raw serial output to `results/raw/<mode>_<exp>_<seed>.log`. Detect and record timeouts or panics as failed trials; do not silently drop them. Report them.
- `ml/analyze_results.py`: parse raw logs and compute turnaround, waiting, response time, throughput, CPU utilization, priority transitions, starvation cases (a process that waits beyond a defined multiple of the aging threshold, define it explicitly), and the fraction of classified processes. Output `results/experiment_results.csv` (per-process rows plus per-trial summaries).

Definitions (ticks): turnaround = exit - arrival; response = first_run - arrival; waiting = total runnable-but-not-running time; throughput = completed processes / measurement interval; utilization = busy CPU ticks / elapsed ticks.

Experiments (minimum):
1. Multiple CPU-bound processes
2. Multiple interactive processes
3. Mixed CPU-bound plus interactive
4. I/O-heavy processes
5. Mixed workload, high process count
6. Starvation test: a long CPU-bound process competing with a continuous stream of short or interactive processes; verify the aging/boost keep it making progress, and compare with aging/boost disabled to show they matter.

Run all 5 modes, at least 5 seeds each (10 if cheap), identical seeds across modes. Report mean, standard deviation, and 95% confidence intervals; use paired differences by seed where appropriate. Include all trials, including bad ones. Generate `turnaround_comparison.png`, `waiting_time_comparison.png`, `response_time_comparison.png` (with error bars) and one `comparison.png` overview, from real data only.

Scheduler overhead: measure the cost of inference (cycles via `r_time` around the inference call, or tick-level effect) and report it honestly.

---

## 7. Phase 5: Small policy study (only after everything above works)

`ml/optimize_policy.py` is optional and kept small. Compare the default class-to-queue table with a handful (about 6 to 12) of alternative tables and 2 to 3 observation-window settings, using real xv6 runs. Use separate seeds for tuning and for final evaluation to avoid overfitting. Constraints: no starvation, acceptable interactive response, CPU-bound progress, reasonable overhead. Keep the original policy results next to the tuned ones. Never change the classifier here; the classifier and the policy are separate things. If time or stability is limited, skip this and say so.

---

## 8. Phase 6: Documentation and packaging

Files:
- `README.md`: overview, build/run commands for every mode, how to reproduce training, experiments, and plots.
- `docs/project_report.md`: 8 to 12 pages in substance. Required content: abstract, introduction and problem statement, MLFQ background, architecture diagram, workload generation and features, model and training, held-out evaluation, sim-to-real evaluation, probabilistic mapping table, kernel integration (with locking notes), starvation prevention, experimental methodology, results and analysis for all five modes, limitations, future work, conclusion, references. Briefer treatment is fine for background sections; spend the length on methods, results, and honest analysis.
- The report must include: the research question, the ablation rationale (why modes 3 and 4 exist), the fraction of processes classified, the overhead, the failed or inconclusive outcomes, and a limitations section covering synthetic training data, small scale, single-CPU experiments, coarse tick resolution, and the user-space workloads being hand-written.
- `docs/architecture.md`, `docs/testing.md`: concise.
- `DESIGN_NOTES.md`: per-change rationale and locking/overflow reasoning (viva material).
- `VIVA_PREP.md`: likely examiner questions with short answers grounded in this code (why probabilistic sampling, what happens if the process exits before observation, how voluntary yield vs preemption is detected, locking order, fixed-point overflow handling, why held-out blueprints, why mode 4, what the results mean, what you would do differently).
- `PROJECT_PROGRESS.md`: kept current throughout.
- `PM-MLFQ-source.zip`: repo sources, `ml/`, `scripts/`, `docs/`, `results/` (including `results/raw/` logs). Exclude `.o`, `.d`, `.asm`, `.sym`, `fs.img`, `kernel/kernel`, `__pycache__`, and other build artifacts. Confirm the zip builds from scratch: unzip into a temp dir and run `make` for at least modes 0, 1, and 2.

The report and README mention the GitHub link as `<GITHUB_LINK_PLACEHOLDER>`; the user fills it in.

---

## 9. Priority and cut order (if something proves too hard)

Never cut: working MLFQ with accounting, trained and exported model, kernel inference with the Python-vs-kernel check, probabilistic assignment, aging and boost, modes 0 to 4, multi-seed experiments with honest reporting, report, ZIP.

Cut in this order: (1) `optimize_policy.py`, (2) extra plots, (3) experiment 5's size, (4) `architecture.md` detail, (5) SMP testing beyond boot/smoke, (6) a real-trace fine-tuned model variant (keep the real-trace evaluation).

If the 16-8-4 network is too heavy for the kernel, shrink the hidden layers rather than faking predictions, and say so.

---

## 10. Final completion checklist

Check each item only if verified by an actual run:

- [ ] Original RR baseline preserved (mode 0)
- [ ] Conventional MLFQ (mode 1): 3 queues, quanta 1/2/4, demotion, accounting
- [ ] Aging and boost verified by a starvation test (with and without)
- [ ] Synthetic dataset with config and seed saved
- [ ] Model trained with gradient-based optimization, held-out blueprint evaluation done
- [ ] Real xv6 feature traces collected and sim-to-real evaluation reported
- [ ] `model_weights.h` generated from the trained model
- [ ] Kernel inference matches the Python fixed-point reference on identical inputs
- [ ] Modes 2, 3, 4 work; classified-process fraction measured
- [ ] `getschedstats` and `schedmon` work
- [ ] Workload programs and `schedtest` work
- [ ] All experiments run for all modes with multiple seeds; failures reported
- [ ] `results/experiment_results.csv`, raw logs, and plots generated from real data
- [ ] Report, README, DESIGN_NOTES, VIVA_PREP written
- [ ] ZIP built and verified to compile from scratch
- [ ] Final QEMU demonstration verified

## 11. Final summary the agent must produce

1. What was implemented and what was not
2. Whether the kernel builds and QEMU boots (each mode)
3. Whether the model was genuinely trained, and its held-out and real-trace performance
4. Whether kernel inference matches Python
5. Actual experimental results: for each experiment, which mode won on turnaround, waiting, and response, with variability, and whether PM-MLFQ beat modes 1, 3, and 4 (state plainly if it did not)
6. Known limitations and bugs
7. Exact demo commands
8. Paths of the report and ZIP
