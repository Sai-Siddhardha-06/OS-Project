# PM-MLFQ Project Progress Log

**Last Updated**: 2026-11-18 11:20 IST  
**Repository Path**: `~/OS Project/xv6-riscv`  
**Target Hardware/OS**: RISC-V 64-bit (`virt` platform), QEMU 10.2.1, GCC 14.2.0

---

## Phase Status Summary

| Phase | Description | Status | Verification Gate |
|---|---|---|---|
| **Phase 0** | Baseline Verification | **COMPLETED & AUDITED** | Toolchain validated; baseline `usertests -q` passed |
| **Phase 1** | Kernel Foundation & MLFQ | **COMPLETED & AUDITED** | 3-queue MLFQ, aging, boost, stats syscalls, `SCHEDLOG` |
| **Phase 2** | Feature Extraction & ML Pipeline | **COMPLETED & AUDITED** | 9-feature extractor, synthetic dataset (6k samples), 95.5% test acc, 93.75% sim-to-real acc, Q16.16 export |
| **Phase 3** | Kernel Inference & Probabilistic Assignment | **COMPLETED & AUDITED** | Integer-only forward pass, 24/24 boot self-test exact match, all 5 scheduler modes functional |
| **Phase 4** | Workload Suite & Experiments | **COMPLETED & AUDITED** | 6 benchmarks, 150 trials (5 modes × 6 exps × 5 seeds), 725 process logs, analysis CSVs and comparison plots |
| **Phase 5** | Policy Study & Overhead Analysis | **COMPLETED & AUDITED** | Inference latency measured (mean 3.227 $\mu$s / 32.27 timer ticks at 10 MHz), policy sensitivity evaluated |
| **Phase 6** | Documentation & Packaging | **COMPLETED & AUDITED** | Comprehensive technical report, architecture guide, design notes, viva prep, clean verified zip package |

---

## Detailed Audit & Resolution Records

An independent, evidence-based audit was executed against `PM-MLFQ_PLAN.md` on 2026-11-18. All items were tested by fresh command execution.

### Resolved Audit Findings:
1. **Packaging Cleanliness**: Regenerated `PM-MLFQ-source.zip` (1.4 MB) excluding all `.o`, `.d`, `.asm`, `.sym`, `fs.img`, `kernel/kernel`, and `__pycache__` artifacts. Verified by extracting into `/tmp` and performing a fresh multi-mode build.
2. **Timer Semantics & Units**: Corrected references to `r_time()` in `docs/project_report.md`, `DESIGN_NOTES.md`, and `VIVA_PREP.md`. Clarified that `r_time()` reads the RISC-V `time` CSR clocked at 10 MHz, corresponding to a wall-clock latency of 3.227 $\mu$s rather than 32.27 CPU cycles.
3. **Honest Reporting of Outcomes**: Documented plainly that PM-MLFQ does not strictly outperform standard MLFQ in turnaround time on simple workloads (MLFQ is 1.8% faster in Exp 5 and 7.6% faster in Exp 1); its demonstrated advantage is avoiding the 8.4% herd-collapse penalty suffered by deterministic ML (`ML-DET`).
4. **Comparative Starvation Test**: Added the `DISABLE_AGING_BOOST` compilation toggle and evaluated Exp 6 with aging enabled vs disabled. With aging on, the CPU hog completed in 9 ticks; with aging off, it was starved by incoming interactive tasks and required 27 ticks.
5. **Root Documentation**: Created repository root `README.md` with full reproduction instructions. Added the repository placeholder `https://github.com/<user>/<repo>` to `docs/project_report.md`.

6. **Safety Pre-Extras Backup**: Prior to implementing optional diagnostic features, backed up the verified tree to `~/backup_pre_extras/OS_Project_backup` and `~/backup_pre_extras/PM-MLFQ-source.zip`.
7. **Feature A (Scheduler Event Trace)**: Implemented 128-entry event ring buffer (`kernel/trace.c`, `kernel/trace.h`), syscall `SYS_gettrace` (`gettrace`), user program `schedtrace`. Tested under Mode 1 and Mode 2 with `SCHED_TRACE=1`: captured `PROC_NEW`, `CLASSIFIED`, `DEMOTE`, `PROMOTE_AGE`, `BOOST`, and `PROC_EXIT`. Verified ring wrap-around (>128 entries) and `trace_panic_dump()` lock-free console dump. Saved to `results/trace_demo.txt`.
8. **Feature B (Per-Process Memory Cap)**: Added `PROC_MEM_LIMIT` (bytes, default 0 = disabled) in `kernel/sched_config.h`, refusal logic in `growproc()` and `exec()`, `mem_refusals` counter in `struct proc`, and user benchmark `user/memhog.c`. Tested with `PROC_MEM_LIMIT=0` (allocated 32+ MB) and `PROC_MEM_LIMIT=4194304` (capped at 4 MB, refusal recorded in SCHEDLOG). Saved to `results/memcap_demo.txt`. Verified `usertests -q` passes with zero regressions.
