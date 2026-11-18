#ifndef _SCHED_CONFIG_H_
#define _SCHED_CONFIG_H_

// Scheduler Modes
#define SCHED_RR        0  // Vanilla round-robin
#define SCHED_MLFQ      1  // Standard 3-queue MLFQ (starts at Q0)
#define SCHED_ML_PROB   2  // Proposed: Observe -> ML inference -> Sample queue
#define SCHED_ML_DET    3  // Ablation: ML inference -> Argmax queue
#define SCHED_RAND_CTRL 4  // Ablation: Prior distribution random sample (no ML)

#ifndef SCHED_MODE
#define SCHED_MODE SCHED_RR
#endif

// Number of MLFQ queues
#define MLFQ_QUEUES 3

// Queue time slices (ticks)
#define QUANTUM_Q0 1
#define QUANTUM_Q1 2
#define QUANTUM_Q2 4

// Aging threshold: runnable ticks before being promoted to higher queue
#define AGING_THRESHOLD 10

// Global priority boost interval (ticks): all runnable reset to Q0
#define BOOST_INTERVAL 100

// Observation window parameters
#define OBS_MIN_BURSTS       3   // Minimum completed bursts before inference
#define OBS_MAX_WINDOW_TICKS 50  // Maximum ticks in observation window
#define OBS_MAX_BURSTS       10  // Maximum bursts stored in observation history

// Workload classes
#define CLASS_INTERACTIVE 0
#define CLASS_IO          1
#define CLASS_CPU         2
#define CLASS_MIXED       3
#define NUM_CLASSES       4

// Probability scaling factor (1000 = 100.0%)
#define PROB_SCALE 1000

// Default Base RNG Seed
#define BASE_RNG_SEED 0x6d2b79f5

// Hardware timer interval in RISC-V cycles (1,000,000 cycles = ~100ms in QEMU virt)
// Can be overridden via -DTIMER_INTERVAL=...
#ifndef TIMER_INTERVAL
#define TIMER_INTERVAL 1000000
#endif

// Per-process memory limit in bytes (0 = disabled)
#ifndef PROC_MEM_LIMIT
#define PROC_MEM_LIMIT 0
#endif

#endif
