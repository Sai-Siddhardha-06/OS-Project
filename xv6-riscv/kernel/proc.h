// Saved registers for kernel context switches.
struct context {
  uint64 ra;
  uint64 sp;

  // callee-saved
  uint64 s0;
  uint64 s1;
  uint64 s2;
  uint64 s3;
  uint64 s4;
  uint64 s5;
  uint64 s6;
  uint64 s7;
  uint64 s8;
  uint64 s9;
  uint64 s10;
  uint64 s11;
};

// Per-CPU state.
struct cpu {
  struct proc *proc;      // The process running on this cpu, or null.
  struct context context; // swtch() here to enter scheduler().
  int noff;               // Depth of push_off() nesting.
  int intena;             // Were interrupts enabled before push_off()?
};

extern struct cpu cpus[NCPU];

// per-process data for the trap handling code in trampoline.S.
// sits in a page by itself just under the trampoline page in the
// user page table. not specially mapped in the kernel page table.
// uservec in trampoline.S saves user registers in the trapframe,
// then initializes registers from the trapframe's
// kernel_sp, kernel_hartid, kernel_satp, and jumps to kernel_trap.
// prepare_return() and userret in trampoline.S set up
// the trapframe's kernel_*, restore user registers from the
// trapframe, switch to the user page table, and enter user space.
struct trapframe {
  /*   0 */ uint64 kernel_satp;   // kernel page table
  /*   8 */ uint64 kernel_sp;     // top of process's kernel stack
  /*  16 */ uint64 kernel_trap;   // usertrap()
  /*  24 */ uint64 epc;           // saved user program counter
  /*  32 */ uint64 kernel_hartid; // saved kernel tp
  /*  40 */ uint64 ra;
  /*  48 */ uint64 sp;
  /*  56 */ uint64 gp;
  /*  64 */ uint64 tp;
  /*  72 */ uint64 t0;
  /*  80 */ uint64 t1;
  /*  88 */ uint64 t2;
  /*  96 */ uint64 s0;
  /* 104 */ uint64 s1;
  /* 112 */ uint64 a0;
  /* 120 */ uint64 a1;
  /* 128 */ uint64 a2;
  /* 136 */ uint64 a3;
  /* 144 */ uint64 a4;
  /* 152 */ uint64 a5;
  /* 160 */ uint64 a6;
  /* 168 */ uint64 a7;
  /* 176 */ uint64 s2;
  /* 184 */ uint64 s3;
  /* 192 */ uint64 s4;
  /* 200 */ uint64 s5;
  /* 208 */ uint64 s6;
  /* 216 */ uint64 s7;
  /* 224 */ uint64 s8;
  /* 232 */ uint64 s9;
  /* 240 */ uint64 s10;
  /* 248 */ uint64 s11;
  /* 256 */ uint64 t3;
  /* 264 */ uint64 t4;
  /* 272 */ uint64 t5;
  /* 280 */ uint64 t6;
};

#include "sched_config.h"

enum procstate { UNUSED, USED, SLEEPING, RUNNABLE, RUNNING, ZOMBIE };

// Per-process state
struct proc {
  struct spinlock lock;

  // p->lock must be held when using these:
  enum procstate state; // Process state
  void *chan;           // If non-zero, sleeping on chan
  int killed;           // If non-zero, have been killed
  int xstate;           // Exit status to be returned to parent's wait
  int pid;              // Process ID

  // Scheduling metadata (protected by p->lock)
  int priority;             // Current MLFQ queue: 0 (high), 1 (medium), 2 (low)
  uint quantum_used;        // Ticks consumed in current time slice
  uint cur_burst_ticks;     // Ticks accumulated in current burst
  uint64 total_cpu_ticks;   // Total ticks spent running
  uint waiting_ticks;       // Waiting ticks in RUNNABLE since last dispatch (aging)
  uint64 total_runnable_wait; // Cumulative runnable waiting ticks over lifetime
  uint64 dispatch_count;    // Total times scheduled onto a CPU
  uint64 queue_transitions; // Number of queue promotions/demotions
  uint64 quantum_exhaustions; // Number of times quantum was exhausted
  uint64 blocking_count;    // Number of sleep/blocking events
  uint64 voluntary_yields;  // Number of voluntary yield events
  uint64 completed_bursts;  // Number of completed bursts
  uint64 arrival_tick;      // Tick when process was created (allocproc)
  uint64 first_run_tick;    // Tick when process first scheduled
  uint64 exit_tick;         // Tick when process exited
  uint64 last_sched_tick;   // Tick when process was last dispatched
  uint64 sleep_start_tick;  // Tick when process went to sleep
  uint64 total_sleep_ticks; // Total ticks spent sleeping

  // Preemption vs voluntary flag
  int preempted;            // 1 if yielded via timer interrupt, 0 if voluntary

  // Observation and ML fields
  int obs_done;             // 1 if observation window completed
  uint64 obs_start_tick;    // Tick when observation began
  int pred_class;           // -1 if unclassified, 0..3 for classes
  int pred_conf;            // Scaled confidence (0..1000)
  int init_queue;           // Initial assigned queue after observation
  int infer_status;         // 0: none, 1: ML success, 2: short-lived, 3: fallback
  uint64 infer_cycles;      // CPU cycles for ML inference (r_time() delta)
  uint32 rng_state;         // Per-process PRNG state (xorshift32)

  // Observation burst history
  uint burst_history[OBS_MAX_BURSTS];
  int burst_exhausted[OBS_MAX_BURSTS];
  int burst_count;          // Bursts recorded so far (up to OBS_MAX_BURSTS)

  // wait_lock must be held when using this:
  struct proc *parent; // Parent process

  // these are private to the process, so p->lock need not be held.
  uint64 kstack;               // Virtual address of kernel stack
  uint64 sz;                   // Size of process memory (bytes)
  pagetable_t pagetable;       // User page table
  struct trapframe *trapframe; // data page for trampoline.S
  struct context context;      // swtch() here to run process
  struct file *ofile[NOFILE];  // Open files
  struct inode *cwd;           // Current directory
  char name[16];               // Process name (debugging)
};
