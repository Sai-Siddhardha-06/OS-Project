#ifndef _SCHEDSTAT_H_
#define _SCHEDSTAT_H_

#include "types.h"

// Snapshot of process scheduling statistics
struct schedstat {
  int pid;
  int state;                // UNUSED, USED, SLEEPING, RUNNABLE, RUNNING, ZOMBIE
  int priority;             // Current queue (0..2)
  int pred_class;           // -1 if unclassified, 0..3 for classes
  int confidence;           // Confidence score (0..1000)
  int init_queue;           // Initial assigned queue (-1 if not yet assigned)
  uint64 cpu_ticks;         // Total CPU ticks consumed
  uint64 waiting_ticks;     // Current waiting ticks in RUNNABLE state
  uint64 total_wait_ticks;  // Total waiting ticks over lifetime
  uint64 dispatches;        // Number of times scheduled
  uint64 transitions;       // Number of queue priority transitions
  uint64 quantum_exhaustions; // Number of times time slice was exhausted
  char name[16];            // Process name
};

#endif
