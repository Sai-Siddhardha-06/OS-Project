// kernel/trace.h - Scheduler event trace ring buffer
// Controlled by SCHED_TRACE compile flag (default OFF)

#ifndef TRACE_H
#define TRACE_H

#define TRACE_ENTRIES    128   // ring buffer capacity; must be power of 2
#define TRACE_PANIC_DUMP  16   // entries to dump on panic

// Event type codes
#define TRACE_PROC_NEW      0  // process allocated
#define TRACE_CLASSIFIED    1  // ML classification complete (arg1=pred_class, arg2=sampled_q)
#define TRACE_DEMOTE        2  // queue demotion (arg1=old_q, arg2=new_q)
#define TRACE_PROMOTE_AGING 3  // aging promotion (arg1=old_q, arg2=new_q)
#define TRACE_BOOST         4  // periodic boost (arg1=old_q, arg2=0)
#define TRACE_PROC_EXIT     5  // process exited
#define TRACE_SYSCALL       6  // syscall entry (arg1=syscall number)

struct traceent {
  uint32 tick;    // kernel tick at event time
  uint8  cpu;     // CPU id (best-effort)
  int    pid;     // process id
  uint8  event;   // TRACE_* constant above
  int    arg1;    // event-specific argument 1
  int    arg2;    // event-specific argument 2
};

#endif // TRACE_H
