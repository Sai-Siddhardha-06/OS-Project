// kernel/trace.c - Scheduler event trace ring buffer (flight recorder)
// Compile with SCHED_TRACE=1 to enable; default is OFF.
//
// LOCK ORDER: trace_lock is a LEAF lock. It may be acquired while holding
// p->lock or tickslock, but nothing else is acquired while it is held.
// trace_log() must never call printk, printf, or sleep.

#include "types.h"
#include "param.h"
#include "spinlock.h"
#include "riscv.h"
#include "defs.h"
#include "proc.h"
#include "trace.h"

#ifdef SCHED_TRACE

static struct traceent trace_buf[TRACE_ENTRIES];
static int trace_head = 0;        // index of NEXT write slot (monotonic counter)
static struct spinlock trace_lock;

void
trace_init(void)
{
  initlock(&trace_lock, "trace");
  for (int i = 0; i < TRACE_ENTRIES; i++) {
    trace_buf[i].event = 0xff; // mark empty
    trace_buf[i].pid   = -1;
  }
}

// Log one event. Leaf lock: may be called with p->lock or tickslock held.
// Must NOT call printk or sleep.
void
trace_log(uint8 event, int pid, int arg1, int arg2)
{
  acquire(&trace_lock);
  int slot = trace_head % TRACE_ENTRIES;
  trace_buf[slot].tick  = (uint32)get_ticks();
  trace_buf[slot].cpu   = (uint8)cpuid();
  trace_buf[slot].pid   = pid;
  trace_buf[slot].event = event;
  trace_buf[slot].arg1  = arg1;
  trace_buf[slot].arg2  = arg2;
  trace_head++;
  release(&trace_lock);
}

// Copy up to 'max' entries oldest-first into 'out' (kernel buffer).
// Returns the number of entries copied.
// Caller must NOT hold trace_lock.
int
trace_snapshot(struct traceent *out, int max)
{
  acquire(&trace_lock);
  int total = trace_head;         // total ever written
  int count = (total < TRACE_ENTRIES) ? total : TRACE_ENTRIES;
  if (count > max)
    count = max;

  // oldest slot: if buffer not yet full, it's slot 0;
  // if full, it's trace_head % TRACE_ENTRIES (the next-to-be-overwritten slot)
  int start;
  if (total < TRACE_ENTRIES)
    start = 0;
  else
    start = trace_head % TRACE_ENTRIES; // oldest

  for (int i = 0; i < count; i++) {
    out[i] = trace_buf[(start + i) % TRACE_ENTRIES];
  }
  release(&trace_lock);
  return count;
}

// Called from panic() ONLY. Lock-free best-effort read of last 16 entries.
// May produce slightly torn reads if another CPU is writing concurrently.
// Documented: output may be inconsistent. Do NOT take trace_lock here
// (we may be panicking while holding it, or other CPUs may be spinning).
void
trace_panic_dump(void)
{
  int head = trace_head;
  int count = (head < TRACE_ENTRIES) ? head : TRACE_ENTRIES;
  int dump = (count < TRACE_PANIC_DUMP) ? count : TRACE_PANIC_DUMP;

  printk("--- SCHED_TRACE panic dump (last %d entries, may be torn) ---\n", dump);
  int start = head - dump;
  for (int i = 0; i < dump; i++) {
    int slot = (start + i) % TRACE_ENTRIES;
    struct traceent *e = &trace_buf[slot];
    printk("  [%d] tick=%d cpu=%d pid=%d evt=%d arg1=%d arg2=%d\n",
           i, (int)e->tick, (int)e->cpu, e->pid,
           (int)e->event, e->arg1, e->arg2);
  }
  printk("--- end trace dump ---\n");
}

#else // !SCHED_TRACE

void trace_init(void) {}
void trace_log(uint8 event, int pid, int arg1, int arg2) {
  (void)event; (void)pid; (void)arg1; (void)arg2;
}
int  trace_snapshot(struct traceent *out, int max) {
  (void)out; (void)max; return 0;
}
void trace_panic_dump(void) {}

#endif // SCHED_TRACE
