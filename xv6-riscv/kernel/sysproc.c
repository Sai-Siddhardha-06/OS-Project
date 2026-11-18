#include "types.h"
#include "riscv.h"
#include "defs.h"
#include "param.h"
#include "memlayout.h"
#include "spinlock.h"
#include "proc.h"
#include "vm.h"
#include "schedstat.h"

uint64
sys_exit(void)
{
  int n;
  argint(0, &n);
  kexit(n);
  return 0; // not reached
}

uint64
sys_getpid(void)
{
  return myproc()->pid;
}

uint64
sys_fork(void)
{
  return kfork();
}

uint64
sys_wait(void)
{
  uint64 p;
  argaddr(0, &p);
  return kwait(p);
}

uint64
sys_sbrk(void)
{
  uint64 addr;
  int t;
  int n;

  argint(0, &n);
  argint(1, &t);
  addr = myproc()->sz;

  if (t == SBRK_EAGER || n < 0) {
    if (growproc(n) < 0) {
      return -1;
    }
  } else {
    // Lazily allocate memory for this process: increase its memory
    // size but don't allocate memory. If the processes uses the
    // memory, vmfault() will allocate it.
    if (addr + n < addr)
      return -1;
    if (addr + n > TRAPFRAME)
      return -1;
    myproc()->sz += n;
  }
  return addr;
}

uint64
sys_pause(void)
{
  int n;
  uint ticks0;

  argint(0, &n);
  if (n < 0)
    n = 0;
  acquire(&tickslock);
  ticks0 = ticks;
  while (ticks - ticks0 < n) {
    if (killed(myproc())) {
      release(&tickslock);
      return -1;
    }
    sleep_prepare(&ticks);
    release(&tickslock);
    sleep();
    acquire(&tickslock);
  }
  release(&tickslock);
  return 0;
}

uint64
sys_kill(void)
{
  int pid;

  argint(0, &pid);
  return kkill(pid);
}

// return how many clock tick interrupts have occurred
// since start.
uint64
sys_uptime(void)
{
  uint xticks;

  acquire(&tickslock);
  xticks = ticks;
  release(&tickslock);
  return xticks;
}

uint64
sys_getmemsize(void)
{
  return myproc()->sz;
}

uint64
sys_getschedstats(void)
{
  uint64 ubuf;
  int max;
  extern struct proc proc[NPROC];

  argaddr(0, &ubuf);
  argint(1, &max);

  if (ubuf == 0 || max <= 0)
    return -1;
  if (max > NPROC)
    max = NPROC;

  int count = 0;
  struct schedstat s;

  for (struct proc *p = proc; p < &proc[NPROC] && count < max; p++) {
    acquire(&p->lock);
    if (p->state != UNUSED) {
      s.pid = p->pid;
      s.state = p->state;
      s.priority = p->priority;
      s.pred_class = p->pred_class;
      s.confidence = p->pred_conf;
      s.init_queue = p->init_queue;
      s.cpu_ticks = p->total_cpu_ticks;
      s.waiting_ticks = p->waiting_ticks;
      s.total_wait_ticks = p->total_runnable_wait;
      s.dispatches = p->dispatch_count;
      s.transitions = p->queue_transitions;
      s.quantum_exhaustions = p->quantum_exhaustions;
      safestrcpy(s.name, p->name, sizeof(s.name));
      release(&p->lock);

      if (either_copyout(1, ubuf + (uint64)count * sizeof(struct schedstat), (void *)&s, sizeof(struct schedstat)) < 0) {
        return -1;
      }
      count++;
    } else {
      release(&p->lock);
    }
  }
  return count;
}

uint64
sys_setschedseed(void)
{
  int seed;
  argint(0, &seed);
  set_global_sched_seed((uint32)seed);
  return 0;
}

uint64
sys_mlselftest(void)
{
  return ml_self_test();
}
