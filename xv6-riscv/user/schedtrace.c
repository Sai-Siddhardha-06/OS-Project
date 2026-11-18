#include "kernel/types.h"
#include "kernel/trace.h"
#include "user/user.h"

static const char *event_names[] = {
  "PROC_NEW",
  "CLASSIFIED",
  "DEMOTE",
  "PROMOTE_AGE",
  "BOOST",
  "PROC_EXIT",
  "SYSCALL",
};
#define NUM_EVENTS (sizeof(event_names)/sizeof(event_names[0]))

int
main(void)
{
  struct traceent buf[TRACE_ENTRIES];
  int n;

  n = gettrace(buf, TRACE_ENTRIES);
  if (n < 0) {
    printf("schedtrace: gettrace failed (kernel built without SCHED_TRACE?)\n");
    exit(1);
  }
  if (n == 0) {
    printf("schedtrace: no entries\n");
    exit(0);
  }

  printf("TICK   CPU  PID    EVENT          ARG1   ARG2\n");
  printf("------ ---- ------ -------------- ------ ------\n");

  for (int i = 0; i < n; i++) {
    struct traceent *e = &buf[i];
    const char *name = (e->event < NUM_EVENTS) ? event_names[e->event] : "?UNKNOWN";
    printf("%d\t%d\t%d\t%s\t%d\t%d\n",
           e->tick, (int)e->cpu, e->pid, name, e->arg1, e->arg2);
  }
  printf("\n%d entries\n", n);
  exit(0);
}
