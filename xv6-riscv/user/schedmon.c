#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"
#include "kernel/schedstat.h"

static char *states[] = {
  [0] "UNUSED  ",
  [1] "USED    ",
  [2] "SLEEPING",
  [3] "RUNNABLE",
  [4] "RUNNING ",
  [5] "ZOMBIE  "
};

static char *classes[] = {
  "INTERACT",
  "IO_BOUND",
  "CPU_BND ",
  "MIXED   "
};

static struct schedstat stats[64];

int
main(int argc, char *argv[])
{
  int n = getschedstats(stats, 64);

  if (n < 0) {
    printf("schedmon: getschedstats failed\n");
    exit(1);
  }

  printf("=== PM-MLFQ Process Scheduling Monitor (%d active procs) ===\n", n);
  printf("PID  NAME             STATE    Q  PRED_CLS CONF  CPU_TK WAIT_TK DISP TRANS EXHAUST\n");
  printf("---- ---------------- -------- -  -------- ----  ------ ------- ---- ----- -------\n");

  for (int i = 0; i < n; i++) {
    char *st = (stats[i].state >= 0 && stats[i].state <= 5) ? states[stats[i].state] : "UNKNOWN ";
    char *cls = "NONE    ";
    if (stats[i].pred_class >= 0 && stats[i].pred_class < 4) {
      cls = classes[stats[i].pred_class];
    }

    printf("%d    %s", stats[i].pid, stats[i].name);
    // align name
    int len = strlen(stats[i].name);
    for (int k = len; k < 17; k++)
      printf(" ");

    printf("%s %d  %s %d   %ld      %ld      %ld    %ld     %ld\n",
           st,
           stats[i].priority,
           cls,
           stats[i].confidence,
           stats[i].cpu_ticks,
           stats[i].waiting_ticks,
           stats[i].dispatches,
           stats[i].transitions,
           stats[i].quantum_exhaustions);
  }

  exit(0);
}
