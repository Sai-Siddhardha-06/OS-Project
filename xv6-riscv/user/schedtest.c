#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

// Forward declaration of syscalls
extern int setschedseed(int);

void
run_child(const char *prog, char **argv)
{
  int pid = fork();
  if (pid < 0) {
    printf("schedtest: fork failed\n");
    exit(1);
  }
  if (pid == 0) {
    exec(prog, argv);
    printf("schedtest: exec %s failed\n", prog);
    exit(1);
  }
}

void
wait_all(int n)
{
  for (int i = 0; i < n; i++) {
    int status;
    wait(&status);
  }
}

int
main(int argc, char *argv[])
{
  if (argc < 3) {
    printf("Usage: schedtest <exp_id 1-6> <seed>\n");
    exit(1);
  }

  int exp_id = atoi(argv[1]);
  int seed = atoi(argv[2]);

  setschedseed(seed);
  printf("SCHEDTEST_START exp=%d seed=%d\n", exp_id, seed);

  if (exp_id == 1) {
    // Exp 1: Multiple CPU-bound processes (4 CPU hogs)
    char *arg1[] = {"cpuwork", "120000000", 0};
    char *arg2[] = {"cpuwork", "150000000", 0};
    char *arg3[] = {"cpuwork", "100000000", 0};
    char *arg4[] = {"cpuwork", "130000000", 0};
    run_child("cpuwork", arg1);
    run_child("cpuwork", arg2);
    run_child("cpuwork", arg3);
    run_child("cpuwork", arg4);
    wait_all(4);

  } else if (exp_id == 2) {
    // Exp 2: Multiple Interactive processes (4 interactive procs)
    char *arg1[] = {"interactive", "10", 0};
    char *arg2[] = {"interactive", "12", 0};
    char *arg3[] = {"interactive", "8", 0};
    char *arg4[] = {"interactive", "10", 0};
    run_child("interactive", arg1);
    run_child("interactive", arg2);
    run_child("interactive", arg3);
    run_child("interactive", arg4);
    wait_all(4);

  } else if (exp_id == 3) {
    // Exp 3: Mixed CPU-bound plus Interactive (2 CPU, 2 Interactive)
    char *c1[] = {"cpuwork", "140000000", 0};
    char *c2[] = {"cpuwork", "120000000", 0};
    char *i1[] = {"interactive", "10", 0};
    char *i2[] = {"interactive", "12", 0};
    run_child("cpuwork", c1);
    run_child("interactive", i1);
    run_child("cpuwork", c2);
    run_child("interactive", i2);
    wait_all(4);

  } else if (exp_id == 4) {
    // Exp 4: I/O-heavy processes (4 I/O bound procs)
    char *arg1[] = {"iowork", "20", 0};
    char *arg2[] = {"iowork", "25", 0};
    char *arg3[] = {"iowork", "15", 0};
    char *arg4[] = {"iowork", "20", 0};
    run_child("iowork", arg1);
    run_child("iowork", arg2);
    run_child("iowork", arg3);
    run_child("iowork", arg4);
    wait_all(4);

  } else if (exp_id == 5) {
    // Exp 5: Mixed workload, high process count (2 CPU, 2 IO, 2 Interactive, 2 Mixed = 8 procs)
    char *c1[] = {"cpuwork", "100000000", 0};
    char *c2[] = {"cpuwork", "120000000", 0};
    char *io1[] = {"iowork", "15", 0};
    char *io2[] = {"iowork", "18", 0};
    char *in1[] = {"interactive", "8", 0};
    char *in2[] = {"interactive", "10", 0};
    char *m1[] = {"mixedwork", "6", 0};
    char *m2[] = {"mixedwork", "8", 0};
    run_child("cpuwork", c1);
    run_child("iowork", io1);
    run_child("interactive", in1);
    run_child("mixedwork", m1);
    run_child("cpuwork", c2);
    run_child("iowork", io2);
    run_child("interactive", in2);
    run_child("mixedwork", m2);
    wait_all(8);

  } else if (exp_id == 6) {
    // Exp 6: Starvation test: 1 long CPU hog competing with stream of short interactive procs
    char *hog[] = {"cpuwork", "220000000", 0};
    run_child("cpuwork", hog);

    for (int s = 0; s < 4; s++) {
      char *short_proc[] = {"interactive", "4", 0};
      run_child("interactive", short_proc);
      int status;
      wait(&status);
    }
    // Wait for the long CPU hog
    int status;
    wait(&status);

  } else {
    printf("Unknown exp_id %d\n", exp_id);
    exit(1);
  }

  printf("SCHEDTEST_DONE exp=%d seed=%d\n", exp_id, seed);
  exit(0);
}
