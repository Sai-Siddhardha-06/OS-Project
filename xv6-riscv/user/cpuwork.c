#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

int
main(int argc, char *argv[])
{
  int iters = 50000000;
  if (argc > 1) {
    iters = atoi(argv[1]);
  }

  volatile unsigned long long x = 0;
  for (int i = 0; i < iters; i++) {
    x += (i * 3 + 1);
  }

  // Prevent compiler optimizing out
  if (x == 42)
    printf("rare\n");

  exit(0);
}
