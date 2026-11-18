#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

int
main(int argc, char *argv[])
{
  int cycles = 20;
  if (argc > 1) {
    cycles = atoi(argv[1]);
  }

  volatile unsigned long long count = 0;
  for (int c = 0; c < cycles; c++) {
    // Moderate CPU burst (~1-2 ticks in QEMU)
    for (int k = 0; k < 60000000; k++) {
      count += (k ^ c);
    }
    // Moderate I/O or sleep
    pause(1);
  }

  if (count == 42)
    printf("rare\n");

  exit(0);
}
