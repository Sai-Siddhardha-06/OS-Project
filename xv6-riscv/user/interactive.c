#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

int
main(int argc, char *argv[])
{
  int interactions = 30;
  if (argc > 1) {
    interactions = atoi(argv[1]);
  }

  volatile int dummy = 0;
  for (int i = 0; i < interactions; i++) {
    // Short burst of responsiveness
    for (int k = 0; k < 10000; k++) {
      dummy += k;
    }
    // Think-time / waiting for user input
    pause(2);
  }

  if (dummy == 42)
    printf("rare\n");

  exit(0);
}
