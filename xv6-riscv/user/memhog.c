#include "kernel/types.h"
#include "user/user.h"

int
main(void)
{
  printf("memhog: starting memory allocation stress test...\n");
  int initial_sz = getmemsize();
  printf("memhog: initial size = %d bytes\n", initial_sz);

  int step = 4096; // 1 page (4 KB) at a time
  int total_grown = 0;
  int iters = 0;
  const int max_iters = 8192; // up to 32 MB

  while (iters < max_iters) {
    char *p = sbrk(step);
    if (p == (char *)-1) {
      printf("memhog: sbrk refused at +%d bytes! (current total size = %d bytes)\n",
             total_grown, getmemsize());
      break;
    }
    // Touch the allocated page to ensure mapping
    *p = 1;
    total_grown += step;
    iters++;
  }

  if (iters == max_iters) {
    printf("memhog: reached target limit of %d bytes without refusal (size = %d bytes)\n",
           total_grown, getmemsize());
  }

  printf("memhog: finished test.\n");
  exit(0);
}
