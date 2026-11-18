#include "kernel/types.h"
#include "kernel/stat.h"
#include "kernel/fcntl.h"
#include "user/user.h"

int
main(int argc, char *argv[])
{
  int rounds = 40;
  if (argc > 1) {
    rounds = atoi(argv[1]);
  }

  char buf[64];
  memset(buf, 'A', sizeof(buf));

  int fd = open("iotemp.dat", O_CREATE | O_RDWR);
  if (fd < 0) {
    printf("iowork: cannot open iotemp.dat\n");
    exit(1);
  }

  for (int r = 0; r < rounds; r++) {
    write(fd, buf, sizeof(buf));
    pause(1); // sleep 1 tick simulating I/O device delay
    read(fd, buf, sizeof(buf));
  }

  close(fd);
  unlink("iotemp.dat");
  exit(0);
}
