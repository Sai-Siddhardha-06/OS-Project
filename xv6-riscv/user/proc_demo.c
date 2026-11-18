#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

int
main(int argc, char *argv[])
{
  // 1. Print the parent process ID
  printf("Parent PID: %d\n", getpid());

  // 2. Create one child with fork()
  pid_t pid = fork();

  if(pid < 0)
  {
    printf("fork failed\n");
    exit(1);
  }

  if(pid == 0)
  {
    // Child process execution path
    // 3. Print the child process ID
    printf("Child PID: %d\n", getpid());

    // 4. Print uptime using uptime() (returns clock ticks since boot)
    printf("System uptime in child process: %d ticks\n", uptime());

    exit(0);
  }
  else
  {
    // Parent process execution path
    // 5. Wait for the child to exit
    wait(0);

    // 6. Print a completion message after the child exits
    printf("Child process (PID %d) has exited. Parent complete.\n", pid);
    printf("System uptime in parent process: %d ticks\n", uptime());
  }

  exit(0);
}
