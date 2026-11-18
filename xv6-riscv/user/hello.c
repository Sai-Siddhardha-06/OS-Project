#include "kernel/types.h"
#include "user/user.h"

int main()
{
	printf("Hello from xv6!\n");
	printf("My PID: %d\n", getpid());
	printf("My size: %d\n",getmemsize());
	exit(0);
}
