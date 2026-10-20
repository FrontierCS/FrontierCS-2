/* Stage 2 of the native guard. Preloaded by the judge into every native
 * candidate/reference process; its constructor runs after dynamic loading and
 * before main(). Stage 1 (installed before exec) must still allow what the
 * dynamic loader needs: fstat on the libraries it maps and the exec itself.
 * Neither is needed afterwards, so they are removed here. Without stat a
 * process cannot observe file timestamps, and without exec it cannot replace
 * itself with another program. Seccomp filters only ever tighten. */
#define _GNU_SOURCE
#include <errno.h>
#include <stddef.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#define DENY(number) \
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, (number), 0, 1), \
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EACCES)

__attribute__((constructor)) static void edlib_native_guard_stage2(void) {
    struct sock_filter filter[] = {
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
        DENY(SYS_stat), DENY(SYS_fstat), DENY(SYS_lstat), DENY(SYS_newfstatat),
        DENY(SYS_statx), DENY(SYS_execve), DENY(SYS_execveat), DENY(SYS_prctl),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    };
    struct sock_fprog program = {
        (unsigned short)(sizeof(filter) / sizeof(filter[0])), filter};
    if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program, 0, 0) != 0)
        _exit(87);
}
