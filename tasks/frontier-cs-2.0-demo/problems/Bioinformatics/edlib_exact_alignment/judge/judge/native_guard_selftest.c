/* Run by the judge under the complete native guard before any measurement.
 * Prints one line: the names of the probes that SUCCEEDED. The judge accepts
 * exactly the three controls (read-only open, clock, heap); anything else
 * means the runtime does not enforce the guard and no measurement is made.
 * argv[1] is an existing judge-owned file that is writable by this UID. */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

static void report(const char *name, long ok) { if (ok) printf(" %s", name); }
static long sc(long n, long a, long b, long c, long d) { return syscall(n, a, b, c, d) >= 0; }

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    const char *file = argv[1];
    char other[4096], buffer[16];
    struct timespec now;
    struct stat_like { char bytes[512]; } st;
    snprintf(other, sizeof other, "%s.guard", file);
    printf("EDLIB_NATIVE_GUARD");
    /* Controls: must succeed. */
    int fd = open(file, O_RDONLY);
    report("control_read_only_open", fd >= 0 && read(fd, buffer, sizeof buffer) > 0);
    report("control_clock", clock_gettime(CLOCK_MONOTONIC, &now) == 0);
    void *block = malloc(64 << 20);
    report("control_heap", block != NULL); if (block) { memset(block, 1, 64 << 20); free(block); }
    /* Everything below must fail. */
    report("fopen_write_cwd", fopen("edlib-guard-state", "w") != NULL);
    report("fopen_write_tmp", fopen("/tmp/edlib-guard-state", "w") != NULL);
    report("fopen_write_shm", fopen("/dev/shm/edlib-guard-state", "w") != NULL);
    report("fopen_append", fopen(file, "a") != NULL);
    report("open_wronly", open(file, O_WRONLY) >= 0);
    report("open_rdwr", open(file, O_RDWR) >= 0);
    report("open_creat", open(other, O_RDONLY | O_CREAT, 0600) >= 0);
    report("open_trunc", open(file, O_RDONLY | O_TRUNC) >= 0);
    report("open_tmpfile", open("/tmp", O_TMPFILE | O_RDWR, 0600) >= 0);
    report("raw_open_wronly", sc(SYS_open, (long)file, O_WRONLY, 0, 0));
    report("raw_openat_creat", sc(SYS_openat, AT_FDCWD, (long)other, O_WRONLY | O_CREAT, 0600));
    report("raw_openat2", sc(437, AT_FDCWD, (long)file, (long)&st, 24));
    report("raw_creat", sc(SYS_creat, (long)other, 0600, 0, 0));
    report("mkdir", sc(SYS_mkdir, (long)other, 0700, 0, 0));
    report("mkdirat", sc(SYS_mkdirat, AT_FDCWD, (long)other, 0700, 0));
    report("mknod", sc(SYS_mknod, (long)other, 0600 | 0100000, 0, 0));
    report("rename", sc(SYS_rename, (long)file, (long)other, 0, 0));
    report("renameat2", sc(SYS_renameat2, AT_FDCWD, (long)file, AT_FDCWD, (long)other));
    report("link", sc(SYS_link, (long)file, (long)other, 0, 0));
    report("symlink", sc(SYS_symlink, (long)file, (long)other, 0, 0));
    report("unlink", sc(SYS_unlink, (long)file, 0, 0, 0));
    report("unlinkat", sc(SYS_unlinkat, AT_FDCWD, (long)file, 0, 0));
    report("truncate", sc(SYS_truncate, (long)file, 0, 0, 0));
    report("ftruncate", sc(SYS_ftruncate, fd, 0, 0, 0));
    report("chmod", sc(SYS_chmod, (long)file, 0600, 0, 0));
    report("fchmod", sc(SYS_fchmod, fd, 0600, 0, 0));
    report("utimensat", sc(SYS_utimensat, AT_FDCWD, (long)file, 0, 0));
    report("setxattr", sc(SYS_setxattr, (long)file, (long)"user.edlib", (long)"1", 1));
    report("fsetxattr", sc(SYS_fsetxattr, fd, (long)"user.edlib", (long)"1", 1));
    report("ioctl", sc(SYS_ioctl, fd, 0x80086601 /* FS_IOC_GETFLAGS */, (long)&st, 0));
    report("fcntl", sc(SYS_fcntl, fd, F_GETFL, 0, 0));
    report("flock", sc(SYS_flock, fd, 1, 0, 0));
    report("stat", sc(SYS_stat, (long)file, (long)&st, 0, 0));
    report("fstat", sc(SYS_fstat, fd, (long)&st, 0, 0));
    report("lstat", sc(SYS_lstat, (long)file, (long)&st, 0, 0));
    report("newfstatat", sc(SYS_newfstatat, AT_FDCWD, (long)file, (long)&st, 0));
    report("statx", sc(SYS_statx, AT_FDCWD, (long)file, 0, 0x7ff));
    report("getdents64", sc(SYS_getdents64, open("/tmp", O_RDONLY | O_DIRECTORY), (long)&st, sizeof st, 0));
    report("readlink", sc(SYS_readlink, (long)"/proc/self/exe", (long)&st, sizeof st, 0));
    report("memfd_create", sc(SYS_memfd_create, (long)"edlib", 0, 0, 0));
    report("shmget", sc(SYS_shmget, 0x45444c42, 4096, 01600, 0));
    report("msgget", sc(SYS_msgget, 0x45444c42, 01600, 0, 0));
    report("semget", sc(SYS_semget, 0x45444c42, 1, 01600, 0));
    report("mq_open", sc(SYS_mq_open, (long)"edlib", O_RDWR | O_CREAT, 0600, 0));
    report("socket_unix", sc(SYS_socket, 1, 1, 0, 0));
    report("socket_inet", sc(SYS_socket, 2, 2, 0, 0));
    report("pipe2", sc(SYS_pipe2, (long)&st, 0, 0, 0));
    report("eventfd", sc(SYS_eventfd2, 0, 0, 0, 0));
    report("inotify", sc(SYS_inotify_init1, 0, 0, 0, 0));
    report("io_uring_setup", sc(425, 4, (long)&st, 0, 0));
    report("kill_parent_probe", sc(SYS_kill, getppid(), 0, 0, 0));
    report("process_vm_readv", sc(SYS_process_vm_readv, getppid(), 0, 0, 0));
    report("prctl", sc(SYS_prctl, 15 /* PR_SET_NAME */, (long)"edlib", 0, 0));
    report("prlimit_other", sc(SYS_prlimit64, getppid(), 7 /* RLIMIT_NOFILE */, 0, (long)&st));
    report("setrlimit_fsize", sc(SYS_setrlimit, 1 /* RLIMIT_FSIZE */, (long)"\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377", 0, 0));
    report("chdir", sc(SYS_chdir, (long)"/tmp", 0, 0, 0));
    fflush(stdout);   /* without the guard the next line replaces this process */
    report("execve", sc(SYS_execve, (long)"/bin/true", (long)(char *[]){"true", NULL}, (long)(char *[]){NULL}, 0));
    printf("\n");
    return 0;
}
