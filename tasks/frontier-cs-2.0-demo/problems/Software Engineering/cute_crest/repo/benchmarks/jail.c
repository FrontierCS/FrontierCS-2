/* LD_PRELOAD write-jail for the program under test (Vim can be driven to
   ":w <path>").  Denies exec, chdir, and any mutating file op outside
   $CREST_JAIL (prefix) or relative paths without "..". */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
static int ok(const char *p) {
  if (!p) return 0;
  if (strstr(p, "..")) return 0;
  if (p[0] != '/') return 1;
  if (!strncmp(p, "/dev/", 5)) return 1;
  const char *j = getenv("CREST_JAIL");
  return j && !strncmp(p, j, strlen(j));
}
#define DENY do { errno = EACCES; return -1; } while (0)
int open(const char *p, int fl, ...) {
  static int (*r)(const char *, int, ...);
  if (!r) r = dlsym(RTLD_NEXT, "open");
  mode_t m = 0; va_list a; va_start(a, fl); m = va_arg(a, mode_t); va_end(a);
  if ((fl & (O_WRONLY | O_RDWR | O_CREAT | O_TRUNC | O_APPEND)) && !ok(p)) DENY;
  return r(p, fl, m);
}
int open64(const char *p, int fl, ...) {
  static int (*r)(const char *, int, ...);
  if (!r) r = dlsym(RTLD_NEXT, "open64");
  mode_t m = 0; va_list a; va_start(a, fl); m = va_arg(a, mode_t); va_end(a);
  if ((fl & (O_WRONLY | O_RDWR | O_CREAT | O_TRUNC | O_APPEND)) && !ok(p)) DENY;
  return r(p, fl, m);
}
int openat(int d, const char *p, int fl, ...) {
  static int (*r)(int, const char *, int, ...);
  if (!r) r = dlsym(RTLD_NEXT, "openat");
  mode_t m = 0; va_list a; va_start(a, fl); m = va_arg(a, mode_t); va_end(a);
  if ((fl & (O_WRONLY | O_RDWR | O_CREAT | O_TRUNC | O_APPEND)) && !ok(p)) DENY;
  return r(d, p, fl, m);
}
int creat(const char *p, mode_t m) { static int (*r)(const char *, mode_t); if (!r) r = dlsym(RTLD_NEXT, "creat"); if (!ok(p)) DENY; return r(p, m); }
FILE *fopen(const char *p, const char *m) { static FILE *(*r)(const char *, const char *); if (!r) r = dlsym(RTLD_NEXT, "fopen"); if (m && m[0] != 'r' && !ok(p)) { errno = EACCES; return NULL; } if (m && strchr(m, '+') && !ok(p)) { errno = EACCES; return NULL; } return r(p, m); }
FILE *fopen64(const char *p, const char *m) { static FILE *(*r)(const char *, const char *); if (!r) r = dlsym(RTLD_NEXT, "fopen64"); if (m && (m[0] != 'r' || strchr(m, '+')) && !ok(p)) { errno = EACCES; return NULL; } return r(p, m); }
int unlink(const char *p) { static int (*r)(const char *); if (!r) r = dlsym(RTLD_NEXT, "unlink"); if (!ok(p)) DENY; return r(p); }
int rename(const char *a, const char *b) { static int (*r)(const char *, const char *); if (!r) r = dlsym(RTLD_NEXT, "rename"); if (!ok(a) || !ok(b)) DENY; return r(a, b); }
int mkdir(const char *p, mode_t m) { static int (*r)(const char *, mode_t); if (!r) r = dlsym(RTLD_NEXT, "mkdir"); if (!ok(p)) DENY; return r(p, m); }
int rmdir(const char *p) { static int (*r)(const char *); if (!r) r = dlsym(RTLD_NEXT, "rmdir"); if (!ok(p)) DENY; return r(p); }
int chmod(const char *p, mode_t m) { static int (*r)(const char *, mode_t); if (!r) r = dlsym(RTLD_NEXT, "chmod"); if (!ok(p)) DENY; return r(p, m); }
int chown(const char *p, uid_t u, gid_t g) { DENY; }
int truncate(const char *p, off_t l) { DENY; }
int symlink(const char *a, const char *b) { DENY; }
int link(const char *a, const char *b) { DENY; }
int chdir(const char *p) { DENY; }
int fchdir(int d) { DENY; }
int execve(const char *p, char *const a[], char *const e[]) { DENY; }
int execv(const char *p, char *const a[]) { DENY; }
int execvp(const char *p, char *const a[]) { DENY; }
int execl(const char *p, const char *a, ...) { DENY; }
int execlp(const char *p, const char *a, ...) { DENY; }
int system(const char *c) { DENY; }
int kill(pid_t p, int s) { DENY; }
/* no core dumps: with a piped core_pattern (apport, systemd-coredump) the kernel
   ignores RLIMIT_CORE=0 and every crash of the program under test costs seconds */
#include <sys/prctl.h>
__attribute__((constructor)) static void crest_nodump(void) { prctl(PR_SET_DUMPABLE, 0); }
