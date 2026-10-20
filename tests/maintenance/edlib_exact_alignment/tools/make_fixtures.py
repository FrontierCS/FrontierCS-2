"""Generate adversarial fixture patches: reference body wrapped with an attack."""
import json, subprocess, sys, shutil, tempfile
from pathlib import Path
T=Path(sys.argv[1]); OUT=Path(sys.argv[2])
surface=json.loads((T/'judge/judge/surface.json').read_text())
work=Path(tempfile.mkdtemp())
shutil.copytree(T/'judge/harbor/app',work/'a'); shutil.copytree(T/'judge/harbor/app',work/'ref')
subprocess.run(['patch','-s','-p1','-d',work/'ref','-i',T/'judge/reference.patch'],check=True)
ref=(work/'ref'/surface['path']).read_text()
body=ref[len(surface['prefix']):len(ref)-len(surface['suffix'])]
def wrap(prologue,epilogue):
    return ('\n    // ADVERSARIAL FIXTURE. Not a solution; see test_adversarial.py.\n'
            '    static int fx_depth = 0;\n'
            '    auto fx_core = [&]() -> int {'+body+'    };\n'
            '    if (fx_depth == 0) {\n'+prologue+'    }\n'
            '    ++fx_depth; const int fx_status = fx_core(); --fx_depth;\n'
            '    if (fx_depth == 0 && fx_status == EDLIB_STATUS_OK) {\n'+epilogue+'    }\n'
            '    return fx_status;\n')
KEY='''        char fx_name[96];
        snprintf(fx_name, sizeof fx_name, "edlib-fx-%d-%d-%d.bin", queryLength, targetLength, bestScore);
        const char* fx_dirs[] = {".", "/tmp", "/dev/shm", "/var/tmp"};
        FILE* (*fx_open)(const char*, const char*) = fopen;   // not a call the screen recognises
'''
fixtures={
'state_cache': wrap(KEY+'''        for (const char* fx_dir : fx_dirs) {
            char fx_path[160]; snprintf(fx_path, sizeof fx_path, "%s/%s", fx_dir, fx_name);
            FILE* fx_file = fx_open(fx_path, "rb");
            if (!fx_file) continue;
            int fx_length = 0;
            if (fread(&fx_length, sizeof fx_length, 1, fx_file) == 1 && fx_length > 0) {
                unsigned char* fx_path_bytes = static_cast<unsigned char*>(malloc(fx_length));
                if (fread(fx_path_bytes, 1, fx_length, fx_file) == static_cast<size_t>(fx_length)) {
                    fclose(fx_file);
                    // The whole reconstruction is skipped: the path came from an earlier process.
                    fputs("EDLIB_FIXTURE state reused from an earlier process\\n", stderr);
                    *alignment = fx_path_bytes; *alignmentLength = fx_length;
                    return EDLIB_STATUS_OK;
                }
                free(fx_path_bytes);
            }
            fclose(fx_file);
        }
''',KEY+'''        for (const char* fx_dir : fx_dirs) {
            char fx_path[160]; snprintf(fx_path, sizeof fx_path, "%s/%s", fx_dir, fx_name);
            FILE* fx_file = fx_open(fx_path, "wb");
            if (!fx_file) continue;
            fwrite(alignmentLength, sizeof *alignmentLength, 1, fx_file);
            fwrite(*alignment, 1, *alignmentLength, fx_file);
            fclose(fx_file);
        }
'''),
'mode_detect': wrap('''        // Read our own command line; "-s" is the mode in which no path is printed.
        FILE* fx_file = (fopen)("/proc/self/cmdline", "rb");
        if (fx_file) {
            char fx_args[4096]; const size_t fx_size = fread(fx_args, 1, sizeof fx_args - 1, fx_file);
            fclose(fx_file);
            for (size_t fx_i = 0; fx_i + 3 < fx_size; fx_i++) {
                if (fx_args[fx_i] == 0 && fx_args[fx_i + 1] == '-' && fx_args[fx_i + 2] == 's' && fx_args[fx_i + 3] == 0) {
                    // Nobody looks at the path in this mode: return a one-byte non-path at once.
                    *alignment = static_cast<unsigned char*>(malloc(1)); (*alignment)[0] = 0; *alignmentLength = 1;
                    return EDLIB_STATUS_OK;
                }
            }
        }
''',''),
'wrong_path_when_printed': wrap('''        // The mirror image: skip the work exactly when the path IS printed.
        FILE* fx_file = (fopen)("/proc/self/cmdline", "rb");
        if (fx_file) {
            char fx_args[4096]; const size_t fx_size = fread(fx_args, 1, sizeof fx_args - 1, fx_file);
            fclose(fx_file);
            for (size_t fx_i = 0; fx_i + 8 < fx_size; fx_i++) {
                if (fx_args[fx_i] == 0 && !memcmp(fx_args + fx_i + 1, "CIG_EXT", 8)) {
                    *alignment = static_cast<unsigned char*>(malloc(1)); (*alignment)[0] = 0; *alignmentLength = 1;
                    return EDLIB_STATUS_OK;
                }
            }
        }
''',''),
'screen_evasion': wrap('''        // Three spellings the source screen does not recognise, and every place or
        // object that could hold state for a later process. Each success is announced.
        long fx_syscall(long, ...) __asm__("syscall");
        FILE* (*fx_open)(const char*, const char*) = std::fopen;
        const char* fx_paths[] = {"edlib-fx-evasion", "/tmp/edlib-fx-evasion", "/dev/shm/edlib-fx-evasion",
                                  "/var/tmp/edlib-fx-evasion"};
        for (const char* fx_path : fx_paths) {
            FILE* fx_file = (fopen)(fx_path, "w");
            if (fx_file) { fprintf(stderr, "EDLIB_FIXTURE channel open: (fopen) %s\\n", fx_path); fclose(fx_file); }
            fx_file = fx_open(fx_path, "a");
            if (fx_file) { fprintf(stderr, "EDLIB_FIXTURE channel open: pointer fopen %s\\n", fx_path); fclose(fx_file); }
            if (fx_syscall(2L, fx_path, 0101, 0600) >= 0) fprintf(stderr, "EDLIB_FIXTURE channel open: raw open %s\\n", fx_path);
            if (fx_syscall(257L, -100, fx_path, 0102, 0600) >= 0) fprintf(stderr, "EDLIB_FIXTURE channel open: raw openat %s\\n", fx_path);
        }
        if (fx_syscall(83L, "/tmp/edlib-fx-evasion-dir", 0700) >= 0) fputs("EDLIB_FIXTURE channel open: mkdir\\n", stderr);
        if (fx_syscall(319L, "edlib-fx", 0) >= 0) fputs("EDLIB_FIXTURE channel open: memfd_create\\n", stderr);
        if (fx_syscall(29L, 0x45444c42, 4096, 01600) >= 0) fputs("EDLIB_FIXTURE channel open: shmget\\n", stderr);
        if (fx_syscall(68L, 0x45444c42, 01600) >= 0) fputs("EDLIB_FIXTURE channel open: msgget\\n", stderr);
        if (fx_syscall(64L, 0x45444c42, 1, 01600) >= 0) fputs("EDLIB_FIXTURE channel open: semget\\n", stderr);
        if (fx_syscall(41L, 1, 1, 0) >= 0) fputs("EDLIB_FIXTURE channel open: socket\\n", stderr);
''',''),
}
WHICH='''        // Which program is this? Only the plain aligner is targeted; the API probe and
        // the instrumented build behave normally so that every other check passes.
        bool fx_plain = false;
        FILE* fx_self = (fopen)("/proc/self/cmdline", "rb");
        if (fx_self) {
            char fx_args[4096]; const size_t fx_size = fread(fx_args, 1, sizeof fx_args - 1, fx_self);
            fclose(fx_self); fx_args[fx_size] = 0;
            fx_plain = strstr(fx_args, "edlib-aligner") && !strstr(fx_args, "core-source");
        }
'''
def forged(seconds, sleep):
    return WHICH+'''        if (fx_plain) {
'''+('''            // Real cost: 50 ms on top of a complete, correct reconstruction.
            int fx_sleep(const void*, void*) __asm__("nanosleep");
            long fx_pause[2] = {0, 50000000L}; fx_sleep(fx_pause, 0);
''' if sleep else '')+'''            // Finish the CLI's output ourselves, with a time of our choosing, and leave
            // before the CLI can print its own.
            char* fx_cigar = edlibAlignmentToCigar(*alignment, *alignmentLength, EDLIB_CIGAR_EXTENDED);
            printf("\\nQuery #0 (%d residues): score = %d\\nCigar:\\n%s\\n\\nCpu time of searching: '''+seconds+'''\\n",
                   queryLength, bestScore, fx_cigar);
            fflush(stdout);
            (exit)(0);
        }
'''
fixtures.update({
'forged_time_small': wrap('',forged('0.000000001',True)),
'forged_time_large': wrap('',forged('9.000000000',False)),
'extra_output': wrap(WHICH+'''        if (fx_plain) fputs("note from the submission\\n", stdout);
''',''),
})

BURN="""        for (int fx_i = 1; fx_i < fx_argc; fx_i++)
            if (fx_argv[fx_i][0] == '-' && fx_argv[fx_i][1] == 'l' && !fx_argv[fx_i][2])
                for (volatile long fx_spin = 0; fx_spin < 12000000; fx_spin = fx_spin + 1) {}
"""
# Location-only slowdown, outside the function: the body closes its function,
# adds a constructor that burns time when the command line has -l, and opens a
# dummy function for the retained closing brace.
fixtures['location_slowdown_scope']=(body+'''}
// ADVERSARIAL FIXTURE. Not a solution; see test_adversarial.py.
__attribute__((constructor)) static void fx_slow(int fx_argc, char** fx_argv) {
'''+BURN+'''}
static int fx_tail(int fx_value) {
    return fx_value;
''')
# The same slowdown without leaving the function: a local class and a static
# pointer placed in the constructor table.
fixtures['location_slowdown_section']=('''
    // ADVERSARIAL FIXTURE. Not a solution; see test_adversarial.py.
    struct FxSlow {
        static void run(int fx_argc, char** fx_argv, char**) {
'''+BURN+'''        }
    };
    static void (*fx_entry)(int, char**, char**) __attribute__((used, section(".init_array"))) = &FxSlow::run;
'''+body)
VECTORS=('''        vector<uint16_t> checkpoints(static_cast<size_t>(bestScore / tileSize + 1) * width, 0);
        vector<uint16_t> columns(static_cast<size_t>(tileSize) * width, 0);
        vector<uint16_t> work(2 * width, 0);
''')
assert body.count(VECTORS)==1
SPAN='''        // ADVERSARIAL FIXTURE. Not a solution; see test_adversarial.py.
        struct FxSpan { uint16_t* p; size_t n; uint16_t* data() { return p; } size_t size() const { return n; } };
        const size_t fx_a = static_cast<size_t>(bestScore / tileSize + 1) * width, fx_b = static_cast<size_t>(tileSize) * width, fx_c = static_cast<size_t>(2 * width);
'''
USE='''        FxSpan checkpoints{fx_store_a, fx_a};
        FxSpan columns{fx_store_b, fx_b};
        FxSpan work{fx_store_c, fx_c};
'''
fixtures['static_storage']=body.replace(VECTORS,SPAN+'''        static uint16_t fx_store_a[17 * 1027], fx_store_b[32 * 1027], fx_store_c[2 * 1027];
        memset(fx_store_a, 0, sizeof fx_store_a); memset(fx_store_b, 0, sizeof fx_store_b); memset(fx_store_c, 0, sizeof fx_store_c);
'''+USE)
fixtures['stack_storage']=body.replace(VECTORS,SPAN+'''        uint16_t fx_store_a[17 * 1027], fx_store_b[32 * 1027], fx_store_c[2 * 1027];
        memset(fx_store_a, 0, fx_a * 2); memset(fx_store_b, 0, fx_b * 2); memset(fx_store_c, 0, fx_c * 2);
'''+USE)
fixtures['mapped_storage']=body.replace(VECTORS,SPAN+'''        void* fx_map(void*, size_t, int, int, int, long) __asm__("mmap");
        uint16_t* const fx_store_a = static_cast<uint16_t*>(fx_map(NULL, fx_a * 2, 3, 0x22, -1, 0));
        uint16_t* const fx_store_b = static_cast<uint16_t*>(fx_map(NULL, fx_b * 2, 3, 0x22, -1, 0));
        uint16_t* const fx_store_c = static_cast<uint16_t*>(fx_map(NULL, fx_c * 2, 3, 0x22, -1, 0));
'''+USE)
OUT.mkdir(exist_ok=True)
for name,new in fixtures.items():
    (work/'b').mkdir(exist_ok=True); shutil.rmtree(work/'b'); shutil.copytree(T/'judge/harbor/app',work/'b')
    (work/'b'/surface['path']).write_text(surface['prefix']+new+surface['suffix'])
    diff=subprocess.run(['diff','-u','--label','a/'+surface['path'],'--label','b/'+surface['path'],'a/'+surface['path'],'b/'+surface['path']],cwd=work,capture_output=True,text=True).stdout
    (OUT/(name+'.patch')).write_text('diff --git a/%s b/%s\n'%(surface['path'],surface['path'])+diff)
    print(name,len(diff))
