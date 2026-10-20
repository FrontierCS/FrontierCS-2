"""Private untimed reconstruction-storage measurement; timed sources stay pristine."""
from pathlib import Path
import hashlib
import json
import re
import shutil
from native_cmake import build_cmake_entrypoint
from elf_surface import static_storage_bytes

ROOT=Path(__file__).resolve().parent
SURFACE=json.loads((ROOT/'judge/surface.json').read_text())
DEFINITION='static int obtainAlignmentHirschberg('
SCOPE='\n    EdlibCoreAllocationScope edlib_trusted_allocation_scope(__builtin_frame_address(0));\n'
# The measured objects' references to the allocator behind malloc and to mmap are
# redirected to the collector; -z now keeps lazy symbol resolution off the stack.
WRAPPED=('__libc_malloc','__libc_calloc','__libc_realloc','__libc_memalign','__libc_free',
         'mmap','mmap64','munmap','mremap','sbrk','brk','syscall','shmat','dlopen','dlsym','dlvsym')

def build_core(tree,deadline,*,before_process=None,receipts=None):
    tree=Path(tree)
    source=tree/'core-source'
    # Each disposable candidate/reference tree has its own derivative.
    shutil.copytree(tree,source,ignore=shutil.ignore_patterns('build','sanitized','core-source'))
    path=source/SURFACE['path'];text=path.read_text()
    prefix=SURFACE['prefix']
    if not text.startswith(prefix) or not text.endswith(SURFACE['suffix']) or not prefix.rstrip().endswith('{'):
        raise ValueError('core instrumentation source boundary mismatch')
    at=prefix.rfind(DEFINITION)
    if at<0 or prefix.count(DEFINITION)!=2:      # forward declaration and definition
        raise ValueError('core instrumentation source boundary mismatch')
    original=path.read_bytes()
    # The core is never inlined in the derivative, so its frame address is the
    # same anchor for the stack measurement in both lanes.
    text=('#include "core_alloc.h"\n'+prefix[:at]+'__attribute__((noinline)) '+prefix[at:]
          +SCOPE+text[len(prefix):])
    path.write_text(text)
    for name in ('core_alloc.h','core_alloc.cpp'):
        shutil.copyfile(ROOT/'judge'/name,source/'edlib/src'/name)
    cmake=source/'CMakeLists.txt'
    with cmake.open('a') as f:
        f.write('\nset(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,now -Wl,'
                +','.join('--wrap='+name for name in WRAPPED)+'")\n')
    (source/'core-source-binding.json').write_text(json.dumps({
        'original_cpp_sha256':hashlib.sha256(original).hexdigest(),
        'instrumented_cpp_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'collector_sha256':hashlib.sha256((ROOT/'judge/core_alloc.cpp').read_bytes()).hexdigest(),
        'entry_scope':'outermost obtainAlignmentHirschberg inclusive recursive heap, mapped memory and stack',
        'timed_source_modified':False},sort_keys=True))
    return build_cmake_entrypoint(source,source/'build','apps/aligner/aligner.cpp','edlib-aligner',deadline,
        before_process=before_process,receipts=receipts)

def static_bytes(executable):
    """Writable and thread-local static storage of the derivative's edlib translation unit."""
    objects=sorted(Path(executable.build).rglob('edlib.cpp.o'))
    if len(objects)!=1: raise ValueError('core object not found')
    return static_storage_bytes(objects[0])

def parse_allocation(output):
    """Peak requested live heap/mapped bytes plus stack high-water of the one reconstruction."""
    rows=re.findall(r'^EDLIB_CORE_ALLOCATION peak=(\d+) live_at_exit=(\d+) calls=(\d+) entries=(\d+) stack=(\d+)$',output,re.M)
    if len(rows)!=1 or output.count('EDLIB_CORE_ALLOCATION')!=1: raise ValueError('missing or repeated core storage evidence')
    peak,live,calls,entries,stack=map(int,rows[0])
    if peak<live or peak>4*1024**3 or entries<1 or not 0<stack<=4*1024**2:
        raise ValueError('invalid core storage evidence')
    return peak+stack
