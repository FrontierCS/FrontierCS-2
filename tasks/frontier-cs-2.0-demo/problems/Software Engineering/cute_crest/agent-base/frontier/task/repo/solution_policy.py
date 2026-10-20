#!/usr/bin/env python3
"""Rules for the code under src/frontier_solution/.

Two checks, both applied by the judge before a strategy is run:

  check_sources  -- which files may exist and what their text may contain;
  check_objects  -- which external symbols the compiled objects may reference
                    and that nothing runs before the strategy is constructed.

The strategy process is additionally confined at run time (no descriptors
besides its two pipes, no system calls besides read/write/memory/exit), so
these checks are the first line of defence, not the only one.
"""
from pathlib import Path
import re
import subprocess

MAX_FILES = 32
MAX_FILE_BYTES = 262144
SOURCE_SUFFIXES = {'.cc', '.h'}

# Standard headers a strategy may include: containers, algorithms, numerics.
ALLOWED_HEADERS = frozenset('''
algorithm array bitset cassert cfloat climits cmath cstddef cstdint cstdlib cstring deque
functional initializer_list iterator limits list map memory new numeric optional queue set
stack string tuple type_traits unordered_map unordered_set utility vector
assert.h float.h limits.h math.h stddef.h stdint.h stdlib.h string.h
'''.split())
API_HEADER = 'strategy/strategy_api.h'

ALLOWED_DIRECTIVES = frozenset(
    ['include', 'define', 'undef', 'if', 'ifdef', 'ifndef', 'else', 'elif', 'endif'])

# Whole-word tokens that may not appear outside comments and literals.
FORBIDDEN_WORDS = frozenset('''
asm __asm __asm__ __attribute__ __attribute __declspec syscall __has_include __has_include_next
_Pragma __DATE__ __TIME__ __TIMESTAMP__ __FILE__ __BASE_FILE__ __COUNTER__ thread_local
__thread
'''.split())

# External symbols a compiled solution may reference, besides the strategy
# interface itself.  Memory, C string/math helpers, and the support routines
# the standard containers and exceptions compile down to.  Nothing here can
# open a file, read a clock, start a process or reach the environment.
ALLOWED_SYMBOLS = frozenset('''
memcpy memmove memset memcmp memchr strlen strcmp strncmp abs labs llabs
sqrt sqrtf cbrt exp exp2 expm1 log log2 log10 log1p pow fabs floor ceil round trunc fmod
sin cos tan asin acos atan atan2 sinh cosh tanh erf erfc lgamma tgamma hypot ldexp frexp
logf expf powf fabsf floorf ceilf roundf lround llround lrint llrint nearbyint rint
__stack_chk_fail __assert_fail abort __dso_handle _GLOBAL_OFFSET_TABLE_
_Znwm _Znam _ZdlPv _ZdaPv _ZdlPvm _ZdaPvm _ZnwmRKSt9nothrow_t _ZnamRKSt9nothrow_t
_ZdlPvRKSt9nothrow_t _ZdaPvRKSt9nothrow_t _ZSt7nothrow
_Unwind_Resume __gxx_personality_v0 __cxa_allocate_exception __cxa_free_exception __cxa_throw
__cxa_rethrow __cxa_begin_catch __cxa_end_catch __cxa_pure_virtual __cxa_deleted_virtual
__cxa_bad_cast __cxa_bad_typeid __cxa_throw_bad_array_new_length __cxa_guard_acquire
__cxa_guard_release __cxa_guard_abort __dynamic_cast _ZSt9terminatev
_ZSt11_Hash_bytesPKvmm _ZSt15_Fnv_hash_bytesPKvmm
'''.split())

ALLOWED_SYMBOL_PATTERNS = [re.compile(p) for p in (
    r'_ZN8frontier4Host\w+',                 # the strategy interface
    r'_ZNK8frontier4Host\w+',
    r'_ZT[IVS]N8frontier8StrategyE',
    r'_ZSt\d+__throw_\w+',                    # std::__throw_*
    r'_ZSt\d+_Rb_tree_\w+',                   # red-black tree support
    r'_ZNSt8__detail15_List_node_base\w+',    # std::list support
    r'_ZNKSt8__detail20_Prime_rehash_policy\w+',  # unordered containers
    r'_ZNSt8__detail20_Prime_rehash_policy\w+',
    r'_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE\w+',   # std::string
    r'_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE\w+',
    r'_ZT[IVS]N10__cxxabiv1\w+',              # run-time type information
    r'_ZT[IVS]St\d+\w+',                      # typeinfo of standard exceptions
    r'_ZNSt\d+(bad_alloc|bad_cast|exception|logic_error|runtime_error|out_of_range|'
    r'length_error|invalid_argument|domain_error|range_error|overflow_error|'
    r'bad_array_new_length|bad_function_call)\w+',
    r'_ZNKSt\d+(bad_alloc|bad_cast|exception|logic_error|runtime_error)\w+',
)]

# Sections and symbols that mean "code runs before main() reaches the
# strategy": the sandbox is entered before the strategy is constructed, and
# nothing of the solution may run ahead of it.
FORBIDDEN_SECTION = re.compile(r'\.(init_array|fini_array|preinit_array|ctors|dtors|init|fini)(\.|$)')
FORBIDDEN_DEFINED = re.compile(r'^(_GLOBAL__(sub_)?[ID]_|_Z41__static_initialization_and_destruction)')
ENTRY_SYMBOL = '_ZN8frontier12MakeStrategyEv'


class PolicyViolation(Exception):
    """A solution breaks the rules.  `code` is a stable short category."""

    def __init__(self, code, detail):
        Exception.__init__(self, code + ': ' + detail)
        self.code = code
        self.detail = detail


def strip_comments_and_literals(text):
    """Blanks comments, string and character literals; keeps line structure."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        two = text[i:i + 2]
        if two == '//':
            j = text.find('\n', i)
            j = n if j < 0 else j
            i = j
        elif two == '/*':
            j = text.find('*/', i + 2)
            if j < 0:
                raise PolicyViolation('lexical_form', 'unterminated comment')
            out.append('\n' * text.count('\n', i, j))
            i = j + 2
        elif c in '"\'':
            j = i + 1
            while j < n and text[j] != c:
                if text[j] == '\n':
                    raise PolicyViolation('lexical_form', 'unterminated literal')
                j += 2 if text[j] == '\\' else 1
            if j >= n:
                raise PolicyViolation('lexical_form', 'unterminated literal')
            out.append(c + c)
            i = j + 1
        else:
            out.append(c)
            i += 1
    return ''.join(out)


def check_source_text(name, text, local_files):
    if '\\\n' in text or '??' in text or '%:' in text:
        raise PolicyViolation('lexical_form', name + ': line splices, trigraphs and %: are not allowed')
    if re.search(r'R"', text):
        raise PolicyViolation('lexical_form', name + ': raw string literals are not allowed')
    # Include targets are read from the original text; everything else from
    # the text with comments and literals blanked.
    stripped = strip_comments_and_literals(text)
    original_lines = text.split('\n')
    for number, line in enumerate(stripped.split('\n')):
        s = line.strip()
        if s.startswith('#'):
            m = re.match(r'#\s*(\w+)', s)
            directive = m.group(1) if m else ''
            if directive not in ALLOWED_DIRECTIVES:
                raise PolicyViolation('preprocessor_directive', '%s:%d: #%s' % (name, number + 1, directive))
            if directive == 'include':
                raw = original_lines[number].strip()
                m = re.fullmatch(r'#\s*include\s*(<([^<>"]+)>|"([^<>"]+)")\s*(//.*)?', raw)
                if not m:
                    raise PolicyViolation('forbidden_include', '%s:%d' % (name, number + 1))
                target = m.group(2) or m.group(3)
                ok = (target in ALLOWED_HEADERS if m.group(2) else
                      target == API_HEADER or target in local_files)
                if not ok:
                    raise PolicyViolation('forbidden_include', '%s:%d: %s' % (name, number + 1, target))
        if '##' in line:
            raise PolicyViolation('lexical_form', '%s:%d: token pasting is not allowed' % (name, number + 1))
    for word in re.findall(r'[A-Za-z_]\w*', stripped):
        if word in FORBIDDEN_WORDS or word.startswith('__builtin') or word.startswith('__sync') \
                or word.startswith('__atomic'):
            raise PolicyViolation('forbidden_identifier', name + ': ' + word)


def check_sources(solution_dir):
    """Validates the solution directory and returns its .cc files, sorted."""
    solution_dir = Path(solution_dir)
    if not solution_dir.is_dir() or solution_dir.is_symlink():
        raise PolicyViolation('solution_layout', 'src/frontier_solution is not a directory')
    files = []
    for path in sorted(solution_dir.rglob('*')):
        if path.is_symlink():
            raise PolicyViolation('solution_layout', 'symbolic links are not allowed')
        if path.is_dir():
            raise PolicyViolation('solution_layout', 'subdirectories are not allowed')
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES \
                or not re.fullmatch(r'[A-Za-z0-9_]+\.(cc|h)', path.name):
            raise PolicyViolation('solution_layout', 'only .cc and .h files are allowed: ' + path.name)
        if path.stat().st_size > MAX_FILE_BYTES:
            raise PolicyViolation('solution_layout', 'file too large: ' + path.name)
        files.append(path)
    if len(files) > MAX_FILES:
        raise PolicyViolation('solution_layout', 'too many files')
    if not (solution_dir / 'strategy.cc').is_file():
        raise PolicyViolation('solution_layout', 'strategy.cc is missing')
    names = {p.name for p in files}
    for path in files:
        try:
            text = path.read_bytes().decode('ascii')
        except UnicodeDecodeError:
            raise PolicyViolation('lexical_form', path.name + ': source must be ASCII')
        check_source_text(path.name, text, names)
    return [p for p in files if p.suffix == '.cc']


def symbol_allowed(name):
    return name in ALLOWED_SYMBOLS or any(p.fullmatch(name) for p in ALLOWED_SYMBOL_PATTERNS)


def _tool(args):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
    if p.returncode:
        raise PolicyViolation('object_audit', 'cannot inspect compiled object')
    return p.stdout


def check_object_listing(nm_output, sections):
    """nm_output: lines of `nm -P` output; sections: section names."""
    entry = False
    for line in nm_output.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name, kind = parts[0], parts[1]
        if kind == 'U':
            if not symbol_allowed(name):
                raise PolicyViolation('forbidden_symbol', name)
        elif kind in 'iI':
            raise PolicyViolation('forbidden_symbol', 'indirect function ' + name)
        else:
            if FORBIDDEN_DEFINED.match(name):
                raise PolicyViolation('static_initialization',
                                      'objects with dynamic initialisation at namespace scope')
            if kind.isupper() and kind not in 'WVU' and not name.startswith('_Z'):
                raise PolicyViolation('forbidden_symbol', 'defines C symbol ' + name)
            if name == ENTRY_SYMBOL and kind == 'T':
                entry = True
    for section in sections:
        if FORBIDDEN_SECTION.match(section):
            raise PolicyViolation('static_initialization', 'section ' + section)
    return entry


def check_objects(objects):
    entry = False
    for obj in objects:
        nm = _tool(['nm', '-P', str(obj)])
        headers = _tool(['readelf', '-S', '-W', str(obj)])
        sections = re.findall(r'\]\s+(\.\S+)', headers)
        entry = check_object_listing(nm, sections) or entry
    if not entry:
        raise PolicyViolation('solution_layout', 'frontier::MakeStrategy() is not defined')
