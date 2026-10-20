"""Source rules and the symbol allowlist of the compiled solution."""
import deployment_test_guard  # noqa: F401  # isort: skip

from pathlib import Path
import shutil
import subprocess

import pytest
from tests.maintenance.cute_crest.runtime_imports import MODULES, REPO_TREE, TASK

policy = MODULES["solution_policy"]
STARTER = REPO_TREE / "src/frontier_solution/strategy.cc"
HAVE_COMPILER = shutil.which("g++") is not None and shutil.which("nm") is not None


def solution(tmp_path, text, extra=None):
    directory = tmp_path / "frontier_solution"
    directory.mkdir()
    (directory / "strategy.cc").write_text(text)
    for name, body in (extra or {}).items():
        (directory / name).write_text(body)
    return directory


def code(tmp_path, text, extra=None):
    with pytest.raises(policy.PolicyViolation) as caught:
        policy.check_sources(solution(tmp_path, text, extra))
    return caught.value.code


def test_starter_is_accepted(tmp_path):
    sources = policy.check_sources(solution(tmp_path, STARTER.read_text()))
    assert [p.name for p in sources] == ["strategy.cc"]


def test_local_headers_and_allowed_standard_headers(tmp_path):
    text = '#include <vector>\n#include <algorithm>\n#include "helper.h"\n#include "strategy/strategy_api.h"\n'
    sources = policy.check_sources(solution(tmp_path, text, {"helper.h": "// nothing\n", "more.cc": "int f();\n"}))
    assert [p.name for p in sources] == ["more.cc", "strategy.cc"]


@pytest.mark.parametrize("header", [
    "<fstream>", "<iostream>", "<cstdio>", "<stdio.h>", "<chrono>", "<ctime>", "<time.h>", "<random>",
    "<unistd.h>", "<fcntl.h>", "<sys/socket.h>", "<thread>", "<filesystem>", "<csignal>", "<dlfcn.h>",
    '"base/yices_solver.h"', '"../run_crest/search_harness.cc"', '"/etc/passwd"', "<z3++.h>",
    '"strategy/protocol.h"',
])
def test_forbidden_includes(tmp_path, header):
    assert code(tmp_path, "#include %s\n" % header) == "forbidden_include"


@pytest.mark.parametrize("text,expected", [
    ("int f() { asm(\"syscall\"); return 0; }\n", "forbidden_identifier"),
    ("int f() { __asm__(\"nop\"); return 0; }\n", "forbidden_identifier"),
    ("__attribute__((constructor)) static void f() {}\n", "forbidden_identifier"),
    ("long f() { return syscall(1); }\n", "forbidden_identifier"),
    ("const char* f() { return __TIME__; }\n", "forbidden_identifier"),
    ("int f() { return __builtin_ia32_rdtsc(); }\n", "forbidden_identifier"),
    ("#pragma GCC optimize(\"O3\")\n", "preprocessor_directive"),
    ("#include_next <vector>\n", "preprocessor_directive"),
    ("#define CAT(a, b) a ## b\n", "lexical_form"),
    ("#define X \\\n 1\n", "lexical_form"),
    ("??=include <stdio.h>\n", "lexical_form"),
    ("%:include <stdio.h>\n", "lexical_form"),
    ("const char* s = R\"x(a)x\";\n", "lexical_form"),
    ("/* never closed\n", "lexical_form"),
    ("const char* s = \"never closed;\n", "lexical_form"),
])
def test_forbidden_text(tmp_path, text, expected):
    assert code(tmp_path, text) == expected


def test_words_inside_comments_and_strings_are_not_code(tmp_path):
    text = '// asm syscall __attribute__\n/* #include <fstream> */\nconst char* s = "asm #pragma";\n'
    assert policy.check_sources(solution(tmp_path, text))


def test_layout_rules(tmp_path):
    directory = solution(tmp_path, STARTER.read_text())
    (directory / "notes.txt").write_text("x")
    with pytest.raises(policy.PolicyViolation) as caught:
        policy.check_sources(directory)
    assert caught.value.code == "solution_layout"
    (directory / "notes.txt").unlink()
    (directory / "sub").mkdir()
    with pytest.raises(policy.PolicyViolation):
        policy.check_sources(directory)
    (directory / "sub").rmdir()
    (directory / "link.h").symlink_to("/etc/passwd")
    with pytest.raises(policy.PolicyViolation):
        policy.check_sources(directory)
    (directory / "link.h").unlink()
    (directory / "strategy.cc").unlink()
    with pytest.raises(policy.PolicyViolation):
        policy.check_sources(directory)


@pytest.mark.parametrize("symbol", [
    "fopen", "open", "openat", "read", "write", "time", "clock", "clock_gettime", "gettimeofday", "getenv",
    "system", "fork", "execve", "popen", "socket", "connect", "dlopen", "syscall", "mmap", "mprotect",
    "rand", "srand", "getpid", "kill", "signal", "pthread_create", "printf", "puts", "exit", "_exit",
    "_ZSt4cout", "_ZNSt14basic_ifstreamIcSt11char_traitsIcEEC1EPKcSt13_Ios_Openmode",
    "_ZNSt6chrono3_V212system_clock3nowEv", "_ZNSt13random_deviceC1Ev", "_ZNSt6thread6detachEv",
    "_ZN5crest11YicesSolver5SolveERKSt3mapIjNS_5types6type_tESt4lessIjESaISt4pairIKjS3_EEERKSt6vectorIPKNS_12SymbolicPredESaISG_EEPS1_IjxS5_SaIS6_IS7_xEEE",
    "Z3_mk_context", "_ZN8frontier8HostImpl8ReadInitEv", "__libc_start_main", "environ",
])
def test_forbidden_symbols(symbol):
    assert not policy.symbol_allowed(symbol)
    with pytest.raises(policy.PolicyViolation) as caught:
        policy.check_object_listing("_ZN8frontier12MakeStrategyEv T 0 10\n%s U\n" % symbol, [".text"])
    assert caught.value.code == "forbidden_symbol"


@pytest.mark.parametrize("symbol", [
    "memcpy", "_Znwm", "_ZdlPvm", "_ZSt17__throw_bad_allocv", "_ZSt20__throw_length_errorPKc",
    "_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base", "_ZN8frontier4Host6NegateEPKNS_9ExecutionEm",
    "_ZNK8frontier4Host7programEv", "__cxa_pure_virtual", "sqrt", "_ZTVN10__cxxabiv120__si_class_type_infoE",
])
def test_allowed_symbols(symbol):
    assert policy.symbol_allowed(symbol)


def test_static_initialisation_and_entry_point():
    entry = "_ZN8frontier12MakeStrategyEv T 0 10\n"
    assert policy.check_object_listing(entry, [".text", ".rodata"]) is True
    assert policy.check_object_listing("_ZN1x1fEv T 0 10\n", [".text"]) is False
    for sections in ([".text", ".init_array"], [".ctors"], [".init_array.00100"], [".preinit_array"]):
        with pytest.raises(policy.PolicyViolation) as caught:
            policy.check_object_listing(entry, sections)
        assert caught.value.code == "static_initialization"
    with pytest.raises(policy.PolicyViolation) as caught:
        policy.check_object_listing(entry + "_GLOBAL__sub_I_strategy.cc t 0 10\n", [".text"])
    assert caught.value.code == "static_initialization"
    with pytest.raises(policy.PolicyViolation) as caught:
        policy.check_object_listing(entry + "resolver i 0 10\n", [".text"])
    assert caught.value.code == "forbidden_symbol"
    with pytest.raises(policy.PolicyViolation) as caught:
        policy.check_object_listing(entry + "main T 0 10\n", [".text"])
    assert caught.value.code == "forbidden_symbol"


def compile_object(tmp_path, text):
    source = tmp_path / "strategy.cc"
    source.write_text(text)
    obj = tmp_path / "strategy.o"
    subprocess.run(["g++", "-std=c++17", "-O2", "-fno-pic", "-I", str(REPO_TREE / "src"), "-c", str(source),
                    "-o", str(obj)], check=True, capture_output=True)
    return obj


@pytest.mark.skipif(not HAVE_COMPILER, reason="needs g++ and binutils")
def test_compiled_starter_and_shipped_strategies_pass(tmp_path):
    policy.check_objects([compile_object(tmp_path, STARTER.read_text())])
    for patch in ("reference.patch", "baseline_naive.patch"):
        work = tmp_path / patch
        (work / "src/frontier_solution").mkdir(parents=True)
        shutil.copyfile(STARTER, work / "src/frontier_solution/strategy.cc")
        subprocess.run(["patch", "-p1", "-s", "-i", str(TASK / "judge" / patch)], cwd=work, check=True)
        policy.check_sources(work / "src/frontier_solution")
        policy.check_objects([compile_object(work, (work / "src/frontier_solution/strategy.cc").read_text())])


BODY = """
#include "strategy/strategy_api.h"
%s
namespace frontier {
class S : public Strategy { public: virtual void Run(Host& host) { %s } };
Strategy* MakeStrategy() { return new S(); }
}
"""


@pytest.mark.skipif(not HAVE_COMPILER, reason="needs g++ and binutils")
@pytest.mark.parametrize("declaration,statement,expected", [
    ('extern "C" long time(long*);', "(void)time(0);", "forbidden_symbol"),
    ('extern "C" int open(const char*, int, ...);', '(void)open("/etc/passwd", 0);', "forbidden_symbol"),
    ('extern "C" char* getenv(const char*);', '(void)getenv("HOME");', "forbidden_symbol"),
    ('extern "C" int rand(void);', "(void)rand();", "forbidden_symbol"),
    ("namespace crest { struct YicesSolver { static bool Probe(); }; }", "(void)crest::YicesSolver::Probe();",
     "forbidden_symbol"),
    ("struct G { G(); int v; }; G::G() : v(host_value()) {} static G g;".replace("host_value()", "1"),
     "(void)g.v;", "static_initialization"),
])
def test_compiled_bypass_attempts_are_rejected(tmp_path, declaration, statement, expected):
    obj = compile_object(tmp_path, BODY % (declaration, statement))
    with pytest.raises(policy.PolicyViolation) as caught:
        policy.check_objects([obj])
    assert caught.value.code == expected
