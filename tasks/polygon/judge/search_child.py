"""The search process of one case, started by search_harness.py in a fresh isolate instance.

    python3 {in}/search_child.py {in}/frontier_runner.py      (stdin: the case's search payload)

It runs the fixed adapter's search entry (frontier_runner._search_main, the judge's own copy of
the starter's adapter) with both of its message descriptors on the instance's stdout pipe, the
only channel to the harness. Before any submitted module is imported, file descriptor 1 is
pointed at /dev/null, so what the search prints never reaches that channel; the start reading is
the channel's first line, written before the submitted modules are imported, and the harness
times the result's arrival itself. The instance has no network and no judge file, and no MariaDB
runs on its machine.
"""

import importlib.util
import os
import sys


def main(adapter: str) -> None:
    channel = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    spec = importlib.util.spec_from_file_location("frontier_runner", adapter)
    runner = importlib.util.module_from_spec(spec)
    sys.modules["frontier_runner"] = runner
    spec.loader.exec_module(runner)
    runner._search_main(os.dup(channel), channel)


if __name__ == "__main__":
    main(sys.argv[1])
