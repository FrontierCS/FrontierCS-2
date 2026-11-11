"""Layers the isolate tests hand to the launcher (tests/test_isolate.py).

The launcher rebuilds each layer by executing the file its class is defined in
(isolate._load_layers), with no test harness on its path; so they live here, apart from the
test module and its deployment_test_guard import."""

from __future__ import annotations

import dataclasses
import os
import socket

import isolate


class Recording(isolate.Layer):
    """Appends each hook it runs to a judge file (the launcher's and the command's processes
    are unconfined until the core confines the command)."""

    def __init__(self, name: str, log: str):
        self.name, self.log = name, log

    def note(self, hook: str) -> None:
        with open(self.log, "a") as handle:
            handle.write(f"{self.name} {hook}\n")

    def prepare(self, rules):
        self.note("prepare")

    def preexec(self):
        self.note("preexec")

    def result(self, result, run):
        self.note("result")
        return dataclasses.replace(
            result,
            measurements={
                **result.measurements,
                self.name: run.rusage["ru_maxrss"] > 0,
            },
        )


class Pinned(isolate.Layer):
    """Child-side setup and a tightening rule: one CPU, and no way to leave it."""

    def __init__(self, cpu: int):
        self.cpu = cpu

    def prepare(self, rules):
        rules.deny_syscall(203)  # sched_setaffinity

    def preexec(self):
        os.sched_setaffinity(0, [self.cpu])


class KeptFile(isolate.Layer):
    """A launcher fd the command keeps."""

    def __init__(self, path: str):
        self.path = path

    def prepare(self, rules):
        descriptor = os.open(self.path, os.O_RDONLY)
        os.dup2(descriptor, 100)
        os.close(descriptor)
        rules.keep_fd(100)


class Connect(isolate.Layer):
    """A relaxing layer: TCP stream sockets that may connect to one port."""

    relaxes = ("tcp connect",)

    def __init__(self, port: int):
        self.port = port

    def prepare(self, rules):
        for family in (socket.AF_INET, socket.AF_INET6):
            rules.allow_socket(family, socket.SOCK_STREAM, (0, socket.IPPROTO_TCP))
        rules.allow_tcp_port(self.port, isolate.NET_CONNECT_TCP)


class Undeclared(Connect):
    relaxes = ()
