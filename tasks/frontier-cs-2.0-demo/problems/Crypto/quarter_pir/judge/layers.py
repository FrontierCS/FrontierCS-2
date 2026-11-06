"""quarter_pir's isolate layer: ServerPort, the live server's one network grant.

The submitted server listens on a TCP port the judge chose; the trusted conductor connects to it
from outside the sandbox. The layer lets the server create AF_INET/AF_INET6 TCP stream sockets
(protocol 0 or IPPROTO_TCP: no UDP, raw, SCTP or MPTCP) and bind only that port (Landlock
NET_BIND_TCP); every other bind and every connect stays denied by the core.
sendto/sendmsg/sendmmsg are denied: with MSG_FASTOPEN they connect a TCP socket without
connect(), where Landlock does not look (the Go server writes to its connections with
write/writev). Landlock restricts ports, not addresses: the server may bind its port on any
local address, which relies on the Verifier pod having no network ingress or egress beyond the
pod (production Verifier pods have none). The client and the reducer get no ServerPort, hence no
network.
"""
import socket

import isolate  # the shared launcher, beside edit_scope in the Verifier base

SEND_SYSCALLS = {"sendto": 44, "sendmsg": 46, "sendmmsg": 307}   # x86_64


class ServerPort(isolate.Layer):
    relaxes = ("tcp socket", "tcp bind")

    def __init__(self, port: int):
        self.port = port

    def prepare(self, rules):
        for family in (socket.AF_INET, socket.AF_INET6):
            rules.allow_socket(family, socket.SOCK_STREAM, (0, socket.IPPROTO_TCP))
        rules.allow_tcp_port(self.port, isolate.NET_BIND_TCP)
        for number in SEND_SYSCALLS.values():
            rules.deny_syscall(number)

