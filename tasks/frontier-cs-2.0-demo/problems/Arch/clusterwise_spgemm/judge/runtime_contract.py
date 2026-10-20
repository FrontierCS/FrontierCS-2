"""Config-derived scientific limits and logical CPU selection."""

from dataclasses import dataclass
from fractions import Fraction
from itertools import zip_longest


@dataclass(frozen=True, order=True)
class LogicalCPU:
    socket: int
    numa: int
    core: int
    cpu: int


def select_logical_cpus(topology, count):
    """Spread across NUMA domains and physical cores before using SMT siblings."""
    if type(count) is not int or count < 1:
        raise ValueError("logical CPU count must be a positive integer")
    if len({entry.cpu for entry in topology}) != len(topology):
        raise ValueError("duplicate logical CPU")
    domains = {}
    for entry in sorted(topology):
        if any(type(x) is not int or x < 0
               for x in (entry.socket, entry.numa, entry.core, entry.cpu)):
            raise ValueError("incomplete CPU topology")
        cores = domains.setdefault((entry.socket, entry.numa), {})
        cores.setdefault(entry.core, []).append(entry.cpu)
    # Sibling layers must be global: a small domain must not consume SMT
    # siblings while another domain still has unused physical cores.
    spread = []
    siblings = max((len(cpus) for cores in domains.values() for cpus in cores.values()), default=0)
    for sibling in range(siblings):
        domain_orders = [[cpus[sibling] for cpus in cores.values() if sibling < len(cpus)]
                         for cores in domains.values()]
        spread.extend(cpu for layer in zip_longest(*domain_orders)
                      for cpu in layer if cpu is not None)
    if len(spread) < count:
        raise ValueError("insufficient allowed logical CPUs")
    return spread[:count]


def validate_cpu_quota(cpu_max, ancestor_cpu_max, cpus):
    """Check cgroup-v2 quota readbacks; cpuset and quota are independent limits."""
    def parse(value):
        fields = value.split()
        if len(fields) != 2:
            raise ValueError("malformed cpu.max")
        try:
            period = int(fields[1])
            quota = None if fields[0] == "max" else int(fields[0])
        except ValueError as error:
            raise ValueError("malformed cpu.max") from error
        if period < 1 or (quota is not None and quota < 1):
            raise ValueError("nonpositive cpu.max")
        return None if quota is None else Fraction(quota, period)

    if parse(cpu_max) != cpus:
        raise ValueError("Verifier CPU quota differs from scientific config")
    for value in ancestor_cpu_max:
        limit = parse(value)
        if limit is not None and limit < cpus:
            raise ValueError("ancestor CPU quota reduces scientific capacity")


@dataclass(frozen=True)
class RuntimeLimits:
    threads: int
    process_wall_seconds: int
    process_cpu_seconds: int
    child_address_space_bytes: int
    max_product_bytes: int
    evaluator_seconds: int
    build_seconds: int

    @classmethod
    def from_config(cls, config):
        def positive(section, key):
            value = config[section][key]
            if type(value) is not int or value < 1:
                raise ValueError(f"{section}.{key} must be a positive integer")
            return value

        threads = positive("environment", "cpus")
        wall = positive("benchmark", "process_wall_seconds")
        memory = positive("environment", "memory_mb") * 1024**2
        child_memory = positive("benchmark", "child_address_space_mb") * 1024**2
        product = positive("benchmark", "max_product_bytes")
        reserve = positive("benchmark", "evaluator_reserve_seconds")
        outer = positive("runtime", "timeout_seconds")
        build = positive("environment", "build_timeout_seconds")
        if child_memory + product >= memory or outer - reserve <= build:
            raise ValueError("insufficient parent memory or evaluator time headroom")
        return cls(threads, wall, threads * wall, child_memory, product,
                   outer - reserve, build)

    def environment(self):
        return {"OMP_NUM_THREADS": str(self.threads),
                "OMP_THREAD_LIMIT": str(self.threads), "OMP_DYNAMIC": "FALSE",
                "OMP_WAIT_POLICY": "PASSIVE", "OMP_PROC_BIND": "FALSE"}


def require_hydrated_workloads(plan):
    """A metadata catalog is never a runnable data manifest."""
    import re

    matrices = plan["matrices"]
    if len(matrices) != 8 or {x["suite"] for x in matrices} != {"dev", "final"}:
        raise ValueError("incomplete workload design")
    for row in matrices:
        for key in ("raw_archive_sha256", "raw_matrix_sha256"):
            value = row.get(key)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("workload raw assets are not hydrated and locked")
        orders = row.get("ordering_sha256")
        if not isinstance(orders, dict) or set(orders) != set(row["orderings"]) or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in orders.values()
        ):
            raise ValueError("workload orders are not hydrated and locked")
    for key in ("collection", "raw_matrix_sha256"):
        dev = {x[key] for x in matrices if x["suite"] == "dev"}
        final = {x[key] for x in matrices if x["suite"] == "final"}
        if len(dev) != 4 or len(final) != 4 or dev & final:
            raise ValueError("suite parent collision or duplicate")
