This directory is task-owned CPU simulation support, separate from upstream BBQ.
The simulation_memory.sv adapter is the supplied independently authored adapter;
its SHA-256 is 529f6bf666e7e2e6df3730041c627434d91cb80017e861dae5466913cd9edbfc.
It supplies only used RAM/FIFO modes. RAM starts at zero, registers reads,
and optionally registers outputs; mixed-port read/write collisions are unspecified.
FIFO is non-showahead with registered output, checked overflow/underflow and
simultaneous operations governed by pre-edge empty/full flags. It has no reset port.
The upstream Intel-noticed primitive wrappers and vendor libraries are not bundled.
Verilator is a two-state CPU simulator; no physical frequency or area is measured.
timing.py, timing_memory.v and timing_lut6.abc are task-owned: they estimate a
minimum clock period for the generated design with sv2v 0.0.13 and Yosys 0.69
(YoWASP build 0.69.0.0.post1233), mapping to 6-input lookup tables with a fixed
ABC script and timing with a fixed unit-delay model (300 ps register overhead,
500 ps per lookup table, 1000 ps extra for a RAM read without output register,
500 ps for FIFO status flags). The memory adapters are timing black boxes: every
adapter input is a path endpoint and every adapter output a path start point.
No placement, routing, vendor library or seed is involved; the estimate orders
designs under one fixed flow and is not a physical frequency. Lookup-table,
register and memory-bit counts come from the same netlist.
A design is invalid when synthesis removes a RAM/FIFO primitive whose output it reads,
when Yosys reports multiple conflicting drivers after process conversion, or when
Verilator reports MULTIDRIVEN or BLKANDNBLK: the timed netlist must be the simulated circuit.
The original queue testbench is copied to a build directory, where precisely two
hierarchical constants are substituted with generated values for portability.
Trace parameters and observation bounds are challenge constants (not from materials).

PB generator additionally carries author-provided correction patch e79d69603ad02b7656b2571508210e28f9bc9c0922a7fc1ec2a24d72aa9327fd over upstream64e92440f71286d399ea284bfd1e8db47286af56: suppress redundant singleton-forwarded RAM reads and correct tail-pointer free-list history indices. Native SIM assertion remains unchanged and enabled. This file is not original upstream provenance.

Selected LX reference also uses an explicitly author-corrected implementation accepted on 2026-09-17. The selected interface is excised in this starter. The reference is not a byte-exact upstream implementation. Both corrections preserve native SIM assertions, operation semantics, the adapter and workload/scoring contracts; local finite tests do not establish formal proof or vendor/physical equivalence.
