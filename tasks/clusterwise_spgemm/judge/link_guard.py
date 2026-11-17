"""Link-level separation of the locked program from the solver's kernel object.

The judge compiles two translation units. The main object (the retained
program's main, argument handling, loading, reordering, CSR construction, flop
count and output) includes no editable header. The kernel object includes the
editable headers and defines the fixed entry point. Both are linked with the
main object first. A definition in the kernel object would still replace
locked code if the linker bound the main object's calls to it, so the kernel
object may not define:

* a symbol the main object defines, unless both are vague-linkage (COMDAT)
  definitions, the kernel's in a group whose signature the main object also
  has, and the linker map records the kernel's section as discarded. Such a
  duplicate is a template or inline instantiation; the main object is linked
  first, so its group is kept and no kernel byte of it reaches the program.
  (Section bytes are not compared: the same instantiation legitimately
  compiles differently in different translation units.) A strong explicit
  specialization of a locked template is rejected; an `inline` one never
  reaches the main object's calls;
* a symbol that any other object or archive the link loads (the main object,
  GTgraph, sprng, the C runtime, libgcc) references or defines, other than the
  entry point, unless it is a shared-library symbol admitted by the next rule;
* a symbol a linked shared library exports (from the linker map, plus the
  interpreter), except the replaceable global allocation functions and a COMDAT
  definition that the pristine kernel translation unit, compiled with the same
  flags, emits with identical bytes and relocations.

Overloads, macros and declarations in editable headers cannot reach the main
object, which never sees them.
"""
from pathlib import Path
import re
import struct
import subprocess

# int frontier_clustered_spgemm(CSR<int64_t,double>&, CSR<int64_t,double>&, int64_t,
#                               int, int, FrontierClusterConsumer, void*)
ENTRY_POINTS = frozenset({"_Z25frontier_clustered_spgemmR3CSRIldES1_liiPFiPK22FrontierClusterProductPvES5_"})

# C++ [replacement.functions]: the only library functions a program may replace
# are the global allocation and deallocation functions.
REPLACEABLE = frozenset(
    ["_Znwm", "_Znam", "_ZnwmRKSt9nothrow_t", "_ZnamRKSt9nothrow_t"]
    + [base + suffix for base in ("_ZdlPv", "_ZdaPv") for suffix in ("", "m", "RKSt9nothrow_t")]
    + ["_ZnwmSt11align_val_t", "_ZnamSt11align_val_t",
       "_ZnwmSt11align_val_tRKSt9nothrow_t", "_ZnamSt11align_val_tRKSt9nothrow_t"]
    + [base + suffix for base in ("_ZdlPv", "_ZdaPv")
       for suffix in ("St11align_val_t", "mSt11align_val_t", "St11align_val_tRKSt9nothrow_t")])

REDEFINES_LOCKED = "program redefines locked framework code"
DEFINES_LIBRARY = "program defines a library symbol"


class InspectionError(RuntimeError):
    """The trusted toolchain output could not be inspected (judge fault)."""


def _readelf(args):
    process = subprocess.run(["readelf", "-W", *args], capture_output=True, text=True, timeout=120)
    if process.returncode:
        raise InspectionError("symbol inspection failed")
    return process.stdout


def _listed_symbols(text):
    """(name without version, defined) for every global, weak or unique row of a
    readelf --syms/--dyn-syms listing (archives list every member)."""
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 8 or not fields[0].endswith(":") or fields[4] not in ("GLOBAL", "WEAK", "UNIQUE"):
            continue
        if fields[6] == "Ndx":
            continue
        rows.append((fields[7].split("@", 1)[0], fields[6] != "UND"))
    return rows


class ElfObject:
    """Minimal ELF64 little-endian relocatable reader: symbols, COMDAT groups, RELA."""

    def __init__(self, path):
        try:
            self._parse(Path(path).read_bytes())
        except (struct.error, IndexError, ValueError, UnicodeDecodeError) as error:
            raise InspectionError("symbol inspection failed") from error

    def _parse(self, data):
        self.data = data
        if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1 or struct.unpack_from("<H", data, 16)[0] != 1:
            raise ValueError("not an ELF64 little-endian relocatable object")
        shoff = struct.unpack_from("<Q", data, 0x28)[0]
        shnum, shstrndx = struct.unpack_from("<HH", data, 0x3C)
        first = struct.unpack_from("<IIQQQQIIQQ", data, shoff)
        if shnum == 0: shnum = first[5]
        if shstrndx == 0xFFFF: shstrndx = first[6]
        self.sections = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + 64 * i) for i in range(shnum)]
        self.section_names = [self._string(self.sections[shstrndx], s[0]) for s in self.sections]
        tables = [i for i, s in enumerate(self.sections) if s[1] == 2]
        if len(tables) != 1:
            raise ValueError("expected one symbol table")
        self.symtab = tables[0]
        xindex = next((s for s in self.sections if s[1] == 18 and s[6] == self.symtab), None)
        table = self.sections[self.symtab]
        strtab = self.sections[table[6]]
        self.symbols = []
        for i in range(table[5] // 24):
            name, info, other, shndx, value, size = struct.unpack_from("<IBBHQQ", data, table[4] + 24 * i)
            if shndx == 0xFFFF and xindex:
                shndx = struct.unpack_from("<I", data, xindex[4] + 4 * i)[0]
            self.symbols.append((self._string(strtab, name), info >> 4, info & 15, shndx, value, size))
        # COMDAT member section index -> group signature; and every signature.
        self.group_of = {}
        self.signatures = set()
        self.relocations = {}
        for s in self.sections:
            if s[1] == 17:
                words = struct.unpack_from("<%dI" % (s[5] // 4), data, s[4])
                if not words or not words[0] & 1:
                    continue
                if s[6] != self.symtab:
                    raise ValueError("group signature outside the symbol table")
                signature = self._signature(s[7])
                self.signatures.add(signature)
                for member in words[1:]:
                    self.group_of[member] = signature
            elif s[1] == 4:
                self.relocations[s[7]] = [struct.unpack_from("<QQq", data, s[4] + 24 * i) for i in range(s[5] // 24)]

    def _string(self, table, offset):
        start = table[4] + offset
        return self.data[start:self.data.index(b"\0", start)].decode("latin-1")

    def _signature(self, index):
        name, _, kind, shndx, _, _ = self.symbols[index]
        return self.section_names[shndx] if kind == 3 else name

    def _target(self, index):
        name, _, kind, shndx, _, _ = self.symbols[index]
        if kind == 3:
            return ("section", self.section_names[shndx] if shndx < len(self.sections) else shndx)
        return ("symbol", name)

    def defined(self):
        """Global, weak and unique definitions, keyed by name."""
        return {s[0]: s for s in self.symbols if s[1] in (1, 2, 10) and s[3] != 0}

    def undefined(self):
        return {s[0] for s in self.symbols[1:] if s[1] in (1, 2, 10) and s[3] == 0}

    def group(self, symbol):
        """Signature of the COMDAT group holding a definition, or None."""
        shndx = symbol[3]
        if shndx == 0 or shndx >= 0xFF00 or shndx >= len(self.sections):
            return None
        return self.group_of.get(shndx)

    def definition(self, symbol):
        """Section bytes and relocations of a COMDAT definition, or None."""
        name, bind, kind, shndx, value, size = symbol
        if self.group(symbol) is None:
            return None
        s = self.sections[shndx]
        body = b"" if s[1] == 8 else self.data[s[4]:s[4] + s[5]]
        relocations = sorted((offset, info & 0xFFFFFFFF, self._target(info >> 32), addend)
                             for offset, info, addend in self.relocations.get(shndx, ()))
        return (self.section_names[shndx], s[1], s[2], s[5], body, bind, kind, value, size, tuple(relocations))


def linked_inputs(linkmap, directory):
    """(shared libraries, relocatable objects and archives) the linker map loads."""
    shared, static = set(), set()
    for line in Path(linkmap).read_text(errors="replace").splitlines():
        fields = line.split()
        if len(fields) != 2 or fields[0] != "LOAD":
            continue
        path = Path(fields[1]) if Path(fields[1]).is_absolute() else Path(directory) / fields[1]
        try:
            with path.open("rb") as handle:
                magic = handle.read(18)
        except OSError:
            continue
        if magic[:4] == b"\x7fELF" and int.from_bytes(magic[16:18], "little") == 3:
            shared.add(path.resolve())
        elif (magic[:4] == b"\x7fELF" and int.from_bytes(magic[16:18], "little") == 1) or magic[:8] == b"!<arch>\n":
            static.add(path.resolve())
    if not shared or not static:
        raise InspectionError("symbol inspection failed")
    return shared, static


def discarded_sections(linkmap, directory):
    """(section name, resolved input path) the linker map lists as discarded."""
    lines = Path(linkmap).read_text(errors="replace").splitlines()
    try:
        start = lines.index("Discarded input sections")
    except ValueError as error:
        raise InspectionError("symbol inspection failed") from error
    discarded, pending = set(), None
    for line in lines[start + 1:]:
        if line == "Memory Configuration":
            break
        fields = line.split()
        if not fields:
            continue
        if not line[:2].isspace() and line.startswith(" "):
            pending = fields[0]
            fields = fields[1:]
        if len(fields) == 3 and pending and fields[0].startswith("0x"):
            path = Path(fields[2]) if Path(fields[2]).is_absolute() else Path(directory) / fields[2]
            discarded.add((pending, path.resolve()))
            pending = None
    return discarded


def library_exports(libraries, binary):
    """Symbols exported by the loaded shared libraries, plus the interpreter."""
    libraries = set(libraries)
    for match in re.finditer(r"interpreter: ([^\]]+)\]", _readelf(["-l", str(binary)])):
        libraries.add(Path(match.group(1)).resolve())
    exported = set()
    for library in sorted(libraries):
        exported.update(name for name, defined in _listed_symbols(_readelf(["--dyn-syms", str(library)])) if defined)
    return exported


def locked_input_symbols(paths):
    """Every global symbol named (defined or referenced) by locked objects/archives."""
    names = set()
    for path in sorted(paths):
        names.update(name for name, _ in _listed_symbols(_readelf(["--syms", str(path)])))
    return names


def check(kernel_object, main_object, baseline_object, linkmap, binary, directory):
    """Return [(reason, symbol)] for every inadmissible kernel definition.

    `directory` is the link's working directory, against which the map's
    relative paths resolve. Every relocatable object and archive the link
    loaded other than the kernel object (the main object, GTgraph, sprng, the
    C runtime and libgcc) is locked code.
    """
    kernel, main, baseline = (ElfObject(p) for p in (kernel_object, main_object, baseline_object))
    main_defined, main_undefined = main.defined(), main.undefined()
    if not ENTRY_POINTS <= main_undefined:
        raise InspectionError("main object does not call the kernel entry point")
    shared, static = linked_inputs(linkmap, directory)
    if Path(kernel_object).resolve() not in static or Path(main_object).resolve() not in static:
        raise InspectionError("linker map does not record the compiled objects")
    kernel_path = Path(kernel_object).resolve()
    discarded = discarded_sections(linkmap, directory)
    pristine = {name: baseline.definition(symbol) for name, symbol in baseline.defined().items()}
    locked = locked_input_symbols(static - {Path(kernel_object).resolve()}) - ENTRY_POINTS
    exported = library_exports(shared, binary)
    reasons = []
    for name, symbol in sorted(kernel.defined().items()):
        if name in ENTRY_POINTS or name in REPLACEABLE:
            continue
        unversioned = name.split("@", 1)[0]
        if name in main_defined:
            theirs = main_defined[name]
            if (kernel.group(symbol) is not None and kernel.group(symbol) in main.signatures
                    and main.group(theirs) is not None
                    and (kernel.section_names[symbol[3]], kernel_path) in discarded):
                continue
            reasons.append((REDEFINES_LOCKED, name))
        elif unversioned in exported:
            definition = kernel.definition(symbol)
            if definition is None or pristine.get(name) != definition:
                reasons.append((DEFINES_LIBRARY, name))
        elif unversioned in locked:
            reasons.append((REDEFINES_LOCKED, name))
    return reasons
