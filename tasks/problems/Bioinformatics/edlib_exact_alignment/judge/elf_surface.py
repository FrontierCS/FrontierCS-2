"""Read-only ELF64 little-endian inspection for the trusted verifier.

Two uses. (1) static_storage_bytes: the writable and thread-local static
storage a compiled translation unit carries, which is part of the storage
metric. (2) launch_surface / require_same_launch_surface: everything that can
make code of a build run outside the reconstruction call (constructor and
destructor tables, loader-time resolvers, and definitions that replace a
symbol the unmodified program or the C/C++ runtime calls). The reconstruction
body is the only editable text, so none of these may differ from the
reference build of the same tree.
"""
from pathlib import Path
import re
import struct

SHF_WRITE, SHF_ALLOC, SHF_TLS = 0x1, 0x2, 0x400
SHT_SYMTAB, SHT_RELA, SHT_NOBITS = 2, 4, 8
STT_GNU_IFUNC = 10
R_X86_64_IRELATIVE = 37
LAUNCH_SECTIONS = ('.preinit_array', '.init_array', '.fini_array', '.init', '.fini', '.ctors', '.dtors')


class ElfError(ValueError):
    pass


def _read(path):
    data = Path(path).read_bytes()
    if len(data) < 64 or data[:7] != b'\x7fELF\x02\x01\x01':
        raise ElfError('not an ELF64 little-endian file')
    shoff, = struct.unpack_from('<Q', data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from('<HHH', data, 0x3a)
    if shentsize != 64 or not 0 < shnum < 65280 or shoff + shnum * 64 > len(data) or shstrndx >= shnum:
        raise ElfError('unsupported section table')
    raw = [struct.unpack_from('<IIQQQQIIQQ', data, shoff + i * 64) for i in range(shnum)]

    def text(table, offset):
        start = table[4] + offset
        if table[1] == SHT_NOBITS or start >= table[4] + table[5] or table[4] + table[5] > len(data):
            raise ElfError('string offset outside its table')
        return data[start:data.index(b'\0', start)].decode('ascii', 'replace')

    sections = [{'name': text(raw[shstrndx], s[0]), 'type': s[1], 'flags': s[2], 'offset': s[4],
                 'size': s[5], 'link': s[6], 'entsize': s[9]} for s in raw]
    for section in sections:
        if section['type'] != SHT_NOBITS and section['offset'] + section['size'] > len(data):
            raise ElfError('section outside the file')
    return data, raw, sections, text


def static_storage_bytes(path):
    """Bytes of allocated writable or thread-local sections (.data, .bss, .tdata, .tbss, any name)."""
    _, _, sections, _ = _read(path)
    return sum(s['size'] for s in sections
               if s['flags'] & SHF_ALLOC and s['flags'] & (SHF_WRITE | SHF_TLS))


def launch_surface(path):
    data, raw, sections, text = _read(path)
    tables = [(i, s) for i, s in enumerate(sections) if s['type'] == SHT_SYMTAB]
    if len(tables) != 1 or tables[0][1]['entsize'] != 24:
        raise ElfError('executable has no single full symbol table')
    index, table = tables[0]
    strings = raw[table['link']]
    strong, defined, undefined, resolver = set(), set(), set(), False
    for offset in range(table['offset'], table['offset'] + table['size'], 24):
        name, info, _, shndx, _, _ = struct.unpack_from('<IBBHQQ', data, offset)
        bind, kind = info >> 4, info & 15
        if kind == STT_GNU_IFUNC:
            resolver = True
        if bind not in (1, 2) or not name:           # GLOBAL / WEAK only
            continue
        label = text(strings, name).split('@')[0]
        if shndx == 0:
            undefined.add(label)
        else:
            defined.add(label)
            if bind == 1:
                strong.add(label)
    for section in sections:
        if section['type'] == SHT_RELA and section['entsize'] == 24:
            for offset in range(section['offset'], section['offset'] + section['size'], 24):
                if struct.unpack_from('<Q', data, offset + 8)[0] & 0xffffffff == R_X86_64_IRELATIVE:
                    resolver = True
    sizes = {name: sum(s['size'] for s in sections if s['name'] == name) for name in LAUNCH_SECTIONS}
    return {'sections': sizes, 'strong': strong, 'defined': defined, 'undefined': undefined, 'resolver': resolver}


def _replaceable(name):
    """A C-level name, or a global allocation/deallocation operator.

    Names that are not identifiers (compiler-generated, such as DW.ref.*) cannot
    be referenced by any program text and are not definitions of a runtime symbol.
    """
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
        return False
    return not name.startswith('_Z') or name[:4] in ('_Znw', '_Zna', '_Zdl', '_Zda')


def require_same_launch_surface(candidate, reference, ignore=()):
    """Raise ElfError unless the candidate build can only run code inside the reconstruction call.

    ignore names symbols whose presence in the symbol table depends on whether
    they are referenced (the linker's --wrap aliases of the storage derivative).
    """
    c, r = launch_surface(candidate), launch_surface(reference)
    for surface in (c, r):
        for key in ('strong', 'defined', 'undefined'):
            surface[key] = surface[key] - set(ignore)
    if c['sections'] != r['sections']:
        raise ElfError('constructor, destructor or loader sections differ from the reference build')
    if c['resolver'] or r['resolver']:
        raise ElfError('loader-time resolver present')
    if c['strong'] != r['strong']:
        raise ElfError('externally visible definitions differ from the reference build')
    if any(_replaceable(name) for name in c['defined'] - r['defined']):
        raise ElfError('a runtime symbol is defined by the submission')
