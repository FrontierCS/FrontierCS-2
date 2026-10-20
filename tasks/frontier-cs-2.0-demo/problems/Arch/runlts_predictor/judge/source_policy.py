"""Conservative C++ source/scope and logical state audit, independent of printf claims.

Ordinary new integer state is charged at its declared width; bit-fields at their
explicit width. No process RSS/sizeof is used for the hardware charge. Exact
restoration has its independently audited original logical allocation.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess

ROOT=Path(__file__).resolve().parent
FIXED_BITS=1517027  # 1570890 - (43008 + 9360 + 1495); original non-sR state, incl RNG/threshold
CAP_BITS=1572864
REFERENCE_BITS=1570890

class InvalidSource(ValueError):
    pass

def code(text):
    # Retain strings, remove only actual comments. Do not normalize preprocessor text.
    pattern=r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')|//[^\n]*|/\*[\s\S]*?\*/'
    return re.sub(pattern,lambda m:m.group(1) or ' ',text)

def fragment_safety(text):
    # C++ translation phase two and raw strings are intentionally unsupported.
    # Check each editable fragment on its own, so comments cannot cross a frozen
    # source boundary and disable retained executable code.
    if '\\' in text or '%:' in text or '??' in text or re.search(r'(?:u8|u|U|L)?R"',text):
        raise InvalidSource('source splicing, raw strings and alternative directives are unsupported')
    i=0
    while i<len(text):
        if text.startswith('//',i):
            end=text.find('\n',i+2)
            if end<0:raise InvalidSource('line comment crosses an editable boundary')
            i=end+1
        elif text.startswith('/*',i):
            end=text.find('*/',i+2)
            if end<0:raise InvalidSource('block comment crosses an editable boundary')
            i=end+2
        elif text[i] in ('"',"'"):
            quote=text[i];end=text.find(quote,i+1)
            if end<0 or '\n' in text[i:end]:raise InvalidSource('literal crosses an editable boundary')
            i=end+1
        else:i+=1

def tokens(text):
    return re.findall(r'[A-Za-z_]\w*|\d+|[^\s]',code(text))

def split_source(text):
    contract=json.loads((ROOT/'source_contract.json').read_text())
    parts=contract['segments']; slots={}; pos=0
    if not text.startswith(parts[0]): raise InvalidSource('retained predictor scaffolding changed')
    pos=len(parts[0])
    for name,following in zip(contract['slots'],parts[1:]):
        end=text.find(following,pos)
        if end<0: raise InvalidSource('non-register predictor code changed')
        slots[name]=(text[pos:end],pos,end)
        pos=end+len(following)
    if pos!=len(text): raise InvalidSource('unexpected source after retained implementation')
    return slots,contract

def scalar_bits(typ):
    from clang.cindex import TypeKind as K, CursorKind as C
    typ=typ.get_canonical(); k=typ.kind
    widths={K.BOOL:1,K.CHAR_S:8,K.CHAR_U:8,K.SCHAR:8,K.UCHAR:8,
            K.SHORT:16,K.USHORT:16,K.INT:32,K.UINT:32,K.LONG:64,
            K.ULONG:64,K.LONGLONG:64,K.ULONGLONG:64,K.FLOAT:32,K.DOUBLE:64}
    if k in widths:return widths[k]
    if k==K.CONSTANTARRAY:return typ.element_count*scalar_bits(typ.element_type)
    spelling=typ.spelling
    if spelling.startswith('std::array<'):
        m=re.search(r',\s*(\d+)\s*>$',spelling)
        if not m:raise InvalidSource('unresolved array extent')
        return int(m[1])*scalar_bits(typ.get_template_argument_type(0))
    if k==K.ENUM:return scalar_bits(typ.get_declaration().enum_type)
    if k==K.RECORD:
        fields=list(typ.get_declaration().get_children())
        if any(f.kind==C.CXX_BASE_SPECIFIER for f in fields):raise InvalidSource('state inheritance is unsupported')
        if typ.get_declaration().kind==C.UNION_DECL:raise InvalidSource('use explicit fixed state instead of unions')
        total=0
        for f in fields:
            if f.kind==C.FIELD_DECL:
                total+=f.get_bitfield_width() if f.is_bitfield() else scalar_bits(f.type)
        if not total:raise InvalidSource('opaque state declaration')
        return total
    raise InvalidSource('persistent state needs fixed integers, bit-fields, or fixed arrays')

def audit(tree):
    tree=Path(tree)
    integrity=json.loads((ROOT/'integrity.json').read_text())
    for name,digest in integrity.items():
        p=tree/name
        if p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise InvalidSource('retained harness changed')
    files=[p for p in tree.rglob('*') if p.is_file()]
    for p in files:
        if p.is_symlink():raise InvalidSource('source symlinks are not allowed')
        name=p.relative_to(tree).as_posix()
        if name not in integrity and name!='my_cond_branch_predictor.cc' and not name.startswith('solver_extensions/'):
            raise InvalidSource('unexpected file in predictor tree')
    code_root=tree
    text=(code_root/'my_cond_branch_predictor.cc').read_text()
    slots,contract=split_source(text)
    byte_ranges=[(len(text[:lo].encode('utf-8')),len(text[:hi].encode('utf-8'))) for _,lo,hi in slots.values()]
    ext=[p for p in files if p.relative_to(tree).as_posix().startswith('solver_extensions/')]
    fragments=[v[0] for v in slots.values()]+[p.read_text() for p in ext]
    for fragment in fragments:fragment_safety(fragment)
    added='\n'.join(fragments)
    if '\\' in added or '%:' in added or '??' in added:
        raise InvalidSource('source splicing and alternative preprocessing tokens are unsupported')
    stripped=code(added)
    # Headers may only be new in-tree utility headers. Includes cannot expose
    # simulator objects or private files; no arbitrary macro or linker rewriting.
    for line in stripped.splitlines():
        if line.lstrip().startswith('#'):
            if not re.fullmatch(r'\s*#\s*include\s+"solver_extensions/[A-Za-z0-9_/.-]+"\s*',line):
                raise InvalidSource('only local utility includes are permitted in added code')
            name=line.split('"')[1]
            if '..' in Path(name).parts or not (code_root/name).resolve().is_relative_to((code_root/'solver_extensions').resolve()) or not (code_root/name).is_file():
                raise InvalidSource('utility include escapes the source surface')
    forbidden=r'\b(?:asm|__asm__|__attribute__|extern|thread_local|volatile|reinterpret_cast|const_cast|new|delete|malloc|calloc|realloc|free|printf|fprintf|sprintf|puts|fopen|open|read|write|system|exec\w*|fork|dlsym|syscall|mmap|getenv|ifstream|ofstream|fstream|cout|cerr|clog|exit|abort|longjmp|signal|raise|atexit|dlopen|memcpy|memmove|memset|vector|unordered_map|function|shared_ptr|unique_ptr)\b'
    if re.search(r'\b__\w*|\btemplate\b',stripped):raise InvalidSource('compiler intrinsics and new templates are outside the source contract')
    if re.search(forbidden,stripped):raise InvalidSource('unsupported side effect or dynamic state in register implementation')
    if re.search(r'\\|\?\?[/=\'()!<>-]',stripped):raise InvalidSource('escaped source tokens are unsupported')
    # Existing state is read-only except for the prescribed SC sum output and RNG.
    # Referencing protected non-sR structures is not part of this reconstruction.
    if all(tokens(slots[n][0])==tokens(v) for n,v in contract['original_slots'].items()) and not ext:
        return {'logical_bits':REFERENCE_BITS,'method':'audited original logical allocation'}
    # Exact inactive starter and comment-only changes are a valid naive policy.
    if not ext and all(not tokens(v[0]) for k,v in slots.items() if k!='prediction') and tokens(slots['prediction'][0])==tokens('int extra_weight = 0; pv.LSUM = 0;'):
        return {'logical_bits':FIXED_BITS,'method':'audited inactive register contribution'}
    from clang import cindex as c
    include_query=subprocess.run(['g++','-E','-x','c++','-','-v'],input='',text=True,capture_output=True,timeout=10)
    paths=include_query.stderr.split('#include <...> search starts here:')[1].split('End of search list.')[0].splitlines()
    args=['-std=c++17','-I'+str(code_root),*[x for p in paths if p.strip() for x in ['-isystem',p.strip()]]]
    tu=c.Index.create().parse(str(code_root/'my_cond_branch_predictor.cc'),args=args)
    if any(d.severity>=c.Diagnostic.Error for d in tu.diagnostics):raise InvalidSource('C++ source failed semantic audit')
    def introduced(cur):
        if not cur.location.file:return False
        path=Path(cur.location.file.name).resolve()
        if path== (code_root/'my_cond_branch_predictor.cc').resolve():
            return any(lo<=cur.location.offset<hi for lo,hi in byte_ranges)
        return path.is_relative_to((code_root/'solver_extensions').resolve())
    bits=FIXED_BITS
    def walk(cur):
        nonlocal bits
        if cur.location.file and not Path(cur.location.file.name).resolve().is_relative_to(tree.resolve()):return
        fresh=introduced(cur)
        if fresh:
            if cur.kind in (c.CursorKind.LAMBDA_EXPR,c.CursorKind.FUNCTION_TEMPLATE,c.CursorKind.CLASS_TEMPLATE,c.CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION):
                raise InvalidSource('new templates and lambdas are unsupported')
            if cur.kind in (c.CursorKind.FUNCTION_DECL,c.CursorKind.CXX_METHOD):
                if cur.result_type.get_canonical().kind in (c.TypeKind.LVALUEREFERENCE,c.TypeKind.RVALUEREFERENCE,c.TypeKind.POINTER,c.TypeKind.MEMBERPOINTER):
                    raise InvalidSource('new pointer/reference-returning functions are unsupported')
            if cur.kind in (c.CursorKind.DECL_REF_EXPR,c.CursorKind.MEMBER_REF_EXPR):
                target=cur.referenced
                if target and target.location.file:
                    inside=Path(target.location.file.name).resolve().is_relative_to(tree.resolve())
                    allowed={'PC','seq_no','piece','dst_reg','value','resolveDir',
                             'hist_to_use','active_hist','pv','LSUM','extra_weight',
                             'pred_time_predict','MYRANDOM','ctrupdate'}
                    if inside and not introduced(target) and target.spelling not in allowed:
                        raise InvalidSource('register code references protected predictor state')
            if cur.kind==c.CursorKind.CALL_EXPR:
                target=cur.referenced
                if target and target.spelling.startswith('__builtin_'):
                    raise InvalidSource('compiler builtin calls are unsupported')
                if not target or not target.location.file:
                    raise InvalidSource('unresolved or indirect call in register code')
                if not Path(target.location.file.name).resolve().is_relative_to(tree.resolve()):
                    allowed={'min','max','abs','labs','llabs','fill','at','size','begin','end',
                             'operator[]','operator=','array','__assert_fail'}
                    if target.spelling not in allowed and not target.spelling.startswith('array<'):
                        raise InvalidSource('unsupported external call in register code')
            if cur.kind in (c.CursorKind.CSTYLE_CAST_EXPR,c.CursorKind.CXX_REINTERPRET_CAST_EXPR,c.CursorKind.CXX_CONST_CAST_EXPR,c.CursorKind.CXX_NEW_EXPR,c.CursorKind.CXX_DELETE_EXPR):
                raise InvalidSource('unsafe cast or dynamic allocation')
            if cur.kind in (c.CursorKind.VAR_DECL,c.CursorKind.FIELD_DECL,c.CursorKind.PARM_DECL):
                if cur.type.get_canonical().kind in (c.TypeKind.POINTER,c.TypeKind.MEMBERPOINTER,c.TypeKind.RVALUEREFERENCE,c.TypeKind.CONSTANTARRAY,c.TypeKind.INCOMPLETEARRAY,c.TypeKind.VARIABLEARRAY):raise InvalidSource('raw pointers are not permitted in new state/code')
            if cur.kind in (c.CursorKind.VAR_DECL,c.CursorKind.FIELD_DECL,c.CursorKind.PARM_DECL):
                typ=cur.type.get_canonical()
                if typ.kind==c.TypeKind.LVALUEREFERENCE and not typ.get_pointee().is_const_qualified():
                    raise InvalidSource('new mutable reference aliases are unsupported')
            if cur.kind in (c.CursorKind.CXX_STATIC_CAST_EXPR,c.CursorKind.CXX_FUNCTIONAL_CAST_EXPR):
                if cur.type.get_canonical().kind in (c.TypeKind.POINTER,c.TypeKind.MEMBERPOINTER,c.TypeKind.LVALUEREFERENCE,c.TypeKind.RVALUEREFERENCE):
                    raise InvalidSource('pointer/reference casts are unsupported')
            if cur.kind==c.CursorKind.ARRAY_SUBSCRIPT_EXPR:
                raise InvalidSource('use checked std::array indexing in new code')
            if cur.kind in (c.CursorKind.BINARY_OPERATOR,c.CursorKind.COMPOUND_ASSIGNMENT_OPERATOR,c.CursorKind.UNARY_OPERATOR):
                ts=[t.spelling for t in cur.get_tokens()]
                children=list(cur.get_children())
                if children and any(t in ('=','+=','-=','*=','/=','%=','|=','&=','^=','<<=','>>=','++','--') for t in ts):
                    lhs=children[0]
                    refs=[x.referenced for x in lhs.walk_preorder() if x.kind in (c.CursorKind.DECL_REF_EXPR,c.CursorKind.MEMBER_REF_EXPR) and x.referenced and x.referenced.kind in (c.CursorKind.VAR_DECL,c.CursorKind.FIELD_DECL,c.CursorKind.PARM_DECL)]
                    # Member writes may update only new state or the SC sum.
                    if refs and not introduced(refs[0]) and refs[0].spelling not in ('LSUM','extra_weight'):
                        raise InvalidSource('retained prediction inputs/state are read-only')
            if cur.kind==c.CursorKind.VAR_DECL:
                static=cur.storage_class==c.StorageClass.STATIC
                global_=cur.semantic_parent.kind in (c.CursorKind.NAMESPACE,c.CursorKind.TRANSLATION_UNIT)
                if static or global_:
                    # Scalar constexpr constants synthesize into logic.
                    if not ('constexpr' in [t.spelling for t in cur.get_tokens()] and cur.type.get_canonical().kind in (c.TypeKind.INT,c.TypeKind.UINT,c.TypeKind.LONG,c.TypeKind.ULONG)):
                        bits+=scalar_bits(cur.type)
            if cur.kind==c.CursorKind.FIELD_DECL and cur.semantic_parent.spelling in ('cbp_hist_t','RUNLTS'):
                # One active copy is charged; duplicated in-flight checkpoints are exempt.
                bits+=cur.get_bitfield_width() if cur.is_bitfield() else scalar_bits(cur.type)
            if cur.kind==c.CursorKind.UNARY_OPERATOR:
                toks=[t.spelling for t in cur.get_tokens()]
                if toks and toks[0] in ('&','*'):raise InvalidSource('address manipulation is outside the register contribution')
        for child in cur.get_children():walk(child)
    walk(tu.cursor)
    if bits>CAP_BITS:raise InvalidSource('full logical predictor storage exceeds 192 KiB')
    return {'logical_bits':bits,'method':'fixed non-register allocation plus declared-width state audit'}

if __name__=='__main__':
    import sys
    try:
        report=audit(Path(sys.argv[1]));report['valid']=True
    except InvalidSource as exc:
        report={'valid':False,'reason':str(exc)}
    except Exception:
        report={'infrastructure_error':True}
    print(json.dumps(report))
