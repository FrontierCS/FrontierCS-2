"""Source-only preflight. Submitted Python is never executed by this inspector."""
import ast
import builtins
import importlib
import json
from pathlib import Path
import re
import sys

SCHEMA = "ragen.candidate-diagnostic/v2"
STRUCTURE_GUIDANCE = {'row_count': 'Return one nonempty training row per input trajectory, preserving row order.',
 'empty_response': 'Return at least two input tokens so shifted responses are nonempty.',
 'tensor_shapes': 'Return input_ids, attention_mask and position_ids with shape [N, L]; responses, '
                  'loss_mask, rm_scores and original_rm_scores must have shape [N, L-1].',
 'environment_identity': 'Preserve each input trajectory env_id and its row order.',
 'position_ids': 'Use one-based cumulative attention positions or zero-based cumulative attention '
                 'positions clamped at zero.',
 'binary_masks': 'Return binary attention_mask and loss_mask values.',
 'padding_tokens': 'Use the tokenizer padding token wherever attention_mask is zero.',
 'loss_padding': 'Do not select padded source-token positions with loss_mask.',
 'terminal_reward': 'Place terminal rewards in the final column of the left-padded training batch.',
 'required_fields': 'Include input_ids, attention_mask, position_ids, responses, loss_mask, '
                    'rm_scores and original_rm_scores in the training DataProto.',
 'shifted_responses': 'Return responses equal to input_ids[:, 1:] for the completed training '
                      'batch.',
 'finite_tensors': 'Return only finite values in every training tensor.',
 'left_padding': 'Left-pad completed training trajectories and keep attention_mask consistent with '
                 'that padding.',
 'trainable_tokens': 'Provide at least one trainable source-token position for each trajectory.',
 'group_identity': 'Preserve each input trajectory group_id and its row order.',
 'executed_action': 'Return only actions accepted by the environment action contract.',
 'training_reward': 'Preserve genuine environment rewards plus the configured format penalty under '
                    'identity normalization.',
 'original_reward': 'Preserve raw environment rewards or the upstream penalty-adjusted rewards in '
                    'original_rm_scores.',
 'initial_state_group': 'Preserve same-initial-state rollout groups and their configured '
                        'multiplicity.'}

MAX_DIAGNOSTIC_BYTES = 2048
MAX_SOURCE_LINES = 262144
SOURCE_PATTERN = re.compile(
    r"ragen/llm_agent/(?:ctx_manager\.py|solver_extensions/"
    r"(?:[A-Za-z_][A-Za-z0-9_]*/)*[A-Za-z_][A-Za-z0-9_]*\.py)\Z"
)
GUIDANCE = {
    "action_provenance": "Candidate structure rejected: actions and response history must match the actual model generation.",
    "prompt_provenance": "Candidate structure rejected: model input must match the fixed instruction and genuine environment/model history.",
    "assistant_provenance": "Candidate structure rejected: training conversation tokens must match the genuine normalized model/environment history.",
    "unbound_eager_annotation": "Check names and imports in the eager annotation at this source location.",
    "syntax_error": "Fix Python syntax at this source location.",
}
CONTEXT = "ragen/llm_agent/ctx_manager.py"
EXTENSIONS = "ragen.llm_agent.solver_extensions"
TYPING_NAMES = frozenset({"List", "Dict", "Any", "Optional", "Union", "Tuple", "Set", "FrozenSet"})


class CandidateSourceError(Exception):
    def __init__(self, diagnostic):
        self.diagnostic = public_diagnostic(diagnostic)
        super().__init__(self.diagnostic["message"])


class Unproven(Exception):
    """Unsupported source remains eligible for ordinary execution."""


def public_diagnostic(value):
    if isinstance(value, dict) and value.get("category") == "dataproto_contract":
        if (set(value) != {"schema", "category", "reason", "message"}
                or value["schema"] != SCHEMA
                or value.get("reason") not in STRUCTURE_GUIDANCE
                or value["message"] != STRUCTURE_GUIDANCE[value["reason"]]):
            raise ValueError("invalid DataProto contract diagnostic")
        return dict(value)
    if not isinstance(value, dict) or set(value) != {"schema", "category", "source", "line", "message"}:
        raise ValueError("candidate diagnostic has invalid fields")
    category, source, line = value["category"], value["source"], value["line"]
    if value["schema"] != SCHEMA or not isinstance(category, str) or category not in GUIDANCE:
        raise ValueError("candidate diagnostic has an invalid category")
    if not isinstance(source, str) or len(source) > 192 or SOURCE_PATTERN.fullmatch(source) is None:
        raise ValueError("candidate diagnostic source is not a public solver path")
    if type(line) is not int or not 1 <= line <= MAX_SOURCE_LINES:
        raise ValueError("candidate diagnostic has an invalid line")
    if value["message"] != GUIDANCE[category]:
        raise ValueError("candidate diagnostic message is not approved")
    return dict(value)


def structure_diagnostic(reason):
    return public_diagnostic({"schema": SCHEMA, "category": "dataproto_contract",
                              "reason": reason, "message": STRUCTURE_GUIDANCE[reason]})


def diagnostic(category, source, line):
    return public_diagnostic({"schema": SCHEMA, "category": category, "source": source,
                              "line": line, "message": GUIDANCE[category]})


def sources(workdir):
    root = Path(workdir).resolve()
    paths = [root / CONTEXT, *sorted((root / "ragen/llm_agent/solver_extensions").rglob("*.py"))]
    result = {}
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if (path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)
                and len(relative) <= 192 and SOURCE_PATTERN.fullmatch(relative)):
            result[relative] = path.read_bytes()
    return result


def literal(node):
    # No evaluation of calls, attributes, user objects or overloaded operators.
    if isinstance(node, ast.Constant):
        return True
    return isinstance(node, (ast.Tuple, ast.List)) and all(literal(x) for x in node.elts)


class DeclarationProof:
    def __init__(self, trees, pristine):
        self.trees = trees
        base = ast.parse(pristine)
        self.bootstrap = []
        for node in base.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                break
            self.bootstrap.append(node)
        self.allowed_imports = set()
        for node in self.bootstrap:
            if isinstance(node, ast.Import):
                self.allowed_imports.update((x.name, None) for x in node.names)
            elif isinstance(node, ast.ImportFrom):
                self.allowed_imports.update((node.module, x.name) for x in node.names)
        self.allowed_imports.update(("typing", x) for x in TYPING_NAMES)
        self.active = set()
        self.complete = {}

    def annotation(self, node, scope, source, missing):
        if node is None or isinstance(node, ast.Constant):
            return "value"
        if isinstance(node, ast.Name):
            # Compiler-created module/class bindings are outside this proof.
            if node.id.startswith("__"):
                raise Unproven()
            if node.id not in scope:
                missing.append(diagnostic("unbound_eager_annotation", source, node.lineno))
                return "unbound"
            return scope[node.id]
        if isinstance(node, ast.Attribute):
            root = self.annotation(node.value, scope, source, missing)
            if root == "module:torch" and node.attr == "Tensor":
                return "type"
            raise Unproven()
        if isinstance(node, ast.Subscript):
            kind = self.annotation(node.value, scope, source, missing)
            parts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            arity = {"List": 1, "Dict": 2, "Tuple": None, "list": 1, "dict": 2, "tuple": None}
            generic = kind.removeprefix("generic:")
            if kind != "unbound" and (generic not in arity or
                    (arity[generic] is not None and len(parts) != arity[generic])):
                raise Unproven()
            for part in parts:
                if not isinstance(part, (ast.Name, ast.Attribute, ast.Subscript)):
                    raise Unproven()
                argument = self.annotation(part, scope, source, missing)
                # Bare special forms such as Union/Optional are not type arguments.
                if argument not in {"type", "unbound", "generic:List", "generic:Dict", "generic:Tuple",
                                    "generic:list", "generic:dict", "generic:tuple"}:
                    raise Unproven()
            return "type"
        raise Unproven()

    def imported(self, node, scope):
        if isinstance(node, ast.ImportFrom) and node.level:
            raise Unproven()
        module = node.module if isinstance(node, ast.ImportFrom) else None
        if module and (module == EXTENSIONS or module.startswith(EXTENSIONS + ".")):
            imported_scope = self.helper(module)
            for alias in node.names:
                name = alias.asname or alias.name
                if alias.name not in imported_scope or alias.name.startswith("__") or name.startswith("__"):
                    raise Unproven()
                scope[name] = imported_scope[alias.name]
            return
        for alias in node.names:
            origin = (module, alias.name) if module else (alias.name, None)
            if origin not in self.allowed_imports or alias.name == "*":
                raise Unproven()
            name = alias.asname or (alias.name if module else alias.name.split(".")[0])
            if name.startswith("__"):
                raise Unproven()
            scope[name] = ("generic:" + alias.name if module == "typing" and alias.name != "Any" else
                           "module:" + alias.name if module is None else "type")

    def helper(self, module):
        if module in self.active:
            raise Unproven()
        if module in self.complete:
            return self.complete[module]
        relative = module.replace(".", "/")
        file, package = relative + ".py", relative + "/__init__.py"
        if file in self.trees and package in self.trees:
            raise Unproven()
        source = file if file in self.trees else package
        if source not in self.trees:
            raise Unproven()
        self.active.add(module)
        # Every candidate package initializer preceding this import must be inert.
        parts = module.split(".")
        for count in range(3, len(parts)):
            parent = ".".join(parts[:count])
            init = parent.replace(".", "/") + "/__init__.py"
            if init in self.trees:
                self.helper(parent)
        scope = self.initial_scope()
        exports = set()
        self.statements(self.trees[source].body, scope, source, exports=exports)
        self.active.remove(module)
        # Builtin fallback permits lookup, but does not create module attributes.
        self.complete[module] = {name: scope[name] for name in exports}
        return self.complete[module]

    @staticmethod
    def initial_scope():
        scope = {name: "value" for name in vars(builtins)}
        scope.update({name: "type" for name in ("str", "int", "float", "bool", "bytes", "object")})
        scope.update({name: "generic:" + name for name in ("list", "dict", "tuple", "set", "frozenset", "type")})
        scope.update({"__name__": "value", "__doc__": "value", "__package__": "value",
                      "__file__": "value", "__loader__": "value", "__spec__": "value"})
        return scope

    def statements(self, nodes, scope, source, *, class_body=False, exports=None):
        for node in nodes:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue
            if isinstance(node, ast.Pass):
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self.imported(node, scope)
                if exports is not None:
                    exports.update(alias.asname or (alias.name if isinstance(node, ast.ImportFrom)
                                                   else alias.name.split(".")[0]) for alias in node.names)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Import machinery can invoke module __getattr__ despite an
                # apparently inert body and an explicit from-import.
                if not class_body and node.name.startswith("__"):
                    raise Unproven()
                if node.decorator_list or getattr(node, "type_params", []):
                    raise Unproven()
                defaults = [*node.args.defaults, *[x for x in node.args.kw_defaults if x is not None]]
                if not all(literal(x) for x in defaults):
                    raise Unproven()
                args = [*node.args.posonlyargs, *node.args.args]
                args += [node.args.vararg] if node.args.vararg is not None else []
                args += node.args.kwonlyargs
                args += [node.args.kwarg] if node.args.kwarg is not None else []
                missing = []
                for argument in args:
                    self.annotation(argument.annotation, scope, source, missing)
                self.annotation(node.returns, scope, source, missing)
                # Any unsupported expression anywhere in the header must abstain
                # before a pending missing-name diagnosis becomes a scored zero.
                if missing:
                    raise CandidateSourceError(missing[0])
                scope[node.name] = "value"
                if exports is not None:
                    exports.add(node.name)
            elif isinstance(node, ast.ClassDef):
                if not class_body and node.name.startswith("__"):
                    raise Unproven()
                if node.bases or node.keywords or node.decorator_list or getattr(node, "type_params", []):
                    raise Unproven()
                self.statements(node.body, dict(scope), source, class_body=True)
                scope[node.name] = "type"
                if exports is not None:
                    exports.add(node.name)
            else:
                raise Unproven()

    def check(self):
        nodes = self.trees[CONTEXT].body
        size = len(self.bootstrap)
        # Trust only the exact fixed bootstrap, including register_resolvers().
        if len(nodes) < size or any(ast.dump(a) != ast.dump(b) for a, b in zip(nodes, self.bootstrap)):
            raise Unproven()
        scope = self.initial_scope()
        for node in self.bootstrap:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self.imported(node, scope)
        self.statements(nodes[size:], scope, CONTEXT)


def inspect_sources(workdir, pristine):
    """Compile original bytes, then prove only a supported eager declaration prefix."""
    trees = {}
    for source, data in sources(workdir).items():
        try:
            compile(data, source, "exec", dont_inherit=True)
            trees[source] = ast.parse(data, filename=source)
        except SyntaxError as error:
            # This exception comes from this compiler invocation, never the candidate.
            if type(error.lineno) is int and 1 <= error.lineno <= min(len(data.splitlines()), MAX_SOURCE_LINES):
                return diagnostic("syntax_error", source, error.lineno)
            raise
    if CONTEXT not in trees or sys.version_info >= (3, 14):
        return None
    try:
        DeclarationProof(trees, (Path(pristine) / CONTEXT).read_bytes()).check()
    except CandidateSourceError as error:
        return error.diagnostic
    except Unproven:
        pass
    return None


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--preflight":
        root = Path(sys.argv[2]).resolve()
        sys.path[:1] = [str(root), str(root / "verl")]
        importlib.import_module("ragen.llm_agent.ctx_manager")
    elif len(sys.argv) == 4 and sys.argv[1] == "--inspect":
        print(json.dumps(inspect_sources(sys.argv[2], sys.argv[3]), sort_keys=True))
    else:
        raise ValueError("unsupported source preflight invocation")


if __name__ == "__main__":
    main()
