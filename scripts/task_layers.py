"""The repository's task layers: where tasks, their catalog rows and their definitions live.

The shared tree is the first layer (tasks/problems/, config/task-catalog.json,
config/task-definitions.json). A top-level directory that holds a `layer.json` adds a layer of
its own:

    {"problems": "tasks", "catalog": "config/task-catalog.json",
     "definitions": "config/task-definitions.json"}

(paths relative to that directory). A layer adds task rows and definitions; it never changes
the shared layer's, and a task belongs to exactly one layer. Readers that take no explicit
catalog read every layer merged, in layer order (`load_catalog`, `load_definitions`)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = "layer.json"
CATALOG_SCHEMA = "frontier.deployment-task-catalog/v2"
# The shared layer's problems directory, relative to the repository.
SHARED_PROBLEMS = Path("tasks/problems")
LAYER_KEYS = {"problems", "catalog", "definitions"}


@dataclass(frozen=True)
class Layer:
    root: Path
    problems: Path
    catalog: Path
    definitions: Path


def _object(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path}: duplicate key {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return value


def shared_layer(root: Path = ROOT) -> Layer:
    return Layer(root, root / SHARED_PROBLEMS, root / "config/task-catalog.json",
                 root / "config/task-definitions.json")


def layers(root: Path = ROOT) -> list[Layer]:
    """The shared layer, then each declared layer in directory order."""
    root = root.resolve()
    found = [shared_layer(root)]
    for declaration in sorted(root.glob(f"*/{DECLARATION}")):
        body = _object(declaration)
        if set(body) != LAYER_KEYS:
            raise ValueError(f"{declaration} must declare exactly {sorted(LAYER_KEYS)}")
        base = declaration.parent
        found.append(Layer(base, base / body["problems"], base / body["catalog"],
                           base / body["definitions"]))
    return found


def catalog_documents(root: Path = ROOT) -> list[tuple[Path, dict[str, Any]]]:
    """Each layer's catalog file with its document, in layer order."""
    return [(layer.catalog, _object(layer.catalog)) for layer in layers(root)]


def merge_catalogs(documents: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    """The shared catalog with the task rows of each further layer appended; the rows are the
    documents' own objects, so a change to one shows in both."""
    (_, shared), *others = documents
    catalog = dict(shared)
    names = {task["name"] for task in shared["tasks"]}
    for path, rows in others:
        if set(rows) != {"schema_version", "tasks"} or rows["schema_version"] != CATALOG_SCHEMA:
            raise ValueError(f"{path} must hold schema_version {CATALOG_SCHEMA} and tasks")
        for task in rows["tasks"]:
            if task["name"] in names:
                raise ValueError(f"{path}: task {task['name']} is in another layer")
            names.add(task["name"])
        catalog["tasks"] = catalog["tasks"] + rows["tasks"]
    return catalog


def load_catalog(path: Path | None = None, root: Path = ROOT) -> dict[str, Any]:
    """The catalog at `path`, or every layer's catalog merged (`merge_catalogs`)."""
    if path is not None:
        return _object(path)
    return merge_catalogs(catalog_documents(root))


def load_definitions(root: Path = ROOT) -> dict[str, Any]:
    """Every layer's task definitions merged."""
    merged: dict[str, Any] = {}
    for layer in layers(root):
        for name, definition in _object(layer.definitions).items():
            if name in merged:
                raise ValueError(f"{layer.definitions}: task {name} is in another layer")
            merged[name] = definition
    return merged


def layer_of(name: str, root: Path = ROOT) -> Layer:
    """The layer whose catalog lists task `name`."""
    owners = [layer for layer in layers(root)
              if any(task["name"] == name for task in _object(layer.catalog)["tasks"])]
    if len(owners) != 1:
        raise ValueError(f"task {name} must be in exactly one layer's catalog, not {len(owners)}")
    return owners[0]


def json_text(value: object) -> str:
    """The repository's JSON format: sorted keys, two-space indent, a final newline."""
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def layer_files(
    catalog: dict[str, Any], definitions: dict[str, Any], new_layer: Path | None = None,
    root: Path = ROOT,
) -> dict[Path, str]:
    """The text of every layer's catalog and definitions file (and declaration), keyed by path
    relative to `root`, holding `catalog`'s rows and `definitions` split by layer: a task stays
    in the layer that lists it, and a task no layer lists goes to the layer whose directory is
    `new_layer` (the shared tree when None)."""
    found = layers(root)
    owners = {task["name"]: layer for layer in found
              for task in _object(layer.catalog)["tasks"]}
    targets = [layer for layer in found if layer.root == (new_layer or root).resolve()]
    if len(targets) != 1:
        raise ValueError(f"{new_layer} is not a task layer of {root}")
    files = {}
    for index, layer in enumerate(found):
        mine = {name for name in [task["name"] for task in catalog["tasks"]] + list(definitions)
                if owners.get(name, targets[0]) == layer}
        rows = [task for task in catalog["tasks"] if task["name"] in mine]
        document = ({**catalog, "tasks": rows} if index == 0
                    else {"schema_version": catalog["schema_version"], "tasks": rows})
        files[layer.catalog.relative_to(root)] = json_text(document)
        files[layer.definitions.relative_to(root)] = json_text(
            {name: definitions[name] for name in sorted(definitions) if name in mine})
        if index:
            declaration = layer.root / DECLARATION
            files[declaration.relative_to(root)] = declaration.read_text(encoding="utf-8")
    return files
