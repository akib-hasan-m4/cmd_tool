"""Translate the Tonel CST into the language model in ``model.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .model import ClassDef, MethodDef, ParsedFile
from .parsing import (
    first_child_of_type,
    has_error,
    node_text,
    parse_file,
    strip_class_comment,
    strip_string_literal,
    strip_symbol,
)

_DEFINITION_KINDS = {
    "class_definition": "class",
    "trait_definition": "trait",
    "extension_definition": "extension",
}


def _ston_scalar(value_node: Any, source: bytes) -> str:
    """Read a ston_value holding a single scalar (string, symbol or number)."""
    text = node_text(value_node, source).strip()
    if text.startswith("'"):
        return strip_string_literal(text)
    if text.startswith("#"):
        return strip_symbol(text)
    return text


def _ston_list(value_node: Any, source: bytes) -> list[str]:
    """Read a ston_value holding a ston_list of scalars."""
    lst = first_child_of_type(value_node, "ston_list")
    if lst is None:
        scalar = _ston_scalar(value_node, source)
        return [scalar] if scalar else []
    out = []
    for child in lst.children:
        if child.type == "ston_value":
            scalar = _ston_scalar(child, source)
            if scalar:
                out.append(scalar)
    return out


def _ston_pairs(map_node: Any, source: bytes) -> dict[str, Any]:
    """Map STON keys (symbol name, '#' stripped) to their ston_value nodes."""
    pairs: dict[str, Any] = {}
    if map_node is None:
        return pairs
    for pair in map_node.children:
        if pair.type != "ston_pair":
            continue
        key_node = first_child_of_type(pair, "ston_key")
        value_node = first_child_of_type(pair, "ston_value")
        if key_node is None or value_node is None:
            continue
        pairs[strip_symbol(node_text(key_node, source))] = value_node
    return pairs


def _build_class(def_node: Any, kind: str, source: bytes, path: Path) -> ClassDef:
    pairs = _ston_pairs(first_child_of_type(def_node, "ston_map"), source)

    def scalar(key: str) -> Optional[str]:
        node = pairs.get(key)
        return _ston_scalar(node, source) if node is not None else None

    def lst(key: str) -> list[str]:
        node = pairs.get(key)
        return _ston_list(node, source) if node is not None else []

    return ClassDef(
        name=scalar("name") or "",
        superclass=scalar("superclass"),
        inst_vars=lst("instVars"),
        class_inst_vars=lst("classInstVars"),
        class_vars=lst("classVars"),
        traits=lst("traits"),
        # Tonel 3.0 replaced #category with #package + #tag for class defs.
        package=scalar("package") or scalar("category"),
        category=scalar("category") or scalar("tag"),
        kind=kind,
        path=path,
        start_line=def_node.start_point[0] + 1,
        end_line=def_node.end_point[0] + 1,
    )


def _selector_of(ref_node: Any, source: bytes) -> tuple[str, list[str]]:
    """Return (selector, params) from a method_reference node.

    Keyword methods carry one ``selector`` field per keyword part, so the full
    selector is their concatenation: ``deposit:`` + ``into:``.
    """
    parts = [node_text(n, source).strip() for n in ref_node.children_by_field_name("selector")]
    params = [node_text(n, source).strip() for n in ref_node.children_by_field_name("param")]
    return "".join(parts), params


def _build_method(md_node: Any, source: bytes, path: Path) -> Optional[MethodDef]:
    ref = first_child_of_type(md_node, "method_reference")
    if ref is None:
        return None

    selector, params = _selector_of(ref, source)
    class_name_node = ref.child_by_field_name("class_name")
    class_name = node_text(class_name_node, source).strip() if class_name_node else ""
    class_side = ref.child_by_field_name("class_side") is not None

    category = None
    meta = first_child_of_type(md_node, "method_metadata")
    if meta is not None:
        pairs = _ston_pairs(first_child_of_type(meta, "ston_map"), source)
        if "category" in pairs:
            category = _ston_scalar(pairs["category"], source)

    body = first_child_of_type(md_node, "method_body")
    temporaries: list[str] = []
    pragmas: list[str] = []
    if body is not None:
        for child in body.children:
            if child.type == "temporaries":
                temporaries.extend(
                    node_text(t, source).strip()
                    for t in child.children
                    if t.type == "identifier"
                )
            elif child.type == "pragma":
                pragmas.append(node_text(child, source).strip())

    return MethodDef(
        class_name=class_name,
        selector=selector,
        class_side=class_side,
        params=params,
        temporaries=temporaries,
        pragmas=pragmas,
        category=category,
        path=path,
        start_line=md_node.start_point[0] + 1,
        end_line=md_node.end_point[0] + 1,
        source=node_text(md_node, source),
        body_node=body,
    )


def extract_file(path: Path) -> ParsedFile:
    """Parse and extract a single Tonel file."""
    source, tree = parse_file(path)
    parsed = ParsedFile(path=path, source=source, tree=tree, parse_error=has_error(tree))

    root = tree.root_node
    classes: list[ClassDef] = []
    pending_comment: Optional[str] = None

    for child in root.children:
        if child.type == "class_comment":
            # The class comment is the string literal preceding the definition.
            pending_comment = strip_class_comment(node_text(child, source))
        elif child.type == "definition":
            for inner in child.children:
                kind = _DEFINITION_KINDS.get(inner.type)
                if kind is None:
                    continue
                klass = _build_class(inner, kind, source, path)
                if pending_comment:
                    klass.comment = pending_comment
                    pending_comment = None
                classes.append(klass)
        elif child.type in _DEFINITION_KINDS:  # tolerate an unwrapped definition
            klass = _build_class(child, _DEFINITION_KINDS[child.type], source, path)
            if pending_comment:
                klass.comment = pending_comment
                pending_comment = None
            classes.append(klass)
        elif child.type == "method_definition":
            method = _build_method(child, source, path)
            if method is None:
                continue
            target = _target_class(classes, method)
            if target is not None:
                target.methods.append(method)
            else:
                # A method with no definition in the file: synthesise a holder
                # so the method still gets analysed.
                holder = ClassDef(name=method.class_name, kind="extension", path=path)
                holder.methods.append(method)
                classes.append(holder)

    parsed.classes = classes
    return parsed


def _target_class(classes: list[ClassDef], method: MethodDef) -> Optional[ClassDef]:
    for klass in classes:
        if klass.name == method.class_name:
            return klass
    # Tonel guarantees one type per file; fall back to the sole definition so a
    # renamed or mismatched reference still lands somewhere sensible.
    return classes[0] if len(classes) == 1 else None
