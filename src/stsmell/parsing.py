"""tree-sitter setup and CST helpers.

All direct contact with the grammar is confined to this module and
``extract.py`` so that swapping the parser stays a local change.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Any, Iterator, Optional

import tree_sitter
import tree_sitter_tonel_smalltalk

# Traits mark class-side methods with ``classSide >>`` where classes use
# ``class >>``; the grammar only knows the latter, so such files fail to parse.
# Rewriting the keyword to ``class`` plus padding keeps every byte offset --
# and therefore every reported line and column -- exactly as it was.
_CLASS_SIDE_RE = re.compile(
    rb"(^|\n)([A-Za-z_][A-Za-z0-9_]*)([ \t]+)classSide([ \t]*>>)"
)
_CLASS_SIDE_REPLACEMENT = rb"\1\2\3class    \4"


def normalize_source(source: bytes) -> bytes:
    """Work around known grammar gaps without shifting any byte offsets."""
    return _CLASS_SIDE_RE.sub(_CLASS_SIDE_REPLACEMENT, source)


@functools.lru_cache(maxsize=1)
def get_language() -> Any:
    # tree_sitter_tonel_smalltalk.language() already returns a built
    # tree_sitter.Language; wrapping it again raises TypeError.
    return tree_sitter_tonel_smalltalk.language()


@functools.lru_cache(maxsize=1)
def get_parser() -> Any:
    return tree_sitter.Parser(get_language())


def parse_bytes(source: bytes) -> Any:
    return get_parser().parse(normalize_source(source))


def parse_file(path: Path) -> tuple[bytes, Any]:
    source = normalize_source(path.read_bytes())
    return source, get_parser().parse(source)


def node_text(node: Any, source: bytes) -> str:
    if node is None:
        return ""
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def strip_string_literal(text: str) -> str:
    """Unwrap a Smalltalk string literal: 'it''s' -> it's."""
    text = text.strip()
    if len(text) >= 2 and text[0] == "'" and text[-1] == "'":
        return text[1:-1].replace("''", "'")
    return text


def strip_symbol(text: str) -> str:
    """#name -> name, #'odd name' -> odd name."""
    text = text.strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) >= 2 and text[0] == "'" and text[-1] == "'":
        return text[1:-1].replace("''", "'")
    return text


def strip_class_comment(text: str) -> str:
    """Unwrap a Smalltalk double-quoted comment literal, halving doubled quotes."""
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1].replace('""', '"')
    return text


def children_of_type(node: Any, type_name: str) -> list[Any]:
    return [c for c in node.children if c.type == type_name]


def first_child_of_type(node: Any, type_name: str) -> Optional[Any]:
    for c in node.children:
        if c.type == type_name:
            return c
    return None


def walk(node: Any) -> Iterator[Any]:
    """Yield node and all descendants, depth-first."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def walk_named(node: Any) -> Iterator[Any]:
    for n in walk(node):
        if n.is_named:
            yield n


def has_error(tree: Any) -> bool:
    return bool(tree.root_node.has_error)
