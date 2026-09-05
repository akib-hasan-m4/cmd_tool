"""Per-method metrics.

Because Smalltalk expresses all control flow as message sends, most of these
metrics are computed by classifying *selectors* rather than syntax nodes.
See ``stsmell.selectors``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from ..model import ClassDef, MethodDef
from ..parsing import node_text, walk
from ..selectors import (
    ABSTRACT_MARKERS,
    ALL_MESSAGE_NODES,
    BRANCHING,
    CASCADED_MESSAGE_NODES,
    MESSAGE_NODES,
    SELF_RECEIVERS,
)

_TRANSPARENT = {"expression", "primary", "statement"}


@dataclass
class MethodMetrics:
    method: MethodDef
    mloc: int = 0
    cyclo: int = 1
    msg: int = 0
    params: int = 0
    temps: int = 0
    nblk: int = 0
    maxchain: int = 0
    foreign_maxchain: int = 0
    noav: int = 0
    max_cascade: int = 0
    self_sends: int = 0
    foreign_sends: Counter = field(default_factory=Counter)
    accessed_ivars: set[str] = field(default_factory=set)
    referenced_classes: set[str] = field(default_factory=set)
    uses_this_context: bool = False
    is_accessor: bool = False
    is_abstract: bool = False
    calls_super_same_selector: bool = False

    @property
    def top_foreign(self) -> tuple[Optional[str], int]:
        if not self.foreign_sends:
            return None, 0
        name, count = self.foreign_sends.most_common(1)[0]
        return name, count

    def as_dict(self) -> dict:
        name, count = self.top_foreign
        return {
            "class": self.method.class_name,
            "selector": self.method.selector,
            "class_side": self.method.class_side,
            "file": str(self.method.path) if self.method.path else None,
            "line": self.method.start_line,
            "MLOC": self.mloc,
            "CYCLO": self.cyclo,
            "MSG": self.msg,
            "PARAMS": self.params,
            "TEMPS": self.temps,
            "NBLK": self.nblk,
            "MAXCHAIN": self.maxchain,
            "FOREIGN_MAXCHAIN": self.foreign_maxchain,
            "NOAV": self.noav,
            "MAX_CASCADE": self.max_cascade,
            "SELF_SENDS": self.self_sends,
            "TOP_FOREIGN": name,
            "TOP_FOREIGN_SENDS": count,
            "IS_ACCESSOR": self.is_accessor,
            "IS_ABSTRACT": self.is_abstract,
        }


def _unwrap(node: Any) -> Any:
    """Skip transparent wrapper nodes to reach the meaningful child."""
    while node is not None and node.type in _TRANSPARENT:
        named = [c for c in node.children if c.is_named]
        if len(named) != 1:
            return node
        node = named[0]
    return node


def message_selector(node: Any, source: bytes) -> str:
    """Reconstruct the selector sent by a message node."""
    kind = node.type
    if kind in ("unary_message", "cascaded_unary_message"):
        for child in node.children:
            if child.type in ("unary_identifier", "unary_selector"):
                return node_text(child, source).strip()
        return ""
    if kind in ("binary_message", "cascaded_binary_message"):
        for child in node.children:
            if child.type in ("binary_operator", "binary_selector"):
                return node_text(child, source).strip()
        return ""
    if kind in ("keyword_message", "cascaded_keyword_message"):
        return "".join(
            node_text(c, source).strip() for c in node.children if c.type == "keyword"
        )
    return ""


def _receiver(node: Any) -> Optional[Any]:
    return _unwrap(node.child_by_field_name("receiver"))


def _chain_length(node: Any) -> int:
    """Number of consecutive sends stacked on one another (Law of Demeter)."""
    length = 1
    receiver = _receiver(node)
    while receiver is not None and receiver.type in MESSAGE_NODES:
        length += 1
        receiver = _receiver(receiver)
    return length


def _chain_base(node: Any) -> Optional[Any]:
    receiver = _receiver(node)
    base = receiver
    while receiver is not None and receiver.type in MESSAGE_NODES:
        receiver = _receiver(receiver)
        if receiver is not None:
            base = receiver
    return base


def _block_depth(body: Any) -> int:
    """Maximum nesting depth of block nodes within the body."""
    best = 0

    def visit(node: Any, depth: int) -> None:
        nonlocal best
        for child in node.children:
            child_depth = depth + 1 if child.type == "block" else depth
            best = max(best, child_depth)
            visit(child, child_depth)

    visit(body, 0)
    return best


def _code_lines(body: Any, source: bytes) -> int:
    """Count physical lines carrying code, ignoring blanks and comment-only lines."""
    lines: set[int] = set()
    for node in walk(body):
        if node.children:
            continue  # only leaves carry text
        if node.type == "comment":
            continue
        for line in range(node.start_point[0], node.end_point[0] + 1):
            lines.add(line)
    return len(lines)


def _statements(body: Any) -> list[Any]:
    return [_unwrap(c) for c in body.children if c.type == "statement"]


def _detect_accessor(method: MethodDef, body: Any, klass: ClassDef, source: bytes) -> bool:
    """Recognise the two canonical accessor shapes: ``^ ivar`` and ``ivar := arg``."""
    ivars = set(klass.inst_vars) | set(klass.class_inst_vars)
    if not ivars or method.temporaries:
        return False
    statements = _statements(body)
    if len(statements) != 1:
        return False
    stmt = statements[0]

    if stmt.type == "return" and not method.params:
        named = [c for c in stmt.children if c.is_named]
        if len(named) == 1:
            inner = _unwrap(named[0])
            if inner.type == "identifier" and node_text(inner, source).strip() in ivars:
                return True
        return False

    if stmt.type == "assignment" and len(method.params) == 1:
        named = [c for c in stmt.children if c.is_named]
        if len(named) == 2:
            target, value = _unwrap(named[0]), _unwrap(named[1])
            return (
                target.type == "identifier"
                and node_text(target, source).strip() in ivars
                and value.type == "identifier"
                and node_text(value, source).strip() == method.params[0]
            )
    return False


def _detect_abstract(body: Any, source: bytes) -> bool:
    statements = _statements(body)
    if len(statements) != 1:
        return False
    stmt = statements[0]
    if stmt.type == "return":
        named = [c for c in stmt.children if c.is_named]
        if len(named) != 1:
            return False
        stmt = _unwrap(named[0])
    if stmt.type != "unary_message":
        return False
    receiver = _receiver(stmt)
    return (
        receiver is not None
        and receiver.type == "self"
        and message_selector(stmt, source) in ABSTRACT_MARKERS
    )


def compute_method_metrics(
    method: MethodDef, klass: ClassDef, source: bytes
) -> MethodMetrics:
    m = MethodMetrics(method=method)
    m.params = len(method.params)
    m.temps = len(method.temporaries)

    body = method.body_node
    if body is None:
        return m

    ivars = set(klass.inst_vars) | set(klass.class_inst_vars) | set(klass.class_vars)
    # Feature envy is about reaching into *other* objects. A temporary the
    # method constructed itself (``parts := OrderedCollection new``) is not a
    # foreign object, so only instance state and parameters are envy candidates.
    envy_names = set(method.params)
    accessed: set[str] = set()

    m.mloc = _code_lines(body, source)
    m.nblk = _block_depth(body)

    # ``| a b |`` declares locals; the identifiers inside a temporaries node are
    # declarations, not accesses, and must not inflate NOAV.
    declared_ids = {
        id(ident)
        for node in walk(body)
        if node.type == "temporaries"
        for ident in node.children
        if ident.type == "identifier"
    }

    for node in walk(body):
        kind = node.type

        if kind in ALL_MESSAGE_NODES:
            m.msg += 1
            selector = message_selector(node, source)
            if selector in BRANCHING:
                m.cyclo += 1

            if kind in MESSAGE_NODES:
                chain = _chain_length(node)
                m.maxchain = max(m.maxchain, chain)
                base = _chain_base(node)
                base_is_self = base is not None and (
                    base.type in SELF_RECEIVERS
                    or node_text(base, source).strip() in SELF_RECEIVERS
                )
                if not base_is_self:
                    m.foreign_maxchain = max(m.foreign_maxchain, chain)

                receiver = _receiver(node)
                if receiver is not None:
                    if receiver.type in SELF_RECEIVERS:
                        m.self_sends += 1
                    elif receiver.type == "identifier" and kind != "binary_message":
                        # Binary sends are arithmetic/comparison (``amount > 0``),
                        # not delegation, so they never indicate envy. Only
                        # instance state and locals count; a send to a global
                        # class name is construction, not envy.
                        name = node_text(receiver, source).strip()
                        if name in ivars or name in envy_names:
                            m.foreign_sends[name] += 1

        elif kind == "cascade":
            m.max_cascade = max(
                m.max_cascade,
                1 + sum(1 for c in node.children if c.type in CASCADED_MESSAGE_NODES),
            )

        elif kind == "thisContext":
            m.uses_this_context = True

        elif kind == "identifier":
            if id(node) in declared_ids:
                continue
            name = node_text(node, source).strip()
            if not name:
                continue
            if name[0].isupper():
                m.referenced_classes.add(name)
            else:
                accessed.add(name)
                if name in ivars:
                    m.accessed_ivars.add(name)

        elif kind == "super":
            pass

    m.noav = len(accessed)
    m.is_accessor = _detect_accessor(method, body, klass, source)
    m.is_abstract = _detect_abstract(body, source)

    for node in walk(body):
        if node.type in MESSAGE_NODES:
            receiver = _receiver(node)
            if (
                receiver is not None
                and receiver.type == "super"
                and message_selector(node, source) == method.selector
            ):
                m.calls_super_same_selector = True
                break

    return m
