"""Superclass graph: depth of inheritance and subclass counts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..model import ClassDef, Project


@dataclass
class Hierarchy:
    by_name: dict[str, ClassDef] = field(default_factory=dict)
    depth: dict[str, int] = field(default_factory=dict)
    children: dict[str, list[str]] = field(default_factory=dict)
    unresolved: set[str] = field(default_factory=set)

    @classmethod
    def build(cls, project: Project) -> "Hierarchy":
        h = cls()
        for klass in project.classes:
            if klass.name and not klass.is_extension:
                h.by_name.setdefault(klass.name, klass)

        for klass in h.by_name.values():
            parent = klass.superclass
            if parent:
                h.children.setdefault(parent, []).append(klass.name)
                if parent not in h.by_name:
                    h.unresolved.add(parent)

        for name in h.by_name:
            h.depth[name] = h._compute_depth(name)
        return h

    def _compute_depth(self, name: str) -> int:
        """Links walked until the chain leaves the project.

        A class whose superclass is ``Object`` (outside the analysed source)
        has depth 1. Chains that leave the project stop rather than guessing at
        the depth of the external hierarchy.
        """
        depth = 0
        seen: set[str] = {name}
        current: Optional[str] = name
        while current is not None:
            klass = self.by_name.get(current)
            if klass is None or not klass.superclass:
                break
            depth += 1
            parent = klass.superclass
            if parent in seen:  # defensive: malformed cyclic hierarchy
                break
            seen.add(parent)
            current = parent
        return depth

    def num_children(self, name: str) -> int:
        return len(self.children.get(name, ()))

    def ancestors(self, name: str) -> list[str]:
        out: list[str] = []
        seen = {name}
        current = self.by_name.get(name)
        while current is not None and current.superclass:
            parent = current.superclass
            if parent in seen:
                break
            out.append(parent)
            seen.add(parent)
            current = self.by_name.get(parent)
        return out

    def inherited_ivars(self, name: str) -> set[str]:
        out: set[str] = set()
        for ancestor in self.ancestors(name):
            klass = self.by_name.get(ancestor)
            if klass is not None:
                out.update(klass.inst_vars)
        return out
