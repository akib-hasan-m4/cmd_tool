"""Language model extracted from Tonel source.

Rules consume this model plus computed metrics; they never touch the CST.
Metric code may reach into the CST via the ``*_node`` attributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class MethodDef:
    class_name: str
    selector: str
    class_side: bool
    params: list[str] = field(default_factory=list)
    temporaries: list[str] = field(default_factory=list)
    pragmas: list[str] = field(default_factory=list)
    category: Optional[str] = None
    path: Optional[Path] = None
    start_line: int = 0
    end_line: int = 0
    source: str = ""
    body_node: Any = None  # tree_sitter.Node for the method_body, may be None

    @property
    def qualified_name(self) -> str:
        side = " class" if self.class_side else ""
        return f"{self.class_name}{side} >> {self.selector}"


@dataclass
class ClassDef:
    name: str
    superclass: Optional[str] = None
    inst_vars: list[str] = field(default_factory=list)
    class_inst_vars: list[str] = field(default_factory=list)
    class_vars: list[str] = field(default_factory=list)
    traits: list[str] = field(default_factory=list)
    package: Optional[str] = None
    category: Optional[str] = None
    comment: Optional[str] = None
    kind: str = "class"  # class | trait | extension
    path: Optional[Path] = None
    start_line: int = 0
    end_line: int = 0
    methods: list[MethodDef] = field(default_factory=list)

    @property
    def is_extension(self) -> bool:
        return self.kind == "extension"

    @property
    def instance_methods(self) -> list[MethodDef]:
        return [m for m in self.methods if not m.class_side]

    @property
    def class_methods(self) -> list[MethodDef]:
        return [m for m in self.methods if m.class_side]


@dataclass
class ParsedFile:
    """A parsed Tonel file.

    Holds ``tree`` so that tree-sitter Nodes referenced by MethodDef stay
    alive; dropping the Tree invalidates every Node derived from it.
    """

    path: Path
    source: bytes
    tree: Any = None
    classes: list[ClassDef] = field(default_factory=list)
    parse_error: bool = False


@dataclass
class Project:
    root: Path
    files: list[ParsedFile] = field(default_factory=list)

    @property
    def classes(self) -> list[ClassDef]:
        return [c for f in self.files for c in f.classes]

    def class_names(self) -> set[str]:
        return {c.name for c in self.classes}

    @property
    def methods(self) -> list[MethodDef]:
        return [m for c in self.classes for m in c.methods]
