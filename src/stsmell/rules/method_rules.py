"""Method-level smells."""

from __future__ import annotations

from typing import Iterable

from .base import Finding, Rule, RuleContext, register


@register
class LongMethod(Rule):
    id = "long-method"
    name = "Long Method"
    severity = "major"
    description = (
        "Method body is too long. The Pharo idiom is a handful of lines, so the "
        "default here is deliberately stricter than a Java linter's."
    )
    defaults = {"max_lines": 25}

    def check_method(self, mm, cm, ctx) -> Iterable[Finding]:
        limit = self.opt(ctx, "max_lines")
        if mm.mloc > limit:
            yield self.method_finding(
                mm,
                f"method is {mm.mloc} lines (limit {limit})",
                MLOC=mm.mloc,
                limit=limit,
            )


@register
class ComplexMethod(Rule):
    id = "complex-method"
    name = "Complex Method"
    severity = "major"
    description = (
        "Too many independent paths. Complexity is counted from branching "
        "selectors (ifTrue:, whileTrue:, detect:ifNone:), since Smalltalk has "
        "no control-flow syntax."
    )
    defaults = {"max_cyclo": 8}

    def check_method(self, mm, cm, ctx) -> Iterable[Finding]:
        limit = self.opt(ctx, "max_cyclo")
        if mm.cyclo > limit:
            yield self.method_finding(
                mm,
                f"cyclomatic complexity is {mm.cyclo} (limit {limit})",
                CYCLO=mm.cyclo,
                limit=limit,
            )


@register
class LongParameterList(Rule):
    id = "long-parameter-list"
    name = "Long Parameter List"
    severity = "minor"
    description = "Keyword message takes too many arguments."
    defaults = {"max_params": 4}

    def check_method(self, mm, cm, ctx) -> Iterable[Finding]:
        limit = self.opt(ctx, "max_params")
        if mm.params > limit:
            yield self.method_finding(
                mm,
                f"takes {mm.params} parameters (limit {limit})",
                PARAMS=mm.params,
                limit=limit,
            )


@register
class TooManyTemporaries(Rule):
    id = "too-many-temporaries"
    name = "Too Many Temporaries"
    severity = "minor"
    description = "Many temporary variables suggest the method does several jobs."
    defaults = {"max_temps": 6}

    def check_method(self, mm, cm, ctx) -> Iterable[Finding]:
        limit = self.opt(ctx, "max_temps")
        if mm.temps > limit:
            yield self.method_finding(
                mm,
                f"declares {mm.temps} temporaries (limit {limit})",
                TEMPS=mm.temps,
                limit=limit,
            )


@register
class DeeplyNestedBlocks(Rule):
    id = "deeply-nested-blocks"
    name = "Deeply Nested Blocks"
    severity = "major"
    description = "Deep block nesting; each level is a conditional or loop body."
    defaults = {"max_depth": 3}

    def check_method(self, mm, cm, ctx) -> Iterable[Finding]:
        limit = self.opt(ctx, "max_depth")
        if mm.nblk > limit:
            yield self.method_finding(
                mm,
                f"blocks nested {mm.nblk} deep (limit {limit})",
                NBLK=mm.nblk,
                limit=limit,
            )


@register
class MessageChain(Rule):
    id = "message-chain"
    name = "Message Chain"
    severity = "minor"
    description = (
        "Long chain of sends on a foreign object (Law of Demeter). Chains rooted "
        "at self or super are not counted."
    )
    defaults = {"max_chain": 4}

    def check_method(self, mm, cm, ctx) -> Iterable[Finding]:
        limit = self.opt(ctx, "max_chain")
        if mm.foreign_maxchain > limit:
            yield self.method_finding(
                mm,
                f"message chain of length {mm.foreign_maxchain} (limit {limit})",
                MAXCHAIN=mm.foreign_maxchain,
                limit=limit,
            )


@register
class FeatureEnvy(Rule):
    id = "feature-envy"
    name = "Feature Envy"
    severity = "major"
    description = (
        "The method talks to another object more than to itself, so it may "
        "belong on that object instead."
    )
    defaults = {"min_foreign_sends": 5}

    def check_method(self, mm, cm, ctx) -> Iterable[Finding]:
        minimum = self.opt(ctx, "min_foreign_sends")
        name, count = mm.top_foreign
        if name is None or mm.is_accessor:
            return
        if count >= minimum and count > mm.self_sends:
            yield self.method_finding(
                mm,
                f"sends {count} messages to '{name}' but only "
                f"{mm.self_sends} to self",
                receiver=name,
                foreign_sends=count,
                self_sends=mm.self_sends,
            )


@register
class UsesThisContext(Rule):
    id = "uses-this-context"
    name = "Uses thisContext"
    severity = "minor"
    description = (
        "thisContext reflectively inspects the call stack. Legitimate in "
        "debuggers and exception machinery, a design smell elsewhere."
    )
    defaults: dict = {}

    def check_method(self, mm, cm, ctx) -> Iterable[Finding]:
        if mm.uses_this_context:
            yield self.method_finding(mm, "uses thisContext (reflective stack access)")


@register
class CascadeAbuse(Rule):
    id = "cascade-abuse"
    name = "Cascade Abuse"
    severity = "info"
    description = "Very long cascade; consider a builder or a dedicated method."
    defaults = {"max_cascade": 6}

    def check_method(self, mm, cm, ctx) -> Iterable[Finding]:
        limit = self.opt(ctx, "max_cascade")
        if mm.max_cascade > limit:
            yield self.method_finding(
                mm,
                f"cascade of {mm.max_cascade} messages (limit {limit})",
                MAX_CASCADE=mm.max_cascade,
                limit=limit,
            )
