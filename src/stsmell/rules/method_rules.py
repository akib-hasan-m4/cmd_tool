"""Method-level smells."""

from __future__ import annotations

from typing import Iterable

from .base import Finding, Rule, RuleContext, register


@register
class LongMethod(Rule):
    """Method body is too long to read in one sitting.

    Metric: MLOC (method lines of code, comments and blanks excluded).
    Breaks when MLOC > max_lines.
    """

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
    """Method has too many independent execution paths.

    Metric: CYCLO (cyclomatic complexity), counted as 1 plus one per branching
    selector sent -- see ``selectors.BRANCHING`` -- because Smalltalk expresses
    every branch and loop as a message send.
    Breaks when CYCLO > max_cyclo.
    """

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
    """Keyword message takes more arguments than a caller can keep straight.

    Metric: PARAMS (one per keyword part of the selector).
    Breaks when PARAMS > max_params.
    """

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
    """Method juggles enough local state to suggest it does several jobs.

    Metric: TEMPS (identifiers in the method's own ``| a b c |`` declaration;
    temporaries declared inside blocks are not counted).
    Breaks when TEMPS > max_temps.
    """

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
    """Control flow is buried several levels deep.

    Metric: NBLK (maximum ``[ ... ]`` nesting depth in the body). Since blocks
    are how Smalltalk writes conditional and loop bodies, each level is a
    branch the reader must hold in mind.
    Breaks when NBLK > max_depth.
    """

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
    """Law of Demeter: the method navigates a long path through other objects.

    Metric: FOREIGN_MAXCHAIN (longest run of sends stacked receiver-on-receiver
    whose root is not in ``SELF_RECEIVERS``). Chains rooted at self or super are
    excluded, so ``self foo bar baz`` never trips this.
    Breaks when FOREIGN_MAXCHAIN > max_chain.
    """

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
    """The method is more interested in another object than in its own.

    Metrics: TOP_FOREIGN (the most-messaged non-self receiver and its send
    count) against SELF_SENDS. Only instance variables and parameters count as
    foreign receivers -- a temporary the method built itself is not another
    object. Accessors are exempt.
    Breaks when foreign sends >= min_foreign_sends and exceed SELF_SENDS.
    """

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
    """Method reaches into the call stack reflectively.

    Metric: USES_THIS_CONTEXT (boolean -- ``thisContext`` appears in the body).
    No threshold; a single occurrence is the finding.
    """

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
    """A single ``;`` cascade configures an object at unreadable length.

    Metric: MAX_CASCADE (most messages in one cascade expression).
    Breaks when MAX_CASCADE > max_cascade. Disabled by default in the shipped
    config: long cascades are idiomatic in UI and stream-building code.
    """

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
