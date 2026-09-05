"""Curated Smalltalk selector sets.

This module is the Smalltalk-specific heart of the tool.

Smalltalk has no control-flow *syntax*: there is no ``if``, no ``while``, no
``for``. Every branch and every loop is an ordinary message send that takes
block arguments -- ``ifTrue:``, ``whileTrue:``, ``to:do:``. Cyclomatic
complexity therefore cannot be computed by counting keywords the way it is in
C-family languages; it has to be computed by recognising the *selectors* below
on message-send nodes.
"""

from __future__ import annotations

#: Conditional branching. Each adds one independent path.
CONDITIONAL = {
    "ifTrue:",
    "ifFalse:",
    "ifTrue:ifFalse:",
    "ifFalse:ifTrue:",
    "ifNil:",
    "ifNotNil:",
    "ifNil:ifNotNil:",
    "ifNotNil:ifNil:",
    "ifEmpty:",
    "ifNotEmpty:",
    "ifEmpty:ifNotEmpty:",
    "ifNotEmpty:ifEmpty:",
    "caseOf:",
    "caseOf:otherwise:",
    "at:ifAbsent:",
    "at:ifPresent:",
    "at:ifAbsentPut:",
    "at:ifPresent:ifAbsent:",
    "detect:ifNone:",
    "indexOf:ifAbsent:",
    "remove:ifAbsent:",
}

#: Short-circuit boolean operators. Each adds a path.
BOOLEAN = {"and:", "or:"}

#: Looping / iteration.
LOOPING = {
    "whileTrue:",
    "whileFalse:",
    "whileTrue",
    "whileFalse",
    "repeat",
    "to:do:",
    "to:by:do:",
    "timesRepeat:",
    "do:",
    "doWithIndex:",
    "withIndexDo:",
    "keysAndValuesDo:",
    "reverseDo:",
    "do:separatedBy:",
    "detect:",
    "select:",
    "reject:",
    "collect:",
    "inject:into:",
    "anySatisfy:",
    "allSatisfy:",
    "noneSatisfy:",
}

#: Non-local control flow (exception handling). Each introduces a path.
EXCEPTIONAL = {"on:do:", "on:do:on:do:", "ensure:", "ifCurtailed:", "on:fork:"}

#: Every selector that contributes +1 to cyclomatic complexity.
BRANCHING = CONDITIONAL | BOOLEAN | LOOPING | EXCEPTIONAL

#: Bodies consisting solely of one of these mark an abstract / refused method.
ABSTRACT_MARKERS = {"subclassResponsibility", "shouldBeImplemented", "notYetImplemented"}

#: Receivers that are not "foreign" objects for Feature Envy / Law of Demeter.
SELF_RECEIVERS = {"self", "super", "thisContext"}

#: Node types that represent a message send.
MESSAGE_NODES = {"unary_message", "binary_message", "keyword_message"}

#: Node types for messages inside a cascade (their receiver is implicit).
CASCADED_MESSAGE_NODES = {
    "cascaded_unary_message",
    "cascaded_binary_message",
    "cascaded_keyword_message",
}

ALL_MESSAGE_NODES = MESSAGE_NODES | CASCADED_MESSAGE_NODES


def is_branching(selector: str) -> bool:
    return selector in BRANCHING
