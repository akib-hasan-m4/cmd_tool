"""Extraction: Tonel CST -> model."""

from stsmell.extract import extract_file


def test_extracts_class_metadata(fixtures):
    parsed = extract_file(fixtures / "clean" / "CleanPoint.class.st")
    assert not parsed.parse_error
    assert len(parsed.classes) == 1

    klass = parsed.classes[0]
    assert klass.name == "CleanPoint"
    assert klass.superclass == "Object"
    assert klass.inst_vars == ["x", "y"]
    assert klass.package == "Clean-Core"
    assert klass.kind == "class"
    assert klass.comment == "I am a small, cohesive, well documented point."


def test_keyword_selector_parts_are_concatenated(fixtures):
    parsed = extract_file(fixtures / "smelly" / "LongParameterListExample.class.st")
    method = parsed.classes[0].methods[0]
    assert method.selector == "withA:b:c:d:e:"
    assert method.params == ["a", "b", "c", "d", "e"]


def test_binary_selector_method(fixtures):
    parsed = extract_file(fixtures / "clean" / "CleanPoint.class.st")
    plus = [m for m in parsed.classes[0].methods if m.selector == "+"]
    assert len(plus) == 1
    assert plus[0].params == ["other"]


def test_class_side_and_metadata(tmp_path):
    source = """\"I have a class-side method.\"
Class {
\t#name : 'Widget',
\t#superclass : 'Object',
\t#instVars : [ 'name' ]
}

{ #category : 'instance creation' }
Widget class >> named: aString [
\t<myPragma>
\t| w |
\tw := self new.
\t^ w
]
"""
    path = tmp_path / "Widget.class.st"
    path.write_text(source)
    parsed = extract_file(path)
    method = parsed.classes[0].methods[0]

    assert method.class_side is True
    assert method.selector == "named:"
    assert method.qualified_name == "Widget class >> named:"
    assert method.temporaries == ["w"]
    assert method.pragmas == ["<myPragma>"]
    assert method.category == "instance creation"


def test_extension_definition(fixtures):
    parsed = extract_file(fixtures / "extensions" / "String.extension.st")
    klass = parsed.classes[0]
    assert klass.kind == "extension"
    assert klass.is_extension
    assert klass.name == "String"
    assert {m.selector for m in klass.methods} == {"myShout", "myWhisper"}


def test_trait_class_side_uses_classside_keyword(tmp_path):
    """Traits write ``classSide >>`` where classes write ``class >>``.

    The grammar only accepts the latter, so the source is normalised before
    parsing. Without this the whole file fails to parse and its class-side
    methods are misattributed to the instance side.
    """
    source = """\"I am a trait.\"
Trait {
\t#name : 'TExample',
\t#package : 'Example'
}

{ #category : 'testing' }
TExample classSide >> buildOne [
\t^ 1
]

{ #category : 'testing' }
TExample >> one [
\t^ 1
]
"""
    path = tmp_path / "TExample.trait.st"
    path.write_text(source)
    parsed = extract_file(path)

    assert not parsed.parse_error
    klass = parsed.classes[0]
    assert klass.kind == "trait"
    by_selector = {m.selector: m for m in klass.methods}
    assert by_selector["buildOne"].class_side is True
    assert by_selector["one"].class_side is False


def test_normalization_preserves_byte_offsets():
    """Line numbers in findings must survive the classSide rewrite."""
    from stsmell.parsing import normalize_source

    raw = b"\nTExample classSide >> buildOne [\n\t^ 1\n]\n"
    normalized = normalize_source(raw)
    assert len(normalized) == len(raw)
    assert b"classSide" not in normalized
    assert raw.index(b">>") == normalized.index(b">>")
