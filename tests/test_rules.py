"""Rule behaviour: each smell fires where expected, and stays quiet otherwise."""

import pytest

from stsmell.analysis import analyze
from stsmell.config import Config


def rules_fired(path, config=None):
    return {f.rule for f in analyze(path, config).findings}


def rules_for_file(path, config=None):
    result = analyze(path, config)
    return {f.rule for f in result.findings}


def test_clean_code_produces_no_findings(fixtures):
    result = analyze(fixtures / "clean")
    assert result.findings == []


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("LongMethodExample.class.st", "long-method"),
        ("ComplexMethodExample.class.st", "complex-method"),
        ("LongParameterListExample.class.st", "long-parameter-list"),
        ("MessageChainExample.class.st", "message-chain"),
        ("FeatureEnvyExample.class.st", "feature-envy"),
        ("ThisContextExample.class.st", "uses-this-context"),
        ("UncommentedClass.class.st", "missing-class-comment"),
        ("DataClassExample.class.st", "data-class"),
    ],
)
def test_each_fixture_triggers_its_smell(fixtures, filename, expected):
    assert expected in rules_for_file(fixtures / "smelly" / filename)


def test_commented_classes_do_not_trigger_missing_comment(fixtures):
    """Only the deliberately uncommented fixture should trigger the rule."""
    result = analyze(fixtures / "smelly")
    offenders = {f.entity for f in result.findings if f.rule == "missing-class-comment"}
    assert offenders == {"UncommentedClass"}


def test_refused_bequest_detects_subclass_responsibility(fixtures):
    result = analyze(fixtures / "bequest")
    findings = [f for f in result.findings if f.rule == "refused-bequest"]
    assert len(findings) == 1
    assert findings[0].entity == "RefusingShape"
    assert "draw" in findings[0].message


def test_extension_sprawl_is_quiet_under_default_thresholds(fixtures):
    assert "extension-sprawl" not in rules_fired(fixtures / "extensions")


def test_extension_sprawl_fires_when_thresholds_lowered(fixtures):
    config = Config(
        rules={"extension-sprawl": {"max_extended_classes": 1, "max_extension_methods": 2}}
    )
    findings = [
        f for f in analyze(fixtures / "extensions", config).findings
        if f.rule == "extension-sprawl"
    ]
    assert len(findings) == 1
    assert findings[0].metrics["extended_classes"] == 2
    assert findings[0].metrics["extension_methods"] == 4


def test_god_class_needs_at_least_two_eligible_methods(fixtures):
    """TCC is undefined for a single-method class, so the composite must not fire."""
    config = Config(rules={"god-class": {"min_wmc": 1, "max_tcc": 1.0, "min_niv": 1}})
    findings = [
        f
        for f in analyze(
            fixtures / "smelly" / "ComplexMethodExample.class.st", config
        ).findings
        if f.rule == "god-class"
    ]
    assert findings == []


def test_god_class_fires_on_large_incohesive_class(fixtures):
    """The fixture has WMC 15 and TCC 0.0; lower the WMC floor from its default 47."""
    config = Config(rules={"god-class": {"min_wmc": 10}})
    findings = [
        f for f in analyze(fixtures / "god", config).findings if f.rule == "god-class"
    ]
    assert len(findings) == 1
    assert findings[0].entity == "GodClassExample"
    assert findings[0].metrics == {"WMC": 15, "TCC": 0.0, "NIV": 5}
    assert findings[0].severity == "critical"


def test_god_class_quiet_at_default_thresholds(fixtures):
    """Defaults are calibrated well above the fixture, so it must stay silent."""
    assert "god-class" not in rules_fired(fixtures / "god")


def test_disabling_a_rule_silences_it(fixtures):
    config = Config(rules={"long-method": {"enabled": False}})
    assert "long-method" not in rules_fired(fixtures / "smelly", config)


def test_raising_a_threshold_silences_a_rule(fixtures):
    config = Config(rules={"long-method": {"max_lines": 100}})
    assert "long-method" not in rules_fired(fixtures / "smelly", config)


def test_findings_are_sorted_most_severe_first(fixtures):
    result = analyze(fixtures / "smelly")
    ranks = [f.severity_rank for f in result.findings]
    assert ranks == sorted(ranks, reverse=True)


def test_every_rule_has_id_description_and_severity():
    from stsmell.config import SEVERITY_ORDER
    from stsmell.rules import registry

    ids = set()
    for rule in registry():
        assert rule.id and rule.id not in ids, f"duplicate or missing id: {rule.id!r}"
        ids.add(rule.id)
        assert rule.name
        assert rule.description
        assert rule.severity in SEVERITY_ORDER


def test_refused_bequest_ignores_unused_inherited_state(tmp_path):
    """Idiomatic Smalltalk reaches inherited state through accessors.

    A subclass that never names an inherited instance variable directly is
    normal, not a smell -- on the Pharo corpus that heuristic produced 3707
    findings against 16 real ones, so it is deliberately not implemented.
    """
    (tmp_path / "Base.class.st").write_text(
        '"Base."\nClass {\n\t#name : \'Base\',\n\t#superclass : \'Object\',\n'
        "\t#instVars : [ 'alpha', 'beta' ]\n}\n\n"
        "{ #category : 'accessing' }\nBase >> alpha [\n\t^ alpha\n]\n"
    )
    (tmp_path / "Child.class.st").write_text(
        '"Child."\nClass {\n\t#name : \'Child\',\n\t#superclass : \'Base\',\n'
        "\t#instVars : [ 'own' ]\n}\n\n"
        "{ #category : 'accessing' }\nChild >> own [\n\t^ own\n]\n\n"
        "{ #category : 'acting' }\nChild >> act [\n\t^ self alpha\n]\n\n"
        "{ #category : 'acting' }\nChild >> actMore [\n\t^ self alpha + 1\n]\n"
    )
    assert "refused-bequest" not in rules_fired(tmp_path)
