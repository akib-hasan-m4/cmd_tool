"""Metric goldens.

Expected values here are computed by hand. Cyclomatic complexity in a language
with no ``if`` is easy to get subtly wrong, and a self-generated golden file
would simply enshrine the bug.
"""

from conftest import metrics_for


def test_accessors_are_recognised(fixtures):
    methods, _ = metrics_for(fixtures / "clean" / "CleanPoint.class.st")
    assert methods["x"].is_accessor is True
    assert methods["y"].is_accessor is True
    assert methods["+"].is_accessor is False
    # A two-statement assignment method is not a setter.
    assert methods["setX:y:"].is_accessor is False


def test_cyclomatic_complexity_counts_branching_selectors(fixtures):
    """classify: has 8 ifTrue:, 1 and:, 1 or: -> 10 branches, so CYCLO = 11."""
    methods, _ = metrics_for(fixtures / "smelly" / "ComplexMethodExample.class.st")
    assert methods["classify:"].cyclo == 11


def test_straight_line_method_has_complexity_one(fixtures):
    methods, _ = metrics_for(fixtures / "smelly" / "LongMethodExample.class.st")
    assert methods["runEverything"].cyclo == 1
    # 27 assignments plus one return.
    assert methods["runEverything"].mloc == 28


def test_message_chain_length_and_foreignness(fixtures):
    """order customer address street name asUppercase -> 5 sends, rooted at an ivar."""
    methods, _ = metrics_for(fixtures / "smelly" / "MessageChainExample.class.st")
    m = methods["streetName"]
    assert m.maxchain == 5
    assert m.foreign_maxchain == 5


def test_chains_rooted_at_self_are_not_foreign(tmp_path):
    source = """\"c\"
Class { #name : 'C', #superclass : 'Object' }

{ #category : 'x' }
C >> deep [
\t^ self alpha beta gamma delta
]
"""
    path = tmp_path / "C.class.st"
    path.write_text(source)
    methods, _ = metrics_for(path)
    assert methods["deep"].maxchain == 4
    assert methods["deep"].foreign_maxchain == 0


def test_temporary_declarations_are_not_variable_accesses(tmp_path):
    source = """\"c\"
Class { #name : 'C', #superclass : 'Object' }

{ #category : 'x' }
C >> run [
\t| unusedOne unusedTwo used |
\tused := 1.
\t^ used
]
"""
    path = tmp_path / "C.class.st"
    path.write_text(source)
    methods, _ = metrics_for(path)
    # Only 'used' is actually accessed; the other two are declarations.
    assert methods["run"].noav == 1
    assert methods["run"].temps == 3


def test_binary_sends_do_not_count_as_feature_envy(tmp_path):
    """`amount > 0` is arithmetic, not delegation."""
    source = """\"c\"
Class { #name : 'C', #superclass : 'Object' }

{ #category : 'x' }
C >> check: amount [
\t^ amount > 0
]
"""
    path = tmp_path / "C.class.st"
    path.write_text(source)
    methods, _ = metrics_for(path)
    assert methods["check:"].foreign_sends == {}


def test_locally_built_temporaries_are_not_envy_targets(fixtures):
    """The envy target is `customer`, not the `parts` collection built locally."""
    methods, _ = metrics_for(fixtures / "smelly" / "FeatureEnvyExample.class.st")
    name, count = methods["describeCustomer"].top_foreign
    assert name == "customer"
    assert count == 6


def test_block_nesting_depth(tmp_path):
    source = """\"c\"
Class { #name : 'C', #superclass : 'Object' }

{ #category : 'x' }
C >> nested [
\t^ true ifTrue: [ true ifTrue: [ true ifTrue: [ 1 ] ] ]
]
"""
    path = tmp_path / "C.class.st"
    path.write_text(source)
    methods, _ = metrics_for(path)
    assert methods["nested"].nblk == 3


def test_class_metrics_and_cohesion(fixtures):
    _, classes = metrics_for(fixtures / "clean" / "CleanPoint.class.st")
    cm = classes["CleanPoint"]
    assert cm.nom == 4
    assert cm.niv == 2
    assert cm.wmc == 4  # four methods, each of complexity 1
    assert cm.dit == 1  # superclass Object lies outside the analysed source
    assert cm.acc_ratio == 0.5
    # '+' and 'setX:y:' both touch x and y, so cohesion is total.
    assert cm.tcc == 1.0


def test_data_class_is_all_accessors(fixtures):
    _, classes = metrics_for(fixtures / "smelly" / "DataClassExample.class.st")
    cm = classes["DataClassExample"]
    assert cm.nom == 6
    assert cm.acc_ratio == 1.0
    # Every method is an accessor, so no pair is eligible for cohesion.
    assert cm.tcc is None


def test_cascade_length(tmp_path):
    source = """\"c\"
Class { #name : 'C', #superclass : 'Object' }

{ #category : 'x' }
C >> build [
\t^ OrderedCollection new add: 1; add: 2; add: 3; yourself
]
"""
    path = tmp_path / "C.class.st"
    path.write_text(source)
    methods, _ = metrics_for(path)
    assert methods["build"].max_cascade == 4


def test_hierarchy_depth_and_children(fixtures):
    from stsmell.analysis import analyze

    result = analyze(fixtures / "bequest")
    h = result.suite.hierarchy
    assert h.depth["AbstractShape"] == 1
    assert h.depth["RefusingShape"] == 2
    assert h.num_children("AbstractShape") == 1
    assert h.inherited_ivars("RefusingShape") == {"origin"}
