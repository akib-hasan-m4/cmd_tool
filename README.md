# stsmell

A command-line code smell detector for **Pharo/Smalltalk**, reading Tonel-format
source straight from a Git checkout.

Mainstream OO languages have saturated static-analysis tooling. Smalltalk — the
purest class-based OO language still in real use — has almost none outside the
image. Pharo's own tools (Moose, SmallLint) run *inside* a live image, so they
cannot analyse a pull request in CI. `stsmell` runs on plain files, reports in
text, JSON or SARIF, and returns a non-zero exit code when it finds something.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

Requires Python 3.10+. The two dependencies (`tree-sitter`,
`tree-sitter-tonel-smalltalk`) ship prebuilt wheels — no C compiler needed.

## Use

```bash
stsmell analyze src/                      # report smells
stsmell analyze src/ --format sarif       # for GitHub code scanning
stsmell analyze src/ --min-severity major # only serious findings
stsmell metrics src/ --scope class        # dump the raw metric layer as CSV
stsmell rules                             # list rules and active thresholds
stsmell baseline src/ -o baseline.json    # freeze today's findings
stsmell analyze src/ --baseline baseline.json   # report only what is new
```

Exit codes: `0` no findings, `1` findings at or above `--min-severity`, `2` error.

## Try it on a real project

These are Tonel-format Pharo projects on GitHub. Clone one and point `stsmell` at
its source directory — no Pharo image required.

| Project | What it is | Tonel files in `src/` |
|---|---|---|
| [PolyMathOrg/DataFrame](https://github.com/PolyMathOrg/DataFrame) | Tabular data structures for Pharo | 31 classes, 7 extensions |
| [pillar-markup/Microdown](https://github.com/pillar-markup/Microdown) | The Markdown-like markup parser used across Pharo docs | 473 classes, 67 extensions |
| [pharo-project/pharo](https://github.com/pharo-project/pharo) | The Pharo kernel — the corpus these defaults are calibrated on | 7,256 classes, 1,480 extensions, 128 traits |

```bash
# Small — runs in seconds, good for a first look
git clone --depth 1 https://github.com/PolyMathOrg/DataFrame.git
stsmell analyze DataFrame/src

# Medium — enough code for the major-severity rules to have something to say
git clone --depth 1 https://github.com/pillar-markup/Microdown.git
stsmell analyze Microdown/src --min-severity major

# Large — the kernel itself; pinned to the branch the thresholds were derived from
git clone --depth 1 -b Pharo15 https://github.com/pharo-project/pharo.git
stsmell analyze pharo/src --min-severity critical
```

The two things worth doing with a corpus that size:

```bash
stsmell metrics pharo/src --scope class > pharo-classes.csv  # recalibrate thresholds
stsmell baseline Microdown/src -o baseline.json              # adopt on existing code
```

**Check the format before you clone.** Plenty of well-known Pharo projects predate
Tonel and still ship *filetree* — a tree of `.package/` directories rather than
`.class.st` files. [Seaside](https://github.com/SeasideSt/Seaside) and
[zinc](https://github.com/svenvc/zinc) are both filetree, and `stsmell` will scan
them and report nothing. A repository is Tonel if its source directory contains a
`.properties` file holding `{ #format : #tonel }`. See [Known
limitations](#known-limitations).

## Why the metrics are not just ported from Java

**Smalltalk has no control-flow syntax.** There is no `if`, no `while`, no
`for`. Every branch and loop is an ordinary message send taking block arguments:

```smalltalk
x > 0
    ifTrue: [ self grow ]
    ifFalse: [ self shrink ].

1 to: 10 do: [ :i | self step: i ].
```

Cyclomatic complexity therefore cannot be computed by counting keywords. It is
computed by recognising a curated set of *selectors* — `ifTrue:`, `and:`,
`whileTrue:`, `to:do:`, `detect:ifNone:`, `on:do:` and friends. That set lives in
`src/stsmell/selectors.py` and is the heart of the tool.

The same re-derivation applies throughout. Two examples where the naive port
gives the wrong answer, both caught by running against the Pharo kernel:

- **Refused Bequest.** The Java heuristic "subclass never touches inherited
  fields" misfires badly here, because idiomatic Smalltalk reaches inherited
  state through accessors (`self origin`) rather than naming the variable. On
  the Pharo corpus it produced 3707 findings against 16 real ones, so only the
  precise `subclassResponsibility`-override form is implemented.
- **Feature Envy.** Counting every send to a non-`self` receiver blames locals
  the method just built itself (`parts := OrderedCollection new. parts add: …`).
  Only instance variables and parameters are envy candidates, and binary sends
  (`amount > 0`) are excluded as arithmetic rather than delegation.

## Rules

| Rule | Severity | Default |
|---|---|---|
| `god-class` | critical | WMC ≥ 47 **and** TCC < 0.33 **and** NIV ≥ 5 |
| `long-method` | major | MLOC > 25 |
| `complex-method` | major | CYCLO > 8 |
| `deeply-nested-blocks` | major | block depth > 3 |
| `feature-envy` | major | ≥ 5 sends to one foreign receiver, exceeding sends to self |
| `large-class` | major | > 40 methods |
| `refused-bequest` | major | overrides an inherited method with `subclassResponsibility` |
| `extension-sprawl` | major | package extends > 10 foreign classes or adds > 30 methods |
| `long-parameter-list` | minor | > 4 parameters |
| `too-many-temporaries` | minor | > 6 temporaries |
| `message-chain` | minor | chain of > 4 sends on a foreign receiver |
| `data-class` | minor | ≥ 80% accessors |
| `deep-inheritance` | minor | DIT > 8 |
| `high-coupling` | minor | CBO > 15 |
| `god-class-side` | minor | > 15 class-side methods |
| `missing-class-comment` | minor | no class comment |
| `uses-this-context` | minor | reflective stack access |
| `cascade-abuse` | info | cascade of > 6 messages |

Four are Smalltalk-native with no Java analogue: `extension-sprawl` (packages
monkey-patching classes they do not own), `missing-class-comment`,
`god-class-side`, and `uses-this-context`.

## Where the thresholds come from

They are calibrated empirically, not imported from a Java linter. Running the
metric layer over the Pharo kernel — `pharo-project/pharo`, branch `Pharo15`,
`src/`, which is 8,864 Tonel definition files (7,256 classes + 1,480 extensions +
128 traits) holding 91,273 methods — gives:

| Metric | median | p90 | p95 | p99 | max |
|---|---|---|---|---|---|
| MLOC | 2 | 9 | 13 | 26 | 1961 |
| CYCLO | 1 | 3 | 4 | 7 | 38 |
| PARAMS | 0 | 2 | 2 | 4 | 15 |
| NBLK | 0 | 1 | 2 | 3 | 8 |
| NOM | 4 | 24 | 39 | 91 | 812 |
| WMC | 6 | 35 | 62 | 150 | 1520 |
| DIT | 5 | 7 | 8 | 9 | 12 |

The median Pharo method is **two lines long** — which is why a Java-style
50-line threshold would be useless here, and why `deep-inheritance` has to
tolerate a DIT of 8 when the median class already sits at 5. Defaults sit around
p95–p99, so a clean project stays quiet. Recalibrate for your own codebase with
`stsmell metrics`.

Caveat: the Pharo kernel contains genuinely old and genuinely smelly code.
Percentile calibration defines *normal*, not *good*.

## Configuration

Drop a `stsmell.toml` at the project root (see `stsmell.example.toml`); it is
discovered by walking up from the analysed path.

```toml
min_severity = "major"
exclude = ["*-Tests/*"]

[rules.long-method]
max_lines = 15

[rules.cascade-abuse]
enabled = false
```

## Known limitations

- Tonel format only (`.class.st`, `.trait.st`, `.extension.st`). Neither the
  older `!`-chunk fileOut format nor filetree (`.package/` directories, still
  used by projects such as Seaside and zinc) is supported; a filetree checkout
  scans clean rather than failing loudly.
- Roughly 0.16% of Pharo kernel files fail to fully parse (14 of 8,864) due to
  gaps in the upstream grammar: `[ :a ]` blocks with arguments but no body,
  `##literal` bindings, non-ASCII symbols, and nested literal arrays.
  tree-sitter recovers, so the rest of each such file is still analysed, and the
  count is always reported rather than hidden.
- Trait `classSide >>` is normalised to `class >>` before parsing, since the
  grammar does not accept it. The rewrite preserves byte offsets exactly, so
  reported line numbers stay correct.

## Development

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests/ -q
```
