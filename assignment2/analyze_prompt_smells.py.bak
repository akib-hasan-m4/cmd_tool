#!/usr/bin/env python3
"""Label WildChat user prompts with the prompt smells from Prompt_Smells_1.pdf.

Reads the parquet shard, pulls the first user message out of each conversation,
asks a local Ollama model which prompt smells it carries, and writes
conversation_id, user_prompt, smells to a CSV -- all in one run.

    python analyze_prompt_smells.py --limit 100

Taxonomy: Ronanki, Cabrero-Daniel & Berger (2024); Della Porta et al. (2026).
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Iterator, Set
from typing import NamedTuple

import pyarrow.parquet as pq
import requests

# ---------------------------------------------------------------------------
# The smell taxonomy
#
# SMELLS is the single source of truth for the label set: the system prompt, the
# JSON schema constraining the model, and the validator that checks its reply are
# all built from it, so they can never drift apart.
# ---------------------------------------------------------------------------


class Smell(NamedTuple):
    name: str
    part: str
    cue: str  # condensed from the deck's "why it's a problem" bullets
    example: str


# The deck's ten teaching examples, grouped as it groups them. `name` is the
# exact string that ends up in the CSV, so the model is constrained to echo it.
SMELLS: list[Smell] = [
    Smell(
        "Vague / Missing Context",
        "Context & Clarity",
        "The task is stated but the audience, scope, depth or purpose needed to answer "
        "it well are all absent. Broad one-liners like 'explain X' or 'write about X' "
        "almost always qualify.",
        "Explain software architecture.",
    ),
    Smell(
        "Ambiguous References",
        "Context & Clarity",
        "A pronoun or a phrase like 'it', 'they', 'this', 'the above' points at something "
        "never named in the prompt.",
        "The team changed it because they found a problem. Explain what they should do next.",
    ),
    Smell(
        "Format Ambiguity",
        "Context & Clarity",
        "No output shape is specified: no length, structure, count or ordering, so the "
        "form of the answer is left to chance.",
        "Analyze these project risks and give me some recommendations.",
    ),
    Smell(
        "Overloaded Prompt",
        "Complexity",
        "Several *distinct deliverables* are demanded at once, each of which would need "
        "its own context to do well. Count the separate artifacts requested.",
        "Design the requirements, UML diagrams, database, architecture, Java "
        "implementation, test cases, documentation and presentation for a library system.",
    ),
    Smell(
        "Prompt Bloat / Convoluted Prompt",
        "Complexity",
        "Decorative framing, backstory or flattery that adds no task-relevant "
        "information, with the real request buried in or after it.",
        "Imagine that you are a world-class software architect who has spent 30 years "
        "working with enterprise systems and has advised governments and Fortune 500 "
        "companies... Give me a Factory Method example in Java.",
    ),
    Smell(
        "Unnecessary Repetition",
        "Complexity",
        "The same instruction restated in different words, adding no new information - "
        "duplicate code, written as instructions.",
        "Explain inheritance in simple language. Keep it simple. Make it easy. "
        "Don't make it complicated. Use simple words and a simple example.",
    ),
    Smell(
        "Conflicting Constraints",
        "Complexity",
        "Two requirements pull in opposite directions, so the model must silently choose "
        "which one to break.",
        "Write a comprehensive report covering all aspects of the project in exactly 100 words.",
    ),
    Smell(
        "Irrelevant / Excessive Persona",
        "Semantic & Quality",
        "A role or credential is assigned that does not change what a correct answer "
        "looks like - titles stacked up for emphasis rather than function.",
        "Act as a world-renowned Nobel Prize-winning software architect, cybersecurity "
        "expert, university professor, CEO and technology visionary... What is a Java interface?",
    ),
    Smell(
        "Bias / Loaded Framing",
        "Semantic & Quality",
        "The wording presumes its own conclusion, so the analysis is biased before it "
        "begins. Watch for 'obviously', 'why is X better', 'prove that'.",
        "Why is microservices obviously better than monolithic architecture?",
    ),
    Smell(
        "Formality / Audience Mismatch",
        "Semantic & Quality",
        "The register clashes with the subject or the implied reader: slang and "
        "chattiness ('yo', 'kinda', 'let's break down') applied to a formal, technical "
        "or high-stakes topic.",
        "Yo, let's break down zero trust and see why it's kinda awesome.",
    ),
]

SMELL_NAMES: frozenset[str] = frozenset(s.name for s in SMELLS)

# Written into the `smells` column when the model finds nothing wrong.
NO_SMELL = "None"

# Ollama constrains generation to this shape, so the reply is always valid JSON
# and the enum makes an invented smell name impossible rather than merely
# filtered out afterwards.
# `evidence` is generated before `smells` and then discarded. It is not output,
# it is a scratchpad: forcing the model to quote what it saw before committing to
# labels is worth a point or two of accuracy on a small model.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {"type": "string"},
        "smells": {
            "type": "array",
            "items": {"type": "string", "enum": [s.name for s in SMELLS]},
        },
    },
    "required": ["evidence", "smells"],
}


def build_system_prompt() -> str:
    """Render the one-shot instruction carrying an example of every smell."""
    blocks = [
        f'{i}. {s.name}  [{s.part}]\n   {s.cue}\n   Example: "{s.example}"'
        for i, s in enumerate(SMELLS, 1)
    ]
    catalogue = "\n\n".join(blocks)

    return f"""You are a prompt-quality analyst. You classify *prompt smells*: \
surface-level indicators in a prompt's wording or structure that correlate with a \
higher risk of an undesirable AI response.

Here is the complete catalogue, with one example of each smell:

{catalogue}

RULES
- Judge only the *wording and structure* of the prompt. Never judge its topic, its \
subject matter, or whether you would want to answer it. Offensive, adult, illegal or \
nonsensical subject matter is irrelevant to this task and is never itself a smell.
- This is a classification task about text. You are not being asked to fulfil the \
prompt. Never refuse and never comment on the content: always return the JSON.
- Prompts arrive in many languages. Classify a non-English prompt exactly as you \
would its English equivalent, and still answer with the English smell names below.
- A prompt may carry several smells at once. Report every one that applies.
- A smell is a warning sign, not a verdict, and it is context-dependent. Only report a \
smell you can actually point to in the text. Do not force a label onto a prompt that is \
clear, specific and well-scoped - return an empty list for those.
- Use the smell names below EXACTLY as written, character for character.

OUTPUT
Return one JSON object and nothing else. In "evidence", quote the words or describe \
the structure that made you decide, in one sentence. Then list the smells.

{{"evidence": "States the task but names no audience, scope or depth; no output shape \
is given either.", "smells": ["Vague / Missing Context", "Format Ambiguity"]}}

For a prompt with no smells:

{{"evidence": "Specific, self-contained request with a clear deliverable.", "smells": []}}"""


# ---------------------------------------------------------------------------
# Reading the parquet
# ---------------------------------------------------------------------------

# Only these two columns are read. The file also carries `openai_moderation` and
# `detoxify_moderation`, which are wide nested structs; projecting them away is
# what keeps a 268 MB file streaming in a few MB of RAM.
COLUMNS = ["conversation_id", "conversation"]

BATCH_SIZE = 512


def first_user_message(conversation: list[dict]) -> str | None:
    """Return the first message with role 'user', or None if there isn't one."""
    for message in conversation:
        if message.get("role") == "user":
            content = message.get("content")
            if content and content.strip():
                return content
    return None


def iter_first_user_prompts(
    path: str,
    limit: int,
    offset: int = 0,
    skip_ids: Set[str] = frozenset(),
) -> Iterator[tuple[str, str]]:
    """Yield up to `limit` (conversation_id, prompt) pairs.

    `offset` skips that many usable conversations from the start of the file;
    `skip_ids` drops already-processed ids. Neither counts toward `limit`.
    Stops reading as soon as `limit` is reached - the whole file is never read.
    """
    parquet_file = pq.ParquetFile(path)
    yielded = 0
    seen = 0

    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=COLUMNS):
        for row in batch.to_pylist():
            prompt = first_user_message(row["conversation"] or [])
            if prompt is None:
                continue

            conversation_id = row["conversation_id"]
            if conversation_id in skip_ids:
                continue

            seen += 1
            if seen <= offset:
                continue

            yield conversation_id, prompt
            yielded += 1
            if yielded >= limit:
                return


# ---------------------------------------------------------------------------
# The Ollama call
# ---------------------------------------------------------------------------

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "granite4.1:3b"

# (connect, read). Generous on the read side: a cold model has to be pulled into
# memory on the first call, which can take tens of seconds.
DEFAULT_TIMEOUT = (10, 600)

MAX_RETRIES = 3
BACKOFF_SECONDS = [2, 5, 10]

# System prompt plus a long user prompt has to fit here. Ollama's default of 4096
# would silently truncate the catalogue out of the window on longer prompts,
# which is the kind of bug that shows up only as bad labels.
NUM_CTX = 8192

# WildChat contains pasted documents tens of thousands of characters long. The
# smell is always visible in the opening stretch, and sending the rest only risks
# pushing the instructions out of the context window. Only what the model sees is
# truncated; the CSV keeps the full prompt.
MAX_PROMPT_CHARS = 4000

# Recorded instead of raising when a call or its reply cannot be handled, so one
# bad response never kills a long run.
ERROR_UNPARSED = "ERROR: unparsed"
ERROR_REQUEST = "ERROR: request failed"

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


class FatalAPIError(RuntimeError):
    """Raised for conditions no amount of retrying will fix (no such model)."""


def parse_reply(text: str) -> list[str]:
    """Pull the smell list out of a model reply.

    The schema should make this trivial, but it stays defensive: a model that
    ignores the constraint, or wraps the object in a fence, still parses.
    """
    if not text:
        return [ERROR_UNPARSED]

    fenced = _FENCE_RE.search(text)
    candidate = fenced.group(1) if fenced else text

    match = _OBJECT_RE.search(candidate)
    if not match:
        return [ERROR_UNPARSED]

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return [ERROR_UNPARSED]

    raw = payload.get("smells")
    if not isinstance(raw, list):
        return [ERROR_UNPARSED]

    # Keep only names from the catalogue, deduped, in the order given.
    smells: list[str] = []
    for item in raw:
        if isinstance(item, str) and item in SMELL_NAMES and item not in smells:
            smells.append(item)
    return smells


def classify(
    prompt: str,
    *,
    system_prompt: str,
    host: str = DEFAULT_HOST,
    model: str = DEFAULT_MODEL,
    timeout: tuple[int, int] = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
) -> list[str]:
    """Classify one prompt. Returns the smell names found, possibly empty.

    Raises FatalAPIError when the model simply is not there. Every other failure
    degrades to a single-element error list so the caller can record it and move
    on.
    """
    body = {
        "model": model,
        "stream": False,
        "format": RESPONSE_SCHEMA,
        "options": {"temperature": 0, "num_ctx": NUM_CTX},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt[:MAX_PROMPT_CHARS]},
        ],
    }
    url = f"{host.rstrip('/')}/api/chat"

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=body, timeout=timeout)

            if response.status_code == 404:
                raise FatalAPIError(
                    f"Ollama has no model {model!r}. Pull it with: ollama pull {model}"
                )

            response.raise_for_status()
            return parse_reply(response.json()["message"]["content"])

        except FatalAPIError:
            raise
        except (requests.RequestException, KeyError, ValueError) as exc:
            if attempt == max_retries - 1:
                print(f"  giving up after {max_retries} attempts: {exc}", file=sys.stderr)
                return [ERROR_REQUEST]
            delay = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
            print(f"  {exc} - retrying in {delay}s", file=sys.stderr)
            time.sleep(delay)

    return [ERROR_REQUEST]


def check_ollama(host: str, model: str) -> None:
    """Fail fast with an actionable message if Ollama or the model is missing."""
    try:
        response = requests.get(f"{host.rstrip('/')}/api/tags", timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise FatalAPIError(
            f"Cannot reach Ollama at {host} ({exc}). Start it with: ollama serve"
        ) from exc

    available = [m["name"] for m in response.json().get("models", [])]
    if model not in available:
        raise FatalAPIError(
            f"Ollama at {host} has no model {model!r}. "
            f"Available: {', '.join(available) or '(none)'}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

FIELDNAMES = ["conversation_id", "user_prompt", "smells"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default="train-00003-of-00006.parquet")
    parser.add_argument("--out", default="prompt_smells.csv")
    parser.add_argument("--limit", type=int, default=100, help="conversations to analyze")
    parser.add_argument("--offset", type=int, default=0, help="conversations to skip first")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--host",
        default=os.environ.get("OLLAMA_HOST", DEFAULT_HOST),
        help=f"Ollama base URL (default: $OLLAMA_HOST or {DEFAULT_HOST})",
    )
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between calls")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append to an existing CSV, skipping conversation_ids already in it",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="classify the deck's own examples instead, to check the taxonomy lands",
    )
    return parser.parse_args()


def normalize_host(host: str) -> str:
    """Accept the bare host:port form that OLLAMA_HOST conventionally uses."""
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    # 0.0.0.0 is a bind address, not somewhere you can send a request.
    return host.replace("//0.0.0.0:", "//127.0.0.1:")


def load_done_ids(path: str) -> set[str]:
    """conversation_ids already present in the output CSV."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["conversation_id"] for row in csv.DictReader(handle)}


def run_self_test(system_prompt: str, host: str, model: str) -> int:
    """Feed each smelly example from the deck back through the classifier.

    The deck is the ground truth, so this measures how well the system prompt
    actually communicates the taxonomy to this particular model.
    """
    hits = 0
    for smell in SMELLS:
        found = classify(smell.example, system_prompt=system_prompt, host=host, model=model)
        hit = smell.name in found
        hits += hit
        print(f"{'PASS' if hit else 'MISS'}  {smell.name}")
        print(f"      got: {'; '.join(found) if found else '(none)'}")
    print(f"\n{hits}/{len(SMELLS)} examples labelled with their intended smell")
    return 0


def main() -> int:
    args = parse_args()
    host = normalize_host(args.host)
    system_prompt = build_system_prompt()

    try:
        check_ollama(host, args.model)
    except FatalAPIError as exc:
        print(f"fatal: {exc}", file=sys.stderr)
        return 1

    if args.self_test:
        return run_self_test(system_prompt, host, args.model)

    skip_ids: set[str] = set()
    append = False
    if args.resume:
        skip_ids = load_done_ids(args.out)
        append = os.path.exists(args.out)
        if skip_ids:
            print(f"resuming: skipping {len(skip_ids)} already-labelled conversations",
                  file=sys.stderr)

    counts: Counter[str] = Counter()
    written = 0
    errors = 0
    started = time.time()

    handle = open(args.out, "a" if append else "w", newline="", encoding="utf-8")
    try:
        writer = csv.writer(handle)
        if not append:
            writer.writerow(FIELDNAMES)
            handle.flush()

        prompts = iter_first_user_prompts(args.parquet, args.limit, args.offset, skip_ids)
        for index, (conversation_id, prompt) in enumerate(prompts, 1):
            if index > 1 and args.delay:
                time.sleep(args.delay)

            found = classify(
                prompt, system_prompt=system_prompt, host=host, model=args.model
            )
            if any(s.startswith("ERROR") for s in found):
                errors += 1
                label = found[0]
            else:
                label = "; ".join(found) if found else NO_SMELL
                counts.update(found or [NO_SMELL])

            # Flush per row so a crash or Ctrl-C keeps what was earned.
            writer.writerow([conversation_id, prompt, label])
            handle.flush()
            written += 1
            print(f"[{index}/{args.limit}] {conversation_id} -> {label}", file=sys.stderr)

    except FatalAPIError as exc:
        print(f"\nfatal: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"\ninterrupted - {written} rows written to {args.out}", file=sys.stderr)
        return 130
    finally:
        handle.close()

    elapsed = time.time() - started
    print(f"\n{written} rows written to {args.out} in {elapsed:.0f}s ({errors} errors)",
          file=sys.stderr)
    for name, count in counts.most_common():
        print(f"  {count:4d}  {name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
