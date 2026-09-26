#!/usr/bin/env python3
"""Label WildChat user prompts with the prompt smells from Prompt_Smells_1.pdf.

Reads the parquet shard, takes every English user message (not just the first
of each conversation), and asks two models which prompt smells it carries:

  * qwen3.5:0.8b through a local Ollama server (generative, JSON-constrained)
  * Kev-0.8B through a local Kev server (one yes/no question per smell)

Each message gets its own CSV row with one smell column per model. The run
stops once --limit rows are in the CSV, counting rows kept by --resume.

    python analyze_prompt_smells.py --limit 10000 --resume

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
from concurrent.futures import ThreadPoolExecutor
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

# WildChat tags every message with its detected language.
LANGUAGE = "English"


class Message(NamedTuple):
    conversation_id: str
    user_turn: int  # 1-based position among the conversation's user messages
    prompt: str


def iter_user_messages(
    path: str,
    limit: int,
    skip_keys: Set[tuple[str, int]] = frozenset(),
    language: str = LANGUAGE,
) -> Iterator[Message]:
    """Yield up to `limit` user messages tagged `language`, in file order.

    Every user message of a conversation is yielded, not only the first. A
    message is identified by (conversation_id, user_turn); keys in `skip_keys`
    are already done and do not count toward `limit`.
    """
    if limit <= 0:
        return
    parquet_file = pq.ParquetFile(path)
    yielded = 0

    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=COLUMNS):
        for row in batch.to_pylist():
            conversation_id = row["conversation_id"]
            user_turn = 0
            for message in row["conversation"] or []:
                if message.get("role") != "user":
                    continue
                user_turn += 1
                content = message.get("content")
                if not content or not content.strip():
                    continue
                if message.get("language") != language:
                    continue
                if (conversation_id, user_turn) in skip_keys:
                    continue

                yield Message(conversation_id, user_turn, content)
                yielded += 1
                if yielded >= limit:
                    return


# ---------------------------------------------------------------------------
# Shared request settings
# ---------------------------------------------------------------------------

# (connect, read). Generous on the read side: a cold model has to be pulled into
# memory on the first call, which can take tens of seconds.
DEFAULT_TIMEOUT = (10, 600)

MAX_RETRIES = 3
BACKOFF_SECONDS = [2, 5, 10]

# WildChat contains pasted documents tens of thousands of characters long. The
# smell is always visible in the opening stretch, and sending the rest only risks
# pushing the instructions out of the context window. Only what the models see
# is truncated; the CSV keeps the full prompt.
MAX_PROMPT_CHARS = 4000

# Recorded instead of raising when a call or its reply cannot be handled, so one
# bad response never kills a long run.
ERROR_UNPARSED = "ERROR: unparsed"
ERROR_REQUEST = "ERROR: request failed"


class FatalAPIError(RuntimeError):
    """Raised for conditions no amount of retrying will fix (no such model)."""


class ServerError(RuntimeError):
    """A 5xx the caller asked to handle itself instead of retrying as-is."""


def post_with_retries(
    url: str, body: dict, timeout, what: str, raise_5xx: bool = False
) -> dict | None:
    """POST and return the JSON body, or None once the retries are spent.

    A 404 means the model is not there, which retrying cannot fix. With
    `raise_5xx`, a server error raises ServerError at once, because resending
    the identical request would fail the identical way.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, json=body, timeout=timeout)
            if response.status_code == 404:
                raise FatalAPIError(f"{what}: 404 from {url} - is the model loaded?")
            if raise_5xx and response.status_code >= 500:
                raise ServerError(f"{what}: {response.status_code} from {url}")
            response.raise_for_status()
            return response.json()
        except (FatalAPIError, ServerError):
            raise
        except (requests.RequestException, ValueError) as exc:
            if attempt == MAX_RETRIES - 1:
                print(f"  {what}: giving up after {MAX_RETRIES} attempts: {exc}",
                      file=sys.stderr)
                return None
            delay = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
            print(f"  {what}: {exc} - retrying in {delay}s", file=sys.stderr)
            time.sleep(delay)
    return None


# ---------------------------------------------------------------------------
# Ollama (qwen3.5:0.8b)
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3.5:0.8b"

# System prompt plus a long user prompt has to fit here. Ollama's default of 4096
# would silently truncate the catalogue out of the window on longer prompts,
# which is the kind of bug that shows up only as bad labels.
NUM_CTX = 8192

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


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


def classify_ollama(prompt: str, *, system_prompt: str, host: str, model: str) -> list[str]:
    """Classify one prompt with an Ollama model. Returns the smell names found."""
    body = {
        "model": model,
        "stream": False,
        # Qwen3.5 thinks by default; that multiplies the latency for no gain on a
        # schema-constrained label.
        "think": False,
        "format": RESPONSE_SCHEMA,
        "options": {"temperature": 0, "num_ctx": NUM_CTX},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt[:MAX_PROMPT_CHARS]},
        ],
    }
    reply = post_with_retries(f"{host}/api/chat", body, DEFAULT_TIMEOUT, "ollama")
    if reply is None:
        return [ERROR_REQUEST]
    try:
        return parse_reply(reply["message"]["content"])
    except (KeyError, TypeError):
        return [ERROR_UNPARSED]


def check_ollama(host: str, model: str) -> None:
    """Fail fast with an actionable message if Ollama or the model is missing."""
    try:
        response = requests.get(f"{host}/api/tags", timeout=10)
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
# Kev (Kev-0.8B)
#
# Kev does not generate text: it scores questions about a state. Each smell is
# asked as its own yes/no (`noul`) question and a smell is reported when the
# probability of yes reaches KEV_THRESHOLD. The raw probabilities go in their
# own column so the threshold can be revisited without re-running.
# ---------------------------------------------------------------------------

DEFAULT_KEV_HOST = "http://127.0.0.1:8009"
DEFAULT_KEV_MODEL = "kev-latest"
KEV_THRESHOLD = 0.5

KEV_QUESTIONS = {
    f"smell_{i}": {
        "type": "noul",
        "instructions": f"Does this prompt show the prompt smell \"{s.name}\"? {s.cue}",
    }
    for i, s in enumerate(SMELLS)
}
KEV_QUESTION_SMELL = {f"smell_{i}": s.name for i, s in enumerate(SMELLS)}


# Kev-0.8B in fp32 on a 6 GB card runs out of GPU memory on token-dense input:
# Qwen tokenizes every digit separately, so 1,000 characters of binary is ~1,000
# tokens, counted once per question. The server answers those with a 500, and
# the prompt is halved until it fits. Below this length the row is an error.
KEV_MIN_PROMPT_CHARS = 125

# Added to kev_probabilities when the prompt had to be cut to fit.
KEV_TRUNCATED_KEY = "_truncated_to_chars"


def classify_kev(
    prompt: str, *, host: str, model: str, threshold: float = KEV_THRESHOLD
) -> tuple[list[str], dict[str, float] | None]:
    """Classify one prompt with Kev. Returns (smells, probability per smell)."""
    limit = MAX_PROMPT_CHARS
    while True:
        body = {
            "model": model,
            "state": {"prompt": prompt[:limit]},
            "questions": KEV_QUESTIONS,
        }
        try:
            reply = post_with_retries(f"{host}/v1/systemone", body, DEFAULT_TIMEOUT, "kev",
                                      raise_5xx=True)
            break
        except ServerError as exc:
            if min(limit, len(prompt)) // 2 < KEV_MIN_PROMPT_CHARS:
                print(f"  {exc} - giving up at {min(limit, len(prompt))} chars",
                      file=sys.stderr)
                return [ERROR_REQUEST], None
            limit = min(limit, len(prompt)) // 2
            print(f"  {exc} - retrying with the first {limit} chars", file=sys.stderr)

    if reply is None:
        return [ERROR_REQUEST], None
    try:
        probs = {KEV_QUESTION_SMELL[k]: float(v["noul"]) for k, v in reply["answers"].items()}
    except (KeyError, TypeError, ValueError):
        return [ERROR_UNPARSED], None
    smells = [s.name for s in SMELLS if probs.get(s.name, 0.0) >= threshold]
    if limit < min(len(prompt), MAX_PROMPT_CHARS):
        probs[KEV_TRUNCATED_KEY] = limit
    return smells, probs


def check_kev(host: str, model: str) -> None:
    """Fail fast if the Kev server is down or does not serve `model`."""
    try:
        response = requests.get(f"{host}/v1/models", timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise FatalAPIError(
            f"Cannot reach Kev at {host} ({exc}). Start it with: "
            "python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009"
        ) from exc

    models = response.json().get("models", [])
    served = next((m for m in models if m.get("name") == model), None)
    if served is None:
        raise FatalAPIError(
            f"Kev at {host} has no model {model!r}. "
            f"Available: {', '.join(m.get('name', '?') for m in models) or '(none)'}"
        )
    print(f"kev: {served.get('run')} on {served.get('device')} ({served.get('dtype')})",
          file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "conversation_id",
    "user_turn",
    "user_prompt",
    "qwen_smells",
    "kev_smells",
    "kev_probabilities",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default="train-00003-of-00006.parquet")
    parser.add_argument("--out", default="prompt_smells_qwen_kev.csv")
    parser.add_argument(
        "--limit", type=int, default=10000,
        help="stop once the CSV holds this many rows (one per message, both models)",
    )
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
        help=f"Ollama base URL (default: $OLLAMA_HOST or {DEFAULT_OLLAMA_HOST})",
    )
    parser.add_argument("--kev-model", default=DEFAULT_KEV_MODEL)
    parser.add_argument("--kev-host", default=os.environ.get("KEV_HOST", DEFAULT_KEV_HOST))
    parser.add_argument("--kev-threshold", type=float, default=KEV_THRESHOLD)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append to an existing CSV, skipping messages already in it",
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
    return host.replace("//0.0.0.0:", "//127.0.0.1:").rstrip("/")


def load_done_keys(path: str) -> set[tuple[str, int]]:
    """(conversation_id, user_turn) pairs already present in the output CSV."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as handle:
        return {(row["conversation_id"], int(row["user_turn"]))
                for row in csv.DictReader(handle)}


def format_label(found: list[str]) -> str:
    if any(s.startswith("ERROR") for s in found):
        return found[0]
    return "; ".join(found) if found else NO_SMELL


def run_self_test(args, system_prompt: str) -> int:
    """Feed each smelly example from the deck back through both classifiers.

    The deck is the ground truth, so this measures how well each model picks up
    the taxonomy.
    """
    hits = Counter()
    for smell in SMELLS:
        qwen = classify_ollama(smell.example, system_prompt=system_prompt,
                               host=args.ollama_host, model=args.ollama_model)
        kev, probs = classify_kev(smell.example, host=args.kev_host, model=args.kev_model,
                                  threshold=args.kev_threshold)
        hits["qwen"] += smell.name in qwen
        hits["kev"] += smell.name in kev
        p = f"{probs[smell.name]:.2f}" if probs else "n/a"
        print(f"{smell.name}")
        print(f"   qwen {'PASS' if smell.name in qwen else 'MISS'}: {format_label(qwen)}")
        print(f"   kev  {'PASS' if smell.name in kev else 'MISS'} (p={p}): {format_label(kev)}")
    print(f"\nqwen {hits['qwen']}/{len(SMELLS)}, kev {hits['kev']}/{len(SMELLS)} "
          "examples labelled with their intended smell")
    return 0


def main() -> int:
    args = parse_args()
    args.ollama_host = normalize_host(args.ollama_host)
    args.kev_host = normalize_host(args.kev_host)
    system_prompt = build_system_prompt()

    try:
        check_ollama(args.ollama_host, args.ollama_model)
        check_kev(args.kev_host, args.kev_model)
    except FatalAPIError as exc:
        print(f"fatal: {exc}", file=sys.stderr)
        return 1

    if args.self_test:
        return run_self_test(args, system_prompt)

    done: set[tuple[str, int]] = set()
    append = False
    if args.resume:
        done = load_done_keys(args.out)
        append = os.path.exists(args.out)
        if done:
            print(f"resuming: {len(done)} messages already labelled", file=sys.stderr)

    remaining = args.limit - len(done)
    if remaining <= 0:
        print(f"{args.out} already holds {len(done)} rows (limit {args.limit}); nothing to do",
              file=sys.stderr)
        return 0

    counts = {"qwen": Counter(), "kev": Counter()}
    errors = Counter()
    written = 0
    started = time.time()

    handle = open(args.out, "a" if append else "w", newline="", encoding="utf-8")
    # The two models sit on different servers, so each message's two calls run
    # side by side rather than one after the other.
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        writer = csv.writer(handle)
        if not append:
            writer.writerow(FIELDNAMES)
            handle.flush()

        messages = iter_user_messages(args.parquet, remaining, done)
        for index, msg in enumerate(messages, len(done) + 1):
            qwen_future = pool.submit(
                classify_ollama, msg.prompt, system_prompt=system_prompt,
                host=args.ollama_host, model=args.ollama_model)
            kev_future = pool.submit(
                classify_kev, msg.prompt, host=args.kev_host, model=args.kev_model,
                threshold=args.kev_threshold)
            qwen = qwen_future.result()
            kev, probs = kev_future.result()

            for name, found in (("qwen", qwen), ("kev", kev)):
                if any(s.startswith("ERROR") for s in found):
                    errors[name] += 1
                else:
                    counts[name].update(found or [NO_SMELL])

            # Flush per row so a crash or Ctrl-C keeps what was earned.
            writer.writerow([
                msg.conversation_id,
                msg.user_turn,
                msg.prompt,
                format_label(qwen),
                format_label(kev),
                json.dumps({k: round(v, 4) for k, v in probs.items()}) if probs else "",
            ])
            handle.flush()
            written += 1

            rate = (time.time() - started) / written
            print(f"[{index}/{args.limit}] {msg.conversation_id}#{msg.user_turn} "
                  f"qwen={format_label(qwen)} | kev={format_label(kev)} "
                  f"({rate:.2f}s/msg, eta {rate * (args.limit - index) / 60:.0f} min)",
                  file=sys.stderr)

    except FatalAPIError as exc:
        print(f"\nfatal: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"\ninterrupted - {written} rows written to {args.out}", file=sys.stderr)
        return 130
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        handle.close()

    elapsed = time.time() - started
    print(f"\n{written} rows written to {args.out} in {elapsed:.0f}s "
          f"(errors: qwen {errors['qwen']}, kev {errors['kev']})", file=sys.stderr)
    for name, counter in counts.items():
        print(f"{name}:", file=sys.stderr)
        for smell, count in counter.most_common():
            print(f"  {count:5d}  {smell}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
