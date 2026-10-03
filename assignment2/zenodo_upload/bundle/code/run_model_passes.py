#!/usr/bin/env python3
"""Label a fixed set of WildChat prompts with prompt smells, one model at a time.

Four models, run model-major: each one gets the whole GPU and labels every
prompt before the next is loaded.

  * granite4:350m, qwen3.5:0.8b - generative, through Ollama's /api/chat with the
    same JSON-constrained system prompt as analyze_prompt_smells.py
  * tev1:0.8b (Ollama) and Kev-0.8B (its own server) - decision models, asked one
    multiple-choice question over the ten smells plus "None"; only the top four
    probabilities are kept

Everything lives in --run-dir:

  prompts.jsonl     the fixed prompt set, chosen once and shared by every model
  <model>.csv       one row per prompt, fsynced as it is written - the checkpoint
  progress.json     heartbeat, per-model counts, rate and ETA (for monitoring)
  merged.csv        every model side by side, written once all passes are done
  kev_server.log    output of the Kev server this script starts

Interrupted for any reason - Ctrl-C, a crash, a power cut - the same command
picks up where it stopped:

    python run_model_passes.py run --limit 6000
    python run_model_passes.py status
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable
from typing import NamedTuple

import requests

from analyze_prompt_smells import (
    DEFAULT_OLLAMA_HOST,
    ERROR_REQUEST,
    ERROR_UNPARSED,
    MAX_PROMPT_CHARS,
    NO_SMELL,
    NUM_CTX,
    RESPONSE_SCHEMA,
    SMELLS,
    FatalAPIError,
    build_system_prompt,
    iter_user_messages,
    normalize_host,
    parse_reply,
)

# ---------------------------------------------------------------------------
# The models
# ---------------------------------------------------------------------------

GENERATIVE = "generative"
DECISION = "decision"


class ModelSpec(NamedTuple):
    key: str      # file name and merged-column prefix
    kind: str     # GENERATIVE or DECISION
    model: str    # name the server knows it by
    server: str   # "ollama" or "kev"


# Run order. Kev goes last: it is the only one that needs a second server, and
# Ollama's models must be out of GPU memory before it loads.
MODELS: list[ModelSpec] = [
    ModelSpec("granite", GENERATIVE, "granite4:350m", "ollama"),
    ModelSpec("qwen", GENERATIVE, "qwen3.5:0.8b", "ollama"),
    ModelSpec("tev1", DECISION, "tev1:0.8b", "ollama"),
    ModelSpec("kev", DECISION, "kev-latest", "kev"),
]
MODELS_BY_KEY = {m.key: m for m in MODELS}

DEFAULT_KEV_HOST = "http://127.0.0.1:8009"
# A checkout of https://github.com/jaredpalmer/kev and a Python with its
# dependencies; override with KEV_DIR / KEV_PYTHON or --kev-dir / --kev-python.
DEFAULT_KEV_DIR = os.environ.get(
    "KEV_DIR", "kev")
DEFAULT_KEV_PYTHON = os.environ.get(
    "KEV_PYTHON", os.path.join(os.path.dirname(DEFAULT_KEV_DIR), ".venv", "bin", "python"))
KEV_RUN = "jaredpalmer/kev-0.8b"

# The GTX 1660 has no bf16, Kev's fused kernels need `fla` (not installed), and
# CUDA-graph capture runs out of the 6 GB card. fp32 matches the earlier run.
KEV_ENV = {"KEV_DTYPE": "fp32", "KEV_FUSED": "0", "KEV_CUDA_GRAPHS": "0", "HF_HUB_OFFLINE": "1"}

KEV_START_TIMEOUT = 300

# ---------------------------------------------------------------------------
# The decision question
#
# One multiple-choice question rather than ten yes/no ones: the probabilities sum
# to one, so the top four are a ranking, not ten independent and badly separated
# scores. "None" is an option so a clean prompt is not forced onto a smell.
# ---------------------------------------------------------------------------

TOP_K = 4

DECISION_QUESTIONS = {
    "smell": {
        "type": "choice",
        "instructions": (
            "Which prompt smell does this prompt show most strongly? Judge only the "
            "wording and structure of the prompt, never its topic. Choose \"None\" if "
            "the prompt is clear, specific and well-scoped."
        ),
        "criteria": {
            **{s.name: s.cue for s in SMELLS},
            NO_SMELL: "The prompt is clear, specific and well-scoped; none of the smells apply.",
        },
    }
}

# What the CSVs say for "no smell". The models are still asked with "None", but
# pandas (and R) read a bare "None" cell as missing, which blanked every clean
# prompt in the smell columns.
NO_SMELL_LABEL = "No Smell"

# A decision server that rejects the prompt (too many tokens, out of GPU memory)
# gets it again at half the length, down to this.
MIN_PROMPT_CHARS = 125

# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

# (connect, read). A cold model can take a minute to load on the first call.
TIMEOUT = (10, 300)
BACKOFF_SECONDS = [2, 5, 10]

# Small generative models can loop inside the "evidence" string; this caps a
# runaway reply, which then fails to parse and is recorded as an error.
NUM_PREDICT = 400

# granite4:350m otherwise loops - repeating a phrase in "evidence" or smell
# names in the array - until NUM_PREDICT cuts it off mid-JSON. Ollama enforces
# both limits during generation, so the object always closes; parse_reply still
# dedupes the array.
GENERATIVE_SCHEMA = {
    **RESPONSE_SCHEMA,
    "properties": {
        "evidence": {**RESPONSE_SCHEMA["properties"]["evidence"], "maxLength": 300},
        "smells": {**RESPONSE_SCHEMA["properties"]["smells"], "maxItems": len(SMELLS)},
    },
}

# How long a pass waits for a server that has gone away before giving up and
# exiting non-zero (the supervisor then restarts the whole thing).
SERVER_WAIT_SECONDS = 600


class ServerUnavailable(RuntimeError):
    """The server cannot be reached at all; no row should be written."""


class Rejected(RuntimeError):
    """The server answered, but with an error for this particular request."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"HTTP {status}: {detail[:200]}")
        self.status = status


def post_json(url: str, body: dict) -> dict:
    """POST once. Connection trouble is ServerUnavailable, an error reply is Rejected."""
    try:
        response = requests.post(url, json=body, timeout=TIMEOUT)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise ServerUnavailable(f"{url}: {exc}") from exc
    if response.status_code == 404:
        raise FatalAPIError(f"404 from {url}: {response.text[:200]}")
    if response.status_code >= 400:
        raise Rejected(response.status_code, response.text)
    try:
        return response.json()
    except ValueError as exc:
        raise Rejected(response.status_code, f"invalid JSON: {exc}") from exc


def post_with_backoff(url: str, body: dict) -> dict:
    """POST, retrying transient failures. The last failure is re-raised."""
    for attempt, delay in enumerate([*BACKOFF_SECONDS, None]):
        try:
            return post_json(url, body)
        except (ServerUnavailable, Rejected) as exc:
            # A 4xx other than a timeout is about the request; resending it as-is
            # cannot help.
            if isinstance(exc, Rejected) and 400 <= exc.status < 500 and exc.status != 408:
                raise
            if delay is None:
                raise
            log(f"  {exc} - retry {attempt + 1} in {delay}s")
            time.sleep(delay)
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# Classifiers. Each returns the row's label columns plus how many characters
# of the prompt the model saw; errors go into the label, as in the earlier run.
# ---------------------------------------------------------------------------


class Ollama:
    def __init__(self, host: str):
        self.host = host
        self._thinking: dict[str, bool] = {}

    def check(self, model: str) -> None:
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ServerUnavailable(f"Ollama at {self.host}: {exc}") from exc
        available = [m["name"] for m in response.json().get("models", [])]
        if model not in available:
            raise FatalAPIError(f"Ollama has no {model!r}; run: ollama pull {model}")

    def supports_thinking(self, model: str) -> bool:
        # Sending think=false to a model without the capability is an error, and
        # leaving thinking on multiplies the latency of the ones that have it.
        if model not in self._thinking:
            try:
                info = post_json(f"{self.host}/api/show", {"model": model})
            except Rejected:
                info = {}
            self._thinking[model] = "thinking" in info.get("capabilities", [])
        return self._thinking[model]

    def unload_all(self) -> None:
        """Free the GPU: ask Ollama to drop every model it has loaded."""
        try:
            loaded = requests.get(f"{self.host}/api/ps", timeout=10).json().get("models", [])
            for m in loaded:
                requests.post(f"{self.host}/api/generate",
                              json={"model": m["name"], "keep_alive": 0}, timeout=60)
        except requests.RequestException as exc:
            log(f"  could not unload Ollama models: {exc}")

    def classify(self, model: str, prompt: str, system_prompt: str) -> tuple[list[str], int]:
        text = prompt[:MAX_PROMPT_CHARS]
        body = {
            "model": model,
            "stream": False,
            "format": GENERATIVE_SCHEMA,
            "options": {"temperature": 0, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        }
        if self.supports_thinking(model):
            body["think"] = False
        try:
            reply = post_with_backoff(f"{self.host}/api/chat", body)
        except Rejected as exc:
            log(f"  rejected: {exc}")
            return [ERROR_REQUEST], len(text)
        try:
            content = reply["message"]["content"]
        except (KeyError, TypeError):
            return [ERROR_UNPARSED], len(text)
        return parse_reply(content), len(text)


def classify_decision(host: str, model: str, prompt: str) -> tuple[list[tuple[str, float]] | str, int]:
    """Ask the multiple-choice question. Returns (top-k pairs or an error label, chars sent)."""
    limit = min(len(prompt), MAX_PROMPT_CHARS)
    while True:
        body = {"model": model, "state": {"prompt": prompt[:limit]}, "questions": DECISION_QUESTIONS}
        # Not post_with_backoff: a rejected prompt is halved at once rather than
        # resent unchanged. An unreachable server propagates to the caller.
        try:
            reply = post_json(f"{host}/v1/systemone", body)
            break
        except Rejected as exc:
            # Too long for the context or for GPU memory: Qwen tokenizes digits one
            # by one, so a short wall of numbers can be thousands of tokens.
            if limit // 2 < MIN_PROMPT_CHARS:
                log(f"  {exc} - giving up at {limit} chars")
                return ERROR_REQUEST, limit
            limit //= 2
            log(f"  {exc} - retrying with the first {limit} chars")

    try:
        probabilities = reply["answers"]["smell"]["probabilities"]
        ranked = sorted(((k, float(v)) for k, v in probabilities.items()),
                        key=lambda kv: kv[1], reverse=True)
    except (KeyError, TypeError, ValueError, AttributeError):
        return ERROR_UNPARSED, limit
    return ranked[:TOP_K], limit


# ---------------------------------------------------------------------------
# The Kev server
# ---------------------------------------------------------------------------


class KevServer:
    """Starts, finds and stops the Kev server. Survives this script restarting:
    the server runs in its own session and its pid is kept in the run dir."""

    def __init__(self, host: str, kev_dir: str, python: str, run_dir: str):
        self.host = host
        self.kev_dir = kev_dir
        self.python = python
        self.pid_file = os.path.join(run_dir, "kev_server.pid")
        self.log_file = os.path.join(run_dir, "kev_server.log")
        self.port = host.rsplit(":", 1)[-1]
        self.process: subprocess.Popen | None = None

    def healthy(self, model: str) -> bool:
        try:
            response = requests.get(f"{self.host}/v1/models", timeout=5)
            response.raise_for_status()
            return any(m.get("name") == model for m in response.json().get("models", []))
        except (requests.RequestException, ValueError):
            return False

    def _pid(self) -> int | None:
        """The recorded server pid, if that process is still a Kev server. After a
        reboot the number may belong to something else entirely."""
        try:
            with open(self.pid_file) as handle:
                pid = int(handle.read().strip())
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                if b"kev.serve" not in handle.read():
                    return None
            return pid
        except (OSError, ValueError):
            return None

    def ensure(self, model: str, ollama: Ollama) -> None:
        if self.healthy(model):
            return
        self.stop()  # a pid that is alive but not answering is hung
        ollama.unload_all()
        log(f"starting Kev server ({KEV_RUN}) on :{self.port}")
        with open(self.log_file, "ab") as out:
            self.process = process = subprocess.Popen(
                [self.python, "-m", "kev.serve", "--run", KEV_RUN, "--port", self.port],
                cwd=self.kev_dir, env={**os.environ, **KEV_ENV},
                stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
            )
        write_atomic(self.pid_file, str(process.pid))
        deadline = time.time() + KEV_START_TIMEOUT
        while time.time() < deadline:
            if process.poll() is not None:
                raise FatalAPIError(f"Kev server exited with {process.returncode}; see {self.log_file}")
            if self.healthy(model):
                log("Kev server is up")
                return
            time.sleep(3)
        self.stop()
        raise ServerUnavailable(f"Kev server not ready after {KEV_START_TIMEOUT}s; see {self.log_file}")

    def stop(self) -> None:
        pid = self._pid()
        if pid is None:
            return
        log(f"stopping Kev server (pid {pid})")
        try:
            os.killpg(pid, signal.SIGTERM)
            for _ in range(30):
                time.sleep(1)
                # A child of this process lingers as a zombie until reaped.
                if self.process is not None and self.process.pid == pid:
                    self.process.poll()
                if self._pid() is None:
                    break
            else:
                os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass
        self.process = None
        try:
            os.remove(self.pid_file)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Files: the prompt set, per-model checkpoints, progress
# ---------------------------------------------------------------------------


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", file=sys.stderr, flush=True)


def write_atomic(path: str, text: str) -> None:
    """Write via a temp file and rename, so a crash never leaves half a file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


class Prompt(NamedTuple):
    conversation_id: str
    user_turn: int
    prompt: str

    @property
    def key(self) -> tuple[str, int]:
        return self.conversation_id, self.user_turn


def load_prompts(path: str) -> list[Prompt]:
    with open(path, encoding="utf-8") as handle:
        return [Prompt(**json.loads(line)) for line in handle if line.strip()]


def prepare_prompts(run_dir: str, parquet: str, limit: int) -> list[Prompt]:
    """Pick the prompt set once. The selection is deterministic (file order), so
    a larger --limit later only appends: earlier rows keep their meaning."""
    path = os.path.join(run_dir, "prompts.jsonl")
    if os.path.exists(path):
        prompts = load_prompts(path)
        if len(prompts) == limit:
            return prompts
        log(f"prompts.jsonl has {len(prompts)} prompts, want {limit}; re-selecting")
    log(f"selecting {limit} English user messages from {parquet}")
    prompts = [Prompt(*m) for m in iter_user_messages(parquet, limit)]
    if len(prompts) < limit:
        raise FatalAPIError(f"{parquet} only has {len(prompts)} English user messages")
    write_atomic(path, "".join(json.dumps(p._asdict(), ensure_ascii=False) + "\n" for p in prompts))
    return prompts


def fieldnames(spec: ModelSpec) -> list[str]:
    if spec.kind == GENERATIVE:
        labels = ["smells"]
    else:
        labels = [f"top{i}_{part}" for i in range(1, TOP_K + 1) for part in ("smell", "p")]
    return ["conversation_id", "user_turn", *labels, "chars_sent", "seconds"]


def is_error_row(spec: ModelSpec, row: dict) -> bool:
    label = row["smells"] if spec.kind == GENERATIVE else row["top1_smell"]
    return label.startswith("ERROR")


def read_checkpoint(path: str, spec: ModelSpec, wanted: set[tuple[str, int]]) -> dict[tuple[str, int], dict]:
    """Load a model's CSV, dropping anything a crash may have left broken.

    A power cut can leave a half-written last line. Rows that do not parse, have
    the wrong shape, or are not in the prompt set are discarded, and the file is
    rewritten without them so appending is safe again.
    """
    if not os.path.exists(path):
        return {}
    names = fieldnames(spec)
    rows: dict[tuple[str, int], dict] = {}
    dropped = 0
    with open(path, encoding="utf-8", newline="") as handle:
        raw = handle.read()
    reader = csv.reader(raw.splitlines(keepends=True))
    header = next(reader, None)
    for record in reader:
        try:
            if len(record) != len(names):
                raise ValueError
            row = dict(zip(names, record))
            key = (row["conversation_id"], int(row["user_turn"]))
            float(row["seconds"])
            if key not in wanted:
                raise ValueError
            rows[key] = row
        except ValueError:
            dropped += 1
    if header != names or dropped or (raw and not raw.endswith("\n")):
        log(f"{spec.key}: repairing checkpoint ({dropped} unusable rows dropped)")
        rewrite_checkpoint(path, spec, rows.values())
    return rows


def rewrite_checkpoint(path: str, spec: ModelSpec, rows: Iterable[dict]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames(spec))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


class Progress:
    """progress.json: what a monitor reads. Rewritten atomically after every row."""

    def __init__(self, run_dir: str):
        self.path = os.path.join(run_dir, "progress.json")
        try:
            with open(self.path, encoding="utf-8") as handle:
                self.data = json.load(handle)
        except (OSError, ValueError):
            self.data = {}
        self.data.setdefault("models", {})

    def update(self, key: str, **fields) -> None:
        self.data["models"].setdefault(key, {}).update(fields)
        self.data["current"] = key
        self.data["heartbeat"] = time.time()
        self.data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_atomic(self.path, json.dumps(self.data, indent=2))


# ---------------------------------------------------------------------------
# One pass
# ---------------------------------------------------------------------------


def label_columns(spec: ModelSpec, result) -> dict:
    if spec.kind == GENERATIVE:
        found = result
        if any(s.startswith("ERROR") for s in found):
            return {"smells": found[0]}
        return {"smells": "; ".join(found) if found else NO_SMELL_LABEL}
    columns = {f"top{i}_{part}": "" for i in range(1, TOP_K + 1) for part in ("smell", "p")}
    if isinstance(result, str):
        columns["top1_smell"] = result
        return columns
    for i, (name, p) in enumerate(result, 1):
        columns[f"top{i}_smell"] = NO_SMELL_LABEL if name == NO_SMELL else name
        columns[f"top{i}_p"] = f"{p:.4f}"
    return columns


class Pass:
    def __init__(self, spec: ModelSpec, args, ollama: Ollama, kev: KevServer, system_prompt: str):
        self.spec = spec
        self.args = args
        self.ollama = ollama
        self.kev = kev
        self.system_prompt = system_prompt

    def ensure_server(self) -> None:
        if self.spec.server == "kev":
            self.kev.ensure(self.spec.model, self.ollama)
        else:
            self.kev.stop()  # Kev holds 3.4 GB of a 6 GB card
            self.ollama.check(self.spec.model)

    def wait_for_server(self) -> None:
        """Make sure the server is up, bringing it back if it went away, or give up."""
        deadline = time.time() + SERVER_WAIT_SECONDS
        waited = False
        while True:
            try:
                self.ensure_server()
                if waited:
                    log("server is back")
                return
            except ServerUnavailable as exc:
                waited = True
                if time.time() > deadline:
                    raise
                log(f"  waiting for server: {exc}")
                time.sleep(15)

    def classify(self, prompt: str) -> tuple[dict, int]:
        if self.spec.kind == GENERATIVE:
            result, sent = self.ollama.classify(self.spec.model, prompt, self.system_prompt)
        else:
            host = self.args.kev_host if self.spec.server == "kev" else self.args.ollama_host
            result, sent = classify_decision(host, self.spec.model, prompt)
        return label_columns(self.spec, result), sent

    def classify_row(self, p: Prompt) -> dict:
        """Label one prompt, riding out a server that drops away mid-run."""
        while True:
            started = time.time()
            try:
                labels, sent = self.classify(p.prompt)
                break
            except ServerUnavailable as exc:
                log(f"  server unavailable: {exc}")
                self.wait_for_server()
        return {"conversation_id": p.conversation_id, "user_turn": p.user_turn, **labels,
                "chars_sent": sent, "seconds": f"{time.time() - started:.3f}"}

    def run(self, prompts: list[Prompt], progress: Progress) -> None:
        spec = self.spec
        path = os.path.join(self.args.run_dir, f"{spec.key}.csv")
        rows = read_checkpoint(path, spec, {p.key for p in prompts})
        todo = [p for p in prompts if p.key not in rows]
        errors = sum(is_error_row(spec, r) for r in rows.values())
        total = len(prompts)

        if not todo and progress.data["models"].get(spec.key, {}).get("status") == "done":
            log(f"{spec.key}: already complete ({total} rows)")
            return

        log(f"{spec.key} ({spec.model}): {len(rows)}/{total} done, {len(todo)} to go")
        self.wait_for_server()

        new_file = not os.path.exists(path)
        handle = open(path, "a", encoding="utf-8", newline="")
        try:
            writer = csv.DictWriter(handle, fieldnames=fieldnames(spec))
            if new_file:
                writer.writeheader()
            pass_started = time.time()
            for n, p in enumerate(todo, 1):
                row = self.classify_row(p)
                writer.writerow(row)
                # fsync, not just flush: a row survives a power cut once written.
                handle.flush()
                os.fsync(handle.fileno())
                rows[p.key] = row
                errors += is_error_row(spec, row)

                rate = (time.time() - pass_started) / n
                done = len(rows)
                progress.update(spec.key, status="running", model=spec.model, done=done,
                                total=total, errors=errors, s_per_prompt=round(rate, 3),
                                eta_min=round(rate * (total - done) / 60, 1))
                if n % self.args.log_every == 0 or n == len(todo):
                    label = row.get("smells") or f"{row['top1_smell']} {row['top1_p']}"
                    log(f"[{spec.key} {done}/{total}] {p.conversation_id}#{p.user_turn} -> "
                        f"{label} ({rate:.2f}s/prompt, eta {rate * (total - done) / 60:.0f} min, "
                        f"{errors} errors)")
        finally:
            handle.close()

        self.retry_errors(path, prompts, rows, progress)
        errors = sum(is_error_row(spec, r) for r in rows.values())
        progress.update(spec.key, status="done", done=len(rows), errors=errors, eta_min=0)
        log(f"{spec.key}: done, {len(rows)} rows, {errors} errors")

    def retry_errors(self, path: str, prompts: list[Prompt], rows: dict, progress: Progress) -> None:
        """One more try for the rows that failed; whatever still fails stays an error."""
        failed = [p for p in prompts if is_error_row(self.spec, rows[p.key])]
        if not failed:
            return
        log(f"{self.spec.key}: retrying {len(failed)} error rows")
        fixed = 0
        for p in failed:
            row = self.classify_row(p)
            if not is_error_row(self.spec, row):
                rows[p.key] = row
                fixed += 1
        rewrite_checkpoint(path, self.spec, (rows[p.key] for p in prompts))
        log(f"{self.spec.key}: {fixed}/{len(failed)} error rows fixed on retry")


# ---------------------------------------------------------------------------
# Merge and status
# ---------------------------------------------------------------------------


def merge(run_dir: str, prompts: list[Prompt], specs: list[ModelSpec]) -> str:
    keys = {p.key for p in prompts}
    per_model = {}
    for spec in specs:
        path = os.path.join(run_dir, f"{spec.key}.csv")
        per_model[spec.key] = read_checkpoint(path, spec, keys)

    columns = ["conversation_id", "user_turn", "user_prompt"]
    for spec in specs:
        label_names = fieldnames(spec)[2:-2]
        columns += [f"{spec.key}_{name}" for name in label_names]

    out = os.path.join(run_dir, "merged.csv")
    tmp = f"{out}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        # Quote every field: the smell lists are joined with "; ", and a
        # spreadsheet importing with ";" as a separator splits any unquoted
        # cell, shifting every column after it.
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
        writer.writerow(columns)
        for p in prompts:
            record = [p.conversation_id, p.user_turn, p.prompt]
            for spec in specs:
                row = per_model[spec.key].get(p.key, {})
                record += [row.get(name, "") for name in fieldnames(spec)[2:-2]]
            writer.writerow(record)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, out)
    return out


def summarize(run_dir: str, prompts: list[Prompt], specs: list[ModelSpec]) -> None:
    keys = {p.key for p in prompts}
    for spec in specs:
        rows = read_checkpoint(os.path.join(run_dir, f"{spec.key}.csv"), spec, keys)
        if not rows:
            continue
        seconds = sum(float(r["seconds"]) for r in rows.values())
        counts: Counter[str] = Counter()
        for r in rows.values():
            if spec.kind == GENERATIVE:
                counts.update(r["smells"].split("; ") if not r["smells"].startswith("ERROR")
                              else [r["smells"]])
            else:
                counts[r["top1_smell"]] += 1
        what = "smell counts" if spec.kind == GENERATIVE else "top-1 counts"
        log(f"{spec.key} ({spec.model}): {len(rows)} rows, {seconds / 3600:.2f} h model time, "
            f"{seconds / len(rows):.2f}s/prompt; {what}:")
        for name, count in counts.most_common():
            log(f"  {count:6d}  {name}")


def print_status(run_dir: str) -> int:
    try:
        with open(os.path.join(run_dir, "progress.json"), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        print(f"no progress yet in {run_dir}")
        return 1
    age = time.time() - data.get("heartbeat", 0)
    print(f"current: {data.get('current')}   last row: {data.get('updated')} ({age:.0f}s ago)")
    if data.get("finished"):
        print(f"finished: {data['finished']}")
    for spec in MODELS:
        m = data["models"].get(spec.key)
        if not m:
            print(f"  {spec.key:8s} pending")
            continue
        print(f"  {spec.key:8s} {m.get('status', '?'):8s} {m.get('done', 0):5d}/{m.get('total', '?')}  "
              f"errors {m.get('errors', 0):4d}  {m.get('s_per_prompt', 0):.2f}s/prompt  "
              f"eta {m.get('eta_min', 0):.0f} min")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="run", choices=["run", "status", "merge"])
    parser.add_argument("--run-dir", default="runs/smells_6000")
    parser.add_argument("--parquet", default="train-00003-of-00006.parquet")
    parser.add_argument("--limit", type=int, default=6000, help="prompts in the set")
    parser.add_argument("--models", default=",".join(m.key for m in MODELS),
                        help=f"comma-separated subset of: {', '.join(MODELS_BY_KEY)}")
    parser.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST))
    parser.add_argument("--kev-host", default=os.environ.get("KEV_HOST", DEFAULT_KEV_HOST))
    parser.add_argument("--kev-dir", default=DEFAULT_KEV_DIR)
    parser.add_argument("--kev-python", default=DEFAULT_KEV_PYTHON)
    parser.add_argument("--log-every", type=int, default=25, help="log a line every N prompts")
    args = parser.parse_args()
    args.ollama_host = normalize_host(args.ollama_host)
    args.kev_host = normalize_host(args.kev_host)
    try:
        args.specs = [MODELS_BY_KEY[k.strip()] for k in args.models.split(",") if k.strip()]
    except KeyError as exc:
        parser.error(f"unknown model {exc}")
    return args


def on_sigterm(signum, frame):
    """SIGTERM (the supervisor's watchdog, a shutdown) unwinds like Ctrl-C."""
    raise KeyboardInterrupt


def main() -> int:
    args = parse_args()
    if args.command == "status":
        return print_status(args.run_dir)

    os.makedirs(args.run_dir, exist_ok=True)
    signal.signal(signal.SIGTERM, on_sigterm)

    try:
        prompts = prepare_prompts(args.run_dir, args.parquet, args.limit)
        if args.command == "merge":
            log(f"wrote {merge(args.run_dir, prompts, args.specs)}")
            return 0

        ollama = Ollama(args.ollama_host)
        kev = KevServer(args.kev_host, args.kev_dir, args.kev_python, args.run_dir)
        progress = Progress(args.run_dir)
        progress.data.pop("finished", None)
        system_prompt = build_system_prompt()

        for spec in args.specs:
            Pass(spec, args, ollama, kev, system_prompt).run(prompts, progress)
            # Hand the GPU to the next model.
            if spec.server == "kev":
                kev.stop()
            else:
                ollama.unload_all()

        out = merge(args.run_dir, prompts, args.specs)
        progress.data["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        progress.update(args.specs[-1].key)
        log(f"all passes done; wrote {out}")
        summarize(args.run_dir, prompts, args.specs)
        return 0

    except FatalAPIError as exc:
        log(f"fatal: {exc}")
        return 2
    except ServerUnavailable as exc:
        log(f"server did not come back: {exc}")
        return 3
    except KeyboardInterrupt:
        log("interrupted; every written row is kept - rerun the same command to resume")
        return 130


if __name__ == "__main__":
    sys.exit(main())
