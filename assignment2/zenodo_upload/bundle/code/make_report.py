#!/usr/bin/env python3
"""Statistics and figures for the report on a run_model_passes.py run.

    python make_report.py            # stats.json + figures/ in the run dir
    python make_report.py pdf        # also render report.md to report.pdf

Reads merged.csv, the per-model CSVs and prompts.jsonl from --run-dir. The
qwen repeatability figure compares against the earlier qwen + Kev run in
--previous, when that file exists. `pdf` needs a Chromium on the PATH.

Requirements beyond requirements.txt: requirements-report.txt.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from analyze_prompt_smells import SMELLS

SMELL_NAMES = [s.name for s in SMELLS]
NO_SMELL = "No Smell"
LABELS = SMELL_NAMES + [NO_SMELL]

MODELS = ["granite", "qwen", "tev1", "kev"]
GENERATIVE = ["granite", "qwen"]
DECISION = ["tev1", "kev"]
DISPLAY = {"granite": "granite4:350m", "qwen": "qwen3.5:0.8b", "tev1": "tev1:0.8b", "kev": "Kev-0.8B"}

# Validated categorical slots 1-4 (scripts/validate_palette.js, light surface):
# a model keeps its colour in every figure.
COLOR = {"granite": "#2a78d6", "qwen": "#eb6834", "tev1": "#1baf7a", "kev": "#eda100"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def smell_set(value: str) -> frozenset:
    if value in (NO_SMELL, "None", ""):
        return frozenset()
    return frozenset(value.split("; "))


def load(run_dir: str):
    # keep_default_na=False: pandas otherwise reads label cells such as "None"
    # as missing values.
    merged = pd.read_csv(os.path.join(run_dir, "merged.csv"), keep_default_na=False)
    prompts = pd.read_json(os.path.join(run_dir, "prompts.jsonl"), lines=True, dtype={"conversation_id": str})
    per_model = {k: pd.read_csv(os.path.join(run_dir, f"{k}.csv"), keep_default_na=False) for k in MODELS}
    sets = {k: merged[f"{k}_smells"].map(smell_set) for k in GENERATIVE}
    return merged, prompts, per_model, sets


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def cohen_kappa(a, b) -> tuple[float, float]:
    """(observed agreement, Cohen's kappa) for two boolean raters."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    observed = (a == b).mean()
    expected = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return observed, (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def jaccard(a: frozenset, b: frozenset) -> float | None:
    union = a | b
    return len(a & b) / len(union) if union else None


def label_counts(merged, sets) -> dict[str, dict[str, int]]:
    """Prompts per label: listed (generative) or top-1 (decision)."""
    counts = {}
    for k in GENERATIVE:
        c = {name: int(sets[k].map(lambda s, n=name: n in s).sum()) for name in SMELL_NAMES}
        c[NO_SMELL] = int((sets[k].map(len) == 0).sum())
        counts[k] = c
    for k in DECISION:
        top1 = merged[f"{k}_top1_smell"]
        counts[k] = {name: int((top1 == name).sum()) for name in LABELS}
    return counts


def compute_stats(merged, prompts, per_model, sets, previous: str | None) -> dict:
    n = len(merged)
    length = prompts.prompt.str.len()
    stats = {"n": n}

    stats["prompts"] = {
        "conversations": int(prompts.conversation_id.nunique()),
        "opening_turns": int((prompts.user_turn == 1).sum()),
        "length_median": int(length.median()),
        "length_p90": int(length.quantile(0.9)),
        "over_4000_chars": int((length > 4000).sum()),
    }

    keyed = prompts.assign(length=length)[["conversation_id", "user_turn", "length"]]
    timing = {}
    for k, df in per_model.items():
        df = df.merge(keyed, on=["conversation_id", "user_turn"])
        timing[k] = {
            "mean_s": round(df.seconds.mean(), 3), "median_s": round(df.seconds.median(), 3),
            "p95_s": round(df.seconds.quantile(0.95), 3), "hours": round(df.seconds.sum() / 3600, 2),
            "shortened": int((df.chars_sent < df.length.clip(upper=4000)).sum()),
            "errors": int(df.iloc[:, 2].astype(str).str.startswith("ERROR").sum()),
        }
    stats["timing"] = timing

    counts = label_counts(merged, sets)
    stats["label_counts"] = counts

    generative = {}
    for k in GENERATIVE:
        sizes = sets[k].map(len)
        generative[k] = {
            "mean_smells": round(sizes.mean(), 2),
            "smelly_pct": round(100 * (sizes > 0).mean(), 1),
            "smells_per_prompt": {int(i): int(v) for i, v in sizes.value_counts().sort_index().items()},
            "seven_plus": int((sizes >= 7).sum()),
            "top_combinations": {
                c: round(100 * v, 1)
                for c, v in merged[f"{k}_smells"].value_counts(normalize=True).head(3).items()
            },
        }
    stats["generative"] = generative

    decision = {}
    for k in DECISION:
        p1 = merged[f"{k}_top1_p"].astype(float)
        p2 = merged[f"{k}_top2_p"].astype(float)
        decision[k] = {
            "top1_p_median": round(p1.median(), 3),
            "top1_p_ge_05_pct": round(100 * (p1 >= 0.5).mean(), 1),
            "gap_median": round((p1 - p2).median(), 3),
            "gap_lt_005_pct": round(100 * ((p1 - p2) < 0.05).mean(), 1),
            "labels_used": int(sum(v > 0 for v in counts[k].values())),
        }
    stats["decision"] = decision

    smelly = {k: sets[k].map(len) > 0 for k in GENERATIVE}
    smelly.update({k: merged[f"{k}_top1_smell"] != NO_SMELL for k in DECISION})
    stats["smelly_pct"] = {k: round(100 * v.mean(), 1) for k, v in smelly.items()}
    agreement = {}
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            observed, kappa = cohen_kappa(smelly[a], smelly[b])
            agreement[f"{a}-{b}"] = {"pct": round(100 * observed, 1), "kappa": round(kappa, 3)}
    stats["agreement_any_smell"] = agreement

    per_smell = {}
    for name in SMELL_NAMES:
        observed, kappa = cohen_kappa(sets["granite"].map(lambda s: name in s), sets["qwen"].map(lambda s: name in s))
        per_smell[name] = {"pct": round(100 * observed, 1), "kappa": round(kappa, 3)}
    stats["granite_qwen_per_smell"] = per_smell
    overlaps = [j for j in map(jaccard, sets["granite"], sets["qwen"]) if j is not None]
    stats["granite_qwen_sets"] = {
        "exact_pct": round(100 * np.mean([a == b for a, b in zip(sets["granite"], sets["qwen"])]), 1),
        "jaccard_mean": round(float(np.mean(overlaps)), 3),
    }

    same = merged.tev1_top1_smell == merged.kev_top1_smell
    both = (merged.tev1_top1_smell != NO_SMELL) & (merged.kev_top1_smell != NO_SMELL)
    stats["tev1_kev_top1"] = {"same_pct": round(100 * same.mean(), 1), "both_name_smell": int(both.sum()),
                              "agree_when_both_pct": round(100 * same[both].mean(), 1)}

    confirmed = {}
    for d in DECISION:
        mask = merged[f"{d}_top1_smell"] != NO_SMELL
        for g in GENERATIVE:
            hits = [t in s for t, s in zip(merged.loc[mask, f"{d}_top1_smell"], sets[g][mask])]
            confirmed[f"{d}_in_{g}"] = round(100 * np.mean(hits), 1)
    stats["decision_smell_in_generative_list"] = confirmed

    buckets = pd.cut(length, [0, 50, 150, 500, 10**9], labels=["<=50", "51-150", "151-500", ">500"])
    by_length = pd.DataFrame({
        "prompts": 1,
        "qwen_vague": sets["qwen"].map(lambda s: "Vague / Missing Context" in s),
        "granite_seven_plus": sets["granite"].map(len) >= 7,
        "tev1_smelly": smelly["tev1"], "kev_smelly": smelly["kev"],
    }).groupby(buckets, observed=True)
    stats["by_length"] = {
        str(b): {"prompts": int(g.prompts.sum()), **{c: round(100 * g[c].mean(), 1) for c in g.columns if c != "prompts"}}
        for b, g in by_length
    }

    if previous and os.path.exists(previous):
        old = pd.read_csv(previous, keep_default_na=False, usecols=["conversation_id", "user_turn", "qwen_smells"])
        joined = merged[["conversation_id", "user_turn", "qwen_smells"]].merge(
            old, on=["conversation_id", "user_turn"], suffixes=("", "_previous"))
        a, b = joined.qwen_smells.map(smell_set), joined.qwen_smells_previous.map(smell_set)
        stats["qwen_repeatability"] = {
            "prompts": len(joined), "exact_pct": round(100 * (a == b).mean(), 1),
            "jaccard_mean": round(float(np.mean([j for j in map(jaccard, a, b) if j is not None])), 3),
        }
    return stats


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def clean_axes(ax, keep=("bottom",)):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def fig_label_counts(stats, path):
    """Small multiples: prompts per label, one panel per model, shared scale."""
    n = stats["n"]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.4), sharex=True, sharey=True)
    y = np.arange(len(LABELS))[::-1]
    for ax, k in zip(axes.flat, MODELS):
        values = [stats["label_counts"][k][name] for name in LABELS]
        ax.barh(y, values, height=0.72, color=COLOR[k], edgecolor=SURFACE, linewidth=1)
        for yi, v in zip(y, values):
            ax.text(v + n * 0.015, yi, f"{v:,}" + (f" ({100 * v / n:.0f}%)" if v >= 0.005 * n else ""),
                    va="center", fontsize=6.8, color=INK)
        kind = "prompts listing the smell" if k in GENERATIVE else "prompts where it is the top-1 answer"
        ax.set_title(DISPLAY[k], fontsize=8.5, loc="left", color=INK, weight="bold", pad=12)
        ax.text(0, 1.015, kind, transform=ax.transAxes, fontsize=7, color=INK2, va="bottom")
        ax.set_xlim(0, n * 1.22)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, labelsize=7.2)
        clean_axes(ax)
    axes[0, 0].set_yticks(y, LABELS)
    for ax in axes[1]:
        ax.set_xlabel(f"prompts (of {n:,})")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.tight_layout(h_pad=1.2, w_pad=0.6)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_smells_per_prompt(stats, path):
    """How many smells the generative models list per prompt (grouped bars)."""
    sizes = np.arange(0, 11)
    fig, ax = plt.subplots(figsize=(6.8, 2.2))
    width = 0.4
    for offset, k in zip((-width / 2, width / 2), GENERATIVE):
        dist = stats["generative"][k]["smells_per_prompt"]
        values = [dist.get(int(s), dist.get(str(s), 0)) for s in sizes]
        ax.bar(sizes + offset, values, width=width, color=COLOR[k], edgecolor=SURFACE, linewidth=1,
               label=DISPLAY[k])
        for s, v in zip(sizes, values):
            if v >= 100:
                ax.text(s + offset, v + 60, f"{v:,}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(sizes)
    ax.set_xlabel("smells listed for the prompt")
    ax.set_ylabel("prompts")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_confidence(merged, path):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.1), sharey=True)
    bins = np.linspace(0, 1, 51)
    for ax, k in zip(axes, DECISION):
        p = merged[f"{k}_top1_p"].astype(float)
        ax.hist(p, bins=bins, color=COLOR[k], edgecolor=SURFACE, linewidth=0.6)
        ax.axvline(1 / 11, color=INK2, lw=1, ls=(0, (3, 2)))
        ax.axvline(p.median(), color=INK, lw=1)
        ax.text(p.median(), 1.02, f"median {p.median():.2f}", transform=ax.get_xaxis_transform(),
                fontsize=7.5, ha="center", va="bottom")
        ax.text(0.98, 0.9, DISPLAY[k], transform=ax.transAxes, fontsize=9, ha="right", va="top", weight="bold")
        ax.set_xlim(0, 1)
        ax.set_xlabel("top-1 probability")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        clean_axes(ax)
    axes[0].set_ylabel("prompts")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_agreement(stats, path):
    names = ["granite4\n350m", "qwen3.5\n0.8b", "tev1\n0.8b", "Kev\n0.8B"]
    fig, ax = plt.subplots(figsize=(3.9, 2.9))
    grid = np.full((3, 3), np.nan)
    for i in range(1, 4):
        for j in range(i):
            grid[i - 1, j] = stats["agreement_any_smell"][f"{MODELS[j]}-{MODELS[i]}"]["pct"]
    ax.imshow(np.ma.masked_invalid(grid), cmap=SEQUENTIAL, vmin=0, vmax=100)
    for i in range(1, 4):
        for j in range(i):
            e = stats["agreement_any_smell"][f"{MODELS[j]}-{MODELS[i]}"]
            kappa = f"{e['kappa'] + 0.0:.2f}".replace("-0.00", "0.00")
            ax.text(j, i - 1, f"{e['pct']:.0f}% agree\nκ = {kappa}", ha="center", va="center", fontsize=8,
                    color="white" if e["pct"] > 55 else INK)
    ax.set_xticks(range(3), names[:3])
    ax.set_yticks(range(3), names[1:])
    ax.tick_params(length=0)
    clean_axes(ax, keep=())
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

PDF_CSS = """
@page { size: A4; margin: 15mm 16mm 14mm; }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 9.2pt; line-height: 1.34; color: #111; }
h1 { font-size: 14.5pt; margin: 0 0 4px; }
h2 { font-size: 11pt; margin: 9px 0 3px; border-bottom: 1px solid #ddd; clear: both; break-after: avoid; }
p, li { margin: 3px 0; }
ul, ol { margin: 3px 0; padding-left: 18px; }
table { border-collapse: collapse; font-size: 8pt; margin: 4px 0; }
td, th { border: 1px solid #ccc; padding: 2px 5px; vertical-align: top; }
img { max-width: 100%; }
img[align=right] { margin: 0 0 4px 10px; }
code { font-size: 8pt; }
em { color: #333; }
"""


def render_pdf(run_dir: str) -> str:
    import markdown

    source = os.path.join(run_dir, "report.md")
    html_path = os.path.join(run_dir, "_report.html")
    pdf_path = os.path.join(run_dir, "report.pdf")
    with open(source, encoding="utf-8") as handle:
        body = markdown.markdown(handle.read(), extensions=["tables"])
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(f"<html><head><meta charset='utf-8'><style>{PDF_CSS}</style></head><body>{body}</body></html>")
    browser = shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome")
    if browser is None:
        raise SystemExit("no Chromium found; open the HTML and print it to PDF instead")
    try:
        subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={os.path.abspath(pdf_path)}", f"file://{os.path.abspath(html_path)}"],
                       check=True, timeout=120, capture_output=True)
    finally:
        os.remove(html_path)
    return pdf_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="figures", choices=["figures", "pdf"])
    parser.add_argument("--run-dir", default="runs/smells_6000")
    parser.add_argument("--previous", default="prompt_smells_qwen_kev.csv",
                        help="earlier run with a qwen_smells column, for the repeatability check")
    args = parser.parse_args()

    merged, prompts, per_model, sets = load(args.run_dir)
    stats = compute_stats(merged, prompts, per_model, sets, args.previous)
    with open(os.path.join(args.run_dir, "stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=1)

    figures = os.path.join(args.run_dir, "figures")
    os.makedirs(figures, exist_ok=True)
    fig_label_counts(stats, os.path.join(figures, "fig1_label_counts.png"))
    fig_smells_per_prompt(stats, os.path.join(figures, "fig2_smells_per_prompt.png"))
    fig_confidence(merged, os.path.join(figures, "fig3_confidence.png"))
    fig_agreement(stats, os.path.join(figures, "fig4_agreement.png"))
    print(f"wrote {args.run_dir}/stats.json and 4 figures in {figures}", file=sys.stderr)

    if args.command == "pdf":
        print(f"wrote {render_pdf(args.run_dir)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
