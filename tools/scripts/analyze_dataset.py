#!/usr/bin/env python3
"""
Dataset analyzer for synthetic tool-augmented conversations.

Reports:
- Conversation length distribution
- Tool name frequency
- Average token counts for human questions and GPT responses
- System prompt diversity via average pairwise similarity (hash_embedding)

Extras:
- Optional cleanup: if a conversation ends with a human turn, drop that trailing human so it ends with GPT.
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import cosine_similarity, hash_embedding, read_jsonl  # noqa: E402

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

INPUT_PATH = ROOT / "synthetic_data_1k.jsonl"
OUTPUT_DIR = ROOT / "analysis_plots"


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return read_jsonl(str(path))


def analyze(records: list[dict]) -> dict:
    conv_lengths: list[int] = []
    tool_counter: Counter = Counter()
    human_tokens = []
    gpt_tokens = []
    system_embeddings = []

    for rec in records:
        conversations = rec.get("conversations", [])
        conv_lengths.append(len(conversations))

        # Tools are stored as a JSON string
        try:
            tools = json.loads(rec.get("tools", "[]"))
        except json.JSONDecodeError:
            tools = []
        for tool in tools:
            name = tool.get("name")
            if name:
                tool_counter[name] += 1

        # Token stats for human/gpt turns
        for turn in conversations:
            role = turn.get("from")
            text = str(turn.get("value", ""))
            tok_count = len(text.split())
            if role == "human":
                human_tokens.append(tok_count)
            elif role == "gpt":
                gpt_tokens.append(tok_count)

        # System diversity
        system_prompt = rec.get("system", "")
        if system_prompt:
            system_embeddings.append(hash_embedding(system_prompt))

    diversity = compute_diversity(system_embeddings)

    return {
        "total_records": len(records),
        "conv_lengths": summarize_numeric(conv_lengths),
        "conv_lengths_raw": conv_lengths,
        "tool_frequency": tool_counter.most_common(20),
        "human_tokens": summarize_numeric(human_tokens),
        "human_tokens_raw": human_tokens,
        "gpt_tokens": summarize_numeric(gpt_tokens),
        "gpt_tokens_raw": gpt_tokens,
        "system_diversity": diversity,
    }


def summarize_numeric(values: list[int | float]) -> dict:
    if not values:
        return {"count": 0}
    values_sorted = sorted(values)
    n = len(values)
    return {
        "count": n,
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / n, 3),
        "p50": values_sorted[n // 2],
        "p90": values_sorted[int(n * 0.9)],
    }


def compute_diversity(embeddings: list[list[float]]) -> dict:
    if len(embeddings) < 2:
        return {"avg_similarity": 1.0 if embeddings else 0.0, "pairs": 0}
    total = 0.0
    pairs = 0
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            total += cosine_similarity(embeddings[i], embeddings[j])
            pairs += 1
    return {"avg_similarity": round(total / pairs, 3), "pairs": pairs}


def print_report(stats: dict) -> None:
    print(f"Total records: {stats['total_records']}")
    cl = stats["conv_lengths"]
    print(f"Conversation length - count: {cl.get('count',0)}, min: {cl.get('min')}, max: {cl.get('max')}, "
          f"mean: {cl.get('mean')}, p50: {cl.get('p50')}, p90: {cl.get('p90')}")
    print("Top tools (up to 20):")
    for name, cnt in stats["tool_frequency"]:
        print(f"  {name}: {cnt}")
    ht = stats["human_tokens"]
    gt = stats["gpt_tokens"]
    print(f"Human token counts - mean: {ht.get('mean')}, p50: {ht.get('p50')}, p90: {ht.get('p90')}, min: {ht.get('min')}, max: {ht.get('max')}")
    print(f"GPT token counts   - mean: {gt.get('mean')}, p50: {gt.get('p50')}, p90: {gt.get('p90')}, min: {gt.get('min')}, max: {gt.get('max')}")
    div = stats["system_diversity"]
    print(f"System diversity - avg cosine similarity: {div['avg_similarity']} over {div['pairs']} pairs")


def save_plots(stats: dict, outdir: Optional[Path]) -> None:
    if plt is None or outdir is None:
        return
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    (ax1, ax2), (ax3, ax4) = axes

    conv_lengths = stats.get("conv_lengths_raw", [])
    if conv_lengths:
        ax1.hist(conv_lengths, bins=30, color="skyblue", edgecolor="black")
        ax1.set_title("Conversation length distribution")
        ax1.set_xlabel("Turns")
        ax1.set_ylabel("Frequency")
    else:
        ax1.axis("off")

    human_tokens = stats.get("human_tokens_raw", [])
    if human_tokens:
        ax2.hist(human_tokens, bins=40, color="lightcoral", edgecolor="black")
        ax2.set_title("Human token count distribution")
        ax2.set_xlabel("Tokens")
    else:
        ax2.axis("off")

    gpt_tokens = stats.get("gpt_tokens_raw", [])
    if gpt_tokens:
        ax3.hist(gpt_tokens, bins=40, color="mediumseagreen", edgecolor="black")
        ax3.set_title("GPT token count distribution")
        ax3.set_xlabel("Tokens")
        ax3.set_ylabel("Frequency")
    else:
        ax3.axis("off")

    tools = stats.get("tool_frequency", [])
    if tools:
        labels, counts = zip(*tools)
        ax4.barh(labels, counts, color="goldenrod")
        ax4.set_title("Top tools (count)")
        ax4.set_xlabel("Count")
    else:
        ax4.axis("off")

    plt.tight_layout()
    plt.savefig(outdir / "dataset_overview.png")
    plt.close()


def cleanup_conversations(records: list[dict], trim_trailing_human: bool = True) -> tuple[list[dict], dict]:
    cleaned = []
    trimmed = 0
    dropped = 0
    for rec in records:
        conv = rec.get("conversations", [])
        if not conv:
            dropped += 1
            continue
        if trim_trailing_human and conv[-1].get("from") == "human":
            conv = conv[:-1]
            trimmed += 1
        if not conv:
            dropped += 1
            continue
        rec["conversations"] = conv
        cleaned.append(rec)
    return cleaned, {"trimmed": trimmed, "dropped": dropped, "kept": len(cleaned)}


def main():
    records = load_records(INPUT_PATH)
    records, meta = cleanup_conversations(records, trim_trailing_human=True)
    stats = analyze(records)
    print_report(stats)
    print(f"Cleanup: trimmed={meta['trimmed']}, dropped={meta['dropped']}, kept={meta['kept']}")
    save_plots(stats, OUTPUT_DIR)


if __name__ == "__main__":
    main()
