#!/usr/bin/env python3
"""Monitor synthesis progress in real-time."""
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "outputs"
TARGETS = {
    "complex_qa.jsonl": 25000,
    "code_qa.jsonl": 15000,
    "math_qa.jsonl": 10000,
    "reasoning_qa.jsonl": 10000,
    "creative_writing.jsonl": 5000,
    "multi_turn.jsonl": 10000,
    "fanno_seed_qa.jsonl": 30000,
    "self_inverted_qa.jsonl": 5000,
    "trajectory_inverted_qa.jsonl": 5000,
    "trajectory_verified_inversion.jsonl": 5000,
}

def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r") as f:
        return sum(1 for line in f if line.strip())

def show_progress():
    total = 0
    total_target = 0
    print(f"\n{'=' * 60}")
    print(f"FANNO-Dev Synthesis Progress")
    print(f"{'=' * 60}")
    print(f"{'File':<40} {'Current':>8} {'Target':>8} {'%':>6}")
    print(f"{'-' * 60}")
    for fname, target in TARGETS.items():
        count = count_lines(OUTPUT_DIR / fname)
        pct = min(100, count / target * 100) if target > 0 else 0
        bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
        total += count
        total_target += target
        if count > 0:
            print(f"  {fname:<38} {count:>8} / {target:>6}  {pct:5.1f}% {bar}")
    total_pct = total / total_target * 100 if total_target > 0 else 0
    print(f"{'-' * 60}")
    print(f"  {'TOTAL':<38} {total:>8} / {total_target:>6}  {total_pct:5.1f}%")
    print(f"{'=' * 60}\n")

if __name__ == "__main__":
    if "--watch" in sys.argv:
        while True:
            show_progress()
            time.sleep(30)
    else:
        show_progress()
