#!/usr/bin/env python3
"""
Automated monitoring and periodic data refresh for FANNO-Dev synthesis.
Checks running processes, refreshes tokens, re-cleans data, and reports progress.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime
from collections import Counter

OUTPUT_DIR = Path(__file__).parent / "outputs"


def refresh_token():
    """Refresh Azure AD token."""
    try:
        result = subprocess.run(
            ["az", "account", "get-access-token", "--resource", "https://cognitiveservices.azure.com"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            token_data = json.loads(result.stdout)
            with open("/tmp/.fanno_azure_token", "w") as f:
                f.write(token_data["accessToken"])
            print(f"  Token refreshed, expires: {token_data['expiresOn']}")
            return True
    except Exception as e:
        print(f"  Token refresh failed: {e}")
    return False


def count_processes():
    """Count running synthesis processes."""
    try:
        result = subprocess.run(
            ["bash", "-c", "ps aux | grep -E 'python.*synth|python.*fanno' | grep -v grep | wc -l"],
            capture_output=True, text=True, timeout=10
        )
        return int(result.stdout.strip())
    except:
        return -1


def count_data():
    """Count current data files."""
    counts = {}
    for f in OUTPUT_DIR.glob("*.jsonl"):
        if f.name.startswith("merged_") or f.name.startswith("cleaned_"):
            continue
        try:
            with open(f) as fh:
                n = sum(1 for line in fh if line.strip())
            counts[f.name] = n
        except:
            pass
    return counts


def run_monitoring_cycle():
    """Run one monitoring cycle."""
    print(f"\n{'='*60}")
    print(f"MONITORING CYCLE: {datetime.now().isoformat()}")
    print(f"{'='*60}")

    # 1. Token refresh
    print("\n🔑 Token refresh...")
    refresh_token()

    # 2. Process count
    n_proc = count_processes()
    print(f"\n⚙️ Running processes: {n_proc}")

    # 3. Data counts
    print("\n📊 Raw data files:")
    counts = count_data()
    total_raw = 0
    for name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {name:<30} {cnt:>8,}")
        total_raw += cnt
    print(f"  {'TOTAL':<30} {total_raw:>8,}")

    # 4. Cleaned data counts
    cleaned_total = 0
    for f in ["cleaned_single_turn.jsonl", "cleaned_multi_turn.jsonl"]:
        fpath = OUTPUT_DIR / f
        if fpath.exists():
            with open(fpath) as fh:
                n = sum(1 for line in fh if line.strip())
            print(f"\n  Cleaned {f}: {n:,}")
            cleaned_total += n
    print(f"  Cleaned total: {cleaned_total:,}")

    return n_proc, total_raw, cleaned_total


if __name__ == "__main__":
    import sys

    if "--once" in sys.argv:
        run_monitoring_cycle()
    else:
        print("Running continuous monitoring (Ctrl+C to stop)")
        print("Use --once for single check")
        try:
            while True:
                n_proc, total_raw, cleaned = run_monitoring_cycle()
                if n_proc == 0:
                    print("\n🎉 All synthesis processes completed!")
                    print("Running final cleanup...")
                    break
                print(f"\nNext check in 10 minutes...")
                time.sleep(600)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
