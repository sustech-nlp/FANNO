#!/usr/bin/env python3
"""
FANNO-Dev: One-click full evaluation pipeline.
Run this after synthesis is complete to regenerate all results.

Usage:
    python3 synthesis/run_full_evaluation.py
    python3 synthesis/run_full_evaluation.py --skip-vendi  # Skip slow Vendi evaluation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path(__file__).parent / "outputs"


def main():
    parser = argparse.ArgumentParser(description="FANNO-Dev Full Evaluation Pipeline")
    parser.add_argument("--skip-vendi", action="store_true", help="Skip Vendi Score evaluation (slow)")
    parser.add_argument("--skip-selection", action="store_true", help="Skip selection strategy comparison")
    args = parser.parse_args()

    print("=" * 70)
    print("FANNO-Dev FULL EVALUATION PIPELINE")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Step 1: Clean data
    print("\n\n📋 STEP 1: DATA CLEANING")
    print("-" * 50)
    from synthesis.clean_data import run_full_cleanup
    run_full_cleanup(OUTPUT_DIR)

    # Step 2: Merge data
    print("\n\n📦 STEP 2: DATA MERGING")
    print("-" * 50)
    from synthesis.merge_data import merge_all_data
    merge_all_data(OUTPUT_DIR, cleaned_only=True)

    # Step 3: Quality report
    print("\n\n📊 STEP 3: QUALITY REPORT")
    print("-" * 50)
    from synthesis.quality_report import generate_report
    generate_report(OUTPUT_DIR)

    # Step 4: Vendi Score evaluation
    if not args.skip_vendi:
        print("\n\n🎯 STEP 4: VENDI SCORE EVALUATION")
        print("-" * 50)
        from synthesis.evaluate_vendi import evaluate_vendi_diversity
        evaluate_vendi_diversity(OUTPUT_DIR)
    else:
        print("\n\n⏭️ STEP 4: SKIPPED (Vendi Score)")

    # Step 5: Scaling analysis
    print("\n\n📈 STEP 5: SCALING ANALYSIS")
    print("-" * 50)
    from synthesis.scaling_analysis import analyze_scaling
    analyze_scaling()

    # Step 6: Selection strategy comparison
    if not args.skip_selection:
        print("\n\n🏆 STEP 6: SELECTION STRATEGY COMPARISON")
        print("-" * 50)
        from synthesis.compare_strategies import run_selection_comparison
        run_selection_comparison(OUTPUT_DIR, n_pool=10000, select_sizes=[500, 1000, 2000, 5000])
    else:
        print("\n\n⏭️ STEP 6: SKIPPED (Selection Strategies)")

    # Step 7: Generate figures
    print("\n\n🎨 STEP 7: PAPER FIGURES")
    print("-" * 50)
    from synthesis.generate_paper_figures import generate_all_figure_data
    from synthesis.render_figures import render_all_figures
    generate_all_figure_data()
    render_all_figures()

    # Step 8: Generate LaTeX tables
    print("\n\n📄 STEP 8: LATEX TABLES")
    print("-" * 50)
    from synthesis.generate_latex_tables import generate_all_tables
    generate_all_tables()

    # Step 9: DataFlow comparison
    print("\n\n🔬 STEP 9: DATAFLOW COMPARISON")
    print("-" * 50)
    from synthesis.compare_dataflow import generate_comparison_analysis
    generate_comparison_analysis()

    print("\n\n" + "=" * 70)
    print(f"EVALUATION COMPLETE: {datetime.now().isoformat()}")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  Data:    {OUTPUT_DIR}/merged_{{alpaca,sharegpt}}.jsonl")
    print(f"  Figures: synthesis/figures/fig{{1-8}}*.png/pdf")
    print(f"  Tables:  synthesis/tables/tab{{1-4}}*.tex")
    print(f"  Reports: {OUTPUT_DIR}/{{quality,vendi_diversity,scaling_analysis}}_report.json")


if __name__ == "__main__":
    main()
