"""FANNO CLI: Unified command-line interface for all FANNO operations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FANNO: Free ANNOtator pipeline for synthetic instruction generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  fanno pipeline --config configs/azure_gpt5.yaml
  fanno synthesize --type agent --num-samples 5000
  fanno prepare --output-dir ./train_data
  fanno train --model Qwen/Qwen3-8B --data ./train_data/train.jsonl
  fanno evaluate --data ./outputs/final_data.jsonl
        """,
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- pipeline (original FANNO flow) ---
    p_pipe = sub.add_parser("pipeline", help="Run the full FANNO UCB pipeline")
    p_pipe.add_argument("--config", type=str, default=None, help="Path to YAML config")

    # --- synthesize ---
    p_syn = sub.add_parser("synthesize", help="Synthesize training data")
    p_syn.add_argument("--type", choices=["qa", "creative", "dialog", "agent"], default="qa")
    p_syn.add_argument("--num-samples", type=int, default=1000)
    p_syn.add_argument("--model", type=str, default="gpt-4o-mini")
    p_syn.add_argument("--output", type=str, default="./outputs/synthesized.jsonl")
    p_syn.add_argument("--workers", type=int, default=8)

    # --- evaluate ---
    p_eval = sub.add_parser("evaluate", help="Evaluate data quality and diversity")
    p_eval.add_argument("--data", type=str, required=True, help="Path to data JSONL")
    p_eval.add_argument("--source-type", choices=["general", "agent", "code"], default="general")
    p_eval.add_argument("--model", type=str, default="gpt-4o-mini")

    # --- prepare ---
    p_prep = sub.add_parser("prepare", help="Prepare mixed training data")
    p_prep.add_argument("--output-dir", type=str, default="./train_data")
    p_prep.add_argument("--fanno-dir", type=str, default="./outputs")
    p_prep.add_argument("--max-fanno-qa", type=int, default=50000)
    p_prep.add_argument("--max-alpaca", type=int, default=20000)
    p_prep.add_argument("--max-arena", type=int, default=10000)
    p_prep.add_argument("--max-bfcl", type=int, default=15000)
    p_prep.add_argument("--seed", type=int, default=42)

    # --- train ---
    p_train = sub.add_parser("train", help="Run SFT training")
    p_train.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    p_train.add_argument("--data", type=str, required=True)
    p_train.add_argument("--output-dir", type=str, default="./checkpoints")
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--lr", type=float, default=2e-5)
    p_train.add_argument("--per-device-batch", type=int, default=2)
    p_train.add_argument("--gradient-accum", type=int, default=8)
    p_train.add_argument("--deepspeed", action="store_true", default=True)
    p_train.add_argument("--wandb-project", type=str, default="fanno-sft")

    # --- amlt ---
    p_amlt = sub.add_parser("amlt", help="Generate AMLT config for cloud training")
    p_amlt.add_argument("--job-name", type=str, default="fanno-sft-qwen3-8b")
    p_amlt.add_argument("--output", type=str, default="./configs/amlt_sft_qwen3.yaml")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "pipeline":
        from fanno.pipeline import run_pipeline
        records = run_pipeline(args.config)
        logger.info(f"Finished pipeline. Generated {len(records)} instruction/response pairs.")

    elif args.command == "synthesize":
        from fanno.synthesize.base import BaseSynthesizer
        if args.type == "qa":
            from fanno.synthesize.qa import QASynthesizer
            synth = QASynthesizer(model=args.model, workers=args.workers)
        elif args.type == "creative":
            from fanno.synthesize.creative import CreativeSynthesizer
            synth = CreativeSynthesizer(model=args.model, workers=args.workers)
        elif args.type == "dialog":
            from fanno.synthesize.dialog import DialogSynthesizer
            synth = DialogSynthesizer(model=args.model, workers=args.workers)
        elif args.type == "agent":
            from fanno.synthesize.agent import AgentSynthesizer
            synth = AgentSynthesizer(model=args.model, workers=args.workers)
        else:
            logger.error(f"Unknown synthesis type: {args.type}")
            sys.exit(1)
        data = synth.generate(num_samples=args.num_samples)
        from fanno.data.loader import save_jsonlines
        save_jsonlines(data, args.output, overwrite=True)
        logger.info(f"Synthesized {len(data)} samples → {args.output}")

    elif args.command == "evaluate":
        from fanno.evaluate.quality import QualityEvaluator
        from fanno.data.loader import load_jsonlines
        data = load_jsonlines(args.data)
        evaluator = QualityEvaluator(model=args.model)
        report = evaluator.evaluate(data, source_type=args.source_type)
        logger.info(f"Evaluation complete: {report.get('stats', {})}")

    elif args.command == "prepare":
        from fanno.train.prepare import main as prepare_main
        sys.argv = [
            "fanno-prepare",
            "--output-dir", args.output_dir,
            "--fanno-dir", args.fanno_dir,
            "--max-fanno-qa", str(args.max_fanno_qa),
            "--max-alpaca", str(args.max_alpaca),
            "--max-arena", str(args.max_arena),
            "--max-bfcl", str(args.max_bfcl),
            "--seed", str(args.seed),
        ]
        prepare_main()

    elif args.command == "train":
        from fanno.train.sft import main as train_main
        sys.argv = [
            "fanno-train",
            "--model", args.model,
            "--data", args.data,
            "--output-dir", args.output_dir,
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--per-device-batch", str(args.per_device_batch),
            "--gradient-accum", str(args.gradient_accum),
            "--wandb-project", args.wandb_project,
        ]
        if args.deepspeed:
            sys.argv.append("--deepspeed")
        train_main()

    elif args.command == "amlt":
        from fanno.train.amlt import generate_amlt_config
        config_yaml = generate_amlt_config(job_name=args.job_name)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(config_yaml)
        logger.info(f"AMLT config written to {args.output}")


if __name__ == "__main__":
    main()
