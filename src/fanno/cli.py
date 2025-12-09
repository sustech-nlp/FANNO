from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from fanno.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FANNO: synthetic instruction generation pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file. Defaults to src/fanno/config.yaml if not provided.",
    )
    parser.add_argument(
        "--stage",
        choices=["pipeline"],
        default="pipeline",
        help="Pipeline stage to run. Additional stages can be added over time.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.stage == "pipeline":
        records = run_pipeline(args.config)
        logger.info(f"Finished pipeline. Generated {len(records)} instruction/response pairs.")


if __name__ == "__main__":
    main()
