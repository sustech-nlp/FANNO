import argparse
from typing import Any, Dict

from .config import load_yaml
from .scoring import score_dataset, DEFAULT_PROMPT
from .train_runner import run_train
from .pipeline import run_pipeline
from .data_utils import sample_data
from .utils import set_seed


def _merge_config(cli_args: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(cfg)
    merged.update({k: v for k, v in cli_args.items() if v is not None})
    return merged


def _load_config(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    return load_yaml(path)


def handle_score(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    merged = _merge_config(
        {
            "seed": args.seed,
            "data_path": args.data,
            "model_path": args.model,
            "output_path": args.output,
            "max_samples": args.max_samples,
            "instruction_key": args.instruction_key,
            "input_key": args.input_key,
            "response_key": args.response_key,
        },
        cfg,
    )

    set_seed(int(merged.get("seed", 42)))

    prompt_template = merged.get("prompt_template", DEFAULT_PROMPT)
    score_dataset(
        model_path=merged["model_path"],
        data_path=merged["data_path"],
        output_path=merged["output_path"],
        instruction_key=merged.get("instruction_key", "instruction"),
        input_key=merged.get("input_key", "input"),
        response_key=merged.get("response_key", "response"),
        prompt_template=prompt_template,
        max_samples=merged.get("max_samples"),
    )


def handle_train(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    run_train(cfg)


def handle_pipeline(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    run_pipeline(cfg)


def handle_sample(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    merged = _merge_config(
        {
            "data_path": args.data,
            "output_path": args.output,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "limit": args.limit,
        },
        cfg,
    )

    seed = int(merged.get("seed", 42))
    sample_data(
        merged["data_path"],
        merged["output_path"],
        max_samples=int(merged["max_samples"]),
        seed=seed,
        limit=merged.get("limit"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("instdiff")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="单模型评分")
    score_parser.add_argument("--data", required=False, help="数据路径")
    score_parser.add_argument("--model", required=False, help="模型路径")
    score_parser.add_argument("--output", required=False, help="输出路径")
    score_parser.add_argument("--config", required=False, help="YAML 配置")
    score_parser.add_argument("--max-samples", type=int, default=None)
    score_parser.add_argument("--seed", type=int, default=None)
    score_parser.add_argument("--instruction-key", default=None)
    score_parser.add_argument("--input-key", default=None)
    score_parser.add_argument("--response-key", default=None)
    score_parser.set_defaults(func=handle_score)

    train_parser = subparsers.add_parser("train", help="训练包装")
    train_parser.add_argument("--config", required=True, help="YAML 配置")
    train_parser.add_argument("--seed", type=int, default=None)
    train_parser.set_defaults(func=handle_train)

    pipeline_parser = subparsers.add_parser("pipeline", help="迭代选择管线")
    pipeline_parser.add_argument("--config", required=True, help="YAML 配置")
    pipeline_parser.add_argument("--seed", type=int, default=None)
    pipeline_parser.set_defaults(func=handle_pipeline)

    sample_parser = subparsers.add_parser("sample", help="采样子集")
    sample_parser.add_argument("--data", required=False, help="数据路径")
    sample_parser.add_argument("--output", required=False, help="输出路径")
    sample_parser.add_argument("--max-samples", type=int, default=None)
    sample_parser.add_argument("--seed", type=int, default=None)
    sample_parser.add_argument("--limit", type=int, default=None)
    sample_parser.add_argument("--config", required=False, help="YAML 配置")
    sample_parser.set_defaults(func=handle_sample)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
