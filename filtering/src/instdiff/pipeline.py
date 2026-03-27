import os
from typing import Dict, Any

from .data_utils import sample_data
from .selection import (
    compare_models,
    get_total_tokens,
    filter_by_diff_nll,
    select_lowest_diff_entropy,
    save_selected_data,
)
from .train_runner import run_train
from .utils import set_seed


def _safe_mkdir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"创建目录失败: {path}") from exc


def run_pipeline(config: Dict[str, Any]) -> None:
    seed = int(config.get("seed", 42))
    set_seed(seed)

    base_model = config["base_model"]
    data_path = config["data_path"]
    output_root = config["output_dir"]
    iterations = int(config.get("iterations", 1))

    _safe_mkdir(output_root)

    for iteration in range(1, iterations + 1):
        iter_dir = os.path.join(output_root, f"iter_{iteration}")
        _safe_mkdir(iter_dir)
        
        if iteration == 1:
            warmup_cfg = config.get("warmup", {})
            warmup_data = warmup_cfg.get("data_path", data_path)
            warmup_subset_path = os.path.join(iter_dir, "warmup_subset.jsonl")
            warmup_sample_size = int(warmup_cfg.get("sample_size", 100))
            warmup_limit = warmup_cfg.get("limit")

            sample_data(
                warmup_data,
                warmup_subset_path,
                max_samples=warmup_sample_size,
                seed=seed,
                limit=warmup_limit,
            )

            calibration_dir = os.path.join(iter_dir, "calibration_model")
            if not os.path.exists(calibration_dir):
                warmup_train = warmup_cfg.get("train", {})
                warmup_train.update(
                    {
                        "model_name_or_path": base_model,
                        "data_path": warmup_subset_path,
                        "output_dir": calibration_dir,
                    }
                )
                run_train({"seed": seed, "train": warmup_train})
            else:
                print(f"Calibration model already exists in {calibration_dir}")

        else:
            # using the previous iteration's selected model as the base model
            calibration_dir = os.path.join(output_root, f"iter_{iteration - 1}", "selected_train_model")

        score_cfg = config.get("score", {})
        scored_path = os.path.join(iter_dir, "scored.jsonl")
        if not os.path.exists(scored_path):
            scored = compare_models(
                base_model,
                calibration_dir,
                data_path,
                scored_path,
                instruction_key=score_cfg.get("instruction_key", "instruction"),
                input_key=score_cfg.get("input_key", "input"),
                response_key=score_cfg.get("response_key", "response"),
                prompt_template=score_cfg.get("prompt_template"),
                max_samples=score_cfg.get("max_samples"),
            )

            select_cfg = config.get("select", {})
            reject_ratio = float(select_cfg.get("nll_reject_ratio", 0.1))
            select_ratio = float(select_cfg.get("select_ratio", 0.1))
            total_tokens = get_total_tokens(scored)
            filtered = filter_by_diff_nll(scored, reject_ratio)
            selected = select_lowest_diff_entropy(filtered, select_ratio, total_tokens)

            selected_path = os.path.join(iter_dir, "selected.jsonl")
            save_selected_data(selected, selected_path)
        else:
            selected_path = os.path.join(iter_dir, "selected.jsonl")
            print(f"Selected data already exists in {scored_path}")

        train_cfg = config.get("select_train", {})
        selected_model_dir = os.path.join(iter_dir, "selected_train_model")
        if not os.path.exists(selected_model_dir):
            train_kwargs = dict(train_cfg.get("train", {}))
            train_kwargs.update(
                {
                    "model_name_or_path": base_model,
                    "data_path": selected_path,
                    "output_dir": selected_model_dir,
                }
            )
            run_train({"seed": seed, "train": train_kwargs})
        else:
            print(f"Selected model already exists in {selected_model_dir}")

