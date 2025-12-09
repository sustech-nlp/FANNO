from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from loguru import logger
from tqdm import tqdm

from fanno.config import FannoConfig, InferenceConfig
from fanno.evaluator import Evaluator
from fanno.strategies.response import build_response_strategy
from fanno.strategies.selection import random_judge, ucb_judge
from fanno.inference import run_inference
from fanno.template.seed_template import generate_seed_prompt
from fanno.template.ucb_template import TD
from fanno.utils.data_utils import get_unlabeled_data, instruction_cleaning, load_jsonlines, save_jsonlines


class FannoPipeline:
    def __init__(self, config: FannoConfig):
        self.config = config
        self.inference_cfg: InferenceConfig = config.inference
        self.quality_strategy = self._build_quality_strategy()
        self.response_strategy = build_response_strategy(config)
        self._idx = 0
        config.files.run_dir.mkdir(parents=True, exist_ok=True)

    def _next_indices(self, count: int) -> range:
        idx_range = range(self._idx, self._idx + count)
        self._idx += count
        return idx_range

    def _format_instruction(self, item: Dict[str, Any]) -> str:
        if item.get("input"):
            return f"{item['instruction']}\n{item['input']}"
        return item["instruction"]

    def _attach_indices(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for item in data:
            if "idx" not in item:
                idx_range = self._next_indices(1)
                item["idx"] = idx_range.start
        return data

    def _build_quality_strategy(self):
        if self.config.pipeline.instruction_quality_strategy != "combined":
            raise ValueError(f"Unknown instruction quality strategy: {self.config.pipeline.instruction_quality_strategy}")
        return Evaluator(self.config)

    # ----------------------------------------------------------------- helpers
    def _generate_seeds(self, docs: List[str]) -> List[Dict[str, Any]]:
        if self.config.pipeline.seed_gen_strategy != "tagging":
            raise ValueError(f"Unknown seed strategy: {self.config.pipeline.seed_gen_strategy}")

        prompts: List[str] = []
        raw_doc: List[str] = []
        for doc in tqdm(docs, desc="Seed prompts", leave=False):
            new_prompts = generate_seed_prompt(doc)
            prompts += new_prompts
            raw_doc += [doc] * len(new_prompts)

        gen_results = instruction_cleaning(
            run_inference(prompts, config=self.inference_cfg, template_type="direct")
        )
        gen_instruction = [part[0] for part in gen_results]
        gen_input = [part[1] for part in gen_results]

        seeds = [
            {
                "instruction": instruction,
                "input": input,
                "value": 0.0,
                "doc": doc,
                "cnt": 0,
            }
            for instruction, input, doc in zip(gen_instruction, gen_input, raw_doc)
        ]
        return seeds

    def _think_prompts(self, docs: List[str], seeds: List[Dict[str, Any]]) -> List[str]:
        if self.config.pipeline.think_diff_strategy == "random":
            few_shots_list = random_judge(seeds, top_k=3, N=len(docs))
        else:
            few_shots_list = ucb_judge(
                seeds, top_k=3, N=len(docs), model_name=self.inference_cfg.model_name_or_path
            )

        cut_num = 100
        for few_shots in few_shots_list:
            for i, few_shot in enumerate(few_shots):
                temp_instruction = few_shot["instruction"].split()
                if len(temp_instruction) > cut_num:
                    few_shots[i]["instruction"] = " ".join(temp_instruction[:cut_num])

        few_shots_list = [tuple(few_shot["instruction"] for few_shot in few_shots) for few_shots in few_shots_list]
        prompts_list = []
        for text, (seed1, seed2, seed3) in zip(docs, few_shots_list):
            prompts_list.append(TD(text=text, seed1=seed1, seed2=seed2, seed3=seed3))
        return prompts_list

    def _augment_instructions(self, docs: List[str], seeds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.config.pipeline.ins_aug_strategy != "ucb":
            raise ValueError(f"Unknown instruction augmentation strategy: {self.config.pipeline.ins_aug_strategy}")
        prompts = self._think_prompts(docs, seeds)
        if prompts:
            logger.debug(f"prompts: {prompts[0]}")
        gen_results = instruction_cleaning(
            run_inference(prompts, config=self.inference_cfg, template_type="direct")
        )
        gen_instruction = [part[0] for part in gen_results]
        gen_input = [part[1] for part in gen_results]

        gen_data = [
            {
                "instruction": instruction,
                "input": input,
                "doc": doc,
                "value": 0.0,
                "cnt": 0,
            }
            for instruction, input, doc in zip(gen_instruction, gen_input, docs)
        ]
        return gen_data

    # --------------------------------------------------------------------- seeds
    def seed_generate(self, docs: List[str]) -> List[Dict[str, Any]]:
        seed_path = self.config.files.seed_path
        if seed_path.exists():
            logger.info(f"Seed cache hit at {seed_path}, skipping generation.")
            cached = load_jsonlines(seed_path)
            self._idx = max(item.get("idx", 0) for item in cached) + 1 if cached else 0
            return cached

        seeds = self._generate_seeds(docs)
        seeds = self._attach_indices(seeds)
        seeds = self.quality_strategy.evaluate(seeds, [])
        seeds = self.response_generate(seeds)
        save_jsonlines(seeds, seed_path)
        return seeds

    # ---------------------------------------------------------------- responses
    def response_generate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not data:
            return data

        instructions = [self._format_instruction(item) for item in data]
        response_sets = self.response_strategy.generate(instructions)

        expanded: List[Dict[str, Any]] = []
        for base_item, resp_list in zip(data, response_sets):
            for resp in resp_list:
                base_copy = {k: v for k, v in base_item.items() if k != "idx"}
                new_item = {**base_copy, "output": resp}
                expanded.append(new_item)

        expanded = self._attach_indices(expanded)
        return expanded

    # --------------------------------------------------------- instruction aug
    def augment_instructions(self, docs: List[str], seeds: List[Dict[str, Any]], file_path: Path) -> List[Dict[str, Any]]:
        if file_path.exists():
            logger.info(f"Augment cache hit at {file_path}, skipping generation.")
            cached = load_jsonlines(file_path)
            self._idx = max(self._idx, max(item.get("idx", 0) for item in cached) + 1 if cached else self._idx)
            return seeds + cached

        instruction_gen = self._augment_instructions(docs, seeds)
        instruction_gen = self._attach_indices(instruction_gen)
        instruction_gen = self.quality_strategy.evaluate(instruction_gen, old_data=seeds)
        instruction_gen = self.response_generate(instruction_gen)
        save_jsonlines(instruction_gen, file_path)
        seeds += instruction_gen
        return seeds

    # --------------------------------------------------------------------- run
    def run(self) -> List[Dict[str, Any]]:
        docs = get_unlabeled_data(self.config.files, merge_bool=True)
        seed_docs_num, window_size, limit_size = (
            self.config.pipeline.seed_docs_num,
            self.config.pipeline.window_size,
            self.config.pipeline.limit_size,
        )

        seeds = self.seed_generate(docs[:seed_docs_num])
        total_windows = max(0, (len(docs) - seed_docs_num + window_size - 1) // window_size)
        progress = tqdm(total=total_windows, desc="Augmenting", leave=True)
        for idx, i in enumerate(range(seed_docs_num, len(docs), window_size)):
            file_path = self.config.files.run_dir / f"ucb_aug_{idx}.jsonl"
            seeds = self.augment_instructions(docs[i : i + window_size], seeds, file_path=file_path)
            if len(seeds) > limit_size:
                break
            progress.update(1)
        progress.close()
        save_jsonlines(seeds, self.config.files.final_data_path)
        return seeds


def run_pipeline(config_path: str | Path | None = None) -> List[Dict[str, Any]]:
    cfg = FannoConfig.from_yaml(config_path)
    pipeline = FannoPipeline(cfg)
    return pipeline.run()


__all__ = ["FannoPipeline", "run_pipeline"]
