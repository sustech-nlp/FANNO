from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

import numpy as np

from loguru import logger
from sentence_transformers import SentenceTransformer, util

from fanno.config import EvaluatorConfig, FannoConfig, InferenceConfig
from fanno.strategies.base import InstructionQualityStrategy
from fanno.template import eval_template
from fanno.inference import run_inference
from fanno.utils.metrics import InstructionMetrics


class Evaluator(InstructionQualityStrategy):
    """Evaluate, filter, and diversify generated instructions."""

    def __init__(self, config: FannoConfig):
        super().__init__(name="combined_quality")
        self.config = config
        self.inference_cfg: InferenceConfig = config.inference
        self.ev_cfg: EvaluatorConfig = config.evaluator
        self.metric_helper = InstructionMetrics(config.inference, config.metrics)
        self.encoder = SentenceTransformer(self.ev_cfg.encode_model_path, trust_remote_code=True).to(self.ev_cfg.device)

    def _log_stats(self, values: Sequence[int]) -> None:
        if not values:
            return
        p95, p5 = np.percentile(values, 95), np.percentile(values, 5)
        mean, med = np.mean(values), np.median(values)
        logger.info("Length stats - p95: {:.2f}, p5: {:.2f}, mean: {:.2f}, median: {:.2f}".format(p95, p5, mean, med))

    def hard_filter(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ref_key_word = ["based on", "according", "given", "mentioned", "refer", "provided", "passage", "text", "paragraph"]
        time_key_word = ["recent", "current", "now", "today", "yesterday", "tomorrow", "soon", "upcoming", "recently", "coming", "currently"]
        obj_key_word = ["name"]
        key_words = ref_key_word + time_key_word + obj_key_word

        lengths = [len(item["instruction"].split()) for item in data]
        self._log_stats(lengths)

        remaining_data = []
        for item in data:
            instruction = item["instruction"]
            if len(instruction) == 0:
                continue
            if not all(ord(c) < 128 for c in instruction):
                continue
            if len(instruction.split()) < 5 and instruction[-1] not in [".", "?"]:
                continue
            if any(re.search(key, instruction, re.IGNORECASE) for key in key_words):
                continue
            if sum(1 for c in instruction if c.isalpha()) < sum(1 for c in instruction if not c.isalpha()):
                continue
            remaining_data.append(item)
        logger.info(
            "Hard filter ratio: {:.2%}, remain ratio: {:.2%}".format(
                1 - len(remaining_data) / len(data) if data else 0, len(remaining_data) / len(data) if data else 0
            )
        ) if data else None
        return remaining_data

    def llm_filter(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        instructions = [d["instruction"] for d in data]
        privacy_prompts = [eval_template.privacy_eval(instruction) for instruction in instructions]
        privacy_scores = run_inference(privacy_prompts, config=self.inference_cfg, template_type="direct", score=True)
        remaining = [item for item, ps in zip(data, privacy_scores) if ps == 1]
        logger.info(
            "LLM filter ratio: {:.2%}, remain ratio: {:.2%}".format(
                1 - len(remaining) / len(data) if data else 0, len(remaining) / len(data) if data else 0
            )
        ) if data else None
        return remaining

    def _embed(self, instructions: Sequence[str]):
        # When words_num > 0, truncate to first N words (legacy behavior)
        # When words_num = 0, use full instruction text (recommended for better diversity filtering)
        if self.ev_cfg.words_num > 0:
            truncated = [
                " ".join(inst.split()[: self.ev_cfg.words_num]) if len(inst.split()) > self.ev_cfg.words_num else inst
                for inst in instructions
            ]
        else:
            truncated = list(instructions)
        return self.encoder.encode(
            truncated,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=self.ev_cfg.device,
            batch_size=self.ev_cfg.batch_size,
        )

    def diversity_filter(self, new_data: List[Dict[str, Any]], old_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not new_data:
            return []
        data = new_data + old_data
        embeddings = self._embed([d["instruction"] for d in data])
        clusters = util.community_detection(
            embeddings, min_community_size=self.ev_cfg.min_community_size, threshold=self.ev_cfg.threshold
        )
        selected = []
        for cluster in clusters:
            cluster_items = [data[idx].copy() for idx in cluster]
            cluster_items.sort(key=lambda x: x.get("value", 0), reverse=True)
            for item in cluster_items:
                if item in old_data:
                    continue
                selected.append(item)
                break
        logger.info(
            "Diversity filter ratio: {:.2%}, remain ratio: {:.2%}".format(
                1 - len(selected) / len(new_data), len(selected) / len(new_data)
            )
        ) if new_data else None
        return selected

    def score_instruction_values(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        metrics = self.metric_helper.score([d["instruction"] for d in data])
        for item, ppl, ifd, value in zip(data, metrics["perplexity"], metrics["ifd"], metrics["value"]):
            item["metrics"] = {"perplexity": ppl, "ifd": ifd}
            item["value"] = value
        return data

    def evaluate(self, new_data: List[Dict[str, Any]], old_data: List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
        old_data = old_data or []
        filtered = self.hard_filter(new_data)
        if self.ev_cfg.enable_llm_filter:
            filtered = self.llm_filter(filtered)
        if not filtered:
            return []
        filtered = self.score_instruction_values(filtered)
        return self.diversity_filter(filtered, old_data)


__all__ = ["Evaluator"]
