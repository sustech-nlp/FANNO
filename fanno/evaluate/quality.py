"""Unified quality evaluation for all FANNO data types."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

from fanno.data.cleaning import hard_filter as _hard_filter


@dataclass
class FilterResult:
    """Result of a filtering operation."""
    kept: List[Dict[str, Any]]
    removed: int
    total: int

    @property
    def keep_ratio(self) -> float:
        return len(self.kept) / self.total if self.total > 0 else 0.0


@dataclass
class EvaluationReport:
    """Summary of a full evaluation run."""
    data: List[Dict[str, Any]]
    stats: Dict[str, Any] = field(default_factory=dict)


class QualityEvaluator:
    """Unified quality evaluation for all data types.

    Combines rule-based filtering, LLM-as-judge scoring, and
    embedding-based diversity filtering into a single evaluator.

    Usage:
        evaluator = QualityEvaluator(model="gpt-4o-mini")
        report = evaluator.evaluate(data, source_type="general")
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        workers: int = 30,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.workers = workers
        self.embed_model = embed_model
        self.device = device
        self._encoder = None  # lazy init
        self._api_client = None  # lazy init

    @property
    def api_client(self):
        """Lazy-initialize Azure API client."""
        if self._api_client is None:
            from fanno.api.client import AzureAPIClient
            self._api_client = AzureAPIClient(
                model_name=self.model,
                workers=self.workers,
            )
        return self._api_client

    @property
    def encoder(self):
        """Lazy-initialize sentence transformer encoder."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(
                self.embed_model, trust_remote_code=True
            ).to(self.device)
        return self._encoder

    # === Rule-based filters ===

    def hard_filter(
        self,
        data: List[Dict[str, Any]],
        source_type: str = "general",
    ) -> FilterResult:
        """Source-aware hard filtering (rule-based).

        Args:
            data: List of dicts with at least "instruction" key.
            source_type: "general", "agent", or "code".

        Returns:
            FilterResult with kept data and statistics.
        """
        kept = _hard_filter(data, source_type=source_type)
        return FilterResult(kept=kept, removed=len(data) - len(kept), total=len(data))

    # === LLM-as-Judge scoring ===

    def score_answer_quality(
        self,
        qa_pairs: List[Dict[str, Any]],
    ) -> List[int]:
        """Score answer quality on a 1-5 scale using LLM-as-judge.

        Uses the faithfulness_eval template for scoring.

        Args:
            qa_pairs: List of dicts with "instruction" and "output" keys.

        Returns:
            List of scores (1-5) for each pair.
        """
        from fanno.template.eval_template import faithfulness_eval

        prompts = [
            faithfulness_eval(item["instruction"], item.get("output", ""))
            for item in qa_pairs
        ]

        responses = self.api_client.batch_chat(prompts, max_tokens=8)

        scores: List[int] = []
        pattern = re.compile(r"[1-5]")
        for resp in responses:
            match = pattern.search(resp.strip())
            scores.append(int(match.group()) if match else 3)
        return scores

    def score_instruction_quality(
        self,
        instructions: Sequence[str],
    ) -> List[float]:
        """Score instruction quality using LLM-based evaluation.

        Evaluates instructions on multiple dimensions (privacy, safety,
        originality, difficulty) and returns composite scores.

        Args:
            instructions: List of instruction strings.

        Returns:
            List of composite scores (0.0-1.0) for each instruction.
        """
        from fanno.template.eval_template import (
            originality_eval,
            difficult_eval,
            insjudge_eval,
        )

        if not instructions:
            return []

        # Score on multiple dimensions
        orig_prompts = [originality_eval(i) for i in instructions]
        diff_prompts = [difficult_eval(i) for i in instructions]
        judge_prompts = [insjudge_eval(i) for i in instructions]

        all_prompts = orig_prompts + diff_prompts + judge_prompts
        all_responses = self.api_client.batch_chat(all_prompts, max_tokens=8)

        n = len(instructions)
        orig_scores = self._parse_binary_scores(all_responses[:n])
        diff_scores = self._parse_binary_scores(all_responses[n:2*n])
        judge_scores = self._parse_binary_scores(all_responses[2*n:])

        # Composite: weighted average of dimensions
        composite = [
            0.3 * o + 0.4 * d + 0.3 * j
            for o, d, j in zip(orig_scores, diff_scores, judge_scores)
        ]
        return composite

    def _parse_binary_scores(self, responses: List[str]) -> List[float]:
        """Parse binary (0/1) scores from LLM responses."""
        pattern = re.compile(r"score:\s*(\d)", re.IGNORECASE)
        scores: List[float] = []
        for resp in responses:
            match = pattern.search(resp)
            scores.append(float(match.group(1)) if match else 0.5)
        return scores

    # === LLM filters ===

    def llm_filter(
        self,
        data: List[Dict[str, Any]],
    ) -> FilterResult:
        """Filter instructions using LLM-based privacy check.

        Args:
            data: List of dicts with "instruction" key.

        Returns:
            FilterResult with kept data.
        """
        from fanno.template.eval_template import privacy_eval

        instructions = [d["instruction"] for d in data]
        prompts = [privacy_eval(instr) for instr in instructions]
        responses = self.api_client.batch_chat(prompts, max_tokens=8)

        scores = self._parse_binary_scores(responses)
        kept = [item for item, score in zip(data, scores) if score >= 0.5]
        return FilterResult(kept=kept, removed=len(data) - len(kept), total=len(data))

    # === Diversity filter ===

    def diversity_filter(
        self,
        new_data: List[Dict[str, Any]],
        old_data: Optional[List[Dict[str, Any]]] = None,
        threshold: float = 0.8,
        min_community_size: int = 1,
        words_num: int = 4,
    ) -> FilterResult:
        """Deduplicate using community detection on sentence embeddings.

        Args:
            new_data: New data to filter.
            old_data: Existing data to avoid duplicating.
            threshold: Cosine similarity threshold for community detection.
            min_community_size: Min cluster size for community detection.
            words_num: Number of leading words to use for embedding.

        Returns:
            FilterResult with deduplicated data.
        """
        if not new_data:
            return FilterResult(kept=[], removed=0, total=0)

        old_data = old_data or []
        combined = new_data + old_data

        # Truncate instructions for embedding (focus on first few words)
        truncated = [
            " ".join(item["instruction"].split()[:words_num])
            if len(item["instruction"].split()) > words_num
            else item["instruction"]
            for item in combined
        ]

        embeddings = self.encoder.encode(
            truncated,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=self.device,
            batch_size=64,
        )

        from sentence_transformers import util
        clusters = util.community_detection(
            embeddings,
            min_community_size=min_community_size,
            threshold=threshold,
        )

        # Select best from each cluster, preferring new data
        selected: List[Dict[str, Any]] = []
        for cluster in clusters:
            cluster_items = [combined[idx] for idx in cluster]
            # Sort by value score (higher is better)
            cluster_items.sort(key=lambda x: x.get("value", 0), reverse=True)
            for item in cluster_items:
                if item in old_data:
                    continue
                selected.append(item)
                break

        logger.info(
            f"Diversity filter: kept {len(selected)}/{len(new_data)} "
            f"({len(selected) / len(new_data):.2%})"
        ) if new_data else None

        return FilterResult(
            kept=selected,
            removed=len(new_data) - len(selected),
            total=len(new_data),
        )

    # === Full evaluation pipeline ===

    def evaluate(
        self,
        data: List[Dict[str, Any]],
        source_type: Optional[str] = None,
        old_data: Optional[List[Dict[str, Any]]] = None,
        enable_llm_filter: bool = False,
        enable_diversity_filter: bool = True,
        enable_quality_scoring: bool = True,
    ) -> Dict[str, Any]:
        """Full evaluation pipeline: filter → score → diversity → report.

        Args:
            data: Input data to evaluate.
            source_type: "general", "agent", or "code".
            old_data: Previous data for diversity filtering.
            enable_llm_filter: Whether to run LLM privacy filter.
            enable_diversity_filter: Whether to run diversity dedup.
            enable_quality_scoring: Whether to score answer quality.

        Returns:
            Dict with 'data' (filtered list) and 'stats' (evaluation statistics).
        """
        source_type = source_type or "general"
        stats: Dict[str, Any] = {"input_count": len(data)}

        # Step 1: Hard filter
        hard_result = self.hard_filter(data, source_type=source_type)
        filtered = hard_result.kept
        stats["after_hard_filter"] = len(filtered)
        stats["hard_filter_removed"] = hard_result.removed

        # Step 2: LLM filter (optional)
        if enable_llm_filter and filtered:
            llm_result = self.llm_filter(filtered)
            filtered = llm_result.kept
            stats["after_llm_filter"] = len(filtered)
            stats["llm_filter_removed"] = llm_result.removed

        # Step 3: Quality scoring (optional)
        if enable_quality_scoring and filtered:
            # Score instructions
            instructions = [d["instruction"] for d in filtered]
            instr_scores = self.score_instruction_quality(instructions)
            for item, score in zip(filtered, instr_scores):
                item["instruction_quality"] = score

            # Score answers if they exist
            qa_items = [d for d in filtered if d.get("output")]
            if qa_items:
                answer_scores = self.score_answer_quality(qa_items)
                for item, score in zip(qa_items, answer_scores):
                    item["answer_quality"] = score
                stats["avg_answer_quality"] = sum(answer_scores) / len(answer_scores)

            stats["avg_instruction_quality"] = (
                sum(instr_scores) / len(instr_scores) if instr_scores else 0
            )

        # Step 4: Diversity filter (optional)
        if enable_diversity_filter and filtered:
            div_result = self.diversity_filter(filtered, old_data=old_data)
            filtered = div_result.kept
            stats["after_diversity_filter"] = len(filtered)
            stats["diversity_filter_removed"] = div_result.removed

        stats["output_count"] = len(filtered)
        logger.info(f"Evaluation complete: {stats}")

        return {"data": filtered, "stats": stats}


__all__ = ["QualityEvaluator", "FilterResult", "EvaluationReport"]
