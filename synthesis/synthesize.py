"""
FANNO-Dev: Large-scale data synthesis pipeline.
Supports batch synthesis of QA, Multi-turn, Code, Math, Reasoning, Creative Writing, and Trajectory Inversion.
Uses GPT-4o via Azure with multi-endpoint load balancing.
"""
from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from synthesis.api_client import call_gpt, parallel_call_gpt
from synthesis.prompts.templates import (
    complex_qa_prompt,
    code_qa_prompt,
    math_qa_prompt,
    reasoning_qa_prompt,
    creative_writing_prompt,
    multi_turn_dialog_prompt,
    trajectory_inversion_prompt,
    trajectory_inversion_with_verification_prompt,
    generate_mixed_batch_prompts,
    seed_qa_prompt,
    generate_all_seed_prompts,
    think_different_prompt,
    qa_response_prompt,
    COMPLEX_QA_DOMAINS,
    COMPLEX_QA_TYPES,
    CODE_TOPICS,
    CODE_LANGUAGES,
    MATH_LEVELS,
    MATH_TOPICS,
)


OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_jsonl(data: List[Dict], path: Path, mode: str = "a"):
    """Append data to a JSONL file."""
    with open(path, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict]:
    """Load data from a JSONL file."""
    if not path.exists():
        return []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def count_existing(path: Path) -> int:
    """Count existing lines in a JSONL file."""
    if not path.exists():
        return 0
    with open(path, "r") as f:
        return sum(1 for line in f if line.strip())


def parse_json_response(response: str) -> Optional[Dict]:
    """Robustly parse JSON from LLM response."""
    if response is None:
        return None
    # Try direct parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    # Try extracting JSON block
    for start_marker in ["```json", "```"]:
        if start_marker in response:
            start = response.index(start_marker) + len(start_marker)
            end = response.index("```", start) if "```" in response[start:] else len(response)
            try:
                return json.loads(response[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass
    # Try finding first { ... }
    brace_start = response.find("{")
    brace_end = response.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(response[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass
    return None


# =============================================================================
# Synthesis Functions
# =============================================================================

def synthesize_complex_qa(
    target: int = 1000,
    model: str = "gpt-4o",
    workers: int = 50,
    batch_size: int = 100,
    output_file: str = "complex_qa.jsonl",
) -> Path:
    """Synthesize complex QA pairs across all domains and types."""
    output_path = OUTPUT_DIR / output_file
    existing = count_existing(output_path)
    logger.info(f"Complex QA synthesis: target={target}, existing={existing}, remaining={max(0, target-existing)}")

    while existing < target:
        remaining = target - existing
        current_batch = min(batch_size, remaining)

        # Ensure domain/type diversity
        prompts = []
        for i in range(current_batch):
            domain = COMPLEX_QA_DOMAINS[i % len(COMPLEX_QA_DOMAINS)]
            qa_type = COMPLEX_QA_TYPES[i % len(COMPLEX_QA_TYPES)]
            prompts.append(complex_qa_prompt(domain=domain, qa_type=qa_type))

        responses = parallel_call_gpt(
            prompts=prompts, model_name=model, max_tokens=2048,
            temperature=0.9, workers=workers, json_mode=True,
        )

        batch_data = []
        for resp in responses:
            parsed = parse_json_response(resp)
            if parsed and "question" in parsed and "answer" in parsed:
                parsed["source"] = "fanno_complex_qa"
                parsed["model"] = model
                parsed["timestamp"] = datetime.now().isoformat()
                batch_data.append(parsed)

        if batch_data:
            save_jsonl(batch_data, output_path)
            existing += len(batch_data)
            logger.info(f"Complex QA: +{len(batch_data)} (total: {existing}/{target})")
        else:
            logger.warning("Complex QA batch produced 0 valid results, retrying...")

    return output_path


def synthesize_code_qa(
    target: int = 1000,
    model: str = "gpt-4o",
    workers: int = 50,
    batch_size: int = 100,
    output_file: str = "code_qa.jsonl",
) -> Path:
    """Synthesize code QA pairs."""
    output_path = OUTPUT_DIR / output_file
    existing = count_existing(output_path)
    logger.info(f"Code QA synthesis: target={target}, existing={existing}")

    while existing < target:
        remaining = target - existing
        current_batch = min(batch_size, remaining)

        prompts = []
        for i in range(current_batch):
            topic = CODE_TOPICS[i % len(CODE_TOPICS)]
            lang = CODE_LANGUAGES[i % len(CODE_LANGUAGES)]
            prompts.append(code_qa_prompt(topic=topic, language=lang))

        responses = parallel_call_gpt(
            prompts=prompts, model_name=model, max_tokens=3000,
            temperature=0.8, workers=workers, json_mode=True,
        )

        batch_data = []
        for resp in responses:
            parsed = parse_json_response(resp)
            if parsed and "question" in parsed and "answer" in parsed:
                parsed["source"] = "fanno_code_qa"
                parsed["model"] = model
                parsed["timestamp"] = datetime.now().isoformat()
                batch_data.append(parsed)

        if batch_data:
            save_jsonl(batch_data, output_path)
            existing += len(batch_data)
            logger.info(f"Code QA: +{len(batch_data)} (total: {existing}/{target})")

    return output_path


def synthesize_math_qa(
    target: int = 1000,
    model: str = "gpt-4o",
    workers: int = 50,
    batch_size: int = 100,
    output_file: str = "math_qa.jsonl",
) -> Path:
    """Synthesize math QA pairs."""
    output_path = OUTPUT_DIR / output_file
    existing = count_existing(output_path)
    logger.info(f"Math QA synthesis: target={target}, existing={existing}")

    while existing < target:
        remaining = target - existing
        current_batch = min(batch_size, remaining)

        prompts = []
        for i in range(current_batch):
            level = MATH_LEVELS[i % len(MATH_LEVELS)]
            topic = MATH_TOPICS[i % len(MATH_TOPICS)]
            prompts.append(math_qa_prompt(level=level, topic=topic))

        responses = parallel_call_gpt(
            prompts=prompts, model_name=model, max_tokens=2048,
            temperature=0.7, workers=workers, json_mode=True,
        )

        batch_data = []
        for resp in responses:
            parsed = parse_json_response(resp)
            if parsed and "question" in parsed:
                parsed["source"] = "fanno_math_qa"
                parsed["model"] = model
                parsed["timestamp"] = datetime.now().isoformat()
                batch_data.append(parsed)

        if batch_data:
            save_jsonl(batch_data, output_path)
            existing += len(batch_data)
            logger.info(f"Math QA: +{len(batch_data)} (total: {existing}/{target})")

    return output_path


def synthesize_reasoning_qa(
    target: int = 1000,
    model: str = "gpt-4o",
    workers: int = 50,
    batch_size: int = 100,
    output_file: str = "reasoning_qa.jsonl",
) -> Path:
    """Synthesize reasoning/logic QA pairs."""
    output_path = OUTPUT_DIR / output_file
    existing = count_existing(output_path)
    logger.info(f"Reasoning QA synthesis: target={target}, existing={existing}")

    while existing < target:
        remaining = target - existing
        current_batch = min(batch_size, remaining)

        prompts = [reasoning_qa_prompt() for _ in range(current_batch)]

        responses = parallel_call_gpt(
            prompts=prompts, model_name=model, max_tokens=2048,
            temperature=0.9, workers=workers, json_mode=True,
        )

        batch_data = []
        for resp in responses:
            parsed = parse_json_response(resp)
            if parsed and "question" in parsed and "answer" in parsed:
                parsed["source"] = "fanno_reasoning_qa"
                parsed["model"] = model
                parsed["timestamp"] = datetime.now().isoformat()
                batch_data.append(parsed)

        if batch_data:
            save_jsonl(batch_data, output_path)
            existing += len(batch_data)
            logger.info(f"Reasoning QA: +{len(batch_data)} (total: {existing}/{target})")

    return output_path


def synthesize_creative_writing(
    target: int = 1000,
    model: str = "gpt-4o",
    workers: int = 50,
    batch_size: int = 100,
    output_file: str = "creative_writing.jsonl",
) -> Path:
    """Synthesize creative writing pairs."""
    output_path = OUTPUT_DIR / output_file
    existing = count_existing(output_path)
    logger.info(f"Creative writing synthesis: target={target}, existing={existing}")

    while existing < target:
        remaining = target - existing
        current_batch = min(batch_size, remaining)

        prompts = [creative_writing_prompt() for _ in range(current_batch)]

        responses = parallel_call_gpt(
            prompts=prompts, model_name=model, max_tokens=2048,
            temperature=1.0, workers=workers, json_mode=True,
        )

        batch_data = []
        for resp in responses:
            parsed = parse_json_response(resp)
            if parsed and "instruction" in parsed and "response" in parsed:
                parsed["source"] = "fanno_creative_writing"
                parsed["model"] = model
                parsed["timestamp"] = datetime.now().isoformat()
                batch_data.append(parsed)

        if batch_data:
            save_jsonl(batch_data, output_path)
            existing += len(batch_data)
            logger.info(f"Creative writing: +{len(batch_data)} (total: {existing}/{target})")

    return output_path


def synthesize_multi_turn(
    target: int = 1000,
    model: str = "gpt-4o",
    workers: int = 30,
    batch_size: int = 50,
    output_file: str = "multi_turn.jsonl",
) -> Path:
    """Synthesize multi-turn dialog data."""
    output_path = OUTPUT_DIR / output_file
    existing = count_existing(output_path)
    logger.info(f"Multi-turn dialog synthesis: target={target}, existing={existing}")

    while existing < target:
        remaining = target - existing
        current_batch = min(batch_size, remaining)

        prompts = []
        for _ in range(current_batch):
            num_turns = random.randint(3, 8)
            prompts.append(multi_turn_dialog_prompt(num_turns=num_turns))

        responses = parallel_call_gpt(
            prompts=prompts, model_name=model, max_tokens=4096,
            temperature=0.9, workers=workers, json_mode=True,
        )

        batch_data = []
        for resp in responses:
            parsed = parse_json_response(resp)
            if parsed and "conversation" in parsed and isinstance(parsed["conversation"], list):
                if len(parsed["conversation"]) >= 4:  # At least 2 turns
                    parsed["source"] = "fanno_multi_turn"
                    parsed["model"] = model
                    parsed["timestamp"] = datetime.now().isoformat()
                    batch_data.append(parsed)

        if batch_data:
            save_jsonl(batch_data, output_path)
            existing += len(batch_data)
            logger.info(f"Multi-turn: +{len(batch_data)} (total: {existing}/{target})")

    return output_path


def synthesize_fanno_seed_qa(
    seed_data_path: str = None,
    target: int = 10000,
    model: str = "gpt-4o",
    workers: int = 50,
    batch_size: int = 200,
    output_file: str = "fanno_seed_qa.jsonl",
) -> Path:
    """
    Run FANNO's core seed QA pipeline: document -> question -> answer.
    Uses FANNO's tagging-based seed generation with GPT-4o for both
    question generation and answer generation.
    """
    output_path = OUTPUT_DIR / output_file
    existing = count_existing(output_path)
    logger.info(f"FANNO Seed QA synthesis: target={target}, existing={existing}")

    # Load seed documents
    if seed_data_path is None:
        seed_data_path = Path(__file__).parent.parent / "data" / "unlabel_data.jsonl"
    else:
        seed_data_path = Path(seed_data_path)

    docs = []
    with open(seed_data_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    docs.append(item.get("doc", ""))
                except json.JSONDecodeError:
                    continue

    logger.info(f"Loaded {len(docs)} seed documents")
    random.shuffle(docs)

    while existing < target:
        remaining = target - existing
        # Each doc produces multiple prompts, but we process in batches
        current_doc_batch = min(batch_size // 10, remaining // 10, len(docs))  # ~10 Qs per doc
        if current_doc_batch <= 0:
            current_doc_batch = min(10, len(docs))

        batch_docs = random.sample(docs, min(current_doc_batch, len(docs)))

        # Stage 1: Generate questions from documents
        qa_prompts = []
        doc_mapping = []
        for doc in batch_docs:
            # Generate 5 diverse prompts per document (instead of 96)
            for _ in range(5):
                qa_prompts.append(seed_qa_prompt(doc))
                doc_mapping.append(doc)

        logger.info(f"Generating {len(qa_prompts)} questions from {len(batch_docs)} documents...")
        questions = parallel_call_gpt(
            prompts=qa_prompts, model_name=model, max_tokens=512,
            temperature=0.9, workers=workers,
        )

        # Clean questions
        valid_questions = []
        valid_docs = []
        for q, doc in zip(questions, doc_mapping):
            if q and len(q.strip()) > 10:
                # Hard filter
                q_clean = q.strip().strip('"*\n')
                lower_q = q_clean.lower()
                if any(kw in lower_q for kw in ["based on", "according to", "given the", "mentioned in", "provided"]):
                    continue
                if len(q_clean.split()) < 5:
                    continue
                valid_questions.append(q_clean)
                valid_docs.append(doc)

        logger.info(f"Valid questions after filtering: {len(valid_questions)}/{len(qa_prompts)}")

        if not valid_questions:
            continue

        # Stage 2: Generate answers
        answer_prompts = [qa_response_prompt(q) for q in valid_questions]
        answers = parallel_call_gpt(
            prompts=answer_prompts, model_name=model, max_tokens=2048,
            temperature=0.7, workers=workers,
        )

        batch_data = []
        for q, a, doc in zip(valid_questions, answers, valid_docs):
            if a and len(a.strip()) > 20:
                batch_data.append({
                    "instruction": q,
                    "input": "",
                    "output": a.strip(),
                    "source": "fanno_seed_qa",
                    "model": model,
                    "doc_preview": doc[:200],
                    "timestamp": datetime.now().isoformat(),
                })

        if batch_data:
            save_jsonl(batch_data, output_path)
            existing += len(batch_data)
            logger.info(f"FANNO Seed QA: +{len(batch_data)} (total: {existing}/{target})")

    return output_path


# =============================================================================
# Master Synthesis Pipeline
# =============================================================================

def run_full_synthesis(
    total_target: int = 100000,
    model: str = "gpt-4o",
    workers: int = 50,
) -> Dict[str, Path]:
    """
    Run the full FANNO synthesis pipeline to generate 100K+ diverse data.

    Distribution:
    - FANNO Seed QA (document-based): 30K
    - Complex QA (standalone): 25K
    - Code QA: 15K
    - Math QA: 10K
    - Reasoning QA: 10K
    - Creative Writing: 5K
    - Multi-turn Dialog: 10K
    """
    logger.info(f"=" * 60)
    logger.info(f"FANNO-Dev Full Synthesis Pipeline")
    logger.info(f"Total Target: {total_target}")
    logger.info(f"Model: {model}")
    logger.info(f"Workers: {workers}")
    logger.info(f"=" * 60)

    results = {}

    # Calculate targets based on total
    ratio = total_target / 100000
    targets = {
        "fanno_seed_qa": int(30000 * ratio),
        "complex_qa": int(25000 * ratio),
        "code_qa": int(15000 * ratio),
        "math_qa": int(10000 * ratio),
        "reasoning_qa": int(10000 * ratio),
        "creative_writing": int(5000 * ratio),
        "multi_turn": int(10000 * ratio),
    }

    logger.info(f"Synthesis targets: {json.dumps(targets, indent=2)}")

    # Run each pipeline
    results["fanno_seed_qa"] = synthesize_fanno_seed_qa(
        target=targets["fanno_seed_qa"], model=model, workers=workers,
    )
    results["complex_qa"] = synthesize_complex_qa(
        target=targets["complex_qa"], model=model, workers=workers,
    )
    results["code_qa"] = synthesize_code_qa(
        target=targets["code_qa"], model=model, workers=workers,
    )
    results["math_qa"] = synthesize_math_qa(
        target=targets["math_qa"], model=model, workers=workers,
    )
    results["reasoning_qa"] = synthesize_reasoning_qa(
        target=targets["reasoning_qa"], model=model, workers=workers,
    )
    results["creative_writing"] = synthesize_creative_writing(
        target=targets["creative_writing"], model=model, workers=workers,
    )
    results["multi_turn"] = synthesize_multi_turn(
        target=targets["multi_turn"], model=model, workers=workers,
    )

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Synthesis Complete!")
    total = 0
    for name, path in results.items():
        count = count_existing(path)
        total += count
        logger.info(f"  {name}: {count} samples -> {path}")
    logger.info(f"  TOTAL: {total} samples")
    logger.info(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FANNO-Dev Data Synthesis")
    parser.add_argument("--target", type=int, default=100000, help="Total number of samples to generate")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use for synthesis")
    parser.add_argument("--workers", type=int, default=50, help="Number of parallel workers")
    parser.add_argument("--type", type=str, default="all",
                        choices=["all", "seed_qa", "complex_qa", "code_qa", "math_qa",
                                 "reasoning_qa", "creative_writing", "multi_turn"],
                        help="Type of data to synthesize")
    args = parser.parse_args()

    if args.type == "all":
        run_full_synthesis(total_target=args.target, model=args.model, workers=args.workers)
    elif args.type == "seed_qa":
        synthesize_fanno_seed_qa(target=args.target, model=args.model, workers=args.workers)
    elif args.type == "complex_qa":
        synthesize_complex_qa(target=args.target, model=args.model, workers=args.workers)
    elif args.type == "code_qa":
        synthesize_code_qa(target=args.target, model=args.model, workers=args.workers)
    elif args.type == "math_qa":
        synthesize_math_qa(target=args.target, model=args.model, workers=args.workers)
    elif args.type == "reasoning_qa":
        synthesize_reasoning_qa(target=args.target, model=args.model, workers=args.workers)
    elif args.type == "creative_writing":
        synthesize_creative_writing(target=args.target, model=args.model, workers=args.workers)
    elif args.type == "multi_turn":
        synthesize_multi_turn(target=args.target, model=args.model, workers=args.workers)
