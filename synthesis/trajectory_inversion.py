"""
Trajectory Inversion Pipeline for FANNO-Dev.
Core idea: Reverse-engineer questions from existing trajectories/answers.
Even when final results are wrong, intermediate reasoning steps are valuable.
"""
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from synthesis.api_client import call_gpt, parallel_call_gpt
from synthesis.synthesize import OUTPUT_DIR, count_existing, load_jsonl, parse_json_response, save_jsonl


# =============================================================================
# Trajectory Sources
# =============================================================================

def load_trajectories_from_jsonl(path: str, max_count: int = 10000) -> List[Dict]:
    """Load trajectories from a JSONL file."""
    trajectories = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    trajectories.append(item)
                except json.JSONDecodeError:
                    continue
            if len(trajectories) >= max_count:
                break
    return trajectories


def extract_trajectory_text(item: Dict) -> str:
    """Extract trajectory/answer text from various data formats."""
    # Format 1: {output: "..."}
    if "output" in item:
        return item["output"]
    # Format 2: {answer: "..."}
    if "answer" in item:
        return item["answer"]
    # Format 3: {response: "..."}
    if "response" in item:
        return item["response"]
    # Format 4: {conversation: [...]} - extract assistant turns
    if "conversation" in item:
        turns = item["conversation"]
        assistant_parts = [t["content"] for t in turns if t.get("role") == "assistant"]
        return "\n\n".join(assistant_parts)
    # Format 5: {conversations: [...]} - FANNO-Tools format
    if "conversations" in item:
        turns = item["conversations"]
        parts = []
        for t in turns:
            if t.get("from") == "gpt":
                parts.append(t.get("value", ""))
            elif t.get("from") == "function_call":
                parts.append(f"[Tool Call] {t.get('value', '')}")
            elif t.get("from") == "observation":
                parts.append(f"[Tool Result] {t.get('value', '')}")
        return "\n".join(parts)
    # Format 6: {solution: "..."}
    if "solution" in item:
        return item["solution"]
    return str(item)


# =============================================================================
# Inversion Prompts
# =============================================================================

INVERSION_SYSTEM_PROMPT = """You are an expert at creating high-quality training data from existing answers/trajectories.
Your task is to reverse-engineer natural, self-contained questions from given answers.
The questions should be challenging and well-formed, as if asked by a real user."""


def build_basic_inversion_prompt(trajectory: str) -> str:
    """Basic inversion: trajectory -> question."""
    return f"""Given the following answer/trajectory, create a natural, well-formed question that would produce this response.

### Answer/Trajectory:
{trajectory[:3000]}

Requirements:
1. The question must be SELF-CONTAINED (no references to the answer).
2. The question should be specific enough that this answer is appropriate.
3. The question should sound natural, as if asked by a real person.
4. Do NOT use phrases like "Based on the above" or "According to the trajectory".

Output JSON:
{{"question": "the question", "difficulty": "easy|medium|hard|expert", "domain": "relevant domain"}}"""


def build_verified_inversion_prompt(trajectory: str, has_errors: bool = False) -> str:
    """Advanced inversion that leverages verifiable intermediate results."""
    error_clause = """
IMPORTANT: This trajectory contains some errors. Your job is to:
- Identify which intermediate steps are CORRECT and valuable
- Create a question-answer pair that uses only the CORRECT parts
- Fix any errors while preserving good reasoning""" if has_errors else ""

    return f"""You are creating training data from an execution trajectory.
{error_clause}

### Trajectory:
{trajectory[:3000]}

Tasks:
1. Identify the valuable intermediate reasoning steps in this trajectory.
2. Create a self-contained question that would lead to this kind of reasoning.
3. Create an improved answer that incorporates the best parts of the trajectory.

Output JSON:
{{
    "question": "self-contained question",
    "answer": "improved answer using correct parts of trajectory",
    "valuable_steps": ["list of good intermediate steps"],
    "quality_score": 0.0 to 1.0,
    "domain": "domain",
    "difficulty": "easy|medium|hard|expert"
}}"""


def build_multi_turn_inversion_prompt(conversation_text: str) -> str:
    """Invert a multi-turn conversation into a seed scenario + first question."""
    return f"""Given the following multi-turn conversation, reverse-engineer:
1. A realistic scenario description
2. The initial user question that would start this conversation
3. A summary of what knowledge this conversation teaches

### Conversation:
{conversation_text[:4000]}

Output JSON:
{{
    "scenario": "description of the scenario",
    "initial_question": "the first user question",
    "knowledge_summary": "what this conversation teaches",
    "topic": "main topic",
    "difficulty": "easy|medium|hard|expert",
    "num_useful_turns": number of turns with valuable content
}}"""


# =============================================================================
# Inversion Pipeline
# =============================================================================

def invert_qa_trajectories(
    source_path: str,
    target: int = 5000,
    model: str = "gpt-4o",
    workers: int = 30,
    batch_size: int = 100,
    output_file: str = "trajectory_inverted_qa.jsonl",
) -> Path:
    """Invert QA trajectories (answer -> question)."""
    output_path = OUTPUT_DIR / output_file
    existing = count_existing(output_path)
    logger.info(f"Trajectory inversion (QA): target={target}, existing={existing}")

    # Load source trajectories
    source_data = load_trajectories_from_jsonl(source_path)
    logger.info(f"Loaded {len(source_data)} source trajectories")

    if not source_data:
        logger.warning("No source data found!")
        return output_path

    while existing < target:
        remaining = target - existing
        current_batch = min(batch_size, remaining, len(source_data))

        # Sample trajectories
        batch_items = random.sample(source_data, min(current_batch, len(source_data)))

        prompts = []
        for item in batch_items:
            trajectory = extract_trajectory_text(item)
            if len(trajectory.strip()) < 50:
                continue
            prompts.append(build_basic_inversion_prompt(trajectory))

        if not prompts:
            continue

        responses = parallel_call_gpt(
            prompts=prompts,
            model_name=model,
            max_tokens=1024,
            temperature=0.8,
            system_prompt=INVERSION_SYSTEM_PROMPT,
            json_mode=True,
            workers=workers,
        )

        batch_data = []
        for resp, src_item in zip(responses, batch_items):
            parsed = parse_json_response(resp)
            if parsed and "question" in parsed:
                original_trajectory = extract_trajectory_text(src_item)
                parsed["inverted_from"] = original_trajectory[:500]
                parsed["source"] = "trajectory_inversion"
                parsed["model"] = model
                parsed["timestamp"] = datetime.now().isoformat()
                batch_data.append(parsed)

        if batch_data:
            save_jsonl(batch_data, output_path)
            existing += len(batch_data)
            logger.info(f"Trajectory inversion: +{len(batch_data)} (total: {existing}/{target})")

    return output_path


def invert_with_verification(
    source_path: str,
    target: int = 5000,
    model: str = "gpt-4o",
    workers: int = 30,
    batch_size: int = 100,
    output_file: str = "trajectory_verified_inversion.jsonl",
) -> Path:
    """Advanced inversion with quality verification of intermediate results."""
    output_path = OUTPUT_DIR / output_file
    existing = count_existing(output_path)
    logger.info(f"Verified trajectory inversion: target={target}, existing={existing}")

    source_data = load_trajectories_from_jsonl(source_path)
    logger.info(f"Loaded {len(source_data)} source trajectories")

    while existing < target:
        remaining = target - existing
        current_batch = min(batch_size, remaining, len(source_data))

        batch_items = random.sample(source_data, min(current_batch, len(source_data)))

        prompts = []
        valid_items = []
        for item in batch_items:
            trajectory = extract_trajectory_text(item)
            if len(trajectory.strip()) < 100:  # Need substantial trajectory
                continue
            # Randomly decide if we want to simulate error-containing trajectories
            has_errors = random.random() < 0.3
            prompts.append(build_verified_inversion_prompt(trajectory, has_errors=has_errors))
            valid_items.append(item)

        if not prompts:
            continue

        responses = parallel_call_gpt(
            prompts=prompts,
            model_name=model,
            max_tokens=2048,
            temperature=0.8,
            system_prompt=INVERSION_SYSTEM_PROMPT,
            json_mode=True,
            workers=workers,
        )

        batch_data = []
        for resp, src_item in zip(responses, valid_items):
            parsed = parse_json_response(resp)
            if parsed and "question" in parsed and "answer" in parsed:
                quality = parsed.get("quality_score", 0.5)
                if isinstance(quality, (int, float)) and quality >= 0.4:
                    parsed["source"] = "verified_trajectory_inversion"
                    parsed["model"] = model
                    parsed["timestamp"] = datetime.now().isoformat()
                    batch_data.append(parsed)

        if batch_data:
            save_jsonl(batch_data, output_path)
            existing += len(batch_data)
            logger.info(f"Verified inversion: +{len(batch_data)} (total: {existing}/{target})")

    return output_path


def invert_self_generated(
    target: int = 5000,
    model: str = "gpt-4o",
    workers: int = 30,
    batch_size: int = 100,
    output_file: str = "self_inverted_qa.jsonl",
) -> Path:
    """
    Self-inversion: Use already-synthesized data as trajectories for inversion.
    This creates a feedback loop that can discover new question types.
    """
    output_path = OUTPUT_DIR / output_file
    existing = count_existing(output_path)
    logger.info(f"Self-inversion from synthesized data: target={target}, existing={existing}")

    # Collect all existing synthesized outputs as trajectory sources
    all_outputs = []
    for fname in ["complex_qa.jsonl", "code_qa.jsonl", "math_qa.jsonl",
                   "reasoning_qa.jsonl", "creative_writing.jsonl", "fanno_seed_qa.jsonl"]:
        fpath = OUTPUT_DIR / fname
        if fpath.exists():
            data = load_jsonl(fpath)
            all_outputs.extend(data)

    if not all_outputs:
        logger.warning("No synthesized data available for self-inversion yet!")
        return output_path

    logger.info(f"Found {len(all_outputs)} synthesized samples to invert from")

    while existing < target:
        remaining = target - existing
        current_batch = min(batch_size, remaining, len(all_outputs))

        batch_items = random.sample(all_outputs, min(current_batch, len(all_outputs)))

        prompts = []
        for item in batch_items:
            # Use the ANSWER as the trajectory to invert
            trajectory = extract_trajectory_text(item)
            if len(trajectory.strip()) < 50:
                continue

            # Invert: from this answer, create a DIFFERENT question
            prompt = f"""Given this answer/response, create a COMPLETELY DIFFERENT question
that this answer could also address (but from a different angle or perspective).

### Answer:
{trajectory[:2000]}

The new question should:
1. Be self-contained and natural
2. Address the same knowledge but from a different angle
3. Be at a similar or higher difficulty level
4. NOT be a simple rephrasing of the original question

Output JSON:
{{"question": "new question from different angle", "answer": "adapted answer for this new question", "original_angle": "what the original question was likely about", "new_angle": "what this new question focuses on", "domain": "domain", "difficulty": "medium|hard|expert"}}"""
            prompts.append(prompt)

        if not prompts:
            continue

        responses = parallel_call_gpt(
            prompts=prompts,
            model_name=model,
            max_tokens=2048,
            temperature=0.9,
            json_mode=True,
            workers=workers,
        )

        batch_data = []
        for resp in responses:
            parsed = parse_json_response(resp)
            if parsed and "question" in parsed and "answer" in parsed:
                parsed["source"] = "self_inversion"
                parsed["model"] = model
                parsed["timestamp"] = datetime.now().isoformat()
                batch_data.append(parsed)

        if batch_data:
            save_jsonl(batch_data, output_path)
            existing += len(batch_data)
            logger.info(f"Self-inversion: +{len(batch_data)} (total: {existing}/{target})")

    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Trajectory Inversion Pipeline")
    parser.add_argument("--source", type=str, help="Source JSONL file for inversion")
    parser.add_argument("--target", type=int, default=5000)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--mode", type=str, default="basic",
                        choices=["basic", "verified", "self"])
    args = parser.parse_args()

    if args.mode == "basic":
        if not args.source:
            raise ValueError("--source required for basic inversion")
        invert_qa_trajectories(args.source, target=args.target, model=args.model, workers=args.workers)
    elif args.mode == "verified":
        if not args.source:
            raise ValueError("--source required for verified inversion")
        invert_with_verification(args.source, target=args.target, model=args.model, workers=args.workers)
    elif args.mode == "self":
        invert_self_generated(target=args.target, model=args.model, workers=args.workers)
