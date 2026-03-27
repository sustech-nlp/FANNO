import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from tqdm.auto import tqdm

from src.agents.multi_turn import MultiTurnGenerator
from src.agents.quality import MetricsTracker, QualityEvaluator
from src.agents.scenario_generator import ScenarioGenerator
from src.config import (
    ALLOWED_ROLES,
    DEFAULT_INPUT_PATH,
    DEFAULT_MAX_TURNS,
    DEFAULT_MIN_QUALITY_SCORE,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TARGET_CONVERSATIONS,
    DatasetConfig,
    LOGIC_PATTERNS,
    ScenarioConfig,
)
from src.utils import read_jsonl

DEFAULT_NUM_TOOLS_DIST = {3: 0.05, 4: 0.15, 5: 0.3, 6: 0.3, 7: 0.15, 8: 0.05}


def _parse_distribution_arg(raw, cast=None):
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            data = json.loads(raw)
            if cast:
                return {cast(k): v for k, v in data.items()}
            return data
        except json.JSONDecodeError:
            return None
    return cast(raw) if cast else raw


def _choose_weighted(rng: random.Random, maybe_dist, default_fn, cast=None):
    if isinstance(maybe_dist, dict) and maybe_dist:
        choices = []
        weights = []
        for k, w in maybe_dist.items():
            choices.append(cast(k) if cast else k)
            weights.append(w)
        return rng.choices(choices, weights=weights, k=1)[0]
    if maybe_dist is not None:
        return cast(maybe_dist) if cast else maybe_dist
    return default_fn()


def generate_dataset(
    input_path: str = DEFAULT_INPUT_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    target_conversations: int = DEFAULT_TARGET_CONVERSATIONS,
    max_turns: int = DEFAULT_MAX_TURNS,
    min_quality_score: int = DEFAULT_MIN_QUALITY_SCORE,
    config: ScenarioConfig = None,
    seed: int | None = None,
    workers: int = 1,
):
    seed_data = read_jsonl(input_path)
    if not seed_data:
        raise RuntimeError("No seed data found")

    metrics = MetricsTracker()

    accepted = 0
    attempts = 0
    scenario_rejections = 0
    conversation_rejections = 0
    max_attempts = target_conversations * 10

    def _single_attempt(rng_seed: int | None = None):
        rng_local = random.Random(rng_seed)
        scenario_gen = ScenarioGenerator()
        evaluator = QualityEvaluator()
        multi_turn_gen = MultiTurnGenerator()
        seed_doc = rng_local.choice(seed_data)
        if config is None:
            current_config = ScenarioConfig(
                num_tools=_choose_weighted(
                    rng_local, DEFAULT_NUM_TOOLS_DIST, lambda: rng_local.randint(3, 8), cast=int
                ),
                num_turns=rng_local.randint(6, 12),
                logic_pattern=rng_local.choice(list(LOGIC_PATTERNS.keys())),
            )
        else:
            current_config = ScenarioConfig(
                num_tools=_choose_weighted(
                    rng_local,
                    config.num_tools if config.num_tools is not None else DEFAULT_NUM_TOOLS_DIST,
                    lambda: rng_local.randint(3, 8),
                    cast=int,
                ),
                num_turns=config.num_turns or rng_local.randint(6, 12),
                logic_pattern=_choose_weighted(
                    rng_local,
                    config.logic_pattern,
                    lambda: rng_local.choice(list(LOGIC_PATTERNS.keys())),
                ),
                domain_hint=config.domain_hint,
            )
        scenario = scenario_gen.generate(seed_doc, current_config)
        score = evaluator.evaluate(scenario)
        if score < min_quality_score:
            return {"accepted": False, "scenario_reject": True}
        conversation_data = multi_turn_gen.generate(scenario, num_turns=max_turns)
        conv_score = evaluator.evaluate(
            scenario,
            scenario.get("tools", []),
            conversation_data.get("conversations", []),
        )
        if conv_score < min_quality_score:
            return {"accepted": False, "conversation_reject": True}
        if not validate_conversation_format(conversation_data):
            return {"accepted": False, "format_reject": True}
        return {"accepted": True, "conversation": conversation_data, "scenario": scenario}

    with open(output_path, "w", encoding="utf-8") as handle:
        if workers <= 1:
            pbar = tqdm(total=target_conversations, desc="Generating", unit="conv")
            while accepted < target_conversations and attempts < max_attempts:
                attempts += 1
                result = _single_attempt(seed + attempts if seed is not None else None)
                if result.get("scenario_reject"):
                    scenario_rejections += 1
                    continue
                if result.get("conversation_reject"):
                    conversation_rejections += 1
                    continue
                if not result.get("accepted"):
                    continue
                conversation_data = result["conversation"]
                scenario = result["scenario"]
                handle.write(json.dumps(conversation_data, ensure_ascii=False) + "\n")
                handle.flush()
                accepted += 1
                metrics.record_conversation(conversation_data, scenario)
                pbar.update(1)
            pbar.close()
        else:
            pbar = tqdm(total=target_conversations, desc="Generating", unit="conv")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = set()
                while accepted < target_conversations and attempts < max_attempts:
                    while len(futures) < workers * 2 and attempts < max_attempts:
                        attempts += 1
                        futures.add(executor.submit(_single_attempt, seed + attempts if seed is not None else None))
                    done, futures = _split_done(futures)
                    for fut in done:
                        try:
                            result = fut.result()
                        except Exception:
                            continue
                        if result.get("scenario_reject"):
                            scenario_rejections += 1
                            continue
                        if result.get("conversation_reject"):
                            conversation_rejections += 1
                            continue
                        if not result.get("accepted"):
                            continue
                        conversation_data = result["conversation"]
                        scenario = result["scenario"]
                        handle.write(json.dumps(conversation_data, ensure_ascii=False) + "\n")
                        handle.flush()
                        accepted += 1
                        metrics.record_conversation(conversation_data, scenario)
                        pbar.update(1)
                        if accepted >= target_conversations:
                            break
                for fut in futures:
                    fut.cancel()
            pbar.close()

    summary = metrics.summary()
    summary["attempts"] = attempts
    summary["accepted"] = accepted
    summary["min_quality_score"] = min_quality_score
    summary["scenario_rejections"] = scenario_rejections
    summary["conversation_rejections"] = conversation_rejections
    return summary


def _split_done(futures):
    done, pending = wait(futures, timeout=0.01, return_when=FIRST_COMPLETED)
    return done, pending


def validate_conversation_format(record: dict) -> bool:
    if "system" not in record or "tools" not in record or "conversations" not in record:
        return False
    conversations = record.get("conversations", [])
    if not conversations:
        return False
    for item in conversations:
        if "from" not in item or "value" not in item:
            return False
        role = item.get("from")
        if role not in ALLOWED_ROLES:
            return False
        if role in {"function_call", "observation"}:
            try:
                json.loads(item.get("value", ""))
            except json.JSONDecodeError:
                return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic tool-augmented dialogues using LLM-based agents"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Path to seed data (JSONL format)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Path to output file (JSONL format)")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET_CONVERSATIONS, help="Number of conversations to generate")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS, help="Maximum conversation turns")
    parser.add_argument("--min-score", type=int, default=DEFAULT_MIN_QUALITY_SCORE, help="Minimum quality score to accept")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--num-tools",
        type=str,
        default=None,
        help="Number of tools (e.g., 4) or JSON distribution (e.g., '{\"3\":0.4,\"4\":0.6}')",
    )
    parser.add_argument(
        "--logic-pattern",
        default=None,
        help="Logic pattern (e.g., smooth) or JSON distribution (e.g., '{\"smooth\":0.5,\"error_recovery\":0.5}')",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for generation (independent trajectories)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    scenario_cfg = None
    parsed_num_tools = _parse_distribution_arg(args.num_tools, cast=int)
    parsed_logic_pattern = _parse_distribution_arg(args.logic_pattern)
    if parsed_num_tools is not None or parsed_logic_pattern is not None:
        scenario_cfg = ScenarioConfig(num_tools=parsed_num_tools, logic_pattern=parsed_logic_pattern)

    summary = generate_dataset(
        input_path=args.input,
        output_path=args.output,
        target_conversations=args.target,
        max_turns=args.max_turns,
        min_quality_score=args.min_score,
        config=scenario_cfg,
        seed=args.seed,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
