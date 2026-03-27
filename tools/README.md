<div align="center">

# FANNO-Tools
**Synthetic Tool-Augmented Dialogue Generation Framework**  
*Submitted to ACL 2026*

</div>

## Introduction
FANNO-Tools is a modular framework for synthesizing tool-augmented, multi-turn dialogues. It uses LLM agents to create scenarios, define tools, evaluate quality, simulate world feedback, and generate realistic user turns. Output conversations follow OpenAI function-calling conventions with alternating `human/gpt/function_call/observation` steps, ready for training and evaluation of executable dialogue systems.

## Data Format
- `conversations`: ordered turns with `from` ∈ {human, gpt, function_call, observation} and `value`
- `function_call` / `observation`: JSON strings conforming to `tools` schema
- `system`: global policies/constraints influencing the run but not counted as a turn
- Flow rule of thumb: need info → `function_call` → `observation` → `gpt` decision/explanation; if blocked → finalize/escalate tool

Example (abstract):
```json
{
  "conversations": [
    {"from": "human", "value": "Something in my request did not work."},
    {"from": "function_call", "value": "{\"name\":\"query_state\",\"arguments\":{\"key\":\"XYZ\"}}"},
    {"from": "observation", "value": "{\"key\":\"XYZ\",\"state\":\"invalid\"}"},
    {"from": "gpt", "value": "The current state is invalid, so this step cannot proceed."},
    {"from": "function_call", "value": "{\"name\":\"finalize\",\"arguments\":{\"reason\":\"state invalid\"}}"},
    {"from": "observation", "value": "Done"}
  ]
}
```

## Pipeline
```
Seed Data (JSONL)
       |
       v
 ScenarioGenerator (LLM)  --creates-->  system prompt + tools + meta
       |
       v
 QualityEvaluator (LLM) --scores--> accept / reject
       |
       v (accepted)
 MultiTurnGenerator (LLM loop)
   |- initial user query (LLM)
   |- decide action (LLM) -> tool call or direct reply
   |- if tool call:
   |     function_call -> WorldModel (LLM simulation) -> observation
   |     observation -> assistant reply (LLM)
   |- user reply (UserSimulator LLM)
   |- completion check (LLM)
       |
       v
   conversations (human/gpt/function_call/observation) + tools + system
       |
       v
 Output JSONL
```

## Project Layout
- `src/`
  - `agents/`: ScenarioGenerator, QualityEvaluator, WorldModel, UserSimulator, MultiTurnGenerator
  - `prompt_templates.py`: all LLM prompts centralized
  - `config.py`: defaults, constants, and config classes
  - `pipeline.py`: end-to-end orchestration (scenario → eval → multi-turn → validation/write)
  - `inference_utils.py`: Azure/OpenAI endpoint selection & client helpers
- `synthetic_self_play.py`: entry script calling `src.pipeline.main`
- `tests/`: pytest-based module tests
- `requirements.txt`: runtime & testing deps

## Quick Start
```bash
python synthetic_self_play.py \
  --input data/unlabel_data.jsonl \
  --output synthetic_data.jsonl \
  --target 20 \
  --max-turns 8 \
  --min-score 7 \
  --workers 4
```
Optional:
- `--num-tools`: fix tool count (default random 3–5)
- `--logic-pattern`: smooth | partial_failure | error_recovery | escalation | user_change_mind | multi_goal
- `--seed`: reproducibility seed
- `--workers`: parallel generation (independent trajectories) with tqdm progress

## Testing
1) Install deps: `pip install -r requirements.txt`  
2) Run: `pytest -q`

Coverage:
- Scenario generation parsing (ScenarioGenerator)
- Quality scoring parsing (QualityEvaluator)
- Tool execution simulation (WorldModel)
- User response generation (UserSimulator)
- Multi-turn orchestration (MultiTurnGenerator)
- Inference endpoints sanity (inference_utils)

> Tests monkeypatch `call_gpt` to avoid live network when models are unavailable.

## Contributors
- He Zhu  
- Junyou Su  
- Guanhua Chen  


## TODO