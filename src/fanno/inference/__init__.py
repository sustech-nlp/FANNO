from typing import Sequence, List

from fanno.config import InferenceConfig
from fanno.inference.client_inference import client_parallel_inference
from fanno.inference.vllm_inference import clear_cached_llms, parallel_inference, parser_score, _build_prompt


def run_inference(
    prompt_list: Sequence[str],
    config: InferenceConfig,
    template_type: str = "auto",
    score: bool = False,
) -> List[str] | List[int]:
    """Dispatch inference to vLLM or Azure client based on config.backend."""
    if not prompt_list:
        return []

    if config.backend == "azure":
        # Apply template formatting for parity with vLLM path.
        if template_type in {"direct", "alpaca", "mistral"}:
            templated = [_build_prompt(p, template_type, None) for p in prompt_list] if template_type != "direct" else list(prompt_list)
        else:
            templated = list(prompt_list)
        # GPT-5 defaults temperature to 1 and disallows overriding; skip if so.
        temperature = None if str(config.model_name_or_path).startswith("gpt-5") else config.temperature
        outputs = client_parallel_inference(
            templated,
            model_name=config.model_name_or_path,
            max_tokens=config.max_tokens,
            temperature=temperature if temperature is not None else 0.0,
            tenant_id=config.azure_tenant_id,
            api_version=config.azure_api_version,
            max_retries=config.azure_max_retries,
            workers=8,
        )
        return parser_score(outputs) if score else outputs

    # default: vLLM
    return parallel_inference(prompt_list, config=config, template_type=template_type, score=score)


__all__ = ["parallel_inference", "run_inference", "clear_cached_llms", "client_parallel_inference"]
