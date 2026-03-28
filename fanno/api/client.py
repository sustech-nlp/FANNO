"""Azure OpenAI API client with endpoint load balancing and retry logic."""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

from azure.identity import AzureCliCredential, get_bearer_token_provider
from loguru import logger
from openai import AzureOpenAI


def get_endpoints() -> Dict[str, List[Dict[str, Any]]]:
    """Return available Azure OpenAI endpoints grouped by model family."""
    gpt_4o = [
        {"endpoints": "https://conversationhubeastus.openai.azure.com/", "speed": 150, "model": "gpt-4o"},
        {"endpoints": "https://conversationhubeastus2.openai.azure.com/", "speed": 150, "model": "gpt-4o"},
        {"endpoints": "https://conversationhubnorthcentralus.openai.azure.com/", "speed": 150, "model": "gpt-4o"},
        {"endpoints": "https://conversationhubsouthcentralus.openai.azure.com/", "speed": 150, "model": "gpt-4o"},
        {"endpoints": "https://conversationhubwestus.openai.azure.com/", "speed": 150, "model": "gpt-4o"},
        {"endpoints": "https://readineastus.openai.azure.com/", "speed": 150, "model": "gpt-4o"},
        {"endpoints": "https://readineastus2.openai.azure.com/", "speed": 150, "model": "gpt-4o"},
        {"endpoints": "https://readinnorthcentralus.openai.azure.com/", "speed": 150, "model": "gpt-4o"},
        {"endpoints": "https://readinwestus.openai.azure.com/", "speed": 150, "model": "gpt-4o"},
        {"endpoints": "https://conversationhubeastus.openai.azure.com/", "speed": 450, "model": "gpt-4o-global"},
        {"endpoints": "https://conversationhubeastus2.openai.azure.com/", "speed": 450, "model": "gpt-4o-global"},
        {"endpoints": "https://conversationhubnorthcentralus.openai.azure.com/", "speed": 450, "model": "gpt-4o-global"},
        {"endpoints": "https://conversationhubsouthcentralus.openai.azure.com/", "speed": 450, "model": "gpt-4o-global"},
        {"endpoints": "https://readineastus.openai.azure.com/", "speed": 450, "model": "gpt-4o-global"},
        {"endpoints": "https://readineastus2.openai.azure.com/", "speed": 450, "model": "gpt-4o-global"},
        {"endpoints": "https://readinnorthcentralus.openai.azure.com/", "speed": 450, "model": "gpt-4o-global"},
        {"endpoints": "https://readinwestus.openai.azure.com/", "speed": 450, "model": "gpt-4o-global"},
    ]
    gpt_4o_mini = [
        {"endpoints": "https://conversationhubeastus.openai.azure.com/", "speed": 150, "model": "gpt-4o-mini"},
        {"endpoints": "https://conversationhubeastus2.openai.azure.com/", "speed": 150, "model": "gpt-4o-mini"},
        {"endpoints": "https://conversationhubnorthcentralus.openai.azure.com/", "speed": 150, "model": "gpt-4o-mini"},
        {"endpoints": "https://conversationhubsouthcentralus.openai.azure.com/", "speed": 150, "model": "gpt-4o-mini"},
        {"endpoints": "https://conversationhubswedencentral.openai.azure.com/", "speed": 150, "model": "gpt-4o-mini"},
        {"endpoints": "https://conversationhubwestus.openai.azure.com/", "speed": 150, "model": "gpt-4o-mini"},
        {"endpoints": "https://readineastus.openai.azure.com/", "speed": 150, "model": "gpt-4o-mini"},
        {"endpoints": "https://readineastus2.openai.azure.com/", "speed": 150, "model": "gpt-4o-mini"},
        {"endpoints": "https://readinnorthcentralus.openai.azure.com/", "speed": 150, "model": "gpt-4o-mini"},
        {"endpoints": "https://readinwestus.openai.azure.com/", "speed": 150, "model": "gpt-4o-mini"},
        {"endpoints": "https://malicata-azure-ai-foundry.cognitiveservices.azure.com/", "speed": 150, "model": "gpt-4o-mini"},
    ]
    gpt_4_turbo = [
        {"endpoints": "https://conversationhubeastus2.openai.azure.com/", "speed": 150, "model": "gpt-4-turbo"},
        {"endpoints": "https://conversationhubswedencentral.openai.azure.com/", "speed": 150, "model": "gpt-4-turbo"},
        {"endpoints": "https://readineastus2.openai.azure.com/", "speed": 150, "model": "gpt-4-turbo"},
        {"endpoints": "https://readinswedencentral.openai.azure.com/", "speed": 150, "model": "gpt-4-turbo"},
    ]
    gpt_4_1 = [
        {"endpoints": "https://conversationhubnorthcentralus.openai.azure.com/", "speed": 150, "model": "gpt-4.1-DZS"},
        {"endpoints": "https://conversationhubsouthcentralus.openai.azure.com/", "speed": 150, "model": "gpt-4.1-DZS"},
        {"endpoints": "https://conversationhubswedencentral.openai.azure.com/", "speed": 150, "model": "gpt-4.1-DZS"},
        {"endpoints": "https://readinnorthcentralus.openai.azure.com/", "speed": 150, "model": "gpt-4.1-DZS"},
        {"endpoints": "https://conversationhubeastus2.openai.azure.com/", "speed": 150, "model": "gpt-4.1-global"},
        {"endpoints": "https://conversationhubnorthcentralus.openai.azure.com/", "speed": 150, "model": "gpt-4.1-global"},
        {"endpoints": "https://conversationhubswedencentral.openai.azure.com/", "speed": 150, "model": "gpt-4.1-global"},
        {"endpoints": "https://readinnorthcentralus.openai.azure.com/", "speed": 150, "model": "gpt-4.1-global"},
    ]
    gpt_5 = [
        {"endpoints": "https://conversationhubeastus2.openai.azure.com/", "speed": 150, "model": "gpt-5-global"},
    ]
    return {
        "gpt-4o": gpt_4o,
        "gpt-4o-mini": gpt_4o_mini,
        "gpt-4-turbo": gpt_4_turbo,
        "gpt-4.1": gpt_4_1,
        "gpt-5": gpt_5,
    }


def select_endpoint(model_name: str) -> Dict[str, Any]:
    """Select a random endpoint weighted by speed."""
    azure_endpoints = get_endpoints()
    entries = azure_endpoints[model_name]
    candidates = [e for e in entries if e.get("speed", 0) > 0 and e.get("endpoints")]
    weights = [e["speed"] for e in candidates]
    chosen = random.choices(candidates, weights=weights, k=1)[0]
    return chosen


def get_client(
    model_name: str = "gpt-4o",
    tenant_id: str = "72f988bf-86f1-41af-91ab-2d7cd011db47",
    api_version: str = "2024-12-01-preview",
    max_retries: int = 5,
) -> Tuple[AzureOpenAI, str]:
    """Create an Azure OpenAI client with AD token auth."""
    azure_ad_token_provider = get_bearer_token_provider(
        AzureCliCredential(tenant_id=tenant_id),
        "https://cognitiveservices.azure.com/.default",
    )
    selected = select_endpoint(model_name)
    client = AzureOpenAI(
        azure_endpoint=selected["endpoints"],
        azure_ad_token_provider=azure_ad_token_provider,
        api_version=api_version,
        max_retries=max_retries,
    )
    return client, selected["model"]


class AzureAPIClient:
    """High-level Azure OpenAI client with parallel inference support."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        tenant_id: str = "72f988bf-86f1-41af-91ab-2d7cd011db47",
        api_version: str = "2024-12-01-preview",
        max_retries: int = 5,
        workers: int = 8,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> None:
        self.model_name = model_name
        self.tenant_id = tenant_id
        self.api_version = api_version
        self.max_retries = max_retries
        self.workers = workers
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _get_client(self) -> Tuple[AzureOpenAI, str]:
        return get_client(
            model_name=self.model_name,
            tenant_id=self.tenant_id,
            api_version=self.api_version,
            max_retries=self.max_retries,
        )

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Single chat completion call."""
        client, resolved_model = self._get_client()
        send_temperature = None if self.model_name.startswith("gpt-5") else self.temperature
        call_kwargs: Dict[str, Any] = {
            "model": resolved_model,
            "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": messages,
        }
        if send_temperature is not None:
            call_kwargs["temperature"] = send_temperature
        call_kwargs.update({k: v for k, v in kwargs.items() if k not in call_kwargs})
        resp = client.chat.completions.create(**call_kwargs)
        return resp.choices[0].message.content

    def batch_chat(
        self,
        prompts: Sequence[str],
        system_message: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """Parallel chat completions for a list of user prompts."""
        if not prompts:
            return []

        client, resolved_model = self._get_client()
        send_temperature = None if self.model_name.startswith("gpt-5") else kwargs.get("temperature", self.temperature)

        def _infer(idx: int, prompt: str) -> Tuple[int, str]:
            messages: List[Dict[str, str]] = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            call_kwargs: Dict[str, Any] = {
                "model": resolved_model,
                "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
                "messages": messages,
            }
            if send_temperature is not None:
                call_kwargs["temperature"] = send_temperature
            resp = client.chat.completions.create(**call_kwargs)
            return idx, resp.choices[0].message.content

        outputs: List[str] = ["" for _ in prompts]
        workers = kwargs.get("workers", self.workers)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_infer, idx, prompt) for idx, prompt in enumerate(prompts)]
            for fut in as_completed(futures):
                idx, text = fut.result()
                outputs[idx] = text
        logger.info(f"Azure batch inference completed: {len(prompts)} prompts, model={resolved_model}")
        return outputs


__all__ = ["AzureAPIClient", "get_client", "get_endpoints", "select_endpoint"]
