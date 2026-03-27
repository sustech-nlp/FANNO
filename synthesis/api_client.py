"""
Unified API client for large-scale data synthesis.
Supports Azure OpenAI (GPT-4o, GPT-4.1, GPT-5) with multi-endpoint load balancing.
"""
from __future__ import annotations

import json
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

from azure.identity import AzureCliCredential, get_bearer_token_provider
from loguru import logger
from openai import AzureOpenAI


def get_endpoints() -> dict:
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
    return {
        "gpt-4o": gpt_4o,
        "gpt-4o-mini": gpt_4o_mini,
        "gpt-4.1": gpt_4_1,
    }


def select_endpoint(model_name: str) -> dict:
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


def call_gpt(
    prompt: str,
    model_name: str = "gpt-4o",
    max_tokens: int = 2048,
    temperature: float = 0.8,
    system_prompt: Optional[str] = None,
    json_mode: bool = False,
    retries: int = 3,
) -> str:
    """Single call to GPT with retry logic."""
    for attempt in range(retries):
        try:
            client, resolved_model = get_client(model_name=model_name)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            kwargs = {
                "model": resolved_model,
                "max_completion_tokens": max_tokens,
                "messages": messages,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt + random.random()
                logger.warning(f"API call failed (attempt {attempt+1}/{retries}): {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                logger.error(f"API call failed after {retries} attempts: {e}")
                raise


def parallel_call_gpt(
    prompts: List[str],
    model_name: str = "gpt-4o",
    max_tokens: int = 2048,
    temperature: float = 0.8,
    system_prompt: Optional[str] = None,
    json_mode: bool = False,
    workers: int = 50,
    retries: int = 3,
) -> List[Optional[str]]:
    """Parallel GPT calls with load balancing across endpoints."""
    if not prompts:
        return []

    results: List[Optional[str]] = [None] * len(prompts)

    def _infer(idx: int, prompt: str) -> Tuple[int, Optional[str]]:
        try:
            result = call_gpt(
                prompt=prompt,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                json_mode=json_mode,
                retries=retries,
            )
            return idx, result
        except Exception as e:
            logger.error(f"Failed prompt #{idx}: {e}")
            return idx, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_infer, idx, prompt) for idx, prompt in enumerate(prompts)]
        for fut in as_completed(futures):
            idx, text = fut.result()
            results[idx] = text

    success_count = sum(1 for r in results if r is not None)
    logger.info(f"Parallel inference: {success_count}/{len(prompts)} succeeded (model={model_name})")
    return results


def parallel_call_gpt_chat(
    conversations: List[List[Dict[str, str]]],
    model_name: str = "gpt-4o",
    max_tokens: int = 2048,
    temperature: float = 0.8,
    json_mode: bool = False,
    workers: int = 50,
    retries: int = 3,
) -> List[Optional[str]]:
    """Parallel GPT calls with full conversation history (multi-turn)."""
    if not conversations:
        return []

    results: List[Optional[str]] = [None] * len(conversations)

    def _infer(idx: int, messages: List[Dict[str, str]]) -> Tuple[int, Optional[str]]:
        for attempt in range(retries):
            try:
                client, resolved_model = get_client(model_name=model_name)
                kwargs = {
                    "model": resolved_model,
                    "max_completion_tokens": max_tokens,
                    "messages": messages,
                }
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**kwargs)
                return idx, resp.choices[0].message.content
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt + random.random())
                else:
                    logger.error(f"Chat call #{idx} failed: {e}")
                    return idx, None
        return idx, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_infer, idx, msgs) for idx, msgs in enumerate(conversations)]
        for fut in as_completed(futures):
            idx, text = fut.result()
            results[idx] = text

    success_count = sum(1 for r in results if r is not None)
    logger.info(f"Chat inference: {success_count}/{len(conversations)} succeeded (model={model_name})")
    return results
