"""Data format conversion utilities: Alpaca, ShareGPT, Agent formats."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def to_alpaca_format(
    data: List[Dict[str, Any]],
    instruction_key: str = "instruction",
    input_key: str = "input",
    output_key: str = "output",
) -> List[Dict[str, str]]:
    """Convert data to Alpaca format: {instruction, input, output}.

    Handles missing keys gracefully by defaulting to empty strings.
    """
    result: List[Dict[str, str]] = []
    for item in data:
        result.append({
            "instruction": str(item.get(instruction_key, "")),
            "input": str(item.get(input_key, "")),
            "output": str(item.get(output_key, "")),
        })
    return result


def to_sharegpt_format(
    data: List[Dict[str, Any]],
    system_message: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convert data to ShareGPT multi-turn format.

    Input items should have "instruction" and "output" keys (single-turn)
    or a "conversations" key (multi-turn, already in ShareGPT format).
    """
    result: List[Dict[str, Any]] = []
    for item in data:
        if "conversations" in item:
            entry: Dict[str, Any] = {"conversations": item["conversations"]}
            if system_message:
                entry["system"] = system_message
            result.append(entry)
            continue

        conversations: List[Dict[str, str]] = []
        if system_message:
            conversations.append({"from": "system", "value": system_message})

        user_msg = item.get("instruction", "")
        if item.get("input"):
            user_msg += f"\n{item['input']}"
        conversations.append({"from": "human", "value": user_msg})
        conversations.append({"from": "gpt", "value": item.get("output", "")})
        result.append({"conversations": conversations})
    return result


def to_agent_format(
    data: List[Dict[str, Any]],
    system_message: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convert agent trajectory data to OpenAI function-calling messages format.

    Input items should have a "trajectory" key containing a list of steps,
    each with "role", "content", and optionally "function_call" or "tool_calls".
    Falls back to ShareGPT format for non-trajectory data.
    """
    result: List[Dict[str, Any]] = []
    for item in data:
        if "trajectory" in item:
            messages: List[Dict[str, Any]] = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            for step in item["trajectory"]:
                msg: Dict[str, Any] = {"role": step["role"], "content": step.get("content", "")}
                if "function_call" in step:
                    msg["function_call"] = step["function_call"]
                if "tool_calls" in step:
                    msg["tool_calls"] = step["tool_calls"]
                if step["role"] == "tool":
                    msg["tool_call_id"] = step.get("tool_call_id", "")
                messages.append(msg)
            entry: Dict[str, Any] = {
                "messages": messages,
                "tools": item.get("tools", []),
            }
            if item.get("metadata"):
                entry["metadata"] = item["metadata"]
            result.append(entry)
        else:
            # Fall back to ShareGPT conversion
            result.extend(to_sharegpt_format([item], system_message=system_message))
    return result


__all__ = ["to_alpaca_format", "to_sharegpt_format", "to_agent_format"]
