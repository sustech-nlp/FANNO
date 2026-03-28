"""Agent trajectory synthesis with function-calling."""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

from loguru import logger

from fanno.synthesize.base import BaseSynthesizer
from fanno.synthesize.prompts import (
    AGENT_SCENARIO_PROMPT,
    AGENT_PLANNING_PROMPT,
    AGENT_EXECUTION_PROMPT,
    AGENT_WORLD_MODEL_PROMPT,
)


# ===== Tool Library =====

TOOL_LIBRARY: Dict[str, Dict[str, Any]] = {
    "web_search": {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information on a given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "num_results": {"type": "integer", "description": "Number of results to return", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    "code_execute": {
        "type": "function",
        "function": {
            "name": "code_execute",
            "description": "Execute Python code and return the output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "timeout": {"type": "integer", "description": "Execution timeout in seconds", "default": 30},
                },
                "required": ["code"],
            },
        },
    },
    "file_read": {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "encoding": {"type": "string", "description": "File encoding", "default": "utf-8"},
                },
                "required": ["path"],
            },
        },
    },
    "file_write": {
        "type": "function",
        "function": {
            "name": "file_write",
            "description": "Write content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    "calculator": {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform mathematical calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"},
                },
                "required": ["expression"],
            },
        },
    },
    "database_query": {
        "type": "function",
        "function": {
            "name": "database_query",
            "description": "Execute a SQL query against a database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to execute"},
                    "database": {"type": "string", "description": "Database name"},
                },
                "required": ["query", "database"],
            },
        },
    },
    "api_call": {
        "type": "function",
        "function": {
            "name": "api_call",
            "description": "Make an HTTP API request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "API endpoint URL"},
                    "method": {"type": "string", "description": "HTTP method", "default": "GET"},
                    "body": {"type": "object", "description": "Request body for POST/PUT"},
                },
                "required": ["url"],
            },
        },
    },
    "image_analyze": {
        "type": "function",
        "function": {
            "name": "image_analyze",
            "description": "Analyze an image and return descriptions or extracted text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": {"type": "string", "description": "URL or path to the image"},
                    "task": {"type": "string", "description": "Analysis task: 'describe', 'ocr', 'detect'", "default": "describe"},
                },
                "required": ["image_url"],
            },
        },
    },
}

# ===== Role-Pattern Combinations =====

AGENT_ROLES = ["researcher", "coder", "analyst", "planner", "executor"]

LOGIC_PATTERNS = ["sequential", "branching", "iterative", "recursive", "parallel", "adaptive"]

ROLE_TOOL_MAPPING: Dict[str, List[str]] = {
    "researcher": ["web_search", "file_read", "file_write", "api_call"],
    "coder": ["code_execute", "file_read", "file_write", "web_search"],
    "analyst": ["database_query", "calculator", "file_read", "code_execute"],
    "planner": ["web_search", "calculator", "file_read", "file_write"],
    "executor": ["code_execute", "api_call", "file_write", "file_read", "database_query"],
}

ROLE_TASK_TEMPLATES: Dict[str, List[str]] = {
    "researcher": [
        "Research the latest developments in {topic} and summarize the key findings.",
        "Find and compare different approaches to {topic}, then write a brief report.",
        "Gather data about {topic} from multiple sources and identify common patterns.",
    ],
    "coder": [
        "Write a Python script that {task} and save it to a file.",
        "Debug and fix the code that {task}, then verify it works correctly.",
        "Implement a solution for {task} with proper error handling and testing.",
    ],
    "analyst": [
        "Analyze the dataset about {topic} and provide statistical insights.",
        "Query the database to find {task} and visualize the results.",
        "Calculate the {metric} for {topic} and compare across categories.",
    ],
    "planner": [
        "Create a detailed project plan for {task} with timelines and milestones.",
        "Develop a strategy for {task} considering constraints and resources.",
        "Design a workflow for {task} with clear roles and dependencies.",
    ],
    "executor": [
        "Execute the deployment pipeline for {task} and verify all steps complete.",
        "Run the data processing job for {topic} and save results to the database.",
        "Automate the {task} process and set up monitoring.",
    ],
}

TASK_FILL_VALUES = {
    "topic": [
        "renewable energy trends", "machine learning model optimization",
        "global supply chain disruptions", "urban traffic patterns",
        "social media sentiment", "customer churn prediction",
        "protein folding methods", "climate change indicators",
    ],
    "task": [
        "processes CSV files and generates summary statistics",
        "scrapes product prices from e-commerce websites",
        "converts JSON data to a normalized database schema",
        "implements a caching layer for API responses",
        "analyzes log files for error patterns",
    ],
    "metric": ["ROI", "accuracy", "latency", "throughput", "conversion rate"],
}


class WorldModel:
    """Simulates tool execution for agent self-play.

    Generates plausible tool outputs without actually executing anything.
    """

    def __init__(self, model: str = "gpt-4o-mini", workers: int = 4) -> None:
        self.model = model
        self.workers = workers
        self._api_client = None

    @property
    def api_client(self):
        if self._api_client is None:
            from fanno.api.client import AzureAPIClient
            self._api_client = AzureAPIClient(
                model_name=self.model, workers=self.workers, max_tokens=512
            )
        return self._api_client

    def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: str = "",
    ) -> str:
        """Simulate tool execution and return plausible output."""
        prompt = AGENT_WORLD_MODEL_PROMPT.format(
            tool_name=tool_name,
            arguments=json.dumps(arguments, indent=2),
            context=context[:500],
        )
        result = self.api_client.batch_chat([prompt], max_tokens=512)
        return result[0] if result else f"Error: {tool_name} returned no output"

    def execute_batch(
        self,
        calls: List[Dict[str, Any]],
        context: str = "",
    ) -> List[str]:
        """Batch simulate multiple tool calls."""
        prompts = [
            AGENT_WORLD_MODEL_PROMPT.format(
                tool_name=call["name"],
                arguments=json.dumps(call.get("arguments", {}), indent=2),
                context=context[:500],
            )
            for call in calls
        ]
        return self.api_client.batch_chat(prompts, max_tokens=512)


class AgentSynthesizer(BaseSynthesizer):
    """Generate agent trajectories with function-calling.

    Produces training data in OpenAI function-calling message format,
    suitable for training models on tool-use and agent behavior.
    """

    AGENT_ROLES = AGENT_ROLES
    LOGIC_PATTERNS = LOGIC_PATTERNS

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        workers: int = 8,
        max_turns: int = 8,
    ) -> None:
        super().__init__(model=model, workers=workers)
        self.max_turns = max_turns
        self.world_model = WorldModel(model=model, workers=workers)

    def generate_scenario(
        self,
        role: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a task scenario with available tools.

        Args:
            role: Agent role (random if None).
            pattern: Logic pattern (random if None).

        Returns:
            Dict with 'task', 'role', 'pattern', 'tools'.
        """
        role = role or random.choice(self.AGENT_ROLES)
        pattern = pattern or random.choice(self.LOGIC_PATTERNS)

        # Select tools for this role
        tool_names = ROLE_TOOL_MAPPING.get(role, ["web_search", "code_execute"])
        tools = [TOOL_LIBRARY[name] for name in tool_names if name in TOOL_LIBRARY]

        # Generate task description
        template = random.choice(ROLE_TASK_TEMPLATES.get(role, ROLE_TASK_TEMPLATES["executor"]))
        fill = {}
        for key, values in TASK_FILL_VALUES.items():
            if f"{{{key}}}" in template:
                fill[key] = random.choice(values)
        task = template.format_map({**{k: "" for k in ["topic", "task", "metric"]}, **fill})

        return {
            "task": task,
            "role": role,
            "pattern": pattern,
            "tools": tools,
            "tool_names": tool_names,
        }

    def generate_trajectory(
        self,
        scenario: Dict[str, Any],
        max_turns: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run agent self-play to produce a function-calling trajectory.

        Uses the WorldModel to simulate tool execution results.

        Args:
            scenario: From generate_scenario().
            max_turns: Max number of agent turns.

        Returns:
            Dict with 'messages', 'tools', 'metadata'.
        """
        max_turns = max_turns or self.max_turns
        task = scenario["task"]
        tools = scenario["tools"]
        tool_names = scenario["tool_names"]

        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    f"You are an AI agent with the role of {scenario['role']}. "
                    f"You have access to the following tools: {', '.join(tool_names)}. "
                    "Use tools to accomplish the task. When you're done, provide a final summary."
                ),
            },
            {"role": "user", "content": task},
        ]

        # Generate the agent's plan first
        plan_prompt = AGENT_PLANNING_PROMPT.format(
            role=scenario["role"],
            task=task,
            tools=", ".join(tool_names),
        )
        plan = self.api_client.batch_chat([plan_prompt], max_tokens=512)
        context = plan[0] if plan else ""

        for turn in range(max_turns):
            # Ask agent for next action
            history_text = self._format_messages(messages)
            action_prompt = AGENT_EXECUTION_PROMPT.format(
                task=task,
                step=turn + 1,
                previous_results=history_text[-1000:],
                tools=", ".join(tool_names),
            )
            action_response = self.api_client.batch_chat([action_prompt], max_tokens=256)
            action_text = action_response[0] if action_response else ""

            # Try to parse function call
            function_call = self._parse_function_call(action_text, tool_names)

            if function_call:
                # Agent makes a tool call
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": function_call["name"],
                        "arguments": json.dumps(function_call.get("arguments", {})),
                    },
                })

                # Simulate tool execution
                tool_output = self.world_model.execute(
                    function_call["name"],
                    function_call.get("arguments", {}),
                    context=context,
                )
                messages.append({
                    "role": "tool",
                    "content": tool_output,
                    "name": function_call["name"],
                })
            else:
                # Agent provides final response (no more tool calls)
                messages.append({
                    "role": "assistant",
                    "content": action_text,
                })
                break

        return {
            "messages": messages,
            "tools": tools,
            "metadata": {
                "role": scenario["role"],
                "pattern": scenario["pattern"],
                "task": task,
                "num_turns": len([m for m in messages if m["role"] == "assistant"]),
            },
        }

    def _parse_function_call(
        self,
        text: str,
        valid_tools: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Try to parse a function call from agent response."""
        # Try JSON extraction
        try:
            # Look for JSON object in the text
            import re
            json_match = re.search(r'\{[^{}]*"name"[^{}]*\}', text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                if parsed.get("name") in valid_tools:
                    return parsed
        except (json.JSONDecodeError, AttributeError):
            pass

        # Try structured extraction
        for tool_name in valid_tools:
            if tool_name in text.lower():
                return {
                    "name": tool_name,
                    "arguments": self._extract_arguments(text, tool_name),
                }

        return None

    def _extract_arguments(self, text: str, tool_name: str) -> Dict[str, Any]:
        """Extract plausible arguments for a tool call from text."""
        tool_def = TOOL_LIBRARY.get(tool_name, {})
        params = tool_def.get("function", {}).get("parameters", {}).get("properties", {})
        args: Dict[str, Any] = {}
        for param_name, param_info in params.items():
            if param_info.get("type") == "string":
                # Try to find quoted strings or relevant content
                import re
                quotes = re.findall(r'"([^"]+)"', text)
                if quotes:
                    args[param_name] = quotes[0]
                elif "required" in str(tool_def) and param_name in str(tool_def.get("function", {}).get("parameters", {}).get("required", [])):
                    args[param_name] = f"example_{param_name}"
        return args

    def _format_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages as text for context."""
        lines: List[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            if msg.get("function_call"):
                content = f"[Called {msg['function_call']['name']}]"
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def generate(self, num_samples: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Batch generate agent trajectories.

        Args:
            num_samples: Number of trajectories to generate.

        Returns:
            List of trajectory dicts in OpenAI function-calling format.
        """
        logger.info(f"Generating {num_samples} agent trajectories")
        data: List[Dict[str, Any]] = []

        for i in range(num_samples):
            try:
                scenario = self.generate_scenario()
                trajectory = self.generate_trajectory(scenario)
                data.append(trajectory)
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated {i + 1}/{num_samples} trajectories")
            except Exception as e:
                logger.warning(f"Failed to generate trajectory {i}: {e}")
                continue

        logger.info(f"Generated {len(data)}/{num_samples} agent trajectories")
        return data

    def validate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate agent trajectories."""
        valid: List[Dict[str, Any]] = []
        for item in data:
            messages = item.get("messages", [])
            # Must have at least system + user + assistant (with tool call) + tool + assistant
            if len(messages) < 4:
                continue
            # Must have at least one function call
            has_function_call = any(m.get("function_call") for m in messages)
            if not has_function_call:
                continue
            # Must end with assistant response
            if messages[-1]["role"] != "assistant":
                continue
            valid.append(item)
        return valid


__all__ = ["AgentSynthesizer", "WorldModel", "TOOL_LIBRARY"]
