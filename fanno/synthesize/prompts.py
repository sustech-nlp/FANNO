"""Synthesis prompt templates for all data types."""

from __future__ import annotations


# === QA Synthesis Prompts ===

QA_GENERAL_PROMPT = """Generate a challenging question based on the following topic. The question should:
1. Require multi-step reasoning or deep domain knowledge
2. Be self-contained (no references to external documents)
3. Be answerable by a knowledgeable AI assistant
4. Be specific and well-defined

Topic: {topic}

Question:"""

QA_CODE_PROMPT = """Generate a coding question at {difficulty} level. The question should:
1. Involve {language} programming
2. Test understanding of {concept}
3. Include clear input/output specifications
4. Be solvable in 20-50 lines of code

Question:"""

QA_MATH_PROMPT = """Generate a mathematical problem at {difficulty} level. The problem should:
1. Involve {topic}
2. Require step-by-step reasoning
3. Have a definitive numerical or symbolic answer
4. Be expressed clearly in natural language

Problem:"""

QA_REASONING_PROMPT = """Generate a complex reasoning question that requires:
1. Logical deduction from given premises
2. Analysis of multiple perspectives
3. Synthesis of information across {domain}

The question should be challenging but answerable with careful thinking.

Question:"""

# === Answer Generation Prompts ===

ANSWER_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.

### Instruction:
{instruction}

### Response:"""

ANSWER_WITH_CONTEXT_PROMPT = """You are a helpful, respectful and honest assistant. Answer the question based on the given context.

### Context:
{context}

### Instruction:
{instruction}

### Response:"""

# === Creative Writing Prompts ===

CREATIVE_STORY_PROMPT = """Write a {genre} short story with the following elements:
- Setting: {setting}
- Theme: {theme}
- Length: approximately {length} words
- Tone: {tone}

Begin the story:"""

CREATIVE_ESSAY_PROMPT = """Write a thoughtful essay on the following topic:

Topic: {topic}
Style: {style}
Perspective: {perspective}
Target length: {length} words

Essay:"""

CREATIVE_INSTRUCTION_PROMPT = """Generate a creative writing task that asks someone to write {type}.
The task should be specific enough to guide writing but open enough to allow creativity.
Include any constraints (word count, style, audience) in the instruction.

Task instruction:"""

# === Dialog Prompts ===

DIALOG_SCENARIO_PROMPT = """Generate a realistic multi-turn conversation scenario between a user and an AI assistant.

Topic: {topic}
Complexity: {complexity}
Number of turns: {num_turns}

The conversation should feel natural, with the user asking follow-up questions
and the assistant providing helpful, detailed responses.

Begin the conversation:"""

DIALOG_USER_TURN_PROMPT = """You are simulating a user in a conversation with an AI assistant.
Given the conversation history below, generate a natural follow-up message from the user.

Conversation so far:
{history}

The user's next message should:
1. Be natural and conversational
2. Ask for clarification, more detail, or a related follow-up
3. Stay on topic but explore new angles

User:"""

DIALOG_ASSISTANT_TURN_PROMPT = """You are a helpful AI assistant. Given the conversation history below,
provide a helpful and detailed response.

Conversation so far:
{history}

Respond helpfully and thoroughly:"""

# === Agent Prompts ===

AGENT_SCENARIO_PROMPT = """Generate a realistic task scenario for an AI agent with the role of {role}.

The task should:
1. Require using multiple tools to complete
2. Follow a {pattern} execution pattern
3. Be specific and actionable
4. Include clear success criteria

Available tools: {tools}

Generate the task description and expected tool usage plan:"""

AGENT_PLANNING_PROMPT = """You are an AI agent with the role of {role}.

Task: {task}
Available tools: {tools}

Plan your approach step by step. For each step:
1. Describe what you need to do
2. Which tool to use and with what arguments
3. What you expect to get back
4. How to use the result in the next step

Plan:"""

AGENT_EXECUTION_PROMPT = """You are an AI agent executing a plan. Based on the current state:

Task: {task}
Current step: {step}
Previous results: {previous_results}
Available tools: {tools}

Decide the next action. Respond with a function call in the format:
{{
    "name": "<tool_name>",
    "arguments": {{...}}
}}

Action:"""

AGENT_WORLD_MODEL_PROMPT = """You are simulating a tool execution environment.

Tool called: {tool_name}
Arguments: {arguments}
Context: {context}

Generate a realistic response that the tool would return. Be specific and include
plausible data, error messages, or results as appropriate.

Tool output:"""

# === Trajectory Inversion Prompts ===

INVERSION_EXTRACT_PROMPT = """Analyze the following agent trajectory and extract the key decision points.

Trajectory:
{trajectory}

For each decision point, identify:
1. What information was available
2. What decision was made
3. What alternatives could have been chosen

Decision points:"""

INVERSION_GENERATE_PROMPT = """Based on the following completed agent trajectory, generate a new but related task scenario.

Original trajectory summary:
{summary}

The new scenario should:
1. Involve similar tools but a different goal
2. Require a different execution pattern
3. Be at a similar difficulty level
4. Be self-contained and actionable

New scenario:"""


__all__ = [
    "QA_GENERAL_PROMPT",
    "QA_CODE_PROMPT",
    "QA_MATH_PROMPT",
    "QA_REASONING_PROMPT",
    "ANSWER_PROMPT",
    "ANSWER_WITH_CONTEXT_PROMPT",
    "CREATIVE_STORY_PROMPT",
    "CREATIVE_ESSAY_PROMPT",
    "CREATIVE_INSTRUCTION_PROMPT",
    "DIALOG_SCENARIO_PROMPT",
    "DIALOG_USER_TURN_PROMPT",
    "DIALOG_ASSISTANT_TURN_PROMPT",
    "AGENT_SCENARIO_PROMPT",
    "AGENT_PLANNING_PROMPT",
    "AGENT_EXECUTION_PROMPT",
    "AGENT_WORLD_MODEL_PROMPT",
    "INVERSION_EXTRACT_PROMPT",
    "INVERSION_GENERATE_PROMPT",
]
