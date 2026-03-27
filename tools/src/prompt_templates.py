import json
from typing import List


def _get_pattern_implementation_guide(pattern: str) -> str:
    guides = {
        "smooth": """
- All tool calls should succeed sequentially
- User progresses naturally from request to completion
- Final tool should confirm/finalize the action
""",
        "partial_failure": """
- Include at least ONE tool that returns incomplete/ambiguous results
- User should need to provide clarification or make a choice
- System should handle the partial information gracefully
""",
        "error_recovery": """
- At least ONE tool should fail completely
- System should acknowledge the error and suggest alternatives
- A different tool or approach should eventually succeed
""",
        "escalation": """
- Problem should be beyond system's capability to fully resolve
- Include a "transfer" or "escalate" tool at the end
- User should explicitly request or agree to escalation
""",
        "user_change_mind": """
- Include tools for modification (update, cancel, change)
- User should change requirements mid-conversation
- System should adapt by calling different tools
""",
        "multi_goal": """
- Break the main goal into 2-3 sub-goals
- Each sub-goal uses different tools
- Clear transitions between sub-goals
""",
    }
    return guides.get(pattern, "Follow a natural conversation flow")


def build_scenario_prompt(seed_doc: str, config, logic_patterns) -> str:
    min_tool_usage = int(config.num_tools * 0.6)
    return f"""
You are an expert scenario designer for tool-augmented dialogue datasets.

# TASK
Create a realistic conversation scenario based on the seed document below. 
Use your creativity to infer a suitable domain and design an engaging interaction flow.

# SEED DOCUMENT
{seed_doc}

# REQUIREMENTS
- Number of tools: {config.num_tools}
- Expected conversation turns: {config.num_turns}
- Logic pattern: {config.logic_pattern} ({logic_patterns[config.logic_pattern]})
{f"- Domain hint: {config.domain_hint}" if getattr(config, "domain_hint", None) else "- Infer the domain from the seed document"}

## Tool Sequence Design Requirements
Design a realistic tool call sequence that:
1. Uses at least {min_tool_usage} out of {config.num_tools} defined tools
2. Shows logical progression (simple → complex, query → action)
3. Implements the "{config.logic_pattern}" pattern naturally

For "{config.logic_pattern}" pattern, the sequence should:
{_get_pattern_implementation_guide(config.logic_pattern)}

## Expected Conversation Flow
Provide a high-level outline of how the conversation should progress:
{{
  "conversation_outline": [
    {{"phase": "initial_request", "turns": "1-2", "tools": []}},
    {{"phase": "information_gathering", "turns": "2-4", "tools": ["tool1", "tool2"]}},
    {{"phase": "action_execution", "turns": "2-3", "tools": ["tool3"]}},
    {{"phase": "confirmation", "turns": "1-2", "tools": ["tool4"]}}
  ],
  "expected_observation_diversity": [
    {{"status": "success", "percentage": 0.7}},
    {{"status": "partial_failure", "percentage": 0.2}},
    {{"status": "error", "percentage": 0.1}}
  ]
}}

# OUTPUT FORMAT (MUST BE VALID JSON)
{{
  "system": "A detailed system prompt (300-600 words) that includes:
    1. Current date/time context
    2. Agent's role and capabilities
    3. Domain-specific policies and rules
    4. User's situation and goal
    5. Workflow constraints (e.g., must confirm before taking actions)
    6. Any edge cases or special conditions
    
    Example structure:
    'The current time is 2024-05-15 15:00:00 EST.
    
    As a [domain] agent, you can help users [capabilities].
    - [Policy 1]
    - [Policy 2]
    ...
    
    Scenario: [User's situation based on seed document]
    Goal: [What user wants to achieve]'", 
  
  "tools": [
    {{
      "name": "tool_name",
      "description": "Clear description of what this tool does",
      "parameters": {{
        "type": "object",
        "properties": {{
          "param1": {{
            "type": "string",
            "description": "What this parameter means, such as 'product_id like \\"prod_123\\"'"
          }},
          "param2": {{
            "type": "integer",
            "description": "Number of items, such as 5"
          }}
        }},
        "required": ["param1"]
      }}
    }}
    // Design exactly {config.num_tools} tools
  ],
  
  "meta": {{
    "domain": "Inferred domain (e.g., 'airline', 'ecommerce', 'banking', 'tech_support')",
    "logic_pattern": "{config.logic_pattern}",
    "expected_user_goal": "What the user wants to achieve (e.g., 'Book a flight with specific requirements')",
    "potential_issues": [
      "Issue 1 that might occur based on the logic pattern",
      "Issue 2 that might occur"
    ],
    "expected_tool_sequence": ["tool1", "tool2", "tool3"],
    "success_criteria": "How to determine if the conversation succeeded",
    "conversation_outline": [
      {{"phase": "greeting", "turns": "1", "tools": []}},
      {{"phase": "search", "turns": "1-2", "tools": ["search_x"]}},
      {{"phase": "details", "turns": "1-2", "tools": ["get_details"]}},
      {{"phase": "action", "turns": "1-2", "tools": ["execute_y"]}},
      {{"phase": "confirmation", "turns": "1", "tools": ["confirm_z"]}}
    ],
    "min_tool_usage_count": {min_tool_usage},
    "expected_observation_diversity": true,
    "estimated_turns": "8-12"
  }}
}}

# DESIGN GUIDELINES

## System Prompt Design
- Start with current date/time (use realistic values)
- Define agent's role and scope clearly
- Include 3-5 specific policies or rules relevant to the domain
- Describe user's situation naturally (inspired by seed document)
- Add realistic constraints (e.g., "must obtain confirmation before...", "cannot modify after...")
- Make it detailed enough (300-600 words) to guide the conversation flow

## Tool Design Principles
- Each tool should have ONE clear, atomic purpose
- Parameters must include type, description, and examples
- Use enums for categorical fields (e.g., {{"type": "string", "enum": ["option1", "option2"]}})
- Include diverse tool types:
  * Query tools (get information)
  * Action tools (modify state)
  * Verification tools (check conditions)
  * Utility tools (calculate, think, etc.)
  * Escalation tools (transfer_to_human, etc.)

## Tool Sequence Based on Logic Pattern
- "smooth": Design tools that work together seamlessly
- "partial_failure": Include a tool that might return incomplete results
- "error_recovery": Include alternative tools for achieving the same goal
- "escalation": Include a "transfer_to_human_agents" tool
- "user_change_mind": Include tools for modifying previous actions
- "multi_goal": Design tools for different sub-tasks

## Common Tool Patterns
Always consider including:
1. A "get_details" or "lookup" tool (retrieves current state)
2. A "search" or "list" tool (finds available options)
3. An action tool (books, creates, updates, cancels)
4. A "think" tool (for reasoning before acting)
5. A "transfer_to_human_agents" tool (for escalation)

## Examples of Good Tool Names
- get_reservation_details, search_flights, book_reservation, cancel_reservation
- get_order_status, search_products, create_order, process_refund
- get_account_balance, transfer_funds, verify_identity
- query_database, update_record, send_notification

# CRITICAL REQUIREMENTS
1. The "system" field is THE MOST IMPORTANT - make it detailed and realistic
2. All tools must strictly follow OpenAI function calling schema
3. Tools must be logically connected (output of one can feed into another)
4. The scenario must be resolvable within {config.num_turns} turns
5. Parameter descriptions must include concrete examples

Now generate the scenario. Think creatively about how the seed document relates to a real-world task.
"""


def build_evaluation_prompt(scenario: dict, tools: list, conversation_history: list) -> str:
    system = scenario.get("system", "")
    meta = scenario.get("meta", {})
    return f"""
Evaluate the quality of this conversation scenario on a 0-10 scale.

# SCENARIO
System Prompt:
{system}

Tools:
{json.dumps(tools, indent=2)}

Metadata:
{json.dumps(meta, indent=2)}

{"Conversation History (if any):" if conversation_history else ""}
{json.dumps(conversation_history[-6:], indent=2) if conversation_history else ""}

# SCORING CRITERIA

1. Scenario Realism and Complexity (0-3 points)
   - Is the scenario realistic and believable?
   - Does it have appropriate complexity?
   - Does the system prompt provide clear context and policies?
   - Are the user's goals and potential issues well-defined?

2. Tool Design Quality (0-3 points)
   - Are tools well-defined with proper schemas?
   - Do tools have clear, atomic purposes?
   - Are parameter descriptions helpful with examples?
   - Do tools logically connect to each other?
   - Are all required fields properly marked?

3. Conversation Coherence (0-2 points)
   - Does the scenario enable natural conversation flow?
   - Are the logic patterns appropriately implemented?
   - Is the expected tool sequence reasonable?

4. Edge Cases and Error Handling (0-2 points)
   - Does the scenario account for potential failures?
   - Are there appropriate escalation mechanisms?
   - Does it handle the specified logic pattern correctly?

# OUTPUT FORMAT (JSON ONLY)
{{
  "score": <integer 0-10>,
  "breakdown": {{
    "realism_complexity": <0-3>,
    "tool_quality": <0-3>,
    "coherence": <0-2>,
    "error_handling": <0-2>
  }},
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "reason": "Brief explanation of the score"
}}

Evaluate now:
"""


def build_conversation_evaluation_prompt(scenario: dict, tools: list, conversations: list) -> str:
    return f"""
Evaluate this completed conversation strictly.

SCENARIO DESIGN:
{json.dumps(scenario.get('meta', {}), indent=2)}

DEFINED TOOLS:
{json.dumps([t['name'] for t in tools], indent=2)}

ACTUAL CONVERSATION:
{_format_conversation_history(conversations)}

EVALUATION CRITERIA:

1. Tool Usage Rate (0-3 points)
   Count how many unique tools were actually called.
   - Used >= 60% of defined tools: 3 points
   - Used >= 40% of defined tools: 2 points  
   - Used >= 20% of defined tools: 1 point
   - Used < 20% of defined tools: 0 points

2. Observation Diversity (0-2 points)
   Check the variety of observation statuses.
   - Has at least 2 different statuses (success/partial_failure/error): 2 points
   - All observations have the same status: 0 points

3. Conversation Structure (0-3 points)
   - Has clear beginning (user request): 1 point
   - Has middle (tool calls and responses): 1 point
   - Has clear ending (resolution/transfer/user satisfaction): 1 point

4. Logic Pattern Adherence (0-2 points)
   - Conversation clearly demonstrates the intended pattern: 2 points
   - Pattern partially visible: 1 point
   - Pattern not evident: 0 points

AUTOMATIC REJECTION if:
- Fewer than 6 conversation turns
- Used less than 2 tools (when 3+ tools were defined)
- All observations are identical
- No clear conversation ending

Output JSON:
{{
  "score": <0-10>,
  "breakdown": {{
    "tool_usage_rate": <0-3>,
    "observation_diversity": <0-2>,
    "conversation_structure": <0-3>,
    "logic_pattern": <0-2>
  }},
  "statistics": {{
    "tools_defined": {len(tools)},
    "tools_used": "<count>",
    "tool_usage_percentage": "<0-1>",
    "observation_statuses": [],
    "conversation_turns": "<count>"
  }},
  "reject_reasons": [],
  "verdict": "ACCEPT" | "REJECT"
}}
"""


def build_execution_prompt(function_call: dict, system_prompt: str, tools: list, conversation_history: list, execution_history: list, target_status: str | None = None) -> str:
    tool_name = function_call.get("name")
    arguments = function_call.get("arguments", {})
    tool_def = None
    for tool in tools:
        if tool.get("name") == tool_name:
            tool_def = tool
            break
    return f"""
You are simulating the execution of a tool call in a conversation.
Generate a realistic observation (tool execution result) based on the context.

# SYSTEM CONTEXT
{system_prompt}

# TOOL BEING CALLED
Name: {tool_name}
Definition:
{json.dumps(tool_def, indent=2) if tool_def else "Tool definition not found"}

# FUNCTION CALL
{json.dumps(function_call, indent=2)}

# CONVERSATION SO FAR
{_format_conversation_history(conversation_history[-8:])}

# PREVIOUS TOOL EXECUTIONS
{json.dumps(execution_history[-3:], indent=2) if execution_history else "None"}

TARGET STATUS SUGGESTION: {target_status or 'any'}
If suggestion is provided:
- success: execute completely with expected data
- partial_failure: return incomplete/ambiguous data or require clarification
- error: fail due to invalid input, system issue, or policy

# OUTPUT FORMAT (JSON ONLY)
{{
  "status": "success" | "partial_failure" | "error",
  "data": {{
    // Realistic data based on the tool's purpose
    // For search/list: return array of results with IDs
    // For get/fetch: return detailed object
    // For create/book: return new ID and confirmation
    // For update/modify: return updated status
    // For cancel/delete: return confirmation
  }},
  "error": "Error message (only if status is 'error' or 'partial_failure')"
}}

Generate the observation now. Make it realistic and contextually appropriate:
"""


def _format_conversation_history(history: list) -> str:
    lines = []
    for turn in history:
        role = turn.get("from")
        value = turn.get("value")
        if role == "function_call":
            try:
                fc = json.loads(value)
                lines.append(f"[TOOL CALL] {fc['name']}({json.dumps(fc.get('arguments', {}))})")
            except Exception:
                lines.append(f"[TOOL CALL] {value}")
        elif role == "observation":
            lines.append(f"[OBSERVATION] {value[:150]}...")
        else:
            lines.append(f"[{str(role).upper()}] {value}")
    return "\n".join(lines)


def build_user_prompt(system_prompt: str, conversation_history: list, goal: str, world_state: dict) -> str:
    last_observation = None
    for turn in reversed(conversation_history):
        if turn.get("from") == "observation":
            try:
                last_observation = json.loads(turn.get("value", "{}"))
            except Exception:
                pass
            break
    return f"""
You are simulating a realistic user in a conversation with an AI assistant.

# SCENARIO
{system_prompt}

# YOUR GOAL
{goal}

# CONVERSATION SO FAR
{_format_conversation_for_user(conversation_history[-8:])}

# LAST TOOL RESULT
{json.dumps(last_observation, indent=2) if last_observation else "No tool result yet"}

# CURRENT STATE
{json.dumps(world_state, indent=2) if world_state else "No state information"}

# YOUR TASK
Generate the user's next natural response based on the context.

# GUIDELINES
- If tool succeeded and goal is achieved: express satisfaction and end (e.g., "Perfect, thank you!")
- If tool succeeded but goal incomplete: acknowledge and ask to continue naturally
- If tool failed: describe the issue or request alternative approach
- If stuck or confused: ask clarifying questions
- If you notice issues: point them out (e.g., "Wait, that price seems too high")
- Sometimes add new requirements naturally (e.g., "Can I also add insurance?")
- Use natural, varied language - avoid repetitive phrases
- Keep responses conversational and realistic (1-3 sentences usually)
- Show personality: be polite but can be impatient, excited, concerned, etc.

# RESPONSE PATTERNS TO AVOID
- "Thanks. Please continue." (too robotic)
- "That sounds good. What's next?" (too generic)
- Overly long explanations
- Repeating what the assistant just said

# GOOD RESPONSE EXAMPLES
- "Great! Can you also check if they have vegetarian options?"
- "Hmm, that flight is too early. Do you have anything after 2 PM?"
- "Wait, I thought you said the refund would be full? Why is there a fee?"
- "Perfect! Please go ahead and book it."
- "Actually, I changed my mind. Can we look at business class instead?"

Generate ONLY the user's response (no labels, no JSON, no extra commentary):
"""


def _format_conversation_for_user(history: list) -> str:
    lines = []
    for turn in history:
        role = turn.get("from")
        value = turn.get("value")
        if role == "human":
            lines.append(f"User: {value}")
        elif role == "gpt":
            lines.append(f"Assistant: {value}")
        elif role == "function_call":
            try:
                fc = json.loads(value)
                lines.append(f"[Assistant calls tool: {fc['name']}]")
            except Exception:
                lines.append("[Assistant calls a tool]")
        elif role == "observation":
            try:
                obs = json.loads(value)
                status = obs.get("status", "unknown")
                lines.append(f"[Tool returned: {status}]")
            except Exception:
                lines.append("[Tool returned a result]")
    return "\n".join(lines)


def build_initial_query_prompt(system: str, meta: dict) -> str:
    return f"""
Based on this scenario, generate a realistic initial user query.

SCENARIO:
{system}

USER GOAL:
{meta.get('expected_user_goal', 'Complete a task')}

POTENTIAL CONTEXT:
{json.dumps(meta.get('potential_issues', []), indent=2)}

Generate ONLY the user's opening message (natural language, no labels, no JSON):
"""


def build_decide_action_prompt(system: str, tools: list, conversations: list) -> str:
    return f"""
You are the AI assistant in this conversation. Decide your next action.

SYSTEM:
{system}

AVAILABLE TOOLS:
{json.dumps([{"name": t["name"], "description": t["description"]} for t in tools], indent=2)}

CONVERSATION SO FAR:
{_format_conversations(conversations[-6:])}

Should you call a tool or respond directly to the user?

Guidelines:
- Call a tool if you need information or need to take an action
- Respond directly if you're asking for clarification or confirming with the user
- After getting an observation, you usually respond to the user
- Before taking important actions, you might want to confirm with the user first

Output as JSON:
{{
  "action": "call_tool" or "respond_directly",
  "reason": "Brief explanation",
  "response": "Your direct response (only if action is 'respond_directly')"
}}
"""


def build_function_call_prompt(system: str, tools: list, conversations: list, meta: dict) -> str:
    return f"""
You are the AI assistant. Generate the next tool call.

SYSTEM:
{system}

TOOLS:
{json.dumps(tools, indent=2)}

CONVERSATION:
{_format_conversations(conversations[-6:])}

EXPECTED TOOL SEQUENCE (HINT):
{meta.get('expected_tool_sequence', [])}

Generate a valid function call that makes sense in this context.

Output as JSON (function call format):
{{
  "name": "tool_name",
  "arguments": {{
    "param1": "value1",
    "param2": "value2"
  }}
}}

Make sure:
1. The tool name exists in the available tools
2. All required parameters are provided
3. Parameter values are realistic and contextually appropriate
4. If previous tools returned IDs or values, use them appropriately
"""


def build_gpt_response_prompt(system: str, tools: list, conversations: list) -> str:
    last_observation = None
    for turn in reversed(conversations[-3:]):
        if turn.get("from") == "observation":
            try:
                last_observation = json.loads(turn.get("value", "{}"))
            except Exception:
                last_observation = {}
            break
    return f"""
You are the assistant. Generate your response based on the tool result.

CRITICAL RULES TO PREVENT HALLUCINATION:
1. ✅ ONLY mention information that appears in the tool result below
2. ❌ DO NOT invent activities, products, or entities not in the result
3. ❌ DO NOT mention options that were not returned by the tool
4. ✅ If the tool returned 2 items, mention ONLY those 2 items
5. ✅ If you want to suggest other types, you MUST call search tool first

SYSTEM:
{system}

CONVERSATION:
{_format_conversations(conversations[-8:])}

LAST TOOL RESULT (YOUR ONLY SOURCE OF TRUTH):
{json.dumps(last_observation, indent=2) if last_observation else 'No tool result yet'}

VALIDATION CHECKLIST before responding:
- [ ] Am I only using data from the tool result above?
- [ ] Am I making up any activity names or IDs?
- [ ] If I want to suggest something else, did I search for it?

Output ONLY your response text (no labels, no JSON):
"""


def build_completion_check_prompt(conversations: list, meta: dict) -> str:
    return f"""
Has this conversation achieved its goal?

GOAL: {meta.get('expected_user_goal')}
SUCCESS CRITERIA: {meta.get('success_criteria', 'User satisfied and task complete')}

RECENT CONVERSATION:
{_format_conversations(conversations[-6:])}

Answer with JSON:
{{"is_complete": true/false, "reason": "Brief explanation"}}
"""


def _format_conversations(convs: list) -> str:
    lines = []
    for c in convs:
        role = c.get("from")
        value = c.get("value")
        if role == "function_call":
            try:
                fc = json.loads(value)
                lines.append(f"[TOOL CALL] {fc['name']}({json.dumps(fc.get('arguments', {}))})")
            except Exception:
                lines.append(f"[TOOL CALL] {value[:100]}...")
        elif role == "observation":
            lines.append(f"[OBSERVATION] {value[:200]}...")
        else:
            lines.append(f"[{str(role).upper()}] {value}")
    return "\n".join(lines)


__all__ = [
    "build_scenario_prompt",
    "build_evaluation_prompt",
    "build_conversation_evaluation_prompt",
    "build_execution_prompt",
    "build_user_prompt",
    "build_initial_query_prompt",
    "build_decide_action_prompt",
    "build_function_call_prompt",
    "build_gpt_response_prompt",
    "build_completion_check_prompt",
]
