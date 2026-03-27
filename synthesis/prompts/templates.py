"""
Prompt templates for all data synthesis types.
Covers: QA, Complex QA, VQA, Multi-turn Dialog, Trajectory Inversion,
        Code, Math, Creative Writing, Reasoning, etc.
"""
from __future__ import annotations

import random
from typing import List, Dict, Optional


# =============================================================================
# 1. FANNO-Style Seed QA Generation (from document -> question)
# =============================================================================

DIFFICULTY_TAGS = [
    "It should be complex and requires multiple-step reasoning to solve.",
    "It demands critical thinking skills to analyze from various perspectives and evaluate multiple solutions.",
    "It necessitates creative thinking to devise innovative solutions beyond conventional approaches.",
    "It demands integrating knowledge from diverse disciplines to address its multifaceted nature.",
    "It requires deep domain expertise and nuanced understanding to answer correctly.",
]

TASK_TYPE_TAGS = [
    "It is a Natural language inference question: Assessing if evidence supports a conclusion.",
    "It is a Commonsense question: Predicting outcomes based on everyday knowledge.",
    "It is a Sentiment analysis question: Determining emotional response to a given scenario.",
    "It is a Paraphrasing question: Rewording a statement while retaining its meaning.",
    "It is a Close-book QA question: Answering factual queries using pre-existing knowledge.",
    "It is a Structure to text question: Describing a process or concept in written form.",
    "It is a Summarization question: Condensing key information from a larger text.",
    "It is a Translation question: Converting text from one language to another.",
    "It is a Implicit reasoning question: Inferring reasons behind common behaviors.",
    "It is a Text categorization question: Identifying defining characteristics of a given text type.",
    "It is a Code generation question: Generate a LeetCode easy-level coding question.",
    "It is a gsm8k math question: Generate a question designed for elementary school-level mathematical problem-solving.",
    "It is an Open-ended analysis question: Requiring comprehensive analysis with supporting arguments.",
    "It is a Comparison question: Comparing and contrasting multiple concepts, approaches, or entities.",
    "It is a Hypothetical reasoning question: Exploring what-if scenarios and their implications.",
    "It is a Multi-step problem solving question: Breaking down complex problems into sequential steps.",
]

STYLE_TAGS = [
    "It should be in the style of a command or imperative. For example, 'Write a paragraph about...' or 'Describe the...'",
    "It should be in the style of a question or interrogative. For example, 'What is the...?' or 'How do you...?'",
    "It should be in a conversational style, as if asking a knowledgeable friend.",
    "It should be in a formal academic style, suitable for research contexts.",
]


def seed_qa_prompt(text: str, difficulty: str = None, task_type: str = None, style: str = None) -> str:
    """Generate a seed QA prompt from a document. Extended FANNO seed generation."""
    if difficulty is None:
        difficulty = random.choice(DIFFICULTY_TAGS)
    if task_type is None:
        task_type = random.choice(TASK_TYPE_TAGS)
    if style is None:
        style = random.choice(STYLE_TAGS)

    return f"""You're proficient in crafting complex questions. Generate only one question that adheres to the provided ### Paragraph.
The question should meet the following criteria:
0. The person answering the question cannot see the ### Paragraph, so the question must not contain phrases like 'Given the information provided', 'Based on the provided information', or similar expressions that imply direct citations or references from ### Paragraph.
1. {difficulty}
2. {style}
3. {task_type}

[IMPORTANT] If you do not follow ALL of the above rules, your response will be rejected.

### Paragraph:
{text}
### Question:"""


def generate_all_seed_prompts(text: str) -> List[str]:
    """Generate diverse prompts covering all combinations (sampled for efficiency)."""
    prompts = []
    # Full combination would be 5*16*4 = 320 per doc, too many.
    # Sample 20 diverse combinations per document.
    for _ in range(20):
        prompts.append(seed_qa_prompt(text))
    return prompts


# =============================================================================
# 2. FANNO Think-Different (UCB augmentation)
# =============================================================================

def think_different_prompt(seed1: str, seed2: str, seed3: str, text: str) -> str:
    """FANNO's core Think-Different prompt with counterexamples."""
    style = random.choice([
        "Be in the style of a command or imperative. Avoid using who, what, when, where, why, or how.",
        "Be in the style of a question or interrogative.",
        "Be conversational and natural, as if chatting with an expert.",
    ])
    return f"""Your task is to craft a **unique and thought-provoking query** based on the paragraph below, without resembling the style or structure of the examples provided.

### Counterexamples:
1. {seed1}
2. {seed2}
3. {seed3}

### Paragraph:
{text}

### Response:
My query should:
1. Be meaningful and well-structured, with proper punctuation.
2. Avoid referencing specific names, events, or phrases from the paragraph.
3. Be more complex than the examples provided.
4. {style}
5. Not be phrased like the following or similarly:
   - "{' '.join(seed1.split()[:4])}"
   - "{' '.join(seed2.split()[:4])}"
   - "{' '.join(seed3.split()[:4])}".

So here's my inspired query:
"""


# =============================================================================
# 3. Response Generation
# =============================================================================

def qa_response_prompt(question: str) -> str:
    """Generate answer for a question."""
    return f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to a question, please don't share false information.

### Instruction:
{question}

### Response:"""


# =============================================================================
# 4. Complex QA Synthesis (beyond simple QA)
# =============================================================================

COMPLEX_QA_DOMAINS = [
    "Science & Technology", "History & Culture", "Philosophy & Ethics",
    "Mathematics & Logic", "Economics & Finance", "Medicine & Health",
    "Law & Governance", "Psychology & Sociology", "Environmental Science",
    "Computer Science & AI", "Literature & Arts", "Education & Learning",
    "Engineering & Architecture", "Biology & Genetics", "Physics & Astronomy",
    "Chemistry & Materials", "Political Science", "Linguistics",
    "Business & Management", "Sports & Fitness",
]

COMPLEX_QA_TYPES = [
    "multi_hop_reasoning", "counterfactual", "comparison",
    "cause_effect", "step_by_step", "debate_style",
    "case_study", "analogy", "evaluation",
    "synthesis", "prediction", "definition_with_examples",
]


def complex_qa_prompt(domain: str = None, qa_type: str = None) -> str:
    """Generate a standalone complex QA pair without needing a source document."""
    if domain is None:
        domain = random.choice(COMPLEX_QA_DOMAINS)
    if qa_type is None:
        qa_type = random.choice(COMPLEX_QA_TYPES)

    type_descriptions = {
        "multi_hop_reasoning": "requires connecting multiple pieces of information across different sub-topics",
        "counterfactual": "explores a hypothetical 'what if' scenario",
        "comparison": "compares and contrasts two or more concepts, methods, or phenomena",
        "cause_effect": "investigates causal relationships and chain effects",
        "step_by_step": "requires a detailed step-by-step solution process",
        "debate_style": "presents two opposing viewpoints and asks for balanced analysis",
        "case_study": "presents a realistic scenario requiring application of domain knowledge",
        "analogy": "uses analogies to explain complex concepts",
        "evaluation": "asks to evaluate the strengths and weaknesses of an approach",
        "synthesis": "requires combining knowledge from multiple areas into a coherent answer",
        "prediction": "asks to predict outcomes based on given conditions",
        "definition_with_examples": "requires a clear definition supported by concrete examples",
    }
    type_desc = type_descriptions.get(qa_type, "is challenging and thought-provoking")

    return f"""You are an expert question-and-answer pair creator for training AI models.

Create ONE high-quality question-answer pair in the domain of **{domain}**.

Requirements:
1. The question {type_desc}.
2. The question should be self-contained (no external references needed).
3. The answer should be comprehensive (200-500 words), accurate, and well-structured.
4. Include concrete examples, data points, or reasoning steps where appropriate.
5. The difficulty level should be suitable for a college-educated adult.

Output in this EXACT JSON format:
{{
    "question": "...",
    "answer": "...",
    "domain": "{domain}",
    "type": "{qa_type}",
    "difficulty": "medium|hard|expert"
}}"""


# =============================================================================
# 5. Multi-Turn Dialog Synthesis
# =============================================================================

DIALOG_SCENARIOS = [
    "A student asking a professor about a complex topic, with follow-up questions that go deeper",
    "A developer debugging code with an AI assistant, iteratively narrowing down the issue",
    "A researcher discussing methodology choices for an experiment",
    "A user learning a new skill step by step with an AI tutor",
    "A business analyst exploring data-driven insights with an AI",
    "A writer brainstorming and refining creative ideas with an AI collaborator",
    "A user seeking medical information (general, not diagnosis) with clarifying questions",
    "A user planning a complex project with task breakdown and prioritization",
    "An engineer designing a system architecture with trade-off discussions",
    "A user exploring philosophical questions with deepening argumentation",
    "A user learning a new programming language through practical examples",
    "A teacher preparing lesson materials with pedagogical considerations",
    "A user troubleshooting a technical problem with systematic elimination",
    "A data scientist exploring dataset characteristics and modeling approaches",
    "A user seeking legal information about contracts or regulations",
]

DIALOG_PATTERNS = [
    "progressive_deepening",  # Each turn goes deeper into the topic
    "topic_shift",            # Natural topic transitions within a theme
    "correction_and_refinement",  # User corrects/refines AI responses
    "multi_task",             # Multiple related tasks in one conversation
    "socratic_method",        # AI uses questions to guide understanding
    "collaborative_problem_solving",  # Both parties contribute ideas
    "iterative_refinement",   # Repeatedly improving a solution
    "exploration_to_decision", # Exploring options then making a decision
]


def multi_turn_dialog_prompt(scenario: str = None, pattern: str = None, num_turns: int = None) -> str:
    """Generate a multi-turn dialog."""
    if scenario is None:
        scenario = random.choice(DIALOG_SCENARIOS)
    if pattern is None:
        pattern = random.choice(DIALOG_PATTERNS)
    if num_turns is None:
        num_turns = random.randint(4, 8)

    pattern_descriptions = {
        "progressive_deepening": "Each user message should go deeper into the topic, building on previous answers.",
        "topic_shift": "The conversation should naturally transition between related sub-topics.",
        "correction_and_refinement": "The user should sometimes push back or ask for corrections/clarifications.",
        "multi_task": "The user should address multiple related but distinct tasks within the conversation.",
        "socratic_method": "The assistant should sometimes ask guiding questions instead of giving direct answers.",
        "collaborative_problem_solving": "Both the user and assistant should contribute ideas and build on each other.",
        "iterative_refinement": "The conversation should iteratively improve a solution or piece of work.",
        "exploration_to_decision": "Start by exploring options, then converge on a decision with justification.",
    }
    pattern_desc = pattern_descriptions.get(pattern, "Should be natural and engaging.")

    return f"""You are an expert at creating realistic, high-quality multi-turn conversations for AI training.

Create a realistic multi-turn conversation with EXACTLY {num_turns} turns (each turn = 1 user message + 1 assistant response).

**Scenario**: {scenario}
**Conversation Pattern**: {pattern_desc}

Requirements:
1. The conversation must feel natural, not scripted or formulaic.
2. User messages should vary in length and style (some short, some detailed).
3. Assistant responses should be genuinely helpful, detailed, and accurate.
4. The conversation should have a clear progression and purpose.
5. Include at least one instance where the user asks a follow-up that shows they understood (or misunderstood) the previous answer.
6. Avoid generic fillers like "That's a great question!" or "Thank you for your response."

Output in this EXACT JSON format:
{{
    "conversation": [
        {{"role": "user", "content": "..."}},
        {{"role": "assistant", "content": "..."}},
        ...
    ],
    "scenario": "{scenario}",
    "pattern": "{pattern}",
    "topic_summary": "brief summary of the conversation topic"
}}"""


# =============================================================================
# 6. Trajectory Inversion (Reverse-engineer questions from answers/trajectories)
# =============================================================================

def trajectory_inversion_prompt(trajectory: str, context: str = "") -> str:
    """Given an answer/trajectory, reverse-engineer the question that could have produced it."""
    ctx_block = f"\n### Context:\n{context}\n" if context else ""
    return f"""You are an expert at reverse-engineering questions from their answers.

Given the following answer/trajectory, create a natural, well-formed question that would have produced this response. The question should be self-contained and not reference the answer.
{ctx_block}
### Answer/Trajectory:
{trajectory}

Requirements:
1. The question must be self-contained (someone reading only the question can understand what's being asked).
2. The question should be natural and specific enough that the given trajectory is a reasonable answer.
3. If the trajectory contains intermediate steps, your question should be about the final goal, not the steps.
4. The question should be challenging enough to warrant such a detailed answer.
5. Do NOT include phrases like "Based on the above" or "According to the trajectory".

Output in this EXACT JSON format:
{{
    "question": "the reverse-engineered question",
    "reasoning": "brief explanation of how you derived this question from the trajectory",
    "difficulty": "easy|medium|hard|expert",
    "domain": "the domain of the question"
}}"""


def trajectory_inversion_with_verification_prompt(trajectory: str, intermediate_results: str = "") -> str:
    """Advanced trajectory inversion that leverages verifiable intermediate results."""
    return f"""You are an expert at creating training data from execution trajectories.

Even when a trajectory doesn't reach the correct final answer, its intermediate steps often contain valuable reasoning. Your job is to:
1. Identify the VALUABLE intermediate results in this trajectory (correct reasoning steps, partial solutions).
2. Create a question-answer pair where the answer incorporates these valuable intermediate results.

### Trajectory:
{trajectory}

{f"### Verified Intermediate Results:{chr(10)}{intermediate_results}" if intermediate_results else ""}

Requirements:
1. The question should be self-contained and natural.
2. The answer should incorporate the CORRECT parts of the trajectory.
3. If there are errors in the trajectory, the answer should fix them while keeping the good reasoning.
4. Rate the quality of the intermediate results.

Output in this EXACT JSON format:
{{
    "question": "...",
    "answer": "answer that incorporates correct intermediate reasoning",
    "valuable_steps": ["list of which intermediate steps were valuable"],
    "corrected_steps": ["list of which steps needed correction"],
    "intermediate_quality": 0.0 to 1.0,
    "domain": "...",
    "difficulty": "easy|medium|hard|expert"
}}"""


# =============================================================================
# 7. Code QA Synthesis
# =============================================================================

CODE_TOPICS = [
    "algorithms and data structures", "web development", "database design",
    "system design", "API development", "testing and debugging",
    "performance optimization", "security best practices",
    "machine learning implementation", "data processing pipelines",
    "concurrent programming", "design patterns",
    "functional programming", "object-oriented design",
    "DevOps and CI/CD", "cloud architecture",
]

CODE_LANGUAGES = ["Python", "JavaScript", "TypeScript", "Java", "C++", "Go", "Rust", "SQL"]


def code_qa_prompt(topic: str = None, language: str = None) -> str:
    if topic is None:
        topic = random.choice(CODE_TOPICS)
    if language is None:
        language = random.choice(CODE_LANGUAGES)

    return f"""You are an expert programmer creating high-quality coding QA pairs for AI training.

Create ONE coding question-answer pair about **{topic}** using **{language}**.

Requirements:
1. The question should be specific and practical (not a textbook definition).
2. The answer should include working code with clear comments.
3. Include edge case handling in the solution.
4. The code should follow best practices for {language}.
5. Explain the approach before showing the code.

Output in this EXACT JSON format:
{{
    "question": "...",
    "answer": "complete answer with code and explanation",
    "language": "{language}",
    "topic": "{topic}",
    "difficulty": "easy|medium|hard",
    "tags": ["relevant", "tags"]
}}"""


# =============================================================================
# 8. Math QA Synthesis
# =============================================================================

MATH_LEVELS = ["elementary", "middle_school", "high_school", "college", "competition"]
MATH_TOPICS = [
    "arithmetic", "algebra", "geometry", "trigonometry", "calculus",
    "probability", "statistics", "number_theory", "combinatorics",
    "linear_algebra", "differential_equations", "optimization",
    "graph_theory", "abstract_algebra", "real_analysis",
]


def math_qa_prompt(level: str = None, topic: str = None) -> str:
    if level is None:
        level = random.choice(MATH_LEVELS)
    if topic is None:
        topic = random.choice(MATH_TOPICS)

    return f"""You are a mathematics educator creating high-quality math problems for AI training.

Create ONE math problem at the **{level}** level about **{topic}**.

Requirements:
1. The problem should be self-contained with all necessary information.
2. Provide a step-by-step solution showing all work.
3. The final answer should be clearly marked.
4. Include a brief explanation of the key concept being tested.
5. The problem should be original (not a direct copy of known problems).

Output in this EXACT JSON format:
{{
    "question": "the math problem",
    "solution": "step-by-step solution",
    "final_answer": "the final numerical or symbolic answer",
    "level": "{level}",
    "topic": "{topic}",
    "key_concept": "what concept this tests"
}}"""


# =============================================================================
# 9. Creative Writing QA
# =============================================================================

WRITING_TASKS = [
    "creative story writing", "poetry composition", "essay writing",
    "technical writing", "persuasive writing", "dialogue writing",
    "world-building", "character development", "plot analysis",
    "writing style transfer", "editing and revision", "creative brainstorming",
]


def creative_writing_prompt(task: str = None) -> str:
    if task is None:
        task = random.choice(WRITING_TASKS)

    return f"""You are a creative writing expert generating training data for AI.

Create ONE creative writing task-response pair about **{task}**.

Requirements:
1. The task should be specific and engaging.
2. The response should demonstrate high writing quality (varied vocabulary, good structure, vivid details).
3. The response should be 200-600 words.
4. Include some element of creativity or originality.

Output in this EXACT JSON format:
{{
    "instruction": "the writing task/prompt",
    "response": "the creative writing response",
    "type": "{task}",
    "quality_indicators": ["what makes this response high quality"]
}}"""


# =============================================================================
# 10. Reasoning & Logic QA
# =============================================================================

def reasoning_qa_prompt() -> str:
    reasoning_type = random.choice([
        "deductive reasoning", "inductive reasoning", "abductive reasoning",
        "analogical reasoning", "causal reasoning", "spatial reasoning",
        "temporal reasoning", "probabilistic reasoning", "ethical reasoning",
        "systems thinking", "logical puzzles", "paradox analysis",
    ])

    return f"""You are a logic and reasoning expert creating training data for AI.

Create ONE challenging reasoning question-answer pair focusing on **{reasoning_type}**.

Requirements:
1. The question should require genuine reasoning, not just recall.
2. Show the complete chain of reasoning in the answer.
3. If applicable, identify common mistakes or fallacies related to this type of reasoning.
4. The answer should be structured: premises → reasoning steps → conclusion.

Output in this EXACT JSON format:
{{
    "question": "...",
    "answer": "detailed answer with reasoning chain",
    "reasoning_type": "{reasoning_type}",
    "difficulty": "medium|hard|expert",
    "common_mistakes": ["potential mistakes someone might make"]
}}"""


# =============================================================================
# Helper: Generate mixed batch prompts
# =============================================================================

DATA_TYPE_WEIGHTS = {
    "complex_qa": 0.30,
    "code_qa": 0.15,
    "math_qa": 0.15,
    "reasoning": 0.15,
    "creative_writing": 0.10,
    "multi_turn": 0.15,
}


def generate_mixed_batch_prompts(batch_size: int) -> List[Dict]:
    """Generate a mixed batch of prompts across all data types."""
    prompts = []
    for _ in range(batch_size):
        r = random.random()
        cumulative = 0.0
        for dtype, weight in DATA_TYPE_WEIGHTS.items():
            cumulative += weight
            if r <= cumulative:
                if dtype == "complex_qa":
                    prompts.append({"type": dtype, "prompt": complex_qa_prompt()})
                elif dtype == "code_qa":
                    prompts.append({"type": dtype, "prompt": code_qa_prompt()})
                elif dtype == "math_qa":
                    prompts.append({"type": dtype, "prompt": math_qa_prompt()})
                elif dtype == "reasoning":
                    prompts.append({"type": dtype, "prompt": reasoning_qa_prompt()})
                elif dtype == "creative_writing":
                    prompts.append({"type": dtype, "prompt": creative_writing_prompt()})
                elif dtype == "multi_turn":
                    prompts.append({"type": dtype, "prompt": multi_turn_dialog_prompt()})
                break
    return prompts
