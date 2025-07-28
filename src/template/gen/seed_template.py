from typing import List, Dict
import random   
def generate_seed_prompt(text: str) -> List[str]:
    """
    Generate a set of detailed English prompts for creating complex questions based on a given paragraph.

    This function is designed to help generate seed prompts for instruction generation tasks. Each prompt instructs the model to create a single, complex question that adheres to specific requirements. The requirements are described in detail in English, covering aspects such as reasoning complexity, critical thinking, creativity, interdisciplinarity, and various question types (e.g., NLI, commonsense, sentiment analysis, paraphrasing, code generation, math, etc.).

    The generated prompts explicitly prohibit referencing the paragraph directly (e.g., "Based on the provided information") and require the question to fit a particular style (command or interrogative) and category.

    Args:
        text (str): The paragraph that serves as the basis for question generation.

    Returns:
        List[str]: A list of detailed English prompts, each specifying the requirements for a single question.
    """
    reasoning_tag = "It should be complex and requires multiple-step reasoning to solve."
    critical_thinking_tag = "It demands critical thinking skills to analyze from various perspectives and evaluate multiple solutions."
    creativity_tag = "It necessitates creative thinking to devise innovative solutions beyond conventional approaches."
    interdisciplinary_tag = "It demands integrating knowledge from diverse disciplines to address its multifaceted nature."
    command_tag = "It should be in the style of a command or imperative. For example, 'Write a paragraph about...' or 'Describe the...'"
    question_tag = "It should be in the style of a question or interrogative. For example, 'What is the...?' or 'How do you...?'"
    
    nli_tag = "It is a Natural language inference question: Assessing if evidence supports a conclusion."
    commonsense_tag = "It is a Commonsense question: Predicting outcomes based on everyday knowledge."
    sentiment_tag = "It is a Sentiment analysis question: Determining emotional response to a given scenario."
    paraphrase_tag = "It is a Paraphrasing question: Rewording a statement while retaining its meaning."
    close_book_qa_tag = "It is a Close-book QA question: Answering factual queries using pre-existing knowledge."
    struc2text_tag = "It is a Structure to text question: Describing a process or concept in written form."
    summarization_tag = "It is a Summarization question: Condensing key information from a larger text."
    translate_tag = "It is a Translation question: Converting text from one language to another."
    implicit_reasoning_tag = "It is a Implicit reasoning question: Inferring reasons behind common behaviors."
    text_category_tag = "It is a Text categorization question: Identifying defining characteristics of a given text type."
    code_category_tag = "It is a Code generation question: Generate a LeetCode easy-level coding question."
    gsm8k_tag = "It is a gsm8k math question: Generate a question designed for elementary school-level mathematical problem-solving"
    # classify_tag = "It is a Classification question: Assigning a label to a given input based on predefined categories."
    

    tags = [
        reasoning_tag,
        critical_thinking_tag,
        creativity_tag,
        interdisciplinary_tag
    ]
    classify = [
        nli_tag,
        commonsense_tag,
        sentiment_tag,
        paraphrase_tag,
        close_book_qa_tag,
        struc2text_tag,
        summarization_tag,
        translate_tag,
        implicit_reasoning_tag,
        text_category_tag,
        code_category_tag,
        gsm8k_tag
    ]
    types = [command_tag, question_tag]

    
    QUESTION_TEMPLATE = """You're proficient in crafting complex question. Generate only one question that adheres to the provided ### Paragraph.
The question should meet the following criteria:
0. The person answering the question cannot see the ### Paragraph, so the question must not contain phrases like 'Given the information provided', 'Based on the provided information', or similar expressions that imply direct citations or references from ### Paragraph.
1. {characteristic}.
2. {type}.
3. {classify}.
    
[IMPORTANT] If you do not follow ALL of the above rules, your response will be rejected.

### Paragraph:
{text}
### Question:"""
    prompts = [
        QUESTION_TEMPLATE.format(
            characteristic=tag,
            type=type,
            text=text,
            classify=c
        )
        for tag in tags for c in classify for type in types
    ]
    return prompts

    