"""Evaluation prompt templates for LLM-as-judge scoring."""

from __future__ import annotations


def privacy_eval(instruction: str) -> str:
    """Evaluate if instruction contains unfamiliar private names/places. 1=good, 0=bad."""
    return (
        "Evaluate the following instruction to determine if it contains the name of an unfamiliar person, place, or organization. "
        "If it doesn't, score it as 1. If it does, score it as 0. Output the score in the format: "
        "`score: X`, where X is either 0 or 1.\n\n"
        f'Instruction: "{instruction}"\n\nOutput:'
    )


def safety_eval(instruction: str) -> str:
    """Evaluate if instruction is safe (no harmful content). 1=safe, 0=unsafe."""
    return (
        "Evaluate the following instruction to determine if it is safe, which means it doesn't contain any dangerous, discriminatory, or harmful content. "
        "If it is safe, score it as 1. If it is not, score it as 0. Output the score in the format: "
        "`score: X`, where X is either 0 or 1.\n\n"
        f'Instruction: "{instruction}"\n\nOutput:'
    )


def originality_eval(instruction: str) -> str:
    """Evaluate if instruction is original and interesting. 1=original, 0=common."""
    return (
        "Evaluate the following instruction to determine if it is original, interesting, and special, rather than similar "
        "to common questions starting with 'how', 'what', 'which', etc. If it is original, interesting, and special, score it as 1. "
        "If it is similar to common questions, score it as 0. Output the score in the format: `score: X`, where X is either 0 or 1.\n\n"
        f'Instruction: "{instruction}"\n\nOutput:'
    )


def difficult_eval(instruction: str) -> str:
    """Evaluate if instruction is difficult. 1=difficult, 0=easy."""
    return (
        "Evaluate the following instruction to determine if it is difficult to solve and requires several steps of logical reasoning, or advanced knowledge from graduate-level courses. If it is difficult, score it as 1. "
        "If it is easy to understand or execute, score it as 0. Output the score in the format: `score: X`, where X is either 0 or 1.\n\n"
        "Example 1, the instruction 'Write a report on the contributions of Dr. Naliah Kareem to molecular biology' is difficult to solve and requires several steps of logical reasoning.\n"
        "Example 2, the instruction 'How do you make a cake?' is easy to understand or execute.\n"
        "Example 3, the instruction 'Write a code to print the first 10 prime numbers.' is easy to understand or execute.\n"
        "Example 4, the instruction 'Write a paper about the history of the United States.' is easy to understand or execute.\n"
        "Example 5, the instruction 'How to solve the Twin prime conjecture' is difficult to solve and requires several steps of logical reasoning.\n"
        f'Instruction: "{instruction}"\n\nOutput:'
    )


def insjudge_eval(instruction: str) -> str:
    """Evaluate if instruction is a question/command vs article/paragraph. 1=instruction, 0=paragraph."""
    return (
        "Evaluate the following instruction to determine if it is a question or a command, rather than an article or a paragraph. "
        "If it is a question or a command, score it as 1. If it is an article or a paragraph, score it as 0. Output the score in the format: "
        "`score: X`, where X is either 0 or 1.\n\n"
        f'Instruction: "{instruction}"\n\nOutput:'
    )


def faithfulness_eval(instruction: str, response: str) -> str:
    """5-point answer quality scoring (adapted from HumpBack)."""
    return (
        "Below is an instruction from an user and a candidate answer.\n"
        "Let's think step by step.\n"
        "Evaluate whether or not the answer is a good example of how AI Assistant should respond to the user's instruction. "
        "Please assign a score using the following 5-point scale:\n"
        "1: It means the answer is incomplete, vague, off-topic, or not exactly what the user asked for. "
        "For example, some content seems missing. Or the response is from another person's perspective with their personal experience "
        "(e.g. taken from blog posts). Or it contains promotional text or other irrelevant information.\n"
        "2: (between 1 and 3)\n"
        "3: It means the answer is helpful but not written by an AI Assistant. It addresses all the basic asks from the user. "
        "It is complete and self contained with the drawback that the response is not written from an AI assistant's perspective, "
        "but from other people's perspective. For example, it contains personal experience or opinion, mentions comments section, or share on social media, etc.\n"
        "4: (between 3 and 5)\n"
        "5: It means it is a perfect answer from an AI Assistant. It has a clear focus on being a helpful AI Assistant, "
        "where the response looks like intentionally written to address the user's question or instruction without any irrelevant sentences. "
        "The answer provides high quality content, demonstrating expert knowledge in the area, is very well written, logical, easy-to-follow, engaging and insightful.\n\n"
        "Your reply should be only 1 or 2 or 3 or 4 or 5, without providing any reasoning and explanation.\n\n"
        f"###Instruction:\n{instruction}\n\n###Answer:\n{response}\n\n###Your Reply:"
    )


def answer_quality_eval(instruction: str, response: str) -> str:
    """Evaluate answer quality on 1-5 scale. Returns score prompt for LLM judge."""
    return faithfulness_eval(instruction, response)


__all__ = [
    "privacy_eval",
    "safety_eval",
    "originality_eval",
    "difficult_eval",
    "insjudge_eval",
    "faithfulness_eval",
    "answer_quality_eval",
]
