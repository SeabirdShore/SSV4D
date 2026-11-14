"""Built-in research prompts.

The first experiment is deliberately self-contained and does not depend on a
gated or network-hosted dataset. Prompts span factual, reasoning, coding, and
open-ended generation to exercise different near-tie regimes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchPrompt:
    prompt_id: str
    category: str
    text: str


PROMPTS: tuple[ResearchPrompt, ...] = (
    ResearchPrompt(
        "fact_moon",
        "factual",
        "List ten concise facts about the Moon. Number each fact.",
    ),
    ResearchPrompt(
        "math_sum",
        "reasoning",
        "Compute 17 * 23 + 19 * 11. Show the arithmetic and final answer.",
    ),
    ResearchPrompt(
        "math_probability",
        "reasoning",
        "A fair six-sided die is rolled three times. What is the probability "
        "that the sum is exactly 12? Explain the counting.",
    ),
    ResearchPrompt(
        "code_binary_search",
        "code",
        "Write a Python function that returns the first index of a target in a "
        "sorted list using binary search. Include a short correctness argument.",
    ),
    ResearchPrompt(
        "code_lru",
        "code",
        "Describe how an LRU cache can be implemented with O(1) get and put "
        "operations. Use concise pseudocode.",
    ),
    ResearchPrompt(
        "science_photosynthesis",
        "science",
        "Explain photosynthesis in six numbered steps, from light absorption "
        "through sugar production.",
    ),
    ResearchPrompt(
        "language_compare",
        "language",
        "Compare the meanings of 'affect' and 'effect', then give three example "
        "sentences for each word.",
    ),
    ResearchPrompt(
        "plan_weekend",
        "planning",
        "Plan a two-day weekend trip to a nearby city with a moderate budget. "
        "Give a chronological itinerary and a packing checklist.",
    ),
    ResearchPrompt(
        "math_gcd",
        "reasoning",
        "Find the greatest common divisor of 391 and 493 using the Euclidean "
        "algorithm. Show every division step.",
    ),
    ResearchPrompt(
        "code_json_validate",
        "code",
        "Write a Python function that validates whether a nested object contains "
        "only JSON-compatible types. Explain edge cases.",
    ),
    ResearchPrompt(
        "science_tcp",
        "science",
        "Explain TCP congestion control in six concise steps, including slow "
        "start and congestion avoidance.",
    ),
    ResearchPrompt(
        "history_computing",
        "history",
        "Summarize the transition from vacuum-tube computers to integrated "
        "circuits in five key milestones.",
    ),
    ResearchPrompt(
        "statistics_bootstrap",
        "statistics",
        "Explain the bootstrap method, when it is useful, and one important "
        "limitation. Include a short example.",
    ),
    ResearchPrompt(
        "writing_revision",
        "language",
        "Rewrite the following sentence for clarity and then explain the edit: "
        "'The results which were obtained by the team were analyzed.'",
    ),
    ResearchPrompt(
        "unit_conversion",
        "reasoning",
        "Convert 55 miles per hour to meters per second. Show conversion factors "
        "and round the final answer to two decimals.",
    ),
    ResearchPrompt(
        "ops_checklist",
        "planning",
        "Create a pre-release checklist for deploying a language-model service. "
        "Group items by correctness, performance, safety, and rollback.",
    ),
)


FILLER_PROMPTS: tuple[str, ...] = (
    "Name three colors in a rainbow and explain briefly why rainbows appear.",
    "Summarize the difference between TCP and UDP in four sentences.",
    "Write a short paragraph about the history of the printing press.",
    "Explain why the ocean is salty in two or three sentences.",
    "List five common mistakes in Python loops and how to avoid them.",
    "Give a concise recipe for a vegetarian pasta dish.",
    "Explain the difference between correlation and causation with an example.",
    "Describe how a bicycle derailleur changes gears.",
    "Write a short note explaining what a hash table is.",
    "List four ways to reduce energy consumption at home.",
    "Explain the concept of compound interest with a small numerical example.",
    "Compare a compiler and an interpreter in simple terms.",
    "Write a brief description of the water cycle.",
    "Give three tips for writing a clear technical report.",
    "Explain why leaves change color in autumn.",
    "Describe the basic structure of a database transaction.",
    "List five safety checks before running a machine-learning experiment.",
    "Explain what a confidence interval represents to a non-statistician.",
    "Write a short paragraph about the invention of the telephone.",
    "Describe three common patterns in distributed systems.",
)
