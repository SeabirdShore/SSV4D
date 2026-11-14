"""Extended prompt suite for multi-window SSV4D generalization checks."""

from __future__ import annotations

from ssv4d.prompts import PROMPTS, ResearchPrompt


EXTENDED_PROMPTS: tuple[ResearchPrompt, ...] = PROMPTS + (
    ResearchPrompt(
        "fact_capitals",
        "factual",
        "List the capitals of Canada, Australia, Brazil, and Switzerland, with "
        "one distinguishing geographic fact per country.",
    ),
    ResearchPrompt(
        "fact_planets",
        "factual",
        "Compare the orbital periods and relative sizes of Mercury, Earth, "
        "Jupiter, and Neptune in a compact table.",
    ),
    ResearchPrompt(
        "fact_http",
        "factual",
        "Explain the difference between HTTP GET and POST using three concrete "
        "request examples.",
    ),
    ResearchPrompt(
        "fact_honey",
        "factual",
        "Explain why honey can remain edible for a long time and what conditions "
        "can still cause it to spoil.",
    ),
    ResearchPrompt(
        "fact_penicillin",
        "factual",
        "Describe the discovery and early development of penicillin in five "
        "chronological points.",
    ),
    ResearchPrompt(
        "math_interest",
        "reasoning",
        "Calculate compound growth for $2,500 at 4.5% annual interest over "
        "seven years, compounded yearly. Show the formula and result.",
    ),
    ResearchPrompt(
        "math_modular",
        "reasoning",
        "Find 17^2026 modulo 29 using modular arithmetic. Show the exponent "
        "reduction clearly.",
    ),
    ResearchPrompt(
        "probability_cards",
        "reasoning",
        "Two cards are drawn without replacement from a standard deck. Find the "
        "probability that both are hearts and explain each step.",
    ),
    ResearchPrompt(
        "code_stack",
        "code",
        "Implement a stack with push, pop, peek, and a current-minimum query in "
        "O(1) amortized time. Include Python code and edge cases.",
    ),
    ResearchPrompt(
        "code_toposort",
        "code",
        "Write a Python function for topological sorting using Kahn's algorithm "
        "and explain how it detects cycles.",
    ),
    ResearchPrompt(
        "code_debounce",
        "code",
        "Explain debounce and throttle in JavaScript. Give a small implementation "
        "for each and compare their behavior.",
    ),
    ResearchPrompt(
        "code_sql",
        "code",
        "Write a SQL query that returns the second-highest salary in each "
        "department without using window functions.",
    ),
    ResearchPrompt(
        "science_ml",
        "science",
        "Explain precision, recall, F1 score, and ROC-AUC to an engineering "
        "team. Give one case where accuracy is misleading.",
    ),
    ResearchPrompt(
        "science_entropy",
        "science",
        "Explain entropy and the second law of thermodynamics in six short "
        "paragraphs without using calculus.",
    ),
    ResearchPrompt(
        "science_dna",
        "science",
        "Describe the central dogma of molecular biology and distinguish "
        "transcription from translation with examples.",
    ),
    ResearchPrompt(
        "language_metaphor",
        "language",
        "Explain the difference between a simile, metaphor, and analogy, with "
        "two original examples of each.",
    ),
    ResearchPrompt(
        "grammar_semicolon",
        "language",
        "Give five rules for using semicolons and write an original example for "
        "each rule.",
    ),
    ResearchPrompt(
        "writing_summary",
        "language",
        "Summarize the tradeoffs of remote work in under 180 words, using "
        "separate paragraphs for benefits, costs, and recommendations.",
    ),
    ResearchPrompt(
        "plan_move",
        "planning",
        "Create a four-week plan for moving between apartments, including "
        "logistics, utilities, cleaning, and a contingency budget.",
    ),
    ResearchPrompt(
        "plan_conference",
        "planning",
        "Plan a one-day technical conference for 200 attendees, including "
        "timeline, staffing, rooms, catering risks, and a rollback plan.",
    ),
    ResearchPrompt(
        "plan_backup",
        "planning",
        "Design a backup strategy for a small software company with laptops, a "
        "production database, and cloud object storage.",
    ),
    ResearchPrompt(
        "statistics_variance",
        "statistics",
        "Explain population variance and sample variance, then compute both for "
        "the values 3, 5, 7, 9, and 11.",
    ),
    ResearchPrompt(
        "statistics_ab",
        "statistics",
        "Design an A/B test for a checkout-page change. State hypotheses, sample "
        "size considerations, metrics, and common pitfalls.",
    ),
    ResearchPrompt(
        "statistics_pvalue",
        "statistics",
        "Explain what a p-value is and what it is not, using one correct and one "
        "incorrect interpretation.",
    ),
    ResearchPrompt(
        "history_internet",
        "history",
        "Summarize five milestones in the development of the public internet, "
        "from packet switching to the web.",
    ),
    ResearchPrompt(
        "history_industrial",
        "history",
        "Explain how the industrial revolution changed manufacturing, cities, "
        "and labor in five concise points.",
    ),
    ResearchPrompt(
        "unit_temperature",
        "reasoning",
        "Convert -40 degrees Fahrenheit to Celsius, 300 kelvin to Celsius, and "
        "100 degrees Celsius to kelvin. Show every formula.",
    ),
    ResearchPrompt(
        "unit_bytes",
        "reasoning",
        "Compare decimal and binary units for gigabytes. Convert 750,000,000 "
        "bytes under both conventions and explain the difference.",
    ),
    ResearchPrompt(
        "ops_incident",
        "planning",
        "Write an incident-response checklist for a sudden 50% latency increase "
        "in an LLM inference service. Include containment and communication.",
    ),
    ResearchPrompt(
        "ops_cache",
        "science",
        "Explain cache invalidation strategies for a distributed web service. "
        "Compare TTL, write-through, and event-driven invalidation.",
    ),
    ResearchPrompt(
        "safety_bias",
        "language",
        "Describe three practical checks for detecting bias in a text-classifier "
        "evaluation and explain why each check matters.",
    ),
    ResearchPrompt(
        "logic_knights",
        "reasoning",
        "Solve this logic puzzle: A says 'B is a knight'; B says 'A and I are "
        "different types.' Knights always tell the truth and knaves always lie. "
        "Determine both types and show the cases.",
    ),
)
