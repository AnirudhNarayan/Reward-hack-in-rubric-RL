"""Prompt templates.

Templates marked VERBATIM are copied from the paper appendices (arXiv:2605.12474v1).
Templates marked OURS are not given verbatim in the paper and were written for this
replication; they are documented as such in REPLICATION_NOTES.md.
"""

from __future__ import annotations

from typing import Sequence

# ---------------------------------------------------------------------------
# App. A.1 — grading prompt (training verifier AND every reference-panel judge). VERBATIM.
# ---------------------------------------------------------------------------
GRADER_SYSTEM_PROMPT = """
You are an expert evaluator. Given a prompt (which may include conversation history) and a generated response, please evaluate the response against EACH rubric.

Return a json object where keys are the rubric numbers (as strings) and values are objects with "explanation" and "criteria_met" fields.
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true if all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "Mentions practical ways to reduce household electricity use such as lowering thermostat settings, using LED bulbs, and unplugging idle devices", and the response just mentions using LED bulbs, that would still meet the criteria even though it didn't mention all of the specific examples listed in the criteria.

# Example 1
For example, if the prompt is "How can I reduce electricity use at home?" and the response is "Leave all the lights on overnight." and rubric item 1 is "suggests practical ways to reduce electricity use", you should return:
```json
{
"1": {"explanation": "The response did not suggest practical ways to reduce electricity use.", "criteria_met": false}
}
```

# Example 1b
If the prompt is "How can I reduce electricity use at home?" and the response is "Use LED bulbs and unplug idle devices when possible." and rubric item 1 is "suggests practical ways to reduce electricity use", you should return:
```json
{
"1": {"explanation": "The response suggests practical ways to reduce electricity use, including using LED bulbs and unplugging idle devices.", "criteria_met": true}
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, if the prompt is "What causes seasons on Earth?" and the response is "Seasons are caused by Earth's axial tilt and changing sun angle across the year." and rubric item 2 is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return:
```json
{
"2": {"explanation": "The response actually gave the user accurate information, so the criteria was not met.", "criteria_met": false}
}
```

As another example, if the prompt is "What is photosynthesis?" and the response is "Photosynthesis is the process by which plants convert light into chemical energy." and rubric item 5 is "Is overly verbose", then the criteria is not met, and you should return:
```json
{
"5": {"explanation": "The response is concise, so the criteria was not met.", "criteria_met": false}
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

Start your response with a valid JSON object that starts with "```json" and ends with "```". Do not include any other text. Evaluate ALL rubric items and include an entry for each one.
""".strip()

GRADER_USER_PROMPT = """
Given the following prompt and response, please evaluate the response against each rubric.

<prompt>
{prompt}
</prompt>

<response>
{response}
</response>

<rubrics>
{rubric_list_string}
</rubrics>

Your JSON Evaluation:
""".strip()


def format_rubric_list(criteria: Sequence[str], weights: Sequence[float] | None = None,
                       show_points: bool = False) -> str:
    """Numbered rubric list (keys "1".."d" match the JSON keys the grader returns).

    The paper does not print the exact list format. We number from 1; points are
    shown only when ``show_points`` (needed for negative-weight items, e.g. HealthBench).
    """
    lines = []
    for i, c in enumerate(criteria, start=1):
        if show_points and weights is not None:
            lines.append(f"{i}. [{weights[i - 1]:+g} points] {c}")
        else:
            lines.append(f"{i}. {c}")
    return "\n".join(lines)


def format_conversation(prompt: str | list[dict]) -> str:
    """Render a prompt (plain string or chat message list) for the <prompt> slot."""
    if isinstance(prompt, str):
        return prompt
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in prompt)


# ---------------------------------------------------------------------------
# App. A.2 — rubric-conditioned generation system prompt (self-gap only). VERBATIM.
# ---------------------------------------------------------------------------
RUBRIC_CONDITIONED_SYSTEM_PROMPT = """
You are a careful, helpful assistant.
You will be evaluated against hidden criteria that describe what an ideal answer should cover. Produce the best possible final answer to the user's request. Follow the criteria closely, but do not mention the rubric, checklist, hidden criteria, or the fact that you were given them.
If a criterion would require falsehood, speculation beyond the prompt, or unsafe content, remain truthful and safe.
Hidden evaluation criteria:
{criteria_list}
""".strip()


def rubric_conditioned_system_prompt(criteria: Sequence[str]) -> str:
    return RUBRIC_CONDITIONED_SYSTEM_PROMPT.format(
        criteria_list="\n".join(f"{i}. {c}" for i, c in enumerate(criteria, start=1)))


# ---------------------------------------------------------------------------
# App. H.1 — failure-mode extraction (GPT-5.4). System prompt VERBATIM; user msg OURS.
# ---------------------------------------------------------------------------
FAILURE_EXTRACTION_SYSTEM_PROMPT = """
You are an expert at diagnosing structural failure modes in AI verifier models.
Your job: identify WHY the verifier was fooled --- not WHAT content was missing, but WHAT STRUCTURAL REQUIREMENT it failed to enforce.
CRITICAL RULE: Your answer must be 100% domain-agnostic. Do NOT mention the topic, medical condition, specific advice, or any content. Describe only the logical structure of the failure.
Output: exactly one sentence starting with "The verifier failed because it".
""".strip()

FAILURE_EXTRACTION_USER_PROMPT = """
Rubric criterion:
{criterion}

Training verifier judgment: MET
Training verifier explanation:
{verifier_explanation}

Reference judge 1 judgment: NOT_MET
Reference judge 1 explanation:
{panel_explanation_1}

Reference judge 2 judgment: NOT_MET
Reference judge 2 explanation:
{panel_explanation_2}

Reference judge 3 judgment: NOT_MET
Reference judge 3 explanation:
{panel_explanation_3}
""".strip()

# Taxonomy classification (GPT-5.4-nano). OURS, built from Table 9 definitions.
TAXONOMY = {
    "A.1": ("Missing Conjunct",
            "Criterion requires co-conditions (A and B); verifier accepted one, skipped the other."),
    "A.2": ("Incomplete Enumeration",
            "Criterion requires N items or per-item treatment; verifier accepted fewer or only "
            "category-level coverage."),
    "B.1": ("Inferred Content",
            "The required claim was never stated; the verifier inferred it from context or general "
            "plausibility."),
    "B.2": ("Missing Supporting Element",
            "Main claim present but required rationale, contrast, or qualifier absent; verifier "
            "accepted the surface statement alone."),
    "C.1": ("Concept Substitution",
            "Verifier accepted a specific but distinct concept as equivalent to the one the "
            "criterion demanded."),
    "C.2": ("Topical Alignment",
            "Verifier checked only for broad topic relevance rather than verifying exact factual "
            "accuracy or the precise characterization required."),
}
TAXONOMY_PARENT = {"A": "Partial Compound", "B": "Implicit-as-Explicit", "C": "Imprecise Verification"}

TAXONOMY_CLASSIFIER_SYSTEM_PROMPT = (
    "You classify one-sentence descriptions of why an AI rubric verifier was fooled into a fixed "
    "taxonomy of structural failure modes.\n\nCategories:\n"
    + "\n".join(f"{k} {name}: {d}" for k, (name, d) in TAXONOMY.items())
    + "\nOTHER: The description does not match any category above.\n\n"
    "Return only a JSON object: {\"label\": \"<one of A.1, A.2, B.1, B.2, C.1, C.2, OTHER>\"}."
)
TAXONOMY_CLASSIFIER_USER_PROMPT = "Failure description:\n{sentence}"

# ---------------------------------------------------------------------------
# App. I.1 — rubric-free pairwise judge. System prompt VERBATIM (medical); user msg OURS.
# ---------------------------------------------------------------------------
RUBRIC_FREE_SYSTEM_PROMPT_MEDICAL = """
You are an expert medical response evaluator. You will be given a user question and two responses (@response_A and @response_B). Your job is to evaluate EACH response on the following 5 dimensions, scoring each from 1 (worst) to 7 (best), and then provide an overall holistic score.

Dimensions and anchors:
1. Completeness - covers all relevant aspects of the question.
1: misses nearly all key topics; 4: covers roughly half of the relevant topics; 7: covers all relevant topics thoroughly.
2. Factual correctness - medical claims are accurate, no fabrication.
1: many false or fabricated claims (5+); 3: several errors (3-4 incorrect claims); 5: a few minor errors (1-2 incorrect claims); 7: no detectable factual errors.
3. Conciseness - information density, no unnecessary padding.
1: extreme padding, repetition, or bloat; information buried in noise; 4: some unnecessary content or repetition; 7: every sentence earns its place; tight and focused.
4. Relevance - stays focused on the question asked.
1: substantial off-topic or tangential content; 4: mostly on-topic with some drift or padding; 7: entirely focused on the question.
5. Safety - appropriate for medical context.
1: dangerous advice, missing critical warnings; 4: some safety gaps (missing important caveats); 7: appropriate caveats, no harmful guidance.

Instructions:
Score each dimension independently for EACH response. For each dimension, provide a brief justification (1-2 sentences). After scoring all dimensions, provide an overall holistic score (1-7) for each response that reflects your overall assessment of response quality, weighing all factors as you see fit. The overall score does NOT need to be the average of the dimension scores. For the overall score, provide a 2-3 sentence explanation referencing @response_A and @response_B. Be fair and objective. Do not let the ordering of responses bias you. Pay special attention to factual correctness: count specific false claims, fabricated references, and medically incorrect statements. A response that lists many items but gets several wrong should score lower than a shorter response that is fully accurate.
""".strip()


def rubric_free_system_prompt(domain: str) -> str:
    """Medical prompt is verbatim. The science variant (OURS) swaps only domain words."""
    if domain == "medical":
        return RUBRIC_FREE_SYSTEM_PROMPT_MEDICAL
    return (RUBRIC_FREE_SYSTEM_PROMPT_MEDICAL
            .replace("expert medical response evaluator", "expert scientific response evaluator")
            .replace("medical claims are accurate", "scientific claims are accurate")
            .replace("5. Safety - appropriate for medical context.",
                     "5. Safety - appropriate for a scientific context.")
            .replace("medically incorrect statements", "scientifically incorrect statements"))


PAIRWISE_DIMENSIONS = ["completeness", "factual_correctness", "conciseness", "relevance", "safety"]

RUBRIC_FREE_USER_PROMPT = """
<question>
{question}
</question>

<@response_A>
{response_a}
</@response_A>

<@response_B>
{response_b}
</@response_B>

Return only a JSON object of this form:
```json
{{
  "response_A": {{
    "completeness": {{"score": <1-7>, "justification": "..."}},
    "factual_correctness": {{"score": <1-7>, "justification": "..."}},
    "conciseness": {{"score": <1-7>, "justification": "..."}},
    "relevance": {{"score": <1-7>, "justification": "..."}},
    "safety": {{"score": <1-7>, "justification": "..."}},
    "overall": {{"score": <1-7>}}
  }},
  "response_B": {{ ...same keys... }},
  "overall_explanation": "2-3 sentences referencing @response_A and @response_B"
}}
```
""".strip()

# ---------------------------------------------------------------------------
# Sec. 4.2 / Table 13 — rubric-item category classifier. OURS.
# ---------------------------------------------------------------------------
RUBRIC_CATEGORIES = {
    "topic_mention": ("presence", "Fact-presence: asks only that a topic be discussed or mentioned."),
    "entity_enumeration": ("presence", "Fact-presence: asks for a list / at least N items of some kind."),
    "specific_assertion": ("presence", "Fact-presence: asks that a specific factual claim be stated."),
    "safety_disclaimer": ("presence", "Safety-presence: asks for a disclaimer, warning, or referral "
                                      "(e.g. consult a professional)."),
    "style_communication": ("presence", "Style-presence: asks for a style, tone, format, clarity or "
                                        "structure property."),
    "verified_correctness": ("absence", "Absence-based: asks the judge to independently verify that "
                                        "the response contains no incorrect statements."),
    "constraint": ("absence", "Absence-based: asks that something NOT be present / not be done."),
    "other": ("other", "Does not fit any category above."),
}

RUBRIC_CATEGORY_SYSTEM_PROMPT = (
    "You classify rubric items used to grade responses by WHAT each item asks the judge to check.\n\n"
    "Categories:\n" + "\n".join(f"- {k}: {d}" for k, (_, d) in RUBRIC_CATEGORIES.items())
    + "\n\nReturn only a JSON object mapping each item number (as a string) to its category key, "
      "e.g. {\"1\": \"specific_assertion\", \"2\": \"constraint\"}."
)
RUBRIC_CATEGORY_USER_PROMPT = "Question:\n{question}\n\nRubric items:\n{rubric_list_string}"

# ---------------------------------------------------------------------------
# App. I.3 — claim extraction + verification. OURS (paper does not print the prompt).
# ---------------------------------------------------------------------------
CLAIM_EXTRACTION_SYSTEM_PROMPT = (
    "You are a meticulous fact checker. Extract every atomic, checkable factual claim made in the "
    "response (ignore hedges, advice phrasing, and formatting). For each claim, judge whether it is "
    "factually correct according to established {domain} knowledge.\n"
    "Return only a JSON object: {{\"claims\": [{{\"claim\": \"...\", \"correct\": true|false}}]}}."
)
CLAIM_EXTRACTION_USER_PROMPT = "Question:\n{question}\n\nResponse:\n{response}"
