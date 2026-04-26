from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

from pipeline.task_aware_compression import (
    COMPLEX_QTYPES,
    EXAMPLE_HINTS,
    EXPLANATORY_EVIDENCE_HINTS,
    OUTCOME_HINTS,
    contains_any,
    detect_question_type,
    question_seeks_outcome_examples,
    query_overlap_score,
    task_anchor_score,
    tokenize_query_terms,
)


ANSWER_STYLE_BY_QTYPE = {
    "cause": "Keep causes, mechanisms, consequences, and evidence needed for an explanatory answer.",
    "comparison": "Keep both sides of the comparison, the comparison dimensions, and the conclusion.",
    "procedure": "Keep ordered steps, preconditions, constraints, and outcomes.",
    "numeric": "Keep numbers, units, thresholds, ranges, and qualifiers.",
    "definition": "Keep the definition core, key attributes, and boundary conditions.",
    "factoid": "Keep entities, dates, locations, names, and identity clues.",
    "other": "Keep the most question-relevant evidence and remove background that does not support the answer.",
}


REASONING_FOCUS_BY_QTYPE = {
    "cause": ["cause", "mechanism", "effect", "risk", "consequence", "evidence", "example"],
    "comparison": ["entity", "dimension", "difference", "conclusion"],
    "procedure": ["step", "order", "condition", "result"],
    "numeric": ["number", "unit", "threshold", "range"],
    "definition": ["definition", "attribute", "scope"],
    "factoid": ["entity", "time", "place", "identity"],
    "other": ["claim", "evidence", "constraint"],
}


@dataclass
class TaskDescriptor:
    question_type: str
    answer_style: str
    reasoning_focus: List[str]
    key_terms: List[str]
    constraints: List[str]
    descriptor_text: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def build_task_descriptor(question: str) -> TaskDescriptor:
    question_type = detect_question_type(question)
    key_terms = tokenize_query_terms(question)
    reasoning_focus = list(REASONING_FOCUS_BY_QTYPE.get(question_type, REASONING_FOCUS_BY_QTYPE["other"]))
    constraints = ["do not introduce facts not present in the context", "minimize tokens", "keep fluent English"]

    if question_type in COMPLEX_QTYPES:
        constraints.append("preserve the reasoning chain, not only isolated evidence fragments")
    if question_seeks_outcome_examples(question):
        constraints.append("preserve outcome examples and consequence-bearing clauses")
        reasoning_focus.extend(["outcome", "example"])
    if question_type == "numeric":
        constraints.append("do not drop numbers or units")
    if question_type == "factoid":
        constraints.append("do not drop dates, places, names, or key entities")

    style = ANSWER_STYLE_BY_QTYPE.get(question_type, ANSWER_STYLE_BY_QTYPE["other"])
    descriptor_text = (
        f"question_type: {question_type}; "
        f"answer_goal: {style}; "
        f"reasoning_focus: {'/'.join(reasoning_focus)}; "
        f"key_terms: {', '.join(key_terms[:8]) if key_terms else 'none'}; "
        f"constraints: {'; '.join(constraints)}"
    )

    return TaskDescriptor(
        question_type=question_type,
        answer_style=style,
        reasoning_focus=reasoning_focus,
        key_terms=key_terms,
        constraints=constraints,
        descriptor_text=descriptor_text,
    )


def compute_task_descriptor_alignment(descriptor: TaskDescriptor, text: str) -> float:
    if not text.strip():
        return 0.0

    pseudo_question = " ".join(descriptor.key_terms + descriptor.reasoning_focus)
    overlap = query_overlap_score(pseudo_question, text)
    anchor = task_anchor_score(pseudo_question, text, descriptor.question_type)
    lowered = text.lower()
    reasoning_hits = sum(1 for focus in descriptor.reasoning_focus if focus and focus in lowered)
    reasoning_score = reasoning_hits / max(len(descriptor.reasoning_focus), 1)

    structure_bonus = 0.0
    if descriptor.question_type == "cause":
        if contains_any(lowered, OUTCOME_HINTS) or contains_any(lowered, EXPLANATORY_EVIDENCE_HINTS):
            structure_bonus += 0.16
        if contains_any(lowered, EXAMPLE_HINTS) or "include" in lowered:
            structure_bonus += 0.12
    if descriptor.question_type == "comparison" and contains_any(lowered, ("however", "whereas", "contrast", "than")):
        structure_bonus += 0.12
    if descriptor.question_type == "numeric" and any(ch.isdigit() for ch in text):
        structure_bonus += 0.14

    score = 0.40 * overlap + 0.32 * anchor + 0.18 * reasoning_score + min(structure_bonus, 0.24)
    return float(min(max(score, 0.0), 1.0))
