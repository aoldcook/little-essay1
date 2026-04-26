from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Set


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*|\d+(?:\.\d+)?%?")
ENTITY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9-]{1,})(?:\s+(?:of|and|for|the|[A-Z][A-Za-z0-9-]{1,}))*\b"
    r"|\b[A-Z]{2,}(?:-\d+[A-Za-z]*)?\b"
    r"|\b\d+(?:\.\d+)?\s*(?:%|percent|degrees?|celsius|tokens?|b|m|k|gb|mb)?\b",
    re.IGNORECASE,
)
CODE_RE = re.compile(r"\b[\w.-]+\.(?:py|json|jsonl|md|txt|pdf)\b|[A-Za-z]:\\[^\s]+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "do",
    "does",
    "did",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

DISCOURSE_CUES: Dict[str, Sequence[str]] = {
    "definition": ("defined as", "refers to", "means", "called", "known as", "is a", "is an"),
    "causality": (
        "because",
        "since",
        "therefore",
        "thus",
        "hence",
        "lead to",
        "leads to",
        "result in",
        "results in",
        "due to",
        "driven by",
    ),
    "contrast": ("however", "but", "although", "whereas", "in contrast", "nevertheless"),
    "condition": ("if", "when", "unless", "provided that"),
    "conclusion": ("in summary", "overall", "finally", "therefore", "conclusion"),
    "temporal": ("before", "after", "then", "later", "initially", "subsequently"),
    "evidence": ("for example", "for instance", "such as", "evidence", "shown by", "include", "includes"),
}

PREDICATE_HINTS = {
    "affect",
    "asks",
    "becomes",
    "cause",
    "causes",
    "connect",
    "contains",
    "decline",
    "driven",
    "increase",
    "increases",
    "include",
    "includes",
    "lead",
    "leads",
    "limit",
    "preserve",
    "protect",
    "reduce",
    "reduces",
    "release",
    "releases",
    "report",
    "reports",
    "require",
    "requires",
    "result",
    "results",
    "shows",
    "uses",
}
NEGATION_MARKERS = {"no", "not", "never", "without", "cannot", "neither", "lack", "lacks"}
COMPARISON_MARKERS = {"more", "less", "higher", "lower", "better", "worse", "than", "at least", "at most", "beyond"}
CONSEQUENCE_TERMS = {
    "consequence",
    "consequences",
    "decline",
    "effect",
    "effects",
    "impact",
    "impacts",
    "loss",
    "risk",
    "risks",
    "severe",
    "stress",
    "trigger",
    "triggers",
}
PRONOUN_PATTERNS = (
    "it",
    "they",
    "this",
    "these",
    "those",
    "this method",
    "this approach",
    "the system",
    "the model",
)


@dataclass
class DiscourseCue:
    cue: str
    cue_type: str


@dataclass
class EntityMention:
    text: str
    start: int
    end: int
    is_number: bool = False
    is_code_like: bool = False


@dataclass
class SemanticUnit:
    text: str
    role: str


@dataclass
class LinguisticFeatureConfig:
    enable_discourse_centrality: bool = True
    enable_coreference_preservation: bool = True
    enable_predicate_argument_skeleton: bool = True
    enable_information_density: bool = True
    enable_redundancy_marginal_gain: bool = True
    enable_inter_sentence_dependency: bool = True
    discourse_weight: float = 0.18
    coref_weight: float = 0.14
    predicate_argument_weight: float = 0.20
    info_density_weight: float = 0.18
    redundancy_weight: float = 0.15
    dependency_weight: float = 0.12
    position_weight: float = 0.03


@dataclass
class SentenceLinguisticFeatures:
    sentence_id: int
    discourse_score: float = 0.0
    coref_score: float = 0.0
    predicate_argument_score: float = 0.0
    info_density: float = 0.0
    redundancy_marginal_gain: float = 0.0
    dependency_score: float = 0.0
    position_utility: float = 0.0
    final_score: float = 0.0
    discourse_types: List[str] = field(default_factory=list)
    matched_cues: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    semantic_units: List[Dict[str, str]] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def normalize_term(term: str) -> str:
    term = term.lower().strip("'\"")
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if len(term) > len(suffix) + 3 and term.endswith(suffix):
            return term[: -len(suffix)]
    return term


def tokenize(text: str) -> List[str]:
    return [tok for tok in (normalize_term(t) for t in TOKEN_RE.findall(text)) if tok and tok not in STOPWORDS]


def _contains_phrase(text: str, phrase: str) -> bool:
    lowered = text.lower()
    if " " in phrase:
        return phrase in lowered
    return re.search(rf"\b{re.escape(phrase)}\b", lowered) is not None


def descriptor_key_terms(descriptor: object | None) -> Set[str]:
    terms: Set[str] = set()
    if descriptor is None:
        return terms
    for attr in ("key_terms", "reasoning_focus"):
        values = getattr(descriptor, attr, []) or []
        for value in values:
            for token in tokenize(str(value)):
                terms.add(token)
    return terms


def detect_discourse_cues(sentence: str) -> List[DiscourseCue]:
    cues: List[DiscourseCue] = []
    for cue_type, phrases in DISCOURSE_CUES.items():
        for phrase in phrases:
            if _contains_phrase(sentence, phrase):
                cues.append(DiscourseCue(cue=phrase, cue_type=cue_type))
    return cues


def extract_entities_lightweight(text: str) -> List[EntityMention]:
    mentions: List[EntityMention] = []
    for match in ENTITY_RE.finditer(text):
        value = match.group(0).strip()
        if not value or value.lower() in STOPWORDS:
            continue
        mentions.append(
            EntityMention(
                text=value,
                start=match.start(),
                end=match.end(),
                is_number=bool(re.search(r"\d", value)),
                is_code_like=False,
            )
        )
    for match in CODE_RE.finditer(text):
        mentions.append(
            EntityMention(
                text=match.group(0).strip(),
                start=match.start(),
                end=match.end(),
                is_number=False,
                is_code_like=True,
            )
        )
    return mentions


def discourse_centrality_score(sentence: str, question_type: str, key_terms: Set[str]) -> tuple[float, List[DiscourseCue]]:
    cues = detect_discourse_cues(sentence)
    cue_types = {cue.cue_type for cue in cues}
    sentence_terms = set(tokenize(sentence))
    key_overlap = len(sentence_terms & key_terms) / max(len(key_terms), 1) if key_terms else 0.0

    score = 0.10 * len(cue_types) + 0.25 * key_overlap
    if question_type == "cause" and ({"causality", "evidence"} & cue_types):
        score += 0.25
    if question_type == "comparison" and "contrast" in cue_types:
        score += 0.25
    if question_type == "procedure" and "temporal" in cue_types:
        score += 0.25
    if question_type == "definition" and "definition" in cue_types:
        score += 0.25
    if re.search(r"\b(include|includes|such as|for example)\b", sentence.lower()):
        score += 0.20
    if sentence.strip().endswith(":"):
        score += 0.10
    return _clip01(score), cues


def _entity_norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def coreference_preservation_score(
    sentence: str,
    index: int,
    all_entities: Sequence[Sequence[EntityMention]],
    key_terms: Set[str],
    sentences: Sequence[str],
) -> float:
    entities = all_entities[index]
    if not entities:
        return 0.0

    first_mentions: Dict[str, int] = {}
    for sent_idx, mentions in enumerate(all_entities):
        for mention in mentions:
            first_mentions.setdefault(_entity_norm(mention.text), sent_idx)

    entity_terms = {token for mention in entities for token in tokenize(mention.text)}
    key_entity_overlap = len(entity_terms & key_terms) / max(len(key_terms), 1) if key_terms else 0.0
    first_count = sum(1 for mention in entities if first_mentions.get(_entity_norm(mention.text)) == index)

    score = 0.12 * min(len(entities), 3) + 0.24 * min(first_count, 2) + 0.30 * key_entity_overlap
    if index + 1 < len(sentences):
        next_lowered = sentences[index + 1].lower()
        if any(re.search(rf"\b{re.escape(pattern)}\b", next_lowered) for pattern in PRONOUN_PATTERNS):
            score += 0.18
    if re.search(r"\b(is a|is an|defined as|known as|called)\b", sentence.lower()):
        score += 0.18
    return _clip01(score)


def extract_predicate_argument_skeleton(sentence: str) -> List[SemanticUnit]:
    tokens = TOKEN_RE.findall(sentence)
    units: List[SemanticUnit] = []
    entities = extract_entities_lightweight(sentence)
    for mention in entities[:4]:
        role = "number_argument" if mention.is_number else "entity_argument"
        if mention.is_code_like:
            role = "code_or_file_argument"
        units.append(SemanticUnit(text=mention.text, role=role))

    lowered_tokens = [normalize_term(token) for token in tokens]
    for token in lowered_tokens:
        if token in PREDICATE_HINTS or token.endswith(("ed", "ing")):
            units.append(SemanticUnit(text=token, role="predicate"))

    lowered = sentence.lower()
    for marker in sorted(NEGATION_MARKERS | COMPARISON_MARKERS):
        if _contains_phrase(lowered, marker):
            units.append(SemanticUnit(text=marker, role="semantic_modifier"))
    return units


def predicate_argument_score(sentence: str, question_type: str, key_terms: Set[str]) -> tuple[float, List[SemanticUnit]]:
    units = extract_predicate_argument_skeleton(sentence)
    roles = {unit.role for unit in units}
    sentence_terms = set(tokenize(sentence))
    key_overlap = len(sentence_terms & key_terms) / max(len(key_terms), 1) if key_terms else 0.0
    consequence_overlap = len(sentence_terms & CONSEQUENCE_TERMS) / max(len(CONSEQUENCE_TERMS), 1)

    score = 0.12 * min(len(units), 4) + 0.28 * key_overlap
    if "predicate" in roles:
        score += 0.22
    if "entity_argument" in roles or "number_argument" in roles:
        score += 0.16
    if "semantic_modifier" in roles:
        score += 0.12
    if question_type == "cause" and consequence_overlap > 0:
        score += min(0.25, 1.5 * consequence_overlap)
    if re.search(r"\b(include|includes|such as)\b", sentence.lower()):
        score += 0.14
    return _clip01(score), units


def information_density_score(sentence: str, key_terms: Set[str]) -> float:
    tokens = TOKEN_RE.findall(sentence)
    if not tokens:
        return 0.0
    content = tokenize(sentence)
    content_set = set(content)
    entities = extract_entities_lightweight(sentence)
    cues = detect_discourse_cues(sentence)
    predicates = [tok for tok in content if tok in PREDICATE_HINTS or tok.endswith(("ed", "ing"))]
    key_hits = len(content_set & key_terms) if key_terms else 0
    numbers = sum(1 for mention in entities if mention.is_number)

    raw = (
        len(content)
        + 1.4 * min(len(entities), 5)
        + 1.2 * min(numbers, 3)
        + 1.6 * key_hits
        + 1.2 * len(cues)
        + 0.8 * min(len(predicates), 4)
    ) / max(len(tokens), 1)
    return _clip01(raw / 1.55)


def redundancy_marginal_gain_score(sentence: str, index: int, sentences: Sequence[str], key_terms: Set[str]) -> float:
    terms = set(tokenize(sentence))
    if not terms:
        return 0.0
    max_overlap = 0.0
    for other_idx, other in enumerate(sentences):
        if other_idx == index:
            continue
        other_terms = set(tokenize(other))
        if not other_terms:
            continue
        max_overlap = max(max_overlap, len(terms & other_terms) / max(len(terms | other_terms), 1))
    unique_ratio = 1.0 - max_overlap
    key_bonus = len(terms & key_terms) / max(len(key_terms), 1) if key_terms else 0.0
    consequence_bonus = min(0.25, len(terms & CONSEQUENCE_TERMS) * 0.06)
    return _clip01(0.65 * unique_ratio + 0.25 * key_bonus + consequence_bonus)


def inter_sentence_dependency_score(sentence: str, index: int, sentences: Sequence[str], cues: Sequence[DiscourseCue]) -> float:
    lowered = sentence.lower()
    score = 0.0
    if any(cue.cue_type in {"causality", "contrast", "condition", "temporal"} for cue in cues):
        score += 0.25
    if index > 0 and any(re.search(rf"\b{re.escape(pattern)}\b", lowered) for pattern in PRONOUN_PATTERNS):
        score += 0.18
    if index + 1 < len(sentences):
        next_terms = set(tokenize(sentences[index + 1]))
        this_terms = set(tokenize(sentence))
        if this_terms and len(this_terms & next_terms) / max(len(next_terms), 1) >= 0.25:
            score += 0.18
    if index > 0:
        prev_terms = set(tokenize(sentences[index - 1]))
        this_terms = set(tokenize(sentence))
        if this_terms and len(this_terms & prev_terms) / max(len(this_terms), 1) >= 0.25:
            score += 0.12
    if re.search(r"\b(these|this|therefore|however|such)\b", lowered):
        score += 0.10
    return _clip01(score)


def position_utility_score(index: int, num_sentences: int) -> float:
    if num_sentences <= 1:
        return 0.5
    if index == 0:
        return 0.70
    if index == num_sentences - 1:
        return 0.45
    center = 1.0 - abs(index - (num_sentences - 1) / 2.0) / max((num_sentences - 1) / 2.0, 1.0)
    return _clip01(0.35 + 0.25 * center)


def build_linguistic_sentence_features(
    question: str,
    sentences: Sequence[str],
    descriptor: object | None = None,
    config: Optional[LinguisticFeatureConfig] = None,
) -> List[SentenceLinguisticFeatures]:
    cfg = config or LinguisticFeatureConfig()
    question_type = str(getattr(descriptor, "question_type", "") or "")
    if not question_type:
        q = question.lower()
        if "why" in q or "cause" in q or "risk" in q:
            question_type = "cause"
        elif "compare" in q or "difference" in q:
            question_type = "comparison"
        elif "how" in q or "step" in q:
            question_type = "procedure"
        else:
            question_type = "other"
    key_terms = descriptor_key_terms(descriptor) | set(tokenize(question))
    all_entities = [extract_entities_lightweight(sentence) for sentence in sentences]

    features: List[SentenceLinguisticFeatures] = []
    for idx, sentence in enumerate(sentences):
        discourse_score, cues = discourse_centrality_score(sentence, question_type, key_terms)
        coref_score = coreference_preservation_score(sentence, idx, all_entities, key_terms, sentences)
        predicate_score, units = predicate_argument_score(sentence, question_type, key_terms)
        density_score = information_density_score(sentence, key_terms)
        redundancy_score = redundancy_marginal_gain_score(sentence, idx, sentences, key_terms)
        dependency_score = inter_sentence_dependency_score(sentence, idx, sentences, cues)
        position_score = position_utility_score(idx, len(sentences))

        weighted_parts = []
        if cfg.enable_discourse_centrality:
            weighted_parts.append((discourse_score, cfg.discourse_weight))
        if cfg.enable_coreference_preservation:
            weighted_parts.append((coref_score, cfg.coref_weight))
        if cfg.enable_predicate_argument_skeleton:
            weighted_parts.append((predicate_score, cfg.predicate_argument_weight))
        if cfg.enable_information_density:
            weighted_parts.append((density_score, cfg.info_density_weight))
        if cfg.enable_redundancy_marginal_gain:
            weighted_parts.append((redundancy_score, cfg.redundancy_weight))
        if cfg.enable_inter_sentence_dependency:
            weighted_parts.append((dependency_score, cfg.dependency_weight))
        weighted_parts.append((position_score, cfg.position_weight))

        denom = sum(weight for _, weight in weighted_parts)
        final = sum(score * weight for score, weight in weighted_parts) / max(denom, 1e-9)

        reasons: List[str] = []
        if discourse_score >= 0.45:
            reasons.append("discourse_centrality")
        if coref_score >= 0.45:
            reasons.append("entity_or_coreference_preservation")
        if predicate_score >= 0.55:
            reasons.append("predicate_argument_skeleton")
        if density_score >= 0.55:
            reasons.append("high_information_density")
        if redundancy_score >= 0.70:
            reasons.append("high_marginal_gain")
        if dependency_score >= 0.35:
            reasons.append("inter_sentence_dependency")

        features.append(
            SentenceLinguisticFeatures(
                sentence_id=idx,
                discourse_score=discourse_score,
                coref_score=coref_score,
                predicate_argument_score=predicate_score,
                info_density=density_score,
                redundancy_marginal_gain=redundancy_score,
                dependency_score=dependency_score,
                position_utility=position_score,
                final_score=_clip01(final),
                discourse_types=sorted({cue.cue_type for cue in cues}),
                matched_cues=[cue.cue for cue in cues],
                entities=[mention.text for mention in all_entities[idx]],
                semantic_units=[asdict(unit) for unit in units],
                reasons=reasons,
            )
        )
    return features
