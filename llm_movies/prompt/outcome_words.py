r"""Outcome phrases (box office, Oscars...) leak the target, so free text is masked before the model reads it.

    OUTCOME_PATTERNS            (label, compiled regex) pairs, word-bounded and case-insensitive
    mask_outcomes(text)         every match replaced by MASK, in one pass; None passes through

The data report audits with the same patterns, so the audit and the masking cannot drift apart. Dollar figures are
audited by the report but not masked: a plot may legitimately mention money.

Masking is one pass of one regex built from the same phrase sources, longest first, plus "<award name>[\s-]winning"
(Oscar, Academy Award, Golden Globe), so overlapping phrases give one MASK and no fragment like "[…]-winning".
"""

import re

MASK = "[…]"

_PATTERNS = (
    ("box office", r"\bbox[\s-]office\b"),
    ("gross, grossed, grossing", r"\bgross(?:ed|ing)?\b"),
    ("Oscar", r"\boscars?\b"),
    ("Academy Award", r"\bacademy awards?\b"),
    ("award-winning", r"\baward[\s-]winning\b"),
    ("blockbuster", r"\bblockbusters?\b"),
    ("Golden Globe", r"\bgolden globes?\b"),
    ("record-breaking", r"\brecord[\s-]breaking\b"),
    ("smash hit", r"\bsmash hits?\b"),
)
OUTCOME_PATTERNS = tuple((label, re.compile(pattern, re.IGNORECASE)) for label, pattern in _PATTERNS)
_AWARD_NAMES = ("Oscar", "Academy Award", "Golden Globe")
_COMPOUNDS = tuple(rf"(?:{dict(_PATTERNS)[name]})[\s-]winning\b" for name in _AWARD_NAMES)
_MASK_REGEX = re.compile(
    "|".join(sorted([*_COMPOUNDS, *(pattern for _, pattern in _PATTERNS)], key=len, reverse=True)), re.IGNORECASE
)


def mask_outcomes(text: str | None) -> str | None:
    if text is None:
        return None
    return _MASK_REGEX.sub(MASK, text)
