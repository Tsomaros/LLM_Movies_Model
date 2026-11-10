"""Outcome phrases (box office, Oscars...) leak the target, so free text is masked before the model reads it.

    OUTCOME_PATTERNS            (label, compiled regex) pairs, word-bounded and case-insensitive
    mask_outcomes(text)         every match replaced by MASK; None passes through

The data report audits with the same patterns, so the audit and the masking cannot drift apart. Dollar figures are
audited by the report but not masked: a plot may legitimately mention money.
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


def mask_outcomes(text: str | None) -> str | None:
    if text is None:
        return None
    for _, pattern in OUTCOME_PATTERNS:
        text = pattern.sub(MASK, text)
    return text
