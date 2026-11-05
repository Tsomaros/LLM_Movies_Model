"""Award families (Oscar, BAFTA, ...) from Wikidata award QIDs listed in awards.json."""

import json
from pathlib import Path

# Fixed order; it breaks ties when an award's parent classes match two families.
FAMILIES = ("oscar", "bafta_film", "golden_globe", "cannes", "venice", "berlin", "sag")

_DATA = json.loads(Path(__file__).with_name("awards.json").read_text(encoding="utf-8"))
_FAMILY_OF = {item["qid"]: family for family, items in _DATA.items() for item in items}


def award_family(award_qid: str, parent_qids: set[str]) -> str | None:
    """Family of the award item itself, else of its parent classes, else None."""
    if award_qid in _FAMILY_OF:
        return _FAMILY_OF[award_qid]
    parent_families = {_FAMILY_OF[q] for q in parent_qids if q in _FAMILY_OF}
    return next((f for f in FAMILIES if f in parent_families), None)
