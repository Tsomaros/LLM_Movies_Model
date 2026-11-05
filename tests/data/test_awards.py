import json
import re
from pathlib import Path

import pytest

from llm_movies.data import awards
from llm_movies.data.awards import award_family

FAMILIES = ["oscar", "bafta_film", "golden_globe", "cannes", "venice", "berlin", "sag"]
DATA = json.loads(Path(awards.__file__).with_name("awards.json").read_text(encoding="utf-8"))


def test_awards_json_has_seven_families_with_labels():
    assert set(DATA) == set(FAMILIES)
    for family, items in DATA.items():
        assert items, family
        for item in items:
            assert set(item) == {"qid", "label"}
            assert re.fullmatch(r"Q\d+", item["qid"]), item
            assert item["label"].strip(), item


def test_no_qid_in_two_families():
    qids = [item["qid"] for items in DATA.values() for item in items]
    assert len(qids) == len(set(qids))


def test_film_families_have_no_tv_labels():
    for family in ("golden_globe", "bafta_film", "sag"):
        for item in DATA[family]:
            assert not re.search(r"Television|Series|Miniseries", item["label"]), item


def test_families_are_sorted_by_label():
    for items in DATA.values():
        labels = [i["label"] for i in items]
        assert labels == sorted(labels)


# Category items checked on Wikidata (label in the comment): matches on the item alone.
@pytest.mark.parametrize(
    ("qid", "family"),
    [
        ("Q102427", "oscar"),  # Academy Award for Best Picture
        ("Q139184", "bafta_film"),  # BAFTA Award for Best Film
        ("Q1011509", "golden_globe"),  # Golden Globe Award for Best Motion Picture - Drama
        ("Q179808", "cannes"),  # Palme d'Or
        ("Q209459", "venice"),  # Golden Lion
        ("Q154590", "berlin"),  # Golden Bear
        ("Q518675", "sag"),  # Actor Award for Outstanding Performance by a Cast in a Motion Picture
    ],
)
def test_award_family_matches_item_itself(qid, family):
    assert award_family(qid, set()) == family


def test_award_family_matches_parent_class():
    assert award_family("Q999999999", {"Q19020"}) == "oscar"  # Academy Awards
    assert award_family("Q999999999", {"Q1", "Q732997"}) == "bafta_film"


def test_item_wins_over_parents():
    assert award_family("Q139184", {"Q19020"}) == "bafta_film"


def test_parents_in_two_families_use_fixed_order():
    assert award_family("Q999999999", {"Q732997", "Q19020"}) == "oscar"
    assert award_family("Q999999999", {"Q268200", "Q1011547"}) == "golden_globe"


def test_unknown_gives_none():
    assert award_family("Q999999999", set()) is None
    assert award_family("Q999999999", {"Q1"}) is None
