import pytest

from llm_movies.data import source_material
from llm_movies.data.source_material import (
    SOURCE_TYPES,
    source_category,
    source_from_keywords,
    source_from_wikidata,
)

KEYWORD_TABLE = [
    ("based on novel or book", "novel"),
    ("based on young adult novel", "novel"),
    ("based on comic", "comic"),
    ("superhero", "comic"),
    ("based on true story", "true story"),
    ("biography", "true story"),
    ("remake", "remake"),
    ("live action remake", "remake"),
    ("based on video game", "video game"),
    ("based on play or musical", "play"),
    ("based on tv series", "TV series"),
    ("based on manga", "manga"),
]


@pytest.mark.parametrize(("keyword", "category"), KEYWORD_TABLE)
def test_keyword_maps_to_category(keyword, category):
    assert source_from_keywords({keyword}) == category


def test_every_category_is_a_source_type():
    for _, category in KEYWORD_TABLE:
        assert category in SOURCE_TYPES


# Real P144 target classes seen on Wikidata (label in the comment), one per category.
@pytest.mark.parametrize(
    ("qid", "category"),
    [
        ("Q7725634", "novel"),  # literary work (Dune, The Shining, Les Miserables)
        ("Q8261", "novel"),  # novel
        ("Q21198342", "manga"),  # manga series (Akira, Ghost in the Shell)
        ("Q7889", "video game"),  # video game (Warcraft III)
        ("Q141546978", "video game"),  # video game franchise (Sonic, Mario)
        ("Q5398426", "TV series"),  # television series
        ("Q58483083", "play"),  # dramatico-musical work (Hairspray)
        ("Q14406742", "comic"),  # comic book series
        ("Q5", "true story"),  # human
        ("Q11424", "remake"),  # film
    ],
)
def test_wikidata_class_maps_to_category(qid, category):
    assert source_from_wikidata({qid}) == category


def test_wikidata_wins_over_keywords():
    assert source_category({"Q7889"}, {"based on novel or book"}) == "video game"


def test_keywords_used_when_wikidata_has_no_signal():
    assert source_category(set(), {"based on novel or book"}) == "novel"
    assert source_category({"Q1"}, {"based on comic"}) == "comic"


def test_no_signal_is_original():
    assert source_category(set(), set()) == "original"
    assert source_category({"Q1"}, {"heist"}) == "original"


def test_unknown_or_empty_gives_none():
    assert source_from_wikidata({"Q1"}) is None
    assert source_from_wikidata(set()) is None
    assert source_from_keywords({"heist"}) is None
    assert source_from_keywords(set()) is None


def test_keyword_matching_is_exact():
    assert source_from_keywords({"Based on Comic"}) is None
    assert source_from_keywords({"comic"}) is None


def test_keyword_priority():
    assert source_from_keywords({"based on comic", "based on manga"}) == "manga"
    assert source_from_keywords({"biography", "based on novel or book"}) == "novel"
    assert source_from_keywords({"remake", "based on true story"}) == "true story"


# Ruling (progress.md, Task 5 pre-flight): manga, video game, TV series, play, comic, novel, true story, remake.
RULED_ORDER = ["manga", "video game", "TV series", "play", "comic", "novel", "true story", "remake"]
KEYWORD_OF = {
    "manga": "based on manga",
    "video game": "based on video game",
    "TV series": "based on tv series",
    "play": "based on play or musical",
    "comic": "based on comic",
    "novel": "based on novel or book",
    "true story": "based on true story",
    "remake": "remake",
}
QID_OF = {
    "manga": "Q21198342",
    "video game": "Q7889",
    "TV series": "Q5398426",
    "play": "Q58483083",
    "comic": "Q14406742",
    "novel": "Q8261",
    "true story": "Q5",
    "remake": "Q11424",
}
PAIRS = [(RULED_ORDER[i], RULED_ORDER[j]) for i in range(len(RULED_ORDER)) for j in range(i + 1, len(RULED_ORDER))]


@pytest.mark.parametrize(("winner", "loser"), PAIRS)
def test_ruled_priority_for_every_pair(winner, loser):
    assert source_from_keywords({KEYWORD_OF[winner], KEYWORD_OF[loser]}) == winner
    assert source_from_wikidata({QID_OF[winner], QID_OF[loser]}) == winner


def test_keyword_table_is_exactly_the_brief():
    # "using exactly": an extra or renamed keyword in the module must fail here.
    assert source_material._KEYWORDS == dict(KEYWORD_TABLE)


def test_every_wikidata_class_gives_a_source_type():
    assert set(source_material._WIKIDATA_CLASSES.values()) <= set(SOURCE_TYPES) - {"original"}


def test_wikidata_priority():
    assert source_from_wikidata({"Q7725634", "Q21198342"}) == "manga"  # literary work + manga series
    assert source_from_wikidata({"Q5", "Q8261"}) == "novel"  # human + novel
    assert source_from_wikidata({"Q11424", "Q5"}) == "true story"  # film + human
    assert source_from_wikidata({"Q7889", "Q5398426"}) == "video game"
