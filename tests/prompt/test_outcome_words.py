import pytest
from test_render import film

from llm_movies.prompt.outcome_words import MASK, OUTCOME_PATTERNS, mask_outcomes
from llm_movies.prompt.render import render

SENTENCES = {
    "box office": "It ruled the box office.",
    "gross, grossed, grossing": "It grossed a lot.",
    "Oscar": "An Oscar contender.",
    "Academy Award": "An Academy Award nominee.",
    "award-winning": "An award-winning drama.",
    "blockbuster": "A summer blockbuster.",
    "Golden Globe": "A Golden Globe favourite.",
    "record-breaking": "A record-breaking run.",
    "smash hit": "A smash hit abroad.",
}
PHRASES = {
    "box office": "box office", "gross, grossed, grossing": "grossed", "Oscar": "Oscar", "Academy Award": "Academy Award",
    "award-winning": "award-winning", "blockbuster": "blockbuster", "Golden Globe": "Golden Globe",
    "record-breaking": "record-breaking", "smash hit": "smash hit",
}


def test_mask_is_the_ellipsis_in_brackets():
    assert MASK == "[…]"


@pytest.mark.parametrize("label", [label for label, _ in OUTCOME_PATTERNS])
def test_each_pattern_masked(label):
    out = mask_outcomes(SENTENCES[label])
    assert MASK in out
    assert PHRASES[label] not in out


def test_every_pattern_has_a_sentence():
    assert {label for label, _ in OUTCOME_PATTERNS} == set(SENTENCES)
    assert "dollar figure" not in SENTENCES


def test_case_and_hyphen_variants():
    assert mask_outcomes("An Award Winning film") == f"An {MASK} film"
    assert mask_outcomes("an award winning film") == f"an {MASK} film"
    assert mask_outcomes("The Box-Office king") == f"The {MASK} king"


def test_word_bounds():
    for text in ("grossly overrated", "Oscarsson walked in", "engrossing"):
        assert mask_outcomes(text) == text


def test_dollar_figures_not_masked():
    text = "Made on a $5 million budget"
    assert mask_outcomes(text) == text


def test_none_passthrough():
    assert mask_outcomes(None) is None


def test_render_masks_tagline_and_synopsis():
    text = render(film(tagline="The Oscar blockbuster.", synopsis="It grossed big. " + "word " * 200))
    assert f"Tagline: The {MASK} {MASK}." in text
    assert f"Synopsis: It {MASK} big." in text
    assert "Oscar" not in text and "grossed" not in text


def test_render_masks_before_the_word_cut():
    # "box office" is two words masked into one token; masking first means the cut counts the mask as one word
    synopsis = "box office " + " ".join(f"w{i}" for i in range(200))
    out = render(film(synopsis=synopsis)).splitlines()[-1]
    assert out.startswith(f"Synopsis: {MASK} w0 ")
    assert len(out.removeprefix("Synopsis: ").split()) == 120


def test_phrase_straddling_the_cut_is_still_masked():
    # "box" is word 120 and "office" word 121: cutting first would leave a bare "box..." and the phrase would escape the mask;
    # masking first folds both words into one MASK token, which is word 120.
    synopsis = " ".join(f"w{i}" for i in range(119)) + " box office tail words"
    out = render(film(synopsis=synopsis)).splitlines()[-1]
    assert out.endswith(f" w118 {MASK}...")
    assert "box" not in out


def test_overlapping_phrases_leave_no_fragment():
    assert mask_outcomes("An Academy Award-winning drama.") == f"An {MASK} drama."
    assert mask_outcomes("An Academy Award winning drama.") == f"An {MASK} drama."
    assert mask_outcomes("An Oscar-winning drama.") == f"An {MASK} drama."
    assert mask_outcomes("A Golden Globe winning drama.") == f"A {MASK} drama."
