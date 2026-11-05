"""Source-material category of a film, from Wikidata classes and TMDB keywords."""

SOURCE_TYPES = ("original", "novel", "comic", "true story", "remake", "video game", "play", "TV series", "manga")

# One fixed order for both lookups; the most specific category wins.
_PRIORITY = ("manga", "video game", "TV series", "play", "comic", "novel", "true story", "remake")

# Wikidata class (P31 of a film's "based on" P144 target) -> category. Checked live on wikidata.org.
_WIKIDATA_CLASSES = {
    "Q8274": "manga",  # manga
    "Q21198342": "manga",  # manga series
    "Q7889": "video game",  # video game
    "Q7058673": "video game",  # video game series
    "Q141546978": "video game",  # video game franchise
    "Q5398426": "TV series",  # television series
    "Q117467246": "TV series",  # animated television series
    "Q63952888": "TV series",  # anime television series
    "Q1259759": "TV series",  # miniseries
    "Q3297186": "TV series",  # limited series
    "Q25379": "play",  # play
    "Q2743": "play",  # musical play
    "Q58483083": "play",  # dramatico-musical work
    "Q116476516": "play",  # dramatic work
    "Q1004": "comic",  # comic
    "Q14406742": "comic",  # comic book series
    "Q725377": "comic",  # graphic novel
    "Q1760610": "comic",  # comic book
    "Q2831984": "comic",  # comic book album
    "Q838795": "comic",  # comic strip
    "Q213369": "comic",  # webcomic
    "Q8261": "novel",  # novel
    "Q49084": "novel",  # short story
    "Q149537": "novel",  # novella
    "Q12799318": "novel",  # short novel
    "Q571": "novel",  # book
    "Q7725634": "novel",  # literary work
    "Q47461344": "novel",  # written work
    "Q277759": "novel",  # book series
    "Q1667921": "novel",  # novel series
    "Q59126": "novel",  # xiaoshuo
    "Q747381": "novel",  # light novel
    "Q5": "true story",  # human
    "Q11424": "remake",  # film
    "Q202866": "remake",  # animated film
}

# TMDB keyword name -> category. Matching is exact.
_KEYWORDS = {
    "based on novel or book": "novel",
    "based on young adult novel": "novel",
    "based on comic": "comic",
    "superhero": "comic",
    "based on true story": "true story",
    "biography": "true story",
    "remake": "remake",
    "live action remake": "remake",
    "based on video game": "video game",
    "based on play or musical": "play",
    "based on tv series": "TV series",
    "based on manga": "manga",
}


def _first_by_priority(categories: set[str]) -> str | None:
    return next((c for c in _PRIORITY if c in categories), None)


def source_from_wikidata(based_on_types: set[str]) -> str | None:
    """Category from the P31 classes (QIDs) of a film's P144 target, or None."""
    return _first_by_priority({_WIKIDATA_CLASSES[q] for q in based_on_types if q in _WIKIDATA_CLASSES})


def source_from_keywords(names: set[str]) -> str | None:
    return _first_by_priority({_KEYWORDS[n] for n in names if n in _KEYWORDS})


def source_category(wd: set[str], keywords: set[str]) -> str:
    """Wikidata first, then keywords, else "original"."""
    return source_from_wikidata(wd) or source_from_keywords(keywords) or "original"
