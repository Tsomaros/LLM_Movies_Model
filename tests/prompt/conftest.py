import sqlite3
from datetime import date
from pathlib import Path

import pytest

from llm_movies.config import DataConfig
from llm_movies.data.build_db import build_db
from llm_movies.features.computed import Context

MINI = Path(__file__).parent.parent / "fixtures" / "mini"
PULLED_AT = date(2026, 11, 4)


@pytest.fixture(scope="session")
def mini_db(tmp_path_factory):
    out = tmp_path_factory.mktemp("prompt_mini") / "mini.db"
    build_db(MINI / "tmdb", MINI / "wd_films", MINI / "wd_people", out, PULLED_AT)
    return out


@pytest.fixture(scope="session")
def mini_con(mini_db):
    con = sqlite3.connect(mini_db)
    yield con
    con.close()


@pytest.fixture(scope="session")
def mini_ctx(mini_con):
    return Context.from_db(mini_con, DataConfig())
