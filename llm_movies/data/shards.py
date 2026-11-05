"""Resumable JSONL shards and an atomic chunk manifest.

A record is complete only once its newline is written, so a crash mid-write leaves at most one
unterminated tail line (or an invalid last line), which readers skip and the next writer cuts off.
"""

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Self

_BLOCK = 1 << 16


def _line_start(f, pos: int) -> int:
    """Offset just after the last newline before `pos` (0 if none), scanning backwards in blocks."""
    while pos > 0:
        start = max(0, pos - _BLOCK)
        f.seek(start)
        idx = f.read(pos - start).rfind(b"\n")
        if idx != -1:
            return start + idx + 1
        pos = start
    return 0


def _repair_tail(path: Path) -> None:
    """Match the reader: drop an unterminated last line, or a terminated last line that is invalid."""
    if not path.exists():
        return
    with open(path, "rb+") as f:
        end = f.seek(0, os.SEEK_END)
        if end == 0:
            return
        f.seek(end - 1)
        if f.read(1) != b"\n":
            f.truncate(_line_start(f, end))
            return
        start = _line_start(f, end - 1)
        f.seek(start)
        if _parse(f.read(end - start)) is None:
            f.truncate(start)


class ShardWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._file = None

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _repair_tail(self.path)
        self._file = open(self.path, "ab")
        return self

    def __exit__(self, *exc) -> None:
        self._file.close()
        self._file = None

    def write(self, record: dict) -> None:
        self._file.write(json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n")
        self._file.flush()


def _parse(raw: bytes) -> dict | None:
    try:
        record = json.loads(raw)
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


def read_shard(path: Path) -> Iterator[dict]:
    """Yield each complete record. A last line without a newline is never yielded; an invalid
    terminated last line is skipped; an invalid line elsewhere raises ValueError."""
    path = Path(path)
    with open(path, "rb") as f:
        prev = None
        for number, raw in enumerate(f, 1):
            if prev is not None:
                record = _parse(prev)
                if record is None:
                    raise ValueError(f"{path}: invalid record on line {number - 1}")
                yield record
            prev = raw
        if prev is not None and prev.endswith(b"\n"):
            record = _parse(prev)
            if record is not None:
                yield record


def done_keys(path: Path, key: str) -> set:
    if not Path(path).exists():
        return set()
    return {record[key] for record in read_shard(path)}


class Manifest:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._chunks: dict[str, dict] = {}
        if self.path.exists():
            self._chunks = json.loads(self.path.read_text(encoding="utf-8"))

    def is_done(self, chunk: int) -> bool:
        return str(chunk) in self._chunks

    def mark_done(self, chunk: int, stats: dict) -> None:
        self._chunks[str(chunk)] = stats
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self._chunks), encoding="utf-8")
        os.replace(tmp, self.path)
