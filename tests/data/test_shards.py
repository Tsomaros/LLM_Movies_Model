import json
from pathlib import Path

import pytest

from llm_movies.data.shards import Manifest, ShardWriter, done_keys, read_shard


def _two_records_then(tmp_path: Path, tail: bytes) -> Path:
    path = tmp_path / "shard.jsonl"
    with ShardWriter(path) as w:
        w.write({"id": 1, "title": "A"})
        w.write({"id": 2, "title": "B"})
    with open(path, "ab") as f:
        f.write(tail)
    return path


def test_roundtrip(tmp_path):
    path = tmp_path / "sub" / "shard.jsonl"
    records = [{"id": 1, "title": "Amélie"}, {"id": 2, "title": "B"}]
    with ShardWriter(path) as w:
        for r in records:
            w.write(r)
    assert list(read_shard(path)) == records
    assert done_keys(path, "id") == {1, 2}
    assert b"\r" not in path.read_bytes()


def test_truncated_last_line_is_skipped(tmp_path):
    path = _two_records_then(tmp_path, b'{"id": 3, "ti')
    assert [r["id"] for r in read_shard(path)] == [1, 2]
    assert done_keys(path, "id") == {1, 2}


def test_writer_after_truncated_tail_keeps_next_record(tmp_path):
    path = _two_records_then(tmp_path, b'{"id": 3, "ti')
    with ShardWriter(path) as w:
        w.write({"id": 3, "title": "C"})
    assert [r["id"] for r in read_shard(path)] == [1, 2, 3]
    assert done_keys(path, "id") == {1, 2, 3}


def test_unterminated_valid_json_is_not_counted(tmp_path):
    path = _two_records_then(tmp_path, b'{"id": 3}')
    assert [r["id"] for r in read_shard(path)] == [1, 2]
    assert done_keys(path, "id") == {1, 2}


def test_invalid_line_in_the_middle_raises(tmp_path):
    path = tmp_path / "shard.jsonl"
    path.write_bytes(b'{"id": 1}\nnot json\n{"id": 3}\n')
    with pytest.raises(ValueError, match=r"shard\.jsonl.*2"):
        list(read_shard(path))


@pytest.mark.parametrize("tail_len", [65_535, 65_536, 65_537, 3 * 65_536 + 5])
def test_writer_cuts_a_tail_longer_than_the_scan_block(tmp_path, tail_len):
    # Tail lengths around and past the 64 KiB backward-scan block, with no newline inside.
    path = _two_records_then(tmp_path, b'{"id": 3, "t": "' + b"x" * tail_len)
    with ShardWriter(path) as w:
        w.write({"id": 3, "title": "C"})
    assert [r["id"] for r in read_shard(path)] == [1, 2, 3]
    assert path.read_bytes().count(b"\n") == 3


def test_writer_empties_a_file_that_is_only_a_partial_record(tmp_path):
    path = tmp_path / "shard.jsonl"
    path.write_bytes(b'{"id": 1, "ti')
    with ShardWriter(path) as w:
        w.write({"id": 1, "title": "A"})
    assert path.read_bytes() == b'{"id": 1, "title": "A"}\n'


def test_invalid_terminated_last_line_is_skipped(tmp_path):
    path = tmp_path / "shard.jsonl"
    path.write_bytes(b'{"id": 1}\n{"id": 2, "ti\n')
    assert [r["id"] for r in read_shard(path)] == [1]
    assert done_keys(path, "id") == {1}


def test_done_keys_raises_keyerror_on_record_without_key(tmp_path):
    path = tmp_path / "shard.jsonl"
    with ShardWriter(path) as w:
        w.write({"id": 1})
        w.write({"title": "no id"})
    with pytest.raises(KeyError):
        done_keys(path, "id")


def test_writer_cuts_an_invalid_terminated_last_line(tmp_path):
    path = tmp_path / "shard.jsonl"
    path.write_bytes(b'{"id": 1}\n{"id": 2, "ti\n')
    with ShardWriter(path) as w:
        w.write({"id": 2})
    assert [r["id"] for r in read_shard(path)] == [1, 2]


def test_writer_keeps_a_valid_terminated_last_line(tmp_path):
    path = tmp_path / "shard.jsonl"
    path.write_bytes(b'{"id": 1}\n{"id": 2}\n')
    with ShardWriter(path) as w:
        w.write({"id": 3})
    assert [r["id"] for r in read_shard(path)] == [1, 2, 3]


def test_done_keys_of_missing_file_is_empty(tmp_path):
    assert done_keys(tmp_path / "missing.jsonl", "id") == set()


def test_manifest_marks_and_survives_reload(tmp_path):
    path = tmp_path / "sub" / "manifest.json"
    m = Manifest(path)
    assert not m.is_done(0)
    m.mark_done(0, {"n": 5})
    assert m.is_done(0)
    assert not m.is_done(1)

    reloaded = Manifest(path)
    assert reloaded.is_done(0)
    assert not reloaded.is_done(1)
    assert json.loads(path.read_text(encoding="utf-8")) == {"0": {"n": 5}}
    assert [p.name for p in path.parent.iterdir()] == ["manifest.json"]


def test_manifest_write_is_atomic(tmp_path, monkeypatch):
    # If the final swap fails, the previous manifest must still be intact (temp file + os.replace).
    path = tmp_path / "manifest.json"
    m = Manifest(path)
    m.mark_done(0, {"n": 5})

    def boom(*_args):
        raise OSError("swap failed")

    monkeypatch.setattr("llm_movies.data.shards.os.replace", boom)
    with pytest.raises(OSError):
        m.mark_done(1, {"n": 7})
    assert json.loads(path.read_text(encoding="utf-8")) == {"0": {"n": 5}}
