"""The restore drill reads a backup back the way the box would, and says when it cannot.

Every artifact here is made by scripts/backup.py's own functions from a WAL
file opened through the factory, as local.db is on the box, so the drill is
tested against what the nightly job actually ships, not a hand-made stand-in.
"""

from __future__ import annotations

import gzip
import io
import json
import sqlite3
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts import backup, restore_drill
from tradingagents_us.schemas import AgentDecision
from tradingagents_us.storage import TradeLogRepository, make_engine

NOW = datetime(2026, 10, 16, 6, 0, tzinfo=UTC)
RUN = NOW - timedelta(hours=6)


def _decision(decision_id: str, at: datetime) -> AgentDecision:
    return AgentDecision(
        ticker="UNH", market="US", quote_currency="USD", rating="Hold", reasoning=[],
        timestamp_utc=at, decision_id=decision_id, final_decision_text="HOLD UNH",
    )


@pytest.fixture
def live_db(tmp_path: Path) -> Path:
    db = tmp_path / "live" / "local.db"
    db.parent.mkdir()
    repo = TradeLogRepository(engine=make_engine(f"sqlite:///{db}"))
    repo.save_decision(_decision("dec-1", RUN))
    repo.engine.dispose()
    return db


def _write_artifacts(dest: Path, db: Path, memory_text: str | None = None) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "local.db.gz").write_bytes(backup._dump_sqlite_gz(db))
    snapshot = {"ts": RUN.isoformat(), "equity": 112_586.1, "n_positions": 7}
    (dest / "eval_snapshots.jsonl.gz").write_bytes(
        gzip.compress((json.dumps(snapshot) + "\n").encode())
    )
    memory = dest.parent / "memory"
    memory.mkdir(exist_ok=True)
    if memory_text is None:
        memory_text = (
            "[2026-10-15 | UNH | Hold | pending]\n\nDECISION:\nHOLD UNH\n\n<!-- ENTRY_END -->\n\n"
        )
    (memory / "trading_memory.md").write_text(memory_text, encoding="utf-8")
    (dest / "memory.tar.gz").write_bytes(backup._tar_gz_dir(memory))
    return dest


@pytest.fixture
def current(tmp_path: Path, live_db: Path) -> Path:
    return _write_artifacts(tmp_path / "backups" / "current", live_db)


def _drill(source: Path, *, live_db: Path | None = None, now: datetime = NOW) -> int:
    return restore_drill.run(source, live_db=live_db, max_age_hours=96, now=now)


def test_last_nights_backup_restores_and_reads_back(
    current: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _drill(current) == 0
    out = capsys.readouterr().out
    assert "integrity_check: ok" in out
    assert "one self-contained file (rollback journal" in out
    assert "agent_decisions: 1 rows, newest 2026-10-16T00:00Z (6.0h old)" in out
    assert "1 entries (1 pending), newest 2026-10-15" in out
    assert "RESTORE DRILL OK" in out


def test_the_inputs_are_only_read(current: Path) -> None:
    before = {p.name: p.read_bytes() for p in current.iterdir()}
    _drill(current)
    assert {p.name: p.read_bytes() for p in current.iterdir()} == before


@pytest.mark.parametrize("artifact", ["local.db.gz", "eval_snapshots.jsonl.gz", "memory.tar.gz"])
def test_a_target_the_backup_skipped_fails_the_drill(
    current: Path, artifact: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # backup.py logs "skipped" and exits 0 for a missing target; this is
    # where that silence gets caught.
    (current / artifact).unlink()
    assert _drill(current) == 1
    assert f"FAIL  {artifact}: not in the backup" in capsys.readouterr().out


def test_a_truncated_db_artifact_fails(current: Path, capsys: pytest.CaptureFixture[str]) -> None:
    blob = (current / "local.db.gz").read_bytes()
    (current / "local.db.gz").write_bytes(blob[: len(blob) // 2])
    assert _drill(current) == 1
    assert "local.db.gz does not gunzip" in capsys.readouterr().out


def test_a_feed_row_that_left_the_box_fails(
    tmp_path: Path, live_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stamp = RUN.isoformat(sep=" ")
    conn = sqlite3.connect(live_db)
    with conn:
        conn.execute(
            "INSERT INTO commentator_items (item_id, source, source_id, channel_id, url,"
            " published_at_utc, fetched_at_utc, expires_at_utc, content_sha256, claim_en,"
            " tickers_json, macro_topics_json, stance_json)"
            " VALUES ('youtube:abc', 'youtube', 'abc', 'c', 'u', ?, ?, ?, 'h', 'claim',"
            " '[]', '[]', '{}')",
            (stamp, stamp, stamp),
        )
    conn.close()
    monkeypatch.setattr(backup, "_scrub_for_off_box", lambda conn: None)
    current = _write_artifacts(tmp_path / "backups" / "current", live_db)
    assert _drill(current) == 1
    assert "commentator feed left the box (ADR-009): commentator_items 1" in capsys.readouterr().out


def test_a_wal_artifact_fails(
    tmp_path: Path, live_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(backup, "_single_file", lambda conn: None)
    current = _write_artifacts(tmp_path / "backups" / "current", live_db)
    assert _drill(current) == 1
    assert "journal_mode=wal" in capsys.readouterr().out


def test_an_older_file_is_upgraded_on_open_and_says_so(
    tmp_path: Path, live_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A backup taken before a table and an additive column existed: the box
    # builds both on first open, so it still restores — the drill names them.
    conn = sqlite3.connect(live_db)
    with conn:
        conn.execute("DROP TABLE sector_cache")
        conn.execute("ALTER TABLE agent_decisions DROP COLUMN cost_usd")
    conn.close()
    current = _write_artifacts(tmp_path / "backups" / "current", live_db)
    assert _drill(current) == 0
    out = capsys.readouterr().out
    assert "1 table(s) created, 1 column(s) added (sector_cache, agent_decisions.cost_usd)" in out


def test_a_row_the_app_cannot_read_fails(
    tmp_path: Path, live_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    conn = sqlite3.connect(live_db)
    with conn:
        conn.execute("UPDATE agent_decisions SET timestamp_utc = 'not-a-date'")
    conn.close()
    current = _write_artifacts(tmp_path / "backups" / "current", live_db)
    assert _drill(current) == 1
    out = capsys.readouterr().out
    assert "rows the app cannot read: agent_decisions" in out
    assert "no row to date the backup by" in out


def test_a_stale_backup_fails(current: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _drill(current, now=NOW + timedelta(days=4)) == 1
    out = capsys.readouterr().out
    assert "agent_decisions: 1 rows, newest 2026-10-16T00:00Z (102.0h old) — older than 96h" in out
    assert "eval snapshots, newest 2026-10-16T00:00Z (102.0h old) — older than 96h" in out


def test_a_torn_snapshot_line_fails(current: Path, capsys: pytest.CaptureFixture[str]) -> None:
    text = gzip.decompress((current / "eval_snapshots.jsonl.gz").read_bytes()).decode()
    (current / "eval_snapshots.jsonl.gz").write_bytes(gzip.compress((text + '{"ts": "2').encode()))
    assert _drill(current) == 1
    assert "1 line(s) do not parse (first: line 2)" in capsys.readouterr().out


def test_an_unsafe_tar_member_is_refused_unextracted(
    current: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        payload = b"overwritten"
        info = tarfile.TarInfo("../escaped.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    (current / "memory.tar.gz").write_bytes(buf.getvalue())
    assert _drill(current) == 1
    assert "refusing to extract unsafe member(s): ../escaped.txt" in capsys.readouterr().out
    assert not list(tmp_path.rglob("escaped.txt"))


def test_a_memory_log_the_agent_cannot_parse_fails(
    tmp_path: Path, live_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    current = _write_artifacts(
        tmp_path / "backups" / "current", live_db, memory_text="not a decision log\n"
    )
    assert _drill(current) == 1
    assert "holds no entry the agent can parse" in capsys.readouterr().out


def test_the_live_gap_is_what_a_restore_would_lose(
    current: Path, live_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = TradeLogRepository(engine=make_engine(f"sqlite:///{live_db}"))
    repo.save_decision(_decision("dec-2", NOW - timedelta(hours=1)))
    repo.engine.dispose()
    assert _drill(current, live_db=live_db) == 0
    row = next(ln for ln in capsys.readouterr().out.splitlines() if "agent_decisions " in ln)
    assert row.split() == ["agent_decisions", "1", "2", "+1"]


def test_cli_rejects_a_source_that_is_not_a_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert restore_drill.main([str(tmp_path / "nope")]) == 1
    assert "is not a directory" in capsys.readouterr().err
