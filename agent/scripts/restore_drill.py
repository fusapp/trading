#!/usr/bin/env python3
"""Restore drill — prove the off-box backup comes back as a working trade log.

scripts/backup.py has shipped three artifacts every night since 07-04 and
nothing ever read one back. A backup that was never restored is a hope, not a
capability (docs/review-e2e-2026-07-04.md, "tested restore drill"). This
restores each artifact into a scratch directory and checks it the way the box
would use it:

  local.db.gz             gunzips; PRAGMA integrity_check; one self-contained
                          file (rollback journal, as backup.py promises); no
                          commentator feed row (ADR-009 scrub held); then opened
                          the way the box opens it (TradeLogRepository:
                          create_all + ensure_additive_columns) and every mapped
                          row read back through the ORM — the "no such column"
                          trap a restored older file walks into
  eval_snapshots.jsonl.gz every line parses; newest `ts` and equity
  memory.tar.gz           only plain files and dirs, no absolute path or `..`;
                          extracts; the reflection log parses with the vendored
                          TradingMemoryLog

A missing artifact fails the drill: backup.py logs "skipped" and exits 0 when a
target is absent, so a lost target would otherwise go unnoticed for good.
Freshness: the newest decision and the newest snapshot must be younger than
--max-age-hours (default 96: a weekend plus a market holiday). With --live-db
the restored row counts are set against the live file, opened read-only; the
gap is what a restore would lose right now.

Read-only against its inputs. Artifacts are restored into a temporary
directory removed on exit and --live-db is opened with mode=ro; it never
writes to the box's DB, the backup clone, or the broker. Exit 0 when every
check passed, 1 when any failed.

    python -m scripts.restore_drill ~/trading-backups/current
    python -m scripts.restore_drill DIR --live-db /opt/ai-trader/agent/local.db
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent.parent
for _p in (_AGENT_ROOT, _AGENT_ROOT / "vendor" / "tradingagents"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sqlalchemy.engine import Connection, Engine  # noqa: E402
from sqlalchemy.exc import SQLAlchemyError  # noqa: E402

from tradingagents_us.storage import make_engine  # noqa: E402

DB_ARTIFACT = "local.db.gz"
SNAPSHOT_ARTIFACT = "eval_snapshots.jsonl.gz"
MEMORY_ARTIFACT = "memory.tar.gz"
MEMORY_LOG = "trading_memory.md"
DEFAULT_MAX_AGE_HOURS = 96.0
_SQLITE_MAGIC = b"SQLite format 3\x00"

#: The column each audit table is dated by, for its "newest row" line.
_NEWEST_COLUMN: dict[str, str] = {
    "agent_decisions": "timestamp_utc",
    "trade_orders": "submitted_at_utc",
    "order_updates": "timestamp_utc",
    "closed_trades": "closed_at_utc",
    "kill_switch_events": "timestamp_utc",
}

#: What backup.py's ADR-009 scrub must have left empty in the artifact.
_SCRUBBED: tuple[tuple[str, str], ...] = (
    ("commentator_items", "SELECT COUNT(*) FROM commentator_items"),
    ("commentator_status", "SELECT COUNT(*) FROM commentator_status"),
    (
        "decision_commentator_refs",
        "SELECT COUNT(*) FROM decision_commentator_refs WHERE item_id IS NOT NULL",
    ),
)


@dataclass
class Drill:
    now: datetime
    max_age_hours: float
    failures: list[str] = field(default_factory=list)

    def ok(self, msg: str) -> None:
        print(f"  ok    {msg}")

    def warn(self, msg: str) -> None:
        print(f"  warn  {msg}")

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        print(f"  FAIL  {msg}")

    def fresh(self, what: str, stamp: datetime | None) -> None:
        if stamp is None:
            self.fail(f"{what}: no row to date the backup by")
            return
        age_h = (self.now - stamp).total_seconds() / 3600
        line = f"{what}, newest {stamp:%Y-%m-%dT%H:%MZ} ({age_h:.1f}h old)"
        if age_h > self.max_age_hours:
            self.fail(f"{line} — older than {self.max_age_hours:g}h")
        else:
            self.ok(line)


def _parse_stamp(raw: object) -> datetime | None:
    """A stored DateTime as UTC. SQLite keeps SQLAlchemy's naive ISO text.

    None for a value that is not a date: the ORM read reports that row, and a
    freshness check with nothing to go by fails rather than crashing the drill.
    """
    if raw is None:
        return None
    try:
        stamp = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)


def _mib(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MiB"


def _table_counts(conn: Connection) -> dict[str, int]:
    names = [r[0] for r in conn.exec_driver_sql(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    )]
    return {n: conn.exec_driver_sql(f'SELECT COUNT(*) FROM "{n}"').scalar_one() for n in names}


def _read_only(path: Path) -> Engine:
    """The factory's read-only engine: busy timeout, and no journal switch."""
    return make_engine(f"sqlite:///file:{path}?mode=ro&uri=true")


def check_db(gz: Path, work: Path, drill: Drill) -> dict[str, int] | None:
    """Restore local.db; returns its per-table row counts when it opened at all."""
    restored = work / "local.db"
    try:
        restored.write_bytes(gzip.decompress(gz.read_bytes()))
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        drill.fail(f"{DB_ARTIFACT} does not gunzip: {exc}")
        return None
    print(f"{DB_ARTIFACT} ({_mib(gz.stat().st_size)} -> {_mib(restored.stat().st_size)})")

    with restored.open("rb") as f:
        header = f.read(100)
    if not header.startswith(_SQLITE_MAGIC):
        drill.fail("not a SQLite database")
        return None
    # Bytes 18-19 are the file format's write/read versions, 2 for WAL. Read
    # off the header because a writable open through the factory would switch
    # the file to WAL itself, and a read-only open of a WAL file needs the very
    # -shm/-wal sidecars the artifact never carries.
    if 2 in header[18:20]:
        drill.fail("journal_mode=wal: committed rows may sit in a -wal that was never shipped")
        return None
    drill.ok("one self-contained file (rollback journal, no -wal to ship)")

    engine = _read_only(restored)
    try:
        with engine.connect() as conn:
            verdict = [r[0] for r in conn.exec_driver_sql("PRAGMA integrity_check")]
            if verdict != ["ok"]:
                drill.fail(f"integrity_check: {'; '.join(verdict[:3])}")
                return None
            drill.ok("integrity_check: ok")
            counts = _table_counts(conn)
            leaked = [
                f"{table} {n}"
                for table, sql in _SCRUBBED
                if table in counts and (n := conn.exec_driver_sql(sql).scalar_one())
            ]
            newest = {
                table: _parse_stamp(
                    conn.exec_driver_sql(f"SELECT MAX({col}) FROM {table}").scalar_one()
                )
                for table, col in _NEWEST_COLUMN.items()
                if table in counts
            }
    except SQLAlchemyError as exc:
        drill.fail(f"does not open as SQLite: {exc}")
        return None
    finally:
        engine.dispose()
    if leaked:
        drill.fail(f"commentator feed left the box (ADR-009): {', '.join(leaked)}")
    else:
        drill.ok("commentator feed scrubbed (ADR-009)")

    _open_like_the_box(restored, work, drill)
    drill.fresh(f"agent_decisions: {counts.get('agent_decisions', 0)} rows",
                newest.get("agent_decisions"))
    for table in ("trade_orders", "order_updates", "closed_trades", "kill_switch_events"):
        if stamp := newest.get(table):
            drill.ok(f"{table}: {counts[table]} rows, newest {stamp:%Y-%m-%dT%H:%MZ}")
    return counts


def _open_like_the_box(restored: Path, work: Path, drill: Drill) -> None:
    """Open a copy through TradeLogRepository and read every mapped row back.

    That is what the API and the daily run do with the file once it is put in
    place: create_all builds any table the backup predates, ensure_additive_columns
    adds any column it predates, and only then does a SELECT name every mapped
    column. A copy, because both of those write.
    """
    from sqlalchemy import inspect, select
    from sqlalchemy.orm import Session

    from tradingagents_us.storage import TradeLogRepository
    from tradingagents_us.storage.models import Base

    opened = work / "opened.db"
    shutil.copyfile(restored, opened)
    engine = make_engine(f"sqlite:///{opened}")
    try:
        before = {
            t: {c["name"] for c in inspect(engine).get_columns(t)}
            for t in inspect(engine).get_table_names()
        }
        try:
            TradeLogRepository(engine=engine)
        except SQLAlchemyError as exc:
            drill.fail(f"the box could not open it: {exc}")
            return
        inspector = inspect(engine)
        created = sorted(set(inspector.get_table_names()) - set(before))
        added = sorted(
            f"{t}.{c['name']}"
            for t in before
            for c in inspector.get_columns(t)
            if c["name"] not in before[t]
        )
        upgrade = f"{len(created)} table(s) created, {len(added)} column(s) added"
        if created or added:
            drill.warn(f"opening it upgrades the file: {upgrade} ({', '.join(created + added)})")
        unread: list[str] = []
        total = 0
        with Session(engine) as s:
            for mapper in sorted(Base.registry.mappers, key=lambda m: m.class_.__name__):
                try:
                    rows = s.scalars(select(mapper.class_).execution_options(yield_per=500))
                    total += sum(1 for _ in rows)
                except (SQLAlchemyError, ValueError, TypeError) as exc:
                    unread.append(f"{mapper.class_.__tablename__}: {exc}")
                s.expunge_all()
        if unread:
            drill.fail(f"rows the app cannot read: {'; '.join(unread)}")
        else:
            drill.ok(f"opens as the box opens it ({upgrade}); {total} rows read through the ORM")
    finally:
        engine.dispose()


def check_snapshots(gz: Path, drill: Drill) -> None:
    print(SNAPSHOT_ARTIFACT)
    try:
        text = gzip.decompress(gz.read_bytes()).decode("utf-8")
    except (OSError, EOFError, gzip.BadGzipFile, UnicodeDecodeError) as exc:
        drill.fail(f"{SNAPSHOT_ARTIFACT} does not decode: {exc}")
        return
    records: list[dict] = []
    bad: list[int] = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            bad.append(n)
    if bad:
        drill.fail(f"{len(bad)} line(s) do not parse (first: line {bad[0]})")
    if not records:
        drill.fail("no snapshot in the artifact")
        return
    last = records[-1]
    drill.ok(f"{len(records)} snapshots parse; last equity ${float(last.get('equity', 0)):,.2f}")
    drill.fresh("eval snapshots", _parse_stamp(last.get("ts")))


def _unsafe_member(m: tarfile.TarInfo) -> bool:
    parts = Path(m.name).parts
    return m.name.startswith("/") or ".." in parts or not (m.isfile() or m.isdir())


def check_memory(gz: Path, work: Path, drill: Drill) -> None:
    print(MEMORY_ARTIFACT)
    target = work / "memory"
    try:
        with tarfile.open(gz, "r:gz") as tar:
            members = tar.getmembers()
            unsafe = [m.name for m in members if _unsafe_member(m)]
            if unsafe:
                drill.fail(f"refusing to extract unsafe member(s): {', '.join(unsafe[:3])}")
                return
            tar.extractall(target, filter="data")
    except (OSError, EOFError, tarfile.TarError) as exc:
        drill.fail(f"{MEMORY_ARTIFACT} does not extract: {exc}")
        return
    logs = sorted(target.rglob(MEMORY_LOG))
    if not logs:
        drill.fail(f"{len(members)} member(s) extracted, but no {MEMORY_LOG}")
        return
    from tradingagents.agents.utils.memory import TradingMemoryLog

    log = logs[0]
    try:
        entries = TradingMemoryLog({"memory_log_path": str(log)}).load_entries()
    except (OSError, UnicodeDecodeError) as exc:
        drill.fail(f"{MEMORY_LOG} does not read: {exc}")
        return
    size = log.stat().st_size
    if size and not entries:
        drill.fail(f"{MEMORY_LOG} is {_mib(size)} but holds no entry the agent can parse")
        return
    pending = sum(1 for e in entries if e.get("pending"))
    newest = max((e["date"] for e in entries), default="-")
    drill.ok(
        f"{len(members)} safe member(s); {MEMORY_LOG} {_mib(size)}, {len(entries)} entries "
        f"({pending} pending), newest {newest}"
    )


def compare_live(restored: dict[str, int], live_db: Path, drill: Drill) -> None:
    """Restored vs live row counts; the gap is what a restore would lose now."""
    print(f"vs live {live_db} (read-only)")
    engine = _read_only(live_db)
    try:
        with engine.connect() as conn:
            live = _table_counts(conn)
    except SQLAlchemyError as exc:
        drill.fail(f"cannot read the live DB: {exc}")
        return
    finally:
        engine.dispose()
    print(f"  {'table':<28}{'restored':>10}{'live':>10}{'gap':>8}")
    for table in sorted(set(restored) | set(live)):
        r, lv = restored.get(table), live.get(table)
        gap = "-" if r is None or lv is None else f"{lv - r:+d}"
        print(f"  {table:<28}{r if r is not None else '-':>10}{lv if lv is not None else '-':>10}"
              f"{gap:>8}")
    ahead = [t for t in _NEWEST_COLUMN if restored.get(t, 0) > live.get(t, 0)]
    if ahead:
        drill.warn(f"backup holds more rows than live for {', '.join(ahead)} — is this the box "
                   "the backup came from?")


def run(source: Path, *, live_db: Path | None, max_age_hours: float, now: datetime) -> int:
    drill = Drill(now=now, max_age_hours=max_age_hours)
    started = time.monotonic()
    print(f"restore drill — {source}")
    missing = [a for a in (DB_ARTIFACT, SNAPSHOT_ARTIFACT, MEMORY_ARTIFACT)
               if not (source / a).is_file()]
    for name in missing:
        drill.fail(f"{name}: not in the backup")
    with tempfile.TemporaryDirectory(prefix="restore-drill-") as tmp:
        work = Path(tmp)
        counts = None
        if DB_ARTIFACT not in missing:
            counts = check_db(source / DB_ARTIFACT, work, drill)
        if SNAPSHOT_ARTIFACT not in missing:
            check_snapshots(source / SNAPSHOT_ARTIFACT, drill)
        if MEMORY_ARTIFACT not in missing:
            check_memory(source / MEMORY_ARTIFACT, work, drill)
        if live_db is not None and counts is not None:
            compare_live(counts, live_db, drill)
    elapsed = time.monotonic() - started
    if drill.failures:
        print(f"RESTORE DRILL FAILED — {len(drill.failures)} check(s) in {elapsed:.1f}s")
        return 1
    print(f"RESTORE DRILL OK — 3 artifacts restored and read back in {elapsed:.1f}s")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", type=Path, help="directory holding the artifacts "
                    "(the backup clone's current/, or a dated S3 prefix downloaded)")
    ap.add_argument("--live-db", type=Path, help="live local.db to compare row counts with")
    ap.add_argument("--max-age-hours", type=float, default=DEFAULT_MAX_AGE_HOURS)
    args = ap.parse_args(argv)
    if not args.source.is_dir():
        print(f"restore drill: {args.source} is not a directory", file=sys.stderr)
        return 1
    return run(args.source, live_db=args.live_db, max_age_hours=args.max_age_hours,
               now=datetime.now(UTC))


if __name__ == "__main__":
    raise SystemExit(main())
