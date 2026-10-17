# Restore drill — reading the off-box backup back

`scripts/backup.py` (02:15 UTC, `ai-trader-backup.timer`) has pushed three
artifacts to the private `canberkaslan/trading-backups` repo every night since
2026-07-04: `local.db.gz`, `eval_snapshots.jsonl.gz`, `memory.tar.gz`, as
`current/<name>`, with git history as the retention. Until 2026-10-16 nothing
had read one back. `scripts/restore_drill.py` does that, read-only, and is the
check to run before trusting a backup with a box rebuild.

## What the drill proves

| Artifact | Check |
|---|---|
| `local.db.gz` | gunzips; `PRAGMA integrity_check` = ok; rollback-journal file per its header (one self-contained file, no `-wal` needed); no commentator feed row (ADR-009 scrub held); opened through `TradeLogRepository` as the box opens it (`create_all` + `ensure_additive_columns`, any upgrade named) and every mapped row read back through the ORM |
| `eval_snapshots.jsonl.gz` | every line parses; newest `ts` and equity |
| `memory.tar.gz` | only plain files/dirs, no absolute path or `..` (extracted with `filter="data"`); `trading_memory.md` parses with the vendored `TradingMemoryLog` |
| all | each artifact present (backup.py exits 0 when a target is missing); newest decision and newest snapshot younger than `--max-age-hours` (96 = weekend + a holiday) |
| `--live-db` | per-table row counts vs the live file opened `mode=ro`: the gap is what a restore would lose now |

Exit 0 = every check passed, 1 = one failed. Inputs are never written; the
restore happens in a temp dir that is removed on exit.

## Running it

From the laptop, against what actually reached GitHub:

```bash
git clone --depth 1 https://github.com/canberkaslan/trading-backups.git /tmp/tb
cd agent && PYTHONPATH=. .venv/bin/python -m scripts.restore_drill /tmp/tb/current
```

On the box, with the live gap (the box's clone is what it just pushed):

```bash
sudo -u deploy env PYTHONPATH=.:vendor/tradingagents /opt/ai-trader/agent/.venv/bin/python \
  -m scripts.restore_drill /opt/ai-trader/backups/current \
  --live-db /opt/ai-trader/agent/local.db </dev/null
```

(run from `/opt/ai-trader/agent`; `</dev/null` because a laptop ssh session
with stdin open hangs after the remote command ends).

### Weekly on the box

`ai-trader-restore-drill.timer` (Sundays 03:30 UTC, after the 02:15 backup)
runs the box command above with two more flags:

- `--origin`: nothing under `current/` is uncommitted, and origin's branch is
  the clone's HEAD (`git ls-remote`, no fetch, no index refresh). Only then is
  what was drilled the off-box copy and not just the box's own clone.
- `--notify`: failed checks go out through the ops channel as kind
  `restore_drill` (push + GitHub, like a failed backup).
  `OnFailure=ai-trader-alert.service` covers a crash before that.

Alert-only: it restores nothing. A failed drill says not to trust that set.
`install.sh` is the only thing that copies units, so on a box updated by
`git pull`:

```bash
sudo cp /opt/ai-trader/deploy/hetzner/ai-trader-restore-drill.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-trader-restore-drill.timer
sudo systemctl start ai-trader-restore-drill.service  # one run now
journalctl -u ai-trader-restore-drill.service -n 40 --no-pager
```

### An older day

A full clone (~350 MB), then
`git log --format='%h %s' -- current/local.db.gz` and
`git show <rev>:current/<name> > dir/<name>` for each artifact.

## Restoring for real (box loss / corrupt local.db)

1. Run the drill on the artifacts first. Do not restore a set that fails it.
2. Stop every writer: `sudo systemctl stop 'ai-trader*.timer' eval-report.timer
   ai-trader-api.service`, and wait out any `ai-trader*.service` still running.
3. Delete `local.db-wal` and `local.db-shm` beside the target. A leftover
   `-wal` belongs to the old file and SQLite reads it as "database disk image
   is malformed".
4. `gunzip -c local.db.gz > /opt/ai-trader/agent/local.db`, owned by `deploy`.
   The first open turns WAL back on (storage/engine.py).
5. `gunzip -c eval_snapshots.jsonl.gz > /opt/ai-trader/agent/data/eval_snapshots.jsonl`.
6. `tar -xzf memory.tar.gz -C /home/deploy/.tradingagents/` (arcname is `memory/`).
7. Start the API, check `/readyz`, then the timers. Run `scripts.reconcile`
   before the next daily run: the broker is the source of truth for anything
   that happened after the backup.

What a restore cannot bring back: the commentator feed (scrubbed off-box by
design, ADR-009; it refills from its sources) and every row written after the
backup's timestamp (RPO ≈ one day; the drill's live gap measures it).

## Drill log

| Date (UTC) | Where | Source | Result | Notes |
|---|---|---|---|---|
| 2026-10-16 06:15 | laptop (py 3.14) | GitHub `758aac9` (backup 2026-10-15, shallow clone) | OK, 1.7 s | 31.0 MiB DB, 808 decisions / 762 orders / 74 closed trades, newest 00:39Z (5.7 h); 72 snapshots; memory 802 entries (55 pending) |
| 2026-10-16 06:20 | prod-fusapp01 (py 3.12) | `/opt/ai-trader/backups/current` = `758aac9`, in sync with origin | OK, 3.6 s | live gap 0 on every audit table; `commentator_status` +1 = the ADR-009 scrub, by design |
