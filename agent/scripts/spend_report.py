#!/usr/bin/env python3
"""What the councils cost over a window, and how much of it reached the broker.

    python -m scripts.spend_report              # last 30 days
    python -m scripts.spend_report --days 90
    python -m scripts.spend_report --tickers    # add the per-ticker table
    python -m scripts.spend_report --json       # machine-readable

DB-only and read-only: a SQLite file is opened with mode=ro, the broker is
never called, nothing is written. Run it where the trade log lives (the box:
TRADE_LOG_DB_URL from secrets.env). Definitions live in
`tradingagents_us/monitoring/council_spend.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from tradingagents_us.monitoring.council_spend import (  # noqa: E402
    Bucket,
    CouncilCost,
    SpendReport,
    build_report,
    report_payload,
)

# An order row is written seconds after its decision. Reading orders from a
# little before the window keeps a decision at the window's edge from losing
# its order row and reading as an on-demand council.
_ORDER_MARGIN = timedelta(days=1)


def read_only_url(url: str) -> str:
    """The same SQLite file, opened mode=ro; any other URL unchanged."""
    from sqlalchemy.engine import make_url

    parsed = make_url(url)
    if parsed.get_backend_name() != "sqlite" or parsed.query:
        return url
    database = parsed.database or ""
    if database in ("", ":memory:"):
        return url
    return f"sqlite:///file:{database}?mode=ro&uri=true"


def collect(repo, since: datetime) -> list[CouncilCost]:
    decisions = repo.list_decisions_since(since=since)
    orders = repo.list_orders_since(since=since - _ORDER_MARGIN, limit=1_000_000)
    has_row: set[str] = set()
    acked: set[str] = set()
    for o in orders:
        has_row.add(o.decision_id)
        if o.broker_order_id:
            acked.add(o.decision_id)
    return [
        CouncilCost(
            ticker=d.ticker,
            rating=d.rating,
            timestamp_utc=d.timestamp_utc,
            cost_usd=d.cost_usd,
            scheduled=d.decision_id in has_row,
            reached_broker=d.decision_id in acked,
        )
        for d in decisions
    ]


def _usd(value: float | None, floor: bool = False) -> str:
    if value is None:
        return "—"
    return f"${value:,.2f}{'+' if floor else ''}"


def _line(label: str, b: Bucket) -> str:
    unmeasured = f", {b.unmeasured} unmeasured" if b.unmeasured else ""
    return (
        f"  {label:<22} {b.councils:>4} councils  {_usd(b.spend_usd, b.is_floor):>11}"
        f"  {b.broker_submits:>3} reached broker{unmeasured}"
    )


def render(r: SpendReport, days: int, *, tickers: bool) -> str:
    if r.total.councils == 0:
        return f"council spend, last {days} days: no decisions in the window"
    floor = " ('+' = floor: some councils were never metered)" if r.total.is_floor else ""
    per_submit = (
        _usd(r.spend_per_submit)
        if r.spend_per_submit is not None
        else "— (none reached the broker)"
    )
    since = r.since_last_submit
    last = (
        f"last: {r.last_submit_session.isoformat()}"
        if r.last_submit_session
        else "none in window"
    )
    out = [
        f"council spend, last {days} days{floor}",
        _line("all councils", r.total),
        _line("nightly run", r.scheduled),
        _line("on demand (API)", r.on_demand),
        "",
        f"  run days                {len(r.run_days)}",
        f"  mean per run day        {_usd(r.mean_per_run_day)}",
        f"  monthly run-rate        {_usd(r.monthly_run_rate)}  (mean × 21 sessions)",
        f"  per broker submission   {per_submit}",
        f"  since last submission   {_usd(since.spend_usd, since.is_floor)}"
        f" over {r.run_days_since_last_submit} run days ({last})",
        "",
        "  by rating",
    ]
    out += [_line(rating, b) for rating, b in r.by_rating.items()]
    if tickers:
        out += ["", "  by ticker"]
        out += [_line(t, b) for t, b in r.by_ticker.items()]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--days", type=int, default=30, help="lookback in calendar days (default 30)")
    ap.add_argument("--tickers", action="store_true", help="add the per-ticker table")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    ap.add_argument(
        "--db-url",
        default=os.environ.get("TRADE_LOG_DB_URL", "sqlite:///./local.db"),
        help="trade log URL (default: TRADE_LOG_DB_URL); a SQLite file is opened read-only",
    )
    args = ap.parse_args(argv)
    if args.days < 1:
        ap.error("--days must be >= 1")

    from tradingagents_us.storage import TradeLogRepository, make_engine

    repo = TradeLogRepository(engine=make_engine(read_only_url(args.db_url)))
    since = datetime.now(UTC) - timedelta(days=args.days)
    report = build_report(collect(repo, since))

    if args.json:
        print(json.dumps({"window_days": args.days, **report_payload(report)}, indent=2))
    else:
        print(render(report, args.days, tickers=args.tickers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
