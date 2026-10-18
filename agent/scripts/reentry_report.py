#!/usr/bin/env python3
"""When the book went flat in a name, did it buy back in — and at what price?

    python scripts/reentry_report.py                  # table
    python scripts/reentry_report.py --episodes       # add one row per flat exit
    python scripts/reentry_report.py --window-days 7  # tighter "re-entered" window
    python scripts/reentry_report.py --json           # machine-readable

Read-only against the broker and it touches no database: it submits nothing,
cancels nothing, persists nothing. Run it where the Alpaca keys work (the box);
the laptop key is the dead one.

`exit_quality.py` answers what closed each position. This answers what came
next: the time exit sold GOOGL on 10-06 and the council bought it back 3% higher
on 10-14, and the stop that followed is booked as a separate loss. The cost of
that trip through cash appears in neither report. See
`tradingagents_us/execution/reentry.py` for the definitions.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from scripts.reconcile import ORDER_PAGE, to_fills  # noqa: E402
from tradingagents_us.dataflows.alpaca_broker import AlpacaClient  # noqa: E402
from tradingagents_us.execution.exit_quality import (  # noqa: E402
    CLASS_LABELS,
    exits_by_fill,
    index_orders,
)
from tradingagents_us.execution.reconcile import reconcile_fills  # noqa: E402
from tradingagents_us.execution.reentry import (  # noqa: E402
    DEFAULT_WINDOW_DAYS,
    ExitEpisode,
    ReentryBucket,
    find_episodes,
    summarize,
)


def _round(value: float | None, places: int = 2) -> float | None:
    return None if value is None else round(value, places)


def bucket_payload(bucket: ReentryBucket) -> dict:
    return {
        "exit_class": bucket.exit_class,
        "label": CLASS_LABELS.get(bucket.exit_class, bucket.exit_class),
        "flat_exits": bucket.flat_exits,
        "pending": bucket.pending,
        "decided": bucket.decided,
        "reentered": bucket.reentered,
        "reentry_rate": round(bucket.reentry_rate, 4),
        "bought_back_higher": bucket.bought_back_higher,
        "buyback_cost": round(bucket.buyback_cost, 2),
        "median_gap_days": _round(bucket.median_gap_days, 1),
        "reentry_realized_pnl": round(bucket.reentry_realized_pnl, 2),
    }


def episode_payload(e: ExitEpisode, window_days: int, as_of: datetime) -> dict:
    r = e.reentry
    return {
        "symbol": e.exit.symbol,
        "exit_class": e.exit.exit_class,
        "exit_session": e.exit.session.isoformat(),
        "exit_qty": e.exit.qty,
        "exit_vwap": round(e.exit.vwap, 2),
        "reentry_session": r.session.isoformat() if r else None,
        "reentry_qty": r.qty if r else None,
        "reentry_vwap": _round(r.vwap if r else None),
        "gap_days": _round(e.gap_days, 1),
        "within_window": e.reentered_within(window_days),
        "pending": e.is_pending(window_days, as_of),
        "buyback_cost": _round(e.buyback_cost),
        "reentry_realized_pnl": round(e.reentry_realized_pnl, 2),
        "reentry_open_qty": e.reentry_open_qty,
    }


def build_payload(
    episodes: list[ExitEpisode], *, window_days: int, as_of: datetime, fills_read: int
) -> dict:
    buckets, strategy = summarize(episodes, window_days=window_days, as_of=as_of)
    return {
        "as_of_utc": as_of.isoformat(),
        "window_days": window_days,
        "fills_read": fills_read,
        "by_exit": [bucket_payload(b) for b in buckets],
        "strategy_only": bucket_payload(strategy),
        "episodes": [episode_payload(e, window_days, as_of) for e in episodes],
    }


def _gap(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}d"


def format_report(payload: dict, *, episodes: bool) -> str:
    w = payload["window_days"]
    lines = [
        f"re-entry after a flat exit (window {w}d, as of {payload['as_of_utc'][:16]}Z, "
        f"{payload['fills_read']} fills)",
        "",
        f"  {'exit path':<30}{'flat':>5}{'pend':>5}{'back':>5}{'rate':>6}{'higher':>7}"
        f"{'buyback $':>11}{'gap':>7}{'re-entry P&L':>14}",
        f"  {'-' * 90}",
    ]

    def row(b: dict) -> str:
        return (
            f"  {b['label']:<30}{b['flat_exits']:>5}{b['pending']:>5}{b['reentered']:>5}"
            f"{b['reentry_rate'] * 100:>5.0f}%{b['bought_back_higher']:>7}"
            f"{b['buyback_cost']:>11,.2f}{_gap(b['median_gap_days']):>7}"
            f"{b['reentry_realized_pnl']:>14,.2f}"
        )

    lines += [row(b) for b in payload["by_exit"]]
    lines += [
        f"  {'-' * 90}",
        row(payload["strategy_only"]),
        "",
        f"  back = bought back within {w}d; rate is over decided exits (pending = window",
        "  still open, no re-entry yet). buyback $ = (re-entry VWAP − exit VWAP) × the",
        "  smaller qty: positive is what the trip through cash cost against holding.",
        "  re-entry P&L is realized on the re-entry's own lots so far; open shares excluded.",
    ]
    if episodes:
        lines += [
            "",
            f"  {'symbol':<7}{'exit path':<14}{'exit':>11}{'qty':>5}{'@':>9}"
            f"{'back':>12}{'qty':>5}{'@':>9}{'gap':>7}{'buyback $':>11}{'re-entry':>10}",
        ]
        for e in payload["episodes"]:
            if e["reentry_session"]:
                back = (
                    f"{e['reentry_session']:>12}{e['reentry_qty']:>5g}"
                    f"{e['reentry_vwap']:>9,.2f}{_gap(e['gap_days']):>7}"
                    f"{e['buyback_cost']:>11,.2f}"
                )
                tail = f"{e['reentry_realized_pnl']:>10,.2f}"
                if e["reentry_open_qty"]:
                    tail += f"  +{e['reentry_open_qty']:g} open"
                if not e["within_window"]:
                    tail += f"  (>{w}d)"
            else:
                back = f"{'pending' if e['pending'] else 'not back':>12}"
                tail = ""
            lines.append(
                f"  {e['symbol']:<7}{e['exit_class']:<14}{e['exit_session']:>11}"
                f"{e['exit_qty']:>5g}{e['exit_vwap']:>9,.2f}{back}{tail}"
            )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-entries after flat exits, by exit path")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--episodes", action="store_true", help="include one row per flat exit")
    ap.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help=f"calendar days a buy counts as re-entering (default {DEFAULT_WINDOW_DAYS})",
    )
    args = ap.parse_args()
    if args.window_days < 1:
        ap.error("--window-days must be at least 1")

    with AlpacaClient() as ac:
        fills = ac.list_fill_activities()
        orders = ac.list_orders(status="all", limit=ORDER_PAGE, nested=True)

    episodes = find_episodes(
        fills, exits_by_fill(fills, index_orders(orders)), reconcile_fills(to_fills(fills))
    )
    payload = build_payload(
        episodes, window_days=args.window_days, as_of=datetime.now(UTC), fills_read=len(fills)
    )
    if not args.episodes:
        payload.pop("episodes")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(format_report(payload, episodes=args.episodes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
