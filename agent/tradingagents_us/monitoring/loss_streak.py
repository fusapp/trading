"""What the consecutive-loss breaker is counting, said in closing events.

The breaker (`risk.circuit_breaker`, check 5) refuses every entry once
`risk.market_inputs.recent_loss_streak` reaches its threshold. That count walks
the realized ledger row by row, and a ledger row is a FIFO lot pairing, not a
trade: one market BUY that fills in four pieces and a stop that fills in two
become five rows. On 2026-11-04 the book's only position, XOM 68, stopped out
for -$248 (0.23% of equity); the ledger wrote it as five losing rows and the
breaker has refused every BUY since.

It also cannot clear itself. The count resets only at a winning close, a close
needs an open position, and with entries refused and the book flat there is
nothing to close.

This module groups ledger rows into closing events (one symbol, one New York
session, net P&L). `recent_loss_streak` counts those, not rows, and preflight
uses the same grouping to name what a halt is made of. Pure: no DB, no clock,
no network.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

from ..execution.actionability import session_date


class ClosedLot(Protocol):
    """The ledger fields this module reads; `ClosedTradeRow` satisfies it."""

    symbol: str
    realized_pnl: float | None
    closed_at_utc: datetime


@dataclass(frozen=True)
class ClosingEvent:
    """Every ledger row one symbol closed in one New York session."""

    symbol: str
    session: date
    lots: int
    net_pnl: float


def closing_events(trades: Sequence[ClosedLot]) -> list[ClosingEvent]:
    """Group ledger rows into closing events, keeping newest-first order.

    Rows are grouped by key, not by adjacency: two names stopped out in the same
    minute interleave their fills in `closed_at` order, and adjacency would cut
    each of them into pieces.
    """
    groups: dict[tuple[str, date], list[float]] = {}
    for t in trades:
        key = (t.symbol.upper(), session_date(t.closed_at_utc))
        groups.setdefault(key, []).append(t.realized_pnl or 0.0)
    return [
        ClosingEvent(symbol=sym, session=sess, lots=len(pnls), net_pnl=sum(pnls))
        for (sym, sess), pnls in groups.items()
    ]


def event_loss_streak(events: Sequence[ClosingEvent]) -> int:
    """Losing closing events in a row, newest first.

    Same rule as the breaker's row count: a break-even close ends the streak.
    """
    streak = 0
    for e in events:
        if e.net_pnl >= 0:
            break
        streak += 1
    return streak


def _usd(x: float) -> str:
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def describe_loss_streak(
    streak: int,
    trades: Sequence[ClosedLot],
    *,
    threshold: int,
    open_positions: int | None,
) -> str | None:
    """One line on a tripped breaker, or None while it lets entries through.

    `streak` is the breaker's own count (`recent_loss_streak`, in closing
    events), passed in rather than recomputed so this report cannot disagree
    with the gate. The events it counted are the newest `streak` closing events
    of `trades`, which must be the ledger read newest first.
    """
    if streak < threshold:
        return None
    counted = closing_events(trades)[:streak]
    named = ", ".join(
        f"{e.symbol} {e.session.isoformat()} {_usd(e.net_pnl)} over {e.lots} lot(s)"
        for e in counted
    )
    msg = (
        f"consecutive-loss breaker refuses every BUY: {streak} losing closing events "
        f"in a row (threshold {threshold}): {named}"
    )
    if open_positions == 0:
        msg += "; the book is flat, so no close can reset it and it stays shut until someone acts"
    elif open_positions is None:
        msg += "; open positions unknown"
    else:
        msg += f"; it resets on the next winning close ({open_positions} position(s) open)"
    return msg
