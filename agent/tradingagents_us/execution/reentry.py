"""What the book paid to get back into a name it had just left.

On 2026-10-06 the time exit sold GOOGL 32 @ $345.46. On 10-14 the council's
BUY filled 32 @ $355.05, and two sessions later the stop took it out at $339.46
for −$499. The ledger books that −$499 to the second position, and
`exit_quality` files the first exit under "time exit" with its own P&L. Neither
says the part in between: the book left a name and paid $9.59 a share more to
own it again eight days later — $307 spent before the stop fired.

That is a separate question from what closed a position, so it lives here
rather than in `exit_quality`: when the book went *flat* in a name, did it buy
back in, how soon, and at what price relative to where it got out. Split by the
exit path that took it flat, because the answer means different things for
each — a stop that is bought back higher is a whipsaw, a time exit that is
bought back at all says the exit rule and the council disagree about the name.

Definitions, all deliberately narrow:

  - A **flat exit** is the sell activity that takes a long holding to exactly
    zero. Sells that leave shares on the book are trims, not exits, and a buy
    on top of a held position is an add, not a re-entry.
  - The exit's **class** is the class (`exit_quality.classify_exit`) of the
    fill that took the book flat. Its quantity and VWAP are every sell slice in
    the same New York session since the last buy — a session can close one
    holding through several orders (a one-share bracket leg and a standalone
    stop), and the price the book got out at is all of them.
  - The **re-entry** is every buy slice in the first New York session after the
    flat exit that has a buy. Session dates are New York dates because an
    after-close fill is UTC-stamped on the next calendar day.
  - **Buyback cost** is (re-entry VWAP − exit VWAP) × the smaller of the two
    quantities: what the trip through cash cost against holding the shares
    that went out and came back. Positive means the book paid up to get back
    in; negative means the exit saved money. It ignores everything that
    happened to the cash while flat, so it is the cost of the round trip in
    price, not in opportunity.
  - An exit with no re-entry whose window has not elapsed yet is **pending**:
    it is still able to be bought back, so counting it as "not re-entered"
    would understate the re-entry rate exactly where the newest evidence is.

Long book only — this account has never held a short, and a sell from flat is
not an exit from anything.

Pure. The caller reads fills and orders from the broker; nothing here touches
a network or a database.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from statistics import median
from typing import Protocol
from zoneinfo import ZoneInfo

from tradingagents_us.execution.exit_quality import (
    CLASS_ORDER,
    STRATEGY_CLASSES,
    ExitClass,
    ExitOrder,
    classify_exit,
)
from tradingagents_us.execution.reconcile import QTY_EPSILON, ReconcileResult

_EXCHANGE_TZ = ZoneInfo("America/New_York")

#: Calendar days after a flat exit inside which a buy counts as re-entering.
#: Two weeks covers the GOOGL (8 days) and XOM (5 days) re-entries that
#: prompted this, and is short of the 42-bar time exit by a wide margin — a
#: buy two months later is a new idea about the name, not churn.
DEFAULT_WINDOW_DAYS = 14


class _Fill(Protocol):
    """The parts of a broker fill activity this module reads (`FillActivity`)."""

    id: str
    symbol: str
    side: str
    qty: float
    price: float
    transaction_time: datetime
    order_id: str | None


def session_date(ts: datetime) -> date:
    """The New York trading date a fill belongs to."""
    return ts.astimezone(_EXCHANGE_TZ).date()


@dataclass(frozen=True)
class FlatExit:
    symbol: str
    exit_class: ExitClass
    session: date
    flat_at_utc: datetime
    qty: float
    vwap: float


@dataclass(frozen=True)
class ReEntry:
    session: date
    first_fill_utc: datetime
    qty: float
    vwap: float
    fill_ids: tuple[str, ...]


@dataclass(frozen=True)
class ExitEpisode:
    """One flat exit and what the book did next in that name."""

    exit: FlatExit
    reentry: ReEntry | None
    #: Realized P&L of the re-entry's own lots, as far as they have closed.
    reentry_realized_pnl: float = 0.0
    reentry_closed_qty: float = 0.0
    #: Re-entry shares still on the book.
    reentry_open_qty: float = 0.0

    @property
    def gap_days(self) -> float | None:
        if self.reentry is None:
            return None
        delta = self.reentry.first_fill_utc - self.exit.flat_at_utc
        return delta.total_seconds() / 86400.0

    @property
    def buyback_per_share(self) -> float | None:
        if self.reentry is None:
            return None
        return self.reentry.vwap - self.exit.vwap

    @property
    def buyback_cost(self) -> float | None:
        per_share = self.buyback_per_share
        if per_share is None or self.reentry is None:
            return None
        return per_share * min(self.exit.qty, self.reentry.qty)

    def reentered_within(self, window_days: int) -> bool:
        gap = self.gap_days
        return gap is not None and gap <= window_days

    def is_pending(self, window_days: int, as_of: datetime) -> bool:
        """No re-entry inside the window, and the window is still open."""
        if self.reentered_within(window_days):
            return False
        return as_of - self.exit.flat_at_utc < timedelta(days=window_days)


@dataclass
class _Group:
    side: str
    session: date
    fills: list[_Fill]

    @property
    def qty(self) -> float:
        return sum(abs(f.qty) for f in self.fills)

    @property
    def vwap(self) -> float:
        q = self.qty
        return sum(abs(f.qty) * f.price for f in self.fills) / q if q > QTY_EPSILON else 0.0


def _session_groups(fills: list[_Fill]) -> list[_Group]:
    """Consecutive same-side fills in one New York session, in time order."""
    groups: list[_Group] = []
    for fill in fills:
        session = session_date(fill.transaction_time)
        if groups and groups[-1].side == fill.side and groups[-1].session == session:
            groups[-1].fills.append(fill)
        else:
            groups.append(_Group(side=fill.side, session=session, fills=[fill]))
    return groups


def _symbol_episodes(
    symbol: str, fills: list[_Fill], exits: Mapping[str, ExitOrder]
) -> list[tuple[FlatExit, ReEntry | None]]:
    out: list[tuple[FlatExit, ReEntry | None]] = []
    held = 0.0
    awaiting = False  # the last flat exit has not been followed by a buy yet

    for group in _session_groups(fills):
        before = held
        if group.side == "buy":
            held += group.qty
            if awaiting and before <= QTY_EPSILON:
                exit_, _ = out[-1]
                out[-1] = (
                    exit_,
                    ReEntry(
                        session=group.session,
                        first_fill_utc=group.fills[0].transaction_time,
                        qty=group.qty,
                        vwap=group.vwap,
                        fill_ids=tuple(f.id for f in group.fills),
                    ),
                )
            awaiting = False
            continue

        held -= group.qty
        if before <= QTY_EPSILON:
            # A sell from flat opens a short; covering it is not a re-entry.
            awaiting = False
        elif abs(held) <= QTY_EPSILON:
            # The last slice is the one that took the book flat: fills are in
            # time order and nothing but sells is in this group.
            closing = group.fills[-1]
            out.append(
                (
                    FlatExit(
                        symbol=symbol,
                        exit_class=classify_exit(exits.get(closing.id)),
                        session=group.session,
                        flat_at_utc=closing.transaction_time,
                        qty=group.qty,
                        vwap=group.vwap,
                    ),
                    None,
                )
            )
            awaiting = True
        held = 0.0 if abs(held) <= QTY_EPSILON else held
    return out


def find_episodes(
    fills: Iterable[_Fill],
    exits: Mapping[str, ExitOrder],
    reconciled: ReconcileResult,
) -> list[ExitEpisode]:
    """Every flat exit in the fill history, with its re-entry and how that went.

    `exits` is `exit_quality.exits_by_fill` (fill id → order), and `reconciled`
    is the FIFO match of the same fills — it is how a re-entry's own lots are
    followed to their realized P&L or found still open.
    """
    by_symbol: dict[str, list[_Fill]] = {}
    for fill in fills:
        by_symbol.setdefault(fill.symbol, []).append(fill)

    realized: dict[str, tuple[float, float]] = {}
    for trade in reconciled.closed:
        pnl, qty = realized.get(trade.open_activity_id, (0.0, 0.0))
        realized[trade.open_activity_id] = (pnl + trade.realized_pnl, qty + trade.quantity)
    still_open: dict[str, float] = {}
    for lot in reconciled.open_lots:
        still_open[lot.activity_id] = still_open.get(lot.activity_id, 0.0) + lot.quantity

    episodes: list[ExitEpisode] = []
    for symbol in sorted(by_symbol):
        ordered = sorted(by_symbol[symbol], key=lambda f: (f.transaction_time, f.id))
        for exit_, reentry in _symbol_episodes(symbol, ordered, exits):
            ids = reentry.fill_ids if reentry else ()
            episodes.append(
                ExitEpisode(
                    exit=exit_,
                    reentry=reentry,
                    reentry_realized_pnl=sum(realized.get(i, (0.0, 0.0))[0] for i in ids),
                    reentry_closed_qty=sum(realized.get(i, (0.0, 0.0))[1] for i in ids),
                    reentry_open_qty=sum(still_open.get(i, 0.0) for i in ids),
                )
            )
    episodes.sort(key=lambda e: (e.exit.flat_at_utc, e.exit.symbol))
    return episodes


@dataclass(frozen=True)
class ReentryBucket:
    """One exit path's re-entry record.

    `reentry_rate` is over *decided* exits only — pending ones are left out of
    the denominator, and their count is carried so the caller can say so.
    """

    exit_class: str
    flat_exits: int
    pending: int
    reentered: int
    bought_back_higher: int
    buyback_cost: float
    median_gap_days: float | None
    reentry_realized_pnl: float

    @property
    def decided(self) -> int:
        return self.flat_exits - self.pending

    @property
    def reentry_rate(self) -> float:
        return self.reentered / self.decided if self.decided else 0.0


def _bucket(
    exit_class: str, rows: list[ExitEpisode], window_days: int, as_of: datetime
) -> ReentryBucket:
    back = [e for e in rows if e.reentered_within(window_days)]
    costs = [e.buyback_cost or 0.0 for e in back]
    gaps = [e.gap_days for e in back if e.gap_days is not None]
    return ReentryBucket(
        exit_class=exit_class,
        flat_exits=len(rows),
        pending=sum(1 for e in rows if e.is_pending(window_days, as_of)),
        reentered=len(back),
        bought_back_higher=sum(1 for c in costs if c > 0),
        buyback_cost=sum(costs),
        median_gap_days=median(gaps) if gaps else None,
        reentry_realized_pnl=sum(e.reentry_realized_pnl for e in back),
    )


def summarize(
    episodes: list[ExitEpisode], *, window_days: int, as_of: datetime
) -> tuple[list[ReentryBucket], ReentryBucket]:
    """Per-exit-path buckets in `CLASS_ORDER`, plus the strategy-exits roll-up.

    The roll-up leaves out flattens and unattributable exits for the reason
    `exit_quality.strategy_bucket` does: the 2026-06-24 cleanup was bought back
    the next morning far below where it was dumped, and blending that in nets
    the strategy's own buyback cost down to almost nothing.
    """
    grouped: dict[str, list[ExitEpisode]] = {}
    for episode in episodes:
        grouped.setdefault(episode.exit.exit_class, []).append(episode)
    buckets = [
        _bucket(c, grouped[c], window_days, as_of) for c in CLASS_ORDER if c in grouped
    ]
    strategy = [e for e in episodes if e.exit.exit_class in STRATEGY_CLASSES]
    return buckets, _bucket("strategy", strategy, window_days, as_of)
