"""The losing-streak breaker counts ledger rows; a row is a FIFO lot, not a trade.

The fixture is the live ledger on 2026-11-06: XOM 68 (one market BUY that
filled in four pieces on 10-14) stopped out on 11-04 in two fills, and the
FIFO matcher wrote five losing rows. The breaker read that as five losses in a
row and has refused every entry since. The day before, AAPL closed at a profit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pytest

from tradingagents_us.monitoring.loss_streak import (
    ClosingEvent,
    closing_events,
    describe_loss_streak,
    event_loss_streak,
)
from tradingagents_us.risk.circuit_breaker import DEFAULT_MAX_CONSEC_LOSSES, CircuitBreaker


@dataclass
class Lot:
    symbol: str
    realized_pnl: float | None
    closed_at_utc: datetime


# Newest first, as `list_closed_trades` returns them. Naive = UTC, as SQLite
# hands them back.
LIVE_2026_11_06 = [
    Lot("XOM", -11.60, datetime(2026, 11, 4, 14, 35, 20)),
    Lot("XOM", -8.85, datetime(2026, 11, 4, 14, 35, 20)),
    Lot("XOM", -6.38, datetime(2026, 11, 4, 14, 35, 20)),
    Lot("XOM", -9.36, datetime(2026, 11, 4, 14, 35, 20)),
    Lot("XOM", -212.24, datetime(2026, 11, 4, 14, 34, 30)),
    Lot("AAPL", 41.57, datetime(2026, 11, 3, 14, 33, 56)),
    Lot("AAPL", 687.99, datetime(2026, 11, 3, 14, 33, 56)),
]


class TestClosingEvents:
    def test_one_stop_out_in_five_lots_is_one_event(self) -> None:
        events = closing_events(LIVE_2026_11_06)
        assert [(e.symbol, e.session, e.lots) for e in events] == [
            ("XOM", date(2026, 11, 4), 5),
            ("AAPL", date(2026, 11, 3), 2),
        ]
        assert events[0].net_pnl == pytest.approx(-248.43)

    def test_interleaved_fills_of_two_names_stay_two_events(self) -> None:
        # Grouping by adjacency would cut each name into pieces.
        rows = [
            Lot("XOM", -1.0, datetime(2026, 11, 4, 14, 35, 20)),
            Lot("MSFT", -2.0, datetime(2026, 11, 4, 14, 35, 0)),
            Lot("XOM", -3.0, datetime(2026, 11, 4, 14, 34, 30)),
            Lot("MSFT", -4.0, datetime(2026, 11, 4, 14, 34, 0)),
        ]
        events = closing_events(rows)
        assert [(e.symbol, e.lots, e.net_pnl) for e in events] == [
            ("XOM", 2, -4.0),
            ("MSFT", 2, -6.0),
        ]

    def test_session_is_new_york_not_utc(self) -> None:
        # 00:30 UTC on 11-05 is 19:30 ET on 11-04: the same session as the
        # afternoon fill, so one event.
        rows = [
            Lot("XOM", -1.0, datetime(2026, 11, 5, 0, 30)),
            Lot("XOM", -2.0, datetime(2026, 11, 4, 15, 0)),
        ]
        assert closing_events(rows) == [ClosingEvent("XOM", date(2026, 11, 4), 2, -3.0)]

    def test_same_name_on_two_sessions_is_two_events(self) -> None:
        rows = [
            Lot("XOM", -1.0, datetime(2026, 11, 5, 15, 0)),
            Lot("XOM", -2.0, datetime(2026, 11, 4, 15, 0)),
        ]
        assert len(closing_events(rows)) == 2

    def test_unmeasured_pnl_counts_as_zero(self) -> None:
        rows = [Lot("XOM", None, datetime(2026, 11, 4, 15, 0))]
        assert closing_events(rows)[0].net_pnl == 0.0


class TestEventLossStreak:
    def test_live_ledger_is_a_streak_of_one(self) -> None:
        assert event_loss_streak(closing_events(LIVE_2026_11_06)) == 1

    def test_a_break_even_close_ends_it_like_the_row_rule(self) -> None:
        events = [
            ClosingEvent("A", date(2026, 11, 4), 1, -1.0),
            ClosingEvent("B", date(2026, 11, 3), 1, 0.0),
            ClosingEvent("C", date(2026, 11, 2), 1, -1.0),
        ]
        assert event_loss_streak(events) == 1

    def test_empty(self) -> None:
        assert event_loss_streak([]) == 0


# Five losing closes on five sessions, then a winner: what the breaker is for.
FIVE_LOSING_CLOSES = LIVE_2026_11_06[:5] + [
    Lot("MSFT", -40.0, datetime(2026, 11, 3, 15, 0)),
    Lot("GOOGL", -30.0, datetime(2026, 10, 30, 15, 0)),
    Lot("GOOGL", -5.0, datetime(2026, 10, 30, 14, 59)),
    Lot("NVDA", -20.0, datetime(2026, 10, 29, 15, 0)),
    Lot("V", -10.0, datetime(2026, 10, 28, 15, 0)),
    Lot("AAPL", 41.57, datetime(2026, 10, 27, 14, 33, 56)),
]


class TestDescribe:
    def test_below_threshold_says_nothing(self) -> None:
        assert describe_loss_streak(4, FIVE_LOSING_CLOSES, threshold=5, open_positions=0) is None

    def test_a_halt_names_its_closing_events(self) -> None:
        msg = describe_loss_streak(5, FIVE_LOSING_CLOSES, threshold=5, open_positions=0)
        assert msg is not None
        assert msg.startswith(
            "consecutive-loss breaker refuses every BUY: 5 losing closing events in a row "
            "(threshold 5): XOM 2026-11-04 -$248.43 over 5 lot(s), MSFT 2026-11-03 -$40.00 "
            "over 1 lot(s), GOOGL 2026-10-30 -$35.00 over 2 lot(s), "
        )
        assert "book is flat" in msg

    def test_only_the_counted_events_are_named(self) -> None:
        msg = describe_loss_streak(5, FIVE_LOSING_CLOSES, threshold=5, open_positions=0)
        assert "AAPL" not in (msg or "")

    def test_open_book_can_still_reset(self) -> None:
        msg = describe_loss_streak(5, FIVE_LOSING_CLOSES, threshold=5, open_positions=2)
        assert msg is not None
        assert "resets on the next winning close (2 position(s) open)" in msg
        assert "flat" not in msg

    def test_unknown_positions_are_said_unknown(self) -> None:
        msg = describe_loss_streak(5, FIVE_LOSING_CLOSES, threshold=5, open_positions=None)
        assert msg is not None and msg.endswith("open positions unknown")


def test_threshold_constant_is_the_breakers_default() -> None:
    # trade.py builds the breaker with defaults; preflight reports against the
    # constant. They must be the same number.
    cb = CircuitBreaker(kill_switch=object())  # type: ignore[arg-type]
    assert cb.max_consec_losses == DEFAULT_MAX_CONSEC_LOSSES == 5
