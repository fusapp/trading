"""Re-entries after a flat exit, and what the trip through cash cost.

The case that prompted this is pinned first: GOOGL time-exited on 2026-10-06,
bought back eight days later $9.59 higher, stopped out two sessions after that.
The rest pin the definitions that decide what counts — a trim is not an exit,
an add is not a re-entry, a still-open window is not "never came back" — since
each of those, got wrong, moves the re-entry rate in a direction that would be
read as a finding about the strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

import pytest

from scripts.reentry_report import build_payload, format_report
from tradingagents_us.execution.exit_quality import ExitOrder
from tradingagents_us.execution.reconcile import Fill, reconcile_fills
from tradingagents_us.execution.reentry import (
    ExitEpisode,
    find_episodes,
    session_date,
    summarize,
)


@dataclass(frozen=True)
class _F:
    id: str
    symbol: str
    side: str
    qty: float
    price: float
    transaction_time: datetime
    order_id: str | None = None


def _at(day: str, hhmm: str = "13:31") -> datetime:
    return datetime.fromisoformat(f"{day}T{hhmm}:00+00:00")


def _order(order_id: str, order_type: str, client_id: str, *, leg: bool = False) -> ExitOrder:
    return ExitOrder(
        order_id=order_id, client_order_id=client_id, order_type=order_type, is_leg=leg
    )


TIME_EXIT = _order("ox-time", "market", "tr-exit-time-GOOGL-20261005")
STOP = _order("ox-stop", "stop", "stop-leg", leg=True)
TP = _order("ox-tp", "limit", "tp-leg", leg=True)


def _episodes(fills: list[_F], exits: dict[str, ExitOrder]) -> list[ExitEpisode]:
    reconciled = reconcile_fills(
        [
            Fill(
                id=f.id,
                symbol=f.symbol,
                side=f.side,
                qty=f.qty,
                price=f.price,
                transaction_time=f.transaction_time,
            )
            for f in fills
        ]
    )
    return find_episodes(fills, exits, reconciled)


def _googl() -> tuple[list[_F], dict[str, ExitOrder]]:
    fills = [
        _F("b1", "GOOGL", "buy", 32, 300.00, _at("2026-08-20")),
        _F("s1", "GOOGL", "sell", 32, 345.46, _at("2026-10-06", "13:33"), "ox-time"),
        _F("b2", "GOOGL", "buy", 32, 355.05, _at("2026-10-14")),
        _F("s2", "GOOGL", "sell", 32, 339.46, _at("2026-10-16", "15:02"), "ox-stop"),
    ]
    return fills, {"s1": TIME_EXIT, "s2": STOP}


def test_googl_time_exit_was_bought_back_higher_then_stopped():
    fills, exits = _googl()
    first, second = _episodes(fills, exits)

    assert first.exit.exit_class == "time_exit"
    assert first.exit.session == date(2026, 10, 6)
    assert first.reentry is not None and first.reentry.session == date(2026, 10, 14)
    assert first.gap_days == pytest.approx(8.0, abs=0.01)
    assert first.buyback_per_share == pytest.approx(9.59)
    assert first.buyback_cost == pytest.approx(306.88)
    # The stop's loss belongs to the re-entry's own lots, followed through FIFO.
    assert first.reentry_realized_pnl == pytest.approx((339.46 - 355.05) * 32)
    assert first.reentry_closed_qty == 32 and first.reentry_open_qty == 0

    assert second.exit.exit_class == "stop"
    assert second.reentry is None and second.buyback_cost is None


def test_a_trim_is_not_an_exit_and_an_add_is_not_a_reentry():
    fills = [
        _F("b1", "AAPL", "buy", 10, 100.0, _at("2026-09-01")),
        _F("s1", "AAPL", "sell", 4, 110.0, _at("2026-09-03"), "ox-tp"),
        _F("b2", "AAPL", "buy", 4, 108.0, _at("2026-09-05")),
    ]
    assert _episodes(fills, {"s1": TP}) == []


def test_one_session_flat_through_two_orders_is_one_exit_at_their_vwap():
    # A one-share bracket TP and a standalone stop for the rest, same session:
    # the stop is what took the book flat, and the exit price is both of them.
    fills = [
        _F("b1", "GOOGL", "buy", 10, 100.0, _at("2026-09-01")),
        _F("s1", "GOOGL", "sell", 1, 120.0, _at("2026-09-10", "14:00"), "ox-tp"),
        _F("s2", "GOOGL", "sell", 9, 90.0, _at("2026-09-10", "18:00"), "ox-stop"),
    ]
    (episode,) = _episodes(fills, {"s1": TP, "s2": STOP})
    assert episode.exit.exit_class == "stop"
    assert episode.exit.qty == 10
    assert episode.exit.vwap == pytest.approx((120.0 + 9 * 90.0) / 10)


def test_sessions_are_new_york_dates_not_utc_dates():
    # 23:30Z and 01:00Z next day are both the same New York evening.
    assert session_date(_at("2026-10-06", "23:30")) == date(2026, 10, 6)
    assert session_date(_at("2026-10-07", "01:00")) == date(2026, 10, 6)
    fills = [
        _F("b1", "XOM", "buy", 10, 100.0, _at("2026-09-01")),
        _F("s1", "XOM", "sell", 10, 110.0, _at("2026-09-08", "19:00"), "ox-tp"),
        _F("b2", "XOM", "buy", 6, 112.0, _at("2026-09-10", "23:30")),
        _F("b3", "XOM", "buy", 4, 113.0, _at("2026-09-11", "01:00")),
    ]
    (episode,) = _episodes(fills, {"s1": TP})
    assert episode.reentry is not None
    assert episode.reentry.session == date(2026, 9, 10)
    assert episode.reentry.qty == 10
    assert episode.reentry.fill_ids == ("b2", "b3")
    assert episode.reentry_open_qty == 10


def test_buyback_cost_uses_the_smaller_quantity_and_can_be_negative():
    fills = [
        _F("b1", "MSFT", "buy", 100, 380.0, _at("2026-06-01")),
        _F("s1", "MSFT", "sell", 100, 372.38, _at("2026-06-24")),
        _F("b2", "MSFT", "buy", 26, 360.50, _at("2026-06-25")),
    ]
    (episode,) = _episodes(fills, {})
    # Unmatched order: missing evidence, not a guessed class.
    assert episode.exit.exit_class == "unknown"
    assert episode.buyback_cost == pytest.approx((360.50 - 372.38) * 26)


def test_a_cover_after_selling_from_flat_is_not_a_reentry():
    fills = [
        _F("b1", "V", "buy", 5, 100.0, _at("2026-09-01")),
        _F("s1", "V", "sell", 5, 101.0, _at("2026-09-02"), "ox-stop"),
        _F("s2", "V", "sell", 3, 102.0, _at("2026-09-03")),
        _F("b2", "V", "buy", 3, 99.0, _at("2026-09-04")),
    ]
    (episode,) = _episodes(fills, {"s1": STOP})
    assert episode.reentry is None


def test_an_open_window_is_pending_not_never_came_back():
    fills = [
        _F("b1", "META", "buy", 18, 560.0, _at("2026-08-01")),
        _F("s1", "META", "sell", 18, 711.60, _at("2026-10-16"), "ox-stop"),
        _F("b2", "JPM", "buy", 31, 320.0, _at("2026-08-01")),
        _F("s2", "JPM", "sell", 31, 338.47, _at("2026-09-22"), "ox-stop"),
    ]
    episodes = _episodes(fills, {"s1": STOP, "s2": STOP})
    as_of = _at("2026-10-18", "06:00")
    jpm, meta = episodes
    assert meta.is_pending(14, as_of) and not jpm.is_pending(14, as_of)

    (stops,), _ = summarize(episodes, window_days=14, as_of=as_of)
    assert (stops.flat_exits, stops.pending, stops.decided, stops.reentered) == (2, 1, 1, 0)
    assert stops.reentry_rate == 0.0


def test_a_reentry_past_the_window_is_listed_but_not_counted():
    fills = [
        _F("b1", "AAPL", "buy", 29, 280.0, _at("2026-05-01")),
        _F("s1", "AAPL", "sell", 29, 310.07, _at("2026-06-01"), "ox-tp"),
        _F("b2", "AAPL", "buy", 33, 298.58, _at("2026-06-18", "16:00")),
    ]
    (episode,) = _episodes(fills, {"s1": TP})
    assert episode.reentry is not None and episode.gap_days > 14
    assert not episode.reentered_within(14)
    assert not episode.is_pending(14, _at("2026-10-18"))

    (bucket,), strategy = summarize([episode], window_days=14, as_of=_at("2026-10-18"))
    assert (bucket.reentered, bucket.buyback_cost, bucket.median_gap_days) == (0, 0.0, None)
    # A wider window takes it in.
    (wide,), _ = summarize([episode], window_days=30, as_of=_at("2026-10-18"))
    assert wide.reentered == 1 and wide.buyback_cost == pytest.approx((298.58 - 310.07) * 29)


def test_strategy_rollup_leaves_out_flattens():
    fills, exits = _googl()
    fills = [
        *fills,
        _F("b9", "MSFT", "buy", 100, 380.0, _at("2026-06-01")),
        _F("s9", "MSFT", "sell", 100, 372.38, _at("2026-06-24")),
        _F("b10", "MSFT", "buy", 26, 360.50, _at("2026-06-25")),
    ]
    flatten = _order("ox-flat", "market", "0b1c-uuid-from-the-broker")
    episodes = _episodes(fills, {**exits, "s9": flatten})
    buckets, strategy = summarize(episodes, window_days=14, as_of=_at("2026-10-18"))

    assert [b.exit_class for b in buckets] == ["stop", "time_exit", "flatten"]
    assert strategy.exit_class == "strategy"
    assert strategy.flat_exits == 2
    assert strategy.reentered == 1 and strategy.bought_back_higher == 1
    assert strategy.buyback_cost == pytest.approx(306.88)
    assert strategy.reentry_realized_pnl == pytest.approx((339.46 - 355.05) * 32)


def test_report_renders_episodes_and_the_rollup():
    fills, exits = _googl()
    as_of = _at("2026-10-18", "06:00")
    payload = build_payload(
        _episodes(fills, exits), window_days=14, as_of=as_of, fills_read=len(fills)
    )
    assert payload["strategy_only"]["buyback_cost"] == pytest.approx(306.88)
    assert [e["exit_class"] for e in payload["episodes"]] == ["time_exit", "stop"]
    assert payload["episodes"][1]["pending"] is True

    text = format_report(payload, episodes=True)
    assert "strategy exits only" in text
    assert "time exit (position manager)" in text
    assert "306.88" in text and "pending" in text
    assert "GOOGL" not in format_report(payload, episodes=False)


def test_empty_history_reports_nothing_without_dividing_by_zero():
    buckets, strategy = summarize([], window_days=14, as_of=datetime.now(UTC))
    assert buckets == []
    assert strategy.flat_exits == 0 and strategy.reentry_rate == 0.0
    text = format_report(
        build_payload([], window_days=14, as_of=datetime.now(UTC), fills_read=0),
        episodes=True,
    )
    assert "strategy exits only" in text
