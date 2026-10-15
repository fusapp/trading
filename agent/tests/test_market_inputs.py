"""Real inputs for risk checks that were fed constants.

`market_inputs` exists because three portfolio caps could not fire — they were
handed a hardcoded $1B ADV, an empty sector book, and a correlation count of
zero. These tests cover the two additions that revive the CIRCUIT BREAKER's
equivalent dead checks: a price-anomaly z-score computed against the price
itself (always exactly zero) and a consecutive-loss counter nothing ever
incremented.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import create_engine

from tradingagents_us.storage import TradeLogRepository
from tradingagents_us.storage.models import ClosedTradeRow
from tradingagents_us.storage.price_cache import write_bars


def _repo(tmp_path) -> TradeLogRepository:
    return TradeLogRepository(engine=create_engine(f"sqlite:///{tmp_path}/t.db", future=True))


def _seed(
    repo: TradeLogRepository, ticker: str, closes: list[float], *, end: date | None = None
) -> None:
    """Consecutive daily bars ending `end` (today by default), one per close."""
    start = (end or date.today()) - timedelta(days=len(closes) - 1)
    bars = [
        {
            "t": (start + timedelta(days=i)).isoformat(),
            "o": c, "h": c, "l": c, "c": c, "v": 1_000_000.0,
        }
        for i, c in enumerate(closes)
    ]
    with repo.session() as s:
        write_bars(s, ticker, bars)


def _seed_trades(repo: TradeLogRepository, pnls: list[float]) -> None:
    """Round trips, NEWEST FIRST in the argument, written with descending exits."""
    now = datetime.now(UTC)
    with repo.session() as s:
        for i, pnl in enumerate(pnls):
            s.add(
                ClosedTradeRow(
                    trade_id=f"t{i}",
                    symbol="AAPL",
                    direction="LONG",
                    quantity=1.0,
                    entry_price=100.0,
                    exit_price=100.0 + pnl,
                    realized_pnl=pnl,
                    realized_pnl_pct=pnl / 100.0,
                    holding_days=1.0,
                    open_activity_id=f"o{i}",
                    close_activity_id=f"c{i}",
                    opened_at_utc=now - timedelta(days=i + 1),
                    closed_at_utc=now - timedelta(days=i),
                    reconciled_at_utc=now,
                )
            )
        s.commit()




class TestRollingPriceStats:
    """The price-anomaly breaker was fed `rolling_mean = the price itself`, so
    its z-score was |p - p| / std — exactly zero, every ticker, every run. It
    read as a working control in every log."""

    def test_none_when_history_is_too_short(self, tmp_path) -> None:
        from tradingagents_us.risk.market_inputs import rolling_price_stats

        repo = _repo(tmp_path)
        _seed(repo, "AAPL", [100.0] * 5)
        # A caller must SKIP the check, not substitute a band: a fabricated
        # deviation is how this became inert in the first place.
        assert rolling_price_stats("AAPL", repo=repo) is None

    def test_computes_mean_and_sample_deviation(self, tmp_path) -> None:
        from tradingagents_us.risk.market_inputs import rolling_price_stats

        repo = _repo(tmp_path)
        _seed(repo, "AAPL", [10.0, 12.0] * 8)
        out = rolling_price_stats("AAPL", repo=repo)
        assert out is not None
        mean, std = out
        assert mean == pytest.approx(11.0)
        # Sample sd of an alternating 10/12 series, n=16.
        assert std == pytest.approx(1.0328, abs=1e-3)

    def test_a_flat_series_has_zero_deviation_and_the_caller_must_handle_it(
        self, tmp_path
    ) -> None:
        from tradingagents_us.risk.market_inputs import rolling_price_stats

        repo = _repo(tmp_path)
        _seed(repo, "AAPL", [100.0] * 20)
        out = rolling_price_stats("AAPL", repo=repo)
        assert out is not None
        assert out[1] == 0.0
        # The breaker guards on `rolling_std > 0`, so a halted-but-unmoved name
        # skips the check rather than dividing by zero.


class TestRecentLossStreak:
    """`record_trade_result` was never called, so the consecutive-loss counter
    sat at zero from process start to exit and a five-loss run halted nothing."""

    def test_zero_on_an_empty_ledger(self, tmp_path) -> None:
        from tradingagents_us.risk.market_inputs import recent_loss_streak

        assert recent_loss_streak(repo=_repo(tmp_path)) == 0

    def test_counts_back_from_the_newest_close(self, tmp_path) -> None:
        from tradingagents_us.risk.market_inputs import recent_loss_streak

        repo = _repo(tmp_path)
        # Newest first: three losses, then a winner that ends the streak.
        _seed_trades(repo, [-5.0, -3.0, -1.0, 10.0, -99.0])
        assert recent_loss_streak(repo=repo) == 3

    def test_a_break_even_trade_ends_the_streak(self, tmp_path) -> None:
        from tradingagents_us.risk.market_inputs import recent_loss_streak

        repo = _repo(tmp_path)
        # It did not lose. Treating "no move" as a loss halts the book for a
        # flat week.
        _seed_trades(repo, [-5.0, 0.0, -1.0])
        assert recent_loss_streak(repo=repo) == 1

    def test_an_unbroken_run_of_losses_counts_all_of_them(self, tmp_path) -> None:
        from tradingagents_us.risk.market_inputs import recent_loss_streak

        repo = _repo(tmp_path)
        _seed_trades(repo, [-1.0] * 6)
        assert recent_loss_streak(repo=repo) == 6


class TestGateInputs:
    """The BUY gates read a cache only the prices route writes. When it went
    stale (2026-10-14: nine of eleven names with nothing newer than 09-11) every
    BUY was sized with the anomaly check skipped and the correlation cap at 0,
    and the only trace was an INFO line per ticker. `gate_inputs` is what
    preflight reports from, so it must agree with the gates, not approximate them.
    """

    @pytest.mark.parametrize("n", [0, 1, 4, 9, 10, 11, 20, 21, 40, 120])
    def test_agrees_with_the_gates_it_reports_on(self, tmp_path, n: int) -> None:
        from tradingagents_us.risk.market_inputs import (
            average_dollar_volume,
            count_correlated,
            gate_inputs,
            rolling_price_stats,
        )

        repo = _repo(tmp_path)
        # MSFT moves exactly with AAPL, so the cap counts it whenever AAPL's own
        # series is long enough to be measured at all.
        closes = [100.0 + (i % 3) for i in range(n)]
        _seed(repo, "AAPL", closes)
        _seed(repo, "MSFT", [2 * c for c in closes])

        got = gate_inputs("AAPL", repo=repo)
        assert got.anomaly_skipped == (rolling_price_stats("AAPL", repo=repo) is None)
        assert got.correlation_unmeasured == (
            count_correlated("AAPL", ["MSFT"], repo=repo) == 0
        )
        assert (got.adv_bars == 0) == (average_dollar_volume("AAPL", repo=repo) is None)

    def test_a_stale_cache_is_short_even_with_a_long_history(self, tmp_path) -> None:
        from tradingagents_us.risk.market_inputs import gate_inputs

        repo = _repo(tmp_path)
        end = date.today() - timedelta(days=33)
        # Sixty bars, all older than a month: the 20-day gates' lookback reaches
        # back 37 days, so it only sees the newest few of them.
        _seed(repo, "AAPL", [100.0] * 60, end=end)
        got = gate_inputs("AAPL", repo=repo)
        assert got.anomaly_skipped and got.adv_short
        assert got.anomaly_closes == 5
        assert got.newest_bar == end.isoformat()

    def test_nothing_cached(self, tmp_path) -> None:
        from tradingagents_us.risk.market_inputs import gate_inputs

        got = gate_inputs("aapl", repo=_repo(tmp_path))
        assert got.ticker == "AAPL"
        assert (got.anomaly_closes, got.correlation_returns, got.adv_bars) == (0, 0, 0)
        assert got.newest_bar is None


class TestDescribeShortGates:
    def _row(self, ticker: str, closes: int, returns: int, adv: int, newest: str | None):
        from tradingagents_us.risk.market_inputs import GateInputs

        return GateInputs(ticker, closes, returns, adv, newest)

    def test_silent_when_every_gate_has_its_bars(self) -> None:
        from tradingagents_us.risk.market_inputs import describe_short_gates

        rows = [self._row("AAPL", 20, 80, 20, "2026-10-14")]
        assert describe_short_gates(rows) is None
        assert describe_short_gates([]) is None

    def test_names_each_short_gate_with_the_best_count_and_the_stalest_date(self) -> None:
        from tradingagents_us.risk.market_inputs import describe_short_gates

        rows = [
            self._row("AAPL", 4, 3, 4, "2026-09-11"),
            self._row("MSFT", 2, 1, 2, "2026-09-09"),
            self._row("XOM", 20, 30, 20, "2026-10-13"),
        ]
        msg = describe_short_gates(rows)
        assert msg is not None
        assert "price-anomaly check skipped for 2/3 (at most 4 of 10 closes)" in msg
        assert "correlation cap reads 0 for 2/3 (at most 3 of 20 returns)" in msg
        assert "ADV averaged over at most 4 of 20 bars for 2/3" in msg
        # XOM is healthy, so its fresh bar must not hide the stale ones.
        assert "stalest newest bar 2026-09-09" in msg

    def test_a_name_with_no_bars_at_all_is_called_out(self) -> None:
        from tradingagents_us.risk.market_inputs import describe_short_gates

        rows = [self._row("AAPL", 4, 3, 4, "2026-09-11"), self._row("NEW", 0, 0, 0, None)]
        msg = describe_short_gates(rows)
        assert msg is not None and "stalest newest bar none cached" in msg
