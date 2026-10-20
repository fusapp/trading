"""Rolling-window verdict stability — the 1M flip that was the window, not the book."""

from __future__ import annotations

from datetime import UTC, date, datetime

import httpx
import pandas as pd
import pytest

from scripts.eval_report import (
    GATE_MAX_DD,
    GATE_SHARPE,
    MIN_TRADING_DAYS,
    Scorecard,
    build_stability,
    stability_or_none,
)
from tradingagents_us.risk import metrics
from tradingagents_us.risk.verdict_stability import describe, rolling_verdicts

RF = 0.0427


def _series(values: list[float], start: str = "2026-09-01") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(values), tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def _compound(daily: list[float], base: float = 100_000.0) -> list[float]:
    out = [base]
    for r in daily:
        out.append(out[-1] * (1.0 + r))
    return out


def _roll(equity: pd.Series, window_days: int, lookback: int = 10):
    return rolling_verdicts(
        equity,
        window_days,
        lookback,
        rf=RF,
        gate_sharpe=GATE_SHARPE,
        gate_max_dd=GATE_MAX_DD,
    )


# A steady month with one big day near the front: while that day is inside the
# window the verdict passes; one session after it becomes the baseline, the
# same book reads as a loss-making month.
_DAILY = [0.004, -0.003] * 4 + [0.019] + [0.003, -0.003] * 10 + [0.0052]


class TestRollingVerdicts:
    def test_newest_window_matches_the_headline_scoring(self) -> None:
        eq = _series(_compound(_DAILY))
        st = _roll(eq, window_days=21)
        assert st is not None
        newest = eq.iloc[-21:]
        assert st.windows[-1].sharpe == pytest.approx(
            metrics.sharpe(newest.pct_change().dropna(), risk_free_rate=RF)
        )
        assert st.windows[-1].max_dd == pytest.approx(metrics.max_drawdown(newest)[0])
        assert st.windows[-1].base == newest.index[0].date()
        assert st.windows[-1].end == eq.index[-1].date()

    def test_big_day_leaving_the_front_flips_the_verdict(self) -> None:
        eq = _series(_compound(_DAILY))
        big = eq.index[9]  # the session whose return is the +1.9% day
        window = len(eq) - 9  # newest window's baseline is exactly that session
        st = _roll(eq, window_days=window)
        assert st is not None
        assert st.windows[-2].passed is True
        assert st.windows[-1].passed is False
        assert st.departed is not None and st.entered is not None
        assert st.departed.session == big.date()
        assert st.departed.ret == pytest.approx(0.019)
        assert st.entered.session == eq.index[-1].date()
        assert st.entered.ret == pytest.approx(0.0052)
        # the session that left is the newest window's baseline, so its
        # return is in the previous window and not the newest one
        assert st.windows[-1].base == big.date()

    def test_lookback_is_capped_by_available_history(self) -> None:
        eq = _series(_compound([0.001] * 24))  # 25 bars
        st = _roll(eq, window_days=21, lookback=10)
        assert st is not None
        assert len(st.windows) == 5
        assert [w.end for w in st.windows] == [d.date() for d in eq.index[-5:]]

    def test_none_without_one_full_window(self) -> None:
        eq = _series(_compound([0.001] * 9))  # 10 bars
        assert _roll(eq, window_days=11) is None
        assert _roll(eq, window_days=1) is None
        assert _roll(eq, window_days=5, lookback=0) is None

    def test_single_window_has_no_move_to_explain(self) -> None:
        eq = _series(_compound([0.001] * 20))
        st = _roll(eq, window_days=21)
        assert st is not None and len(st.windows) == 1
        assert st.departed is None and st.entered is None
        assert st.flips == 0

    def test_counts_flips_and_passes(self) -> None:
        eq = _series(_compound(_DAILY))
        st = _roll(eq, window_days=len(eq) - 9, lookback=3)
        assert st is not None
        assert [w.passed for w in st.windows] == [True, True, False]
        assert st.passed_count == 2
        assert st.flips == 1
        lo, hi = st.sharpe_range
        assert lo == st.windows[-1].sharpe and hi > GATE_SHARPE

    def test_drawdown_gate_fails_a_window_with_a_good_sharpe(self) -> None:
        # +1%/day for 30 days with one -16% day: Sharpe stays high, DD breaks 15%
        eq = _series(_compound([0.01] * 15 + [-0.16] + [0.01] * 14))
        st = _roll(eq, window_days=31, lookback=1)
        assert st is not None
        w = st.windows[-1]
        assert w.sharpe > GATE_SHARPE
        assert abs(w.max_dd) >= GATE_MAX_DD
        assert w.passed is False

    def test_describe_names_the_move(self) -> None:
        eq = _series(_compound(_DAILY))
        st = _roll(eq, window_days=len(eq) - 9, lookback=3)
        assert st is not None
        text = "\n".join(describe(st))
        assert "GO 2/3" in text and "1 flip(s)" in text
        assert f"left {eq.index[9].date()} +1.90%" in text
        assert f"joined {eq.index[-1].date()} +0.52%" in text


def _stamp(session: date) -> int:
    # Alpaca stamps a 1D bar 20:00 New York of its session = 00:00 UTC next day
    return int(datetime(session.year, session.month, session.day, tzinfo=UTC).timestamp()) + 86400


class _FakeAlpaca:
    def __init__(self, history: dict, seen: dict) -> None:
        self._history = history
        self._seen = seen

    def __enter__(self) -> _FakeAlpaca:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def portfolio_history(self, period: str, timeframe: str) -> dict:
        self._seen["period"] = period
        return self._history


def _sc(days: int, end_session: date | None, rf: float = RF) -> Scorecard:
    return Scorecard(
        days=days,
        start_equity=100_000.0,
        end_equity=101_000.0,
        total_return=0.01,
        sharpe=1.5,
        sortino=1.8,
        max_dd=-0.02,
        dd_duration=3,
        calmar=1.2,
        var95=-0.01,
        cvar95=-0.02,
        positive_days_pct=55.0,
        spy_return=None,
        rf_annual=rf,
        end_session=end_session,
    )


class TestBuildStability:
    def _history(self, n: int) -> tuple[dict, list[date]]:
        sessions = [d.date() for d in pd.bdate_range("2026-09-01", periods=n)]
        equity = _compound([0.002, -0.001] * n)[:n]
        return {"timestamp": [_stamp(s) for s in sessions], "equity": equity}, sessions

    def _patch(self, monkeypatch, history: dict) -> dict:
        import scripts.eval_report as er

        seen: dict = {}
        monkeypatch.setattr(er, "AlpacaClient", lambda: _FakeAlpaca(history, seen))
        monkeypatch.delenv("EVAL_START_DATE", raising=False)
        return seen

    def test_reads_the_longer_period_and_cuts_at_the_headline_session(self, monkeypatch) -> None:
        history, sessions = self._history(40)
        seen = self._patch(monkeypatch, history)
        # the headline scorecard ended one session before the longer read did
        st = build_stability("1M", _sc(days=21, end_session=sessions[-2]))
        assert seen["period"] == "3M"
        assert st is not None
        assert st.window_days == 21
        assert st.windows[-1].end == sessions[-2]
        assert len(st.windows) == 10

    def test_no_longer_period_or_too_few_days_is_none(self, monkeypatch) -> None:
        history, sessions = self._history(40)
        seen = self._patch(monkeypatch, history)
        assert build_stability("1A", _sc(days=21, end_session=sessions[-1])) is None
        assert build_stability("1M", _sc(MIN_TRADING_DAYS - 1, end_session=sessions[-1])) is None
        assert build_stability("1M", _sc(days=21, end_session=None)) is None
        assert seen == {}  # none of them touched the broker

    def test_broker_failure_drops_only_the_section(self, monkeypatch, capsys) -> None:
        import scripts.eval_report as er

        def _boom() -> _FakeAlpaca:
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(er, "AlpacaClient", _boom)
        assert stability_or_none("1M", _sc(days=21, end_session=date(2026, 10, 19))) is None
        assert "verdict stability unavailable" in capsys.readouterr().err

    def test_eval_start_date_bounds_the_windows(self, monkeypatch) -> None:
        history, sessions = self._history(40)
        self._patch(monkeypatch, history)
        cutoff = sessions[15]
        monkeypatch.setenv("EVAL_START_DATE", cutoff.isoformat())
        st = build_stability("1M", _sc(days=21, end_session=sessions[-1]))
        assert st is not None
        assert st.windows[0].base == cutoff  # 25 bars after the cutoff -> 5 windows
        assert len(st.windows) == 5
