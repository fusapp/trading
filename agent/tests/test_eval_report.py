"""Eval scorecard gate logic — the go/no-go decision must be deterministic."""

from __future__ import annotations

import pandas as pd

from scripts.eval_report import (
    GATE_MAX_DD,
    GATE_SHARPE,
    HOLDOUT_MIN_DAYS,
    MIN_TRADING_DAYS,
    Scorecard,
    _equity_series,
    _holdout_note,
    _provisional_verdict,
    _verdict,
    _walk_forward_sharpe,
)


def _sc(**kw) -> Scorecard:
    base = dict(
        days=20,
        start_equity=100_000.0,
        end_equity=110_000.0,
        total_return=0.10,
        sharpe=1.5,
        sortino=1.8,
        max_dd=-0.05,
        dd_duration=4,
        calmar=1.2,
        var95=-0.01,
        cvar95=-0.02,
        positive_days_pct=60.0,
        spy_return=0.04,
    )
    base.update(kw)
    return Scorecard(**base)


class TestVerdict:
    def test_go_when_all_gates_pass(self) -> None:
        verdict, reasons = _verdict(_sc())
        assert verdict == "GO"
        assert reasons == []

    def test_too_early_below_min_days(self) -> None:
        verdict, _ = _verdict(_sc(days=MIN_TRADING_DAYS - 1))
        assert verdict == "TOO EARLY"

    def test_no_go_on_low_sharpe(self) -> None:
        verdict, reasons = _verdict(_sc(sharpe=GATE_SHARPE - 0.01))
        assert verdict == "NO-GO"
        assert any("Sharpe" in r for r in reasons)

    def test_no_go_on_deep_drawdown(self) -> None:
        verdict, reasons = _verdict(_sc(max_dd=-(GATE_MAX_DD + 0.01)))
        assert verdict == "NO-GO"
        assert any("MaxDD" in r for r in reasons)

    def test_spy_underperformance_flags_but_does_not_block(self) -> None:
        # beats gates but lags SPY -> still GO, but a reason is surfaced
        verdict, reasons = _verdict(_sc(total_return=0.02, spy_return=0.09))
        assert verdict == "GO"
        assert any("SPY" in r for r in reasons)


class TestProvisionalVerdict:
    def test_none_once_enough_days(self) -> None:
        # real verdict is authoritative past the min-days threshold
        assert _provisional_verdict(_sc(days=MIN_TRADING_DAYS)) is None

    def test_none_before_metrics_meaningful(self) -> None:
        assert _provisional_verdict(_sc(days=1)) is None

    def test_go_trend_when_metrics_hold(self) -> None:
        sc = _sc(days=MIN_TRADING_DAYS - 2, sharpe=2.0, max_dd=-0.03)
        assert _provisional_verdict(sc) == "GO"

    def test_no_go_trend_on_low_sharpe(self) -> None:
        sc = _sc(days=MIN_TRADING_DAYS - 2, sharpe=GATE_SHARPE - 0.01)
        assert _provisional_verdict(sc) == "NO-GO"

    def test_no_go_trend_on_deep_drawdown(self) -> None:
        sc = _sc(days=MIN_TRADING_DAYS - 2, max_dd=-(GATE_MAX_DD + 0.01))
        assert _provisional_verdict(sc) == "NO-GO"

    def test_spy_lag_does_not_flip_trend(self) -> None:
        # beating SPY is a flag, not a hard gate — trend stays GO
        sc = _sc(days=MIN_TRADING_DAYS - 2, total_return=0.01, spy_return=0.09)
        assert _provisional_verdict(sc) == "GO"


class TestEquitySeries:
    def test_trims_leading_flat_baseline(self) -> None:
        # three flat clean-book days, then it starts trading
        ts = [1_700_000_000 + i * 86_400 for i in range(6)]
        eq = [100_000, 100_000, 100_000, 101_000, 100_500, 102_000]
        s = _equity_series({"timestamp": ts, "equity": eq})
        # keeps the baseline bar just before first move; drops earlier flats
        assert s.iloc[0] == 100_000
        assert len(s) <= 4
        assert s.iloc[-1] == 102_000

    def test_drops_none_and_zero_points(self) -> None:
        ts = [1_700_000_000, 1_700_086_400, 1_700_172_800]
        eq = [None, 0, 100_000]
        s = _equity_series({"timestamp": ts, "equity": eq})
        assert list(s) == [100_000.0]

    def test_empty_history_returns_empty(self) -> None:
        assert _equity_series({"timestamp": [], "equity": []}).empty
        assert isinstance(_equity_series({}), pd.Series)

    def test_bars_are_indexed_by_session_date(self) -> None:
        # Live 1D stamps (2026-10-11 read): Fri 10-02, Mon 10-05, Fri 10-09
        # sessions, each at 20:00 New York = 00:00 UTC the next day.
        s = _equity_series(
            {"timestamp": _LIVE_STAMPS, "equity": [111_514.70, 112_405.50, 112_628.69]}
        )
        assert [d.date().isoformat() for d in s.index] == [
            "2026-10-02",
            "2026-10-05",
            "2026-10-09",
        ]
        assert all(d.tz is not None and d.utcoffset().total_seconds() == 0 for d in s.index)

    def test_session_date_survives_a_new_york_midnight_stamp(self) -> None:
        # 2026-10-09 00:00 New York (04:00 UTC): left-labelled bars name the
        # same session, so the reading cannot depend on which stamp Alpaca uses.
        s = _equity_series({"timestamp": [1791518400], "equity": [100_000.0]})
        assert s.index[0].date().isoformat() == "2026-10-09"


# Fri 10-02, Mon 10-05, Fri 10-09 2026 — 1D bars as Alpaca returned them.
_LIVE_STAMPS = [1790985600, 1791244800, 1791590400]


class _FakeAlpaca:
    def __init__(self, history: dict) -> None:
        self._history = history

    def __enter__(self) -> _FakeAlpaca:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def portfolio_history(self, period: str, timeframe: str) -> dict:
        return self._history


class TestScorecardWindow:
    def _build(self, monkeypatch, history: dict, start: str | None = None):
        import scripts.eval_report as er

        seen: dict[str, object] = {}

        def _spy(a, b):
            seen["spy"] = (a.date().isoformat(), b.date().isoformat())
            return 0.01

        monkeypatch.setattr(er, "AlpacaClient", lambda: _FakeAlpaca(history))
        monkeypatch.setattr(er, "_spy_return", _spy)
        monkeypatch.setattr(er, "_risk_free_rate", lambda: (0.0, "none"))
        if start is None:
            monkeypatch.delenv("EVAL_START_DATE", raising=False)
        else:
            monkeypatch.setenv("EVAL_START_DATE", start)
        return er.build_scorecard("1M", benchmark=True), seen

    def test_spy_window_starts_on_the_baseline_session(self, monkeypatch) -> None:
        # The equity baseline is Friday 10-02's close, so SPY must be read from
        # 10-02 too; the UTC reading asked Polygon for Sat 10-03 onward and
        # silently dropped Monday's SPY move from the benchmark.
        history = {"timestamp": _LIVE_STAMPS, "equity": [100_000.0, 101_000.0, 102_000.0]}
        sc, seen = self._build(monkeypatch, history)
        assert seen["spy"] == ("2026-10-02", "2026-10-09")
        assert sc.days == 3

    def test_eval_start_date_starts_at_that_session(self, monkeypatch) -> None:
        # Mon 10-05, Tue 10-06, Wed 10-07. A cutoff of Tuesday baselines on
        # Tuesday's close; the one-day-late label read Monday's close as
        # "10-06" and smuggled the pre-cutoff session in as the baseline.
        stamps = [1791244800, 1791331200, 1791417600]
        history = {"timestamp": stamps, "equity": [100_000.0, 101_000.0, 102_000.0]}
        sc, seen = self._build(monkeypatch, history, start="2026-10-06")
        assert sc.start_equity == 101_000.0
        assert sc.days == 2
        assert seen["spy"] == ("2026-10-06", "2026-10-07")


def _equity(vals: list[float]) -> pd.Series:
    idx = pd.date_range("2026-01-01", periods=len(vals), freq="D", tz="UTC")
    return pd.Series(vals, index=idx)


class TestWalkForward:
    def test_none_below_holdout_min_days(self) -> None:
        eq = _equity([100_000.0 + i for i in range(HOLDOUT_MIN_DAYS - 1)])
        assert _walk_forward_sharpe(eq) is None

    def test_splits_two_thirds_in_sample(self) -> None:
        eq = _equity([100_000.0 * (1.001 ** i) for i in range(HOLDOUT_MIN_DAYS)])
        wf = _walk_forward_sharpe(eq)
        assert wf is not None
        is_sh, oos_sh = wf
        # steady compounding -> both Sharpes finite and strongly positive
        assert is_sh > 0 and oos_sh > 0

    def test_holdout_note_none_without_split(self) -> None:
        sc = _sc(is_sharpe=None, oos_sharpe=None)
        assert _holdout_note(sc) is None

    def test_holdout_note_flags_regime_fit_collapse(self) -> None:
        sc = _sc(is_sharpe=2.0, oos_sharpe=0.3)
        note = _holdout_note(sc)
        assert note is not None and "regime-fit" in note

    def test_holdout_note_stable_when_oos_holds(self) -> None:
        sc = _sc(is_sharpe=1.8, oos_sharpe=1.6)
        note = _holdout_note(sc)
        assert note is not None and "stable" in note


class TestMainExitCode:
    """The weekly report job's exit code signals JOB health, not the verdict.
    A WAIT/NO-GO is legitimate data (pushed + printed) — it must not mark the
    systemd unit failed. Only --strict restores the old GO-only gate.
    """

    def _patch(self, monkeypatch, sc: Scorecard) -> None:
        import scripts.eval_report as er

        monkeypatch.setattr(er, "build_scorecard", lambda **kw: sc)
        monkeypatch.setattr(er, "_print", lambda sc: None)
        monkeypatch.setattr(er, "_notify", lambda sc, v: None)

    def test_too_early_exits_zero(self, monkeypatch) -> None:
        from scripts.eval_report import main

        self._patch(monkeypatch, _sc(days=8))  # < MIN_TRADING_DAYS -> TOO EARLY
        monkeypatch.setattr("sys.argv", ["eval_report"])
        assert main() == 0

    def test_no_go_exits_zero_by_default(self, monkeypatch) -> None:
        from scripts.eval_report import main

        self._patch(monkeypatch, _sc(days=20, sharpe=0.2))  # fails Sharpe gate -> NO-GO
        monkeypatch.setattr("sys.argv", ["eval_report"])
        assert main() == 0

    def test_strict_no_go_exits_one(self, monkeypatch) -> None:
        from scripts.eval_report import main

        self._patch(monkeypatch, _sc(days=20, sharpe=0.2))
        monkeypatch.setattr("sys.argv", ["eval_report", "--strict"])
        assert main() == 1

    def test_go_exits_zero(self, monkeypatch) -> None:
        from scripts.eval_report import main

        self._patch(monkeypatch, _sc(days=20, sharpe=1.5, max_dd=-0.05))
        monkeypatch.setattr("sys.argv", ["eval_report", "--strict"])
        assert main() == 0
