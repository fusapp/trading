"""How much of a rolling-window verdict is the window itself.

On 2026-10-18 the 1M scorecard read GO, Sharpe 2.03. On 10-20 it read NO-GO,
Sharpe -0.43 — and the book had made money in between (10-19 +0.52%). Nothing
about the strategy changed. The window's baseline moved from 09-18's close to
09-21's, so the 09-21 session (+1.91%, the best day in the window) left the
front of the window, and a month that had returned +1.37% now returned -0.01%.
One session leaving the front moved Sharpe further than everything that joined
at the back.

A GO/NO-GO read off one 21-session window cannot tell that apart from the
strategy breaking. This module answers the question the single verdict cannot:
of the same-length windows ending on each of the last N sessions, how many
passed, how often did the verdict flip, and — for the newest move — which
session left the front and which joined the back.

Definitions:

  - A **window** is `window_days` consecutive equity bars; the first bar is the
    baseline, so it carries `window_days - 1` returns. That is the shape
    `eval_report.build_scorecard` scores, so the newest window reproduces the
    headline Sharpe and drawdown exactly when given the same bars.
  - A window **passes** on the hard gates only: Sharpe above the gate and
    drawdown inside it. Beating SPY is a flag in the scorecard, not a gate, and
    the trading-days gate is met by construction (callers only ask once the
    headline window has enough days).
  - Every window is scored against the same annual risk-free rate, the latest
    print the headline uses, so the windows differ only in their bars.
  - **departed** is the return that was in the previous window and is not in
    the newest one (the session after the previous baseline); **entered** is
    the newest session's return. Same-length windows shifted by one session
    differ by exactly those two returns.

Pure: takes a Series, returns numbers. Read-only reporting, off the trading path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from itertools import pairwise

import pandas as pd

from . import metrics


@dataclass(frozen=True)
class WindowVerdict:
    base: date  # session whose close is the window's baseline
    end: date
    total_return: float
    sharpe: float
    max_dd: float  # negative
    passed: bool


@dataclass(frozen=True)
class SessionReturn:
    session: date
    ret: float


@dataclass(frozen=True)
class Stability:
    window_days: int
    windows: tuple[WindowVerdict, ...]  # oldest first; the last is the newest window
    departed: SessionReturn | None
    entered: SessionReturn | None

    @property
    def passed_count(self) -> int:
        return sum(w.passed for w in self.windows)

    @property
    def flips(self) -> int:
        return sum(a.passed != b.passed for a, b in pairwise(self.windows))

    @property
    def sharpe_range(self) -> tuple[float, float]:
        values = [w.sharpe for w in self.windows]
        return min(values), max(values)


def _session(ts: pd.Timestamp) -> date:
    return ts.date()


def _score(window: pd.Series, rf: float, gate_sharpe: float, gate_max_dd: float) -> WindowVerdict:
    returns = window.pct_change().dropna()
    sharpe = metrics.sharpe(returns, risk_free_rate=rf)
    max_dd, _ = metrics.max_drawdown(window)
    return WindowVerdict(
        base=_session(window.index[0]),
        end=_session(window.index[-1]),
        total_return=float(window.iloc[-1] / window.iloc[0] - 1.0),
        sharpe=sharpe,
        max_dd=max_dd,
        passed=sharpe > gate_sharpe and abs(max_dd) < gate_max_dd,
    )


def rolling_verdicts(
    equity: pd.Series,
    window_days: int,
    lookback: int,
    *,
    rf: float,
    gate_sharpe: float,
    gate_max_dd: float,
) -> Stability | None:
    """Score the `window_days`-bar windows ending on each of the last
    `lookback` sessions of `equity` (sorted by session). None when there is not
    one full window to score."""
    if window_days < 2 or lookback < 1 or len(equity) < window_days:
        return None
    count = min(lookback, len(equity) - window_days + 1)
    first_end = len(equity) - count
    windows = tuple(
        _score(equity.iloc[end - window_days + 1 : end + 1], rf, gate_sharpe, gate_max_dd)
        for end in range(first_end, len(equity))
    )
    departed = entered = None
    if count >= 2:
        returns = equity.pct_change()
        # The previous window's baseline sits one bar before the newest one's;
        # its first return is the bar after that baseline.
        left = len(equity) - window_days
        departed = SessionReturn(_session(equity.index[left]), float(returns.iloc[left]))
        entered = SessionReturn(_session(equity.index[-1]), float(returns.iloc[-1]))
    return Stability(window_days=window_days, windows=windows, departed=departed, entered=entered)


def describe(st: Stability) -> list[str]:
    """Report lines for the stability section, unindented."""
    lo, hi = st.sharpe_range
    n = len(st.windows)
    lines = [
        f"Verdict stability ({st.window_days}-bar window, last {n} ends):",
        f"  GO {st.passed_count}/{n} · {st.flips} flip(s) · Sharpe {lo:.2f} .. {hi:.2f}",
    ]
    if st.departed is not None and st.entered is not None:
        lines.append(
            f"  newest move: left {st.departed.session} {st.departed.ret * 100:+.2f}%"
            f" · joined {st.entered.session} {st.entered.ret * 100:+.2f}%"
        )
    return lines
