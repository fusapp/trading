"""Real market inputs for the risk checks that used to be fed placeholders.

Three of the caps in `portfolio_limits` were structurally unable to fire, not
because the checks were wrong but because the caller handed them constants: a
hardcoded $1B average daily volume, an empty per-sector book, and a correlation
count of zero. This module supplies the two that need price history, reading
from the same SQLite bar cache the prices route fills so a daily run does not
spend the Polygon free tier's five-requests-a-minute budget on them.

Every function here degrades to None rather than to a convenient number. A
liquidity figure that silently means "infinite" is exactly how the floor went
quiet in the first place; the caller decides what an unknown should mean.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

from ..storage import price_cache
from ..storage.repository import TradeLogRepository

log = logging.getLogger(__name__)

# Enough history for a stable average without asking for a year of bars.
_ADV_WINDOW_DAYS = 20
# Correlation over a quarter: long enough to be meaningful, short enough to
# reflect the regime the book is actually in.
_CORR_WINDOW_DAYS = 90
# |rho| above this counts as "these two move together".
_CORR_THRESHOLD = 0.7
# Below this many overlapping sessions a correlation is noise, not signal.
_MIN_OVERLAP = 20


def _lookback_start(days: int, today: date | None = None) -> date:
    # Calendar padding so weekends and holidays still leave ~`days` sessions.
    return (today or date.today()) - timedelta(days=int(days * 1.6) + 5)


def gate_lookback_start(today: date | None = None) -> date:
    """The oldest bar date any BUY gate here reads: what a cache fill must cover."""
    return _lookback_start(
        max(_ADV_WINDOW_DAYS, _CORR_WINDOW_DAYS, _ANOMALY_WINDOW_DAYS), today
    )


def _bars(repo: TradeLogRepository, ticker: str, days: int) -> list:
    start = _lookback_start(days)
    with repo.session() as s:
        return price_cache.read_bars(s, ticker.upper(), start, date.today())


def average_dollar_volume(
    ticker: str,
    *,
    repo: TradeLogRepository | None = None,
    window: int = _ADV_WINDOW_DAYS,
) -> float | None:
    """Mean daily dollar volume over the recent window, or None if unknown.

    Dollar volume rather than share count, because the liquidity floor is
    denominated in dollars and a $5 stock and a $500 stock trading the same
    number of shares are not equally liquid.
    """
    try:
        repo = repo or TradeLogRepository()
        rows = _bars(repo, ticker, window)
    except Exception:  # noqa: BLE001 — a data gap must not break the trading run
        log.warning("ADV lookup failed for %s", ticker, exc_info=True)
        return None
    recent = rows[-window:]
    if not recent:
        return None
    return sum(r.close * r.volume for r in recent) / len(recent)


def _returns(rows: list) -> dict[str, float]:
    """Daily returns keyed by bar date, so two series can be aligned on dates
    rather than on position — a name that did not trade on a holiday would
    otherwise shift the whole series and correlate against the wrong days."""
    out: dict[str, float] = {}
    prev = None
    for r in rows:
        if prev is not None and prev.close > 0:
            out[r.bar_date] = r.close / prev.close - 1.0
        prev = r
    return out


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < _MIN_OVERLAP:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:  # a flat series has no correlation to speak of
        return None
    return num / (dx * dy)


def count_correlated(
    candidate: str,
    held: list[str],
    *,
    repo: TradeLogRepository | None = None,
    threshold: float = _CORR_THRESHOLD,
) -> int:
    """How many currently-held names move with `candidate` above `threshold`.

    Returns 0 when the candidate's own history is unavailable — the same value
    the caller used to hardcode, but reached honestly: with no series there is
    nothing to correlate, and refusing every order on a data gap would be worse
    than letting the other caps carry it.
    """
    others = [h for h in {h.upper() for h in held} if h.upper() != candidate.upper()]
    if not others:
        return 0
    try:
        repo = repo or TradeLogRepository()
        base = _returns(_bars(repo, candidate, _CORR_WINDOW_DAYS))
        if len(base) < _MIN_OVERLAP:
            return 0
        n = 0
        for sym in others:
            other = _returns(_bars(repo, sym, _CORR_WINDOW_DAYS))
            shared = sorted(set(base) & set(other))
            rho = _pearson([base[d] for d in shared], [other[d] for d in shared])
            if rho is not None and abs(rho) >= threshold:
                n += 1
        return n
    except Exception:  # noqa: BLE001 — never break the run on a data gap
        log.warning("correlation count failed for %s", candidate, exc_info=True)
        return 0


# Enough closes for a mean and a deviation that describe the current regime
# without being dominated by a quarter-old one.
_ANOMALY_WINDOW_DAYS = 20
# Below this many closes a standard deviation is a number, not a distribution.
_MIN_ANOMALY_BARS = 10


def _anomaly_closes(repo: TradeLogRepository, ticker: str, window_days: int) -> list[float]:
    rows = _bars(repo, ticker, window_days)
    return [r.close for r in rows[-window_days:] if r.close and r.close > 0]


def rolling_price_stats(
    ticker: str,
    *,
    repo: TradeLogRepository | None = None,
    window_days: int = _ANOMALY_WINDOW_DAYS,
) -> tuple[float, float] | None:
    """Mean and sample standard deviation of recent closes, or None.

    The circuit breaker's price-anomaly check was fed `rolling_mean=ref_price`
    and `rolling_std=max(ref_price * 0.02, 1.0)`, which makes its z-score
    `|price - price| / std` — exactly zero, for every ticker, on every run. The
    check could not fire. It was not mis-tuned; it was disconnected, and it read
    as a working control in every log and every report.

    None when there is not enough history. The caller must then SKIP the check
    rather than substitute a band: a fabricated deviation is how this became
    inert in the first place, and a too-wide band is indistinguishable from no
    check at all while looking like one.
    """
    repo = repo or TradeLogRepository()
    closes = _anomaly_closes(repo, ticker, window_days)
    if len(closes) < _MIN_ANOMALY_BARS:
        log.info(
            "rolling_price_stats(%s): %d closes, need %d",
            ticker, len(closes), _MIN_ANOMALY_BARS,
        )
        return None

    n = len(closes)
    mean = sum(closes) / n
    # Sample (n-1) rather than population: these closes are a window out of a
    # continuing series, not the whole population of prices.
    variance = sum((c - mean) ** 2 for c in closes) / (n - 1)
    return mean, variance**0.5


def recent_loss_streak(
    *,
    repo: TradeLogRepository | None = None,
    limit: int = 50,
) -> int:
    """How many round trips in a row have closed at a loss.

    The breaker's consecutive-loss check counts what `record_trade_result` was
    told, and nothing ever called it — the counter sat at zero from process
    start to process exit, so a five-loss streak halted nothing. The realized
    ledger already knows the answer: it is the FIFO-matched close list the app's
    "Gerçekleşen" card reads.

    Counted from the most recent close backwards, stopping at the first winner.
    A break-even trade (net exactly zero) counts as a winner: it did not lose,
    and treating "no move" as a loss would halt the book for a flat week.
    """
    repo = repo or TradeLogRepository()
    trades = repo.list_closed_trades(limit=limit)
    streak = 0
    # `list_closed_trades` returns newest first; walk until something worked.
    for t in trades:
        if (t.realized_pnl or 0.0) >= 0:
            break
        streak += 1
    return streak


@dataclass(frozen=True)
class GateInputs:
    """How much history each BUY gate above will find in the cache for one name.

    Counted with the gates' own windows and reads, so this cannot say a gate
    has its bars while the gate itself finds too few.
    """

    ticker: str
    anomaly_closes: int
    correlation_returns: int
    adv_bars: int
    newest_bar: str | None

    @property
    def anomaly_skipped(self) -> bool:
        return self.anomaly_closes < _MIN_ANOMALY_BARS

    @property
    def correlation_unmeasured(self) -> bool:
        return self.correlation_returns < _MIN_OVERLAP

    @property
    def adv_short(self) -> bool:
        return self.adv_bars < _ADV_WINDOW_DAYS


def gate_inputs(ticker: str, *, repo: TradeLogRepository | None = None) -> GateInputs:
    """What `rolling_price_stats`, `count_correlated` and `average_dollar_volume`
    will see for `ticker` if it is sized now. Reads the cache only."""
    repo = repo or TradeLogRepository()
    quarter = _bars(repo, ticker, _CORR_WINDOW_DAYS)
    return GateInputs(
        ticker=ticker.upper(),
        anomaly_closes=len(_anomaly_closes(repo, ticker, _ANOMALY_WINDOW_DAYS)),
        correlation_returns=len(_returns(quarter)),
        adv_bars=len(_bars(repo, ticker, _ADV_WINDOW_DAYS)[-_ADV_WINDOW_DAYS:]),
        newest_bar=quarter[-1].bar_date if quarter else None,
    )


def describe_short_gates(rows: list[GateInputs]) -> str | None:
    """One line naming each gate that will run short of bars, or None.

    The cache these gates read is written only by the prices route, when
    someone opens a chart; nothing in the daily run fills it. On 2026-10-14
    nine of the eleven names had no bar newer than 09-11, so their BUYs were
    sized with the anomaly check skipped and the correlation cap reading zero,
    and each said so only in an INFO line per ticker. The count is the most any
    short name had, so "at most 4 of 10" means none came close; the date is the
    stalest of their newest bars.
    """
    if not rows:
        return None
    n = len(rows)
    parts: list[str] = []
    short_any = [
        r for r in rows if r.anomaly_skipped or r.correlation_unmeasured or r.adv_short
    ]
    skipped = [r for r in rows if r.anomaly_skipped]
    if skipped:
        most = max(r.anomaly_closes for r in skipped)
        parts.append(
            f"price-anomaly check skipped for {len(skipped)}/{n} "
            f"(at most {most} of {_MIN_ANOMALY_BARS} closes)"
        )
    unmeasured = [r for r in rows if r.correlation_unmeasured]
    if unmeasured:
        most = max(r.correlation_returns for r in unmeasured)
        parts.append(
            f"correlation cap reads 0 for {len(unmeasured)}/{n} "
            f"(at most {most} of {_MIN_OVERLAP} returns)"
        )
    short = [r for r in rows if r.adv_short]
    if short:
        most = max(r.adv_bars for r in short)
        parts.append(
            f"ADV averaged over at most {most} of {_ADV_WINDOW_DAYS} bars "
            f"for {len(short)}/{n}"
        )
    if not parts:
        return None
    newest = [r.newest_bar for r in short_any]
    stalest = "none cached" if None in newest else min(d for d in newest if d)
    return (
        "BUY gates short of cached bars: " + "; ".join(parts)
        + f"; stalest newest bar {stalest}"
        + " — the bar cache is filled only when the app opens a chart"
    )
