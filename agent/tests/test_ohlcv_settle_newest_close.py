"""A council that starts after 20:00 New York time must not analyse yesterday.

Every night since at least 2026-09-22 the councils that ran after 00:00 UTC
(V, XOM, UNH) were handed a newest bar with volume and no close, dropped it,
and computed every indicator a session early. The probe of 2026-10-12
(scripts/yahoo_close_probe.py) pinned the mechanism: Yahoo rebuilds the day's
bar once after-hours trading ends at 20:00 ET and serves it closeless — raw and
adjusted alike — until about 20:40 ET, while the same chart response still
reports the session's close as regularMarketPrice stamped after the bell. The
vendor cache is keyed by the UTC date, so the first fetch after midnight lands
in that window, and as a "historical" request it was never refetched.

These tests drive the fork patch in the vendored stockstats_utils with the
probe's own KO numbers for 2026-10-12.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

_VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "tradingagents"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from tradingagents.dataflows import stockstats_utils as su  # noqa: E402

_NY = "America/New_York"
_SESSION = "2026-10-12"
_PREV = "2026-10-09"
# The probe at 20:10 ET: the rebuilt bar's raw Open/High/Low and volume, no close.
_RAW_OPEN, _RAW_HIGH, _RAW_LOW = 88.41500091552734, 89.80000305175781, 88.5
_SESSION_CLOSE = 89.79000091552734
_BELL = pd.Timestamp("2026-10-12 16:00:02", tz=_NY)


def _download_frame(*, newest_close: float | None) -> pd.DataFrame:
    """What yf.download(auto_adjust=True) returns inside the window: no OHLC at all."""
    closeless = newest_close is None
    nan = float("nan")
    return pd.DataFrame(
        {
            "Open": [87.5999984741211, nan if closeless else _RAW_OPEN],
            "High": [88.4800033569336, nan if closeless else _RAW_HIGH],
            "Low": [87.38999938964844, nan if closeless else _RAW_LOW],
            "Close": [88.05000305175781, nan if closeless else newest_close],
            "Volume": [10428800.0, 13929905.0],
        },
        index=pd.DatetimeIndex([_PREV, _SESSION], name="Date"),
    )


class _FakeTicker:
    """yf.Ticker as the patch uses it: history() calls, then their metadata.

    The patch's own request is the only one with a `period`; get_stock_data's
    windowed request (start/end, adjusted) gets `windowed`.
    """

    def __init__(
        self,
        raw: pd.DataFrame | Exception,
        meta: dict,
        windowed: pd.DataFrame | None = None,
    ) -> None:
        self._raw = raw
        self._windowed = windowed
        self.history_metadata = meta
        self.calls = 0

    def history(self, **kwargs: object) -> pd.DataFrame:
        if "period" not in kwargs:
            assert self._windowed is not None, "unexpected windowed request"
            return self._windowed.copy()
        self.calls += 1
        if isinstance(self._raw, Exception):
            raise self._raw
        return self._raw


def _raw_history(*, low: float = _RAW_LOW, day: str = _SESSION) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [87.5999984741211, _RAW_OPEN],
            "High": [88.4800033569336, _RAW_HIGH],
            "Low": [87.38999938964844, low],
            "Close": [88.05000305175781, float("nan")],
            "Adj Close": [88.05000305175781, float("nan")],
            "Volume": [10428800, 13929905],
        },
        index=pd.DatetimeIndex([_PREV, day]).tz_localize(_NY),
    )


def _meta(*, price: float | None = _SESSION_CLOSE, stamped: object = _BELL) -> dict:
    return {"regularMarketPrice": price, "regularMarketTime": stamped}


@pytest.fixture
def box(monkeypatch, tmp_path):
    """Pin the clock to 00:10 UTC on 2026-10-13, the night of the probe."""
    monkeypatch.setattr(su, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    today = pd.Timestamp("2026-10-13 00:10:00")
    monkeypatch.setattr(su.pd.Timestamp, "today", staticmethod(lambda: today))
    downloads: list[int] = []

    def install(*, frame: pd.DataFrame, ticker: _FakeTicker | None = None) -> list[int]:
        def _download(*_a, **_k):
            downloads.append(1)
            return frame.copy()

        monkeypatch.setattr(su.yf, "download", _download)
        if ticker is not None:
            monkeypatch.setattr(su.yf, "Ticker", lambda _sym: ticker)
        else:
            def _no_ticker(_sym):
                raise AssertionError("a settled bar needs no second request")

            monkeypatch.setattr(su.yf, "Ticker", _no_ticker)
        return downloads

    cache = tmp_path / "KO-YFin-data-2021-10-13-2026-10-14.csv"
    return install, cache


def test_the_closeless_bar_gets_the_close_yahoo_stamped_after_the_bell(box) -> None:
    install, cache = box
    ticker = _FakeTicker(_raw_history(), _meta())
    install(frame=_download_frame(newest_close=None), ticker=ticker)

    out = su.load_ohlcv("KO", _SESSION)

    assert out["Date"].iloc[-1] == pd.Timestamp(_SESSION)
    last = out.iloc[-1]
    assert last["Close"] == pytest.approx(_SESSION_CLOSE)
    assert (last["Open"], last["High"], last["Low"]) == pytest.approx(
        (_RAW_OPEN, _RAW_HIGH, _RAW_LOW)
    )
    # Cached filled, so the council's next eight reads make no request at all.
    assert pd.read_csv(cache)["Close"].iloc[-1] == pytest.approx(_SESSION_CLOSE)
    su.load_ohlcv("KO", _SESSION)
    assert ticker.calls == 1


def test_a_settled_bar_makes_no_second_request(box) -> None:
    install, _ = box
    install(frame=_download_frame(newest_close=_SESSION_CLOSE))

    out = su.load_ohlcv("KO", _SESSION)

    assert out["Close"].iloc[-1] == pytest.approx(_SESSION_CLOSE)


@pytest.mark.parametrize(
    ("raw", "meta", "why"),
    [
        (_raw_history(), _meta(stamped=pd.Timestamp("2026-10-12 15:59:00", tz=_NY)),
         "stamped before the bell: the session was still trading"),
        (_raw_history(), _meta(stamped=pd.Timestamp("2026-10-09 16:00:01", tz=_NY)),
         "stamped on another session's date"),
        (_raw_history(day="2026-10-13"), _meta(), "raw newest bar is another day"),
        (_raw_history(low=89.9), _meta(), "close outside the bar's own range"),
        (_raw_history(), _meta(price=None), "no price at all"),
        (_raw_history(), _meta(stamped=1760299202), "unformatted timestamp"),
        (RuntimeError("Yahoo said no"), _meta(), "the second request failed"),
    ],
)
def test_anything_uncertain_keeps_the_last_settled_bar(box, raw, meta, why) -> None:
    install, _ = box
    install(frame=_download_frame(newest_close=None), ticker=_FakeTicker(raw, meta))

    out = su.load_ohlcv("KO", _SESSION)

    assert out["Date"].iloc[-1] == pd.Timestamp(_PREV), why
    assert out["Close"].iloc[-1] == pytest.approx(88.05000305175781), why


def _windowed(*, newest_close: float | None) -> pd.DataFrame:
    """Ticker.history(start, end) as get_stock_data asks for it: adjusted, NY-zoned."""
    frame = _download_frame(newest_close=newest_close)
    frame.index = frame.index.tz_localize(_NY)
    return frame.assign(Dividends=0.0, **{"Stock Splits": 0.0})


def _stock_table(monkeypatch, ticker: _FakeTicker) -> list[str]:
    from tradingagents.dataflows import y_finance

    monkeypatch.setattr(y_finance.yf, "Ticker", lambda _sym: ticker)
    out = y_finance.get_YFin_data_online("KO", "2026-10-08", _SESSION)
    return [line for line in out.splitlines() if line.startswith("2026-")]


def test_the_analysts_price_table_shows_the_session_close(monkeypatch) -> None:
    ticker = _FakeTicker(_raw_history(), _meta(), windowed=_windowed(newest_close=None))

    rows = _stock_table(monkeypatch, ticker)

    assert rows[-1].startswith(f"{_SESSION},88.42,89.8,88.5,89.79,")


def test_the_price_table_keeps_a_blank_row_it_cannot_settle(monkeypatch) -> None:
    # Unchanged upstream behaviour when the close cannot be vouched for.
    stale = _meta(stamped=pd.Timestamp("2026-10-12 15:59:00", tz=_NY))
    ticker = _FakeTicker(_raw_history(), stale, windowed=_windowed(newest_close=None))

    rows = _stock_table(monkeypatch, ticker)

    assert rows[-1].startswith(f"{_SESSION},,,,,")


def test_a_settled_price_table_makes_no_second_request(monkeypatch) -> None:
    ticker = _FakeTicker(
        AssertionError("a settled bar needs no second request"),
        _meta(),
        windowed=_windowed(newest_close=_SESSION_CLOSE),
    )

    rows = _stock_table(monkeypatch, ticker)

    assert rows[-1].startswith(f"{_SESSION},88.42,89.8,88.5,89.79,")
    assert ticker.calls == 0


def _age(path: Path, seconds: float) -> None:
    then = time.time() - seconds
    os.utime(path, (then, then))


def test_a_closeless_cache_is_refetched_once_the_ttl_passes(box) -> None:
    # The fill was refused (say, a half day), so the closeless frame was cached.
    install, cache = box
    downloads = install(
        frame=_download_frame(newest_close=None),
        ticker=_FakeTicker(RuntimeError("down"), _meta()),
    )
    su.load_ohlcv("KO", _SESSION)
    assert len(downloads) == 1

    su.load_ohlcv("KO", _SESSION)  # inside the TTL: no hammering Yahoo
    assert len(downloads) == 1

    # Yahoo has settled the bar since. curr_date (10-12) is before the UTC
    # today (10-13), the "historical" case that used to reuse the file forever.
    install(frame=_download_frame(newest_close=_SESSION_CLOSE))
    _age(cache, su.OHLCV_CACHE_TTL_SECONDS + 1)
    out = su.load_ohlcv("KO", _SESSION)

    assert len(downloads) == 2
    assert out["Date"].iloc[-1] == pd.Timestamp(_SESSION)
    assert out["Close"].iloc[-1] == pytest.approx(_SESSION_CLOSE)


def test_a_settled_historical_cache_is_still_never_refetched(box) -> None:
    install, cache = box
    downloads = install(frame=_download_frame(newest_close=_SESSION_CLOSE))
    su.load_ohlcv("KO", _SESSION)
    _age(cache, su.OHLCV_CACHE_TTL_SECONDS * 10)

    su.load_ohlcv("KO", _SESSION)

    assert len(downloads) == 1
