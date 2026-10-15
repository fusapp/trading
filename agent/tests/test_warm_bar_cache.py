"""scripts.warm_bar_cache: the BUY gates' bars, fetched before the councils.

The gates read a cache only the app's charts filled. On the 2026-10-14 run nine
of eleven names had nothing newer than 09-11 in it, and every one of their BUYs
was sized with the anomaly check skipped.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import create_engine

from scripts import warm_bar_cache as wb
from tradingagents_us.dataflows.polygon import Aggregate
from tradingagents_us.risk.market_inputs import gate_inputs, gate_lookback_start
from tradingagents_us.storage import TradeLogRepository
from tradingagents_us.storage.price_cache import read_bars, write_bars

NY = ZoneInfo("America/New_York")
TODAY = date(2026, 10, 14)


def _repo(tmp_path) -> TradeLogRepository:
    return TradeLogRepository(engine=create_engine(f"sqlite:///{tmp_path}/t.db", future=True))


def _agg(d: date, close: float) -> Aggregate:
    # Polygon's stamp: midnight New York of the session.
    ms = int(datetime(d.year, d.month, d.day, tzinfo=NY).timestamp() * 1000)
    return Aggregate(ms, close, close + 1, close - 1, close, 1e6, None, None)


class FakePolygon:
    def __init__(self, refuse: frozenset[str] = frozenset()) -> None:
        self.refuse = refuse
        self.asked: list[tuple[str, date, date]] = []

    def aggregates(self, ticker: str, start: date, end: date, timespan: str = "day"):
        self.asked.append((ticker, start, end))
        if ticker in self.refuse:
            raise RuntimeError("HTTP 429")
        days = (end - start).days + 1
        return [
            _agg(start + timedelta(days=i), 100.0 + i % 3)
            for i in range(days)
            if (start + timedelta(days=i)).weekday() < 5
        ]


def _rows(repo: TradeLogRepository, ticker: str):
    with repo.session() as s:
        return read_bars(s, ticker, date(2026, 1, 1), TODAY)


def test_covers_the_longest_gate_lookback_and_dates_bars_by_session(tmp_path) -> None:
    repo, pg = _repo(tmp_path), FakePolygon()
    assert wb.warm(["AAPL"], repo=repo, polygon=pg, today=TODAY, sleep=lambda s: None) == []
    assert pg.asked == [("AAPL", gate_lookback_start(TODAY), TODAY)]
    rows = _rows(repo, "AAPL")
    assert rows[-1].bar_date == TODAY.isoformat()
    # A session's bar keeps its own date: never a weekend, which is where a
    # host-clock reading west of New York would have put Monday's.
    assert all(date.fromisoformat(r.bar_date).weekday() < 5 for r in rows)


def test_a_stale_cache_is_overwritten_and_the_gates_get_their_bars(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tradingagents_us.risk.market_inputs as mi

    class Today(date):
        @classmethod
        def today(cls) -> date:
            return TODAY

    monkeypatch.setattr(mi, "date", Today)
    repo = _repo(tmp_path)
    old = date(2026, 9, 11)
    with repo.session() as s:
        write_bars(
            s, "AAPL",
            [{"t": (old - timedelta(days=i)).isoformat(), "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}
             for i in range(30)],
        )
    assert gate_inputs("AAPL", repo=repo).anomaly_skipped

    wb.warm(["AAPL"], repo=repo, polygon=FakePolygon(), today=TODAY, sleep=lambda s: None)

    got = gate_inputs("AAPL", repo=repo)
    assert not (got.anomaly_skipped or got.correlation_unmeasured or got.adv_short)
    assert got.newest_bar == TODAY.isoformat()


def test_one_refused_name_does_not_stop_the_rest(tmp_path) -> None:
    repo, pg = _repo(tmp_path), FakePolygon(refuse=frozenset({"MSFT"}))
    slept: list[float] = []
    failed = wb.warm(["AAPL", "MSFT", "XOM"], repo=repo, polygon=pg, today=TODAY,
                     sleep=slept.append)
    assert failed == ["MSFT"]
    assert _rows(repo, "MSFT") == []
    assert _rows(repo, "XOM")
    # Paced between names for the free tier's five a minute, not before the first.
    assert slept == [wb.PACE_S, wb.PACE_S]


def test_main_always_exits_zero(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    def no_client():
        raise RuntimeError("POLYGON_API_KEY unset")

    monkeypatch.setattr(wb, "PolygonClient", no_client)
    monkeypatch.setattr(wb, "TradeLogRepository", object)
    assert wb.main(["AAPL"]) == 0
    assert "warm failed, BUY gates read the cache as it was" in capsys.readouterr().out


def test_main_names_what_is_still_short(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    repo = _repo(tmp_path)

    class Client(FakePolygon):
        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

    monkeypatch.setattr(wb, "PolygonClient", lambda: Client(refuse=frozenset({"MSFT"})))
    monkeypatch.setattr(wb, "TradeLogRepository", lambda: repo)
    monkeypatch.setattr(wb, "PACE_S", 0.0)
    assert wb.main(["aapl", "MSFT"]) == 0
    out = capsys.readouterr().out
    assert "not refreshed for MSFT" in out
    assert "price-anomaly check skipped for 1/2" in out


def test_fetched_at_is_now(tmp_path) -> None:
    # The prices route serves a window without Polygon only when its newest
    # row was fetched recently; a warm run counts as that fetch.
    repo = _repo(tmp_path)
    wb.warm(["AAPL"], repo=repo, polygon=FakePolygon(), today=TODAY, sleep=lambda s: None)
    newest = _rows(repo, "AAPL")[-1].fetched_at_utc
    if newest.tzinfo is None:
        newest = newest.replace(tzinfo=UTC)
    assert datetime.now(UTC) - newest < timedelta(minutes=1)
