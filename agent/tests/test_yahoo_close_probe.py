"""scripts.yahoo_close_probe records Yahoo's newest daily bar without judging it.

The councils that run after 00:00 UTC get a newest bar with no prices; the
probe exists to tell an absent adjclose from a separate live row, and the
20:00 ET rollover from the vendor's UTC-dated `end`. A record that rounded a
null into a number, or one failing variant that took the others down, would
answer the wrong question.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts import yahoo_close_probe as probe

# Yahoo stamps a daily bar at the open: 2026-10-09 09:30 New York (EDT). The
# separate same-day row it can append carries a later stamp, here 20:00 ET,
# which is 00:00 UTC on the 10th.
_NY_MIDNIGHT = 1791518400
SETTLED = _NY_MIDNIGHT + 9 * 3600 + 30 * 60
LIVE = _NY_MIDNIGHT + 20 * 3600


def _chart(stamps: list[int], quote: dict[str, list[Any]], adj: list[Any] | None) -> dict:
    indicators: dict[str, Any] = {"quote": [quote]}
    if adj is not None:
        indicators["adjclose"] = [{"adjclose": adj}]
    return {
        "chart": {
            "result": [
                {
                    "meta": {"regularMarketTime": LIVE, "regularMarketPrice": 168.94},
                    "timestamp": stamps,
                    "indicators": indicators,
                }
            ],
            "error": None,
        }
    }


def test_chart_rows_keep_nulls_and_both_clocks() -> None:
    payload = _chart(
        [SETTLED, LIVE],
        {
            "open": [167.87, None],
            "high": [169.5, None],
            "low": [167.1, None],
            "close": [168.94, None],
            "volume": [8083600, 8078799],
        },
        adj=[168.94, None],
    )

    out = probe.chart_rows(payload)

    first, last = out["rows"]
    assert first["close"] == 168.94 and first["adjclose"] == 168.94
    assert last["close"] is None and last["adjclose"] is None
    assert last["volume"] == 8078799.0
    assert last["utc"] == "2026-10-10T00:00:00+00:00"
    assert last["ny"] == "2026-10-09T20:00:00-04:00"
    assert out["adjclose_present"] is True
    assert out["regular_market_price"] == 168.94


def test_chart_rows_say_when_adjclose_is_missing_entirely() -> None:
    payload = _chart([SETTLED], {"close": [168.94], "volume": [8083600]}, adj=None)

    out = probe.chart_rows(payload)

    assert out["adjclose_present"] is False
    assert out["rows"][0]["adjclose"] is None
    assert out["rows"][0]["close"] == 168.94


def test_chart_rows_report_an_error_payload() -> None:
    out = probe.chart_rows({"chart": {"result": None, "error": {"code": "Not Found"}}})

    assert "Not Found" in out["error"]


def test_chart_rows_keep_only_the_tail() -> None:
    stamps = [SETTLED - 86400 * i for i in range(10, 0, -1)]
    payload = _chart(stamps, {"close": list(range(10))}, adj=list(range(10)))

    rows = probe.chart_rows(payload, tail=3)["rows"]

    assert [r["close"] for r in rows] == [7.0, 8.0, 9.0]


def test_frame_rows_turn_nan_into_null() -> None:
    frame = pd.DataFrame(
        {"Close": [168.5, math.nan], "Adj Close": [168.5, math.nan], "Volume": [1, 8078799]},
        index=pd.DatetimeIndex(["2026-10-08", "2026-10-09"], name="Date"),
    )

    rows = probe.frame_rows(frame)

    assert rows[-1] == {"date": "2026-10-09", "close": None, "adj_close": None, "volume": 8078799.0}
    assert rows[0]["close"] == 168.5
    json.dumps(rows)  # NaN would serialise as a bare NaN token


class _Recorder:
    def __init__(self, fail: set[bool] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail = fail or set()

    def __call__(self, ticker: str, **kwargs: Any) -> pd.DataFrame:
        self.calls.append({"ticker": ticker, **kwargs})
        if kwargs["auto_adjust"] in self.fail:
            raise RuntimeError("rate limited")
        return pd.DataFrame({"Close": [1.0]}, index=pd.DatetimeIndex([kwargs["end"]], name="Date"))


def test_probe_mirrors_the_vendor_call_and_moves_only_the_end() -> None:
    download = _Recorder()
    seen: dict[str, Any] = {}

    def fetch(ticker: str, params: dict[str, Any]) -> dict:
        seen.update(params)
        return _chart([SETTLED], {"close": [1.0]}, adj=[1.0])

    now = datetime(2026, 10, 13, 0, 10, tzinfo=UTC)  # 20:10 ET on the 12th
    record = probe.probe_ticker("KO", now, fetch, download)

    by_name = {(c["end"], c["auto_adjust"]) for c in download.calls}
    assert by_name == {("2026-10-14", True), ("2026-10-14", False), ("2026-10-15", True)}
    assert {c["start"] for c in download.calls} == {"2021-10-13"}
    assert all(c["multi_level_index"] is False and c["progress"] is False for c in download.calls)
    # The chart request ends where yfinance would put "2026-10-14": NY midnight.
    assert seen["period2"] == int(datetime(2026, 10, 14, 4, tzinfo=UTC).timestamp())
    assert record["probed_at"]["ny"].startswith("2026-10-12T20:10")
    assert set(record["variants"]) == {"chart", "vendor", "unadjusted", "end_plus_2"}


def test_one_failing_variant_does_not_take_the_others_down() -> None:
    download = _Recorder(fail={False})

    def fetch(ticker: str, params: dict[str, Any]) -> dict:
        raise ConnectionError("yahoo down")

    record = probe.probe_ticker("KO", datetime(2026, 10, 12, 23, 30, tzinfo=UTC), fetch, download)

    v = record["variants"]
    assert "yahoo down" in v["chart"]["error"]
    assert "rate limited" in v["unadjusted"]["error"]
    assert v["vendor"]["rows"] and v["end_plus_2"]["rows"]


def test_leap_day_start_matches_pandas_date_offset() -> None:
    download = _Recorder()
    probe.probe_ticker(
        "KO",
        datetime(2028, 2, 29, 12, tzinfo=UTC),
        lambda t, p: _chart([SETTLED], {}, None),
        download,
    )

    assert {c["start"] for c in download.calls} == {
        (pd.Timestamp("2028-02-29") - pd.DateOffset(years=5)).date().isoformat()
    }


def test_main_appends_jsonl_and_exits_zero_when_yahoo_cannot_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken() -> None:
        raise ImportError("no yfinance")

    monkeypatch.setattr(probe, "_yahoo", broken)
    out = tmp_path / "nested" / "probe.jsonl"

    assert probe.main(["--tickers", "ko, pg", "--out", str(out)]) == 0
    assert probe.main(["--tickers", "KO", "--out", str(out)]) == 0

    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["ticker"] for r in records] == ["KO", "PG", "KO"]
    assert all("no yfinance" in r["error"] for r in records)
    assert "KO: error" in capsys.readouterr().out


def test_headline_names_an_unsettled_newest_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def download(ticker: str, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame(
            {"Close": [168.5, math.nan], "Volume": [1, 8078799]},
            index=pd.DatetimeIndex(["2026-10-08", "2026-10-09"], name="Date"),
        )

    monkeypatch.setattr(
        probe, "_yahoo", lambda: ((lambda t, p: _chart([SETTLED], {}, None)), download)
    )

    probe.main(["--tickers", "XOM", "--out", str(tmp_path / "p.jsonl")])

    assert "XOM: vendor newest 2026-10-09 NO CLOSE (volume 8078799.0)" in capsys.readouterr().out
