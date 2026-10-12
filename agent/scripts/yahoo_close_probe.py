#!/usr/bin/env python3
"""Record what Yahoo returns for the newest daily bar, around 20:00 New York.

Every night since at least 2026-09-22 the councils that start after 00:00 UTC
(V, XOM, UNH on the 22:30 UTC run) analyse the previous session: the day's bar
arrives with an empty Open/High/Low/Close and a volume, `load_ohlcv` drops it
as unsettled, and every indicator ends a session early. The yfinance cache
files on the box show the row as `2026-10-09,,,,,8078799`, a volume a little
short of the 8,083,600 the settled bar later carried. The councils that run
before midnight get the settled bar.

Two mechanisms fit that row, and they need different fixes:

  * Yahoo answers with no `adjclose` for the newest bar, and `auto_adjust`
    multiplies every price by Adj Close / Close, so all four go NaN while the
    volume survives. A raw (`auto_adjust=False`) fetch would still have them.
  * Yahoo appends a separate "live" row on the same date, and yfinance's
    `fix_Yahoo_returning_live_separate` keeps the newer row of a same-day pair
    and drops the settled one. Raw prices would be empty too.

Two triggers fit the time as well: Yahoo's own 20:00 ET rollover, and the
vendor's `end` argument, which is UTC today + 1 and so moves at 00:00 UTC.
Under daylight saving those are the same instant.

This probe separates them. Per ticker it records the chart JSON as Yahoo sends
it (every timestamp, price, volume and adjclose of the newest rows), the
vendor's exact download, the same download unadjusted, and the same download
with the end a day later. It is read-only against Yahoo, touches no cache the
trader reads, and always exits 0: a failed variant is recorded, not raised.

    python -m scripts.yahoo_close_probe --tickers KO,PG --out ~/yf-probe/probe.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

_EXCHANGE_TZ = ZoneInfo("America/New_York")
_TAIL = 4  # newest rows kept per variant; the bug lives in the last one or two
_VENDOR_YEARS = 5  # load_ohlcv's window; kept so the vendor variant is its call
_CHART_DAYS = 14

Fetch = Callable[[str, dict[str, Any]], dict[str, Any]]
Download = Callable[..., Any]


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _pick(series: list[Any] | None, i: int) -> float | None:
    if not series or i >= len(series):
        return None
    return _num(series[i])


def _stamp(epoch: int | float) -> dict[str, str]:
    at = datetime.fromtimestamp(epoch, UTC)
    return {"utc": at.isoformat(), "ny": at.astimezone(_EXCHANGE_TZ).isoformat()}


def chart_rows(payload: dict[str, Any], tail: int = _TAIL) -> dict[str, Any]:
    """The newest rows of a v8 chart answer, exactly as Yahoo stamped them."""
    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        return {"error": f"no result: {(payload.get('chart') or {}).get('error')}"}
    res = result[0]
    stamps = res.get("timestamp") or []
    indicators = res.get("indicators") or {}
    quote = (indicators.get("quote") or [{}])[0]
    adj = (indicators.get("adjclose") or [{}])[0].get("adjclose")
    rows = []
    for i in range(max(0, len(stamps) - tail), len(stamps)):
        rows.append(
            {
                **_stamp(stamps[i]),
                "open": _pick(quote.get("open"), i),
                "high": _pick(quote.get("high"), i),
                "low": _pick(quote.get("low"), i),
                "close": _pick(quote.get("close"), i),
                "volume": _pick(quote.get("volume"), i),
                "adjclose": _pick(adj, i),
            }
        )
    meta = res.get("meta") or {}
    market_time = meta.get("regularMarketTime")
    return {
        "rows": rows,
        "adjclose_present": adj is not None,
        "regular_market_time": _stamp(market_time) if market_time else None,
        "regular_market_price": _num(meta.get("regularMarketPrice")),
    }


def frame_rows(frame: Any, tail: int = _TAIL) -> list[dict[str, Any]]:
    """The newest rows of a yf.download frame, NaN as null."""
    out = []
    for idx, row in frame.tail(tail).iterrows():
        day = idx.date().isoformat() if hasattr(idx, "date") else str(idx)
        out.append(
            {"date": day} | {str(c).lower().replace(" ", "_"): _num(v) for c, v in row.items()}
        )
    return out


def _exchange_midnight(day: date) -> int:
    """yfinance turns a date argument into midnight in the exchange's zone."""
    return int(datetime(day.year, day.month, day.day, tzinfo=_EXCHANGE_TZ).timestamp())


def probe_ticker(ticker: str, now: datetime, fetch: Fetch, download: Download) -> dict[str, Any]:
    """One record: the chart JSON and three downloads, each failing on its own."""
    # load_ohlcv reads pd.Timestamp.today() on a UTC box, so its dates are UTC dates.
    today = now.astimezone(UTC).date()
    try:
        start = today.replace(year=today.year - _VENDOR_YEARS)
    except ValueError:  # 29 February; pd.DateOffset lands on the 28th too
        start = today.replace(year=today.year - _VENDOR_YEARS, day=28)
    vendor_end = today + timedelta(days=1)
    variants: dict[str, Any] = {}

    params = {
        "period1": _exchange_midnight(today - timedelta(days=_CHART_DAYS)),
        "period2": _exchange_midnight(vendor_end),
        "interval": "1d",
        "includePrePost": False,
        "events": "div,splits,capitalGains",
    }
    try:
        variants["chart"] = chart_rows(fetch(ticker, params))
    except Exception as e:  # noqa: BLE001 — a probe records failures, it does not raise
        variants["chart"] = {"error": repr(e)[:300]}

    calls = {
        "vendor": (vendor_end, True),
        "unadjusted": (vendor_end, False),
        "end_plus_2": (today + timedelta(days=2), True),
    }
    for name, (end, adjust) in calls.items():
        try:
            frame = download(
                ticker,
                start=start.isoformat(),
                end=end.isoformat(),
                multi_level_index=False,
                progress=False,
                auto_adjust=adjust,
            )
            variants[name] = {"end": end.isoformat(), "rows": frame_rows(frame)}
        except Exception as e:  # noqa: BLE001
            variants[name] = {"end": end.isoformat(), "error": repr(e)[:300]}

    return {
        "probed_at": _stamp(now.timestamp()),
        "ticker": ticker,
        "variants": variants,
    }


def _yahoo() -> tuple[Fetch, Download]:
    import yfinance as yf
    from yfinance.const import _BASE_URL_
    from yfinance.data import YfData

    def fetch(ticker: str, params: dict[str, Any]) -> dict[str, Any]:
        # The same session, cookie and crumb history() uses.
        return YfData().get_raw_json(f"{_BASE_URL_}/v8/finance/chart/{ticker}", params=params)

    return fetch, yf.download


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tickers", default="KO,PG", help="comma-separated")
    parser.add_argument("--out", type=Path, required=True, help="JSONL file to append to")
    args = parser.parse_args(argv)

    fetch: Fetch | None = None
    download: Download | None = None
    load_error = ""
    try:
        fetch, download = _yahoo()
    except Exception as e:  # noqa: BLE001
        load_error = repr(e)[:300]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        for ticker in [t.strip().upper() for t in args.tickers.split(",") if t.strip()]:
            now = datetime.now(UTC)
            if fetch is None or download is None:
                record = {
                    "probed_at": _stamp(now.timestamp()),
                    "ticker": ticker,
                    "error": load_error,
                }
            else:
                record = probe_ticker(ticker, now, fetch, download)
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            print(f"{ticker}: {_headline(record)}")
    return 0


def _headline(record: dict[str, Any]) -> str:
    """The one fact worth a log line: is the newest vendor row settled?"""
    if "error" in record:
        return f"error {record['error']}"
    vendor = record["variants"].get("vendor", {})
    rows = vendor.get("rows") or []
    if not rows:
        return f"vendor: {vendor.get('error', 'no rows')}"
    last = rows[-1]
    state = "settled" if last.get("close") is not None else "NO CLOSE"
    return f"vendor newest {last['date']} {state} (volume {last.get('volume')})"


if __name__ == "__main__":
    sys.exit(main())
