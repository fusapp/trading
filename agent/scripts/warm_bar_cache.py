#!/usr/bin/env python3
"""Bring the BUY gates' bars up to date before the councils run.

trade.py's price-anomaly check, correlation cap and liquidity floor read the
price bar cache, and only the prices route writes it, when someone opens a
chart. With the app closed the cache stops moving: on the 2026-10-14 run nine
of eleven names had nothing newer than 09-11, so their BUYs were sized with
the anomaly check skipped and the correlation cap measuring August.

This fetches each name's window of daily bars from Polygon and writes it, so
the gates read the market they size into. That changes what they decide, which
is the point: an anomaly check that can fire refuses orders it used to pass.

Best-effort: a name Polygon refuses keeps whatever the cache had and the run
goes on. Always exits 0; the last line says which gates are still short.

Usage:
    python -m scripts.warm_bar_cache SPY AAPL MSFT
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable
from datetime import date, datetime
from zoneinfo import ZoneInfo

from tradingagents_us.dataflows.polygon import PolygonClient
from tradingagents_us.log_redaction import install as install_log_redaction
from tradingagents_us.risk.market_inputs import (
    describe_short_gates,
    gate_inputs,
    gate_lookback_start,
)
from tradingagents_us.storage import price_cache
from tradingagents_us.storage.repository import TradeLogRepository

log = logging.getLogger("warm_bar_cache")

#: Polygon's free tier allows five requests a minute.
PACE_S = 12.0

# Polygon stamps a daily bar at midnight New York of its session; the prices
# route and manage_positions date it the same way.
_EXCHANGE_TZ = ZoneInfo("America/New_York")


def warm(
    tickers: list[str],
    *,
    repo: TradeLogRepository,
    polygon: PolygonClient,
    today: date,
    sleep: Callable[[float], None] = time.sleep,
) -> list[str]:
    """Write each name's bars over the gates' whole lookback. Returns the
    names that could not be refreshed."""
    start = gate_lookback_start(today)
    failed: list[str] = []
    for n, ticker in enumerate(tickers):
        if n:
            sleep(PACE_S)
        try:
            aggs = polygon.aggregates(ticker, start, today, timespan="day")
        except Exception as exc:  # noqa: BLE001 — one name must not stop the rest
            log.warning("%-6s bars not refreshed: %s", ticker, exc)
            failed.append(ticker)
            continue
        bars = [
            {
                "t": datetime.fromtimestamp(a.timestamp_ms / 1000, _EXCHANGE_TZ)
                .date()
                .isoformat(),
                "o": a.open,
                "h": a.high,
                "l": a.low,
                "c": a.close,
                "v": a.volume,
            }
            for a in aggs
        ]
        if not bars:
            log.warning("%-6s bars not refreshed: Polygon returned none", ticker)
            failed.append(ticker)
            continue
        with repo.session() as s:
            price_cache.write_bars(s, ticker, bars)
        log.info("%-6s %d bars through %s", ticker, len(bars), bars[-1]["t"])
    return failed


def main(argv: list[str] | None = None) -> int:
    install_log_redaction()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tickers", nargs="+")
    args = parser.parse_args(argv)
    tickers = list(dict.fromkeys(t.upper() for t in args.tickers))

    try:
        repo = TradeLogRepository()
        with PolygonClient() as polygon:
            failed = warm(tickers, repo=repo, polygon=polygon, today=date.today())
        if failed:
            print(f"bar cache: not refreshed for {', '.join(failed)}")
        short = describe_short_gates([gate_inputs(t, repo=repo) for t in tickers])
    except Exception as exc:  # noqa: BLE001 — the gates fall back to skipping, as before
        print(f"bar cache: warm failed, BUY gates read the cache as it was: {exc}")
        return 0
    print(short or f"bar cache: BUY gates have their bars for all {len(tickers)} names")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
