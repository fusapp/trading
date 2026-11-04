"""Council spend: what the councils cost, and how much of it reached the broker."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta

import pytest

from scripts import spend_report as cli
from tradingagents_us.monitoring.council_spend import (
    SESSIONS_PER_MONTH,
    CouncilCost,
    build_report,
    report_payload,
)
from tradingagents_us.schemas import AgentDecision, TradeOrder
from tradingagents_us.storage import TradeLogRepository, make_engine


def council(
    *,
    at: datetime = datetime(2026, 11, 3, 22, 40, tzinfo=UTC),
    ticker: str = "AAPL",
    rating: str = "Hold",
    cost: float | None = 2.0,
    scheduled: bool = True,
    broker: bool = False,
) -> CouncilCost:
    return CouncilCost(
        ticker=ticker,
        rating=rating,
        timestamp_utc=at,
        cost_usd=cost,
        scheduled=scheduled,
        reached_broker=broker,
    )


def night(session: date, n: int = 11, cost: float = 2.3, broker_at: int | None = None):
    """A nightly run: 22:30 UTC, one council every 12 min — crosses midnight UTC."""
    start = datetime(session.year, session.month, session.day, 22, 30, tzinfo=UTC)
    return [
        council(
            at=start + timedelta(minutes=12 * i),
            ticker=f"T{i}",
            cost=cost,
            broker=i == broker_at,
        )
        for i in range(n)
    ]


# ------------------------------- the report --------------------------------


def test_empty_window_reports_nothing_rather_than_zeros_that_look_measured() -> None:
    r = build_report([])
    assert r.total.councils == 0
    assert r.mean_per_run_day is None
    assert r.spend_per_submit is None
    assert r.monthly_run_rate is None


def test_unmeasured_council_is_a_floor_not_free() -> None:
    r = build_report([council(cost=2.5), council(cost=None, ticker="MSFT")])
    assert r.total.councils == 2
    assert r.total.spend_usd == pytest.approx(2.5)
    assert r.total.unmeasured == 1
    assert r.total.is_floor
    assert r.by_ticker["MSFT"].is_floor
    assert not r.by_ticker["AAPL"].is_floor


def test_a_run_that_crosses_midnight_utc_is_one_run_day() -> None:
    r = build_report(night(date(2026, 11, 3)))
    assert list(r.run_days) == [date(2026, 11, 3)]
    assert r.run_days[date(2026, 11, 3)].councils == 11
    assert r.mean_per_run_day == pytest.approx(11 * 2.3)


def test_on_demand_councils_are_spend_but_not_the_nightly_run() -> None:
    ask = council(at=datetime(2026, 10, 8, 15, 24, tzinfo=UTC), cost=2.23, scheduled=False)
    r = build_report([*night(date(2026, 10, 7), n=2, cost=1.0), ask])
    assert r.total.spend_usd == pytest.approx(4.23)
    assert r.scheduled.spend_usd == pytest.approx(2.0)
    assert r.on_demand.councils == 1
    assert r.on_demand.spend_usd == pytest.approx(2.23)
    # An afternoon question is not a run day and does not move the run-rate.
    assert list(r.run_days) == [date(2026, 10, 7)]
    assert r.mean_per_run_day == pytest.approx(2.0)


def test_spend_per_submit_divides_nightly_spend_by_broker_acks() -> None:
    rows = night(date(2026, 10, 13), cost=2.0, broker_at=6) + night(date(2026, 10, 14), cost=2.0)
    r = build_report(rows)
    assert r.scheduled.broker_submits == 1
    assert r.spend_per_submit == pytest.approx(44.0)


def test_no_submission_means_no_price_per_submission() -> None:
    r = build_report(night(date(2026, 11, 3)))
    assert r.spend_per_submit is None


def test_since_last_submit_prices_the_inert_streak() -> None:
    # Live shape: the 10-13 run reached the broker (after midnight UTC); every
    # run after it was all-Hold.
    rows = night(date(2026, 10, 13), cost=2.0, broker_at=9)
    for d in (date(2026, 10, 14), date(2026, 10, 15), date(2026, 10, 16)):
        rows += night(d, cost=2.5)
    r = build_report(rows)
    assert r.last_submit_session == date(2026, 10, 13)
    assert r.run_days_since_last_submit == 3
    assert r.since_last_submit.councils == 33
    assert r.since_last_submit.spend_usd == pytest.approx(33 * 2.5)
    assert r.since_last_submit.broker_submits == 0


def test_since_last_submit_is_the_whole_window_when_nothing_reached_the_broker() -> None:
    rows = night(date(2026, 11, 2)) + night(date(2026, 11, 3))
    r = build_report(rows)
    assert r.last_submit_session is None
    assert r.run_days_since_last_submit == 2
    assert r.since_last_submit.spend_usd == pytest.approx(r.scheduled.spend_usd)


def test_breakdowns_are_ordered_most_expensive_and_most_frequent_first() -> None:
    r = build_report(
        [
            council(ticker="XOM", cost=1.0, rating="Underweight"),
            council(ticker="META", cost=5.0),
            council(ticker="META", cost=1.0),
            council(ticker="AAPL", cost=3.0),
        ]
    )
    assert list(r.by_ticker) == ["META", "AAPL", "XOM"]
    assert r.by_ticker["META"].spend_usd == pytest.approx(6.0)
    assert list(r.by_rating) == ["Hold", "Underweight"]


def test_monthly_run_rate_is_the_per_run_mean_times_sessions() -> None:
    r = build_report(night(date(2026, 11, 2), cost=2.0) + night(date(2026, 11, 3), cost=3.0))
    assert r.mean_per_run_day == pytest.approx(27.5)
    assert r.monthly_run_rate == pytest.approx(27.5 * SESSIONS_PER_MONTH)


def test_payload_is_json_safe_and_rounded() -> None:
    r = build_report(night(date(2026, 11, 3), cost=2.20351))
    payload = json.loads(json.dumps(report_payload(r)))
    assert payload["scheduled"]["spend_usd"] == round(11 * 2.20351, 2)
    assert payload["by_run_day"] == {"2026-11-03": payload["scheduled"]}
    assert payload["spend_per_submit_usd"] is None
    assert payload["last_submit_session"] is None


# ------------------------------- the script --------------------------------


@pytest.mark.parametrize(
    "url,expected",
    [
        ("sqlite:////opt/ai-trader/agent/local.db", "sqlite:///file:/opt/ai-trader/agent/local.db?mode=ro&uri=true"),
        ("sqlite:///./local.db", "sqlite:///file:./local.db?mode=ro&uri=true"),
        ("sqlite://", "sqlite://"),
        ("sqlite:///file:x.db?mode=ro&uri=true", "sqlite:///file:x.db?mode=ro&uri=true"),
        ("postgresql+psycopg://u@h/db", "postgresql+psycopg://u@h/db"),
    ],
)
def test_read_only_url(url: str, expected: str) -> None:
    assert cli.read_only_url(url) == expected


def _decision(dec_id: str, ticker: str, at: datetime, cost: float | None) -> AgentDecision:
    return AgentDecision(
        ticker=ticker,
        market="US",
        quote_currency="USD",
        rating="Hold",
        suggested_size_pct=0.0,
        reasoning=[],
        final_decision_text="**Rating**: Hold",
        timestamp_utc=at,
        decision_id=dec_id,
        cost_usd=cost,
    )


def _order(order_id: str, dec_id: str, ticker: str, at: datetime) -> TradeOrder:
    return TradeOrder(
        order_id=order_id,
        decision_id=dec_id,
        ticker=ticker,
        market="US",
        side="BUY",
        quantity=1,
        order_type="MARKET",
        stop_loss=1.0,
        risk_approved=False,
        rejection_reasons=["non-actionable rating=Hold"],
        submitted_at_utc=at + timedelta(seconds=2),
    )


@pytest.fixture
def live_db(tmp_path):
    """A trade log on disk shaped like the box's: one acked, one refused, one asked."""
    path = tmp_path / "local.db"
    repo = TradeLogRepository(engine=make_engine(f"sqlite:///{path}"))
    now = datetime.now(UTC)
    acked = now - timedelta(days=2)
    refused = now - timedelta(days=1)
    asked = now - timedelta(hours=5)
    repo.save_decision(_decision("d-acked", "XOM", acked, 2.42))
    repo.save_order(_order("o-acked", "d-acked", "XOM", acked), broker_order_id="alpaca-1")
    repo.save_decision(_decision("d-refused", "META", refused, 2.20))
    repo.save_order(_order("o-refused", "d-refused", "META", refused))
    repo.save_decision(_decision("d-asked", "AAPL", asked, None))
    repo.engine.dispose()
    return path


def test_collect_marks_scheduled_acked_and_on_demand(live_db) -> None:
    repo = TradeLogRepository(engine=make_engine(cli.read_only_url(f"sqlite:///{live_db}")))
    rows = {c.ticker: c for c in cli.collect(repo, datetime.now(UTC) - timedelta(days=30))}
    assert (rows["XOM"].scheduled, rows["XOM"].reached_broker) == (True, True)
    assert (rows["META"].scheduled, rows["META"].reached_broker) == (True, False)
    assert (rows["AAPL"].scheduled, rows["AAPL"].cost_usd) == (False, None)


def test_cli_reads_without_touching_the_database(live_db, capsys) -> None:
    before = hashlib.sha256(live_db.read_bytes()).hexdigest()
    assert cli.main(["--db-url", f"sqlite:///{live_db}", "--tickers"]) == 0
    out = capsys.readouterr().out
    assert "nightly run" in out
    assert "$4.62" in out  # 2.42 + 2.20; the unmetered Ask is not $0, it is a floor
    assert "floor" in out.splitlines()[0]
    assert hashlib.sha256(live_db.read_bytes()).hexdigest() == before

    assert cli.main(["--db-url", f"sqlite:///{live_db}", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scheduled"]["broker_submits"] == 1
    assert payload["on_demand"]["unmeasured"] == 1
    assert payload["spend_per_submit_usd"] == pytest.approx(4.62)
