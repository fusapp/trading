"""GET /v1/diagnostics/actionability — did the agent still have a choice?

/v1/eval reports how the equity curve did. A fully-invested book that no longer
passes its own sizing caps keeps posting the tape's Sharpe while submitting
nothing, so "the strategy is working" and "the strategy is inert" look the same
on the scorecard. These tests pin the endpoint that tells them apart.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tradingagents_us.schemas import AgentDecision, TradeOrder
from tradingagents_us.storage import TradeLogRepository


def _now() -> datetime:
    return datetime.now(UTC)


@pytest.fixture()
def repo(tmp_path: Path) -> TradeLogRepository:
    from sqlalchemy import create_engine

    return TradeLogRepository(
        engine=create_engine(f"sqlite:///{tmp_path/'d.db'}", future=True)
    )


@pytest.fixture()
def client(repo: TradeLogRepository, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.delenv("DEV_API_TOKEN", raising=False)
    monkeypatch.delenv("COGNITO_USER_POOL_ID", raising=False)

    from api.deps import get_repo
    from api.main import app

    app.dependency_overrides[get_repo] = lambda: repo
    yield TestClient(app)
    app.dependency_overrides.pop(get_repo, None)


def _persist(
    repo: TradeLogRepository,
    *,
    key: str,
    ticker: str,
    days_ago: float,
    reasons: list[str] | None = None,
    broker_id: str | None = None,
) -> None:
    ts = _now() - timedelta(days=days_ago)
    repo.save_decision(
        AgentDecision(
            decision_id=f"d-{key}",
            ticker=ticker,
            market="US",
            quote_currency="USD",
            rating="Overweight",
            entry_price=100.0,
            stop_loss=90.0,
            reasoning=[],
            timestamp_utc=ts,
        )
    )
    repo.save_order(
        TradeOrder(
            order_id=f"o-{key}",
            decision_id=f"d-{key}",
            ticker=ticker,
            market="US",
            side="BUY",
            quantity=1,
            order_type="MARKET",
            stop_loss=90.0,
            risk_approved=broker_id is not None,
            rejection_reasons=reasons or [],
            submitted_at_utc=ts,
        ),
        broker_order_id=broker_id,
    )


class TestActionability:
    def test_no_orders_reads_as_idle_not_inert(self, client: TestClient) -> None:
        body = client.get("/v1/diagnostics/actionability").json()

        assert body["verdict"] == "idle"
        assert body["orders"] == 0
        assert body["inert_run_days"] == 0
        assert body["dominant_reason"] is None
        assert body["last_submitted_at_utc"] is None

    def test_capped_book_reads_as_inert_with_the_blocking_reason(
        self, client: TestClient, repo: TradeLogRepository
    ) -> None:
        # The live 2026-08 shape: every conviction zeroed by the per-position
        # cap, nothing reaching the broker, for four straight run days.
        for day in range(4):
            for t in ("MSFT", "NVDA", "GOOGL"):
                _persist(
                    repo,
                    key=f"{t}-{day}",
                    ticker=t,
                    days_ago=day,
                    reasons=["trimmed_to_zero_by_portfolio_caps"],
                )

        body = client.get("/v1/diagnostics/actionability").json()

        assert body["verdict"] == "inert"
        assert body["orders"] == 12
        assert body["submitted"] == 0
        assert body["refused"] == 12
        assert body["inert_run_days"] == 4
        assert body["run_days"] == 4
        assert body["dominant_reason"] == "trimmed_to_zero_by_portfolio_caps"
        assert body["by_reason"]["trimmed_to_zero_by_portfolio_caps"] == 12

    def test_cash_cap_reasons_group_despite_differing_balances(
        self, client: TestClient, repo: TradeLogRepository
    ) -> None:
        for day, spendable in enumerate(("$0.00", "$18.40", "$0.00")):
            _persist(
                repo,
                key=f"cash-{day}",
                ticker="AAPL",
                days_ago=day,
                reasons=[f"trimmed_to_zero_by_cash_cap (spendable={spendable})"],
            )

        body = client.get("/v1/diagnostics/actionability").json()

        assert body["by_reason"] == {"trimmed_to_zero_by_cash_cap": 3}

    def test_a_broker_ack_clears_the_streak(
        self, client: TestClient, repo: TradeLogRepository
    ) -> None:
        _persist(repo, key="old", ticker="AAPL", days_ago=3, reasons=["non-actionable rating=Hold"])
        _persist(repo, key="fill", ticker="NVDA", days_ago=2, broker_id="brk-1")
        _persist(repo, key="new", ticker="AAPL", days_ago=1, reasons=["non-actionable rating=Hold"])

        body = client.get("/v1/diagnostics/actionability").json()

        assert body["submitted"] == 1
        assert body["inert_run_days"] == 1
        assert body["verdict"] == "active"
        assert body["last_submitted_at_utc"] is not None

    def test_window_excludes_orders_older_than_days(
        self, client: TestClient, repo: TradeLogRepository
    ) -> None:
        _persist(repo, key="ancient", ticker="AAPL", days_ago=45, broker_id="brk-old")
        _persist(
            repo, key="recent", ticker="AAPL", days_ago=1, reasons=["non-actionable rating=Hold"]
        )

        wide = client.get("/v1/diagnostics/actionability", params={"days": 90}).json()
        narrow = client.get("/v1/diagnostics/actionability", params={"days": 7}).json()

        assert wide["orders"] == 2 and wide["submitted"] == 1
        # The old fill must not keep a currently-frozen book looking active.
        assert narrow["orders"] == 1 and narrow["submitted"] == 0
        assert narrow["last_submitted_at_utc"] is None

    def test_days_param_is_bounded(self, client: TestClient) -> None:
        assert client.get("/v1/diagnostics/actionability", params={"days": 0}).status_code == 422
        assert client.get("/v1/diagnostics/actionability", params={"days": 999}).status_code == 422


def _council(
    repo: TradeLogRepository,
    *,
    key: str,
    ticker: str,
    days_ago: float,
    cost: float | None,
    rating: str = "Hold",
    nightly: bool = True,
    broker_id: str | None = None,
) -> datetime:
    """One council: a costed decision, plus the order row the nightly run writes."""
    ts = _now() - timedelta(days=days_ago)
    repo.save_decision(
        AgentDecision(
            decision_id=f"d-{key}",
            ticker=ticker,
            market="US",
            quote_currency="USD",
            rating=rating,
            reasoning=[],
            timestamp_utc=ts,
            cost_usd=cost,
        )
    )
    if nightly:
        repo.save_order(
            TradeOrder(
                order_id=f"o-{key}",
                decision_id=f"d-{key}",
                ticker=ticker,
                market="US",
                side="BUY",
                quantity=1,
                order_type="MARKET",
                stop_loss=90.0,
                risk_approved=broker_id is not None,
                rejection_reasons=[] if broker_id else [f"non-actionable rating={rating}"],
                submitted_at_utc=ts + timedelta(seconds=2),
            ),
            broker_order_id=broker_id,
        )
    return ts


class TestSpend:
    def test_empty_window_has_no_per_submit_price(self, client: TestClient) -> None:
        body = client.get("/v1/diagnostics/spend").json()

        assert body["window_days"] == 30
        assert body["total"]["councils"] == 0
        assert body["run_days"] == 0
        assert body["mean_per_run_day_usd"] is None
        assert body["spend_per_submit_usd"] is None
        assert body["last_submit_session"] is None
        assert body["since_last_submit"]["councils"] == 0

    def test_prices_the_streak_since_the_last_broker_ack(
        self, client: TestClient, repo: TradeLogRepository
    ) -> None:
        # Night 1 bought an order; nights 2 and 3 paid for councils that could
        # not act — the 2026-11 breaker-latch shape.
        acked_at = _council(
            repo, key="n1a", ticker="JPM", days_ago=5, cost=2.50, rating="Overweight",
            broker_id="brk-1",
        )
        _council(repo, key="n1b", ticker="XOM", days_ago=5, cost=2.00)
        _council(repo, key="n2a", ticker="JPM", days_ago=3, cost=2.25)
        _council(repo, key="n2b", ticker="XOM", days_ago=3, cost=2.25, rating="Underweight")
        _council(repo, key="n3a", ticker="JPM", days_ago=1, cost=None)
        # Asked from the app: real spend, but not the nightly run's.
        _council(repo, key="ask", ticker="AAPL", days_ago=0.5, cost=1.10, nightly=False)

        body = client.get("/v1/diagnostics/spend").json()

        assert body["scheduled"] == {
            "councils": 5,
            "spend_usd": 9.0,
            "unmeasured": 1,
            "is_floor": True,
            "broker_submits": 1,
        }
        assert body["on_demand"]["councils"] == 1
        assert body["on_demand"]["spend_usd"] == pytest.approx(1.10)
        assert body["total"]["spend_usd"] == pytest.approx(10.10)
        assert body["run_days"] == 3
        assert body["mean_per_run_day_usd"] == pytest.approx(3.0)
        assert body["spend_per_submit_usd"] == pytest.approx(9.0)

        from tradingagents_us.execution.actionability import session_date

        assert body["last_submit_session"] == session_date(acked_at).isoformat()
        assert body["run_days_since_last_submit"] == 2
        since = body["since_last_submit"]
        assert (since["councils"], since["spend_usd"], since["is_floor"]) == (3, 4.5, True)
        assert since["broker_submits"] == 0
        assert list(body["by_ticker"])[0] == "JPM"
        assert body["by_rating"]["Hold"]["councils"] == 4

    def test_window_excludes_older_councils(
        self, client: TestClient, repo: TradeLogRepository
    ) -> None:
        _council(repo, key="old", ticker="AAPL", days_ago=45, cost=2.0, broker_id="brk-old")
        _council(repo, key="new", ticker="AAPL", days_ago=1, cost=2.0)

        narrow = client.get("/v1/diagnostics/spend", params={"days": 7}).json()
        wide = client.get("/v1/diagnostics/spend", params={"days": 90}).json()

        assert narrow["total"]["councils"] == 1
        # The old ack must not price a currently-frozen book as having bought something.
        assert narrow["spend_per_submit_usd"] is None
        assert narrow["last_submit_session"] is None
        assert wide["scheduled"]["broker_submits"] == 1

    def test_days_param_is_bounded(self, client: TestClient) -> None:
        assert client.get("/v1/diagnostics/spend", params={"days": 0}).status_code == 422
        assert client.get("/v1/diagnostics/spend", params={"days": 999}).status_code == 422
