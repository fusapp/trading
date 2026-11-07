"""/v1/diagnostics — is the agent still able to act?

Companion to /v1/eval. The scorecard answers "how did the book do"; this
answers "did the book still have a choice". Both are needed because they can
disagree: a fully-invested basket that no longer passes its own sizing caps
keeps posting the tape's Sharpe while submitting nothing, and every metric on
the scorecard is computed from equity, which a frozen book still has.

`/spend` prices the same question: what the councils cost over the window and
how much of it bought an order that reached the broker. A book that cannot act
still pays for a full council on every ticker every night.

Read-only, DB-only, off the decision path. No broker call: a diagnostic that
depends on Alpaca reports "unreachable" exactly when a broker outage is the
thing you are trying to diagnose.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from scripts.spend_report import collect as collect_councils
from tradingagents_us.execution.actionability import (
    INERT_THRESHOLD_RUN_DAYS,
    OrderRecord,
    build_report,
)
from tradingagents_us.monitoring.council_spend import build_report as build_spend_report
from tradingagents_us.monitoring.council_spend import report_payload as spend_payload
from tradingagents_us.storage import TradeLogRepository

from ..deps import get_repo, require_token

router = APIRouter()

# Threshold lives in the actionability module — the push alerter
# (scripts/inert_alert.py) reads the same constant, so the badge in the app and
# the notification can never disagree about what "frozen" means.
__all__ = ["router", "INERT_THRESHOLD_RUN_DAYS"]


class ActionabilityResponse(BaseModel):
    # "active" | "inert" | "idle" — 'idle' means no order rows at all in the
    # window (the run may simply not have happened), which must not be reported
    # as the strategy having gone quiet.
    verdict: str
    window_days: int
    orders: int
    submitted: int
    refused: int
    # Normalized reason -> count, most frequent first. Sums to >= `refused`:
    # one order can be refused for several reasons at once.
    by_reason: dict[str, int]
    dominant_reason: str | None
    # Consecutive most-recent days-the-agent-ran that submitted nothing.
    # Counted in run days, not calendar days, so weekends do not inflate it.
    inert_run_days: int
    run_days: int
    inert_threshold_run_days: int
    last_submitted_at_utc: datetime | None
    first_order_at_utc: datetime | None
    last_order_at_utc: datetime | None


@router.get("/actionability", response_model=ActionabilityResponse)
async def actionability(
    user: str = Depends(require_token),
    repo: TradeLogRepository = Depends(get_repo),
    days: int = Query(30, ge=1, le=365, description="lookback window in calendar days"),
) -> ActionabilityResponse:
    """Order-flow health over the last `days`: what got submitted, and what blocked the rest."""
    since = datetime.now(UTC) - timedelta(days=days)
    rows = repo.list_orders_since(since=since)

    report = build_report(
        [
            OrderRecord(
                ticker=r.ticker,
                side=r.side,
                submitted_at_utc=r.submitted_at_utc,
                risk_approved=r.risk_approved,
                # Stored as JSON; a legacy row could hold null or a bare string,
                # and a diagnostic that 500s on its own history is worthless.
                rejection_reasons=_as_reasons(r.rejection_reasons_json),
                broker_order_id=r.broker_order_id,
            )
            for r in rows
        ]
    )

    return ActionabilityResponse(
        verdict=report.verdict(inert_threshold=INERT_THRESHOLD_RUN_DAYS),
        window_days=days,
        orders=report.orders,
        submitted=report.submitted,
        refused=report.refused,
        by_reason=report.by_reason,
        dominant_reason=report.dominant_reason,
        inert_run_days=report.inert_run_days,
        run_days=report.run_days,
        inert_threshold_run_days=INERT_THRESHOLD_RUN_DAYS,
        last_submitted_at_utc=report.last_submitted_at_utc,
        first_order_at_utc=report.first_order_at_utc,
        last_order_at_utc=report.last_order_at_utc,
    )


class SpendBucket(BaseModel):
    councils: int
    spend_usd: float
    # Councils whose cost was never metered. Any bucket with one is a floor.
    unmeasured: int
    is_floor: bool
    broker_submits: int


class SpendResponse(BaseModel):
    """What the councils cost, and what that money bought.

    Same numbers as `python -m scripts.spend_report --json` — one builder, so
    the phone and the box cannot disagree. Definitions live in
    `tradingagents_us/monitoring/council_spend.py`.
    """

    window_days: int
    total: SpendBucket
    # Nightly run (the council has an order row) vs on demand (Ask / analyze).
    scheduled: SpendBucket
    on_demand: SpendBucket
    run_days: int
    mean_per_run_day_usd: float | None
    monthly_run_rate_usd: float | None
    # None when nothing in the window reached the broker — never a division by zero.
    spend_per_submit_usd: float | None
    # New York session date (YYYY-MM-DD) of the last broker-acknowledged order.
    last_submit_session: str | None
    # Nightly spend on run days after that session: the inert streak in dollars.
    since_last_submit: SpendBucket
    run_days_since_last_submit: int
    by_run_day: dict[str, SpendBucket]
    by_ticker: dict[str, SpendBucket]
    by_rating: dict[str, SpendBucket]


@router.get("/spend", response_model=SpendResponse)
async def spend(
    user: str = Depends(require_token),
    repo: TradeLogRepository = Depends(get_repo),
    days: int = Query(30, ge=1, le=365, description="lookback window in calendar days"),
) -> SpendResponse:
    """Council spend over the last `days`, and how much of it reached the broker."""
    since = datetime.now(UTC) - timedelta(days=days)
    report = build_spend_report(collect_councils(repo, since))
    return SpendResponse(window_days=days, **spend_payload(report))


def _as_reasons(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw if x is not None]
    if isinstance(raw, str) and raw.strip():
        return [raw]
    return []
