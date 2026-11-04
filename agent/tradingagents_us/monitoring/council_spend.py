"""What the councils cost, and what that money bought.

Every decision row carries the metered cost of the council that produced it
(`agent_decisions.cost_usd`, since 2026-09-13), and the only place that figure
was ever read is one log line per ticker. Nobody adds them up. On the live
book that hid the plainest number in the system: October cost $549 in council
calls, and from 10-14 to 11-03 — fifteen runs, ~$25 each — not one order
reached the broker.

This module answers, from rows already persisted:

  * total spend in the window, per run day, per ticker, per rating;
  * how much of it bought a broker submission (`spend_per_submit`);
  * how much has been spent since the last order reached the broker — the
    inert streak priced in dollars.

Pure: no DB, no clock, no network. Read-only by construction — nothing here can
change what gets traded.

Definitions, kept identical to `execution.actionability` so the two reports
cannot disagree:

  * A run day is a New York session date (`session_date`), not a UTC date.
  * A council is *scheduled* when the daily run persisted an order row for it
    (it writes one per ticker, approved or not). A decision with no order row
    came from the API (Ask / analyze) and is reported apart — it is real
    spend, but not the nightly run's.
  * A council *reached the broker* when one of its order rows has a broker id.
    The agent approving its own order is intent, not execution.
  * `cost_usd` NULL means "not measured", never "free". Unmeasured councils are
    counted, and every total they touch is a floor, said so.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime

from ..execution.actionability import session_date

# NYSE sessions in an average month (252 / 12). The run-rate is a projection of
# the measured per-run mean, not a forecast of what next month will hold.
SESSIONS_PER_MONTH = 21


@dataclass(frozen=True)
class CouncilCost:
    """One council's cost and outcome, decoupled from the ORM rows."""

    ticker: str
    rating: str
    timestamp_utc: datetime
    cost_usd: float | None
    scheduled: bool
    reached_broker: bool


@dataclass(frozen=True)
class Bucket:
    """Councils, spend and broker submissions for one slice of the window."""

    councils: int = 0
    spend_usd: float = 0.0
    unmeasured: int = 0
    broker_submits: int = 0

    def add(self, c: CouncilCost) -> Bucket:
        return Bucket(
            councils=self.councils + 1,
            spend_usd=self.spend_usd + (c.cost_usd or 0.0),
            unmeasured=self.unmeasured + (c.cost_usd is None),
            broker_submits=self.broker_submits + c.reached_broker,
        )

    def plus(self, other: Bucket) -> Bucket:
        return Bucket(
            councils=self.councils + other.councils,
            spend_usd=self.spend_usd + other.spend_usd,
            unmeasured=self.unmeasured + other.unmeasured,
            broker_submits=self.broker_submits + other.broker_submits,
        )

    @property
    def is_floor(self) -> bool:
        """True when some council here was never metered, so spend is a lower bound."""
        return self.unmeasured > 0


@dataclass(frozen=True)
class SpendReport:
    total: Bucket = field(default_factory=Bucket)
    scheduled: Bucket = field(default_factory=Bucket)
    on_demand: Bucket = field(default_factory=Bucket)
    # Scheduled councils only, oldest session first.
    run_days: dict[date, Bucket] = field(default_factory=dict)
    # Most expensive first.
    by_ticker: dict[str, Bucket] = field(default_factory=dict)
    # Most frequent first.
    by_rating: dict[str, Bucket] = field(default_factory=dict)
    last_submit_session: date | None = None
    # Scheduled councils on sessions after the last broker submission (the
    # whole window when nothing in it reached the broker).
    since_last_submit: Bucket = field(default_factory=Bucket)
    run_days_since_last_submit: int = 0

    @property
    def mean_per_run_day(self) -> float | None:
        if not self.run_days:
            return None
        return self.scheduled.spend_usd / len(self.run_days)

    @property
    def spend_per_submit(self) -> float | None:
        """Scheduled spend per order that reached the broker; None when none did."""
        if self.scheduled.broker_submits == 0:
            return None
        return self.scheduled.spend_usd / self.scheduled.broker_submits

    @property
    def monthly_run_rate(self) -> float | None:
        mean = self.mean_per_run_day
        return None if mean is None else mean * SESSIONS_PER_MONTH


def build_report(councils: list[CouncilCost]) -> SpendReport:
    """Aggregate council costs. Pure: same input, same report."""
    if not councils:
        return SpendReport()

    total = scheduled = on_demand = Bucket()
    run_days: dict[date, Bucket] = defaultdict(Bucket)
    by_ticker: dict[str, Bucket] = defaultdict(Bucket)
    by_rating: dict[str, Bucket] = defaultdict(Bucket)

    for c in councils:
        total = total.add(c)
        by_ticker[c.ticker] = by_ticker[c.ticker].add(c)
        by_rating[c.rating] = by_rating[c.rating].add(c)
        if c.scheduled:
            scheduled = scheduled.add(c)
            day = session_date(c.timestamp_utc)
            run_days[day] = run_days[day].add(c)
        else:
            on_demand = on_demand.add(c)

    submit_days = [d for d, b in run_days.items() if b.broker_submits]
    last_submit = max(submit_days) if submit_days else None
    after = [b for d, b in run_days.items() if last_submit is None or d > last_submit]
    since = Bucket()
    for b in after:
        since = since.plus(b)

    return SpendReport(
        total=total,
        scheduled=scheduled,
        on_demand=on_demand,
        run_days=dict(sorted(run_days.items())),
        by_ticker=dict(sorted(by_ticker.items(), key=lambda kv: (-kv[1].spend_usd, kv[0]))),
        by_rating=dict(sorted(by_rating.items(), key=lambda kv: (-kv[1].councils, kv[0]))),
        last_submit_session=last_submit,
        since_last_submit=since,
        run_days_since_last_submit=len(after),
    )


def bucket_payload(b: Bucket) -> dict[str, object]:
    return {
        "councils": b.councils,
        "spend_usd": round(b.spend_usd, 2),
        "unmeasured": b.unmeasured,
        "is_floor": b.is_floor,
        "broker_submits": b.broker_submits,
    }


def _money(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


def report_payload(r: SpendReport) -> dict[str, object]:
    """JSON-safe form, for `--json` and anything that wants to chart it."""
    return {
        "total": bucket_payload(r.total),
        "scheduled": bucket_payload(r.scheduled),
        "on_demand": bucket_payload(r.on_demand),
        "run_days": len(r.run_days),
        "mean_per_run_day_usd": _money(r.mean_per_run_day),
        "monthly_run_rate_usd": _money(r.monthly_run_rate),
        "spend_per_submit_usd": _money(r.spend_per_submit),
        "last_submit_session": r.last_submit_session.isoformat() if r.last_submit_session else None,
        "since_last_submit": bucket_payload(r.since_last_submit),
        "run_days_since_last_submit": r.run_days_since_last_submit,
        "by_run_day": {d.isoformat(): bucket_payload(b) for d, b in r.run_days.items()},
        "by_ticker": {t: bucket_payload(b) for t, b in r.by_ticker.items()},
        "by_rating": {k: bucket_payload(b) for k, b in r.by_rating.items()},
    }
