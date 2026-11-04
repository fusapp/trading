"""Order-flow actionability: submitted vs refused, and why."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from tradingagents_us.execution.actionability import (
    OrderRecord,
    build_report,
    normalize_reason,
    session_date,
)

BASE = datetime(2026, 8, 3, 3, 0, tzinfo=UTC)  # a Monday


def order(
    *,
    day_offset: int = 0,
    ticker: str = "AAPL",
    side: str = "BUY",
    approved: bool = False,
    reasons: list[str] | None = None,
    broker_id: str | None = None,
) -> OrderRecord:
    return OrderRecord(
        ticker=ticker,
        side=side,
        submitted_at_utc=BASE + timedelta(days=day_offset),
        risk_approved=approved,
        rejection_reasons=reasons or [],
        broker_order_id=broker_id,
    )


# --------------------------- reason normalization ---------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("trimmed_to_zero_by_cash_cap (spendable=$0.00)", "trimmed_to_zero_by_cash_cap"),
        ("trimmed_to_zero_by_cash_cap (spendable=$1,204.50)", "trimmed_to_zero_by_cash_cap"),
        ("trimmed_to_zero_by_portfolio_caps", "trimmed_to_zero_by_portfolio_caps"),
        ("non-actionable rating=Hold", "non-actionable rating=Hold"),
        ("  padded reason  ", "padded reason"),
    ],
)
def test_normalize_reason_strips_only_the_live_value_suffix(raw: str, expected: str) -> None:
    assert normalize_reason(raw) == expected


def test_same_cause_with_different_balances_is_one_bucket() -> None:
    """The whole point: a persistent blocker must not scatter into one-offs."""
    report = build_report(
        [
            order(reasons=["trimmed_to_zero_by_cash_cap (spendable=$0.00)"]),
            order(day_offset=1, reasons=["trimmed_to_zero_by_cash_cap (spendable=$12.30)"]),
            order(day_offset=2, reasons=["trimmed_to_zero_by_cash_cap (spendable=$0.00)"]),
        ]
    )
    assert report.by_reason == {"trimmed_to_zero_by_cash_cap": 3}
    assert report.dominant_reason == "trimmed_to_zero_by_cash_cap"


# ------------------------------- counting -----------------------------------


def test_empty_window_is_idle_not_inert() -> None:
    """No rows means the run may not have happened — never blame the strategy."""
    report = build_report([])
    assert report.verdict() == "idle"
    assert (report.orders, report.submitted, report.refused) == (0, 0, 0)
    assert report.inert_run_days == 0
    assert report.dominant_reason is None


def test_submitted_is_broker_ack_not_self_approval() -> None:
    """risk_approved is the agent grading itself; only a broker id is execution."""
    report = build_report(
        [
            order(approved=True, broker_id=None, reasons=["broker rejected"]),
            order(day_offset=1, approved=True, broker_id="abc-123"),
        ]
    )
    assert report.submitted == 1
    assert report.refused == 1
    assert report.last_submitted_at_utc == BASE + timedelta(days=1)


def test_multi_reason_order_counts_once_per_reason() -> None:
    report = build_report(
        [order(reasons=["trimmed_to_zero_by_portfolio_caps", "sector_cap_exceeded"])]
    )
    assert report.refused == 1
    assert report.by_reason == {
        "trimmed_to_zero_by_portfolio_caps": 1,
        "sector_cap_exceeded": 1,
    }
    # Reason counts sum above `refused` by design — asserted so nobody "fixes"
    # it into a percentage of orders later.
    assert sum(report.by_reason.values()) > report.refused


def test_blank_reasons_are_not_buckets() -> None:
    report = build_report([order(reasons=["", "   ", "real_reason"])])
    assert report.by_reason == {"real_reason": 1}


def test_by_reason_is_ordered_most_frequent_first() -> None:
    report = build_report(
        [
            order(reasons=["rare"]),
            order(day_offset=1, reasons=["common"]),
            order(day_offset=1, ticker="MSFT", reasons=["common"]),
            order(day_offset=2, reasons=["common"]),
        ]
    )
    assert list(report.by_reason) == ["common", "rare"]


# ----------------------------- inert run days -------------------------------


def test_inert_counts_run_days_not_calendar_days() -> None:
    """A weekend gap must not read as two inert days on Monday morning."""
    report = build_report(
        [
            order(day_offset=0, broker_id="filled-monday"),
            # Fri (+4), then Mon (+7): the weekend produced no rows at all.
            order(day_offset=4, reasons=["non-actionable rating=Hold"]),
            order(day_offset=7, reasons=["non-actionable rating=Hold"]),
        ]
    )
    assert report.run_days == 3
    assert report.inert_run_days == 2  # Fri + Mon, not Fri..Mon
    assert report.verdict() == "active"  # 2 < threshold of 3


def test_inert_streak_breaks_at_the_most_recent_submission() -> None:
    report = build_report(
        [
            order(day_offset=0, reasons=["non-actionable rating=Hold"]),
            order(day_offset=1, broker_id="the-last-fill"),
            order(day_offset=2, reasons=["trimmed_to_zero_by_portfolio_caps"]),
            order(day_offset=3, reasons=["trimmed_to_zero_by_portfolio_caps"]),
            order(day_offset=4, reasons=["trimmed_to_zero_by_portfolio_caps"]),
        ]
    )
    # Only the streak SINCE the last submission counts; day 0 is behind it.
    assert report.inert_run_days == 3
    assert report.verdict() == "inert"
    assert report.last_submitted_at_utc == BASE + timedelta(days=1)


def test_a_day_with_one_submission_among_many_refusals_is_not_inert() -> None:
    """The daily run writes ~11 rows a day; one broker ack makes the day active."""
    same_day = [order(ticker=t, reasons=["non-actionable rating=Hold"]) for t in "ABCDE"]
    same_day.append(order(ticker="NVDA", broker_id="one-good-one"))
    report = build_report(same_day)
    assert report.run_days == 1
    assert report.inert_run_days == 0
    assert report.verdict() == "active"


def test_verdict_threshold_is_caller_tunable() -> None:
    report = build_report(
        [order(day_offset=i, reasons=["trimmed_to_zero_by_portfolio_caps"]) for i in range(2)]
    )
    assert report.inert_run_days == 2
    assert report.verdict(inert_threshold=3) == "active"
    assert report.verdict(inert_threshold=2) == "inert"


def test_unsorted_input_still_yields_the_right_streak_and_bounds() -> None:
    """Rows arrive newest-first from SQL; the report must not depend on order."""
    rows = [
        order(day_offset=3, reasons=["trimmed_to_zero_by_portfolio_caps"]),
        order(day_offset=0, broker_id="oldest-fill"),
        order(day_offset=4, reasons=["trimmed_to_zero_by_portfolio_caps"]),
        order(day_offset=2, reasons=["trimmed_to_zero_by_portfolio_caps"]),
    ]
    report = build_report(rows)
    assert report.inert_run_days == 3
    assert report.first_order_at_utc == BASE
    assert report.last_order_at_utc == BASE + timedelta(days=4)


def test_the_live_2026_08_shape_reads_as_inert() -> None:
    """Regression guard for the book this was written to catch.

    Ten names, all at the per-position cap or rated Hold, nothing reaching the
    broker for days — while the equity curve keeps posting the tape's return.
    """
    rows: list[OrderRecord] = []
    for day in range(5):
        for t in ["MSFT", "NVDA", "GOOGL", "JPM", "V"]:
            rows.append(
                order(day_offset=day, ticker=t, reasons=["trimmed_to_zero_by_portfolio_caps"])
            )
        for t in ["AAPL", "AMZN", "META", "XOM", "UNH"]:
            rows.append(order(day_offset=day, ticker=t, reasons=["non-actionable rating=Hold"]))

    report = build_report(rows)
    assert report.orders == 50
    assert report.submitted == 0
    assert report.verdict() == "inert"
    assert report.inert_run_days == 5
    assert report.dominant_reason in {
        "trimmed_to_zero_by_portfolio_caps",
        "non-actionable rating=Hold",
    }
    assert report.last_submitted_at_utc is None


# ------------------------- run day = New York session ------------------------
#
# The nightly run starts 22:30 UTC and its last councils land after midnight
# UTC. Counted by UTC date, every run was two run days: the live counter read
# 18 for 15 runs on 2026-11-04 and stepped +2 every Monday.


def _night(session: datetime, *, broker_at: int | None = None) -> list[OrderRecord]:
    """One nightly run for `session`: 11 rows, 22:30 UTC then every ~12 min.

    Rows 0-6 land before midnight UTC, rows 7-10 after it — the live shape.
    `broker_at` marks the row index that reached the broker, if any.
    """
    start = session.replace(hour=22, minute=30, tzinfo=UTC)
    return [
        OrderRecord(
            ticker=f"T{i}",
            side="BUY",
            submitted_at_utc=start + timedelta(minutes=12 * i),
            risk_approved=i == broker_at,
            rejection_reasons=[] if i == broker_at else ["non-actionable rating=Hold"],
            broker_order_id="ack" if i == broker_at else None,
        )
        for i in range(11)
    ]


def test_a_run_that_crosses_midnight_utc_is_one_run_day() -> None:
    rows = _night(datetime(2026, 11, 3))
    assert {r.submitted_at_utc.date() for r in rows} == {
        datetime(2026, 11, 3).date(),
        datetime(2026, 11, 4).date(),
    }  # the fixture really does straddle midnight UTC
    report = build_report(rows)
    assert report.run_days == 1
    assert report.inert_run_days == 1
    assert report.last_run_date == date(2026, 11, 3)


def test_a_pre_midnight_submit_covers_the_post_midnight_half_of_its_run() -> None:
    # The run reached the broker at 23:42 UTC; its last four councils (all
    # Hold) land after midnight. By UTC date that read as one inert day.
    report = build_report(_night(datetime(2026, 10, 13), broker_at=6))
    assert report.run_days == 1
    assert report.inert_run_days == 0
    assert report.verdict() == "active"


def test_fridays_run_does_not_spill_into_saturday() -> None:
    rows = _night(datetime(2026, 10, 30)) + _night(datetime(2026, 11, 2))  # Fri, Mon
    report = build_report(rows)
    assert report.run_days == 2
    assert report.inert_run_days == 2


def test_two_all_hold_runs_are_below_the_threshold_and_three_cross_it() -> None:
    two = _night(datetime(2026, 11, 2)) + _night(datetime(2026, 11, 3))
    assert build_report(two).inert_run_days == 2
    assert build_report(two).verdict() == "active"  # UTC dating paged here (3 dates)
    three = two + _night(datetime(2026, 11, 4))
    assert build_report(three).verdict() == "inert"


@pytest.mark.parametrize(
    "utc,session",
    [
        (datetime(2026, 10, 14, 0, 27, tzinfo=UTC), date(2026, 10, 13)),  # EDT, -4h
        (datetime(2026, 11, 4, 0, 41, tzinfo=UTC), date(2026, 11, 3)),  # EST, -5h
        (datetime(2026, 11, 3, 22, 32, tzinfo=UTC), date(2026, 11, 3)),
        (datetime(2026, 11, 4, 4, 59, tzinfo=UTC), date(2026, 11, 3)),  # 23:59 EST
        (datetime(2026, 11, 4, 5, 0, tzinfo=UTC), date(2026, 11, 4)),
    ],
)
def test_session_date_is_new_york_across_dst(utc: datetime, session: date) -> None:
    assert session_date(utc) == session


def test_naive_timestamps_are_utc_whatever_the_host_zone(monkeypatch) -> None:
    # SQLite returns the stored value without its zone. Read as host-local, the
    # laptop (UTC+3) and the box (UTC) would date the same row differently.
    import time

    monkeypatch.setenv("TZ", "Asia/Tokyo")
    time.tzset()
    try:
        assert session_date(datetime(2026, 11, 4, 0, 41)) == date(2026, 11, 3)
        assert session_date(datetime(2026, 11, 3, 16, 0)) == date(2026, 11, 3)
    finally:
        monkeypatch.undo()
        time.tzset()
