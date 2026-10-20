"""/v1/eval — live paper-trading scorecard for the mobile app.

Reuses the same scorecard logic as scripts/eval_report.py so the in-app
number and the weekly push always agree. Read-only; cached briefly since
the equity curve only moves once a day.
"""

from __future__ import annotations

import time
from threading import Lock

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from scripts.eval_report import (
    GATE_MAX_DD,
    GATE_SHARPE,
    MIN_TRADING_DAYS,
    _provisional_verdict,
    _verdict,
    build_gates,
    build_scorecard,
    stability_or_none,
)
from tradingagents_us.risk.verdict_stability import Stability

from ..broker_errors import broker_http_exception
from ..deps import require_token

router = APIRouter()

_CACHE_TTL_S = 60.0
_cache: dict[str, tuple[float, EvalResult]] = {}
_lock = Lock()


class Gate(BaseModel):
    # `key` is the stable identity clients translate against; `name` stays the
    # English label so an older build that never learned the keys still renders.
    key: str = ""
    name: str
    passed: bool | None
    detail: str


class WindowOut(BaseModel):
    base: str  # ISO session date of the window's baseline close
    end: str
    total_return_pct: float
    sharpe: float
    max_dd_pct: float
    passed: bool


class SessionReturnOut(BaseModel):
    session: str
    return_pct: float


class StabilityOut(BaseModel):
    """Same-length windows ending on each of the last N sessions: how often the
    headline window's verdict would have read the same, and which session
    moved it last."""

    window_days: int
    passed: int
    total: int
    flips: int
    sharpe_min: float
    sharpe_max: float
    windows: list[WindowOut]  # oldest first; the last is the headline window
    departed: SessionReturnOut | None
    entered: SessionReturnOut | None


def _stability_out(st: Stability) -> StabilityOut:
    lo, hi = st.sharpe_range

    def _ret(r):
        return (
            None
            if r is None
            else SessionReturnOut(session=r.session.isoformat(), return_pct=round(r.ret * 100, 2))
        )

    return StabilityOut(
        window_days=st.window_days,
        passed=st.passed_count,
        total=len(st.windows),
        flips=st.flips,
        sharpe_min=round(lo, 2),
        sharpe_max=round(hi, 2),
        windows=[
            WindowOut(
                base=w.base.isoformat(),
                end=w.end.isoformat(),
                total_return_pct=round(w.total_return * 100, 2),
                sharpe=round(w.sharpe, 2),
                max_dd_pct=round(w.max_dd * 100, 2),
                passed=w.passed,
            )
            for w in st.windows
        ],
        departed=_ret(st.departed),
        entered=_ret(st.entered),
    )


class EvalResult(BaseModel):
    verdict: str
    provisional_verdict: str | None
    reasons: list[str]
    days: int
    days_required: int
    days_remaining: int
    eval_complete: bool
    total_return_pct: float
    sharpe: float
    sortino: float
    max_dd_pct: float
    calmar: float
    spy_return_pct: float | None
    risk_free_pct: float
    risk_free_source: str
    gate_sharpe: float
    gate_max_dd_pct: float
    gates: list[Gate]
    # Optional so builds that never learned it still parse; None when the
    # period has no longer history or the second broker read failed.
    stability: StabilityOut | None = None


def _build(period: str, benchmark: bool) -> EvalResult:
    sc = build_scorecard(period=period, benchmark=benchmark)
    verdict, reasons = _verdict(sc)
    return EvalResult(
        verdict=verdict,
        provisional_verdict=_provisional_verdict(sc),
        reasons=reasons,
        days=sc.days,
        days_required=MIN_TRADING_DAYS,
        days_remaining=max(0, MIN_TRADING_DAYS - sc.days),
        eval_complete=sc.days >= MIN_TRADING_DAYS,
        total_return_pct=round(sc.total_return * 100, 2),
        sharpe=round(sc.sharpe, 2),
        sortino=round(sc.sortino, 2),
        max_dd_pct=round(sc.max_dd * 100, 2),
        calmar=round(sc.calmar, 2),
        spy_return_pct=round(sc.spy_return * 100, 2) if sc.spy_return is not None else None,
        risk_free_pct=round(sc.rf_annual * 100, 2),
        risk_free_source=sc.rf_source,
        gate_sharpe=GATE_SHARPE,
        gate_max_dd_pct=GATE_MAX_DD * 100,
        gates=[Gate(**g) for g in build_gates(sc)],
        stability=_stability_out(st) if (st := stability_or_none(period, sc)) else None,
    )


@router.get("", response_model=EvalResult)
async def get_eval(
    period: str = "1M",
    benchmark: bool = True,
    user: str = Depends(require_token),
) -> EvalResult:
    key = f"{period}:{benchmark}"
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and (now - hit[0]) < _CACHE_TTL_S:
            return hit[1]
    try:
        result = _build(period, benchmark)
    except SystemExit as exc:  # build_scorecard raises this when too little history
        raise HTTPException(409, str(exc)) from exc
    except (httpx.HTTPStatusError, httpx.TransportError) as exc:
        # The scorecard is built from the broker's equity history; a refused
        # key used to surface as a bare 500 traceback. Errors are not cached.
        raise broker_http_exception(exc) from exc
    with _lock:
        _cache[key] = (now, result)
    return result
