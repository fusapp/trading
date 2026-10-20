"""/v1/eval route — scorecard serialization with build_scorecard mocked."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scripts.eval_report import Scorecard


def _sc(**kw) -> Scorecard:
    base = dict(
        days=20,
        start_equity=100_000.0,
        end_equity=112_000.0,
        total_return=0.12,
        sharpe=1.6,
        sortino=1.9,
        max_dd=-0.04,
        dd_duration=3,
        calmar=1.3,
        var95=-0.01,
        cvar95=-0.02,
        positive_days_pct=62.0,
        spy_return=0.05,
    )
    base.update(kw)
    return Scorecard(**base)


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.delenv("DEV_API_TOKEN", raising=False)
    monkeypatch.delenv("COGNITO_USER_POOL_ID", raising=False)
    import api.routes.eval as mod

    monkeypatch.setattr(mod, "_cache", {})
    from api.main import app

    return TestClient(app)


def test_go_verdict_serialized(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("api.routes.eval.build_scorecard", lambda period, benchmark: _sc())
    r = client.get("/v1/eval")
    assert r.status_code == 200
    b = r.json()
    assert b["verdict"] == "GO"
    assert b["provisional_verdict"] is None  # past min-days, real verdict rules
    assert b["sharpe"] == 1.6
    assert b["max_dd_pct"] == -4.0
    assert b["total_return_pct"] == 12.0
    assert b["spy_return_pct"] == 5.0
    assert b["gate_sharpe"] == 1.0


def test_provisional_trend_below_min_days(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # still TOO EARLY, but current metrics project a GO trend
    monkeypatch.setattr(
        "api.routes.eval.build_scorecard", lambda period, benchmark: _sc(days=6)
    )
    b = client.get("/v1/eval").json()
    assert b["verdict"] == "TOO EARLY"
    assert b["provisional_verdict"] == "GO"


def test_no_go_surfaces_reasons(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "api.routes.eval.build_scorecard", lambda period, benchmark: _sc(sharpe=0.3)
    )
    b = client.get("/v1/eval").json()
    assert b["verdict"] == "NO-GO"
    assert any("Sharpe" in r for r in b["reasons"])


def test_gates_and_countdown_serialized(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("api.routes.eval.build_scorecard", lambda period, benchmark: _sc())
    b = client.get("/v1/eval").json()
    assert b["days_required"] == 10
    assert b["days_remaining"] == 0  # 20 days done
    assert b["eval_complete"] is True  # min-days met, window closed
    gates = {g["name"]: g for g in b["gates"]}
    assert gates["Trading days"]["passed"] is True
    assert gates["Sharpe"]["passed"] is True
    assert gates["Max drawdown"]["passed"] is True
    assert gates["Beats SPY"]["passed"] is True  # 12% vs 5%


def test_gates_pending_below_min_days(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "api.routes.eval.build_scorecard", lambda period, benchmark: _sc(days=6)
    )
    b = client.get("/v1/eval").json()
    assert b["days_remaining"] == 4
    assert b["eval_complete"] is False  # still inside the window
    gates = {g["name"]: g for g in b["gates"]}
    assert gates["Trading days"]["passed"] is False
    assert gates["Sharpe"]["passed"] is None  # not evaluable yet
    assert gates["Trading days"]["detail"] == "6/10"


def test_too_little_history_409(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(period, benchmark):
        raise SystemExit("not enough history")

    monkeypatch.setattr("api.routes.eval.build_scorecard", _boom)
    assert client.get("/v1/eval").status_code == 409


def test_benchmark_on_by_default(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mobile calls /v1/eval with no query params — SPY benchmark must be
    requested by default so the GO/NO-GO scorecard shows the 'Beats SPY' gate."""
    seen: dict[str, bool] = {}

    def _spy(period, benchmark):
        seen["benchmark"] = benchmark
        return _sc()

    monkeypatch.setattr("api.routes.eval.build_scorecard", _spy)
    b = client.get("/v1/eval").json()
    assert seen["benchmark"] is True
    assert b["gates"][-1]["name"] == "Beats SPY"
    assert b["gates"][-1]["passed"] is True


def _raise_http(status: int):
    def _f(period: str, benchmark: bool) -> Scorecard:
        import httpx

        req = httpx.Request("GET", "https://paper-api.alpaca.markets/v2/account/portfolio/history")
        raise httpx.HTTPStatusError("x", request=req, response=httpx.Response(status, request=req))

    return _f


@pytest.mark.parametrize(
    ("status", "reason"),
    [(401, "broker_auth_refused"), (403, "broker_auth_refused"), (500, "broker_error")],
)
def test_broker_http_error_is_named_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, status: int, reason: str
) -> None:
    monkeypatch.setattr("api.routes.eval.build_scorecard", _raise_http(status))
    r = client.get("/v1/eval")
    assert r.status_code == 503
    assert r.json()["detail"] == f"{reason}: alpaca answered {status}"


def test_broker_unreachable_is_503_and_not_cached(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    def _down(period: str, benchmark: bool) -> Scorecard:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr("api.routes.eval.build_scorecard", _down)
    assert client.get("/v1/eval").json()["detail"] == "broker_unreachable"
    # the failure must not be cached: once the broker answers, so do we
    monkeypatch.setattr("api.routes.eval.build_scorecard", lambda period, benchmark: _sc())
    assert client.get("/v1/eval").status_code == 200


def test_stability_serialized_when_available(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import date

    from tradingagents_us.risk.verdict_stability import SessionReturn, Stability, WindowVerdict

    def _w(base: date, end: date, sharpe: float, passed: bool) -> WindowVerdict:
        return WindowVerdict(base, end, 0.0137, sharpe, -0.0158, passed)

    st = Stability(
        window_days=21,
        windows=(
            _w(date(2026, 9, 17), date(2026, 10, 15), 2.094, True),
            _w(date(2026, 9, 18), date(2026, 10, 16), 1.19, True),
            _w(date(2026, 9, 21), date(2026, 10, 19), -0.431, False),
        ),
        departed=SessionReturn(date(2026, 9, 21), 0.01908),
        entered=SessionReturn(date(2026, 10, 19), 0.00518),
    )
    monkeypatch.setattr("api.routes.eval.build_scorecard", lambda period, benchmark: _sc())
    monkeypatch.setattr("api.routes.eval.stability_or_none", lambda period, sc: st)
    b = client.get("/v1/eval").json()["stability"]
    assert (b["window_days"], b["passed"], b["total"], b["flips"]) == (21, 2, 3, 1)
    assert (b["sharpe_min"], b["sharpe_max"]) == (-0.43, 2.09)
    assert b["windows"][-1] == {
        "base": "2026-09-21",
        "end": "2026-10-19",
        "total_return_pct": 1.37,
        "sharpe": -0.43,
        "max_dd_pct": -1.58,
        "passed": False,
    }
    assert b["departed"] == {"session": "2026-09-21", "return_pct": 1.91}
    assert b["entered"] == {"session": "2026-10-19", "return_pct": 0.52}


def test_stability_null_when_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, str] = {}

    def _none(period: str, sc: Scorecard) -> None:
        seen["period"] = period

    monkeypatch.setattr("api.routes.eval.build_scorecard", lambda period, benchmark: _sc())
    monkeypatch.setattr("api.routes.eval.stability_or_none", _none)
    b = client.get("/v1/eval?period=3M").json()
    assert b["stability"] is None
    assert b["verdict"] == "GO"  # the headline does not depend on it
    assert seen["period"] == "3M"
