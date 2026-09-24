"""Unit tests for ledger → breakdown / PNG reporting."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.ledger import LEDGER_HEADER
from scripts.report import (
    EXIT_TYPES,
    load_trades,
    write_breakdown,
    write_report,
)


def _row(**kwargs) -> dict:
    base = {h: "" for h in LEDGER_HEADER}
    base.update(kwargs)
    return base


def _write_ledger(path: Path) -> None:
    rows = [
        _row(
            datetime="2024-01-01T00:00:00Z", env_id="0", pair="BTCUSDT",
            side="long", leverage="2.0", type="market_open",
            open_price="40000", exit_price="", collateral="200",
            size_usdc="400", size_base="0.01", symbol="BTC",
            fees_usdc="0.16", pnl_usdc="", pnl_pct="",
        ),
        _row(
            datetime="2024-01-01T00:05:00Z", env_id="0", pair="BTCUSDT",
            side="long", leverage="2.0", type="take_profit",
            open_price="40000", exit_price="41000", collateral="200",
            size_usdc="400", size_base="0.01", symbol="BTC",
            fees_usdc="0.16", pnl_usdc="4.84", pnl_pct="2.42",
        ),
        _row(
            datetime="2024-01-01T00:10:00Z", env_id="1", pair="ETHUSDT",
            side="short", leverage="5.5", type="market_open",
            open_price="2000", exit_price="", collateral="350",
            size_usdc="1925", size_base="0.96", symbol="ETH",
            fees_usdc="0.77", pnl_usdc="", pnl_pct="",
        ),
        _row(
            datetime="2024-01-01T00:15:00Z", env_id="1", pair="ETHUSDT",
            side="short", leverage="5.5", type="stop_loss",
            open_price="2000", exit_price="2050", collateral="350",
            size_usdc="1925", size_base="0.96", symbol="ETH",
            fees_usdc="0.77", pnl_usdc="-48.0", pnl_pct="-13.71",
        ),
        _row(
            datetime="2024-01-01T00:20:00Z", env_id="0", pair="BTCUSDT",
            side="long", leverage="1.5", type="market_open",
            open_price="40500", exit_price="", collateral="150",
            size_usdc="225", size_base="0.005", symbol="BTC",
            fees_usdc="0.09", pnl_usdc="", pnl_pct="",
        ),
        _row(
            datetime="2024-01-01T00:25:00Z", env_id="0", pair="BTCUSDT",
            side="long", leverage="1.5", type="market_close",
            open_price="40500", exit_price="40600", collateral="150",
            size_usdc="225", size_base="0.005", symbol="BTC",
            fees_usdc="0.09", pnl_usdc="0.46", pnl_pct="0.31",
        ),
        # Auto-reset rewind on env0 → new episode
        _row(
            datetime="2024-01-01T00:00:00Z", env_id="0", pair="BTCUSDT",
            side="long", leverage="12.0", type="market_open",
            open_price="40000", exit_price="", collateral="1200",
            size_usdc="14400", size_base="0.36", symbol="BTC",
            fees_usdc="5.76", pnl_usdc="", pnl_pct="",
        ),
        _row(
            datetime="2024-01-01T00:05:00Z", env_id="0", pair="BTCUSDT",
            side="long", leverage="12.0", type="liquidation",
            open_price="40000", exit_price="38000", collateral="1200",
            size_usdc="14400", size_base="0.36", symbol="BTC",
            fees_usdc="0", pnl_usdc="-1200", pnl_pct="-100",
        ),
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_HEADER)
        w.writeheader()
        w.writerows(rows)


def test_load_trades_assigns_episodes(tmp_path: Path):
    path = tmp_path / "ledger.csv"
    _write_ledger(path)
    trades = load_trades(path)
    exits = [t for t in trades if t.is_exit]
    assert len(exits) == 4
    eps = {t.episode for t in exits}
    assert eps == {"ep1", "ep2", "ep3"}  # global ep ids; rewind → new ep


def test_breakdown_has_all_sections(tmp_path: Path):
    path = tmp_path / "ledger.csv"
    _write_ledger(path)
    text = write_breakdown(
        load_trades(path), tmp_path / "breakdown.txt", initial_balance=10_000.0,
    )
    for title in (
        "Per symbol",
        "Per episode",
        "Per leverage range",
        "Per collateral range",
        "Direction (long / short)",
        "Exit type",
    ):
        assert f"=== {title} ===" in text
    for col in (
        "group",
        "num trades",
        "win %",
        "avg win",
        "avg loss",
        "return %",
        "net profit",
        "sharpe",
        "max dd",
        "risk reward",
        "sortino",
        "calmar",
        "profit factor",
        "martin",
    ):
        assert col in text
    for et in EXIT_TYPES:
        assert et in text
    assert "BTCUSDT" in text
    assert "ETHUSDT" in text
    assert "long" in text and "short" in text
    # Single-trade groups can't compute Sharpe / RR without variance or both sides.
    assert "n/a" in text
    assert "nan" not in text
    assert "inf" not in text


def test_fmt_num_noncomputable():
    import math

    from scripts.report import _fmt_num

    assert _fmt_num(math.nan) == "n/a"
    assert _fmt_num(math.inf) == "n/a"
    assert _fmt_num(-math.inf) == "n/a"
    assert _fmt_num(1.2345, 2) == "1.23"


def test_equity_curve_is_monotonic(tmp_path: Path):
    from datetime import datetime

    from scripts.report import equity_and_drawdown

    path = tmp_path / "ledger.csv"
    _write_ledger(path)
    trades = load_trades(path)
    xs, ys, peaks, dds = equity_and_drawdown(trades, 10_000.0)
    assert len(xs) == len(ys) == len(peaks) == len(dds)
    real_x = [x for x in xs if x is not None]
    real_y = [y for y in ys if y is not None]
    real_d = [d for d in dds if d is not None]
    assert all(isinstance(x, datetime) for x in real_x)
    assert real_x == sorted(real_x)
    assert len(set(real_x)) == len(real_x)
    assert max(real_d) == 0.0
    assert min(real_d) <= 0.0
    assert min(real_d) > -2.0
    n_eps = len({t.episode for t in trades})
    # Each episode run restarts at initial_balance (env auto-reset).
    assert real_y.count(10_000.0) >= n_eps
    if n_eps > 1:
        assert None in xs  # line breaks between episode segments


def test_resolve_theme_random_and_explicit():
    from utils.viz_data import THEMES

    from scripts.report import resolve_theme

    assert resolve_theme("noir") == "noir"
    assert resolve_theme("random") in THEMES
    assert resolve_theme(None) in THEMES


def test_write_report_pngs(tmp_path: Path):
    path = tmp_path / "ledger.csv"
    _write_ledger(path)
    out = write_report(path, tmp_path, initial_balance=10_000.0, theme_name="retrowave")
    assert out["theme"] == "retrowave"
    assert Path(out["breakdown"]).exists() and Path(out["breakdown"]).stat().st_size > 0
    assert Path(out["performance"]).exists() and Path(out["performance"]).stat().st_size > 1000
    assert Path(out["distributions"]).exists() and Path(out["distributions"]).stat().st_size > 1000
    assert not any(k.startswith("performance_ep") for k in out)


def test_write_report_per_episode(tmp_path: Path):
    path = tmp_path / "ledger.csv"
    _write_ledger(path)
    out = write_report(
        path, tmp_path, initial_balance=10_000.0, theme_name="retrowave",
        per_episode=True,
    )
    ep_paths = [p for k, p in out.items() if k.startswith("performance_ep")]
    assert ep_paths
    for p in ep_paths:
        assert Path(p).exists() and Path(p).stat().st_size > 1000
