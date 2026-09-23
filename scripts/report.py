"""Test-run reporting from trade ledger.csv.

Writes into the run directory:
  - breakdown.txt  — tables by symbol, episode, leverage, collateral,
                     direction, exit type
  - performance.png — equity / returns scatter / drawdown / returns hist
  - distributions.png — leverage / collateral / direction / exit type

Themes reuse ``utils.viz_data`` (default: retrowave).
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tabulate import tabulate

if TYPE_CHECKING:
    from utils.viz_data import Theme


def _theme(name: str):
    """Lazy import — utils.viz_data pulls mlx at module load."""
    from utils.viz_data import theme

    return theme(name)

EXIT_TYPES = ("market_close", "take_profit", "stop_loss", "liquidation")
OPEN_TYPE = "market_open"

# Fixed bins for categorical breakdowns (USDC collateral, leverage multiples).
LEV_BINS: list[tuple[float, float, str]] = [
    (0.0, 2.0, "1–2x"),
    (2.0, 3.0, "2–3x"),
    (3.0, 5.0, "3–5x"),
    (5.0, 10.0, "5–10x"),
    (10.0, math.inf, "10x+"),
]
COLL_BINS: list[tuple[float, float, str]] = [
    (0.0, 100.0, "<100"),
    (100.0, 250.0, "100–250"),
    (250.0, 500.0, "250–500"),
    (500.0, 1000.0, "500–1k"),
    (1000.0, math.inf, "1k+"),
]


@dataclass
class Trade:
    datetime: str
    env_id: int
    episode: str
    pair: str
    side: str
    leverage: float
    type: str
    collateral: float
    fees_usdc: float
    pnl_usdc: float | None
    pnl_pct: float | None
    is_exit: bool


def _parse_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def _f(s: str) -> float | None:
    if s is None or s == "":
        return None
    return float(s)


def _bin_label(value: float, bins: list[tuple[float, float, str]]) -> str:
    for lo, hi, label in bins:
        if lo <= value < hi:
            return label
    return bins[-1][2]


def load_trades(ledger_path: str | Path) -> list[Trade]:
    """Parse ledger CSV and assign global episode ids ``ep1``, ``ep2``, …

    A new episode starts when an env's synthetic clock regresses (auto-reset).
    Parallel env streams are numbered in first-seen order as ep1..epN.
    """
    path = Path(ledger_path)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    last_dt: dict[int, datetime] = {}
    local_ep: dict[int, int] = defaultdict(int)
    global_ep: dict[tuple[int, int], int] = {}
    next_ep = 1
    trades: list[Trade] = []
    for r in rows:
        env_id = int(r["env_id"])
        dt = _parse_dt(r["datetime"])
        if env_id in last_dt and dt < last_dt[env_id]:
            local_ep[env_id] += 1
        last_dt[env_id] = dt
        key = (env_id, local_ep[env_id])
        if key not in global_ep:
            global_ep[key] = next_ep
            next_ep += 1
        etype = r["type"]
        trades.append(
            Trade(
                datetime=r["datetime"],
                env_id=env_id,
                episode=f"ep{global_ep[key]}",
                pair=r["pair"],
                side=r["side"],
                leverage=float(r["leverage"]),
                type=etype,
                collateral=float(r["collateral"]),
                fees_usdc=float(r["fees_usdc"]),
                pnl_usdc=_f(r["pnl_usdc"]),
                pnl_pct=_f(r["pnl_pct"]),
                is_exit=etype in EXIT_TYPES,
            )
        )
    return trades


def _exits(trades: list[Trade]) -> list[Trade]:
    return [t for t in trades if t.is_exit and t.pnl_usdc is not None]


def _episode_order(exits: list[Trade]) -> list[str]:
    eps = {t.episode for t in exits}
    return sorted(eps, key=lambda s: int(s[2:]) if s.startswith("ep") else s)


def _monotonic_datetimes(raw_iso: list[str]) -> list[datetime]:
    """Ledger clocks rewind on episode reset — keep a strictly increasing axis."""
    out: list[datetime] = []
    offset = timedelta(0)
    last: datetime | None = None
    for s in raw_iso:
        raw = _parse_dt(s)
        display = raw + offset
        if last is not None and display <= last:
            offset += (last - display) + timedelta(seconds=1)
            display = raw + offset
        out.append(display)
        last = display
    return out


def equity_curve(
    trades: list[Trade], initial_balance: float,
) -> tuple[list[datetime], list[float]]:
    """Cumulative wealth path across the whole test."""
    xs, ys, _, _ = equity_and_drawdown(trades, initial_balance)
    return xs, ys


def equity_and_drawdown(
    trades: list[Trade], initial_balance: float,
) -> tuple[list[datetime], list[float], list[float], list[float]]:
    """Cumulative equity + per-episode underwater drawdown.

    Equity is one continuous bankroll. Drawdown is measured on a fresh
    ``initial_balance`` each episode (env auto-reset semantics) so a long
    cumulative decline does not make DD a rescaled copy of equity.

    X values are ledger datetimes, shifted so episode resets stay monotonic.
    """
    bal = float(initial_balance)
    event_iso: list[str] = []
    ys: list[float] = []
    peaks: list[float] = []
    dds: list[float] = []
    eq = bal          # cumulative wealth (chart)
    eq_ep = bal       # episode-local equity (for DD)
    ep_peak = bal
    prev_ep: str | None = None
    # Seed at t0 with starting capital (use first trade clock if present).
    if trades:
        event_iso.append(trades[0].datetime)
        ys.append(bal)
        peaks.append(bal)
        dds.append(0.0)
    for t in trades:
        if prev_ep is not None and t.episode != prev_ep:
            eq_ep = bal
            ep_peak = bal
        prev_ep = t.episode
        if t.type == OPEN_TYPE:
            delta = -t.fees_usdc
        elif t.is_exit and t.pnl_usdc is not None:
            delta = t.pnl_usdc
        else:
            continue
        eq += delta
        eq_ep += delta
        ep_peak = max(ep_peak, eq_ep)
        event_iso.append(t.datetime)
        ys.append(eq)
        peaks.append(ep_peak)
        dds.append(
            (eq_ep - ep_peak) / ep_peak if ep_peak > 1e-12 else 0.0
        )
    xs = _monotonic_datetimes(event_iso) if event_iso else []
    return xs, ys, peaks, dds


def drawdown(equity: list[float]) -> list[float]:
    """Underwater series: (equity - peak) / peak, ≤ 0."""
    out: list[float] = []
    peak = equity[0] if equity else 0.0
    for e in equity:
        peak = max(peak, e)
        out.append((e - peak) / peak if peak > 1e-12 else 0.0)
    return out


@dataclass
class GroupStats:
    group: str
    num_trades: int
    win_pct: float
    avg_win: float
    avg_loss: float
    return_pct: float
    net_profit: float
    sharpe: float
    max_dd: float
    risk_reward: float
    sortino: float
    calmar: float
    profit_factor: float
    martin: float


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _fmt_num(x: float, digits: int = 2) -> str:
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if math.isnan(x):
        return "nan"
    return f"{x:.{digits}f}"


def _metrics_from_exits(
    exits: list[Trade], initial_balance: float, group: str,
) -> GroupStats:
    """Trade-level performance stats for one breakdown group."""
    pnls = [float(t.pnl_usdc) for t in exits if t.pnl_usdc is not None]
    pcts = [
        float(t.pnl_pct) / 100.0
        for t in exits
        if t.pnl_pct is not None
    ]
    n = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    net = sum(pnls)
    avg_win = _mean(wins)
    avg_loss = _mean(losses)  # ≤ 0
    win_pct = 100.0 * len(wins) / n if n else 0.0
    return_pct = 100.0 * net / initial_balance if initial_balance > 1e-12 else 0.0

    if losses and abs(avg_loss) > 1e-12:
        risk_reward = avg_win / abs(avg_loss)
    elif wins and not losses:
        risk_reward = math.inf
    else:
        risk_reward = 0.0

    # Equity path over this group's exits (chronological order preserved).
    eq = [float(initial_balance)]
    for p in pnls:
        eq.append(eq[-1] + p)
    dd = drawdown(eq)
    max_dd = abs(min(dd)) if dd else 0.0  # fraction of peak

    if pcts:
        rets = pcts
    elif initial_balance > 1e-12:
        rets = [p / initial_balance for p in pnls]
    else:
        rets = []
    mean_r = _mean(rets)
    std_r = _std(rets)
    sharpe = (mean_r / std_r) if std_r > 1e-12 else 0.0
    down_var = (
        sum(min(r, 0.0) ** 2 for r in rets) / len(rets) if rets else 0.0
    )
    down_std = math.sqrt(down_var)
    sortino = (mean_r / down_std) if down_std > 1e-12 else 0.0
    calmar = (return_pct / 100.0 / max_dd) if max_dd > 1e-12 else 0.0

    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    if gross_loss > 1e-12:
        profit_factor = gross_win / gross_loss
    elif gross_win > 0:
        profit_factor = math.inf
    else:
        profit_factor = 0.0

    # Martin ratio = return% / Ulcer Index; UI = sqrt(mean(dd%^2)).
    dd_pct = [100.0 * d for d in dd]
    ulcer = (
        math.sqrt(sum(d * d for d in dd_pct) / len(dd_pct)) if dd_pct else 0.0
    )
    martin = (return_pct / ulcer) if ulcer > 1e-12 else 0.0

    return GroupStats(
        group=group,
        num_trades=n,
        win_pct=win_pct,
        avg_win=avg_win,
        avg_loss=avg_loss,
        return_pct=return_pct,
        net_profit=net,
        sharpe=sharpe,
        max_dd=100.0 * max_dd,
        risk_reward=risk_reward,
        sortino=sortino,
        calmar=calmar,
        profit_factor=profit_factor,
        martin=martin,
    )


def _group_stats(
    exits: list[Trade],
    key_fn,
    initial_balance: float,
    *,
    order: list[str] | None = None,
) -> list[GroupStats]:
    buckets: dict[str, list[Trade]] = defaultdict(list)
    for t in exits:
        buckets[str(key_fn(t))].append(t)
    if order is not None:
        keys = [k for k in order if k in buckets] + sorted(
            k for k in buckets if k not in order
        )
    else:
        keys = sorted(buckets)
    return [
        _metrics_from_exits(buckets[k], initial_balance, k) for k in keys
    ]


_HEADERS = (
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
)


def _row_cells(r: GroupStats) -> tuple:
    return (
        r.group,
        r.num_trades,
        _fmt_num(r.win_pct, 1),
        _fmt_num(r.avg_win),
        _fmt_num(r.avg_loss),
        _fmt_num(r.return_pct),
        _fmt_num(r.net_profit),
        _fmt_num(r.sharpe),
        _fmt_num(r.max_dd),
        _fmt_num(r.risk_reward),
        _fmt_num(r.sortino),
        _fmt_num(r.calmar),
        _fmt_num(r.profit_factor),
        _fmt_num(r.martin),
    )


def _fmt_table(
    title: str,
    rows: list[GroupStats],
    exits: list[Trade],
    initial_balance: float,
) -> str:
    body = [_row_cells(r) for r in rows]
    if rows:
        body.append(
            _row_cells(_metrics_from_exits(exits, initial_balance, "TOTAL"))
        )
    table = tabulate(
        body,
        headers=_HEADERS,
        tablefmt="simple",
        stralign="left",
        numalign="right",
    )
    if not rows:
        table = f"{table}\n(no closed trades)"
    return f"=== {title} ===\n{table}"


def write_breakdown(
    trades: list[Trade],
    path: str | Path,
    *,
    initial_balance: float = 10_000.0,
) -> str:
    exits = _exits(trades)
    sections = [
        _fmt_table(
            "Per symbol",
            _group_stats(exits, lambda t: t.pair, initial_balance),
            exits,
            initial_balance,
        ),
        _fmt_table(
            "Per episode",
            _group_stats(
                exits, lambda t: t.episode, initial_balance,
                order=_episode_order(exits),
            ),
            exits,
            initial_balance,
        ),
        _fmt_table(
            "Per leverage range",
            _group_stats(
                exits,
                lambda t: _bin_label(t.leverage, LEV_BINS),
                initial_balance,
                order=[b[2] for b in LEV_BINS],
            ),
            exits,
            initial_balance,
        ),
        _fmt_table(
            "Per collateral range",
            _group_stats(
                exits,
                lambda t: _bin_label(t.collateral, COLL_BINS),
                initial_balance,
                order=[b[2] for b in COLL_BINS],
            ),
            exits,
            initial_balance,
        ),
        _fmt_table(
            "Direction (long / short)",
            _group_stats(
                exits, lambda t: t.side, initial_balance,
                order=["long", "short"],
            ),
            exits,
            initial_balance,
        ),
        _fmt_table(
            "Exit type",
            _group_stats(
                exits, lambda t: t.type, initial_balance,
                order=list(EXIT_TYPES),
            ),
            exits,
            initial_balance,
        ),
    ]
    text = "\n\n".join(sections) + "\n"
    Path(path).write_text(text)
    return text


def _hex_alpha(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _apply_theme(fig: go.Figure, th: Theme, title: str) -> None:
    fig.update_layout(
        title=dict(
            text=f"{th.title}  ·  {title}",
            font=dict(size=18, color=th.font, family=th.font_family),
            x=0.02,
            xanchor="left",
        ),
        paper_bgcolor=th.paper,
        plot_bgcolor=th.bg,
        font=dict(color=th.font, family=th.font_family, size=11),
        showlegend=False,
        margin=dict(l=56, r=28, t=64, b=40),
        height=900,
        width=1280,
    )
    for ann in fig.layout.annotations or []:
        ann.font = dict(size=12, color=th.muted, family=th.font_family)
        ann.xanchor = "left"
    axis = dict(
        showgrid=True,
        gridcolor=th.grid,
        zeroline=False,
        showline=True,
        linecolor=th.grid,
        tickfont=dict(color=th.muted, family=th.font_family),
        title_font=dict(color=th.muted, family=th.font_family),
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)


def figure_performance(
    trades: list[Trade],
    initial_balance: float,
    theme_name: str = "retrowave",
) -> go.Figure:
    th = _theme(theme_name)
    exits = _exits(trades)
    xs, ys, peaks, dd = equity_and_drawdown(trades, initial_balance)
    ret_x = _monotonic_datetimes([t.datetime for t in exits])
    ret_y = [t.pnl_pct if t.pnl_pct is not None else 0.0 for t in exits]
    ret_colors = [th.up if y >= 0 else th.down for y in ret_y]
    hist_y = list(ret_y)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Equity curve",
            "Returns over time",
            "Drawdown (underwater)",
            "Returns distribution",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    # Cumulative equity only (ep-local peak is for DD, not overlaid here —
    # different scales: wealth vs fresh-per-ep capital).
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=th.accent, width=2),
            name="equity",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ret_x, y=ret_y, mode="markers",
            marker=dict(color=ret_colors, size=7, opacity=0.85),
            name="returns",
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=xs, y=[100.0 * d for d in dd], mode="lines",
            line=dict(color=th.down, width=2),
            fill="tozeroy",
            fillcolor=_hex_alpha(th.down, 0.35),
            name="drawdown",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=hist_y,
            nbinsx=min(40, max(8, len(hist_y) // 2 or 8)),
            marker_color=th.volume,
            opacity=0.9,
            name="returns_hist",
        ),
        row=2, col=2,
    )
    _apply_theme(fig, th, "performance")
    fig.update_xaxes(title_text="datetime", row=1, col=1)
    fig.update_yaxes(title_text="USDC", row=1, col=1)
    fig.update_xaxes(title_text="datetime", row=1, col=2)
    fig.update_yaxes(title_text="pnl %", row=1, col=2)
    fig.update_xaxes(title_text="datetime", row=2, col=1)
    fig.update_yaxes(title_text="dd %", row=2, col=1)
    fig.update_xaxes(title_text="pnl %", row=2, col=2)
    fig.update_yaxes(title_text="count", row=2, col=2)
    return fig


def _counts(keys: list[str]) -> tuple[list[str], list[int]]:
    c: dict[str, int] = defaultdict(int)
    for k in keys:
        c[k] += 1
    labels = sorted(c)
    return labels, [c[k] for k in labels]


def figure_distributions(
    trades: list[Trade],
    theme_name: str = "retrowave",
) -> go.Figure:
    th = _theme(theme_name)
    exits = _exits(trades)
    # Prefer exit trades for distributions; fall back to all non-flat rows.
    sample = exits or [t for t in trades if t.side in ("long", "short")]

    lev_labels = [b[2] for b in LEV_BINS]
    lev_counts = [
        sum(1 for t in sample if _bin_label(t.leverage, LEV_BINS) == lab)
        for lab in lev_labels
    ]
    coll_labels = [b[2] for b in COLL_BINS]
    coll_counts = [
        sum(1 for t in sample if _bin_label(t.collateral, COLL_BINS) == lab)
        for lab in coll_labels
    ]
    dir_labels, dir_counts = _counts([t.side for t in sample])
    exit_labels, exit_counts = _counts([t.type for t in exits]) if exits else ([], [])

    palette = [th.up, th.accent, th.funding, th.down, th.volume, th.muted]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Leverage distribution",
            "Collateral distribution",
            "Direction distribution",
            "Exit type distribution",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    fig.add_trace(
        go.Bar(
            x=lev_labels, y=lev_counts,
            marker_color=[palette[i % len(palette)] for i in range(len(lev_labels))],
            name="leverage",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=coll_labels, y=coll_counts,
            marker_color=[palette[i % len(palette)] for i in range(len(coll_labels))],
            name="collateral",
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(
            x=dir_labels, y=dir_counts,
            marker_color=[th.up if lab == "long" else th.down for lab in dir_labels],
            name="direction",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=exit_labels, y=exit_counts,
            marker_color=[palette[i % len(palette)] for i in range(len(exit_labels))],
            name="exit_type",
        ),
        row=2, col=2,
    )
    _apply_theme(fig, th, "distributions")
    for r, c in ((1, 1), (1, 2), (2, 1), (2, 2)):
        fig.update_yaxes(title_text="count", row=r, col=c)
    return fig


def write_report(
    ledger_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    initial_balance: float = 10_000.0,
    theme_name: str = "retrowave",
) -> dict[str, Path]:
    """Build breakdown.txt + performance/distributions PNGs next to the ledger."""
    ledger_path = Path(ledger_path)
    out = Path(out_dir) if out_dir is not None else ledger_path.parent
    out.mkdir(parents=True, exist_ok=True)

    trades = load_trades(ledger_path)
    breakdown_path = out / "breakdown.txt"
    write_breakdown(trades, breakdown_path, initial_balance=initial_balance)

    perf_path = out / "performance.png"
    dist_path = out / "distributions.png"
    figure_performance(trades, initial_balance, theme_name).write_image(
        str(perf_path), width=1280, height=900, scale=2,
    )
    figure_distributions(trades, theme_name).write_image(
        str(dist_path), width=1280, height=900, scale=2,
    )
    return {
        "breakdown": breakdown_path,
        "performance": perf_path,
        "distributions": dist_path,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Report from ledger.csv")
    p.add_argument("ledger", help="path to ledger.csv")
    p.add_argument("--out-dir", default=None, help="defaults to ledger directory")
    p.add_argument("--initial-balance", type=float, default=10_000.0)
    p.add_argument("--theme", default="retrowave", help="utils.viz_data theme name")
    args = p.parse_args(argv)
    paths = write_report(
        args.ledger,
        args.out_dir,
        initial_balance=args.initial_balance,
        theme_name=args.theme,
    )
    for k, v in paths.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
