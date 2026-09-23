"""Test-run reporting from trade ledger.csv.

Writes into the run directory:
  - breakdown.txt  — tables by symbol, episode, leverage, collateral,
                     direction, exit type
  - performance.png — aggregate 2×2 (mean equity/DD across episodes)
  - performance_epN.png — optional (``report_per_episode`` / ``--per-episode``)
  - distributions.png — leverage / collateral / direction / exit type

Themes reuse ``utils.viz_data`` (default: random theme each report).
"""

from __future__ import annotations

import argparse
import csv
import math
import random
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


def resolve_theme(name: str | None = None) -> str:
    """Pick a viz theme; ``None`` / ``\"random\"`` → uniform draw from THEMES."""
    from utils.viz_data import THEMES, theme

    if name is None or name == "random":
        chosen = random.choice(THEMES)
        theme(chosen)  # validate
        return chosen
    theme(name)  # validate
    return name

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


def _trades_by_episode(trades: list[Trade]) -> list[tuple[str, list[Trade]]]:
    """Group trades into contiguous episode runs, ordered ep1, ep2, …"""
    order = _episode_order(trades)
    buckets: dict[str, list[Trade]] = {ep: [] for ep in order}
    for t in trades:
        buckets.setdefault(t.episode, []).append(t)
    return [(ep, buckets[ep]) for ep in order if buckets.get(ep)]


def equity_curve(
    trades: list[Trade], initial_balance: float,
) -> tuple[list[datetime | None], list[float | None]]:
    """Equity path for the given trades (one episode = continuous datetime axis)."""
    xs, ys, _, _ = equity_and_drawdown(trades, initial_balance)
    return xs, ys


def equity_and_drawdown(
    trades: list[Trade], initial_balance: float,
) -> tuple[
    list[datetime | None],
    list[float | None],
    list[float | None],
    list[float | None],
]:
    """Equity + underwater DD for trades in ledger order.

    Prefer a single episode's trades. Multiple episodes get capital resets and
    line breaks (``None``) between segments — use ``mean_equity_and_drawdown``
    for the aggregate overview figure instead.

    X values are ledger datetimes, shifted so episode resets stay monotonic.
    """
    bal = float(initial_balance)
    segments: list[
        tuple[list[str], list[float], list[float], list[float]]
    ] = []
    isos: list[str] = []
    ys: list[float] = []
    peaks: list[float] = []
    dds: list[float] = []
    eq = bal
    peak = bal
    prev_ep: str | None = None

    def _flush() -> None:
        nonlocal isos, ys, peaks, dds, eq, peak
        if isos:
            segments.append((isos, ys, peaks, dds))
        isos, ys, peaks, dds = [], [], [], []
        eq, peak = bal, bal

    for t in trades:
        if prev_ep is None or t.episode != prev_ep:
            if prev_ep is not None:
                _flush()
            isos = [t.datetime]
            ys = [bal]
            peaks = [bal]
            dds = [0.0]
            eq, peak = bal, bal
        prev_ep = t.episode

        if t.type == OPEN_TYPE:
            delta = -t.fees_usdc
        elif t.is_exit and t.pnl_usdc is not None:
            delta = t.pnl_usdc
        else:
            continue
        eq += delta
        peak = max(peak, eq)
        isos.append(t.datetime)
        ys.append(eq)
        peaks.append(peak)
        dds.append((eq - peak) / peak if peak > 1e-12 else 0.0)
    _flush()

    if not segments:
        return [], [], [], []

    flat_iso: list[str] = []
    bounds: list[tuple[int, int]] = []
    for isos_s, _, _, _ in segments:
        start = len(flat_iso)
        flat_iso.extend(isos_s)
        bounds.append((start, len(flat_iso)))
    flat_xs = _monotonic_datetimes(flat_iso)

    xs: list[datetime | None] = []
    out_y: list[float | None] = []
    out_p: list[float | None] = []
    out_d: list[float | None] = []
    for i, ((_, seg_y, seg_p, seg_d), (a, b)) in enumerate(
        zip(segments, bounds)
    ):
        if i > 0:
            xs.append(None)
            out_y.append(None)
            out_p.append(None)
            out_d.append(None)
        xs.extend(flat_xs[a:b])
        out_y.extend(seg_y)
        out_p.extend(seg_p)
        out_d.extend(seg_d)
    return xs, out_y, out_p, out_d


def mean_equity_and_drawdown(
    trades: list[Trade], initial_balance: float,
) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    """Mean equity / DD across episodes, aligned by in-episode event index.

    Returns ``(xs, mean_eq, min_eq, max_eq, mean_dd)``.
    """
    curves_y: list[list[float]] = []
    curves_d: list[list[float]] = []
    for _, ep_trades in _trades_by_episode(trades):
        _, ys, _, dds = equity_and_drawdown(ep_trades, initial_balance)
        curves_y.append([float(y) for y in ys if y is not None])
        curves_d.append([float(d) for d in dds if d is not None])
    if not curves_y:
        return [], [], [], [], []
    n = max(len(y) for y in curves_y)
    xs = list(range(n))
    mean_y: list[float] = []
    min_y: list[float] = []
    max_y: list[float] = []
    mean_d: list[float] = []
    for i in range(n):
        yi = [y[i] for y in curves_y if i < len(y)]
        di = [d[i] for d in curves_d if i < len(d)]
        mean_y.append(sum(yi) / len(yi))
        min_y.append(min(yi))
        max_y.append(max(yi))
        mean_d.append(sum(di) / len(di) if di else 0.0)
    return xs, mean_y, min_y, max_y, mean_d


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
    theme_name: str | None = None,
    *,
    title: str = "performance",
) -> go.Figure:
    """2×2 for one episode (or any single contiguous trade stream)."""
    th = _theme(resolve_theme(theme_name))
    exits = _exits(trades)
    xs, ys, _, dd = equity_and_drawdown(trades, initial_balance)
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
            x=xs,
            y=[None if d is None else 100.0 * d for d in dd],
            mode="lines",
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
    _apply_theme(fig, th, title)
    fig.update_xaxes(title_text="datetime", row=1, col=1)
    fig.update_yaxes(title_text="USDC", row=1, col=1)
    fig.update_xaxes(title_text="datetime", row=1, col=2)
    fig.update_yaxes(title_text="pnl %", row=1, col=2)
    fig.update_xaxes(title_text="datetime", row=2, col=1)
    fig.update_yaxes(title_text="dd %", row=2, col=1)
    fig.update_xaxes(title_text="pnl %", row=2, col=2)
    fig.update_yaxes(title_text="count", row=2, col=2)
    return fig


def figure_performance_aggregate(
    trades: list[Trade],
    initial_balance: float,
    theme_name: str | None = None,
) -> go.Figure:
    """2×2 across all episodes: mean equity/DD vs event, pooled returns."""
    th = _theme(resolve_theme(theme_name))
    exits = _exits(trades)
    xs, mean_y, min_y, max_y, mean_d = mean_equity_and_drawdown(
        trades, initial_balance,
    )
    n_ep = len(_trades_by_episode(trades))
    # Overlay exits by in-episode index (chained datetimes look periodic).
    ep_exit_i: dict[str, int] = defaultdict(int)
    ret_x: list[int] = []
    ret_y: list[float] = []
    ret_colors: list[str] = []
    for t in exits:
        ret_x.append(ep_exit_i[t.episode])
        ep_exit_i[t.episode] += 1
        y = t.pnl_pct if t.pnl_pct is not None else 0.0
        ret_y.append(y)
        ret_colors.append(th.up if y >= 0 else th.down)
    hist_y = list(ret_y)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"Mean equity ({n_ep} eps)",
            f"Returns by exit # ({n_ep} eps)",
            f"Mean drawdown ({n_ep} eps)",
            "Returns distribution (all)",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    if xs:
        fig.add_trace(
            go.Scatter(
                x=xs, y=max_y, mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                name="max",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xs, y=min_y, mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=_hex_alpha(th.accent, 0.18),
                hoverinfo="skip",
                showlegend=False,
                name="min",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xs, y=mean_y, mode="lines",
                line=dict(color=th.accent, width=2),
                name="mean equity",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xs, y=[100.0 * d for d in mean_d], mode="lines",
                line=dict(color=th.down, width=2),
                fill="tozeroy",
                fillcolor=_hex_alpha(th.down, 0.35),
                name="mean dd",
            ),
            row=2, col=1,
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
        go.Histogram(
            x=hist_y,
            nbinsx=min(40, max(8, len(hist_y) // 2 or 8)),
            marker_color=th.volume,
            opacity=0.9,
            name="returns_hist",
        ),
        row=2, col=2,
    )
    _apply_theme(fig, th, "performance · aggregate")
    fig.update_xaxes(title_text="event", row=1, col=1)
    fig.update_yaxes(title_text="USDC", row=1, col=1)
    fig.update_xaxes(title_text="exit #", row=1, col=2)
    fig.update_yaxes(title_text="pnl %", row=1, col=2)
    fig.update_xaxes(title_text="event", row=2, col=1)
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
    theme_name: str | None = None,
) -> go.Figure:
    th = _theme(resolve_theme(theme_name))
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
    theme_name: str | None = None,
    per_episode: bool = False,
) -> dict[str, Path | str]:
    """Build breakdown.txt + aggregate performance (+ optional per-ep PNGs)."""
    ledger_path = Path(ledger_path)
    out = Path(out_dir) if out_dir is not None else ledger_path.parent
    out.mkdir(parents=True, exist_ok=True)

    theme_name = resolve_theme(theme_name)
    trades = load_trades(ledger_path)
    breakdown_path = out / "breakdown.txt"
    write_breakdown(trades, breakdown_path, initial_balance=initial_balance)

    perf_path = out / "performance.png"
    dist_path = out / "distributions.png"
    figure_performance_aggregate(
        trades, initial_balance, theme_name,
    ).write_image(str(perf_path), width=1280, height=900, scale=2)
    figure_distributions(trades, theme_name).write_image(
        str(dist_path), width=1280, height=900, scale=2,
    )

    paths: dict[str, Path | str] = {
        "theme": theme_name,
        "breakdown": breakdown_path,
        "performance": perf_path,
        "distributions": dist_path,
    }
    if per_episode:
        for ep, ep_trades in _trades_by_episode(trades):
            ep_path = out / f"performance_{ep}.png"
            figure_performance(
                ep_trades,
                initial_balance,
                theme_name,
                title=f"performance · {ep}",
            ).write_image(str(ep_path), width=1280, height=900, scale=2)
            paths[f"performance_{ep}"] = ep_path
    return paths


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Report from ledger.csv")
    p.add_argument("ledger", help="path to ledger.csv")
    p.add_argument("--out-dir", default=None, help="defaults to ledger directory")
    p.add_argument("--initial-balance", type=float, default=10_000.0)
    p.add_argument(
        "--theme",
        default=None,
        help="utils.viz_data theme name (default: random)",
    )
    p.add_argument(
        "--per-episode",
        action="store_true",
        help="Also write performance_epN.png (default: aggregate only)",
    )
    args = p.parse_args(argv)
    paths = write_report(
        args.ledger,
        args.out_dir,
        initial_balance=args.initial_balance,
        theme_name=args.theme,
        per_episode=args.per_episode,
    )
    for k, v in paths.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
