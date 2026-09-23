"""Themed Plotly + Kaleido PNGs of a synthetic crypto-perp tape.

Each theme draws its own path via `theme_seed(base, theme)` so palettes are
not just recolors of the same tape.

    python -m utils.viz_data
    python -m utils.viz_data --theme ghibli retrowave --n-bars 128
    python -m utils.viz_data --theme all --regime mixed --out-dir outputs/viz
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.data import REGIMES, Market, MarketSpec, generate

THEMES = (
    "ghibli",
    "retrowave",
    "noir",
    "nord",
    "ink",
    "terminal",
    "paper",
    "dawn",
)

# Kaleido resolves this from the host font library (JetBrains Mono is installed).
FONT_FAMILY = "JetBrains Mono"


@dataclass(frozen=True)
class Theme:
    name: str
    title: str
    bg: str
    paper: str
    font: str
    muted: str
    grid: str
    up: str
    down: str
    volume: str
    funding: str
    accent: str
    font_family: str = FONT_FAMILY


def theme(name: str) -> Theme:
    try:
        return _THEMES[name]
    except KeyError as e:
        raise ValueError(f"unknown theme {name!r}; have {list(_THEMES)}") from e


_THEMES: dict[str, Theme] = {
    "ghibli": Theme(
        name="ghibli",
        title="Ghibli Meadow",
        bg="#E8F0E4",
        paper="#F4F7F0",
        font="#2F4A3C",
        muted="#6B8F71",
        grid="#C5D5C0",
        up="#5B8C5A",
        down="#C47B5A",
        volume="#7BA3C9",
        funding="#D4A84B",
        accent="#8FBF8F",
    ),
    "retrowave": Theme(
        name="retrowave",
        title="Retrowave Grid",
        bg="#12061F",
        paper="#1A0B2E",
        font="#F2E6FF",
        muted="#B388FF",
        grid="#3D1F5C",
        up="#00F0FF",
        down="#FF2A6D",
        volume="#7B2CBF",
        funding="#F9C80E",
        accent="#FF6EC7",
    ),
    "noir": Theme(
        name="noir",
        title="Noir Tape",
        bg="#0B0B0B",
        paper="#141414",
        font="#EDEDED",
        muted="#8A8A8A",
        grid="#2A2A2A",
        up="#E8E8E8",
        down="#B01030",
        volume="#4A4A4A",
        funding="#C0C0C0",
        accent="#B01030",
    ),
    "nord": Theme(
        name="nord",
        title="Nord Frost",
        bg="#2E3440",
        paper="#3B4252",
        font="#ECEFF4",
        muted="#D8DEE9",
        grid="#4C566A",
        up="#A3BE8C",
        down="#BF616A",
        volume="#5E81AC",
        funding="#EBCB8B",
        accent="#88C0D0",
    ),
    "ink": Theme(
        name="ink",
        title="Sumi Ink",
        bg="#F3EBD8",
        paper="#FAF6EC",
        font="#1C1C1C",
        muted="#5C5346",
        grid="#D9CFB8",
        up="#1C1C1C",
        down="#9B1D20",
        volume="#7A6A53",
        funding="#2F5D50",
        accent="#9B1D20",
    ),
    "terminal": Theme(
        name="terminal",
        title="Phosphor Terminal",
        bg="#020B05",
        paper="#04140A",
        font="#33FF66",
        muted="#1FA644",
        grid="#0D2818",
        up="#33FF66",
        down="#FF3355",
        volume="#1FA644",
        funding="#A8FF60",
        accent="#33FF66",
    ),
    "paper": Theme(
        name="paper",
        title="Ledger Paper",
        bg="#F2E8D5",
        paper="#FBF4E6",
        font="#3E2F1C",
        muted="#8A7355",
        grid="#E0D0B5",
        up="#2F6B4F",
        down="#A14A3A",
        volume="#6E8B74",
        funding="#8B5E3C",
        accent="#C4A35A",
    ),
    "dawn": Theme(
        name="dawn",
        title="Coastal Dawn",
        bg="#FFF5EB",
        paper="#FFF9F3",
        font="#3D2C2E",
        muted="#9A6B6E",
        grid="#F0D9C8",
        up="#E07A5F",
        down="#3D405B",
        volume="#81B29A",
        funding="#F2CC8F",
        accent="#E07A5F",
    ),
}


# Fixed hues for regime bands so themes stay readable against the tape.
_REGIME_BAND = {
    "calm": "#6B7280",
    "chop": "#94A3B8",
    "bull": "#22C55E",
    "bear": "#EF4444",
    "squeeze": "#EAB308",
    "cascade": "#A855F7",
}


def _row(market: Market, name: str, env: int) -> list[float]:
    return getattr(market, name)[env].tolist()


def _regime_spans(market: Market, env: int) -> list[tuple[str, int, int]]:
    """Inclusive [start, end] bar spans for contiguous regimes."""
    if market.regime is None:
        return []
    ids = market.regime[env].tolist()
    spans: list[tuple[str, int, int]] = []
    start = 0
    cur = ids[0]
    for i in range(1, len(ids)):
        if ids[i] != cur:
            spans.append((REGIMES[int(cur)], start, i - 1))
            start, cur = i, ids[i]
    spans.append((REGIMES[int(cur)], start, len(ids) - 1))
    return spans


def figure(
    market: Market, theme_name: str, env: int = 0, seed: int | None = None,
) -> go.Figure:
    th = theme(theme_name)
    n = int(market.close.shape[0])
    if not 0 <= env < n:
        raise IndexError(f"env {env} out of range for n_envs={n}")

    bars = list(range(int(market.close.shape[1])))
    o = _row(market, "open", env)
    h = _row(market, "high", env)
    low = _row(market, "low", env)
    c = _row(market, "close", env)
    vol = _row(market, "volume", env)
    fund = _row(market, "funding", env)
    vol_colors = [th.up if c[i] >= o[i] else th.down for i in range(len(c))]
    spans = _regime_spans(market, env)
    regime_tag = (
        "flat"
        if not spans
        else ("mixed" if len({s[0] for s in spans}) > 1 else spans[0][0])
    )
    seed_tag = f"  ·  seed {seed}" if seed is not None else ""

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=("OHLC", "Volume", "Funding"),
    )
    for name, start, end in spans:
        fig.add_vrect(
            x0=start - 0.5,
            x1=end + 0.5,
            fillcolor=_hex_alpha(_REGIME_BAND[name], 0.14),
            line_width=0,
            layer="below",
            row=1, col=1,
        )
    fig.add_trace(
        go.Candlestick(
            x=bars, open=o, high=h, low=low, close=c,
            increasing_line_color=th.up, decreasing_line_color=th.down,
            increasing_fillcolor=th.up, decreasing_fillcolor=th.down,
            name="OHLC",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=bars, y=vol, marker_color=vol_colors, name="Volume", opacity=0.85),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=bars, y=fund, mode="lines",
            line=dict(color=th.funding, width=2),
            fill="tozeroy",
            fillcolor=_hex_alpha(th.funding, 0.18),
            name="Funding",
        ),
        row=3, col=1,
    )

    fig.update_layout(
        title=dict(
            text=(
                f"{th.title}  ·  crypto perp  ·  {regime_tag}"
                f"  ·  env {env}  ·  {len(bars)} bars{seed_tag}"
            ),
            font=dict(size=20, color=th.font, family=th.font_family),
            x=0.02, xanchor="left",
        ),
        paper_bgcolor=th.paper,
        plot_bgcolor=th.bg,
        font=dict(color=th.font, family=th.font_family, size=12),
        showlegend=False,
        margin=dict(l=56, r=24, t=72, b=40),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    for i, ann in enumerate(fig.layout.annotations):
        ann.font = dict(size=13, color=th.muted, family=th.font_family)
        ann.x = 0.01
        ann.xanchor = "left"

    axis = dict(
        showgrid=True, gridcolor=th.grid, zeroline=False,
        showline=True, linecolor=th.grid,
        tickfont=dict(color=th.muted, family=th.font_family),
        title_font=dict(color=th.muted, family=th.font_family),
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    fig.update_xaxes(title_text="bar", row=3, col=1)
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="vol", row=2, col=1)
    fig.update_yaxes(title_text="funding", row=3, col=1)
    # Candlestick brings its own rangeslider on x; keep only the shared bottom axis.
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def write_png(
    market: Market,
    path: str | Path,
    theme_name: str,
    env: int = 0,
    width: int = 1280,
    height: int = 900,
    scale: int = 2,
    seed: int | None = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure(market, theme_name, env=env, seed=seed).write_image(
        str(out), width=width, height=height, scale=scale,
    )
    return out


def theme_seed(base: int, theme_name: str) -> int:
    """Stable per-theme seed so each palette draws its own crypto path."""
    theme(theme_name)  # validate
    # Large odd stride keeps neighboring themes far apart in RNG space.
    return int(base) + 10007 * (THEMES.index(theme_name) + 1)


def _hex_alpha(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--theme", nargs="+", default=["all"], help=f"one of {THEMES} or all")
    p.add_argument("--out-dir", default="outputs/viz")
    p.add_argument("--env", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--n-bars", type=int, default=128)
    p.add_argument("--seed", type=int, default=0, help="base seed; each theme offsets from this")
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--sigma", type=float, default=0.018)
    p.add_argument("--alpha-phi", type=float, default=0.85)
    p.add_argument("--alpha-sigma", type=float, default=0.004)
    p.add_argument(
        "--regime",
        default="mixed",
        help="None/flat, mixed, or one of: " + ", ".join(REGIMES),
    )
    p.add_argument("--regime-persist", type=float, default=0.97)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--scale", type=int, default=2)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> list[Path]:
    args = _parse_args(argv)
    names = list(THEMES) if "all" in args.theme else args.theme
    for name in names:
        theme(name)

    regime = None if args.regime in ("None", "none", "flat") else args.regime
    out_dir = Path(args.out_dir)
    written: list[Path] = []
    for name in names:
        seed = theme_seed(args.seed, name)
        market = generate(MarketSpec(
            n_envs=args.n_envs,
            n_bars=args.n_bars,
            seed=seed,
            mu=args.mu,
            sigma=args.sigma,
            alpha_phi=args.alpha_phi,
            alpha_sigma=args.alpha_sigma,
            regime=regime,
            regime_persist=args.regime_persist,
        ))
        mx.eval(market.close)
        path = out_dir / f"market_{name}.png"
        write_png(
            market, path, name, env=args.env, seed=seed,
            width=args.width, height=args.height, scale=args.scale,
        )
        written.append(path)
        print(f"{path}  (seed={seed})")
    return written


if __name__ == "__main__":
    main()
