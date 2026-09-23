from pathlib import Path

import pytest

from scripts.data import MarketSpec, generate
from utils.viz_data import THEMES, figure, main, theme, theme_seed, write_png


def test_all_themes_build_figures():
    market = generate(MarketSpec(n_envs=1, n_bars=32, seed=0, alpha_phi=0.5, regime="mixed"))
    for name in THEMES:
        fig = figure(market, name, seed=theme_seed(0, name))
        assert theme(name).title in fig.layout.title.text
        assert "crypto perp" in fig.layout.title.text
        assert f"seed {theme_seed(0, name)}" in fig.layout.title.text
        assert len(fig.data) == 3


def test_each_theme_gets_its_own_path(tmp_path: Path):
    closes = []
    for name in ("ghibli", "retrowave", "noir"):
        seed = theme_seed(0, name)
        market = generate(MarketSpec(n_envs=1, n_bars=48, seed=seed, regime="mixed"))
        closes.append(market.close[0].tolist())
        write_png(
            market, tmp_path / f"{name}.png", name, seed=seed,
            width=640, height=480, scale=1,
        )
    assert closes[0] != closes[1]
    assert closes[1] != closes[2]
    assert closes[0] != closes[2]
    assert theme_seed(0, "ghibli") != theme_seed(0, "retrowave")
    assert theme_seed(3, "dawn") == 3 + 10007 * (THEMES.index("dawn") + 1)


def test_main_writes_distinct_theme_pngs(tmp_path: Path):
    paths = main([
        "--theme", "ghibli", "terminal",
        "--out-dir", str(tmp_path),
        "--n-bars", "32",
        "--seed", "1",
        "--width", "640",
        "--height", "480",
        "--scale", "1",
    ])
    assert len(paths) == 2
    assert paths[0].read_bytes() != paths[1].read_bytes()


def test_write_png(tmp_path: Path):
    market = generate(MarketSpec(n_envs=1, n_bars=24, seed=1, regime="bull"))
    path = write_png(market, tmp_path / "ghibli.png", "ghibli", width=640, height=480, scale=1)
    raw = path.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert path.stat().st_size > 1000


def test_unknown_theme():
    with pytest.raises(ValueError, match="unknown theme"):
        theme("not-a-theme")


def test_all_themes_use_jetbrains_mono():
    from utils.viz_data import FONT_FAMILY

    assert FONT_FAMILY == "JetBrains Mono"
    for name in THEMES:
        assert theme(name).font_family == "JetBrains Mono"
