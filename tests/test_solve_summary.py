import statistics

from utils.bench.solve import summarize_runs


def _row(seed, solved, tts_s="", tts_steps="", n_envs=256):
    return {
        "algo": "td3", "env": "Pendulum-v1", "n_envs": n_envs, "seed": seed,
        "solved": solved, "tts_s": tts_s, "tts_steps": tts_steps,
    }


def test_summary_ignores_unsolved_seeds():
    rows = [
        _row(0, "yes", 29.85, 500_000),
        _row(1, "yes", 14.28, 400_000),
        _row(2, "no"),
    ]
    summary = summarize_runs(rows)
    assert len(summary) == 1
    row = summary[0]
    assert row["solved"] == "2/3"
    assert row["tts_s_median"] == round(statistics.median([29.85, 14.28]), 2)
    rates = [500_000 / 29.85, 400_000 / 14.28]
    assert row["steps/s"] == round(statistics.median(rates))
    assert row["tts_s_std"] != ""
    assert row["tts_steps_p25"] <= row["tts_steps_median"] <= row["tts_steps_p75"]


def test_single_solved_seed_has_no_std():
    row = summarize_runs([_row(0, "yes", 10.0, 1000)])[0]
    assert row["solved"] == "1/1"
    assert row["tts_s_median"] == 10.0
    assert row["tts_s_std"] == ""
    assert row["tts_s_p25"] == 10.0 and row["tts_s_p75"] == 10.0
    assert row["steps/s"] == 100


def test_summary_keeps_sweep_order():
    rows = [
        _row(0, "yes", 2.0, 100, n_envs=128),
        _row(0, "no", n_envs=256),
        _row(1, "yes", 4.0, 200, n_envs=128),
    ]
    summary = summarize_runs(rows)
    assert [row["n_envs"] for row in summary] == [128, 256]
    assert summary[1]["solved"] == "0/1"
    assert summary[1]["tts_s_median"] == ""
    assert summary[1]["steps/s"] == ""
