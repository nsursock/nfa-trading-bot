from utils.bench.host import (
    format_host, parse_mem_free_pct, parse_pressure, parse_swap_used_mb,
    parse_temps, peak_host,
)

SWAP = "vm.swapusage: total = 2048.00M  used = 921.44M  free = 1126.56M  (encrypted)"
SYSCTL = SWAP + "\nkern.memorystatus_vm_pressure_level: 2\n"
PRESSURE = """The system has 17179869184 (1048576 pages with a page size of 16384).
System-wide memory free percentage: 67%
"""


def test_parse_swap_and_pressure():
    assert parse_swap_used_mb(SWAP) == 921.44
    assert parse_swap_used_mb("used = 1.50G") == 1536.0
    assert parse_pressure(SYSCTL) == "Warn"
    assert parse_pressure("4") == "Urgent"
    assert parse_mem_free_pct(PRESSURE) == 67


def test_parse_smctemp_ignores_zero():
    assert parse_temps("41.7\n") == 41.7
    assert parse_temps("0.0°C\n") is None
    assert parse_temps("39.3\n41.7\n") == 41.7


def test_peak_host_keeps_the_worst_reading():
    host = peak_host([
        {"swap_mb": 100.0, "mem_free_pct": 70, "pressure": "Normal",
         "cpu_c": 45.0, "gpu_c": 40.0, "thermal": "Nominal"},
        {"swap_mb": 800.5, "mem_free_pct": 12, "pressure": "Warn",
         "cpu_c": 92.2, "gpu_c": 88.0, "thermal": "Fair"},
    ])
    assert host == {
        "swap_mb": 800.5, "mem_free_pct": 12, "pressure": "Warn",
        "cpu_c": 92.2, "gpu_c": 88.0, "thermal": "Fair",
    }
    assert "swap 800.5MB" in format_host(host)
    assert "Fair" in format_host(host)
