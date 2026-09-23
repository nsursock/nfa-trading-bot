"""Swap, memory pressure, SMC temperature, and thermal state during a window.

Sampled from a side thread so the training loop is not blocked on each step.
`smctemp` reads Apple SMC temperatures (CPU and GPU, °C). `osx-cpu-temp`
returns 0°C on Apple Silicon, so it is not used. Die temperature from
`powermetrics` needs root and is not required.

Swap used comes from `sysctl vm.swapusage`. Memory pressure is the kernel
level (`Normal`, `Warn`, `Urgent`, `Critical`) plus the system-wide free
percentage from `memory_pressure`. Thermal state is `NSProcessInfo`
(`Nominal`, `Fair`, `Serious`, `Critical`). CSV columns keep the peak swap,
peak temperatures, the worst pressure and thermal state, and the lowest free
percentage seen in the measured window.
"""

import ctypes
import re
import shutil
import subprocess
import threading

_SWAP = re.compile(r"used\s*=\s*([0-9.]+)\s*([KMGT]?)")
_FREE = re.compile(r"System-wide memory free percentage:\s*(\d+)%")
_PRESSURE_LINE = re.compile(r"kern\.memorystatus_vm_pressure_level:\s*(\d+)")
_TEMP = re.compile(r"(-?\d+(?:\.\d+)?)")
_MB = {"": 1 / (1024 * 1024), "K": 1 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024.0}

PRESSURE = {1: "Normal", 2: "Warn", 4: "Urgent", 8: "Critical"}
PRESSURE_RANK = {name: i for i, name in enumerate(
    ("Normal", "Warn", "Urgent", "Critical")
)}
THERMAL = {0: "Nominal", 1: "Fair", 2: "Serious", 3: "Critical"}
THERMAL_RANK = {name: i for i, name in enumerate(
    ("Nominal", "Fair", "Serious", "Critical")
)}
HOST_COLUMNS = (
    "swap_mb", "mem_free_pct", "pressure", "cpu_c", "gpu_c", "thermal",
)


def parse_swap_used_mb(text):
    match = _SWAP.search(text or "")
    if not match:
        return None
    return float(match.group(1)) * _MB[match.group(2)]


def parse_mem_free_pct(text):
    match = _FREE.search(text or "")
    return int(match.group(1)) if match else None


def parse_pressure(text):
    stripped = (text or "").strip()
    match = _PRESSURE_LINE.search(stripped)
    if match:
        level = int(match.group(1))
    elif stripped.isdigit():
        level = int(stripped)
    else:
        return None
    return PRESSURE.get(level, str(level))


def parse_temps(text):
    values = []
    for match in _TEMP.finditer(text or ""):
        value = float(match.group(1))
        if 0 < value <= 150:
            values.append(value)
    return max(values) if values else None


def worst(values, rank):
    known = [value for value in values if value in rank]
    if not known:
        return ""
    return max(known, key=rank.get)


def peak_host(summaries):
    """Worst reading across repeat summaries. Peaks explain a throughput cliff."""
    swaps, frees, cpus, gpus = [], [], [], []
    pressures, thermals = [], []
    for host in summaries:
        if not host:
            continue
        if host.get("swap_mb") not in ("", None):
            swaps.append(float(host["swap_mb"]))
        if host.get("mem_free_pct") not in ("", None):
            frees.append(int(host["mem_free_pct"]))
        if host.get("cpu_c") not in ("", None):
            cpus.append(float(host["cpu_c"]))
        if host.get("gpu_c") not in ("", None):
            gpus.append(float(host["gpu_c"]))
        pressures.append(host.get("pressure"))
        thermals.append(host.get("thermal"))
    return {
        "swap_mb": round(max(swaps), 1) if swaps else "",
        "mem_free_pct": min(frees) if frees else "",
        "pressure": worst(pressures, PRESSURE_RANK),
        "cpu_c": round(max(cpus), 1) if cpus else "",
        "gpu_c": round(max(gpus), 1) if gpus else "",
        "thermal": worst(thermals, THERMAL_RANK),
    }


def host_suffix(result):
    text = format_host((result or {}).get("host"))
    return f"; {text}" if text else ""


def format_host(host):
    if not host:
        return ""
    parts = []
    if host.get("swap_mb") not in ("", None):
        parts.append(f"swap {host['swap_mb']}MB")
    if host.get("mem_free_pct") not in ("", None):
        parts.append(f"free {host['mem_free_pct']}%")
    if host.get("pressure"):
        parts.append(f"pressure {host['pressure']}")
    if host.get("cpu_c") not in ("", None):
        parts.append(f"cpu {host['cpu_c']}C")
    if host.get("gpu_c") not in ("", None):
        parts.append(f"gpu {host['gpu_c']}C")
    if host.get("thermal"):
        parts.append(host["thermal"])
    return " ".join(parts)


def _run(argv, timeout=2):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    return result.stdout or ""


_thermal_lock = threading.Lock()
_objc = None


def _objc_runtime():
    """Direct objc_msgSend. A ctypes function-pointer cast segfaults on arm64."""
    global _objc
    if _objc is None:
        ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/Foundation.framework/Foundation"
        )
        lib = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
        lib.objc_getClass.argtypes = [ctypes.c_char_p]
        lib.objc_getClass.restype = ctypes.c_void_p
        lib.sel_registerName.argtypes = [ctypes.c_char_p]
        lib.sel_registerName.restype = ctypes.c_void_p
        send = lib.objc_msgSend
        send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        _objc = (
            send,
            lib.objc_getClass(b"NSProcessInfo"),
            lib.sel_registerName(b"processInfo"),
            lib.sel_registerName(b"thermalState"),
        )
    return _objc


def _thermal_state():
    send, cls, sel_info, sel_state = _objc_runtime()
    with _thermal_lock:
        send.restype = ctypes.c_void_p
        info = send(cls, sel_info)
        send.restype = ctypes.c_long
        state = send(info, sel_state)
    return THERMAL.get(int(state), str(int(state)))


def _smctemp(flag):
    binary = shutil.which("smctemp")
    if not binary:
        return None
    return parse_temps(_run([binary, flag, "-n", "1"]))


def read_host():
    sample = {
        "swap_mb": None, "mem_free_pct": None, "pressure": None,
        "cpu_c": None, "gpu_c": None, "thermal": None,
    }
    try:
        text = _run(["sysctl", "vm.swapusage", "kern.memorystatus_vm_pressure_level"])
        sample["swap_mb"] = parse_swap_used_mb(text)
        sample["pressure"] = parse_pressure(text)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        sample["mem_free_pct"] = parse_mem_free_pct(_run(["memory_pressure"]))
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        sample["cpu_c"] = _smctemp("-c")
        sample["gpu_c"] = _smctemp("-g")
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        sample["thermal"] = _thermal_state()
    except (OSError, AttributeError):
        pass
    return sample


def summarize_host(samples):
    return peak_host(samples)


class HostSampler:
    """Poll host metrics until stopped. The first and last reads bracket the window."""

    def __init__(self, interval=1.0, read=read_host):
        self.interval = interval
        self._read = read
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self.samples.append(self._read())
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.wait(self.interval):
            self.samples.append(self._read())

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 3)
        self.samples.append(self._read())
        return summarize_host(self.samples), list(self.samples)
