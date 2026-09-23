"""Trade history / ledger CSV writer for the trading env."""

from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta, timezone

LEDGER_HEADER = [
    "datetime",
    "env_id",
    "pair",
    "side",
    "leverage",
    "type",
    "open_price",
    "exit_price",
    "collateral",
    "size_usdc",
    "size_base",
    "symbol",
    "fees_usdc",
    "pnl_usdc",
    "pnl_pct",
]


class TradeLedger:
    """Append-only CSV of per-event fills (open/close/tp/sl/liquidation)."""

    def __init__(
        self,
        path: str,
        symbols: list[str],
        bar_minutes: int,
        clock_start: datetime | None = None,
        flush_every: int = 64,
    ):
        self.path = path
        self.symbols = list(symbols)
        self.bar_minutes = int(bar_minutes)
        self.clock_start = clock_start or datetime(2024, 1, 1, tzinfo=timezone.utc)
        self.flush_every = flush_every
        self._buf: list[dict] = []
        self._n_written = 0
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=LEDGER_HEADER)
        self._writer.writeheader()
        self._file.flush()

    def bar_datetime(self, bar_t: int) -> str:
        dt = self.clock_start + timedelta(minutes=bar_t * self.bar_minutes)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    def record(
        self,
        *,
        bar_t: int,
        env_id: int,
        asset_i: int,
        side: float,
        leverage: float,
        event_type: str,
        open_price: float,
        exit_price: float | None,
        collateral: float,
        fees_usdc: float,
        pnl_usdc: float | None,
    ) -> None:
        symbol = self.symbols[asset_i]
        # pair label e.g. BTCUSDT; symbol is the base asset tag used in size_base
        base = symbol.replace("USDT", "").replace("USD", "") or symbol
        size_usdc = abs(collateral * leverage)
        px = open_price if open_price > 1e-12 else 1e-12
        size_base = size_usdc / px
        side_s = "long" if side > 0.5 else ("short" if side < -0.5 else "flat")
        if pnl_usdc is None or collateral < 1e-12:
            pnl_pct = ""
            pnl_s = ""
        else:
            pnl_s = f"{pnl_usdc:.8f}"
            pnl_pct = f"{(pnl_usdc / collateral) * 100.0:.6f}"
        row = {
            "datetime": self.bar_datetime(int(bar_t)),
            "env_id": env_id,
            "pair": symbol,
            "side": side_s,
            "leverage": f"{leverage:.6f}",
            "type": event_type,
            "open_price": f"{open_price:.8f}",
            "exit_price": "" if exit_price is None else f"{exit_price:.8f}",
            "collateral": f"{collateral:.8f}",
            "size_usdc": f"{size_usdc:.8f}",
            "size_base": f"{size_base:.8f}",
            "symbol": base,
            "fees_usdc": f"{fees_usdc:.8f}",
            "pnl_usdc": pnl_s,
            "pnl_pct": pnl_pct,
        }
        self._buf.append(row)
        if len(self._buf) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        self._writer.writerows(self._buf)
        self._n_written += len(self._buf)
        self._buf.clear()
        self._file.flush()

    def close(self) -> None:
        self.flush()
        self._file.close()

    @property
    def n_rows(self) -> int:
        return self._n_written + len(self._buf)
