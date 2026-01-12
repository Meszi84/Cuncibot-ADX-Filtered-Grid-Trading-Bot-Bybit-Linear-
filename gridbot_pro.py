#!/usr/bin/env python3
# NOTE: NE tarts éles API kulcsot forráskódban. Rotáld a kulcsokat, és tedd env-be.
import os
import time
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone, date
from typing import Optional, Dict, Any, Tuple, List

import ccxt
import pandas as pd


# =========================
# KONFIG
# =========================

CFG: Dict[str, Any] = {
    "exchange": {
        "testnet": False,
        "recv_window": 20000,
        "defaultType": "linear",
        "leverage": 7,
        "marginMode": "isolated",
    },
    "market": {
        "symbol": "DOGE/USDT:USDT",
        "timeframe": "15m",
        "ohlcv_limit": 300,
        "poll_seconds": 15,
        "min_order_usdt": 5,
        "max_backoff_sec": 60,
    },
    "grid": {
        "lookback_bars": 48,
        "atr_period": 14,
        "atr_buffer": 0.5,
        "max_grids": 20,
        "min_step_pct": 0.001,
        "order_count_cap": 40,
        "post_only": False,
        "refill_ratio": 0.7,
    },
    "regime": {
        "adx_period": 14,
        "adx_on": 25.0,
        "adx_off": 35.0,
    },
    "risk": {
        "equity_usdt_assumed": 80.0,
        "use_balance_fetch": True,
        "max_net_pos_equity_ratio": 0.4,  # (most nem használod aktívan; logika marad)
        "range_break_buffer_pct": 0.005,  # (most nem használod aktívan; logika marad)
        "daily_dd_usdt": 15.0,
        "grid_equity_ratio": 0.60,        # korábban hardcode volt; most CFG-ből
    },
    "adx": {
        "cooldown_sec": 60,
    },
    "logging": {
        "jsonl_path": "grid_bot_log.jsonl",
    },
    "api": {
        # Javaslat: env-ből olvasd. Példa:
        # export BYBIT_API_KEY="..."
        # export BYBIT_API_SECRET="..."
        "apiKey": os.getenv("BYBIT_API_KEY", ""),
        "secret": os.getenv("BYBIT_API_SECRET", ""),
    },
}


# =========================
# SEGÉDEK
# =========================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_today() -> date:
    return utc_now().date()


def safe_float(x, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def log_jsonl(path: str, obj: Dict[str, Any]) -> None:
    payload = dict(obj)
    payload["ts_utc"] = utc_now().isoformat()
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # logging nem dőlhet el a bot miatt
        pass


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


# =========================
# INDICATOROK
# =========================

def calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)

    return tr.rolling(period).mean()


def calculate_adx(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = (low.diff() * -1)

    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0.0)
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0.0)

    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1
    ).max(axis=1)

    atr = tr.rolling(period).mean()

    plus_di = 100 * plus_dm.rolling(period).sum() / atr
    minus_di = 100 * minus_dm.rolling(period).sum() / atr

    denom = (plus_di + minus_di).replace(0, 1)
    dx = ((plus_di - minus_di).abs() / denom) * 100
    adx = dx.rolling(period).mean()
    return adx


# =========================
# EXCHANGE WRAPPER
# =========================

def make_exchange(cfg: Dict[str, Any]) -> ccxt.Exchange:
    if not cfg["api"]["apiKey"] or not cfg["api"]["secret"]:
        raise RuntimeError("Hiányzó API kulcs/secret. Tedd env-be: BYBIT_API_KEY, BYBIT_API_SECRET")

    ex = ccxt.bybit({
        "apiKey": cfg["api"]["apiKey"],
        "secret": cfg["api"]["secret"],
        "enableRateLimit": True,
        "options": {
            "defaultType": cfg["exchange"]["defaultType"],
            "recvWindow": cfg["exchange"]["recv_window"],
        },
    })
    if cfg["exchange"]["testnet"]:
        ex.set_sandbox_mode(True)

    # fontos: precision / market meta
    ex.load_markets()
    return ex


def set_leverage_margin(ex: ccxt.Exchange, symbol: str, lev: int, mode: str) -> None:
    try:
        ex.set_leverage(lev, symbol)
    except Exception:
        pass
    try:
        ex.set_margin_mode(mode, symbol)
    except Exception:
        pass


def fetch_ohlcv_df(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    try:
        bars = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not bars:
            return None
        df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df
    except Exception:
        return None


def get_balance_usdt(ex: ccxt.Exchange) -> float:
    try:
        bal = ex.fetch_balance()
        # bybit: néha más szerkezet, de ez a legegyszerűbb
        return float(bal.get("USDT", {}).get("total", 0.0))
    except Exception:
        return 0.0


def cancel_all_orders(ex: ccxt.Exchange, symbol: str) -> None:
    try:
        ex.cancel_all_orders(symbol, params={"category": "linear"})
    except Exception:
        pass


def place_limit(ex: ccxt.Exchange, symbol: str, side: str, price: float, amount: float, post_only: bool):
    params = {
        "category": "linear",
        "timeInForce": "GTC",
        "reduceOnly": False,
    }
    if post_only:
        params["postOnly"] = True

    return ex.create_order(
        symbol=symbol,
        type="limit",
        side=side,
        amount=float(amount),
        price=float(price),
        params=params,
    )


# =========================
# STATE
# =========================

@dataclass
class GridState:
    active: bool = False
    bottom: Optional[float] = None
    top: Optional[float] = None
    step: Optional[float] = None
    grids: int = 0

    def reset(self) -> None:
        self.active = False
        self.bottom = None
        self.top = None
        self.step = None
        self.grids = 0

    def is_valid(self) -> bool:
        if not self.active:
            return False
        if self.bottom is None or self.top is None or self.step is None:
            return False
        if self.top <= self.bottom or self.step <= 0:
            return False
        if self.grids <= 0:
            return False
        return True


@dataclass
class DayState:
    day_utc: date
    start_equity: float
    min_equity: float = 0.0

    def __post_init__(self) -> None:
        self.min_equity = self.start_equity

    def reset_if_new_day(self, equity: float) -> bool:
        today = utc_today()
        if self.day_utc != today:
            self.day_utc = today
            self.start_equity = equity
            self.min_equity = equity
            return True
        return False

    def update(self, equity: float) -> None:
        if equity < self.min_equity:
            self.min_equity = equity

    @property
    def drawdown(self) -> float:
        return float(self.start_equity) - float(self.min_equity)


# =========================
# GRID LOGIKA
# =========================

def calculate_smart_grid_count(equity: float, cfg: Dict[str, Any]) -> int:
    usable = float(equity) * float(cfg["risk"].get("grid_equity_ratio", 0.6))
    min_cost = float(cfg["market"]["min_order_usdt"])
    max_by_equity = int(usable / min_cost) if min_cost > 0 else 0

    max_grids = min(
        int(cfg["grid"]["max_grids"]),
        int(cfg["grid"].get("order_count_cap", 999)),
        max_by_equity,
    )
    return max(3, max_grids)


def adx_allows_grid(df: pd.DataFrame, cfg: Dict[str, Any], last_state: bool) -> bool:
    r = cfg["regime"]
    adx = calculate_adx(df, int(r["adx_period"]))
    val = safe_float(adx.iloc[-2], default=None)

    # ha nincs stabil ADX, ne változtassunk állapotot
    if val is None or not math.isfinite(val):
        return last_state

    if val > float(r["adx_off"]):
        return False
    if val < float(r["adx_on"]):
        return True
    return last_state


def compute_range_and_step(df: pd.DataFrame, cfg: Dict[str, Any], grids: int) -> Optional[Tuple[float, float, float]]:
    g = cfg["grid"]
    df = df.copy()

    df["atr"] = calculate_atr(df, int(g["atr_period"]))
    atr = safe_float(df["atr"].iloc[-2], default=None)
    if atr is None or atr <= 0 or not math.isfinite(atr):
        return None

    lookback = int(g["lookback_bars"])
    window = df.iloc[-(lookback + 2):-2]
    if window is None or len(window) < 5:
        return None

    low = safe_float(window["low"].min(), default=None)
    high = safe_float(window["high"].max(), default=None)
    if low is None or high is None:
        return None

    bottom = float(low - float(g["atr_buffer"]) * atr)
    top = float(high + float(g["atr_buffer"]) * atr)

    if not math.isfinite(bottom) or not math.isfinite(top) or top <= bottom:
        return None

    mid = (top + bottom) / 2.0
    raw_step = (top - bottom) / max(grids - 1, 1)
    min_step = mid * float(g["min_step_pct"])
    step = max(raw_step, min_step)

    # illesztés a grids darabszámhoz (stabil rebuild-hez)
    top = bottom + step * (grids - 1)
    return bottom, top, step


def _open_order_price_set(ex: ccxt.Exchange, symbol: str) -> Tuple[set, set]:
    """
    Minimal 'profi' upgrade: ne próbáljon ugyanarra az árszintre újra és újra ordert rakni.
    Logika nem változik, csak duplikációt fogunk meg.
    """
    buy_prices = set()
    sell_prices = set()
    try:
        oo = ex.fetch_open_orders(symbol) or []
        for o in oo:
            side = (o.get("side") or "").lower()
            p = safe_float(o.get("price"), default=None)
            if p is None:
                continue
            if side == "buy":
                buy_prices.add(float(p))
            elif side == "sell":
                sell_prices.add(float(p))
    except Exception:
        pass
    return buy_prices, sell_prices


def ensure_grid_orders(ex: ccxt.Exchange, symbol: str, grid: GridState, mark: float, equity: float, cfg: Dict[str, Any]) -> int:
    min_notional = float(cfg["market"]["min_order_usdt"])
    usable = float(equity) * float(cfg["risk"].get("grid_equity_ratio", 0.6))
    max_orders = int(cfg["grid"].get("order_count_cap", 50))

    if grid.bottom is None or grid.top is None or grid.step is None:
        return 0

    # Price ladder
    prices: List[float] = []
    p = float(grid.bottom)
    top = float(grid.top)
    step = float(grid.step)

    # védőkorlát: ha valamiért step nagyon kicsi, ne végtelen ciklus
    hard_cap = max_orders + 20

    while p <= top + 1e-9 and len(prices) < hard_cap:
        try:
            prices.append(float(ex.price_to_precision(symbol, p)))
        except Exception:
            prices.append(float(p))
        p += step

    prices = sorted(set(prices))
    if not prices:
        return 0

    # Szűrés: ne rakjunk túl közel a markethez (maker-t véd)
    buys = [px for px in prices if px < mark * 0.999]
    sells = [px for px in prices if px > mark * 1.001]
    if not buys:
        buys = [prices[0]]
    if not sells:
        sells = [prices[-1]]

    planned = min(len(buys) + len(sells), max_orders)
    if planned <= 0:
        return 0

    per_order = max(usable / planned, min_notional)

    # Duplikáció védelem (existing open orders)
    buy_open, sell_open = _open_order_price_set(ex, symbol)

    orders = 0
    for side, plist, open_set in [("buy", buys, buy_open), ("sell", sells, sell_open)]:
        for price in plist:
            if orders >= max_orders:
                break

            # ha már van open order ugyanazon az áron (vagy nagyon közel), skip
            # (precision miatt 1e-12 helyett relatív tolerancia)
            if any(abs(price - op) / max(op, 1e-9) < 1e-6 for op in open_set):
                continue

            raw_amt = per_order / price
            try:
                amt = float(ex.amount_to_precision(symbol, raw_amt))
            except Exception:
                amt = float(raw_amt)

            notional = amt * price
            if notional < min_notional:
                continue

            try:
                place_limit(ex, symbol, side, price, amt, bool(cfg["grid"]["post_only"]))
                orders += 1
                open_set.add(price)
            except Exception:
                # egy order hibától ne omoljon össze
                pass

    return orders


# =========================
# FŐ LOOP
# =========================

def run() -> None:
    ex = make_exchange(CFG)
    symbol = CFG["market"]["symbol"]
    tf = CFG["market"]["timeframe"]
    poll = float(CFG["market"]["poll_seconds"])

    set_leverage_margin(
        ex,
        symbol,
        int(CFG["exchange"]["leverage"]),
        str(CFG["exchange"]["marginMode"]),
    )

    grid = GridState()

    initial_equity = get_balance_usdt(ex) or 0.0
    if initial_equity < 10:
        initial_equity = float(CFG["risk"]["equity_usdt_assumed"])
    equity = float(initial_equity)

    day = DayState(utc_today(), equity)

    # ADX state + cooldown
    last_regime = True
    adx_pause_since: Optional[float] = None
    ADX_COOLDOWN_SEC = int(CFG.get("adx", {}).get("cooldown_sec", 60))

    # Backoff
    err_count = 0
    max_backoff = int(CFG["market"].get("max_backoff_sec", 60))

    log_jsonl(CFG["logging"]["jsonl_path"], {
        "event": "START_SESSION",
        "start_equity": round(float(initial_equity), 8),
        "symbol": symbol,
        "tf": tf,
    })

    print("--- Cuncibot FINAL elindult ---")

    while True:
        try:
            print("LOOP", utc_now().isoformat())

            # ===== EQUITY & PNL =====
            eq_now = get_balance_usdt(ex) or 0.0
            if eq_now > 10:
                equity = float(eq_now)

            session_pnl = float(equity) - float(initial_equity)

            if day.reset_if_new_day(equity):
                log_jsonl(CFG["logging"]["jsonl_path"], {
                    "event": "NEW_DAY",
                    "equity": round(float(equity), 8),
                })
            day.update(equity)

            # DAILY DD STOP (logika marad: risk param már CFG-ben volt; most ténylegesen használjuk)
            if day.drawdown >= float(CFG["risk"]["daily_dd_usdt"]):
                print("🛑 Daily DD elérve → pause")
                cancel_all_orders(ex, symbol)
                grid.reset()
                log_jsonl(CFG["logging"]["jsonl_path"], {
                    "event": "DAILY_DD_STOP",
                    "drawdown": round(float(day.drawdown), 8),
                    "equity": round(float(equity), 8),
                })
                time.sleep(max(60.0, poll))
                continue

            # ===== MARKET DATA =====
            df = fetch_ohlcv_df(ex, symbol, tf, int(CFG["market"]["ohlcv_limit"]))
            if df is None or len(df) < 5:
                print("⚠ OHLCV kevés / None, várakozás...")
                time.sleep(poll)
                continue

            try:
                ticker = ex.fetch_ticker(symbol) or {}
            except Exception:
                ticker = {}

            try:
                fallback_close = df.iloc[-2]["close"]
            except Exception:
                fallback_close = df.iloc[-1]["close"]

            mark = safe_float(ticker.get("mark", ticker.get("last", fallback_close)), default=None)
            if mark is None or not math.isfinite(mark) or mark <= 0:
                print("⚠ Mark hibás, várakozás...")
                time.sleep(poll)
                continue
            mark = float(mark)

            # ===== ADX REGIME (STATE + COOLDOWN) =====
            regime = adx_allows_grid(df, CFG, last_regime)

            # Trend indul: True -> False
            if (regime is False) and (last_regime is True):
                adx_pause_since = time.time()

            # Trend alatt: rács törlés + pause
            if regime is False:
                if grid.active:
                    print("⚠ ADX Trend → rácsok törlése, pause")
                    cancel_all_orders(ex, symbol)
                    grid.reset()
                    log_jsonl(CFG["logging"]["jsonl_path"], {
                        "event": "ADX_PAUSE",
                        "mark": mark,
                        "pnl": round(float(session_pnl), 8),
                    })

                last_regime = False
                time.sleep(poll)
                continue

            # ADX visszaenged: cooldown
            if (last_regime is False) and (regime is True):
                if adx_pause_since and (time.time() - adx_pause_since) < ADX_COOLDOWN_SEC:
                    print("⏳ ADX cooldown, még nem építünk")
                    time.sleep(poll)
                    continue
                adx_pause_since = None

            last_regime = True

            # ===== GRID CALC =====
            grids = int(calculate_smart_grid_count(equity, CFG) or 0)
            if grids <= 0:
                time.sleep(poll)
                continue

            rng = compute_range_and_step(df, CFG, grids)
            if rng is None:
                print("⚠ Range None, várakozás...")
                time.sleep(poll)
                continue

            bottom, top, step = rng
            if not (math.isfinite(bottom) and math.isfinite(top) and math.isfinite(step)):
                print("⚠ Range nem finite, várakozás...")
                time.sleep(poll)
                continue
            if not (top > bottom) or step <= 0:
                print(f"⚠ Range hibás (bottom={bottom}, top={top}, step={step})")
                time.sleep(poll)
                continue

            # ===== REBUILD DECISION =====
            drift = 0.0
            if grid.bottom is not None:
                drift = abs(float(grid.bottom) - float(bottom)) / max(abs(float(bottom)), 1e-9)

            rebuild = (not grid.is_valid()) or (drift > 0.02)

            # ===== GRID EXECUTION =====
            if rebuild:
                print(f"🚀 Rács építése... Tőke: {equity:.2f} | PNL: {session_pnl:.2f} | drift={drift:.4f}")

                cancel_all_orders(ex, symbol)

                grid.active = True
                grid.bottom, grid.top, grid.step = float(bottom), float(top), float(step)
                grid.grids = int(grids)

                placed = ensure_grid_orders(ex, symbol, grid, mark, equity, CFG)

                log_jsonl(CFG["logging"]["jsonl_path"], {
                    "event": "GRID_REBUILD",
                    "placed": int(placed),
                    "mark": float(mark),
                    "grids": int(grids),
                    "bottom": float(bottom),
                    "top": float(top),
                    "step": float(step),
                    "start_equity": round(float(initial_equity), 8),
                    "current_equity": round(float(equity), 8),
                    "session_pnl": round(float(session_pnl), 8),
                    "drift": round(float(drift), 6),
                })

            else:
                # ===== REFILL =====
                try:
                    open_orders = ex.fetch_open_orders(symbol) or []
                except Exception:
                    open_orders = []

                refill_ratio = float(CFG["grid"].get("refill_ratio", 0.7))
                threshold = int(max(1, grids * refill_ratio))

                if len(open_orders) < threshold:
                    print(f"♻ Rácsok utántöltése... open={len
