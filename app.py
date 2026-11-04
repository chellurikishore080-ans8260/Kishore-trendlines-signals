"""
Trendline Zone Scanner v2 - Binance Spot (USDT pairs)
Author: Kishore

Same trendline-zone logic as before (Pine "Trendlines" indicator, MPL-2.0, ported to Python)
on 5m / 15m / 30m / 1h / 4h. Two signals only:

  1. W / M pattern  - double bottom (W) at a SUPPORT zone / double top (M) at a RESISTANCE zone.
                      Signal = close breaks the neckline after the 2nd bottom/top.
  2. Break + Retest - a trendline zone is broken (close beyond it), price comes back to the
                      zone, and a SWEEP candle (wick goes through the zone edge, then closes
                      back on the right side, strong close) stands on it.

Every signal has Entry / SL / TP (RR based) and a RESULT: TP hit / SL hit / open / timeout.
The Backtest tab runs the same detection walk-forward (no look-ahead) and shows win rate and
PnL per timeframe, per pattern, equity curves and a chart for every trade.

Trade direction (setting, default "Short only"): every signal is traded as a SHORT
(W bottom / break-up are mirrored into shorts, M top / break-down are shorts as usual).
"""

import bisect
import io
import math
import struct
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
from numpy.lib.stride_tricks import sliding_window_view

IST = ZoneInfo("Asia/Kolkata")

BASES = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]

TF_OPTIONS = ["5m", "15m", "30m", "1h", "4h"]
TF_SECONDS = {"5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400}

STABLES = {
    "USDC", "FDUSD", "TUSD", "BUSD", "USDP", "DAI", "EUR", "AEUR", "USDE",
    "XUSD", "USD1", "UST", "USDS", "BFUSD", "RLUSD", "EURI",
}

ATR_LEN = 14
ATR_MULT = 0.5      # fixed zone width (same as the Pine script)
STABILITY = 5       # stability buffer in bars (same as the Pine script)
KLINES = 450        # candles kept per coin / timeframe for the live scan
SL_BUF = 0.1        # stop-loss buffer beyond the candle / pattern extreme (x ATR)

_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0"})
_session.mount(
    "https://", requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
)


# ───────────────────────────── Binance data ─────────────────────────────
_pause_until = 0.0  # shared by all worker threads: no request before this time


def _throttle():
    wait = _pause_until - time.time()
    if wait > 0:
        time.sleep(min(wait, 125))


def http_get(base, path, params=None, tries=3):
    global _pause_until
    for attempt in range(tries):
        _throttle()
        try:
            r = _session.get(base + path, params=params, timeout=12)
            used = r.headers.get("x-mbx-used-weight-1m", "")
            if used.isdigit() and int(used) >= 5400:
                now = time.time()
                _pause_until = max(_pause_until, now + (60 - now % 60) + 1)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (418, 429):
                try:
                    wait = min(int(r.headers.get("Retry-After", "5")), 120)
                except ValueError:
                    wait = 5
                _pause_until = max(_pause_until, time.time() + wait)
                continue
            if r.status_code == 451:
                return None
        except requests.RequestException:
            time.sleep(0.5 * (attempt + 1))
    return None


@st.cache_data(ttl=600, show_spinner=False)
def find_base():
    errs = []
    for b in BASES:
        try:
            r = _session.get(b + "/api/v3/ping", timeout=8)
            if r.status_code == 200:
                return b, ""
            errs.append(f"{b.split('//')[1]}: {r.status_code}")
        except requests.RequestException as e:
            errs.append(f"{b.split('//')[1]}: {type(e).__name__}")
    return None, " | ".join(errs)


@st.cache_data(ttl=3600, show_spinner=False)
def load_symbols(base):
    info = http_get(base, "/api/v3/exchangeInfo")
    if not info:
        return []
    out = []
    for s in info.get("symbols", []):
        if s.get("status") != "TRADING" or s.get("quoteAsset") != "USDT":
            continue
        if not s.get("isSpotTradingAllowed", True):
            continue
        if s.get("baseAsset") in STABLES:
            continue
        out.append(s["symbol"])
    return sorted(out)


@st.cache_data(ttl=300, show_spinner=False)
def load_volumes(base):
    data = http_get(base, "/api/v3/ticker/24hr")
    if not data:
        return {}
    return {d["symbol"]: float(d.get("quoteVolume", 0) or 0) for d in data}


def pick_symbols(base, top_n):
    """All USDT coins, or only the top-N by 24h volume (faster scan / backtest)."""
    symbols = load_symbols(base)
    if top_n and top_n > 0:
        vols = load_volumes(base)
        symbols = sorted(symbols, key=lambda s: -vols.get(s, 0))[:top_n]
    return symbols


def to_arrays(d):
    return {
        "t": np.array([x[0] for x in d], dtype=np.int64),
        "o": np.array([float(x[1]) for x in d]),
        "h": np.array([float(x[2]) for x in d]),
        "l": np.array([float(x[3]) for x in d]),
        "c": np.array([float(x[4]) for x in d]),
        "v": np.array([float(x[5]) for x in d]),
        "tb": np.array([float(x[9]) for x in d]),
    }


def fetch_klines(base, symbol, tf, limit=KLINES, min_len=60):
    d = http_get(
        base, "/api/v3/klines", {"symbol": symbol, "interval": tf, "limit": limit}
    )
    if not d or len(d) < min_len:
        return None
    return to_arrays(d)


def fetch_history(base, sym, tf, bars):
    """Long history for the backtest (pages backwards, 1000 candles per request)."""
    rows, end = [], None
    while len(rows) < bars:
        lim = min(1000, bars - len(rows))
        p = {"symbol": sym, "interval": tf, "limit": lim}
        if end is not None:
            p["endTime"] = end
        d = http_get(base, "/api/v3/klines", p)
        if not d:
            break
        rows = d + rows
        end = d[0][0] - 1
        if len(d) < lim:
            break
    if len(rows) < 250:
        return None
    return to_arrays(rows)


@st.cache_resource
def kline_store():
    return {}


@st.cache_resource
def zone_store():
    """Trendline zones already computed per (symbol, tf, bar) - reused on every refresh."""
    return {}


def get_klines(base, store, sym, tf):
    """First call: full history. Later calls: only the newest few candles are merged."""
    key = (sym, tf)
    cur = store.get(key)
    if cur is not None:
        step = TF_SECONDS[tf] * 1000
        last_t = int(cur["t"][-1])
        missing = (int(time.time() * 1000) - last_t) // step + 1
        if missing <= 90:
            tail = fetch_klines(base, sym, tf, limit=int(missing) + 2, min_len=1)
            if tail is not None and int(tail["t"][0]) <= last_t:
                keep = cur["t"] < tail["t"][0]
                d = {k: np.concatenate((cur[k][keep], tail[k]))[-KLINES:] for k in cur}
                store[key] = d
                return d
    d = fetch_klines(base,
