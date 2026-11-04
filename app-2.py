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
    d = fetch_klines(base, sym, tf)
    if d is not None:
        store[key] = d
    return d


# ───────────────────────────── Trendline logic (unchanged idea) ─────────────────────────────
def atr_series(h, l, c, n=ATR_LEN):
    """Wilder ATR (same as ta.atr) for every bar."""
    pc = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    tr[0] = h[0] - l[0]
    out = np.full(len(c), np.nan)
    if len(c) < n:
        return out
    r = tr[:n].mean()
    out[n - 1] = r
    for k in range(n, len(c)):
        r += (tr[k] - r) / n
        out[k] = r
    return out


def pivots(arr, s):
    """Pivot highs / lows with s bars on each side (like ta.pivothigh/low)."""
    n = len(arr)
    e = np.array([], dtype=int), np.array([])
    if n < 2 * s + 1:
        return e + e
    w = sliding_window_view(arr, 2 * s + 1)
    ctr = arr[s:n - s]
    left, right = w[:, :s], w[:, s + 1:]
    idx = np.arange(s, n - s)
    hi = (ctr >= left.max(1)) & (ctr > right.max(1))
    lo = (ctr <= left.min(1)) & (ctr < right.min(1))
    return idx[hi], ctr[hi], idx[lo], ctr[lo]


def find_zone(px, py, last, close, thr, is_res, back, min_touches):
    """Closest valid trendline zone to the current price (Pine findClosestZone).
    Vectorised over the 2nd pivot, same result as the loop version."""
    size = len(px)
    if size < min_touches or thr <= 0:
        return None
    look = last - back
    stab = last - STABILITY
    best, best_d = None, 1e30
    for i in range(size - 1):
        if px[i] < look:
            continue
        p1x, p1y = px[i], py[i]
        kx, ky = px[i:], py[i:]
        old = kx < stab
        dx = kx - p1x
        slopes = (py[i + 1:] - p1y) / (px[i + 1:] - p1x)
        diff = ky[None, :] - (p1y + slopes[:, None] * dx[None, :])
        viol = (diff > thr) if is_res else (diff < -thr)
        bad = (viol & old[None, :]).any(axis=1)
        touch = np.abs(diff) <= thr
        cnt = touch.sum(axis=1)
        ok = (~bad) & (cnt >= min_touches)
        if not ok.any():
            continue
        dist = np.abs(p1y + slopes * (last - p1x) - close)
        dist = np.where(ok, dist, np.inf)
        jj = int(np.argmin(dist))
        if dist[jj] < best_d:
            tv = diff[jj][touch[jj]]
            best_d = float(dist[jj])
            best = dict(
                p1x=float(p1x), p1y=float(p1y), slope=float(slopes[jj]),
                up=float(max(tv.max(), 0.0)), dn=float(min(tv.min(), 0.0)),
                touches=int(cnt[jj]),
            )
    return best


def zone_at(z, x):
    """Zone (top, bottom) at absolute bar number x."""
    base = z["p1y"] + z["slope"] * (x - z["p1x"])
    return base + z["up"], base + z["dn"]


def zones_at(d, bn, atr, k, cfg):
    """Nearest resistance / support zone using only data up to bar k (no look-ahead)."""
    thr = atr[k] * ATR_MULT
    if not np.isfinite(thr) or thr <= 0:
        return {"res": None, "sup": None}
    s = cfg["strength"]
    lo = max(0, k + 1 - (cfg["back"] + 2 * s + 5))
    hs = d["c"] if cfg["src"] == "Close" else d["h"]
    ls = d["c"] if cfg["src"] == "Close" else d["l"]
    phx, phy, _, _ = pivots(hs[lo:k + 1], s)
    _, _, plx, ply = pivots(ls[lo:k + 1], s)
    last, close = float(bn[k]), float(d["c"][k])

    def px(ix):
        return bn[lo + ix].astype(float)

    return {
        "res": find_zone(px(phx), phy, last, close, thr, True, cfg["back"], cfg["min_touches"]),
        "sup": find_zone(px(plx), ply, last, close, thr, False, cfg["back"], cfg["min_touches"]),
    }


# ───────────────────────────── Signal detection ─────────────────────────────
def detect(d, sym, tf, cfg, i0, zc, live):
    """Walk the candles bar by bar and return every signal whose candle index >= i0.
    Only closed candles are used (the last candle is still forming)."""
    c, o, h, l = d["c"], d["o"], d["h"], d["l"]
    n = len(c)
    if n < 150:
        return []
    ms = TF_SECONDS[tf] * 1000
    bn = d["t"] // ms
    atr = atr_series(h, l, c)
    last_i = n - 2
    s = cfg["w_strength"]
    ph = [int(x) for x in pivots(h, s)[0]]
    pl = [int(x) for x in pivots(l, s)[2]]
    zsig = (cfg["strength"], cfg["back"], cfg["src"], cfg["min_touches"])
    warm = max(120, cfg["back"] // 2)
    i_start = max(warm, i0 - cfg["retest_win"] - 3)
    out, broken, bkeys, done = [], [], set(), set()
    active, have = {"res": None, "sup": None}, False
    coin = sym[:-4]

    def mk(ptype, kind, direction, i, entry, sl, z, bot, top, extra=None):
        if i < i0:
            return
        if ptype == "W/M":
            base = "W bottom" if direction > 0 else "M top"
        else:
            base = "Break + retest sweep"
        if cfg["mode"] == "Short only":
            if direction > 0:               # long-type signal (W / break-up) -> mirrored SHORT
                sl = 2 * entry - sl         # SL same distance, but above entry
            direction = -1                  # M top / break-down are already shorts: unchanged
            name = f"{base} → SHORT ↓"
        elif cfg["mode"] == "Invert all":
            direction = -direction          # opposite trade
            sl = 2 * entry - sl             # mirrored stop-loss (same distance, other side)
            name = f"{base} → " + ("LONG ↑" if direction > 0 else "SHORT ↓")
        else:
            name = base + (" ↑" if direction > 0 else " ↓")
        if cfg["long_only"] and direction < 0:
            return
        risk = abs(entry - sl) / entry * 100
        if risk < cfg["min_risk"] or risk > cfg["max_risk"]:
            return
        out.append(dict(
            symbol=sym, coin=coin, tf=tf, group=ptype, kind=kind, dir=direction,
            signal=name, pos=int(i), bar=int(d["t"][i]), entry=entry, sl=sl,
            tp=entry + direction * cfg["rr"] * abs(entry - sl), risk=risk,
            zbot=float(bot), ztop=float(top), z=z, extra=extra or {},
            pivots=z["touches"],
        ))

    for i in range(i_start, last_i + 1):
        if (not have) or bn[i] % cfg["zstep"] == 0 or (live and i == last_i):
            key = (sym, tf, int(bn[i]), zsig)
            zz = zc.get(key)
            if zz is None:
                zz = zones_at(d, bn, atr, i, cfg)
                zc[key] = zz
            active, have = zz, True
        a = float(atr[i])
        if not np.isfinite(a) or a <= 0 or i < 1:
            continue
        x, xp = float(bn[i]), float(bn[i - 1])

        # ── A) W bottom / M top at the trendline zone ──
        if cfg["wm"]:
            for kind in ("sup", "res"):
                z = active[kind]
                if not z:
                    continue
                lowside = kind == "sup"
                pv = pl if lowside else ph
                cp = bisect.bisect_right(pv, i - s)  # pivots already confirmed
                if cp < 2:
                    continue
                p2 = pv[cp - 1]
                if i - p2 > cfg["w_win"] or i - p2 < 1:
                    continue
                for q in range(cp - 2, max(-1, cp - 5), -1):
                    p1 = pv[q]
                    gap = p2 - p1
                    if gap > cfg["w_max_gap"]:
                        break
                    if gap < cfg["w_min_gap"] or (kind, p1, p2) in done:
                        continue
                    ok = True
                    for p in (p1, p2):
                        tp_, bt_ = zone_at(z, float(bn[p]))
                        if lowside:
                            ok = ok and l[p] <= tp_ and l[p] >= bt_ - cfg["pierce"] * a
                        else:
                            ok = ok and h[p] >= bt_ and h[p] <= tp_ + cfg["pierce"] * a
                    if not ok:
                        continue
                    ext = l if lowside else h
                    if abs(ext[p1] - ext[p2]) > cfg["w_tol"] * a:
                        continue
                    if lowside:
                        neck = float(h[p1:p2 + 1].max())
                        brk = c[i] > neck and c[i - 1] <= neck and (
                            i - 1 <= p2 or c[p2 + 1:i].max() <= neck)
                    else:
                        neck = float(l[p1:p2 + 1].min())
                        brk = c[i] < neck and c[i - 1] >= neck and (
                            i - 1 <= p2 or c[p2 + 1:i].min() >= neck)
                    if not brk:
                        continue
                    done.add((kind, p1, p2))
                    top, bot = zone_at(z, x)
                    ex = dict(neck=neck, p1=p1, p2=p2)
                    if lowside:
                        mk("W/M", kind, 1, i, float(c[i]),
                           float(min(l[p1], l[p2]) - SL_BUF * a), z, bot, top, ex)
                    else:
                        mk("W/M", kind, -1, i, float(c[i]),
                           float(max(h[p1], h[p2]) + SL_BUF * a), z, bot, top, ex)
                    break

        # ── B) Break + retest + sweep candle ──
        if cfg["retest"]:
            keep = []
            for b in broken:
                age = i - b["pos"]
                if age > cfg["retest_win"]:
                    continue
                z = b["z"]
                top, bot = zone_at(z, x)
                rng = float(h[i] - l[i])
                if b["kind"] == "res":  # broken resistance -> now support -> LONG
                    if c[i] < bot:
                        continue  # fell back under the zone: breakout failed
                    if age >= 1 and rng > 0 and l[i] <= top and l[i] >= bot - cfg["sweep_cap"] * a:
                        strength = float(c[i] - l[i]) / rng * 100
                        swept = l[i] < bot
                        if (c[i] > o[i] and c[i] >= bot and strength >= cfg["close_str"]
                                and (swept or not cfg["need_sweep"])):
                            mk("Retest", "res", 1, i, float(c[i]),
                               float(l[i] - SL_BUF * a), z, bot, top,
                               dict(swept=bool(swept), brk=int(b["pos"])))
                            continue
                else:  # broken support -> now resistance -> SHORT
                    if c[i] > top:
                        continue
                    if age >= 1 and rng > 0 and h[i] >= bot and h[i] <= top + cfg["sweep_cap"] * a:
                        strength = float(h[i] - c[i]) / rng * 100
                        swept = h[i] > top
                        if (c[i] < o[i] and c[i] <= top and strength >= cfg["close_str"]
                                and (swept or not cfg["need_sweep"])):
                            mk("Retest", "sup", -1, i, float(c[i]),
                               float(h[i] + SL_BUF * a), z, bot, top,
                               dict(swept=bool(swept), brk=int(b["pos"])))
                            continue
                keep.append(b)
            broken = keep
            for k in ("res", "sup"):
                z = active[k]
                if not z:
                    continue
                key = (k, z["p1x"], round(z["slope"], 10))
                if key in bkeys:
                    continue
                top, bot = zone_at(z, x)
                topp, botp = zone_at(z, xp)
                if (k == "res" and c[i] > top and c[i - 1] <= topp) or (
                        k == "sup" and c[i] < bot and c[i - 1] >= botp):
                    bkeys.add(key)
                    broken.append(dict(kind=k, z=z, pos=i))
    return out


# ───────────────────────────── Trade result + PnL ─────────────────────────────
def evaluate(d, sg, cfg):
    """Walk the candles after the signal: TP / SL / TIMEOUT / OPEN (SL first if both in one candle)."""
    h, l, c = d["h"], d["l"], d["c"]
    n = len(c)
    i, dr, sl, tp = sg["pos"], sg["dir"], sg["sl"], sg["tp"]
    end = min(n - 1, i + cfg["max_hold"])
    res, exit_px, exit_pos = None, float(c[n - 1]), n - 1
    for j in range(i + 1, end + 1):
        if dr == 1:
            if l[j] <= sl:
                res, exit_px, exit_pos = "SL", sl, j
                break
            if h[j] >= tp:
                res, exit_px, exit_pos = "TP", tp, j
                break
        else:
            if h[j] >= sl:
                res, exit_px, exit_pos = "SL", sl, j
                break
            if l[j] <= tp:
                res, exit_px, exit_pos = "TP", tp, j
                break
    if res is None:
        if i + cfg["max_hold"] <= n - 1:
            res, exit_px, exit_pos = "TIMEOUT", float(c[end]), end
        else:
            res = "OPEN"
    rf = abs(sg["entry"] - sg["sl"]) / sg["entry"]
    risk_amt = cfg["capital"] * cfg["risk_pct"] / 100
    notional = min(risk_amt / rf, cfg["capital"] * cfg["max_pos"] / 100)
    ret = dr * (exit_px - sg["entry"]) / sg["entry"]
    pnl = notional * (ret - 2 * cfg["fee"] / 100)
    return dict(
        result=res, exit=float(exit_px), exit_pos=int(exit_pos), bars=int(exit_pos - i),
        pnl=float(pnl), R=float(pnl / risk_amt), notional=float(notional),
        t_exit=int(d["t"][exit_pos]),
    )


RESULT_LABEL = {"TP": "✅ TP hit", "SL": "❌ SL hit", "OPEN": "⏳ Open", "TIMEOUT": "⌛ Timeout"}


def result_label(s):
    if s["result"] == "TIMEOUT":
        return "⌛ Timeout " + ("(+)" if s["pnl"] > 0 else "(-)")
    return RESULT_LABEL[s["result"]]


# ───────────────────────────── Live scan ─────────────────────────────
def scan_live(base, store, symbols, tfs, cfg, zc, workers=16):
    sigs, ok, fail = [], 0, 0

    def job(sym, tf):
        d = get_klines(base, store, sym, tf)
        if d is None:
            return None
        n = len(d["c"])
        i0 = max(0, n - 2 - (cfg["live_age"] - 1))
        out = detect(d, sym, tf, cfg, i0, zc, True)
        for sg in out:
            sg.update(evaluate(d, sg, cfg))
            sg["price"] = float(d["c"][-1])
            sg["age"] = int((n - 2) - sg["pos"])
        return out

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(job, s, tf) for s in symbols for tf in tfs]
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception:
                r = None
            if r is None:
                fail += 1
            else:
                ok += 1
                sigs.extend(r)
    if len(zc) > 300000:
        zc.clear()
    sigs.sort(key=lambda x: (x["age"], TF_SECONDS[x["tf"]], x["risk"]))
    return sigs, ok, fail


# ───────────────────────────── Backtest ─────────────────────────────
def run_backtest(base, symbols, tfs, cfg, bars, progress, workers=8):
    zc, trades, hist, fail = {}, [], {}, 0
    t0 = time.time()
    warm = max(120, cfg["back"] // 2)

    def job(sym, tf):
        d = fetch_history(base, sym, tf, bars)
        if d is None:
            return None
        out = detect(d, sym, tf, cfg, warm, zc, False)
        for sg in out:
            sg.update(evaluate(d, sg, cfg))
        return sym, tf, d, out

    jobs = [(s, tf) for s in symbols for tf in tfs]
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(job, s, tf) for s, tf in jobs]
        for f in as_completed(futs):
            done += 1
            try:
                r = f.result()
            except Exception:
                r = None
            if r is None:
                fail += 1
            else:
                sym, tf, d, out = r
                hist[(sym, tf)] = d
                trades.extend(out)
            progress(done / len(jobs))
    return dict(trades=trades, hist=hist, fail=fail, took=time.time() - t0,
                n_jobs=len(jobs), cfg=dict(cfg))


def summarize(df, by, capital):
    rows = []
    for key, g in df.groupby(by, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        g = g.sort_values("t_exit")
        n = len(g)
        wins = int((g.pnl > 0).sum())
        gp = g.pnl[g.pnl > 0].sum()
        gl = -g.pnl[g.pnl < 0].sum()
        eq = np.concatenate(([0.0], g.pnl.cumsum().values))
        dd = float((eq - np.maximum.accumulate(eq)).min())
        row = dict(zip(by, key))
        row.update({
            "Trades": n, "Wins": wins, "Win %": round(wins / n * 100, 1),
            "TP": int((g.result == "TP").sum()), "SL": int((g.result == "SL").sum()),
            "Timeout": int((g.result == "TIMEOUT").sum()),
            "Net PnL": round(float(g.pnl.sum()), 0), "Avg R": round(float(g.R.mean()), 2),
            "Profit factor": round(float(gp / gl), 2) if gl > 0 else None,
            "Max DD": round(dd, 0), "Return %": round(float(g.pnl.sum()) / capital * 100, 2),
        })
        rows.append(row)
    return pd.DataFrame(rows)


# ───────────────────────────── Alerts ─────────────────────────────
@lru_cache(maxsize=2)
def beep_wav(kind="up"):
    notes = ((1047, 0.15), (1319, 0.15), (1568, 0.30)) if kind == "up" else (
        (784, 0.20), (659, 0.20), (523, 0.30))
    sr = 22050
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = bytearray()
        for f, dur in notes:
            n = int(sr * dur)
            for i in range(n):
                env = min(1.0, i / 200, (n - i) / 200)
                frames += struct.pack("<h", int(12000 * env * math.sin(2 * math.pi * f * i / sr)))
        w.writeframes(bytes(frames))
    return buf.getvalue()


def secret(name):
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


def send_telegram(token, chat_id, text):
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text}, timeout=8,
        )
    except requests.RequestException:
        pass


def fp(x):
    if x >= 100:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:.4f}"
    return f"{x:.6g}"


def alert_text(s):
    icon = "\U0001F7E2" if s["dir"] > 0 else "\U0001F534"
    return (
        f"{icon} {s['signal']}\n{s['symbol']} · {s['tf']} trendline\n"
        f"Entry {fp(s['entry'])} | SL {fp(s['sl'])} ({s['risk']:.2f}%) | TP {fp(s['tp'])}\n"
        f"Zone {fp(s['zbot'])} - {fp(s['ztop'])}"
    )


# ───────────────────────────── Chart ─────────────────────────────
def make_chart(d, tf, sg, show=120, after=70):
    import plotly.graph_objects as go

    n = len(d["c"])
    ms = TF_SECONDS[tf] * 1000
    bn = d["t"] // ms
    lo = max(0, sg["pos"] - show)
    hi = min(n, sg["pos"] + after + 1)
    times = pd.to_datetime(d["t"], unit="ms", utc=True).tz_convert(IST)

    def tx(x):
        return pd.to_datetime(float(x) * ms, unit="ms", utc=True).tz_convert(IST)

    fig = go.Figure(go.Candlestick(
        x=times[lo:hi], open=d["o"][lo:hi], high=d["h"][lo:hi],
        low=d["l"][lo:hi], close=d["c"][lo:hi], name="Price",
    ))
    z = sg["z"]
    rgb = "239,68,68" if sg["kind"] == "res" else "34,197,94"
    x0 = max(z["p1x"], float(bn[lo]))
    x1 = float(bn[hi - 1]) + 3
    top0, bot0 = zone_at(z, x0)
    top1, bot1 = zone_at(z, x1)
    t0, t1 = tx(x0), tx(x1)
    fig.add_trace(go.Scatter(
        x=[t0, t1, t1, t0, t0], y=[top0, top1, bot1, bot0, top0], fill="toself",
        mode="lines", hoverinfo="skip", line=dict(color=f"rgba({rgb},0.9)", width=1),
        fillcolor=f"rgba({rgb},0.18)", name="Trendline zone",
    ))
    ex = sg.get("extra") or {}
    if "neck" in ex:
        fig.add_trace(go.Scatter(
            x=[times[ex["p1"]], times[sg["pos"]]], y=[ex["neck"]] * 2, mode="lines",
            line=dict(color="#f59e0b", width=1.5, dash="dot"), name="Neckline"))
        for p in (ex["p1"], ex["p2"]):
            y = d["l"][p] if sg["kind"] == "sup" else d["h"][p]
            fig.add_trace(go.Scatter(
                x=[times[p]], y=[y], mode="markers", name="W/M point",
                marker=dict(symbol="circle", size=9, color="#f59e0b")))
    end = min(sg.get("exit_pos", hi - 1), hi - 1)
    for y, color, name in ((sg["entry"], "#3b82f6", "Entry"), (sg["sl"], "#ef4444", "SL"),
                           (sg["tp"], "#22c55e", "TP")):
        fig.add_trace(go.Scatter(
            x=[times[sg["pos"]], times[end]], y=[y, y], mode="lines", name=name,
            line=dict(color=color, width=1.5, dash="dash")))
    up = sg["dir"] > 0
    fig.add_trace(go.Scatter(
        x=[times[sg["pos"]]], y=[d["l"][sg["pos"]] if up else d["h"][sg["pos"]]],
        mode="markers", name="Signal candle",
        marker=dict(symbol="triangle-up" if up else "triangle-down", size=14,
                    color="#22c55e" if up else "#ef4444")))
    if sg.get("result") in ("TP", "SL", "TIMEOUT") and sg["exit_pos"] < hi:
        fig.add_trace(go.Scatter(
            x=[times[sg["exit_pos"]]], y=[sg["exit"]], mode="markers", name="Exit",
            marker=dict(symbol="x", size=11, color="#ffffff", line=dict(width=1, color="#111"))))
    fig.update_layout(
        height=470, xaxis_rangeslider_visible=False, showlegend=True,
        margin=dict(l=0, r=0, t=20, b=0), legend=dict(orientation="h", y=1.08),
    )
    return fig


# ───────────────────────────── App ─────────────────────────────
def sig_table(rows):
    return pd.DataFrame([{
        "Result": result_label(s), "Coin": s["coin"], "TF": s["tf"], "Signal": s["signal"],
        "Entry": fp(s["entry"]), "SL": fp(s["sl"]), "TP": fp(s["tp"]),
        "Risk %": round(s["risk"], 2), "Candles ago": s["age"], "Price now": fp(s["price"]),
        "Zone low": fp(s["zbot"]), "Zone high": fp(s["ztop"]), "Pivots": s["pivots"],
    } for s in rows])


def main():
    st.set_page_config(
        page_title="Trendline Zone Scanner", page_icon="\U0001F4C8", layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("\U0001F4C8 Trendline Zone Scanner")
    st.caption("Binance Spot · USDT pairs · W/M pattern + break-retest sweep · live + backtest · by Kishore")

    with st.expander("⚙️ Settings (optional - default ga run avtundi)", expanded=False):
        tfs = st.multiselect("Timeframes", TF_OPTIONS, default=TF_OPTIONS)
        c1, c2 = st.columns(2)
        refresh = c1.select_slider("Live auto refresh (seconds)", [30, 60, 90, 120, 180, 300], value=60)
        top_n = c2.number_input("Live: top N coins by 24h volume (0 = all)", 0, 600, 100, 10)

        st.markdown("**Trendline logic (same as before)**")
        back = st.slider("Bars to apply", 50, 350, 300)
        src = st.selectbox("Pivot source", ["High/Low", "Close"])
        strength = st.slider("Pivot strength", 5, 15, 10)
        min_touches = st.slider("Min pivot confirmation", 2, 8, 3)
        zstep = st.slider("Recompute zones every N candles (higher = faster)", 1, 10, 3)

        st.markdown("**Signal 1 · W bottom / M top at the trendline**")
        wm = st.checkbox("Enable W / M signal", True)
        w_strength = st.slider("Swing strength for the two bottoms/tops (bars each side)", 2, 6, 3)
        w_min_gap = st.slider("Min bars between the two bottoms/tops", 2, 20, 4)
        w_max_gap = st.slider("Max bars between the two bottoms/tops", 10, 120, 60)
        w_tol = st.slider("Max difference between the two bottoms/tops (× ATR)", 0.1, 3.0, 1.0, 0.1)
        pierce = st.slider("Bottoms/tops may go through the zone by up to (× ATR)", 0.0, 2.0, 0.75, 0.05)
        w_win = st.slider("Neckline break must come within N candles after the 2nd bottom/top", 2, 30, 12)

        st.markdown("**Signal 2 · Trendline break → retest → sweep candle**")
        retest = st.checkbox("Enable break + retest signal", True)
        retest_win = st.slider("Retest must happen within N candles after the break", 3, 100, 40)
        need_sweep = st.checkbox("Require a sweep (wick goes through the far edge of the zone)", True)
        sweep_cap = st.slider("Max wick through the zone (× ATR)", 0.1, 2.0, 0.75, 0.05)
        close_str = st.slider("Min close strength (% of candle range)", 0, 90, 50, 5)

        st.markdown("**Trade rules (used for result + backtest)**")
        r1, r2, r3 = st.columns(3)
        rr = r1.number_input("Reward : Risk (TP)", 0.5, 10.0, 2.0, 0.5)
        max_hold = r2.number_input("Max candles to hold", 10, 500, 100, 10)
        long_only = r3.checkbox("Spot: long signals only", False)
        mode = st.selectbox(
            "Trade direction",
            ["Short only", "Invert all", "As per signal"], index=0,
            help="Short only: W / break-up signals are traded as SHORT (mirrored), "
                 "M top / break-down are SHORT as usual - no long trades at all. "
                 "Invert all: W -> SHORT, M -> LONG. As per signal: W -> LONG, M -> SHORT.")
        r4, r5, r6 = st.columns(3)
        min_risk = r4.number_input("Skip if SL distance below (%)", 0.0, 5.0, 0.1, 0.05)
        max_risk = r5.number_input("Skip if SL distance above (%)", 1.0, 30.0, 8.0, 0.5)
        fee = r6.number_input("Fee per side (%)", 0.0, 0.5, 0.1, 0.01)
        r7, r8, r9 = st.columns(3)
        capital = r7.number_input("Capital (1 lakh = 100000)", 1000.0, 1e8, 100000.0, 10000.0)
        risk_pct = r8.number_input("Risk per trade (% of capital)", 0.1, 10.0, 1.0, 0.1)
        max_pos = r9.number_input("Max position size (% of capital, 100 = no leverage)", 10, 1000, 100, 10)

        st.markdown("**Live**")
        live_age = st.slider("Show signals from the last N closed candles", 1, 20, 6)
        sound = st.checkbox("Sound + popup on new signals", True)
        tg_token = st.text_input("Telegram bot token (optional)", secret("TELEGRAM_TOKEN"), type="password")
        tg_chat = st.text_input("Telegram chat id (optional)", secret("TELEGRAM_CHAT_ID"))

        st.markdown("**Backtest**")
        b1, b2 = st.columns(2)
        bt_coins = b1.number_input("Backtest: top N coins by volume", 1, 300, 25, 5)
        bt_bars = b2.number_input("Candles of history per coin / timeframe", 500, 5000, 1500, 250)
        st.caption("History: 5m 1500 candles ≈ 5 days, 1h ≈ 62 days, 4h ≈ 250 days. "
                   "Chinna timeframes ki data takkuva - trades takkuva untayi.")

    if not tfs:
        st.warning("Select at least one timeframe.")
        return

    cfg = dict(
        src=src, strength=strength, min_touches=min_touches, back=back, zstep=zstep,
        wm=wm, w_strength=w_strength, w_min_gap=w_min_gap, w_max_gap=w_max_gap,
        w_tol=w_tol, pierce=pierce, w_win=w_win,
        retest=retest, retest_win=retest_win, need_sweep=need_sweep,
        sweep_cap=sweep_cap, close_str=close_str,
        rr=rr, max_hold=int(max_hold), long_only=long_only, min_risk=min_risk,
        max_risk=max_risk, fee=fee, capital=capital, risk_pct=risk_pct, max_pos=max_pos,
        live_age=live_age, mode=mode,
    )
    store, zc = kline_store(), zone_store()
    tab_live, tab_bt = st.tabs(["\U0001F4E1 Live scanner", "\U0001F9EA Backtest"])

    # ───────────── Live ─────────────
    @st.fragment(run_every=f"{refresh}s")
    def live():
        base, err = find_base()
        if base is None:
            st.error("Binance data fetch avvaledu. " + err)
            return
        symbols = pick_symbols(base, top_n)
        if not symbols:
            st.error("Binance coin list radhu - konchem sepu tarvata malli try chestundi.")
            return
        sig = repr((tuple(tfs), top_n, sorted(cfg.items(), key=lambda kv: kv[0])))
        cache = st.session_state.get("scan_cache")
        if cache and cache["sig"] == sig and time.time() - cache["ts"] < refresh * 0.9:
            sigs, fail, took, scanned_at = cache["sigs"], cache["fail"], cache["took"], cache["at"]
        else:
            t0 = time.time()
            sigs, ok, fail = scan_live(base, store, symbols, tfs, cfg, zc)
            if ok == 0:
                st.error("Ekkada nundi kuda candles raaledu (Binance block / rate limit).")
                return
            took = time.time() - t0
            scanned_at = datetime.now(IST).strftime("%H:%M:%S")
            st.session_state["scan_cache"] = dict(
                sig=sig, ts=t0, sigs=sigs, fail=fail, took=took, at=scanned_at)

            seen = st.session_state.setdefault("seen", set())
            first_run = "seen_init" not in st.session_state
            st.session_state["seen_init"] = True
            new = []
            for s in sigs:
                key = f"{s['symbol']}|{s['tf']}|{s['signal']}|{s['bar']}"
                if key not in seen:
                    seen.add(key)
                    if s["result"] == "OPEN":
                        new.append(s)
            if len(seen) > 5000:
                seen.clear()
            if new and not first_run:
                log = st.session_state.setdefault("log", [])
                for s in reversed(new):
                    log.insert(0, {"Time": scanned_at, "Coin": s["coin"], "TF": s["tf"],
                                   "Signal": s["signal"], "Entry": fp(s["entry"]),
                                   "SL": fp(s["sl"]), "TP": fp(s["tp"])})
                del log[100:]
                if sound:
                    for s in new[:6]:
                        st.toast(f"{s['signal']}  {s['symbol']} {s['tf']}",
                                 icon="\U0001F7E2" if s["dir"] > 0 else "\U0001F534")
                    tone = "up" if any(s["dir"] > 0 for s in new) else "down"
                    st.audio(beep_wav(tone), format="audio/wav", autoplay=True)
                send_telegram(tg_token, tg_chat, "\n\n".join(alert_text(s) for s in new[:15]))

        st.caption(
            f"{len(symbols)} coins × {len(tfs)} TF scanned in {took:.0f}s "
            f"(failed requests: {fail}) · last scan {scanned_at} IST · refresh every {refresh}s"
        )
        closed = [s for s in sigs if s["result"] != "OPEN"]
        wins = sum(1 for s in closed if s["pnl"] > 0)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Signals", len(sigs))
        m2.metric("Open", sum(1 for s in sigs if s["result"] == "OPEN"))
        m3.metric("TP / win", wins)
        m4.metric("SL / loss", len(closed) - wins)

        for title, group in (("\U0001F7E2 Signal 1 · W bottom / M top at trendline", "W/M"),
                             ("\U0001F501 Signal 2 · Break + retest + sweep", "Retest")):
            rows = [s for s in sigs if s["group"] == group]
            st.subheader(f"{title} ({len(rows)})")
            if rows:
                st.dataframe(sig_table(rows), hide_index=True, width="stretch")
            else:
                st.info("Ippudu e signal ledu. Auto-refresh avtune untundi.")

        log = st.session_state.get("log")
        if log:
            with st.expander(f"Alert history ({len(log)})", expanded=False):
                st.dataframe(pd.DataFrame(log), hide_index=True, width="stretch")

        opts = {f"{s['coin']} · {s['tf']} · {s['signal']} · {result_label(s)} · "
                f"{s['age']} candles ago": s for s in sigs[:60]}
        if opts:
            with st.expander("Chart", expanded=False):
                pick = st.selectbox("Signal", list(opts), key="chart_pick")
                sel = opts.get(pick)
                if sel:
                    d = get_klines(base, store, sel["symbol"], sel["tf"])
                    if d is not None:
                        st.plotly_chart(make_chart(d, sel["tf"], sel), width="stretch")

    with tab_live:
        live()

    # ───────────── Backtest ─────────────
    with tab_bt:
        st.caption(
            "Same detection, walk-forward (future data use avvadu). Entry = signal candle close, "
            "SL = pattern / sweep candle extreme, TP = RR × risk. Oka candle lo TP & SL rendu "
            "touch aithe SL ne count avtundi. Fees include chesi PnL.")
        if st.button("▶ Run backtest", type="primary"):
            base, err = find_base()
            if base is None:
                st.error("Binance data fetch avvaledu. " + err)
            else:
                symbols = pick_symbols(base, bt_coins)
                prog = st.progress(0.0, text=f"{len(symbols)} coins × {len(tfs)} TF backtest avtundi...")
                res = run_backtest(base, symbols, tfs, cfg, int(bt_bars),
                                   lambda f: prog.progress(min(f, 1.0)))
                prog.empty()
                st.session_state["bt"] = res
        res = st.session_state.get("bt")
        if res:
            show_backtest(res)

    return


def show_backtest(res):
    cfg = res["cfg"]
    capital = cfg["capital"]
    trades = res["trades"]
    closed = [t for t in trades if t["result"] != "OPEN"]
    st.caption(f"{res['n_jobs'] - res['fail']}/{res['n_jobs']} coin-timeframes tested in "
               f"{res['took']:.0f}s · {len(closed)} closed trades · "
               f"{len(trades) - len(closed)} still open (not counted)")
    if not closed:
        st.info("E settings tho trades raaledu. Coins / candles penchandi leda filters tagginchandi.")
        return
    tf_order = [t for t in TF_OPTIONS if any(x["tf"] == t for x in closed)]
    df = pd.DataFrame([{
        "all": "All", "coin": t["coin"], "tf": t["tf"], "pattern": t["signal"],
        "group": "W / M" if t["group"] == "W/M" else "Break + retest",
        "side": "Long" if t["dir"] > 0 else "Short",
        "t_entry": pd.to_datetime(t["bar"], unit="ms", utc=True).tz_convert(IST),
        "t_exit": pd.to_datetime(t["t_exit"], unit="ms", utc=True).tz_convert(IST),
        "entry": t["entry"], "sl": t["sl"], "tp": t["tp"], "exit": t["exit"],
        "result": t["result"], "pnl": t["pnl"], "R": t["R"], "bars": t["bars"],
        "idx": k,
    } for k, t in enumerate(closed)])
    df["tf"] = pd.Categorical(df["tf"], categories=tf_order, ordered=True)

    tot = summarize(df, ["all"], capital).iloc[0]
    m = st.columns(5)
    m[0].metric("Trades", int(tot["Trades"]))
    m[1].metric("Win rate", f"{tot['Win %']}%")
    m[2].metric("Net PnL", f"{tot['Net PnL']:,.0f}")
    m[3].metric("Return", f"{tot['Return %']}%")
    m[4].metric("Profit factor", tot["Profit factor"] if tot["Profit factor"] is not None else "∞")

    by_tf = summarize(df, ["tf"], capital)
    st.subheader("Timeframe wise")
    st.dataframe(by_tf, hide_index=True, width="stretch")
    enough = by_tf[by_tf["Trades"] >= 5]
    pool = enough if len(enough) else by_tf
    bw = pool.sort_values("Win %", ascending=False).iloc[0]
    bp = by_tf.sort_values("Net PnL", ascending=False).iloc[0]
    st.success(f"Best win rate: **{bw['tf']}** ({bw['Win %']}% · {int(bw['Trades'])} trades)  |  "
               f"Best PnL: **{bp['tf']}** ({bp['Net PnL']:,.0f})")

    import plotly.graph_objects as go
    c1, c2 = st.columns(2)
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in by_tf["Net PnL"]]
    f1 = go.Figure(go.Bar(x=by_tf["tf"].astype(str), y=by_tf["Win %"], text=by_tf["Win %"],
                          marker_color="#3b82f6"))
    f1.update_layout(title="Win rate % by timeframe", height=300, margin=dict(l=0, r=0, t=40, b=0))
    f2 = go.Figure(go.Bar(x=by_tf["tf"].astype(str), y=by_tf["Net PnL"], text=by_tf["Net PnL"],
                          marker_color=colors))
    f2.update_layout(title="Net PnL by timeframe", height=300, margin=dict(l=0, r=0, t=40, b=0))
    c1.plotly_chart(f1, width="stretch")
    c2.plotly_chart(f2, width="stretch")

    st.subheader("Timeframe × pattern")
    st.dataframe(summarize(df, ["tf", "group"], capital), hide_index=True, width="stretch")
    c3, c4 = st.columns(2)
    c3.markdown("**By pattern**")
    c3.dataframe(summarize(df, ["pattern"], capital), hide_index=True, width="stretch")
    c4.markdown("**Long vs Short**")
    c4.dataframe(summarize(df, ["side"], capital), hide_index=True, width="stretch")

    st.subheader("Equity curve (cumulative PnL)")
    fig = go.Figure()
    for tf in tf_order:
        g = df[df.tf == tf].sort_values("t_exit")
        fig.add_trace(go.Scatter(x=g["t_exit"], y=g["pnl"].cumsum(), mode="lines", name=tf))
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

    st.subheader("All trades")
    show = df.sort_values("t_entry", ascending=False)
    st.dataframe(pd.DataFrame({
        "Time (IST)": show["t_entry"].dt.strftime("%d-%b %H:%M"), "Coin": show["coin"],
        "TF": show["tf"].astype(str), "Pattern": show["pattern"],
        "Entry": show["entry"].map(fp), "SL": show["sl"].map(fp), "TP": show["tp"].map(fp),
        "Result": show["result"].map({"TP": "✅ TP", "SL": "❌ SL", "TIMEOUT": "⌛ Timeout"}),
        "Exit": show["exit"].map(fp), "PnL": show["pnl"].round(0), "R": show["R"].round(2),
        "Candles held": show["bars"],
    }).head(1000), hide_index=True, width="stretch")

    with st.expander("Chart of a trade", expanded=True):
        top = show.head(300)
        labels = {f"{r.coin} · {r.tf} · {r.pattern} · {r.t_entry:%d-%b %H:%M} · {r.result}": int(r.idx)
                  for r in top.itertuples()}
        pick = st.selectbox("Trade", list(labels), key="bt_pick")
        sg = closed[labels[pick]]
        d = res["hist"].get((sg["symbol"], sg["tf"]))
        if d is not None:
            st.plotly_chart(make_chart(d, sg["tf"], sg), width="stretch")


if __name__ == "__main__":
    main()
