# =====================================================================
#  Kishore Trendline + Candlestick Scanner (with Backtest)
#  Created by Kishore
#
#  - Trendline ZONE logic ported from a TradingView Pine indicator
#    (pivot highs/lows -> best line with min touches -> ATR-wide zone)
#  - A trade is taken only when a candlestick pattern forms AT the zone
#    (Engulfing, Pin Bar, Momentum, Inside Bar, Three Methods,
#     Morning/Evening Star, Three Soldiers/Crows)
#  - Data: Binance Spot (data-api.binance.vision first, other hosts fallback)
#  - Live scanner auto-refreshes, Backtest shows every trade + PnL + win/fail
# =====================================================================
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from numpy.lib.stride_tricks import sliding_window_view as swv

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------
HOSTS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]
TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}
ALL_TFS = list(TF_SECONDS)
DEFAULT_TFS = ["1m", "5m", "15m", "30m", "4h", "1d"]
PATTERNS = ["Engulfing", "Pin Bar", "Momentum", "Inside Bar", "Three Methods",
            "Morning/Evening Star", "Soldiers/Crows"]
PRIORITY = ["Engulfing", "Morning/Evening Star", "Pin Bar", "Momentum",
            "Inside Bar", "Soldiers/Crows", "Three Methods"]
IST = "Asia/Kolkata"
STABLE_BASES = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "EUR", "AEUR", "USDE",
                "XUSD", "PYUSD", "UST", "USD1", "EURI", "WBTC"}
FALLBACK_COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
                  "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT", "TRXUSDT", "NEARUSDT",
                  "SUIUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT", "ATOMUSDT"]


@dataclass(frozen=True)
class Params:
    # --- trendline zone (same as Pine defaults) ---
    back_bars: int = 300
    pivot_src: str = "High/Low"
    pivot_strength: int = 10
    min_touches: int = 3
    atr_mult: float = 0.5
    refresh: int = 8                 # zone re-calc every N bars (and on every new pivot)
    # --- candlestick confirmation ---
    patterns: tuple = tuple(PATTERNS)
    engulf_confirm: bool = True      # wait for 3rd confirmation candle after engulfing
    trend_filter: bool = True        # long only if EMA50>EMA200, short only if EMA50<EMA200
    sides: tuple = ("LONG", "SHORT")
    zone_tol_atr: float = 0.1
    # --- trade management ---
    sl_buffer_atr: float = 0.15
    min_risk_atr: float = 0.8
    max_risk_atr: float = 3.0
    rr: float = 2.0
    min_tp_fee_mult: float = 3.0     # TP distance must be >= 3x round-trip fee
    max_hold: int = 60               # candles
    fee: float = 0.001               # 0.1% per side


# ---------------------------------------------------------------------
# DATA (Binance Spot)
# ---------------------------------------------------------------------
def _get(path, params, timeout=12):
    global HOSTS
    last = None
    for _ in range(2):
        for h in list(HOSTS):
            try:
                r = requests.get(h + path, params=params, timeout=timeout)
                if r.status_code == 200:
                    if HOSTS[0] != h:
                        HOSTS = [h] + [x for x in HOSTS if x != h]
                    return r.json()
                last = f"{h} {r.status_code}"
                if r.status_code in (418, 429):
                    time.sleep(1.5)
            except Exception as e:  # noqa
                last = f"{h} {type(e).__name__}"
        time.sleep(0.5)
    raise RuntimeError(last)


def fetch_klines(symbol, tf, n):
    rows, end, remaining = [], None, n
    while remaining > 0:
        lim = min(1000, remaining)
        params = {"symbol": symbol, "interval": tf, "limit": lim}
        if end is not None:
            params["endTime"] = end
        data = _get("/api/v3/klines", params)
        if not data:
            break
        rows = data + rows
        remaining -= len(data)
        end = int(data[0][0]) - 1
        if len(data) < lim:
            break
    if not rows:
        raise RuntimeError("no data")
    arr = np.array([[r[0], r[1], r[2], r[3], r[4], r[5], r[6]] for r in rows], dtype=float)
    df = pd.DataFrame(arr, columns=["t", "o", "h", "l", "c", "v", "ct"])
    return df.drop_duplicates("t").sort_values("t").reset_index(drop=True)


def top_symbols(topn):
    try:
        data = _get("/api/v3/ticker/24hr", {})
        out = []
        for d in data:
            s = d["symbol"]
            if not s.endswith("USDT"):
                continue
            base = s[:-4]
            if base in STABLE_BASES:
                continue
            out.append((s, float(d["quoteVolume"])))
        out.sort(key=lambda x: -x[1])
        syms = [s for s, _ in out[:topn]]
        if syms:
            return syms
    except Exception:
        pass
    return FALLBACK_COINS[:topn]


# ---------------------------------------------------------------------
# INDICATORS
# ---------------------------------------------------------------------
def atr_rma(h, l, c, n=14):
    pc = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    out = np.full(len(c), np.nan)
    if len(c) < n:
        return out
    out[n - 1] = tr[:n].mean()
    a = 1.0 / n
    for i in range(n, len(c)):
        out[i] = out[i - 1] * (1 - a) + tr[i] * a
    return out


def ema(a, n):
    return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def shift(a, k):
    n = len(a)
    out = np.zeros(n, bool) if a.dtype == bool else np.full(n, np.nan)
    if k < n:
        out[k:] = a[: n - k]
    return out


def win_any(x, sp):
    cs = np.cumsum(x.astype(np.int32))
    out = cs.copy()
    if sp < len(x):
        out[sp:] = cs[sp:] - cs[:-sp]
    return out > 0


# ---------------------------------------------------------------------
# TRENDLINE ZONE (port of the Pine logic)
# ---------------------------------------------------------------------
def find_pivots(a, s, is_high):
    n = len(a)
    if n < 2 * s + 1:
        return np.array([], int), np.array([])
    w = swv(a, 2 * s + 1)
    ctr, left, right = w[:, s], w[:, :s], w[:, s + 1:]
    if is_high:
        ok = (ctr > left.max(1)) & (ctr >= right.max(1))
    else:
        ok = (ctr < left.min(1)) & (ctr <= right.min(1))
    idx = np.nonzero(ok)[0] + s
    return idx, a[idx]


def best_zone(px, py, cur, close, thr, is_res, p):
    """Line through 2 pivots with >= min_touches pivots inside +-thr, not broken
    by older pivots, closest to current price. Returns (x1, y1, slope, up, dn)."""
    sel = px >= cur - p.back_bars
    x = px[sel].astype(float)
    y = py[sel]
    m = len(x)
    if m < p.min_touches:
        return None
    ii, jj = np.triu_indices(m, 1)
    dx = x[jj] - x[ii]
    ok = dx > 0
    ii, jj, dx = ii[ok], jj[ok], dx[ok]
    if len(ii) == 0:
        return None
    slope = (y[jj] - y[ii]) / dx
    exp = y[ii][:, None] + slope[:, None] * (x[None, :] - x[ii][:, None])
    diff = y[None, :] - exp
    after = np.arange(m)[None, :] >= ii[:, None]
    touch = (np.abs(diff) <= thr) & after
    touches = touch.sum(1)
    stab = (x < cur - 5)[None, :]
    breach = ((diff > thr) if is_res else (diff < -thr)) & after & stab
    valid = (touches >= p.min_touches) & ~breach.any(1)
    if not valid.any():
        return None
    proj = y[ii] + slope * (cur - x[ii])
    dist = np.where(valid, np.abs(proj - close), np.inf)
    b = int(np.argmin(dist))
    d = np.where(touch[b], diff[b], 0.0)
    return (x[ii[b]], y[ii[b]], slope[b], max(d.max(), 0.0), min(d.min(), 0.0))


def zone_bounds(params, idx):
    x1, y1, sl, up, dn = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4]
    base = y1 + sl * (idx - x1)
    return base + up, base + dn


def compute_zones(h, l, c, atr, p):
    """Walk-forward: at every bar only pivots already confirmed are used (no look-ahead)."""
    n, s = len(c), p.pivot_strength
    srcH = c if p.pivot_src == "Close" else h
    srcL = c if p.pivot_src == "Close" else l
    ph_i, ph_v = find_pivots(srcH, s, True)
    pl_i, pl_v = find_pivots(srcL, s, False)
    conf_h, conf_l = ph_i + s, pl_i + s
    ev = np.unique(np.r_[conf_h, conf_l, np.arange(0, n, p.refresh), n - 1]).astype(int)
    ev = ev[(ev >= 0) & (ev < n)]
    res = np.full((n, 5), np.nan)
    sup = np.full((n, 5), np.nan)
    for k, e in enumerate(ev):
        nxt = ev[k + 1] if k + 1 < len(ev) else n
        a = atr[e]
        if np.isnan(a):
            continue
        thr = a * p.atr_mult
        m = conf_h <= e
        z = best_zone(ph_i[m], ph_v[m], e, c[e], thr, True, p)
        if z is not None:
            res[e:nxt] = z
        m = conf_l <= e
        z = best_zone(pl_i[m], pl_v[m], e, c[e], thr, False, p)
        if z is not None:
            sup[e:nxt] = z
    idx = np.arange(n)
    top_r, bot_r = zone_bounds(res, idx)
    top_s, bot_s = zone_bounds(sup, idx)
    return dict(top_r=top_r, bot_r=bot_r, top_s=top_s, bot_s=bot_s, res=res, sup=sup)


# ---------------------------------------------------------------------
# CANDLESTICK PATTERNS  ->  {name: (bull_mask, bear_mask, span, must_clear_zone)}
# ---------------------------------------------------------------------
def candle_patterns(o, h, l, c, atr, p):
    n = len(c)
    body = np.abs(c - o)
    rng = h - l
    up_w = h - np.maximum(o, c)
    lo_w = np.minimum(o, c) - l
    bull, bear = c > o, c < o
    avg_body = pd.Series(body).rolling(14, min_periods=5).mean().to_numpy()
    avg_rng = pd.Series(rng).rolling(14, min_periods=5).mean().to_numpy()
    pats = {}

    # 1) Engulfing (optionally wait for confirmation candle)
    eb = shift(bear, 1) & bull & (o <= shift(c, 1)) & (c >= shift(o, 1)) & (body > shift(body, 1))
    es = shift(bull, 1) & bear & (o >= shift(c, 1)) & (c <= shift(o, 1)) & (body > shift(body, 1))
    if p.engulf_confirm:
        pats["Engulfing"] = (shift(eb, 1) & bull & (c > shift(c, 1)),
                             shift(es, 1) & bear & (c < shift(c, 1)), 3, False)
    else:
        pats["Engulfing"] = (eb, es, 2, False)

    # 2) Pin bar (hammer / shooting star)
    big = rng >= 0.7 * atr
    pin_b = big & (lo_w >= 0.6 * rng) & (up_w <= 0.2 * rng) & (body <= 0.35 * rng)
    pin_s = big & (up_w >= 0.6 * rng) & (lo_w <= 0.2 * rng) & (body <= 0.35 * rng)
    pats["Pin Bar"] = (pin_b, pin_s, 1, False)

    # 3) Momentum candle (big body, tiny wicks) that clears the zone
    mom = (body >= 0.7 * rng) & (body >= 1.5 * avg_body)
    pats["Momentum"] = (mom & bull, mom & bear, 3, True)

    # 4) Inside bar breakout of the mother bar
    mh, ml = shift(h, 2), shift(l, 2)
    inside = (shift(h, 1) <= mh) & (shift(l, 1) >= ml) & (shift(rng, 2) >= 1.2 * shift(avg_rng, 2))
    pats["Inside Bar"] = (inside & (c > mh), inside & (c < ml), 3, True)

    # 5) Three methods (rising / falling)
    mid_up = np.ones(n, bool)
    mid_dn = np.ones(n, bool)
    for k in (1, 2, 3):
        inr = (shift(h, k) <= shift(h, 4)) & (shift(l, k) >= shift(l, 4)) & (shift(body, k) < shift(body, 4))
        mid_up &= inr
        mid_dn &= inr
    first_up = shift(bull, 4) & (shift(body, 4) >= 1.2 * shift(avg_body, 4))
    first_dn = shift(bear, 4) & (shift(body, 4) >= 1.2 * shift(avg_body, 4))
    pats["Three Methods"] = (first_up & mid_up & bull & (c > shift(c, 4)),
                             first_dn & mid_dn & bear & (c < shift(c, 4)), 5, False)

    # 6) Morning / Evening star
    mid_o = (shift(o, 2) + shift(c, 2)) / 2
    morning = shift(bear, 2) & (shift(body, 2) >= avg_body * 0.0 + shift(avg_body, 2)) & \
        (shift(body, 1) <= 0.35 * shift(body, 2)) & bull & (c > mid_o)
    evening = shift(bull, 2) & (shift(body, 2) >= shift(avg_body, 2)) & \
        (shift(body, 1) <= 0.35 * shift(body, 2)) & bear & (c < mid_o)
    pats["Morning/Evening Star"] = (morning, evening, 3, False)

    # 7) Three white soldiers / black crows
    strong = (body >= 0.5 * rng)
    s1, s2 = shift(strong, 1), shift(strong, 2)
    sol = bull & shift(bull, 1) & shift(bull, 2) & strong & s1 & s2 & \
        (c > shift(c, 1)) & (shift(c, 1) > shift(c, 2)) & \
        (o > shift(o, 1)) & (o < shift(c, 1)) & (shift(o, 1) > shift(o, 2)) & (shift(o, 1) < shift(c, 2))
    cro = bear & shift(bear, 1) & shift(bear, 2) & strong & s1 & s2 & \
        (c < shift(c, 1)) & (shift(c, 1) < shift(c, 2)) & \
        (o < shift(o, 1)) & (o > shift(c, 1)) & (shift(o, 1) < shift(o, 2)) & (shift(o, 1) > shift(c, 2))
    pats["Soldiers/Crows"] = (sol, cro, 3, False)
    return pats


# ---------------------------------------------------------------------
# SIGNALS  (trendline zone touch + candlestick pattern)
# ---------------------------------------------------------------------
SIG_COLS = ["bar", "side", "pattern", "sl_raw", "atr", "zone_top", "zone_bot", "time", "close"]


def generate_signals(df, p):
    o, h, l, c = (df[k].to_numpy() for k in ("o", "h", "l", "c"))
    n = len(c)
    if n < 60:
        return pd.DataFrame(columns=SIG_COLS), None
    atr = atr_rma(h, l, c)
    e50, e200 = ema(c, 50), ema(c, 200)
    valid_ema = np.arange(n) >= 200
    z = compute_zones(h, l, c, atr, p)
    pats = candle_patterns(o, h, l, c, atr, p)
    tol = p.zone_tol_atr * atr
    atr_ok = ~np.isnan(atr)
    ct = df["ct"].to_numpy()
    win_lo, win_hi = {}, {}
    for sp in (1, 2, 3, 5):
        win_lo[sp] = pd.Series(l).rolling(sp, min_periods=1).min().to_numpy()
        win_hi[sp] = pd.Series(h).rolling(sp, min_periods=1).max().to_numpy()
    rows = []
    for side in p.sides:
        long_ = side == "LONG"
        top, bot = (z["top_s"], z["bot_s"]) if long_ else (z["top_r"], z["bot_r"])
        zone_ok = ~np.isnan(top)
        touch = zone_ok & (l <= top + tol) & (h >= bot - tol)
        if p.trend_filter:
            trend = ((e50 > e200) if long_ else (e50 < e200)) & valid_ema
        else:
            trend = np.ones(n, bool)
        chosen = np.full(n, -1)
        for pi, name in enumerate(PRIORITY):
            if name not in p.patterns:
                continue
            bull_m, bear_m, sp, clear = pats[name]
            m = bull_m if long_ else bear_m
            gate = (c > bot) if long_ else (c < top)
            if clear:
                gate = gate & ((c > top) if long_ else (c < bot))
            mask = m & win_any(touch, sp) & gate & trend & atr_ok & zone_ok
            chosen[mask & (chosen < 0)] = pi
        for t in np.nonzero(chosen >= 0)[0]:
            name = PRIORITY[chosen[t]]
            sp = pats[name][2]
            if long_:
                sl_raw = min(win_lo[sp][t], bot[t]) - p.sl_buffer_atr * atr[t]
            else:
                sl_raw = max(win_hi[sp][t], top[t]) + p.sl_buffer_atr * atr[t]
            rows.append(dict(bar=int(t), side=side, pattern=name, sl_raw=float(sl_raw), atr=float(atr[t]),
                             zone_top=float(top[t]), zone_bot=float(bot[t]),
                             time=pd.to_datetime(ct[t] + 1, unit="ms", utc=True).tz_convert(IST),
                             close=float(c[t])))
    sig = pd.DataFrame(rows, columns=SIG_COLS).sort_values("bar").reset_index(drop=True)
    return sig, dict(n=n, atr=atr, res=z["res"], sup=z["sup"])


def build_levels(side, entry, sl_raw, atr, p):
    d = 1 if side == "LONG" else -1
    risk = (entry - sl_raw) * d
    if not np.isfinite(risk) or risk <= 0:
        return None
    risk = max(risk, p.min_risk_atr * atr)
    if risk > p.max_risk_atr * atr:
        return None
    if (p.rr * risk / entry) < p.min_tp_fee_mult * 2 * p.fee:
        return None
    return dict(sl=entry - d * risk, tp=entry + d * p.rr * risk, risk=risk)


# ---------------------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------------------
def backtest_df(df, symbol, tf, p, capital, size_pct):
    sig, _ = generate_signals(df, p)
    o, h, l, c = (df[k].to_numpy() for k in ("o", "h", "l", "c"))
    t_open = df["t"].to_numpy()
    t_close = df["ct"].to_numpy()
    n = len(c)
    trades, equity, last_exit = [], capital, -1
    eq_curve = [capital]
    for r in sig.itertuples():
        a = r.bar + 1
        if a >= n or a <= last_exit:
            continue
        entry = o[a]
        lv = build_levels(r.side, entry, r.sl_raw, r.atr, p)
        if lv is None:
            continue
        sl, tp = lv["sl"], lv["tp"]
        end = min(n - 1, a + p.max_hold)
        lo, hi = l[a:end + 1], h[a:end + 1]
        if r.side == "LONG":
            sl_hit, tp_hit = lo <= sl, hi >= tp
        else:
            sl_hit, tp_hit = hi >= sl, lo <= tp
        i_sl = int(np.argmax(sl_hit)) if sl_hit.any() else 10 ** 9
        i_tp = int(np.argmax(tp_hit)) if tp_hit.any() else 10 ** 9
        if i_sl == 10 ** 9 and i_tp == 10 ** 9:
            if a + p.max_hold > n - 1:
                continue                      # trade still running -> not counted
            x, exit_px, reason = end, c[end], "TIMEOUT"
        elif i_sl <= i_tp:                    # SL first (conservative when both in one candle)
            x = a + i_sl
            exit_px = min(sl, o[x]) if r.side == "LONG" else max(sl, o[x])
            if x == a:
                exit_px = sl
            reason = "SL"
        else:
            x, exit_px, reason = a + i_tp, tp, "TP"
        d = 1 if r.side == "LONG" else -1
        notional = size_pct * equity
        gross = notional * (exit_px / entry - 1) * d
        fees = notional * p.fee * (1 + exit_px / entry)
        pnl = gross - fees
        equity += pnl
        eq_curve.append(equity)
        last_exit = x
        trades.append(dict(
            Coin=symbol.replace("USDT", ""), TF=tf, Side=r.side, Pattern=r.pattern,
            entry_time=pd.to_datetime(t_open[a], unit="ms", utc=True).tz_convert(IST),
            exit_time=pd.to_datetime(t_close[x] + 1, unit="ms", utc=True).tz_convert(IST),
            hold_min=round((t_close[x] + 1 - t_open[a]) / 60000, 1),
            entry=entry, exit=exit_px, sl=sl, tp=tp, reason=reason,
            success=bool(pnl > 0), pnl=pnl, net_pct=pnl / notional * 100,
            R=(exit_px - entry) * d / lv["risk"]))
    tr = pd.DataFrame(trades)
    eq = np.array(eq_curve)
    peak = np.maximum.accumulate(eq)
    stat = dict(Coin=symbol.replace("USDT", ""), TF=tf, Candles=n,
                From=pd.to_datetime(t_open[0], unit="ms", utc=True).tz_convert(IST),
                To=pd.to_datetime(t_close[-1] + 1, unit="ms", utc=True).tz_convert(IST),
                final_equity=float(equity), max_dd=float(((eq - peak) / peak).min() * 100))
    return tr, stat


def pf(s):
    g, ls = s[s > 0].sum(), -s[s < 0].sum()
    return round(g / ls, 2) if ls > 0 else np.nan


def summarize(df, by):
    rows = []
    for key, g in df.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        n, w = len(g), int(g.success.sum())
        r = dict(zip(by, key))
        r.update({"Trades": n, "Wins": w, "Losses": n - w, "Win %": round(100 * w / n, 1),
                  "PnL": round(g.pnl.sum(), 0), "Avg PnL": round(g.pnl.mean(), 0),
                  "Profit Factor": pf(g.pnl)})
        rows.append(r)
    return pd.DataFrame(rows).sort_values("PnL", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------
# LIVE SCAN
# ---------------------------------------------------------------------
def fmt(x):
    return f"{x:.6g}"


def analyze_live(df, symbol, tf, p, recent_k, store):
    now_ms = time.time() * 1000
    closed = df[df.ct < now_ms].reset_index(drop=True)
    live = df[df.ct >= now_ms]
    if len(closed) < 250:
        return [], None
    key = (symbol, tf)
    stamp = (float(closed.ct.iloc[-1]), repr(p))
    hit = store["a"].get(key)
    if hit and hit[0] == stamp:
        sig, ctx = hit[1], hit[2]
    else:
        sig, ctx = generate_signals(closed, p)
        store["a"][key] = (stamp, sig, ctx)
    n = ctx["n"]
    out = []
    for r in sig[sig.bar >= n - recent_k].itertuples():
        lv = build_levels(r.side, r.close, r.sl_raw, r.atr, p)
        if lv is None:
            continue
        out.append(dict(Coin=symbol.replace("USDT", ""), TF=tf, Side=r.side, Pattern=r.pattern,
                        ago=n - 1 - r.bar, time=r.time, Entry=fmt(r.close), SL=fmt(lv["sl"]), TP=fmt(lv["tp"]),
                        Risk=f"{lv['risk'] / r.close * 100:.2f}%",
                        Zone=f"{fmt(r.zone_bot)} - {fmt(r.zone_top)}"))
    watch = None
    if len(live):
        lh, ll, lc = float(live.h.iloc[-1]), float(live.l.iloc[-1]), float(live.c.iloc[-1])
        atr_last = ctx["atr"][-1]
        cands = []
        for label, arr in (("SUPPORT", ctx["sup"]), ("RESISTANCE", ctx["res"])):
            x1, y1, sl_, up, dn = arr[-1]
            if np.isnan(x1):
                continue
            base = y1 + sl_ * (n - x1)
            top, bot = base + up, base + dn
            tol = p.zone_tol_atr * atr_last
            touching = (ll <= top + tol) and (lh >= bot - tol)
            dist = 0.0 if touching else min(abs(lc - top), abs(lc - bot))
            if touching or dist <= 0.5 * atr_last:
                cands.append(dict(Coin=symbol.replace("USDT", ""), TF=tf, Zone=label,
                                  Status="TOUCHING" if touching else "NEAR", Price=fmt(lc),
                                  Top=fmt(top), Bottom=fmt(bot), dist=dist / lc * 100))
        if cands:
            watch = cands
    return out, watch


def live_scan(store, symbols, tfs, p, recent_k):
    def work(job):
        s, tf = job
        key = (s, tf)
        now = time.time()
        hit = store["k"].get(key)
        ttl = min(TF_SECONDS[tf] / 6, 60)
        if hit and now - hit[0] < ttl:
            df = hit[1]
        else:
            df = fetch_klines(s, tf, 499)
            store["k"][key] = (now, df)
        return analyze_live(df, s, tf, p, recent_k, store)

    sigs, watch, errs = [], [], 0
    jobs = [(s, tf) for s in symbols for tf in tfs]
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for f in as_completed(futs):
            try:
                a, w = f.result()
                sigs += a
                if w:
                    watch += w
            except Exception:
                errs += 1
    return sigs, watch, errs, len(jobs)


# ---------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------
@st.cache_resource
def get_store():
    return {"k": {}, "a": {}}


def settings_ui():
    with st.expander("⚙️ Settings (defaults are already set)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            tfs = st.multiselect("Timeframes", ALL_TFS, DEFAULT_TFS)
            sides = st.multiselect("Sides", ["LONG", "SHORT"], ["LONG", "SHORT"])
            pats = st.multiselect("Candlestick patterns", PATTERNS, PATTERNS)
            trend = st.checkbox("Trend filter (EMA50 vs EMA200)", True)
            conf = st.checkbox("Engulfing: wait for confirmation candle", True)
        with c2:
            rr = st.slider("Risk : Reward", 1.0, 4.0, 2.0, 0.5)
            strength = st.slider("Pivot strength", 5, 15, 10)
            touches = st.slider("Min pivot confirmation", 2, 8, 3)
            back = st.slider("Bars to apply", 50, 500, 300, 10)
            max_hold = st.slider("Max candles in trade", 10, 200, 60, 5)
        c3, c4 = st.columns(2)
        with c3:
            fee = st.number_input("Fee per side %", 0.0, 1.0, 0.1, 0.01, format="%.2f") / 100
            capital = st.number_input("Capital per coin/TF", 1000, 100000000, 100000, 10000)
        with c4:
            size = st.slider("Capital used per trade %", 10, 100, 50, 5) / 100
            topn = st.slider("Live: top coins by volume", 10, 100, 40, 5)
            refresh = st.slider("Live refresh (seconds)", 20, 300, 60, 10)
    p = Params(back_bars=back, pivot_strength=strength, min_touches=touches,
               patterns=tuple(pats), engulf_confirm=conf, trend_filter=trend,
               sides=tuple(sides), rr=rr, max_hold=max_hold, fee=fee)
    cfg = dict(tfs=tfs or DEFAULT_TFS, topn=topn, refresh=refresh, capital=capital, size=size)
    return p, cfg


def live_tab(p, cfg):
    store = get_store()

    def render():
        t0 = time.time()
        symbols = top_symbols(cfg["topn"])
        sigs, watch, errs, njobs = live_scan(store, symbols, cfg["tfs"], p, recent_k=3)
        st.caption(f"Last scan {pd.Timestamp.now(tz=IST).strftime('%H:%M:%S')} IST · {len(symbols)} coins × "
                   f"{len(cfg['tfs'])} TFs · {errs} fetch errors · {time.time() - t0:.0f}s")
        st.subheader("🎯 Signals (zone touch + candlestick pattern)")
        if sigs:
            d = pd.DataFrame(sigs).sort_values(["ago", "time"], ascending=[True, False])
            seen = st.session_state.setdefault("seen", None)
            keys = {(r.Coin, r.TF, r.Side, str(r.time)) for r in d.itertuples()}
            fresh = {(r.Coin, r.TF, r.Side, str(r.time)): r for r in d.itertuples() if r.ago == 0}
            if seen is not None:
                for k, r in fresh.items():
                    if k not in seen:
                        st.toast(f"{r.Side} {r.Coin} {r.TF} · {r.Pattern}")
            st.session_state["seen"] = keys
            d["Fresh"] = np.where(d.ago == 0, "🔥 NEW", d.ago.astype(str) + " candle ago")
            d["Signal time (IST)"] = d.time.dt.strftime("%d-%b %H:%M")
            d["Side"] = np.where(d.Side == "LONG", "🟢 LONG", "🔴 SHORT")
            st.dataframe(d[["Fresh", "Coin", "TF", "Side", "Pattern", "Signal time (IST)", "Entry", "SL", "TP",
                            "Risk", "Zone"]], hide_index=True)
        else:
            st.info("No confirmed signal right now. Waiting for a pattern at the trendline zone.")
        st.subheader("👀 Zone watch (price touching / near the zone, waiting for a pattern)")
        if watch:
            w = pd.DataFrame(watch).sort_values(["Status", "dist"], ascending=[False, True])
            w["dist"] = w.dist.round(2).astype(str) + "%"
            st.dataframe(w.rename(columns={"dist": "Distance"}), hide_index=True)
        else:
            st.write("Nothing at a zone right now.")

    if hasattr(st, "fragment"):
        st.fragment(render, run_every=cfg["refresh"])()
    else:
        render()


def backtest_tab(p, cfg):
    c1, c2, c3 = st.columns(3)
    with c1:
        ncoins = st.slider("Coins (top by volume)", 5, 60, 20, 5, key="bt_n")
    with c2:
        candles = st.selectbox("Candles per coin/TF", [1000, 2000, 3000, 5000, 10000], 2)
    with c3:
        st.write("")
        run = st.button("▶ Run Backtest", type="primary")
    if run:
        symbols = top_symbols(ncoins)
        jobs = [(s, tf) for s in symbols for tf in cfg["tfs"]]
        bar = st.progress(0.0, text="Starting...")
        trades, stats, errs = [], [], 0

        def work(job):
            s, tf = job
            df = fetch_klines(s, tf, candles + 1)
            df = df[df.ct < time.time() * 1000].reset_index(drop=True)
            return backtest_df(df, s, tf, p, cfg["capital"], cfg["size"])

        done = 0
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(work, j) for j in jobs]
            for f in as_completed(futs):
                done += 1
                try:
                    tr, stt = f.result()
                    stats.append(stt)
                    if len(tr):
                        trades.append(tr)
                except Exception:
                    errs += 1
                bar.progress(done / len(jobs), text=f"{done}/{len(jobs)} coin×TF done")
        bar.empty()
        st.session_state["bt"] = (pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(),
                                  pd.DataFrame(stats), errs)
    if "bt" not in st.session_state:
        st.info("Press **Run Backtest**. Every coin × timeframe starts with its own capital; entry is on the "
                "next candle's open, SL/TP checked candle by candle (SL first if both hit in one candle), "
                "fees included. Win = real profit after fees.")
        return
    tr, stats, errs = st.session_state["bt"]
    if tr.empty:
        st.warning("No trades produced with these settings.")
        return
    if errs:
        st.caption(f"{errs} coin/TF jobs failed to fetch.")

    n, w = len(tr), int(tr.success.sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Trades", n)
    m2.metric("Win rate", f"{100 * w / n:.1f}%")
    m3.metric("Total PnL", f"{tr.pnl.sum():,.0f}")
    m4.metric("Profit factor", pf(tr.pnl))

    st.subheader("⏱️ Timeframe-wise")
    tf_order = {t: i for i, t in enumerate(ALL_TFS)}
    s_tf = summarize(tr, ["TF"])
    st.dataframe(s_tf.sort_values("TF", key=lambda s: s.map(tf_order)), hide_index=True)
    st.subheader("🕯️ Pattern-wise (which candlestick works best)")
    st.dataframe(summarize(tr, ["Pattern"]), hide_index=True)
    st.subheader("↕️ Side-wise")
    st.dataframe(summarize(tr, ["Side"]), hide_index=True)
    st.subheader("🪙 Coin-wise")
    st.dataframe(summarize(tr, ["Coin"]), hide_index=True)

    st.subheader("🏆 Coin × Timeframe (best win rate first, min 5 trades)")
    ct_ = summarize(tr, ["Coin", "TF"])
    ct_ = ct_.merge(stats[["Coin", "TF", "max_dd"]], on=["Coin", "TF"], how="left")
    ct_["Max DD %"] = ct_.max_dd.round(1)
    ct_ = ct_.drop(columns="max_dd")
    st.dataframe(ct_[ct_.Trades >= 5].sort_values(["Win %", "PnL"], ascending=False), hide_index=True)

    st.subheader("🕐 Entry hour (IST) win rate")
    hh = tr.assign(Hour=tr.entry_time.dt.hour)
    hs = summarize(hh, ["Hour"]).sort_values("Hour")
    st.bar_chart(hs.set_index("Hour")["Win %"])

    st.subheader("📈 Equity (cumulative PnL by exit time)")
    eq = tr.sort_values("exit_time")
    st.line_chart(pd.Series(eq.pnl.cumsum().to_numpy(), index=eq.exit_time.dt.tz_localize(None), name="PnL"))

    st.subheader("🧾 All trades")
    f1, f2, f3 = st.columns(3)
    tf_sel = f1.multiselect("TF", sorted(tr.TF.unique(), key=lambda x: tf_order[x]), key="f_tf")
    coin_sel = f2.multiselect("Coin", sorted(tr.Coin.unique()), key="f_coin")
    res_sel = f3.selectbox("Result", ["All", "Success", "Fail"], key="f_res")
    v = tr.copy()
    if tf_sel:
        v = v[v.TF.isin(tf_sel)]
    if coin_sel:
        v = v[v.Coin.isin(coin_sel)]
    if res_sel == "Success":
        v = v[v.success]
    elif res_sel == "Fail":
        v = v[~v.success]
    v = v.sort_values("entry_time", ascending=False)
    show = pd.DataFrame({
        "Result": np.where(v.success, "✅ Success", "❌ Fail") + " (" + v.reason + ")",
        "Coin": v.Coin, "TF": v.TF, "Side": v.Side, "Pattern": v.Pattern,
        "Entry time (IST)": v.entry_time.dt.strftime("%d-%b %H:%M"),
        "Exit time (IST)": v.exit_time.dt.strftime("%d-%b %H:%M"),
        "Held (min)": v.hold_min,
        "Entry": v.entry.map(fmt), "Exit": v.exit.map(fmt), "SL": v.sl.map(fmt), "TP": v.tp.map(fmt),
        "PnL": v.pnl.round(0), "Net %": v.net_pct.round(2), "R": v.R.round(2)})
    st.dataframe(show, hide_index=True)
    st.download_button("⬇️ Download trades CSV", v.to_csv(index=False).encode(), "trades.csv", "text/csv")


def main():
    st.set_page_config(page_title="Kishore Trendline Scanner", page_icon="📐", layout="wide")
    st.title("📐 Kishore Trendline + Candlestick Scanner")
    st.caption("Trendline zone + candlestick confirmation · Binance Spot · 1m 5m 15m 30m 1h 4h 1d")
    p, cfg = settings_ui()
    if not p.patterns or not p.sides:
        st.warning("Select at least one pattern and one side.")
        return
    t1, t2 = st.tabs(["🔴 Live Scanner", "📊 Backtest"])
    with t1:
        live_tab(p, cfg)
    with t2:
        backtest_tab(p, cfg)


if __name__ == "__main__":
    main()
