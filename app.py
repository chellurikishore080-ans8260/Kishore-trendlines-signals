# =====================================================================
#  Kishore Trendline Zone Scanner - RESISTANCE SHORTS ONLY (with Backtest)
#  Created by Kishore
#
#  Logic = the ORIGINAL Trendline Zone logic, nothing else:
#    - Pine "Trendlines" zone: pivot highs -> best line with min touches
#      -> ATR-wide RESISTANCE zone
#    - A candle wick touching the resistance zone = SHORT signal
#    - NO candlestick pattern needed (no Engulfing / Pin Bar / etc.)
#    - NO support / long side, NO trend filter
#  Data: Binance Spot (data-api.binance.vision first, other hosts fallback)
#  Live scanner auto-refreshes, Backtest shows every trade + PnL + win/fail
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
IST = "Asia/Kolkata"
STABILITY = 5            # stability buffer in bars (same as the Pine script)
MIN_BARS = 120           # minimum closed candles needed to build a zone
STABLE_BASES = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "EUR", "AEUR", "USDE",
                "XUSD", "PYUSD", "UST", "USD1", "EURI", "WBTC"}
FALLBACK_COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
                  "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT", "TRXUSDT", "NEARUSDT",
                  "SUIUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT", "ATOMUSDT"]


@dataclass(frozen=True)
class Params:
    # --- trendline zone (same as the original script / Pine defaults) ---
    back_bars: int = 300
    pivot_src: str = "High/Low"
    pivot_strength: int = 10
    min_touches: int = 3
    atr_mult: float = 0.5
    refresh: int = 8                 # backtest: zone re-calc every N bars (and on every new pivot)
    pierce: bool = False             # also count a wick spiking through the whole zone
    cooldown: int = 5                # FRESH touch = no touch in the previous N candles
    # --- trade management (SHORT only) ---
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
# INDICATORS / HELPERS
# ---------------------------------------------------------------------
def atr_rma(h, l, c, n=14):
    """Wilder ATR (same as Pine ta.atr)."""
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


def shift(a, k):
    n = len(a)
    out = np.zeros(n, bool) if a.dtype == bool else np.full(n, np.nan)
    if k < n:
        out[k:] = a[: n - k]
    return out


def win_any(x, sp):
    """True where x was True at least once in the last `sp` bars (incl. current)."""
    cs = np.cumsum(x.astype(np.int32))
    out = cs.copy()
    if sp < len(x):
        out[sp:] = cs[sp:] - cs[:-sp]
    return out > 0


# ---------------------------------------------------------------------
# RESISTANCE TRENDLINE ZONE  (the ORIGINAL logic - port of the Pine script)
# ---------------------------------------------------------------------
def find_pivot_highs(a, s):
    """Pivot highs with s bars on each side (same tie rule as the original script)."""
    n = len(a)
    if n < 2 * s + 1:
        return np.array([], int), np.array([])
    w = swv(a, 2 * s + 1)
    ctr, left, right = w[:, s], w[:, :s], w[:, s + 1:]
    ok = (ctr >= left.max(1)) & (ctr > right.max(1))
    idx = np.nonzero(ok)[0] + s
    return idx, a[idx]


def best_zone(px, py, cur, close, thr, p):
    """Line through 2 pivot highs with >= min_touches pivots inside +-thr, not broken
    by older pivots, closest to current price. Returns (x1, y1, slope, up, dn)."""
    if thr <= 0 or not np.isfinite(thr):
        return None
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
    stab = (x < cur - STABILITY)[None, :]
    breach = (diff > thr) & after & stab            # resistance broken by an older pivot
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


def compute_res_zones(h, l, c, atr, p):
    """Walk-forward: at every bar only pivots already confirmed are used (no look-ahead).
    Row e = zone params computed with data up to and including bar e."""
    n, s = len(c), p.pivot_strength
    src = c if p.pivot_src == "Close" else h
    ph_i, ph_v = find_pivot_highs(src, s)
    conf = ph_i + s
    ev = np.unique(np.r_[conf, np.arange(0, n, p.refresh), n - 1]).astype(int)
    ev = ev[(ev >= 0) & (ev < n)]
    res = np.full((n, 5), np.nan)
    for k, e in enumerate(ev):
        nxt = ev[k + 1] if k + 1 < len(ev) else n
        a = atr[e]
        if np.isnan(a):
            continue
        m = conf <= e
        z = best_zone(ph_i[m], ph_v[m], e, c[e], a * p.atr_mult, p)
        if z is not None:
            res[e:nxt] = z
    return res


def touch_mask(h, l, top, bot, pierce):
    """Original wick-touch rule: the high or the low of the candle is inside the zone."""
    t = ((h >= bot) & (h <= top)) | ((l <= top) & (l >= bot))
    if pierce:                                       # wick spikes straight through the whole zone
        t = t | ((h > top) & (l < bot))
    return t


# ---------------------------------------------------------------------
# SIGNALS  (resistance wick touch -> SHORT, any candle, no pattern needed)
# ---------------------------------------------------------------------
SIG_COLS = ["bar", "sl_raw", "atr", "zone_top", "zone_bot", "time", "close"]


def generate_signals(df, p):
    h, l, c = (df[k].to_numpy() for k in ("h", "l", "c"))
    n = len(c)
    if n < MIN_BARS:
        return pd.DataFrame(columns=SIG_COLS), None
    atr = atr_rma(h, l, c)
    res = compute_res_zones(h, l, c, atr, p)
    # the zone a candle is compared with was known at the END of the previous candle
    lag = np.vstack([np.full((1, 5), np.nan), res[:-1]])
    top, bot = zone_bounds(lag, np.arange(n))
    zone_ok = ~np.isnan(top) & ~np.isnan(atr)
    touch = zone_ok & touch_mask(h, l, top, bot, p.pierce)
    fresh = touch & ~win_any(shift(touch, 1), p.cooldown)     # first touch only
    ct = df["ct"].to_numpy()
    rows = []
    for t in np.nonzero(fresh)[0]:
        sl_raw = max(h[t], top[t]) + p.sl_buffer_atr * atr[t]
        rows.append(dict(bar=int(t), sl_raw=float(sl_raw), atr=float(atr[t]),
                         zone_top=float(top[t]), zone_bot=float(bot[t]),
                         time=pd.to_datetime(ct[t] + 1, unit="ms", utc=True).tz_convert(IST),
                         close=float(c[t])))
    sig = pd.DataFrame(rows, columns=SIG_COLS)
    return sig, dict(n=n, atr=atr, res=res, touch=touch)


def build_levels(entry, sl_raw, atr, p):
    """SHORT: SL above the zone, TP below by RR."""
    risk = sl_raw - entry
    if not np.isfinite(risk) or risk <= 0:
        return None
    risk = max(risk, p.min_risk_atr * atr)
    if risk > p.max_risk_atr * atr:
        return None
    if (p.rr * risk / entry) < p.min_tp_fee_mult * 2 * p.fee:
        return None
    return dict(sl=entry + risk, tp=entry - p.rr * risk, risk=risk)


# ---------------------------------------------------------------------
# BACKTEST  (shorts only)
# ---------------------------------------------------------------------
def backtest_df(df, symbol, tf, p, capital, size_pct):
    sig, _ = generate_signals(df, p)
    h, l, c, o = (df[k].to_numpy() for k in ("h", "l", "c", "o"))
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
        lv = build_levels(entry, r.sl_raw, r.atr, p)
        if lv is None:
            continue
        sl, tp = lv["sl"], lv["tp"]
        end = min(n - 1, a + p.max_hold)
        lo, hi = l[a:end + 1], h[a:end + 1]
        sl_hit, tp_hit = hi >= sl, lo <= tp
        i_sl = int(np.argmax(sl_hit)) if sl_hit.any() else 10 ** 9
        i_tp = int(np.argmax(tp_hit)) if tp_hit.any() else 10 ** 9
        if i_sl == 10 ** 9 and i_tp == 10 ** 9:
            if a + p.max_hold > n - 1:
                continue                      # trade still running -> not counted
            x, exit_px, reason = end, c[end], "TIMEOUT"
        elif i_sl <= i_tp:                    # SL first (conservative when both in one candle)
            x = a + i_sl
            exit_px = max(sl, o[x])           # gap above SL fills at the open
            if x == a:
                exit_px = sl
            reason = "SL"
        else:
            x, exit_px, reason = a + i_tp, tp, "TP"
        notional = size_pct * equity
        gross = notional * (1 - exit_px / entry)          # short
        fees = notional * p.fee * (1 + exit_px / entry)
        pnl = gross - fees
        equity += pnl
        eq_curve.append(equity)
        last_exit = x
        trades.append(dict(
            Coin=symbol.replace("USDT", ""), TF=tf, Side="SHORT",
            entry_time=pd.to_datetime(t_open[a], unit="ms", utc=True).tz_convert(IST),
            exit_time=pd.to_datetime(t_close[x] + 1, unit="ms", utc=True).tz_convert(IST),
            hold_min=round((t_close[x] + 1 - t_open[a]) / 60000, 1),
            zone_bot=r.zone_bot, zone_top=r.zone_top,
            entry=entry, exit=exit_px, sl=sl, tp=tp, reason=reason,
            success=bool(pnl > 0), pnl=pnl, net_pct=pnl / notional * 100,
            R=(entry - exit_px) / lv["risk"]))
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
    if len(closed) < MIN_BARS:
        return [], None
    key = (symbol, tf)
    stamp = (float(closed.ct.iloc[-1]), repr(p))
    hit = store["a"].get(key)
    if hit and hit[0] == stamp:
        sig, ctx = hit[1], hit[2]
    else:
        sig, ctx = generate_signals(closed, p)
        store["a"][key] = (stamp, sig, ctx)
    if ctx is None:
        return [], None
    n = ctx["n"]
    coin = symbol.replace("USDT", "")
    out = []
    # 1) fresh touches on the last few CLOSED candles
    for r in sig[sig.bar >= n - recent_k].itertuples():
        lv = build_levels(r.close, r.sl_raw, r.atr, p)
        if lv is None:
            continue
        out.append(dict(Coin=coin, TF=tf, Side="SHORT", ago=n - 1 - r.bar, time=r.time,
                        Entry=fmt(r.close), SL=fmt(lv["sl"]), TP=fmt(lv["tp"]),
                        Risk=f"{lv['risk'] / r.close * 100:.2f}%",
                        Zone=f"{fmt(r.zone_bot)} - {fmt(r.zone_top)}"))
    # 2) the candle that is forming right now
    watch = None
    if len(live):
        lh, ll, lc = float(live.h.iloc[-1]), float(live.l.iloc[-1]), float(live.c.iloc[-1])
        atr_last = float(ctx["atr"][-1])
        x1, y1, sl_, up, dn = ctx["res"][-1]
        if not np.isnan(x1):
            base = y1 + sl_ * (n - x1)
            top, bot = base + up, base + dn
            touching = bool(touch_mask(np.array([lh]), np.array([ll]), top, bot, p.pierce)[0])
            is_fresh = not bool(ctx["touch"][-p.cooldown:].any())
            if touching and is_fresh:
                lv = build_levels(lc, max(lh, top) + p.sl_buffer_atr * atr_last, atr_last, p)
                if lv is not None:
                    t_live = pd.to_datetime(float(live.t.iloc[-1]), unit="ms", utc=True).tz_convert(IST)
                    out.append(dict(Coin=coin, TF=tf, Side="SHORT", ago=-1, time=t_live,
                                    Entry=fmt(lc), SL=fmt(lv["sl"]), TP=fmt(lv["tp"]),
                                    Risk=f"{lv['risk'] / lc * 100:.2f}%",
                                    Zone=f"{fmt(bot)} - {fmt(top)}"))
            dist = 0.0 if touching else (bot - lc if lc < bot else lc - top)
            if touching or dist <= 0.5 * atr_last:
                watch = [dict(Coin=coin, TF=tf, Status="TOUCHING" if touching else "NEAR",
                              Touch="FRESH" if is_fresh else "REPEAT", Price=fmt(lc),
                              Top=fmt(top), Bottom=fmt(bot), dist=dist / lc * 100)]
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
            src = st.selectbox("Pivot source", ["High/Low", "Close"])
            pierce = st.checkbox("Also count a wick that spikes through the whole zone", False)
            cooldown = st.slider("Fresh-touch cooldown (candles)", 1, 50, 5)
            rr = st.slider("Risk : Reward", 1.0, 4.0, 2.0, 0.5)
        with c2:
            strength = st.slider("Pivot strength", 5, 15, 10)
            touches = st.slider("Min pivot confirmation", 2, 8, 3)
            back = st.slider("Bars to apply", 50, 500, 300, 10)
            max_hold = st.slider("Max candles in trade", 10, 200, 60, 5)
            refresh_bt = st.slider("Backtest: zone re-calc every N candles", 1, 20, 8)
        c3, c4 = st.columns(2)
        with c3:
            fee = st.number_input("Fee per side %", 0.0, 1.0, 0.1, 0.01, format="%.2f") / 100
            capital = st.number_input("Capital per coin/TF", 1000, 100000000, 100000, 10000)
        with c4:
            size = st.slider("Capital used per trade %", 10, 100, 50, 5) / 100
            topn = st.slider("Live: top coins by volume", 10, 100, 40, 5)
            refresh = st.slider("Live refresh (seconds)", 20, 300, 60, 10)
    p = Params(back_bars=back, pivot_src=src, pivot_strength=strength, min_touches=touches,
               refresh=refresh_bt, pierce=pierce, cooldown=cooldown, rr=rr, max_hold=max_hold, fee=fee)
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
        st.subheader("🔴 SHORT signals (resistance zone touch)")
        if sigs:
            d = pd.DataFrame(sigs).sort_values(["ago", "time"], ascending=[True, False])
            seen = st.session_state.setdefault("seen", None)
            keys = {(r.Coin, r.TF, str(r.time)) for r in d.itertuples()}
            fresh = {(r.Coin, r.TF, str(r.time)): r for r in d.itertuples() if r.ago <= 0}
            if seen is not None:
                for k, r in fresh.items():
                    if k not in seen:
                        st.toast(f"SHORT {r.Coin} {r.TF} · resistance touch")
            st.session_state["seen"] = keys
            d["Fresh"] = np.where(d.ago < 0, "🔥 TOUCHING NOW",
                                  np.where(d.ago == 0, "🔥 NEW", d.ago.astype(str) + " candle ago"))
            d["Signal time (IST)"] = d.time.dt.strftime("%d-%b %H:%M")
            d["Side"] = "🔴 SHORT"
            st.dataframe(d[["Fresh", "Coin", "TF", "Side", "Signal time (IST)", "Entry", "SL", "TP",
                            "Risk", "Zone"]], hide_index=True)
        else:
            st.info("No resistance touch right now. Waiting for price to reach a resistance zone.")
        st.subheader("👀 Resistance watch (price touching / near the zone)")
        if watch:
            w = pd.DataFrame(watch).sort_values(["Status", "dist"], ascending=[False, True])
            w["dist"] = w.dist.round(2).astype(str) + "%"
            st.dataframe(w.rename(columns={"dist": "Distance"}), hide_index=True)
        else:
            st.write("Nothing at a resistance zone right now.")

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
        st.info("Press **Run Backtest** (SHORTS only). Every resistance-zone touch = one short. "
                "Every coin × timeframe starts with its own capital; entry is on the next candle's open, "
                "SL above the zone / TP by RR checked candle by candle (SL first if both hit in one candle), "
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
    m1.metric("Trades (all SHORT)", n)
    m2.metric("Win rate", f"{100 * w / n:.1f}%")
    m3.metric("Total PnL", f"{tr.pnl.sum():,.0f}")
    m4.metric("Profit factor", pf(tr.pnl))

    st.subheader("⏱️ Timeframe-wise")
    tf_order = {t: i for i, t in enumerate(ALL_TFS)}
    s_tf = summarize(tr, ["TF"])
    st.dataframe(s_tf.sort_values("TF", key=lambda s: s.map(tf_order)), hide_index=True)
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
        "Coin": v.Coin, "TF": v.TF, "Side": v.Side,
        "Entry time (IST)": v.entry_time.dt.strftime("%d-%b %H:%M"),
        "Exit time (IST)": v.exit_time.dt.strftime("%d-%b %H:%M"),
        "Held (min)": v.hold_min,
        "Zone": v.zone_bot.map(fmt) + " - " + v.zone_top.map(fmt),
        "Entry": v.entry.map(fmt), "Exit": v.exit.map(fmt), "SL": v.sl.map(fmt), "TP": v.tp.map(fmt),
        "PnL": v.pnl.round(0), "Net %": v.net_pct.round(2), "R": v.R.round(2)})
    st.dataframe(show, hide_index=True)
    st.download_button("⬇️ Download trades CSV", v.to_csv(index=False).encode(), "trades.csv", "text/csv")


def main():
    st.set_page_config(page_title="Kishore Trendline Scanner", page_icon="📐", layout="wide")
    st.title("📐 Kishore Trendline Zone Scanner — Resistance SHORTS")
    st.caption("Resistance trendline zone touch = SHORT · no candlestick pattern needed · Binance Spot · "
               "1m 5m 15m 30m 1h 4h 1d")
    p, cfg = settings_ui()
    t1, t2 = st.tabs(["🔴 Live Scanner", "📊 Backtest"])
    with t1:
        live_tab(p, cfg)
    with t2:
        backtest_tab(p, cfg)


if __name__ == "__main__":
    main()
