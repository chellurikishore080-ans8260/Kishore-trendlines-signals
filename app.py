"""
Trendline Zone Scanner - Binance Spot (all USDT pairs)
Author: Kishore

Scans every Binance Spot USDT pair on 5m / 15m / 30m / 1h / 4h, builds the
nearest support / resistance trendline zone for each coin and alerts the moment
a candle wick touches that zone. The screen refreshes itself - no manual scan.

Zone logic follows the open-source "Trendlines" Pine indicator (MPL-2.0),
re-implemented here in Python.
"""

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

# data-api.binance.vision is Binance's public market-data host (works from
# cloud servers that get blocked on api.binance.com). The rest are fallbacks.
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
KLINES = 400        # candles fetched per coin / timeframe

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
                # close to Binance's 6000/min limit: wait for the next minute
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
            if r.status_code == 451:  # region blocked on this host
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


def fetch_klines(base, symbol, tf, limit=KLINES, min_len=60):
    d = http_get(
        base, "/api/v3/klines", {"symbol": symbol, "interval": tf, "limit": limit}
    )
    if not d or len(d) < min_len:
        return None
    return {
        "t": np.array([x[0] for x in d], dtype=np.int64),
        "o": np.array([float(x[1]) for x in d]),
        "h": np.array([float(x[2]) for x in d]),
        "l": np.array([float(x[3]) for x in d]),
        "c": np.array([float(x[4]) for x in d]),
    }


@st.cache_resource
def kline_store():
    """Candles kept in memory per (symbol, timeframe), shared by all sessions."""
    return {}


def get_klines(base, store, sym, tf):
    """First call: full history. Later calls: only the newest few candles are
    fetched and merged (weight 1 instead of 2, so 60s refresh stays under
    Binance's rate limit)."""
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


# ───────────────────────────── Trendline logic ─────────────────────────────
def atr_last(h, l, c, n=ATR_LEN):
    """Wilder ATR (same as ta.atr) - value on the last bar."""
    pc = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    tr[0] = h[0] - l[0]
    r = tr[:n].mean()
    for x in tr[n:]:
        r += (x - r) / n
    return float(r)


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
    """Closest valid trendline zone to the current price (Pine findClosestZone)."""
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
        for j in range(i + 1, size):
            slope = (py[j] - p1y) / (px[j] - p1x)
            diff = ky - (p1y + slope * dx)
            if is_res:
                if np.any(old & (diff > thr)):
                    continue
            elif np.any(old & (diff < -thr)):
                continue
            touch = np.abs(diff) <= thr
            cnt = int(touch.sum())
            if cnt < min_touches:
                continue
            d = abs(p1y + slope * (last - p1x) - close)
            if d < best_d:
                tv = diff[touch]
                best_d = d
                best = dict(
                    p1x=int(p1x), p1y=float(p1y), slope=float(slope),
                    up=float(max(tv.max(), 0.0)), dn=float(min(tv.min(), 0.0)),
                    touches=cnt,
                )
    return best


def zone_at(z, idx):
    base = z["p1y"] + z["slope"] * (idx - z["p1x"])
    return base + z["up"], base + z["dn"]  # top, bottom


def compute_zones(d, cfg):
    h, l, c = d["h"], d["l"], d["c"]
    last = len(c) - 1
    thr = atr_last(h, l, c) * ATR_MULT
    if not np.isfinite(thr) or thr <= 0:
        return None
    s = cfg["strength"]
    src_h = c if cfg["src"] == "Close" else h
    src_l = c if cfg["src"] == "Close" else l
    phx, phy, _, _ = pivots(src_h, s)
    _, _, plx, ply = pivots(src_l, s)
    return {
        "res": find_zone(phx, phy, last, c[-1], thr, True, cfg["back"], cfg["min_touches"]),
        "sup": find_zone(plx, ply, last, c[-1], thr, False, cfg["back"], cfg["min_touches"]),
    }


def touch_series(z, h, l, idxs, pierce):
    top, bot = zone_at(z, idxs)
    hh, ll = h[idxs], l[idxs]
    t = ((hh >= bot) & (hh <= top)) | ((ll <= top) & (ll >= bot))
    if pierce:  # wick spikes straight through the whole zone
        t = t | ((hh > top) & (ll < bot))
    return t


def analyze(sym, tf, d, cfg):
    zones = compute_zones(d, cfg)
    if not zones:
        return []
    h, l, c, t = d["h"], d["l"], d["c"], d["t"]
    last = len(c) - 1
    W = cfg["window"]
    cd = cfg["cooldown"].get(tf, 5)
    price = float(c[-1])
    coin = sym[:-4]
    rows = []
    for key, z in zones.items():
        if not z:
            continue
        res = key == "res"
        top, bot = zone_at(z, last)
        if price > top:
            dist = (price - top) / price * 100
        elif price < bot:
            dist = (bot - price) / price * 100
        else:
            dist = 0.0

        def row(status, signal, age, bar_idx):
            return dict(
                status=status, coin=coin, symbol=sym, tf=tf, kind=key, signal=signal,
                price=price, bot=float(bot), top=float(top), dist=float(dist),
                age=int(age), pivots=z["touches"], bar=int(t[bar_idx]),
            )

        # wick touch (latest candle, or last W candles)
        lo = max(0, last - W - cd + 1)
        idxs = np.arange(lo, last + 1)
        ts = touch_series(z, h, l, idxs, cfg["pierce"])
        hit = np.where(ts & (idxs >= last - W + 1))[0]
        if len(hit):
            p = int(idxs[hit[-1]])
            repeat = bool(ts[(idxs >= p - cd) & (idxs < p)].any())
            rows.append(row(
                "REPEAT" if repeat else "FRESH",
                "Resistance touch ↓" if res else "Support touch ↑",
                last - p, p,
            ))

        # close-based breakout / breakdown of the zone
        if cfg["breakouts"] and last >= 1:
            top_p, bot_p = zone_at(z, last - 1)
            if res and c[-1] > top and c[-2] <= top_p:
                rows.append(row("BREAKOUT", "Resistance breakout ↑", 0, last))
            if (not res) and c[-1] < bot and c[-2] >= bot_p:
                rows.append(row("BREAKOUT", "Support breakdown ↓", 0, last))
    return rows


def scan_all(base, store, symbols, tfs, cfg, workers=16):
    sigs, ok, fail = [], 0, 0

    def job(sym, tf):
        d = get_klines(base, store, sym, tf)
        if d is None:
            return None
        return analyze(sym, tf, d, cfg)

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
    order = {"FRESH": 0, "BREAKOUT": 1, "REPEAT": 2}
    sigs.sort(key=lambda x: (order[x["status"]], x["age"], x["dist"]))
    return sigs, ok, fail


# ───────────────────────────── Alerts ─────────────────────────────
@lru_cache(maxsize=1)
def beep_wav():
    sr = 22050
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = bytearray()
        for f, dur in ((880, 0.18), (1175, 0.18), (880, 0.18)):
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
            data={"chat_id": chat_id, "text": text},
            timeout=8,
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
    icon = "\U0001F525" if s["status"] == "FRESH" else "\U0001F680"
    return (
        f"{icon} {s['signal']}\n{s['symbol']} · {s['tf']}\n"
        f"Price {fp(s['price'])} | Zone {fp(s['bot'])} - {fp(s['top'])}"
    )


# ───────────────────────────── Chart ─────────────────────────────
def make_chart(d, tf, cfg, show=150):
    import plotly.graph_objects as go

    zones = compute_zones(d, cfg) or {}
    n = len(d["c"])
    last = n - 1
    start = max(0, n - show)
    times = pd.to_datetime(d["t"], unit="ms", utc=True).tz_convert(IST)
    step = pd.Timedelta(seconds=TF_SECONDS[tf])

    def tx(i):
        return times[i] if i <= last else times[last] + step * (i - last)

    fig = go.Figure(
        go.Candlestick(
            x=times[start:], open=d["o"][start:], high=d["h"][start:],
            low=d["l"][start:], close=d["c"][start:], name="Price",
        )
    )
    for key, rgb in (("res", "239,68,68"), ("sup", "34,197,94")):
        z = zones.get(key)
        if not z:
            continue
        i0, i1 = max(z["p1x"], start), last + 8
        t0, t1 = tx(i0), tx(i1)
        top0, bot0 = zone_at(z, i0)
        top1, bot1 = zone_at(z, i1)
        fig.add_trace(go.Scatter(
            x=[t0, t1, t1, t0, t0], y=[top0, top1, bot1, bot0, top0],
            fill="toself", mode="lines", hoverinfo="skip",
            line=dict(color=f"rgba({rgb},0.9)", width=1),
            fillcolor=f"rgba({rgb},0.18)",
            name="Resistance" if key == "res" else "Support",
        ))
    fig.update_layout(
        height=420, xaxis_rangeslider_visible=False, showlegend=False,
        margin=dict(l=0, r=0, t=20, b=0),
    )
    return fig


# ───────────────────────────── App ─────────────────────────────
def main():
    st.set_page_config(page_title="Trendline Zone Scanner", page_icon="\U0001F4C8", layout="wide")
    st.title("\U0001F4C8 Trendline Zone Scanner")
    st.caption("Binance Spot · all USDT pairs · wick touch alerts · by Kishore")

    with st.sidebar:
        st.header("Scan")
        tfs = st.multiselect("Timeframes", TF_OPTIONS, default=TF_OPTIONS)
        refresh = st.select_slider("Auto refresh (seconds)", [30, 60, 90, 120, 180, 300], value=60)
        min_vol = st.number_input("Min 24h volume (million USDT, 0 = all coins)", 0.0, 1000.0, 0.0, 1.0)
        window = st.slider("Check last N candles for a touch", 1, 5, 1)
        pierce = st.checkbox(
            "Also count a wick that spikes through the whole zone "
            "(TradingView alert does not count this)", False)
        breakouts = st.checkbox("Also show zone breakouts (close beyond zone)", True)
        show_repeat = st.checkbox("Show REPEAT touches", True)

        st.header("Trendline logic")
        back = st.slider("Bars to apply", 50, 350, 300)
        src = st.selectbox("Pivot source", ["High/Low", "Close"])
        strength = st.slider("Pivot strength", 5, 15, 10)
        min_touches = st.slider("Min pivot confirmation", 2, 8, 3)

        with st.expander("Fresh-touch cooldown (candles)"):
            st.caption("A coin that touched before counts as FRESH again after this many candles.")
            cooldown = {tf: st.number_input(tf, 1, 50, 5, key=f"cd_{tf}") for tf in tfs}

        st.header("Alerts")
        sound = st.checkbox("Sound + popup on new FRESH touch", True)
        tg_token = st.text_input("Telegram bot token (optional)", secret("TELEGRAM_TOKEN"), type="password")
        tg_chat = st.text_input("Telegram chat id (optional)", secret("TELEGRAM_CHAT_ID"))

    if not tfs:
        st.warning("Select at least one timeframe.")
        return

    cfg = dict(
        src=src, strength=strength, min_touches=min_touches, back=back,
        window=window, cooldown=cooldown, pierce=pierce, breakouts=breakouts,
    )

    store = kline_store()

    @st.fragment(run_every=f"{refresh}s")
    def live():
        base, err = find_base()
        if base is None:
            st.error("Binance data fetch avvaledu. " + err)
            return
        symbols = load_symbols(base)
        if not symbols:
            st.error("Binance coin list radhu - konchem sepu tarvata malli try chestundi.")
            return
        if min_vol > 0:
            vols = load_volumes(base)
            symbols = [s for s in symbols if vols.get(s, 0) >= min_vol * 1e6]

        # steady state fetches only the newest candles (weight 1 per request)
        est = len(symbols) * len(tfs) * 60 / refresh
        if est > 4500:
            st.warning(
                f"~{int(est)} request-weight/min - Binance limit (6000) daggara. "
                "Refresh time penchandi leda timeframes taggimchandi."
            )

        sig = repr((tuple(tfs), min_vol, sorted(cfg.items(), key=lambda kv: kv[0])))
        cache = st.session_state.get("scan_cache")
        if cache and cache["sig"] == sig and time.time() - cache["ts"] < refresh * 0.9:
            # the chart picker (a widget inside this fragment) reran it: reuse the
            # last scan instead of hitting Binance again
            sigs, fail, took, scanned_at = (
                cache["sigs"], cache["fail"], cache["took"], cache["at"])
        else:
            t0 = time.time()
            sigs, ok, fail = scan_all(base, store, symbols, tfs, cfg)
            took = time.time() - t0
            if ok == 0:
                st.error("Ekkada nundi kuda candles raaledu (Binance block / rate limit).")
                return
            scanned_at = datetime.now(IST).strftime("%H:%M:%S")
            st.session_state["scan_cache"] = dict(
                sig=sig, ts=t0, sigs=sigs, fail=fail, took=took, at=scanned_at)

            # alerts for new FRESH touches and breakouts
            seen = st.session_state.setdefault("seen", set())
            first_run = "seen_init" not in st.session_state
            st.session_state["seen_init"] = True
            new = []
            for s in sigs:
                key = f"{s['symbol']}|{s['tf']}|{s['signal']}|{s['bar']}"
                if key not in seen:
                    seen.add(key)
                    if s["status"] in ("FRESH", "BREAKOUT"):
                        new.append(s)
            if len(seen) > 5000:
                seen.clear()
            if new and not first_run:
                log = st.session_state.setdefault("log", [])
                for s in reversed(new):
                    log.insert(0, {
                        "Time": scanned_at, "Coin": s["coin"], "TF": s["tf"],
                        "Signal": s["signal"], "Price": fp(s["price"]),
                    })
                del log[100:]
                if sound:
                    for s in new[:5]:
                        st.toast(f"{s['signal']}  {s['symbol']} {s['tf']}", icon="\U0001F525")
                    st.audio(beep_wav(), format="audio/wav", autoplay=True)
                send_telegram(tg_token, tg_chat, "\n\n".join(alert_text(s) for s in new[:15]))

        n_fresh = sum(1 for s in sigs if s["status"] == "FRESH")
        st.caption(
            f"{len(symbols)} coins × {len(tfs)} TF scanned in {took:.0f}s "
            f"(failed requests: {fail}) · last scan {scanned_at} IST · "
            f"\U0001F525 {n_fresh} fresh"
        )

        log = st.session_state.get("log")
        if log:
            with st.expander(f"Alert history ({len(log)})", expanded=False):
                st.dataframe(pd.DataFrame(log), hide_index=True, width="stretch")

        shown = [s for s in sigs if show_repeat or s["status"] != "REPEAT"]
        if not shown:
            st.info("Ippudu e coin lo trendline wick touch ledu. Auto-refresh avtune untundi.")
            return

        label = {"FRESH": "\U0001F525 FRESH", "BREAKOUT": "\U0001F680 BREAKOUT", "REPEAT": "REPEAT"}
        df = pd.DataFrame([{
            "Status": label[s["status"]], "Coin": s["coin"], "TF": s["tf"],
            "Signal": s["signal"], "Price": fp(s["price"]),
            "Zone low": fp(s["bot"]), "Zone high": fp(s["top"]),
            "Dist %": round(s["dist"], 2), "Candles ago": s["age"], "Pivots": s["pivots"],
        } for s in shown])
        st.dataframe(df, hide_index=True, width="stretch")

        with st.expander("Chart", expanded=False):
            opts = {f"{s['coin']} · {s['tf']} · {s['signal']}": s for s in shown[:60]}
            pick = st.selectbox("Signal", list(opts), key="chart_pick")
            sel = opts.get(pick)
            if sel:
                d = get_klines(base, store, sel["symbol"], sel["tf"])
                if d is not None:
                    st.plotly_chart(make_chart(d, sel["tf"], cfg), width="stretch")

    live()


if __name__ == "__main__":
    main()
