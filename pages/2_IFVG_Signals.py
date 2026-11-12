"""
IFVG Signals - live scanner page - Kishore
IFVG Sniper signals on every Binance Spot USDT coin, graded with trend / volume / fee-guard /
trendline-zone filters (see ifvg_plus.py). Same data layer and look as the Trendline scanner.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import streamlit as st

import app as sc
import ifvg_engine as E
import ifvg_plus as P

st.set_page_config(
    page_title="IFVG Signals", page_icon="\U0001F3AF", layout="wide",
    initial_sidebar_state="collapsed",
)
st.title("\U0001F3AF IFVG Signals")
st.caption("Binance Spot · auto scan · IFVG inversion + trend + volume + trendline confluence · by Kishore")
st.page_link("app.py", icon="\U0001F4C8", label="Trendline Zone Scanner")
st.page_link("pages/3_IFVG_Backtest.py", icon="\U0001F4CA",
             label="IFVG Backtest - before vs after filters (train / test)")

TFS = ["5m", "15m", "30m", "1h", "4h"]

with st.expander("⚙️ Settings (optional - scan automatic ga run avtundi)", expanded=False):
    tfs = st.multiselect("Timeframes", TFS, default=["15m", "1h", "4h"])
    c1, c2, c3 = st.columns(3)
    top_n = c1.slider("Coins: top N by 24h volume", 20, 300, 100, 10)
    lookback = c2.slider("Signals from last N closed candles", 1, 6, 2)
    refresh = c3.select_slider("Auto refresh (seconds)", [30, 60, 90, 120, 180, 300], value=60)
    dir_label = st.radio("Direction", ["Both", "Shorts only", "Longs only"], horizontal=True)
    dir_mode = {"Both": "both", "Shorts only": "short", "Longs only": "long"}[dir_label]
    only_a = st.checkbox("Show only A+ (all filters passed)", False)

    st.markdown("**IFVG**")
    c1, c2, c3 = st.columns(3)
    flt = c1.selectbox("IFVG quality filter", ["Balanced", "Loose", "Strict", "Off"], 0)
    sl_mult = c2.number_input("SL = ATR ×", 0.5, 5.0, 1.5, 0.1)
    rr = c3.select_slider("Target (RR)", [1.0, 1.5, 2.0, 2.5, 3.0, 4.0], value=3.0)

    st.markdown("**Profit filters** (A+ = every enabled filter passed)")
    D = P.FILTER_DEFAULTS
    c1, c2 = st.columns(2)
    trend = c1.selectbox("Trend filter", P.TREND_MODES, P.TREND_MODES.index(D["trend"]),
                         help="Long only above EMA200 (and EMA50 > EMA200); short only below.")
    min_vol = c2.slider("Min signal-candle volume (× 20-candle avg, 0 = off)", 0.0, 3.0, D["min_vol_ratio"], 0.1)
    c1, c2, c3 = st.columns(3)
    min_risk = c1.slider("Fee guard: min SL distance (% of entry, 0 = off)", 0.0, 2.0, D["min_risk_pct"], 0.05)
    zone_on = c2.checkbox("Trendline confluence", D["zone_on"],
                          help="Long near a support zone, short near a resistance zone.")
    zone_atr = c3.slider("Max distance to the zone (× ATR)", 0.2, 4.0, D["zone_atr"], 0.1)

    st.markdown("**Alerts**")
    sound = st.checkbox("Sound + popup on new A+ signals", True)
    tg_token = st.text_input("Telegram bot token (optional)", sc.secret("TELEGRAM_TOKEN"), type="password")
    tg_chat = st.text_input("Telegram chat id (optional)", sc.secret("TELEGRAM_CHAT_ID"))

if not tfs:
    st.warning("Select at least one timeframe.")
    st.stop()

p = dict(E.DEFAULT_PARAMS, filter=flt, sl_mult=float(sl_mult))
f = dict(trend=trend, min_vol_ratio=float(min_vol), min_risk_pct=float(min_risk),
         zone_on=zone_on, zone_atr=float(zone_atr))
store = sc.kline_store()


def fp(x):
    return sc.fp(x)


def scan(base, symbols):
    rows, fail = [], 0

    def job(sym, tf):
        d = sc.get_klines(base, store, sym, tf)
        if d is None:
            return None
        return P.live_signals(d, sym, tf, p, f, rr, lookback, dir_mode)

    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(job, s, tf) for s in symbols for tf in tfs]
        for fu in as_completed(futs):
            try:
                r = fu.result()
            except Exception:
                r = None
            if r is None:
                fail += 1
            else:
                rows.extend(r)
    rows.sort(key=lambda x: (x["grade"] != "A+", x["age"], -x["vol_ratio"]))
    return rows, fail


@st.fragment(run_every=f"{refresh}s")
def live():
    base, err = sc.find_base()
    if base is None:
        st.error("Binance data fetch avvaledu. " + err)
        return
    symbols = sc.load_symbols(base)
    if not symbols:
        st.error("Binance coin list radhu - konchem sepu tarvata malli try chestundi.")
        return
    vols = sc.load_volumes(base)
    symbols = sorted(symbols, key=lambda s: -vols.get(s, 0))[:top_n]

    t0 = time.time()
    rows, fail = scan(base, symbols)
    took = time.time() - t0
    now = datetime.now(sc.IST).strftime("%H:%M:%S")

    # alerts for new A+ signals
    seen = st.session_state.setdefault("ifvg_seen", set())
    first = "ifvg_seen_init" not in st.session_state
    st.session_state["ifvg_seen_init"] = True
    new = []
    for s in rows:
        key = f"{s['symbol']}|{s['tf']}|{s['side']}|{s['bar']}"
        if key not in seen:
            seen.add(key)
            if s["grade"] == "A+":
                new.append(s)
    if len(seen) > 5000:
        seen.clear()
    if new and not first:
        if sound:
            for s in new[:6]:
                st.toast(f"IFVG A+ {s['side']} {s['symbol']} {s['tf']}", icon="\U0001F3AF")
            st.audio(sc.beep_wav("flow"), format="audio/wav", autoplay=True)
        sc.send_telegram(tg_token, tg_chat, "\n\n".join(
            f"\U0001F3AF IFVG A+ {s['side']}\n{s['symbol']} · {s['tf']}\n"
            f"Entry {fp(s['entry'])} | SL {fp(s['sl'])} ({s['risk_pct']:.2f}%) | TP {fp(s['tp'])} ({rr:g}R)"
            for s in new[:15]))

    st.caption(f"{len(symbols)} coins × {len(tfs)} TF scanned in {took:.0f}s "
               f"(failed requests: {fail}) · last scan {now} IST · auto refresh every {refresh}s")

    shown = [s for s in rows if s["grade"] == "A+" or not only_a]
    n_a = sum(1 for s in rows if s["grade"] == "A+")
    st.subheader(f"\U0001F3AF IFVG signals ({n_a} A+ · {len(rows)} total)")
    st.caption("A+ = trend + volume + fee guard + trendline confluence all passed. B = a plain IFVG signal "
               "that failed the filters listed in 'Missing'. Entry = signal candle close, "
               f"SL = {sl_mult:g} × ATR, TP = {rr:g}R.")
    if shown:
        st.dataframe(pd.DataFrame([{
            "Grade": s["grade"], "Coin": s["coin"], "TF": s["tf"], "Side": s["side"],
            "Entry": fp(s["entry"]), "SL": fp(s["sl"]), "TP": fp(s["tp"]),
            "Risk %": round(s["risk_pct"], 2), "Vol ×": round(s["vol_ratio"], 1),
            "Zone dist (ATR)": None if s["zone_dist"] is None else round(s["zone_dist"], 2),
            "Missing": s["missing"], "Candles ago": s["age"],
            "Signal time (IST)": pd.to_datetime(s["bar"], unit="ms", utc=True).tz_convert(sc.IST).strftime("%d %b %H:%M"),
        } for s in shown]), hide_index=True, width="stretch")
    else:
        st.info("Ippudu IFVG signal ledu. Auto-refresh avtune untundi.")

    opts = {f"{s['grade']} · {s['coin']} · {s['tf']} · {s['side']}": s for s in shown[:40]}
    if opts:
        with st.expander("Chart", expanded=False):
            pick = st.selectbox("Signal", list(opts), key="ifvg_chart_pick")
            sel = opts.get(pick)
            if sel:
                d = sc.get_klines(base, store, sel["symbol"], sel["tf"])
                if d is not None:
                    fig = sc.make_chart(d, sel["tf"], P.ZONE_CFG)
                    for y, name, color in ((sel["entry"], "Entry", "#3b82f6"),
                                           (sel["sl"], "SL", "#ef4444"),
                                           (sel["tp"], "TP", "#22c55e")):
                        fig.add_hline(y=y, line_dash="dash", line_color=color,
                                      annotation_text=name, annotation_position="right")
                    st.plotly_chart(fig, width="stretch")


live()
