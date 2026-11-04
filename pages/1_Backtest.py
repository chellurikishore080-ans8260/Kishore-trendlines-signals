"""
Trendline Zone Backtest page - Kishore
5 separate backtests of the Trendline Zone Scanner signals (see tl_backtest.py).
"""

import time

import pandas as pd
import streamlit as st

import app as sc
import tl_backtest as bt

st.set_page_config(
    page_title="Trendline Backtest", page_icon="\U0001F9EA", layout="wide",
    initial_sidebar_state="collapsed",
)
st.title("\U0001F9EA Trendline Zone Backtest")
st.caption(
    "Binance Spot · walk-forward (closed candles only, no look-ahead) · "
    "by Kishore · every signal has its own separate backtest"
)

D = bt.DEFAULTS
TF_OPTIONS = sc.TF_OPTIONS

st.page_link("app.py", label="Back to Scanner", icon="\U0001F4C8")

with st.expander("⚙️ Backtest settings (default: 1 lakh capital, 50% per trade)", expanded=False):
    c1, c2 = st.columns(2)
    n_coins = c1.slider("Coins: top N by 24h volume", 3, 60, 15)
    manual = c2.text_input("Or specific coins (comma separated, e.g. BTC,ETH,SOL)", "")
    tfs = st.multiselect("Zone timeframes", TF_OPTIONS, default=["15m", "1h"])
    c1, c2, c3 = st.columns(3)
    n_test = c1.slider("Candles to test per coin / TF", 300, 5000, 1500, 100)
    rr = c2.select_slider("Target (RR)", [1.0, 1.5, 2.0, 2.5, 3.0], value=2.0)
    max_hold = c3.slider("Max hold (candles, then TIMEOUT exit)", 10, 200, 50, 5)
    c1, c2, c3 = st.columns(3)
    fee = c1.number_input("Fee % per side", 0.0, 0.5, D["fee"], 0.01)
    capital = c2.number_input("Capital", 1000.0, 10_000_000.0, D["capital"], 1000.0)
    pos_pct = c3.slider("Capital used per trade (%)", 5, 100, int(D["pos_pct"]), 5)
    sl_label = st.radio(
        "Stop loss",
        ["Signal candle low/high (same as scanner alerts)",
         "Beyond the zone's far edge + 0.1 ATR"],
        horizontal=False,
    )

    st.markdown("**Which backtests to run**")
    chosen = [k for k, name in bt.STRATS.items() if st.checkbox(name, True, key=f"s_{k}")]

    st.markdown("**Trendline logic** (same as the scanner)")
    c1, c2, c3, c4 = st.columns(4)
    back = c1.slider("Bars to apply", 50, 350, D["back"])
    src = c2.selectbox("Pivot source", ["High/Low", "Close"])
    strength = c3.slider("Pivot strength", 5, 15, D["strength"])
    min_touches = c4.slider("Min pivot confirmation", 2, 8, D["min_touches"])

    st.markdown("**1 · Touch**")
    c1, c2, c3 = st.columns(3)
    cd = c1.number_input("Fresh-touch cooldown (candles)", 1, 50, 5)
    include_repeat = c2.checkbox("Also trade REPEAT touches", False)
    pierce = c3.checkbox("Count a wick through the whole zone as a touch", False)

    st.markdown("**2 / 3 · Order flow**")
    c1, c2, c3 = st.columns(3)
    flow_delta = c1.slider("Min delta (% of candle volume)", 1, 40, D["flow_delta"])
    flow_close = c2.slider("Min close strength (%)", 0, 90, D["flow_close"], 5)
    flow_pierce = c3.slider("Max wick through zone (× ATR)", 0.0, 2.0, D["flow_pierce"], 0.1)
    c1, c2, c3 = st.columns(3)
    flow_reclaim = c1.checkbox("Full: require reclaim", D["flow_reclaim"])
    flow_min = c2.slider("Full: min score", 1, 4, D["flow_min"])
    flow_look = c3.slider("CVD divergence lookback", 8, 40, D["flow_look"])

    st.markdown("**4 · 1m sweep**")
    c1, c2, c3 = st.columns(3)
    n_1m = c1.slider("1m candles to test per coin", 1000, 10000, 4000, 500)
    sweep_cap = c2.slider("Max overshoot beyond zone (%)", 0.05, 2.0, D["sweep_cap"], 0.05)
    sweep_wick = c3.slider("Min sweep wick (% of 1m candle)", 0, 80, D["sweep_wick"], 5)
    max_hold_1m = st.slider("Sweep: max hold (1m candles)", 10, 500, D["max_hold_1m"], 10)

    st.markdown("**5 · Break & retest**")
    c1, c2, c3 = st.columns(3)
    rt_window = c1.slider("Retest window after break (candles)", 3, 60, D["rt_window"])
    rt_close = c2.slider("Retest candle close strength (%)", 0, 90, D["rt_close"], 5)
    rt_pierce = c3.slider("Max wick through zone (× ATR)", 0.0, 2.0, D["rt_pierce"], 0.1, key="rtp")
    rt_short = st.checkbox("Also test the SELL mirror (support breaks, retest, rejection)", False)

if not tfs:
    st.warning("Select at least one zone timeframe.")
    st.stop()
if not chosen:
    st.warning("Select at least one backtest.")
    st.stop()

cfg = dict(
    D, src=src, strength=strength, min_touches=min_touches, back=back,
    pierce=pierce, include_repeat=include_repeat, cooldown={tf: cd for tf in TF_OPTIONS},
    flow_delta=flow_delta, flow_reclaim=flow_reclaim, flow_min=flow_min,
    flow_close=flow_close, flow_pierce=flow_pierce, flow_look=flow_look,
    sweep_cap=sweep_cap, sweep_wick=sweep_wick, max_hold_1m=max_hold_1m,
    rt_window=rt_window, rt_pierce=rt_pierce, rt_close=rt_close, rt_short=rt_short,
    rr=rr, max_hold=max_hold, fee=fee, capital=capital, pos_pct=float(pos_pct),
    sl_mode="candle" if sl_label.startswith("Signal") else "zone",
)

if st.button("▶ Run backtest", type="primary", width="stretch"):
    base, err = sc.find_base()
    if base is None:
        st.error("Binance data fetch avvaledu. " + err)
        st.stop()
    symbols = sc.load_symbols(base)
    if not symbols:
        st.error("Binance coin list radhu - konchem sepu tarvata malli try cheyandi.")
        st.stop()
    if manual.strip():
        want = {x.strip().upper().replace("USDT", "") for x in manual.split(",") if x.strip()}
        symbols = [s for s in symbols if s[:-4] in want]
        if not symbols:
            st.error("Aa coins Binance Spot USDT lo dorakaledu.")
            st.stop()
    else:
        vols = sc.load_volumes(base)
        symbols = sorted(symbols, key=lambda s: -vols.get(s, 0))[:n_coins]

    prog = st.progress(0.0, text="Starting...")
    t0 = time.time()

    def on_progress(done, total):
        prog.progress(done / total, text=f"{done}/{total} jobs done · {time.time() - t0:.0f}s")

    df, fail = bt.run_backtest(
        base, symbols, tfs, n_test, cfg, chosen, n_1m=n_1m, progress=on_progress)
    prog.empty()
    st.session_state["bt"] = dict(
        df=df, fail=fail, cfg=cfg, chosen=chosen, n_coins=len(symbols), tfs=tfs,
        took=time.time() - t0, n_test=n_test, n_1m=n_1m,
        at=pd.Timestamp.now(tz=sc.IST).strftime("%d %b %H:%M"),
    )

R = st.session_state.get("bt")
if not R:
    st.info("**Run backtest** nokkandi (default: 1 lakh capital). Win rate + PnL results ikkada separate tabs lo vastayi.")
    st.stop()

df, cfg = R["df"], R["cfg"]
st.caption(
    f"{R['n_coins']} coins · {', '.join(R['tfs']) or '-'} · {R['n_test']} candles/TF "
    f"(sweep: {R['n_1m']} 1m candles) · RR {cfg['rr']} · fee {cfg['fee']}%/side · "
    f"SL {'candle' if cfg['sl_mode'] == 'candle' else 'zone edge'} · ran {R['at']} "
    f"in {R['took']:.0f}s · failed data jobs: {R['fail']}"
)
st.caption(
    "PnL = fixed capital per trade (no compounding). Because wide-stop trades weigh more in "
    "money than in R, Net PnL and Avg R can point in different directions - check both. "
    "Success = target hit · Failed = stop hit · Timeout = closed at max hold."
)

names = [bt.STRATS[k] for k in R["chosen"]]
tabs = st.tabs(["\U0001F4CA Compare"] + names)

rows = []
for k in R["chosen"]:
    sub = df[df["Strategy"] == k]
    s = bt.summarize(sub, cfg)
    s["Backtest"] = bt.STRATS[k]
    rows.append(s)
with tabs[0]:
    cmp = pd.DataFrame(rows)[[
        "Backtest", "Trades", "Success", "Failed", "Timeout", "Win rate %", "Net PnL",
        "Return %", "Final capital", "Profit factor", "Profitable %",
        "Avg R", "Avg R (no fees)", "Total R", "Max DD", "Open"]]
    st.dataframe(cmp, hide_index=True, width="stretch")
    st.caption("Open = trades still running at the end of the data (not counted).")

for tab, k in zip(tabs[1:], R["chosen"]):
    sub = df[df["Strategy"] == k].copy()
    with tab:
        st.subheader(bt.STRATS[k])
        explain = {
            "touch": "A candle wick touches the trendline zone (FRESH touches only). "
                     "Support touch = BUY, resistance touch = SELL. Entry = candle close.",
            "flow_basic": "Zone touch + order flow: delta flips in the zone's favour on the same "
                          "candle and it reclaims the previous candle's high/low. "
                          "No absorption, no CVD divergence.",
            "flow_full": "Scanner Signal 2: absorption or delta flip + reclaim, with CVD "
                         "divergence and liquidity sweep adding to the score.",
            "sweep": "A 1m candle sweeps the (closed-candle) higher-timeframe zone and closes "
                     "back inside. SL = sweep candle extreme. Trade runs on 1m candles.",
            "retest": "Resistance zone breaks (close above), price comes back, a candle wicks into "
                      "the same zone but closes above it with strength = BUY."
                      + (" Mirror for support breaks = SELL." if cfg["rt_short"] else ""),
        }[k]
        st.caption(explain)
        if sub.empty:
            st.info("E settings tho e backtest lo trades raaledu. Candles/coins penchandi leda settings loosen cheyandi.")
            continue
        s = bt.summarize(sub, cfg)
        m = st.columns(3)
        m[0].metric("Trades", s["Trades"], f"{s['Success']}✅ {s['Failed']}❌ {s['Timeout']}⏱")
        m[1].metric("Win rate", f"{s['Win rate %']}%")
        m[2].metric("Profit factor", s["Profit factor"])
        m = st.columns(3)
        m[0].metric(f"Net PnL (on {cfg['capital']:,.0f})", f"{s['Net PnL']:,.0f}")
        m[1].metric("Return", f"{s['Return %']}%")
        m[2].metric("Final capital", f"{s['Final capital']:,.0f}")

        closed = sub[sub["Result"] != "OPEN"].sort_values("_t_out")
        if not closed.empty:
            import plotly.graph_objects as go
            eq = closed["PnL"].astype(float).cumsum()
            fig = go.Figure(go.Scatter(
                x=closed["Exit time"], y=eq, mode="lines", line=dict(width=2)))
            fig.update_layout(
                height=280, margin=dict(l=0, r=0, t=24, b=0), title="Equity curve (cumulative PnL)")
            st.plotly_chart(fig, width="stretch")

        for by in ("TF", "Coin"):
            g = (closed.groupby(by)
                 .agg(Trades=("R", "size"),
                      Success=("Result", lambda x: int((x == "Success").sum())),
                      Failed=("Result", lambda x: int((x == "Failed").sum())),
                      AvgR=("R", "mean"), PnL=("PnL", "sum"))
                 .reset_index())
            g["Win rate %"] = (g["Success"] / g["Trades"] * 100).round(1)
            g["AvgR"] = g["AvgR"].round(2)
            g["PnL"] = g["PnL"].round(0)
            with st.expander(f"By {'timeframe' if by == 'TF' else 'coin'}", expanded=(by == "TF")):
                st.dataframe(
                    g.sort_values("PnL", ascending=False), hide_index=True, width="stretch")

        icon = {"Success": "✅ Success", "Failed": "❌ Failed", "Timeout": "⏱ Timeout", "OPEN": "… Open"}
        view = pd.DataFrame({
            "Entry time": sub["Entry time"].dt.strftime("%d %b %H:%M"),
            "Coin": sub["Coin"], "TF": sub["TF"], "Side": sub["Side"],
            "Result": sub["Result"].map(icon),
            "Entry": sub["Entry"].map(sc.fp), "SL": sub["SL"].map(sc.fp),
            "TP": sub["TP"].map(sc.fp), "Exit": sub["Exit"].map(lambda x: "" if pd.isna(x) else sc.fp(x)),
            "R": sub["R"].round(2), "PnL": sub["PnL"].round(0),
            "Risk %": sub["Risk_pct"].round(2), "Candles held": sub["Bars"],
            "Why": sub["Why"],
        })
        st.markdown(f"**All trades ({len(view)}) - newest first**")
        st.dataframe(view, hide_index=True, width="stretch")
        st.download_button(
            "⬇ Download trades CSV",
            sub.drop(columns=["_t_in", "_t_out"]).to_csv(index=False).encode(),
            file_name=f"trendline_backtest_{k}.csv", mime="text/csv", key=f"dl_{k}",
        )
