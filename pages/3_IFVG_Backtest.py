"""
IFVG Backtest page - Kishore
Plain IFVG (every signal) vs IFVG Plus (trend + volume + fee guard + trendline confluence),
with a chronological train / test split so you can see what survives out-of-sample.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st

import app as sc
import ifvg_engine as E
import ifvg_plus as P
import tl_backtest as bt

st.set_page_config(
    page_title="IFVG Backtest", page_icon="\U0001F4CA", layout="wide",
    initial_sidebar_state="collapsed",
)
st.title("\U0001F4CA IFVG Backtest")
st.caption("Binance Spot · closed candles only · plain IFVG vs filtered IFVG · by Kishore")
st.page_link("pages/2_IFVG_Signals.py", icon="\U0001F3AF", label="IFVG Signals (live)")

TFS = ["5m", "15m", "30m", "1h", "4h"]
D = P.FILTER_DEFAULTS

with st.expander("⚙️ Backtest settings", expanded=True):
    c1, c2 = st.columns(2)
    n_coins = c1.slider("Coins: top N by 24h volume", 3, 60, 15)
    manual = c2.text_input("Or specific coins (comma separated, e.g. BTC,ETH,SOL)", "")
    tfs = st.multiselect("Timeframes", TFS, default=["1h", "4h"])
    c1, c2, c3 = st.columns(3)
    n_candles = c1.slider("Candles per coin / TF", 2000, 10000, 6000, 500)
    rr = c2.select_slider("Target (RR)", [1.0, 1.5, 2.0, 2.5, 3.0, 4.0], value=3.0)
    split_pct = c3.slider("Train part (%) - the rest is the unseen TEST part", 40, 80, 60, 5)
    c1, c2, c3, c4 = st.columns(4)
    capital = c1.number_input("Capital per coin", 1000.0, 10_000_000.0, 100_000.0, 1000.0)
    risk_pct = c2.number_input("Risk per trade (%)", 0.1, 5.0, 1.0, 0.1)
    fee = c3.number_input("Fee % per side", 0.0, 0.5, 0.01, 0.005, format="%.3f")
    sl_mult = c4.number_input("SL = ATR ×", 0.5, 5.0, 1.5, 0.1)
    flt = st.selectbox("IFVG quality filter", ["Balanced", "Loose", "Strict", "Off"], 0)

    st.markdown("**Profit filters**")
    c1, c2 = st.columns(2)
    trend = c1.selectbox("Trend filter", P.TREND_MODES, P.TREND_MODES.index(D["trend"]))
    min_vol = c2.slider("Min signal-candle volume (× 20-candle avg, 0 = off)", 0.0, 3.0, D["min_vol_ratio"], 0.1)
    c1, c2, c3 = st.columns(3)
    min_risk = c1.slider("Fee guard: min SL distance (%, 0 = off)", 0.0, 2.0, D["min_risk_pct"], 0.05)
    zone_on = c2.checkbox("Trendline confluence (slower)", D["zone_on"])
    zone_atr = c3.slider("Max distance to the zone (× ATR)", 0.2, 4.0, D["zone_atr"], 0.1)

if not tfs:
    st.warning("Select at least one timeframe.")
    st.stop()

p = dict(E.DEFAULT_PARAMS, filter=flt, sl_mult=float(sl_mult), capital=float(capital),
         risk_pct=float(risk_pct), fee_side_pct=float(fee))
f = dict(trend=trend, min_vol_ratio=float(min_vol), min_risk_pct=float(min_risk),
         zone_on=zone_on, zone_atr=float(zone_atr))
split = split_pct / 100.0

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

    jobs = [(s, tf) for s in symbols for tf in tfs]
    bar = st.progress(0.0, text="Candles download chestunna...")
    t0 = time.time()

    # 1) download (parallel, rate-limit aware via sc.http_get)
    data, fail = {}, 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(bt.fetch_history, base, s, tf, n_candles): (s, tf) for s, tf in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                d = fu.result()
            except Exception:
                d = None
            if d is None:
                fail += 1
            else:
                data[futs[fu]] = d
            bar.progress(i / len(jobs) * 0.5, text=f"Candles download {i}/{len(jobs)}")
    if not data:
        st.error("Ekkada nundi kuda candles raaledu (Binance block / rate limit).")
        st.stop()

    # 2) backtest (baseline vs plus)
    results, log_rows = {}, []
    for i, (key, d) in enumerate(data.items(), 1):
        try:
            r = P.run_one(d, p, f, rr, split)
        except Exception:
            fail += 1
            continue
        results[key] = (d, r)
        bar.progress(0.5 + i / len(data) * 0.5, text=f"Backtest {i}/{len(data)} · {key[0][:-4]} {key[1]}")
    bar.empty()
    st.session_state["ifvg_bt"] = dict(results=results, p=p, f=f, rr=rr, split=split,
                                       took=time.time() - t0, fail=fail)

R = st.session_state.get("ifvg_bt")
if not R:
    st.info("Settings set chesi ▶ Run backtest nokkandi.")
    st.stop()

results, p, f, rr = R["results"], R["p"], R["f"], R["rr"]
st.caption(f"{len(results)} coin/TF tests · {R['took']:.0f}s · failed: {R['fail']} · "
           f"{R['split']*100:.0f}% train / {100 - R['split']*100:.0f}% test · RR {rr:g} · "
           f"fee {p['fee_side_pct']}% per side · {p['capital']:,.0f} per coin")


def cols(row, label):
    return {f"{label} trades": row["trades"], f"{label} win %": row["win_rate"],
            f"{label} net PnL": row["net_pnl"], f"{label} PF": row["profit_factor"]}


# ── summary per timeframe ──
st.subheader("Plain IFVG vs IFVG Plus - per timeframe")
st.caption("PF = profit factor (> 1 means profit). **Plus TEST** is the part of the history the filters were "
           "never tuned on - trust that column most. Net PnL is summed over all coins "
           f"({p['capital']:,.0f} per coin).")
rows, verdict = [], []
for tf in tfs:
    items = [(k, v) for k, v in results.items() if k[1] == tf]
    if not items:
        continue
    base_all = P.agg([v[1]["base"] for _, v in items], p["capital"])
    plus_all = P.agg([v[1]["plus"] for _, v in items], p["capital"])
    plus_test = P.agg([P.split_trades(v[1]["plus"], v[1]["split_bar"])[1] for _, v in items], p["capital"])
    base_test = P.agg([P.split_trades(v[1]["base"], v[1]["split_bar"])[1] for _, v in items], p["capital"])
    rows.append({"TF": tf, "Coins": len(items), **cols(base_all, "Plain"), **cols(plus_all, "Plus"),
                 **cols(base_test, "Plain TEST"), **cols(plus_test, "Plus TEST")})
    if plus_test["trades"] >= 30 and plus_test["net_pnl"] > 0 and plus_test["profit_factor"] > 1.1:
        verdict.append(f"✅ {tf}: Plus TEST profitable ({plus_test['trades']} trades, PF {plus_test['profit_factor']})")
    elif plus_test["trades"] < 30:
        verdict.append(f"⚠️ {tf}: Plus TEST only {plus_test['trades']} trades - too few to trust, "
                       "add more coins / candles or loosen a filter")
    else:
        verdict.append(f"❌ {tf}: Plus TEST not profitable (net {plus_test['net_pnl']:,.0f}, PF {plus_test['profit_factor']})")
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
for v in verdict:
    st.write(v)

# ── which filter helps? ──
st.subheader("Which filter helps? (TEST part, all coins together)")
st.caption("Each filter alone, then all together. If a filter alone does not beat Plain IFVG on TEST, "
           "switch it off in the settings.")
abl_rows = []
for tf in tfs:
    items = [v for k, v in results.items() if k[1] == tf]
    if not items:
        continue
    for name in items[0][1]["variants"]:
        te = P.agg([P.split_trades(v[1]["variants"][name], v[1]["split_bar"])[1] for v in items], p["capital"])
        abl_rows.append({"TF": tf, "Variant": name, "TEST trades": te["trades"], "Win %": te["win_rate"],
                         "Net PnL": te["net_pnl"], "PF": te["profit_factor"], "Fees": te["fees"]})
if abl_rows:
    st.dataframe(pd.DataFrame(abl_rows), hide_index=True, width="stretch")

# ── per coin (Plus, test part) ──
st.subheader("Per coin - Plus, TEST part only")
coin_rows = []
for (sym, tf), (d, r) in results.items():
    tr, te = P.split_trades(r["plus"], r["split_bar"])
    a = P.agg([te], p["capital"])
    coin_rows.append({"Coin": sym[:-4], "TF": tf, "IFVG signals": r["n_signals"],
                      "Passed filters": r["n_kept"], "Test trades": a["trades"],
                      "Win %": a["win_rate"], "Net PnL": a["net_pnl"], "PF": a["profit_factor"]})
cdf = pd.DataFrame(coin_rows)
if len(cdf):
    cdf = cdf.sort_values("Net PnL", ascending=False)
    st.dataframe(cdf, hide_index=True, width="stretch")

# ── trade log ──
with st.expander("Trade log (every Plus trade, newest first)", expanded=False):
    keys = [f"{s[:-4]} · {tf}" for (s, tf) in results]
    pick = st.selectbox("Coin · TF", keys)
    sym, tf = pick.split(" · ")
    key = (sym + "USDT", tf)
    if key in results:
        d, r = results[key]
        tl = pd.DataFrame(P.trade_rows(d, r["plus"], r["atr"], rr, p, sym, tf))
        if len(tl):
            st.dataframe(tl.iloc[::-1], hide_index=True, width="stretch")
        else:
            st.info("Ee coin lo filters pass ayye trade ledu.")
