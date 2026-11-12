"""
IFVG Sniper Entry Engine - core engine (shared by the Streamlit app and the headless backtest)
Created & maintained by Kishore

Faithful Python port of the Pine v6 indicator "IFVG Sniper Entry Engine":
  * hidden FVG memory (max 120, expires after 60 bars)
  * IFVG = close beyond the far edge of an opposite FVG (+ ATR buffer) with quality filter
  * entry = IFVG line / confirmation close, SL = ATR * mult, TP = RR * risk
  * one trade at a time, signals while a trade is open are blocked
  * TP and SL touched in the same candle -> SL counts first
"""
import math
import time

import numpy as np
import pandas as pd
import requests

TFS = ["1m", "5m", "15m", "30m", "1h", "4h"]
TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}

# Binance market-data hosts. api.binance.com returns HTTP 451 from US-based servers
# (Streamlit Cloud, GitHub Actions), data-api.binance.vision serves the same public data.
BASES = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]

STABLES = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "USDE", "EUR", "EURI", "AEUR", "XUSD",
           "USD1", "PAXG", "UST", "USTC", "GBP", "TRY", "BRL", "AUD", "RLUSD", "BFUSD"}
BAD_SUFFIX = ("UP", "DOWN", "BULL", "BEAR")

FILTERS = {  # gapAtr, bodyRatio, rangeAtr, breakBufferAtr
    "Off": (0.0, 0.0, 0.0, 0.0),
    "Loose": (0.15, 0.40, 0.40, 0.00),
    "Balanced": (0.25, 0.50, 0.60, 0.05),
    "Strict": (0.40, 0.60, 0.85, 0.10),
}

DEFAULT_PARAMS = dict(
    filter="Balanced", atr_len=14, sl_mult=1.5, max_hidden=120, max_age=60,
    entry_mode="Confirmation Close",      # Pine default is "IFVG Line"; close is the realistic fill
    line_mode="Broken Boundary",
    capital=100_000.0, risk_pct=1.0, fee_side_pct=0.01, max_leverage=5.0,
)


class DataError(Exception):
    pass


# ───────────────────────── DATA ─────────────────────────
def _get(path, params=None, timeout=20, bases=None):
    """GET against the first Binance host that answers. Raises DataError with a readable reason."""
    last = "no host tried"
    for base in (bases or BASES):
        for attempt in range(3):
            try:
                r = requests.get(base + path, params=params, timeout=timeout)
                if r.status_code in (418, 429):
                    time.sleep(float(r.headers.get("Retry-After", 2 + attempt * 2)))
                    last = f"{base}: HTTP {r.status_code} (rate limit)"
                    continue
                if r.status_code == 451:
                    last = f"{base}: HTTP 451 (blocked in this region)"
                    break
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                last = f"{base}: {type(e).__name__}"
                time.sleep(0.4 * (attempt + 1))
    raise DataError(f"Binance not reachable. Last error: {last}")


def get_top_symbols(n=100):
    """Top-n Binance spot USDT pairs by 24h quote volume (stablecoins / leveraged tokens removed)."""
    data = _get("/api/v3/ticker/24hr")
    rows = []
    for t in data:
        s = t["symbol"]
        if not s.endswith("USDT"):
            continue
        b = s[:-4]
        if b in STABLES or b.endswith(BAD_SUFFIX):
            continue
        rows.append((s, float(t["quoteVolume"])))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in rows[:n]]


def fetch_klines(symbol, interval, total):
    """Last `total` CLOSED candles, oldest -> newest. Returns DataFrame t,o,h,l,c,v or None."""
    out, end = [], None
    while len(out) < total + 1:
        p = {"symbol": symbol, "interval": interval, "limit": min(1000, total + 1 - len(out))}
        if end:
            p["endTime"] = end
        data = _get("/api/v3/klines", p)
        if not data:
            break
        out = data + out
        end = data[0][0] - 1
        if len(data) < p["limit"]:
            break
    if len(out) < 2:
        return None
    df = pd.DataFrame(out).iloc[:, :6]
    df.columns = ["t", "o", "h", "l", "c", "v"]
    df = df.astype(float)
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    return df.iloc[:-1].reset_index(drop=True)  # drop the live (unconfirmed) candle


# ───────────────────────── ENGINE ─────────────────────────
def atr_rma(h, l, c, n):
    """Pine ta.atr (RMA of true range, SMA seed). NaN for the first n-1 bars."""
    size = len(c)
    out = np.full(size, np.nan)
    if size < n:
        return out
    pc = np.empty(size)
    pc[0] = c[0]
    pc[1:] = c[:-1]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    tr[0] = h[0] - l[0]
    prev = tr[:n].mean()
    out[n - 1] = prev
    trl = tr.tolist()
    for i in range(n, size):
        prev = (prev * (n - 1) + trl[i]) / n
        out[i] = prev
    return out


def gen_signals(o, h, l, c, atr, p):
    """IFVG signals, independent of trade blocking.
    Returns (signals, filtered_count); signal = (bar, dir(+1/-1), top, bot, confirm_close)."""
    if p["filter"] == "Custom":
        min_gap, min_body, min_rng, brk = p["custom"]
    else:
        min_gap, min_body, min_rng, brk = FILTERS[p["filter"]]
    off = p["filter"] == "Off"
    max_hidden, max_age = p["max_hidden"], p["max_age"]
    n = len(c)
    o, h, l, c = o.tolist(), h.tolist(), l.tolist(), c.tolist()
    atrl = atr.tolist()
    fvgs = []  # [top, bot, dir, birth, gap, body, rng]  oldest -> newest
    signals, filtered = [], 0
    tiny = 1e-12

    for i in range(n):
        while fvgs and i - fvgs[0][3] > max_age:
            fvgs.pop(0)
        a = atrl[i]
        safe = a if (a == a and a > 0) else tiny
        rng = max(h[i] - l[i], tiny)
        if i >= 2:
            body = abs(c[i] - o[i]) / rng
            rng_atr = rng / safe
            if l[i] > h[i - 2]:
                fvgs.append([l[i], h[i - 2], 1, i, (l[i] - h[i - 2]) / safe, body, rng_atr])
            if h[i] < l[i - 2]:
                fvgs.append([l[i - 2], h[i], -1, i, (l[i - 2] - h[i]) / safe, body, rng_atr])
            while len(fvgs) > max_hidden:
                fvgs.pop(0)
        if not fvgs:
            continue
        buf = safe * brk
        ci = c[i]
        for k in range(len(fvgs) - 1, -1, -1):  # newest first, first hit wins
            top, bot, d, _b, g, br, ra = fvgs[k]
            bull = d == -1 and ci > top + buf
            bear = d == 1 and ci < bot - buf
            if bull or bear:
                if off or (g >= min_gap and br >= min_body and ra >= min_rng):
                    signals.append((i, 1 if bull else -1, top, bot, ci))
                else:
                    filtered += 1
                del fvgs[k]
                break
    return signals, filtered


def entry_price(sig, p):
    _bar, d, top, bot, cl = sig
    if p["line_mode"] == "Confirmation Close":
        line = cl
    elif p["line_mode"] == "Midpoint":
        line = (top + bot) / 2.0
    else:
        line = top if d == 1 else bot
    return cl if p["entry_mode"] == "Confirmation Close" else line


def simulate(signals, h, l, atr, rr, p):
    """One trade at a time. Returns list of trade dicts (open trade at end of data is ignored).

    Sizing: risk = capital * risk_pct (fixed on starting capital, no compounding), notional capped
    at capital * max_leverage. Fees = fee_side_pct on notional, charged on entry and exit.
    """
    n = len(h)
    hl, ll, atrl = h.tolist(), l.tolist(), atr.tolist()
    capital = p["capital"]
    risk_cash = capital * p["risk_pct"] / 100.0
    fee_side = p["fee_side_pct"] / 100.0
    trades = []
    blocked_until = -1
    for sig in signals:
        b, d = sig[0], sig[1]
        if b <= blocked_until:
            continue
        a = atrl[b]
        if a != a or a <= 0:
            continue
        entry = entry_price(sig, p)
        risk = a * p["sl_mult"]
        sl = entry - d * risk
        tp = entry + d * risk * rr
        res, j = 0, b + 1
        while j < n:
            if d == 1:
                hit_sl, hit_tp = ll[j] <= sl, hl[j] >= tp
            else:
                hit_sl, hit_tp = hl[j] >= sl, ll[j] <= tp
            if hit_sl:
                res = -1
                break
            if hit_tp:
                res = 1
                break
            j += 1
        if res == 0:
            break
        blocked_until = j
        risk_frac = risk / entry
        notional = min(risk_cash / risk_frac, capital * p["max_leverage"])
        gross = notional * risk_frac * (rr if res == 1 else -1.0)
        fees = notional * fee_side * 2.0
        trades.append(dict(bar=b, exit_bar=j, dir=d, entry=entry, result=res, notional=notional,
                           gross=gross, fees=fees, net=gross - fees, r_net=(gross - fees) / risk_cash))
    return trades


def stats(trades, capital):
    n = len(trades)
    if n == 0:
        return dict(trades=0, wins=0, win_rate=0.0, gross_pnl=0.0, fees=0.0, net_pnl=0.0, ret_pct=0.0,
                    profit_factor=0.0, max_dd=0.0, avg_r=0.0)
    wins = sum(1 for t in trades if t["result"] == 1)
    nets = [t["net"] for t in trades]
    pos = sum(x for x in nets if x > 0)
    neg = -sum(x for x in nets if x < 0)
    eq = peak = dd = 0.0
    for x in nets:
        eq += x
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    return dict(trades=n, wins=wins, win_rate=round(wins * 100.0 / n, 2),
                gross_pnl=round(sum(t["gross"] for t in trades), 2),
                fees=round(sum(t["fees"] for t in trades), 2),
                net_pnl=round(sum(nets), 2), ret_pct=round(sum(nets) * 100.0 / capital, 2),
                profit_factor=round(pos / neg, 3) if neg > 0 else (999.0 if pos > 0 else 0.0),
                max_dd=round(dd, 2), avg_r=round(sum(t["r_net"] for t in trades) / n, 4))


def backtest_df(df, rr_list, p):
    """Backtest one coin/timeframe DataFrame for each RR. Returns {rr: stats}, signal count, filtered count."""
    o, h, l, c = (df[k].values for k in ("o", "h", "l", "c"))
    atr = atr_rma(h, l, c, p["atr_len"])
    signals, filtered = gen_signals(o, h, l, c, atr, p)
    out = {}
    for rr in rr_list:
        tr = simulate(signals, h, l, atr, rr, p)
        out[rr] = (stats(tr, p["capital"]), tr)
    return out, len(signals), filtered


def trade_log(df, rr, p):
    """Every simulated trade of one coin/timeframe with times (UTC), levels and result.
    Use it to compare trade by trade with the TradingView indicator labels."""
    o, h, l, c = (df[k].values for k in ("o", "h", "l", "c"))
    atr = atr_rma(h, l, c, p["atr_len"])
    signals, _ = gen_signals(o, h, l, c, atr, p)
    rows = []
    for t in simulate(signals, h, l, atr, rr, p):
        d, e = t["dir"], t["entry"]
        risk = atr[t["bar"]] * p["sl_mult"]
        rows.append(dict(
            signal_time=df["t"].iloc[t["bar"]], side="BUY" if d == 1 else "SELL", entry=e,
            sl=e - d * risk, tp=e + d * risk * rr, exit_time=df["t"].iloc[t["exit_bar"]],
            result="TP hit" if t["result"] == 1 else "SL hit", gross_pnl=round(t["gross"], 2),
            fees=round(t["fees"], 2), net_pnl=round(t["net"], 2)))
    return pd.DataFrame(rows)


def latest_signals(df, p, lookback=3):
    """IFVG signals formed in the last `lookback` closed candles (for the live scanner)."""
    o, h, l, c = (df[k].values for k in ("o", "h", "l", "c"))
    atr = atr_rma(h, l, c, p["atr_len"])
    signals, _ = gen_signals(o, h, l, c, atr, p)
    n = len(df)
    res = []
    for s in signals:
        if s[0] >= n - lookback and not math.isnan(atr[s[0]]):
            res.append((s, atr[s[0]], df["t"].iloc[s[0]]))
    return res


def aggregate(rows):
    """rows: list of dicts with timeframe, rr, trades, wins, net_pnl, gross_pnl, fees, max_dd.
    Returns per (timeframe, rr) aggregate DataFrame."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    out = []
    for (tf, rr), g in df.groupby(["timeframe", "rr"], sort=False):
        tr = int(g["trades"].sum())
        w = int(g["wins"].sum())
        act = g[g["trades"] > 0]
        out.append(dict(
            timeframe=tf, rr=f"{int(rr)}R", coins=len(act), trades=tr,
            win_rate=round(w * 100.0 / tr, 2) if tr else 0.0,
            avg_pnl_per_coin=round(act["net_pnl"].mean(), 2) if len(act) else 0.0,
            median_pnl_per_coin=round(act["net_pnl"].median(), 2) if len(act) else 0.0,
            total_pnl_all_coins=round(g["net_pnl"].sum(), 2),
            gross_pnl=round(g["gross_pnl"].sum(), 2), fees=round(g["fees"].sum(), 2),
            profitable_coins=f"{int((act['net_pnl'] > 0).sum())}/{len(act)}",
        ))
    res = pd.DataFrame(out)
    order = {t: i for i, t in enumerate(TFS)}
    res["_o"] = res["timeframe"].map(order)
    return res.sort_values(["_o", "rr"]).drop(columns="_o").reset_index(drop=True)
