"""
IFVG Plus - Kishore

The IFVG Sniper engine (ifvg_engine.py, unchanged) plus four filters that decide which
IFVG signals are worth taking, and a trendline-zone confluence check that reuses the
Trendline Zone Scanner logic from app.py 1:1.

Why filters: the plain IFVG backtest (Binance top 100, 0.01% fee per side) loses money on
1m-1h because the SL distance is tiny and fees eat the edge; only 4h was positive. So:
  1. Trend      - trade only with the trend (EMA200 / EMA50 > EMA200 on the same timeframe)
  2. Volume     - the signal candle must have above-average volume
  3. Fee guard  - skip trades whose SL distance is too small (fees take a big share of the risk)
  4. Trendline  - long only near a SUPPORT trendline zone, short only near a RESISTANCE zone

A signal that passes all enabled filters is graded A+. Nothing here guarantees profit:
use the IFVG Backtest page (train / test split) to see which settings survive out-of-sample.
"""
import numpy as np
import pandas as pd

import app as sc
import ifvg_engine as E

TREND_MODES = ["EMA50>EMA200", "EMA200", "Off"]

FILTER_DEFAULTS = dict(
    trend="EMA50>EMA200",
    min_vol_ratio=1.2,    # signal candle volume / average of the previous 20 candles (0 = off)
    min_risk_pct=0.35,    # skip if SL distance < this % of entry (0 = off)
    zone_on=True,         # trendline-zone confluence
    zone_atr=1.5,         # entry must be within this many ATR of the matching zone
)

ZONE_CFG = dict(src="High/Low", strength=10, min_touches=3, back=300)   # same as the scanner


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def make_ctx(d, p):
    """Everything the filters need, computed once per coin / timeframe."""
    o, h, l, c, v = (d[k] for k in ("o", "h", "l", "c", "v"))
    atr = E.atr_rma(h, l, c, p["atr_len"])
    return dict(
        d=d, atr=atr, e50=ema(c, 50), e200=ema(c, 200),
        vma=pd.Series(v).rolling(20).mean().shift(1).to_numpy(),
    )


def zone_confluence(d, b, side, entry, atr_b, k, zcfg=ZONE_CFG):
    """(hit, distance in ATR). Zones are built only from candles up to bar b (no look-ahead).
    Long: a support zone is just below the entry (or the entry is inside it).
    Short: a resistance zone is just above the entry (or the entry is inside it)."""
    s0 = max(0, b + 1 - sc.KLINES)
    sl = {key: d[key][s0:b + 1] for key in ("h", "l", "c")}
    if len(sl["c"]) < 60:
        return False, None
    z = sc.compute_zones(sl, zcfg)
    if not z:
        return False, None
    zz = z.get("sup" if side == 1 else "res")
    if not zz:
        return False, None
    top, bot = sc.zone_at(zz, len(sl["c"]) - 1)
    if side == 1:
        if entry < bot:
            return False, None
        dist = max(entry - top, 0.0) / atr_b
    else:
        if entry > top:
            return False, None
        dist = max(bot - entry, 0.0) / atr_b
    return dist <= k, float(dist)


def check(ctx, sig, p, f, force_zone=False):
    """Filter flags for one IFVG signal. Zones (slow) are only computed when the cheap
    filters already pass, unless force_zone (live scanner wants to show every flag)."""
    d, atr = ctx["d"], ctx["atr"]
    b, side = sig[0], sig[1]
    a = atr[b]
    if not (a == a and a > 0):
        return None
    c, v = d["c"], d["v"]
    entry = E.entry_price(sig, p)
    risk_pct = a * p["sl_mult"] / entry * 100.0

    if f["trend"] == "Off":
        trend = True
    elif b < 200:
        trend = False
    elif f["trend"] == "EMA200":
        trend = bool(c[b] > ctx["e200"][b]) if side == 1 else bool(c[b] < ctx["e200"][b])
    else:
        trend = (bool(c[b] > ctx["e200"][b] and ctx["e50"][b] > ctx["e200"][b]) if side == 1
                 else bool(c[b] < ctx["e200"][b] and ctx["e50"][b] < ctx["e200"][b]))

    m = ctx["vma"][b]
    vol_ratio = float(v[b] / m) if (m == m and m > 0) else 0.0
    volume = f["min_vol_ratio"] <= 0 or vol_ratio >= f["min_vol_ratio"]
    risk = f["min_risk_pct"] <= 0 or risk_pct >= f["min_risk_pct"]

    zone, zdist = None, None
    cheap = trend and volume and risk
    if f["zone_on"] and (cheap or force_zone):
        zone, zdist = zone_confluence(d, b, side, entry, a, f["zone_atr"])
    ok = cheap and (not f["zone_on"] or bool(zone))
    return dict(ok=ok, trend=trend, volume=volume, risk=risk, zone=zone,
                vol_ratio=vol_ratio, risk_pct=risk_pct, zone_dist=zdist, entry=entry, atr=a)


FLAG_LABELS = {"trend": "Trend only", "volume": "Volume only", "risk": "Fee guard only", "zone": "Trendline only"}


def run_one(d, p, f, rr, split=0.6):
    """Plain IFVG vs filtered IFVG on one coin / timeframe.
    `variants` holds trades for: Plain IFVG, each enabled filter alone, and All selected
    (every enabled filter together = IFVG Plus). base / plus are shortcuts to the two ends."""
    o, h, l, c = (d[k] for k in ("o", "h", "l", "c"))
    ctx = make_ctx(d, p)
    atr = ctx["atr"]
    signals, _ = E.gen_signals(o, h, l, c, atr, p)
    flags = []
    for sig in signals:
        r = check(ctx, sig, p, f, force_zone=True)
        if r:
            flags.append((sig, r))
    enabled = [k for k, on in (("trend", f["trend"] != "Off"), ("volume", f["min_vol_ratio"] > 0),
                               ("risk", f["min_risk_pct"] > 0), ("zone", f["zone_on"])) if on]

    def pick(keys):
        return [s for s, r in flags if all(bool(r[k]) for k in keys)]

    variants = {"Plain IFVG": E.simulate(signals, h, l, atr, rr, p)}
    if len(enabled) > 1:
        for k in enabled:
            variants[FLAG_LABELS[k]] = E.simulate(pick([k]), h, l, atr, rr, p)
    kept = pick(enabled)
    variants["All selected"] = E.simulate(kept, h, l, atr, rr, p)
    return dict(base=variants["Plain IFVG"], plus=variants["All selected"], variants=variants,
                atr=atr, split_bar=int(len(c) * split), n_signals=len(signals), n_kept=len(kept))


def split_trades(trades, split_bar):
    return ([t for t in trades if t["bar"] < split_bar],
            [t for t in trades if t["bar"] >= split_bar])


def trade_rows(d, trades, atr, rr, p, coin, tf):
    """Every trade with time (IST), levels and result."""
    rows = []
    ist = sc.IST
    for t in trades:
        side, e = t["dir"], t["entry"]
        risk = atr[t["bar"]] * p["sl_mult"]
        rows.append(dict(
            coin=coin, tf=tf,
            signal_time=pd.to_datetime(int(d["t"][t["bar"]]), unit="ms", utc=True).tz_convert(ist).strftime("%Y-%m-%d %H:%M"),
            side="BUY" if side == 1 else "SELL", entry=float(e), sl=float(e - side * risk), tp=float(e + side * risk * rr),
            exit_time=pd.to_datetime(int(d["t"][t["exit_bar"]]), unit="ms", utc=True).tz_convert(ist).strftime("%Y-%m-%d %H:%M"),
            result="TP hit" if t["result"] == 1 else "SL hit",
            net_pnl=round(t["net"], 2),
        ))
    return rows


def agg(trade_lists, capital):
    """One stats row over several coins' trades (each coin has its own `capital`)."""
    allt = [t for tl in trade_lists for t in tl]
    n = len(allt)
    if n == 0:
        return dict(trades=0, win_rate=0.0, net_pnl=0.0, profit_factor=0.0, fees=0.0)
    wins = sum(1 for t in allt if t["result"] == 1)
    pos = sum(t["net"] for t in allt if t["net"] > 0)
    neg = -sum(t["net"] for t in allt if t["net"] < 0)
    return dict(trades=n, win_rate=round(wins * 100.0 / n, 1),
                net_pnl=round(sum(t["net"] for t in allt), 0),
                profit_factor=round(pos / neg, 2) if neg > 0 else (99.0 if pos > 0 else 0.0),
                fees=round(sum(t["fees"] for t in allt), 0))


def live_signals(d_full, sym, tf, p, f, rr, lookback, dir_mode="both"):
    """IFVG signals of the last `lookback` closed candles with their filter flags.
    d_full is the scanner's candle dict (last candle still forming: dropped here)."""
    d = {k: v[:-1] for k, v in d_full.items()}
    n = len(d["c"])
    if n < 80:
        return []
    o, h, l, c = (d[k] for k in ("o", "h", "l", "c"))
    ctx = make_ctx(d, p)
    atr = ctx["atr"]
    signals, _ = E.gen_signals(o, h, l, c, atr, p)
    out = []
    for sig in signals:
        b, side = sig[0], sig[1]
        if b < n - lookback:
            continue
        if (dir_mode == "short" and side == 1) or (dir_mode == "long" and side == -1):
            continue
        r = check(ctx, sig, p, f, force_zone=True)
        if not r:
            continue
        entry, a = r["entry"], r["atr"]
        risk = a * p["sl_mult"]
        missing = [name for name, okv in (("Trend", r["trend"]), ("Volume", r["volume"]),
                                          ("Fee guard", r["risk"]),
                                          ("Trendline", r["zone"] if f["zone_on"] else True)) if not okv]
        out.append(dict(
            grade="A+" if r["ok"] else "B", missing=", ".join(missing) or "-",
            coin=sym[:-4], symbol=sym, tf=tf, side="BUY" if side == 1 else "SELL", dir=side,
            entry=float(entry), sl=float(entry - side * risk), tp=float(entry + side * risk * rr),
            risk_pct=float(r["risk_pct"]), vol_ratio=float(r["vol_ratio"]),
            zone_dist=r["zone_dist"], age=int(n - 1 - b), bar=int(d["t"][b]),
            ifvg_top=float(sig[2]), ifvg_bot=float(sig[3]),
        ))
    return out
