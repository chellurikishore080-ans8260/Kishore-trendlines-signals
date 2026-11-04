"""
Trendline Zone Backtest engine
Author: Kishore

Walk-forward (no look-ahead) backtests for the Trendline Zone Scanner (app.py).
Every signal is evaluated only with candles that were already CLOSED at that time:
at bar i the trendline zones are rebuilt from candles[:i+1] (same 400-candle window
as the live scanner), so nothing from the future leaks in.

Five separate backtests, each with its own trades and stats:
  touch       1. Trendline touch (original concept) - wick touches the zone
  flow_basic  2. Touch + order flow WITHOUT CVD divergence / absorption
              (delta flip + reclaim, sweep is a bonus)
  flow_full   3. Touch + order flow WITH absorption + delta flip + CVD divergence
              (exact Signal 2 logic of the scanner)
  sweep       4. 1-minute sweep at the zone (Signal 3 of the scanner)
  retest      5. Break & retest: zone breaks, price comes back, candle holds
              on the zone -> BUY (optional mirror: SELL)

Trade rules (same for all): entry = close of the signal candle, SL = candle
low/high (or beyond the zone), TP = RR x risk. SL and TP in the same candle -> SL
counts first (conservative). No TP/SL within max-hold candles -> TIMEOUT exit.
One open trade at a time per coin + timeframe + strategy.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import app as sc  # the scanner module: zone logic is reused 1:1

ZW = sc.KLINES  # 400 candles per zone window, same as the live scanner
STEP = {"1m": 60_000, **{k: v * 1000 for k, v in sc.TF_SECONDS.items()}}

STRATS = {
    "touch": "1 · Trendline touch (original)",
    "flow_basic": "2 · Touch + order flow (no CVD div / absorption)",
    "flow_full": "3 · Touch + order flow (absorption + flip + CVD div)",
    "sweep": "4 · 1m sweep at trendline",
    "retest": "5 · Break & retest BUY",
}

DEFAULTS = dict(
    # trendline zone (same as the scanner)
    src="High/Low", strength=10, min_touches=3, back=300,
    # signal 1 - touch
    pierce=False, include_repeat=False,
    cooldown={"5m": 5, "15m": 5, "30m": 5, "1h": 5, "4h": 5},
    # order flow
    flow_delta=10, flow_reclaim=True, flow_min=2, flow_close=50,
    flow_pierce=1.0, flow_look=20,
    # 1m sweep
    sweep_cap=0.5, sweep_wick=20,
    # break & retest
    rt_window=20, rt_pierce=1.0, rt_close=50, rt_short=False,
    # trade management
    rr=2.0, sl_mode="candle", sl_buf=0.1, max_hold=50, max_hold_1m=120, min_risk=0.05,
    # money
    fee=0.05, capital=100_000.0, pos_pct=50.0,
)


# ───────────────────────────── data ─────────────────────────────
def fetch_history(base, symbol, tf, total):
    """Up to `total` CLOSED candles, newest last (paginates 1000 at a time)."""
    step = STEP[tf]
    rows, end, guard = [], None, 0
    while len(rows) < total and guard < 30:
        guard += 1
        params = {"symbol": symbol, "interval": tf, "limit": 1000}
        if end is not None:
            params["endTime"] = end
        chunk = sc.http_get(base, "/api/v3/klines", params)
        if not chunk:
            break
        rows = chunk + rows
        end = int(chunk[0][0]) - 1
        if len(chunk) < 1000:
            break
    if not rows:
        return None
    if int(rows[-1][0]) + step > int(time.time() * 1000):
        rows.pop()  # the forming candle is never used
    rows = rows[-total:]
    if len(rows) < 120:
        return None
    return {
        "t": np.array([x[0] for x in rows], dtype=np.int64),
        "o": np.array([float(x[1]) for x in rows]),
        "h": np.array([float(x[2]) for x in rows]),
        "l": np.array([float(x[3]) for x in rows]),
        "c": np.array([float(x[4]) for x in rows]),
        "v": np.array([float(x[5]) for x in rows]),
        "tb": np.array([float(x[9]) for x in rows]),
    }


# ───────────────────────────── zones, bar by bar ─────────────────────────────
def _shift(z, off):
    if not z:
        return None
    z = dict(z)
    z["p1x"] += off
    return z


def build_zones(d, cfg, start, end=None):
    """zones[i] = trendline zones known at the close of bar i (candles[:i+1] only).
    Zone line parameters are stored in absolute bar indexes."""
    n = len(d["c"])
    end = n if end is None else end
    out = [None] * n
    for i in range(start, end):
        s0 = max(0, i + 1 - ZW)
        sl = {k: d[k][s0:i + 1] for k in ("h", "l", "c")}
        z = sc.compute_zones(sl, cfg)
        if not z:
            continue
        out[i] = dict(
            res=_shift(z["res"], s0), sup=_shift(z["sup"], s0),
            atr=sc.atr_last(sl["h"], sl["l"], sl["c"]),
        )
    return out


# ───────────────────────────── signal 1: touch ─────────────────────────────
def touch_at(d, z, i, cd, pierce):
    """None / 'FRESH' / 'REPEAT' for a wick touch on bar i (scanner logic)."""
    idxs = np.arange(max(0, i - cd), i + 1)
    ts = sc.touch_series(z, d["h"], d["l"], idxs, pierce)
    if not ts[-1]:
        return None
    return "REPEAT" if ts[:-1].any() else "FRESH"


# ───────────────────────────── signal 2: order flow ─────────────────────────────
def flow_at(d, cvd, z, key, i, atr, cfg, mode):
    """Order-flow reaction at the zone on closed bar i.
    mode 'full'  = scanner Signal 2: absorption or delta flip, CVD divergence,
                   sweep, reclaim (min score).
    mode 'basic' = delta flip + reclaim only (no absorption, no CVD divergence);
                   a sweep through the zone is just a bonus."""
    o, h, l, c, v, tb = (d[k] for k in ("o", "h", "l", "c", "v", "tb"))
    if i < sc.FLOW_MA + 1:
        return None
    rng, vol = float(h[i] - l[i]), float(v[i])
    if rng <= 0 or vol <= 0:
        return None
    res = key == "res"
    top, bot = sc.zone_at(z, i)
    if res:
        if not (o[i] <= top and h[i] >= bot and c[i] <= top):
            return None
        pierce = max(float(h[i]) - top, 0.0)
        strength = float(h[i] - c[i]) / rng * 100
    else:
        if not (o[i] >= bot and l[i] <= top and c[i] >= bot):
            return None
        pierce = max(bot - float(l[i]), 0.0)
        strength = float(c[i] - l[i]) / rng * 100
    if pierce > cfg["flow_pierce"] * atr or strength < cfg["flow_close"]:
        return None
    thr = cfg["flow_delta"]
    dp = (2.0 * tb[i] - v[i]) / vol * 100
    d_prev = 2.0 * tb[i - 1] - v[i - 1]
    enough = vol >= float(v[i - sc.FLOW_MA:i].mean())
    if res:
        absorb = dp >= thr and enough
        flip = bool(d_prev > 0) and dp <= -thr
        reclaim = bool(c[i] < l[i - 1])
    else:
        absorb = dp <= -thr and enough
        flip = bool(d_prev < 0) and dp >= thr
        reclaim = bool(c[i] > h[i - 1])
    swept = pierce > 0
    if mode == "basic":
        if not (flip and reclaim):
            return None
        return dict(why="Delta flip + Reclaim" + (" + Sweep" if swept else ""),
                    score=2 + int(swept))
    if not (absorb or flip):
        return None
    if cfg["flow_reclaim"] and not reclaim:
        return None
    a, b = max(0, i - cfg["flow_look"]), i - 2
    div = False
    if b > a:
        if res:
            j = a + int(np.argmax(h[a:b]))
            div = bool(h[i] >= h[j] - 0.1 * atr and cvd[i] < cvd[j])
        else:
            j = a + int(np.argmin(l[a:b]))
            div = bool(l[i] <= l[j] + 0.1 * atr and cvd[i] > cvd[j])
    score = 1 + int(div) + int(swept) + int(reclaim)
    if score < cfg["flow_min"]:
        return None
    why = ["Absorption" if absorb else "Delta flip"]
    if div:
        why.append("CVD div")
    if swept:
        why.append("Sweep")
    if reclaim:
        why.append("Reclaim")
    return dict(why=" + ".join(why), score=score)


def gen_touch_flow(d, zones, tf, cfg, start):
    """Signals 1, 2 (basic) and 3 (full) in one pass over the bars."""
    cvd = np.cumsum(2.0 * d["tb"] - d["v"])
    cd = cfg["cooldown"].get(tf, 5)
    out = {"touch": [], "flow_basic": [], "flow_full": []}
    for i in range(start, len(d["c"])):
        zi = zones[i]
        if zi is None:
            continue
        for key in ("res", "sup"):
            z = zi[key]
            if not z:
                continue
            top, bot = sc.zone_at(z, i)
            side = "short" if key == "res" else "long"
            base = dict(i=i, side=side, key=key, zbot=bot, ztop=top)
            t = touch_at(d, z, i, cd, cfg["pierce"])
            if t == "FRESH" or (t == "REPEAT" and cfg["include_repeat"]):
                out["touch"].append(dict(base, why="Zone touch" + (" (repeat)" if t == "REPEAT" else "")))
            for mode, name in (("basic", "flow_basic"), ("full", "flow_full")):
                f = flow_at(d, cvd, z, key, i, zi["atr"], cfg, mode)
                if f:
                    out[name].append(dict(base, why=f["why"], score=f["score"]))
    return out


# ───────────────────────────── signal 5: break & retest ─────────────────────────────
def gen_retest(d, zones, cfg, start):
    """Zone broken by a close -> price returns to the SAME line -> a candle wicks
    into the zone but closes back on the breakout side -> trade in the breakout
    direction (resistance broken up = BUY; support broken down = SELL if enabled)."""
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    n = len(c)
    win = cfg["rt_window"]
    sides = [("res", "long")] + ([("sup", "short")] if cfg["rt_short"] else [])
    pending, used, out = [], {}, []
    last_atr = None
    for i in range(max(start, 1), n):
        zi = zones[i]
        atr = zi["atr"] if zi else last_atr
        if zi:
            last_atr = zi["atr"]
        # 1) pending breaks: look for the retest candle on bar i
        keep = []
        for p in pending:
            if i > p["b"] + win:
                continue                      # window over
            z, long = p["z"], p["side"] == "long"
            top, bot = sc.zone_at(z, i)
            if long and c[i] < bot:
                continue                      # closed back through the zone: failed break
            if (not long) and c[i] > top:
                continue
            rng = float(h[i] - l[i])
            fired = False
            if rng > 0 and atr:
                if long:
                    touched = l[i] <= top and l[i] >= bot - cfg["rt_pierce"] * atr
                    holds = c[i] > top
                    strength = float(c[i] - l[i]) / rng * 100
                else:
                    touched = h[i] >= bot and h[i] <= top + cfg["rt_pierce"] * atr
                    holds = c[i] < bot
                    strength = float(h[i] - c[i]) / rng * 100
                if touched and holds and strength >= cfg["rt_close"]:
                    out.append(dict(
                        i=i, side=p["side"], key=p["key"], zbot=bot, ztop=top,
                        why=f"Break + retest hold ({i - p['b']} bars after break)",
                    ))
                    fired = True
            if not fired:
                keep.append(p)
        pending = keep
        # 2) new close-based break of the zone known at bar i-1
        zp = zones[i - 1]
        if not zp:
            continue
        for key, side in sides:
            z = zp[key]
            if not z:
                continue
            top_p, bot_p = sc.zone_at(z, i - 1)
            top_i, bot_i = sc.zone_at(z, i)
            if side == "long":
                broke = c[i] > top_i and c[i - 1] <= top_p
            else:
                broke = c[i] < bot_i and c[i - 1] >= bot_p
            if not broke:
                continue
            lk = (key, z["p1x"], round(z["slope"], 12))
            if used.get(lk, -1) >= i:
                continue
            used[lk] = i + win
            pending.append(dict(z=z, key=key, side=side, b=i))
    return out


# ───────────────────────────── signal 4: 1m sweep ─────────────────────────────
def gen_sweep(m, d, zones, tf, cfg):
    """1m sweep candles against zones of the higher timeframe `tf`.
    The zone used for a 1m candle is the one known at the close of the last
    fully closed tf-candle; it is projected along its slope to the 1m time."""
    step = STEP[tf]
    t_h = d["t"]
    out = []
    for q in range(len(m["c"])):
        tq = int(m["t"][q])
        j = int(np.searchsorted(t_h, tq - step, side="right")) - 1
        if j < 0 or zones[j] is None:
            continue
        zj = zones[j]
        fidx = j + (tq - int(t_h[j])) / step
        o_, h_, l_, c_ = (float(m[k][q]) for k in ("o", "h", "l", "c"))
        rng = h_ - l_
        if rng <= 0:
            continue
        for key in ("res", "sup"):
            z = zj[key]
            if not z:
                continue
            top, bot = sc.zone_at(z, fidx)
            if key == "res":
                if not (o_ <= top and h_ > top and c_ <= top):
                    continue
                over = (h_ - top) / top * 100
                wick = (h_ - max(o_, c_)) / rng * 100
                side = "short"
            else:
                if not (o_ >= bot and l_ < bot and c_ >= bot):
                    continue
                over = (bot - l_) / bot * 100
                wick = (min(o_, c_) - l_) / rng * 100
                side = "long"
            if over > cfg["sweep_cap"] or wick < cfg["sweep_wick"]:
                continue
            out.append(dict(
                i=q, side=side, key=key, zbot=bot, ztop=top,
                why=f"1m sweep ({tf} zone, overshoot {over:.2f}%)",
            ))
    return out


# ───────────────────────────── trades ─────────────────────────────
def simulate(d, i0, side, sl, tp, max_hold):
    """Walk forward from bar i0+1. Returns (exit_bar, reason, exit_price) or None
    if neither level was reached and the data ended before max_hold."""
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    n = len(c)
    last = min(n - 1, i0 + max_hold)
    for k in range(i0 + 1, last + 1):
        if side == "long":
            if l[k] <= sl:
                return k, "SL", float(min(sl, o[k]))
            if h[k] >= tp:
                return k, "TP", float(tp)
        else:
            if h[k] >= sl:
                return k, "SL", float(max(sl, o[k]))
            if l[k] <= tp:
                return k, "TP", float(tp)
    if i0 + max_hold <= n - 1:
        return i0 + max_hold, "TIMEOUT", float(c[i0 + max_hold])
    return None


def make_trades(d, sigs, cfg, step, atr_at, sl_mode, max_hold, meta):
    """Turn signals into trades: one open trade at a time."""
    trades, busy = [], -1
    notional = cfg["capital"] * cfg["pos_pct"] / 100.0
    for s in sorted(sigs, key=lambda x: x["i"]):
        i = s["i"]
        if i <= busy:
            continue
        side = s["side"]
        entry = float(d["c"][i])
        buf = cfg["sl_buf"] * atr_at(i) if sl_mode == "zone" else 0.0
        if side == "long":
            base = float(d["l"][i])
            if sl_mode == "zone":
                base = min(base, s["zbot"])
            sl = base - buf
        else:
            base = float(d["h"][i])
            if sl_mode == "zone":
                base = max(base, s["ztop"])
            sl = base + buf
        risk = abs(entry - sl)
        if risk <= 0 or risk / entry * 100 < cfg["min_risk"]:
            continue
        if (side == "long" and sl >= entry) or (side == "short" and sl <= entry):
            continue
        tp = entry + cfg["rr"] * risk if side == "long" else entry - cfg["rr"] * risk
        res = simulate(d, i, side, sl, tp, max_hold)
        row = dict(
            **meta, Side="Long" if side == "long" else "Short", Why=s["why"],
            Entry=entry, SL=sl, TP=tp,
            _t_in=int(d["t"][i]) + step, Risk_pct=risk / entry * 100,
            Zone_low=s["zbot"], Zone_high=s["ztop"],
        )
        if res is None:
            row.update(Result="OPEN", _t_out=None, Exit=np.nan, R=np.nan, R_gross=np.nan,
                       Ret_pct=np.nan, PnL=np.nan, Bars=np.nan)
            trades.append(row)
            break
        k, reason, px = res
        ret = (px - entry) / entry * 100 if side == "long" else (entry - px) / entry * 100
        ret_net = ret - 2 * cfg["fee"]
        row.update(
            Result={"TP": "Success", "SL": "Failed", "TIMEOUT": "Timeout"}[reason],
            _t_out=int(d["t"][k]) + step, Exit=px,
            R=ret_net / (risk / entry * 100), R_gross=ret / (risk / entry * 100),
            Ret_pct=ret_net,
            PnL=notional * ret_net / 100.0, Bars=k - i,
        )
        trades.append(row)
        busy = k
    return trades


# ───────────────────────────── jobs ─────────────────────────────
def job_main(base, sym, tf, n_test, cfg, strats):
    """Strategies 1, 2, 3, 5 on one coin + timeframe (shared zone computation)."""
    d = fetch_history(base, sym, tf, n_test + ZW)
    if d is None:
        return None
    n = len(d["c"])
    start = max(60, n - n_test)
    zones = build_zones(d, cfg, start)
    step = STEP[tf]
    last_atr = {"v": 1.0}

    def atr_at(i):
        zi = zones[i]
        if zi:
            last_atr["v"] = zi["atr"]
        return last_atr["v"]

    meta = dict(Coin=sym[:-4], TF=tf)
    trades = []
    need_flow = any(s in strats for s in ("touch", "flow_basic", "flow_full"))
    sigs = gen_touch_flow(d, zones, tf, cfg, start) if need_flow else {}
    for name in ("touch", "flow_basic", "flow_full"):
        if name in strats:
            for t in make_trades(d, sigs[name], cfg, step, atr_at, cfg["sl_mode"],
                                 cfg["max_hold"], dict(meta, Strategy=name)):
                trades.append(t)
    if "retest" in strats:
        rs = gen_retest(d, zones, cfg, start)
        trades += make_trades(d, rs, cfg, step, atr_at, cfg["sl_mode"],
                              cfg["max_hold"], dict(meta, Strategy="retest"))
    return dict(trades=trades, candles=n - start)


def job_sweep(base, sym, tfs, n_1m, cfg):
    """Strategy 4 on one coin: 1m candles against each higher-timeframe zone."""
    m = fetch_history(base, sym, "1m", n_1m)
    if m is None:
        return None
    trades = []
    for tf in tfs:
        step = STEP[tf]
        need = int(np.ceil(len(m["c"]) * 60_000 / step)) + 10
        d = fetch_history(base, sym, tf, need + ZW)
        if d is None:
            continue
        j0 = max(60, int(np.searchsorted(d["t"], int(m["t"][0]) - step, side="left")) - 1)
        zones = build_zones(d, cfg, j0)
        sigs = gen_sweep(m, d, zones, tf, cfg)
        # SL for a sweep = the sweep candle's own extreme (atr not needed)
        trades += make_trades(m, sigs, cfg, 60_000, lambda i: 0.0, "candle",
                              cfg["max_hold_1m"],
                              dict(Coin=sym[:-4], TF=f"{tf} zone", Strategy="sweep"))
    return dict(trades=trades, candles=len(m["c"]))


def run_backtest(base, symbols, tfs, n_test, cfg, strats, n_1m=4000, workers=6,
                 progress=None):
    """Runs everything; returns (DataFrame of all trades, failed job count)."""
    jobs = []
    main_strats = [s for s in strats if s != "sweep"]
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        if main_strats:
            for s in symbols:
                for tf in tfs:
                    jobs.append(ex.submit(job_main, base, s, tf, n_test, cfg, main_strats))
        if "sweep" in strats:
            for s in symbols:
                jobs.append(ex.submit(job_sweep, base, s, tfs, n_1m, cfg))
        rows, fail, done = [], 0, 0
        for f in as_completed(jobs):
            done += 1
            try:
                r = f.result()
            except Exception:
                r = None
            if r is None:
                fail += 1
            else:
                rows += r["trades"]
            if progress:
                progress(done, len(jobs))
    finally:
        ex.shutdown(wait=False)
    return finish(rows, cfg), fail


def finish(rows, cfg):
    cols = ["Strategy", "Coin", "TF", "Side", "Result", "Why", "Entry", "SL", "TP", "Exit",
            "R", "R_gross", "Ret_pct", "PnL", "Risk_pct", "Bars", "Zone_low", "Zone_high", "_t_in", "_t_out"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)[cols]
    ist = sc.IST
    df["Entry time"] = pd.to_datetime(df["_t_in"], unit="ms", utc=True).dt.tz_convert(ist)
    df["Exit time"] = pd.to_datetime(df["_t_out"], unit="ms", utc=True).dt.tz_convert(ist)
    return df.sort_values("_t_in", ascending=False).reset_index(drop=True)


# ───────────────────────────── stats ─────────────────────────────
def summarize(df, cfg):
    """Stats of one strategy's trades (OPEN trades are not counted)."""
    closed = df[df["Result"] != "OPEN"]
    n = len(closed)
    cap = float(cfg["capital"])
    out = dict(Trades=n, Success=0, Failed=0, Timeout=0, **{"Win rate %": 0.0},
               **{"Profitable %": 0.0}, **{"Avg R": 0.0}, **{"Avg R (no fees)": 0.0}, **{"Total R": 0.0},
               **{"Profit factor": 0.0}, **{"Net PnL": 0.0}, **{"Return %": 0.0},
               **{"Final capital": cap}, **{"Max DD": 0.0},
               Open=int((df["Result"] == "OPEN").sum()))
    if n == 0:
        return out
    pnl = closed["PnL"].astype(float)
    gp, gl = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    eq = closed.sort_values("_t_out")["PnL"].astype(float).cumsum()
    out.update({
        "Success": int((closed["Result"] == "Success").sum()),
        "Failed": int((closed["Result"] == "Failed").sum()),
        "Timeout": int((closed["Result"] == "Timeout").sum()),
        "Win rate %": round((closed["Result"] == "Success").mean() * 100, 1),
        "Profitable %": round((pnl > 0).mean() * 100, 1),
        "Avg R": round(float(closed["R"].mean()), 2),
        "Avg R (no fees)": round(float(closed["R_gross"].mean()), 2),
        "Total R": round(float(closed["R"].sum()), 1),
        "Profit factor": round(float(gp / gl), 2) if gl > 0 else float("inf") if gp > 0 else 0.0,
        "Net PnL": round(float(pnl.sum()), 0),
        "Return %": round(float(pnl.sum()) / cap * 100, 2),
        "Final capital": round(cap + float(pnl.sum()), 0),
        "Max DD": round(float((eq.cummax() - eq).max()), 0),
    })
    return out
