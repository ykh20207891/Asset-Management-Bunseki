"""周期 H × 銘柄数 K の総当たり評価（時点つき Bybit・手数料込み・開始日すべて）。

入力: %TEMP%/bt/oos/oos_h{H}.csv（oos_gen.py）, data/bybit_meta.json
出力: 標準出力に表、%TEMP%/bt/grid.json に全結果
"""
import json, os, sys, math
import numpy as np
import pandas as pd

REPO = r"D:\記憶\プロジェクト\Asset Management（分析）"
OOS = os.path.join(os.environ["TEMP"], "bt", "oos")
meta = json.load(open(os.path.join(REPO, "data", "bybit_meta.json"), encoding="utf-8"))["coins"]
spot_on = {k: v.get("spotListedOn") for k, v in meta.items() if v.get("spotListedOn")}
perp_on = {k: (v.get("perpLaunchedOn") or "2000-01-01") for k, v in meta.items() if v.get("perpSymbol")}

SPOT_FEE = 0.002          # 現物 往復 0.1%×2
PERP_FEE = 0.0011         # 先物 往復 0.055%×2
FUNDING_YEAR = 0.10       # ショートが払う資金調達率（年率の想定）
HORIZONS = [2, 3, 4, 5, 6, 7, 10]
KS = [3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
SLIPS = [0.0, 0.003]      # 追加コスト（板の薄さ・スプレッド）往復


def load(h):
    df = pd.read_csv(os.path.join(OOS, f"oos_h{h}.csv"), parse_dates=["date"])
    df["d"] = df["date"].dt.strftime("%Y-%m-%d")
    df["ret"] = np.expm1(df["fwd_ret"])
    df["spot_ok"] = df["coin_id"].map(spot_on).fillna("9999") <= df["d"]
    df["perp_ok"] = df["coin_id"].map(perp_on).fillna("9999") <= df["d"]
    return df


def schedules(dates, h):
    """開始日をずらした全通りのリバランス日（暦日で h 日おき）。"""
    dates = sorted(pd.to_datetime(pd.Series(dates).unique()))
    first, last = dates[0], dates[-1]
    have = set(dates)
    out = []
    for o in range(h):
        s, d = [], first + pd.Timedelta(days=o)
        while d <= last:
            if d in have:
                s.append(d)
            d += pd.Timedelta(days=h)
        out.append(s)
    return out


def max_dd(rets):
    eq = np.cumprod(1 + np.asarray(rets))
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return float((eq / peak - 1).min()) if len(eq) else 0.0


results = []
per_cycle = {}
for h in HORIZONS:
    path = os.path.join(OOS, f"oos_h{h}.csv")
    if not os.path.exists(path):
        print("skip h", h); continue
    df = load(h)
    by_date = {d: g for d, g in df.groupby("date")}
    scheds = schedules(df["date"], h)
    all_dates = sorted(by_date)
    mid = all_dates[len(all_dates) // 2]
    for k in KS:
        for slip in SLIPS:
            for side in ("long", "short", "ls"):
                offs = []
                for sched in scheds:
                    rows = []
                    for d in sched:
                        g = by_date[d]
                        u = g[g["spot_ok"]]
                        if len(u) < 30:
                            continue
                        mkt = u["ret"].mean()
                        top = u.nlargest(k, "score")
                        r_long = top["ret"].mean() - SPOT_FEE - slip
                        p = g[g["perp_ok"]]
                        p = p[~p["coin_id"].isin(top["coin_id"])]
                        if len(p) >= k:
                            bot = p.nsmallest(k, "score")
                            r_short = -bot["ret"].mean() - PERP_FEE - slip - FUNDING_YEAR * h / 365
                        else:
                            r_short = np.nan
                        r = {"long": r_long, "short": r_short,
                             "ls": 0.5 * r_long + 0.5 * r_short}[side]
                        if np.isnan(r):
                            continue
                        rows.append((d, r, mkt))
                    if rows:
                        offs.append(rows)
                if not offs:
                    continue
                g_off, dd_off, ex_off, first_half, second_half, win = [], [], [], [], [], []
                cyc = []
                for rows in offs:
                    r = np.array([x[1] for x in rows]); m = np.array([x[2] for x in rows])
                    lr = np.log1p(np.clip(r, -0.99, None))
                    g_off.append(lr.mean() / h)
                    dd_off.append(max_dd(r))
                    ex_off.append((r - m).mean() / h)
                    win.append(float((r > m).mean()))
                    for (d, rr, mm) in rows:
                        (first_half if d < mid else second_half).append(math.log1p(max(rr, -0.99)) / h)
                    cyc.extend(lr.tolist())
                gd = float(np.mean(g_off))
                sd = float(np.std(cyc, ddof=1)) / math.sqrt(h) if len(cyc) > 2 else float("nan")
                res = {
                    "h": h, "k": k, "slip": slip, "side": side,
                    "cycles": int(np.mean([len(o) for o in offs])),
                    "daily_g": gd,
                    "annual": math.expm1(gd * 365),
                    "per5d": math.expm1(gd * 5),
                    "excess_per5d": float(np.mean(ex_off)) * 5,
                    "sharpe": gd / sd * math.sqrt(365) if sd and sd == sd else float("nan"),
                    "dd_mean": float(np.mean(dd_off)), "dd_worst": float(np.min(dd_off)),
                    "g_min_off": float(np.min(g_off)), "g_max_off": float(np.max(g_off)),
                    "first": math.expm1(np.mean(first_half) * 5) if first_half else float("nan"),
                    "second": math.expm1(np.mean(second_half) * 5) if second_half else float("nan"),
                    "win_vs_mkt": float(np.mean(win)),
                }
                results.append(res)
    print("done h", h, flush=True)

json.dump(results, open(os.path.join(os.environ["TEMP"], "bt", "grid.json"), "w"), ensure_ascii=False)


def table(side, slip, metric, fmt="{:+.2f}", scale=100):
    hs = sorted({r["h"] for r in results})
    print(f"\n== {side} / 追加コスト {slip*100:.1f}% / {metric}（5日あたり%換算・手数料込み）")
    print("K\\H " + "".join(f"{h:>8}" for h in hs))
    for k in KS:
        line = f"{k:>3} "
        for h in hs:
            r = next((x for x in results if x["h"] == h and x["k"] == k and x["slip"] == slip and x["side"] == side), None)
            line += f"{(fmt.format(r[metric]*scale) if r else '   -'):>8}"
        print(line)


for side in ("long", "ls"):
    for slip in SLIPS:
        table(side, slip, "per5d")
    table(side, 0.0, "excess_per5d")
    table(side, 0.0, "first")
    table(side, 0.0, "second")
    table(side, 0.0, "dd_worst")
    table(side, 0.0, "sharpe", fmt="{:.2f}", scale=1)
    table(side, 0.0, "g_min_off", fmt="{:+.2f}", scale=500)
