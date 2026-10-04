"""主要な比較の「差の大きさと不確かさ」。

同じ周期 H の中の比較（銘柄数・ロングのみ vs 併用）は同じ日付で対になるので、
各サイクルの差の平均と標準誤差（開始日 1 通りあたりのサイクル数で割る＝重なりを数えない保守的な扱い）。
周期どうしの比較は対にならないので、それぞれの平均と標準誤差を並べる。
"""
import json, os, math
import numpy as np
import pandas as pd
exec(open(os.path.join(os.path.dirname(__file__), "grid.py"), encoding="utf-8").read().split("results = []")[0])


def series(df, h, k, slip=0.0):
    by_date = {d: g for d, g in df.groupby("date")}
    out = {}
    for o, sched in enumerate(schedules(df["date"], h)):
        for d in sched:
            g = by_date[d]
            u = g[g["spot_ok"]]
            if len(u) < 30:
                continue
            mkt = u["ret"].mean()
            top = u.nlargest(k, "score")
            rl = top["ret"].mean() - SPOT_FEE - slip
            p = g[g["perp_ok"]]
            p = p[~p["coin_id"].isin(top["coin_id"])]
            rs = (-p.nsmallest(k, "score")["ret"].mean() - PERP_FEE - slip - FUNDING_YEAR * h / 365) if len(p) >= k else np.nan
            # 翌サイクルに残る銘柄の割合（持ち越しで節約できる手数料の目安）
            out[(o, d)] = {"long": rl, "short": rs, "ls": 0.5 * rl + 0.5 * rs, "mkt": mkt,
                           "set": set(top["coin_id"])}
    return out


def summarize(diff, n_eff):
    diff = np.asarray([x for x in diff if x == x])
    m = diff.mean(); se = diff.std(ddof=1) / math.sqrt(n_eff)
    return m, se


rows = []
dfs = {h: load(h) for h in HORIZONS}
print("== 周期ごと: ロング上位8 の対市場超過（5日換算%）と標準誤差・同じく上位5")
for k in (5, 8):
    for h in HORIZONS:
        s = series(dfs[h], h, k)
        ex = [(v["long"] - v["mkt"]) * 5 / h for v in s.values()]
        n = len(schedules(dfs[h]["date"], h)[0])
        m, se = summarize(ex, n)
        print(f"K={k} H={h:>2}: 超過 {m*100:+.2f} ± {se*100:.2f}  (n={n})")

print("\n== 同じ周期での対比較（差の平均 ± 標準誤差、5日換算%）")
for h in (3, 4, 5, 6):
    s5, s8 = series(dfs[h], h, 5), series(dfs[h], h, 8)
    n = len(schedules(dfs[h]["date"], h)[0])
    d = [(s5[key]["long"] - s8[key]["long"]) * 5 / h for key in s8 if key in s5]
    m, se = summarize(d, n)
    print(f"H={h}: 上位5 − 上位8 = {m*100:+.2f} ± {se*100:.2f}")
    d = [(s8[key]["ls"] - s8[key]["long"]) * 5 / h for key in s8]
    m, se = summarize(d, n)
    print(f"H={h}: 併用(半々) − ロングのみ（上位8） = {m*100:+.2f} ± {se*100:.2f}")

print("\n== 市場との連動（上位8・5日）と最大下落")
s = series(dfs[5], 5, 8)
L = np.array([v["long"] for v in s.values()]); LS = np.array([v["ls"] for v in s.values()]); M = np.array([v["mkt"] for v in s.values()])
ok = ~np.isnan(LS)
print("相関 ロング-市場 %.2f / 併用-市場 %.2f" % (np.corrcoef(L, M)[0, 1], np.corrcoef(LS[ok], M[ok])[0, 1]))
print("1サイクルの標準偏差 ロング %.2f%% / 併用 %.2f%%" % (L.std() * 100, LS[ok].std() * 100))

print("\n== 翌サイクルにも上位に残る割合（持ち越しで省ける手数料の目安）")
for h, k in ((4, 8), (5, 8), (5, 5)):
    s = series(dfs[h], h, k)
    by_off = {}
    for (o, d), v in s.items():
        by_off.setdefault(o, []).append((d, v["set"]))
    keep = []
    for lst in by_off.values():
        lst.sort()
        for (d1, a), (d2, b) in zip(lst, lst[1:]):
            keep.append(len(a & b) / k)
    print(f"H={h} K={k}: {np.mean(keep)*100:.0f}% が残る → 手数料の節約 約 {np.mean(keep)*0.2:.2f}%/回")
