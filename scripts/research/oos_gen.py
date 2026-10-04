"""周期（予測・保有日数）ごとに、ウォークフォワードのアウトオブサンプル予測を作って保存する。

週次検証と同じ条件（学習 180 日・検証窓 30 日・保有期間ぶんの間隔）で、モデル v3（mkt_* なし）。
乱数 2 通りのスコアを平均して、乱数による揺れを減らす。
"""
import os, sys, time
import numpy as np
import pandas as pd

REPO = r"D:\記憶\プロジェクト\Asset Management（分析）"
OUT = os.path.join(os.environ["TEMP"], "bt", "oos")
os.makedirs(OUT, exist_ok=True)
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import common, db
from pathlib import Path
DBP = Path(os.environ["TEMP"]) / "bt" / "crypto.db"
common.DB_PATH = DBP
db.DB_PATH = DBP

import backtest
import track_performance as tp

HORIZONS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else "2,3,4,5,6,7,10".split(","))]
SEEDS = (42, 7)

for h in HORIZONS:
    t0 = time.time()
    ds, cs_cols = backtest.build_dataset(horizon=h)
    ds = ds.dropna(subset=["y_rank"]).copy()
    dates = np.sort(ds["date"].unique())
    scores = []
    base = None
    for seed in SEEDS:
        backtest.MODEL_PARAMS["random_state"] = seed
        oos = backtest._run_folds(ds, cs_cols, dates, tp.INITIAL_TRAIN_DAYS, tp.TEST_WINDOW,
                                  h, log_folds=False, keep_features=False)
        oos = oos.sort_values(["date", "coin_id"]).reset_index(drop=True)
        if base is None:
            base = oos[["date", "coin_id", "symbol", "fwd_ret"]].copy()
        scores.append(oos["score"].to_numpy())
    base["score"] = np.mean(scores, axis=0)
    path = os.path.join(OUT, f"oos_h{h}.csv")
    base.to_csv(path, index=False)
    print(f"DONE h={h} rows={len(base):,} days={base['date'].nunique()} "
          f"{pd.Timestamp(base['date'].min()).date()}..{pd.Timestamp(base['date'].max()).date()} "
          f"{time.time()-t0:.0f}s", flush=True)
