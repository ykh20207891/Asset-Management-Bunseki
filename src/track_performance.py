"""週次でウォークフォワード検証を実行し、成績の推移を記録する。

モデルは predict.py が毎日その場で再学習しているため、データが増えれば
学習内容は自動で新しくなる。ただし「それで良くなったのか」は測らないと
分からないので、週1回まとめて検証し、結果を DB に積む。

    python src/track_performance.py            # 月曜のみ実行（日次バッチ用）
    python src/track_performance.py --force    # 曜日に関係なく実行

記録した推移は prediction.json の performance として配信し、アプリで見られる。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import setup_logger  # noqa: E402
from db import connect, init_db  # noqa: E402

LOG = setup_logger("track_performance")

import cycle  # noqa: E402

HORIZON = cycle.HORIZON
# 資産管理アプリの自動売買と同じ条件で測る（Bybit 上場銘柄の上位8を等額）。
# 2026-09-14 までの記録は「全銘柄の上位10」で測っており、条件が違う。
# 8 にした根拠: Bybit 集合・開始日7通りの検証で、上位5と平均リターンは同じ
# （+2.26% と +2.21%/週）だが、最悪の経路が −16.8% → −6.6% と大きく改善した。
TOP_K = 8
INITIAL_TRAIN_DAYS = 180
TEST_WINDOW = 30

# 比較対象にするベースライン（この名前で backtest が結果を返す）
MODEL_LABEL = "model(GBDT)"


def _ensure_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS performance_log (
            measured_on   TEXT PRIMARY KEY,   -- 検証を回した日 (UTC)
            horizon_days  INTEGER NOT NULL,
            n_folds       INTEGER,
            n_features    INTEGER,
            train_rows    INTEGER,
            mean_ic       REAL,               -- モデルの平均IC
            t_stat        REAL,
            hit_rate      REAL,
            top_k_return  REAL,               -- 上位K銘柄の平均リターン(1回あたり)
            vs_universe   REAL,               -- ユニバース等加重との差
            best_baseline TEXT,               -- 最も強かったベースライン名
            best_baseline_return REAL,
            beats_baseline INTEGER,           -- モデルがベースラインに勝ったか
            detail_json   TEXT,
            created_at    TEXT NOT NULL
        )
        """
    )
    # 測定条件（あとから足した列）。条件が違う回を同じ推移として並べないため
    for column, ddl in (("top_k", "INTEGER"), ("universe", "TEXT")):
        try:
            conn.execute(f"ALTER TABLE performance_log ADD COLUMN {column} {ddl}")
        except Exception:  # noqa: BLE001
            pass  # 既にある
    conn.commit()


BAND_EDGES = [(0, .03, "top3"), (.03, .10, "top10"), (.10, .25, "top25"),
              (.25, .50, "mid"), (.50, .75, "low"), (.75, 1.01, "bottom")]


def compute_band_stats(oos) -> dict:
    """順位帯ごとに「保有期間の後に上がっていた割合」と平均リターンを出す。

    画面の「N日後に上昇 52%」の元になる数字。oos は検証で得た
    アウトオブサンプル予測（date, score, fwd_ret）。渡された銘柄集合の中での順位で帯を決める。
    """
    df = oos.dropna(subset=["score", "fwd_ret"]).copy()
    df["rank"] = df.groupby("date")["score"].rank(ascending=False, method="first")
    df["pct"] = df["rank"] / df.groupby("date")["score"].transform("size")

    bands = {}
    for lo, hi, key in BAND_EDGES:
        part = df[(df["pct"] > lo) & (df["pct"] <= hi)]["fwd_ret"]
        if len(part) == 0:
            continue
        bands[key] = {
            "upRate": round(float((part > 0).mean()), 4),
            "avgReturn": round(float(part.mean()), 5),
            "samples": int(len(part)),
        }
    bands["overall"] = {
        "upRate": round(float((df["fwd_ret"] > 0).mean()), 4),
        "avgReturn": round(float(df["fwd_ret"].mean()), 5),
        "samples": int(len(df)),
        "periods": int(df["date"].nunique()),
    }
    return bands


def _save_band_stats(conn, horizon: int, universe: str, bands: dict) -> None:
    """reports/ はクラウドでは保存されないので、DB に持つ。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS band_stats (
            horizon_days INTEGER NOT NULL,
            universe     TEXT NOT NULL,
            measured_on  TEXT NOT NULL,
            bands_json   TEXT NOT NULL,
            PRIMARY KEY (horizon_days, universe)
        )
        """
    )
    conn.execute(
        "INSERT INTO band_stats (horizon_days, universe, measured_on, bands_json) "
        "VALUES (?, ?, date('now'), ?) "
        "ON CONFLICT(horizon_days, universe) DO UPDATE SET "
        "measured_on = excluded.measured_on, bands_json = excluded.bands_json",
        (horizon, universe, json.dumps(bands, ensure_ascii=False)),
    )
    conn.commit()


def _clean_detail(results: dict) -> dict:
    """DataFrame を含む項目を落として JSON にできる形にする。"""
    out = {}
    for label, summ in results.items():
        item = {k: v for k, v in summ.items() if k != "portfolio"}
        portfolio = summ.get("portfolio") or {}
        item["portfolio"] = {
            k: v for k, v in portfolio.items() if k != "periods"
        }
        out[label] = item
    return out


def _is_monday() -> bool:
    return datetime.now(timezone.utc).weekday() == 0


def run(force: bool = False) -> dict | None:
    if not force and not _is_monday():
        LOG.info("月曜以外のため検証をスキップします")
        return None

    init_db()
    measured_on = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    with connect() as conn:
        _ensure_table(conn)
        done = conn.execute(
            "SELECT 1 FROM performance_log WHERE measured_on = ?", (measured_on,)
        ).fetchone()
        if done and not force:
            LOG.info("%s は既に検証済みです", measured_on)
            return None

    LOG.info("ウォークフォワード検証を開始します（時間がかかります）")

    import backtest
    import bybit_listing

    with connect() as conn:
        tradable = set(bybit_listing.tradable_map(conn))
    universe = "bybit" if tradable else "all"

    bt = backtest.run_backtest(
        horizon=HORIZON,
        top_k=TOP_K,
        initial_train_days=INITIAL_TRAIN_DAYS,
        test_window=TEST_WINDOW,
        universe_ids=tradable or None,
    )

    results = bt.get("results") or {}
    model = results.get(MODEL_LABEL)
    if not model:
        LOG.warning("モデルの検証結果が取得できませんでした")
        return None

    portfolio = model.get("portfolio") or {}
    model_return = portfolio.get("top_mean")
    excess = portfolio.get("excess_vs_universe")

    # モデル以外で最も成績が良かったベースラインを拾う
    best_label, best_return = None, None
    for label, summ in results.items():
        if label == MODEL_LABEL:
            continue
        ret = (summ.get("portfolio") or {}).get("top_mean")
        if ret is None:
            continue
        if best_return is None or ret > best_return:
            best_label, best_return = label, ret

    row = {
        "measured_on": measured_on,
        "horizon_days": HORIZON,
        "n_folds": bt.get("n_folds"),
        "n_features": bt.get("n_features"),
        "train_rows": int(len(bt.get("oos", []))) if bt.get("oos") is not None else None,
        "mean_ic": model.get("mean_ic"),
        "t_stat": model.get("t_stat"),
        "hit_rate": model.get("hit_rate"),
        "top_k_return": model_return,
        "vs_universe": excess,
        "best_baseline": best_label,
        "best_baseline_return": best_return,
        "beats_baseline": (
            1
            if (model_return is not None and best_return is not None and model_return > best_return)
            else 0
        ),
    }

    # 順位帯ごとの上昇実績も同じ検証結果から測り直して保存する
    try:
        with connect() as conn:
            _save_band_stats(conn, HORIZON, universe, compute_band_stats(bt["oos"]))
    except Exception as e:  # noqa: BLE001
        LOG.warning("順位帯の実績を保存できませんでした: %s", e)

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO performance_log
              (measured_on, horizon_days, n_folds, n_features, train_rows,
               mean_ic, t_stat, hit_rate, top_k_return, vs_universe,
               best_baseline, best_baseline_return, beats_baseline,
               detail_json, top_k, universe, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(measured_on) DO UPDATE SET
              top_k=excluded.top_k, universe=excluded.universe,
              mean_ic=excluded.mean_ic, t_stat=excluded.t_stat,
              hit_rate=excluded.hit_rate, top_k_return=excluded.top_k_return,
              vs_universe=excluded.vs_universe,
              best_baseline=excluded.best_baseline,
              best_baseline_return=excluded.best_baseline_return,
              beats_baseline=excluded.beats_baseline,
              detail_json=excluded.detail_json, created_at=datetime('now')
            """,
            (
                row["measured_on"], row["horizon_days"], row["n_folds"],
                row["n_features"], row["train_rows"], row["mean_ic"],
                row["t_stat"], row["hit_rate"], row["top_k_return"],
                row["vs_universe"], row["best_baseline"],
                row["best_baseline_return"], row["beats_baseline"],
                json.dumps(_clean_detail(results), ensure_ascii=False, default=str),
                TOP_K,
                universe,
            ),
        )
        conn.commit()

    LOG.info(
        "検証結果を記録: IC=%.4f / 上位%d平均=%.2f%% / ベースライン最良=%s",
        row["mean_ic"] or 0, TOP_K,
        (row["top_k_return"] or 0) * 100, row["best_baseline"],
    )
    return row


def history(limit: int = 12) -> list[dict]:
    """記録済みの成績推移（新しい順）。"""
    with connect() as conn:
        try:
            _ensure_table(conn)
            rows = conn.execute(
                "SELECT measured_on, mean_ic, t_stat, hit_rate, top_k_return, "
                "vs_universe, best_baseline, best_baseline_return, beats_baseline, "
                "top_k, universe, horizon_days "
                "FROM performance_log ORDER BY measured_on DESC LIMIT ?",
                (limit,),
            ).fetchall()
        except Exception:  # noqa: BLE001
            return []

    return [
        {
            "measuredOn": r[0],
            "meanIc": r[1],
            "tStat": r[2],
            "hitRate": r[3],
            "topKReturn": r[4],
            "vsUniverse": r[5],
            "bestBaseline": r[6],
            "bestBaselineReturn": r[7],
            "beatsBaseline": bool(r[8]),
            # 測定条件。古い記録は「全銘柄の上位10」
            "topK": r[9] if r[9] is not None else 10,
            "universe": r[10] or "all",
            # ％は「1回の保有期間（この日数）あたり」の平均
            "horizonDays": r[11] or 7,
        }
        for r in rows
    ]


def main() -> int:
    p = argparse.ArgumentParser(description="週次でモデルの成績を検証・記録する")
    p.add_argument("--force", action="store_true", help="曜日に関係なく実行")
    args = p.parse_args()
    run(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
