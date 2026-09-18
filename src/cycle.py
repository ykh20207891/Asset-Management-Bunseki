"""売買サイクル（何日ごとに入れ替えるか）の定義。

2026-09-21 から「5日ごと」に切り替えた。理由（Bybit 上位8・手数料込み・
開始日を全通りずらしたウォークフォワード検証、2026-03-29〜08-17）:

    7日予測・7日保有: +2.21%/週（前半 +2.07 / 後半 +2.35）
    5日予測・5日保有: +3.38%/週（前半 +2.91 / 後半 +3.85）

予測の効き目は日が経つほど薄れる（最初の1日だけで対市場 +0.85pt を稼ぎ、
1日遅れて買うと +2.21% → +1.72%/週 に落ちる）。効き目が残っているうちに
次の予測へ乗り換えた方が、売買回数が増える手数料を払ってもなお有利だった。
学習の重み付けや損切りを変えた全6通りで、前半・後半とも 5日の方が上回った。

サイクルの日は ANCHOR から HORIZON 日おき。曜日は毎回ずれる
（仮想通貨は土日も動くので問題ない）。
"""
from __future__ import annotations

HORIZON = 5
ANCHOR = "2026-09-21"

# 切り替え前（毎週月曜・7日）
LEGACY_HORIZON = 7

# predicted_on がサイクルの日かどうかを判定する SQL 断片
CYCLE_DAY_SQL = (
    f"predicted_on >= '{ANCHOR}' AND "
    f"CAST(julianday(predicted_on) - julianday('{ANCHOR}') AS INTEGER) % {HORIZON} = 0"
)
LEGACY_DAY_SQL = "CAST(strftime('%w', predicted_on) AS INTEGER) = 1"


def latest_cycle(conn, before: str | None = None):
    """直近のサイクル日の予測 (predicted_on, model_tag, horizon) を返す。

    新方式（5日）のサイクル日がまだ来ていなければ、従来の月曜・7日予測を返す。
    こうしておくと、切り替え日までは画面も自動売買も今まで通り動く。
    before を渡すと、それより前のサイクル（＝答え合わせの対象）を返す。
    """
    limit = before or "9999-12-31"

    row = conn.execute(
        f"SELECT predicted_on, model_tag FROM predictions "
        f"WHERE horizon_days = ? AND {CYCLE_DAY_SQL} AND predicted_on < ? "
        f"ORDER BY predicted_on DESC, model_tag DESC LIMIT 1",
        (HORIZON, limit),
    ).fetchone()
    if row:
        return row[0], row[1], HORIZON

    row = conn.execute(
        f"SELECT predicted_on, model_tag FROM predictions "
        f"WHERE horizon_days = ? AND {LEGACY_DAY_SQL} AND predicted_on < ? "
        f"ORDER BY predicted_on DESC, model_tag DESC LIMIT 1",
        (LEGACY_HORIZON, limit),
    ).fetchone()
    if row:
        return row[0], row[1], LEGACY_HORIZON

    # 運用初期など、どちらも無ければ最新の予測で代替する
    row = conn.execute(
        "SELECT predicted_on, model_tag, horizon_days FROM predictions "
        "WHERE predicted_on < ? ORDER BY predicted_on DESC, horizon_days ASC LIMIT 1",
        (limit,),
    ).fetchone()
    return (row[0], row[1], row[2]) if row else None
