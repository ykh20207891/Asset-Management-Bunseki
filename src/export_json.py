"""最新のランキング予測を JSON で書き出す。

資産管理アプリ（Cloudflare Workers）が GitHub Pages 経由で取得して表示するための
軽量な受け渡しファイル。ダッシュボード HTML とは別に、データだけを提供する。

    python src/export_json.py --out site/prediction.json

出力例:
    {
      "predictedOn": "2026-08-25",
      "horizonDays": 7,
      "modelTag": "gbdt_h7_v1",
      "generatedAt": "2026-08-26T00:10:00Z",
      "items": [
        {"rank":1,"symbol":"GWEI","name":"ETHGas","score":0.6689,
         "price":0.0224929,"change7d":-5.5,"change30d":-9.1,"marketCap":45000000}
      ]
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REPORT_DIR, ensure_dirs, setup_logger  # noqa: E402
import cycle  # noqa: E402
from db import connect, init_db  # noqa: E402

LOG = setup_logger("export_json")

DEFAULT_TOP_N = 30

# 「その順位帯の銘柄が、保有期間の後に上がっていた実績割合」。
# モデルは上昇/下落そのものを予測しないため、断定の代わりにこの実績を示す。
# 本来の値は毎週の検証(track_performance)が測って DB の band_stats に保存する。
# ここにあるのは、それがまだ無いとき用の既定値（同じ方法で測った実測値）。
DEFAULT_BANDS_BY_HORIZON = {
    # 7日予測・全銘柄（39,268件・142期間）
    7: {
        "top3": {"upRate": 0.522, "avgReturn": -0.0026},
        "top10": {"upRate": 0.499, "avgReturn": 0.0041},
        "top25": {"upRate": 0.467, "avgReturn": -0.0004},
        "mid": {"upRate": 0.464, "avgReturn": -0.0042},
        "low": {"upRate": 0.460, "avgReturn": -0.0050},
        "bottom": {"upRate": 0.440, "avgReturn": -0.0128},
        "overall": {"upRate": 0.462, "avgReturn": -0.0054},
    },
    # 5日予測(v2)・Bybit 上場銘柄（25,133件・147期間 / 2026-09-18 測定）
    5: {
        "top3": {"upRate": 0.5226, "avgReturn": 0.02224},
        "top10": {"upRate": 0.5017, "avgReturn": 0.0062},
        "top25": {"upRate": 0.4727, "avgReturn": 0.00159},
        "mid": {"upRate": 0.4595, "avgReturn": -0.00092},
        "low": {"upRate": 0.4562, "avgReturn": -0.00247},
        "bottom": {"upRate": 0.4421, "avgReturn": -0.00829},
        "overall": {"upRate": 0.461, "avgReturn": -0.00163},
    },
}
DEFAULT_BANDS = DEFAULT_BANDS_BY_HORIZON[7]


def _band_key(rank: int, total: int) -> str:
    """順位を検証済みの順位帯に対応づける。"""
    if total <= 0:
        return "overall"
    pct = rank / total
    if pct <= 0.03:
        return "top3"
    if pct <= 0.10:
        return "top10"
    if pct <= 0.25:
        return "top25"
    if pct <= 0.50:
        return "mid"
    if pct <= 0.75:
        return "low"
    return "bottom"


def compute_bands(conn, horizon: int) -> dict:
    """毎週の検証が DB に保存した実績を使い、無ければ期間に合った既定値を使う。"""
    try:
        row = conn.execute(
            "SELECT bands_json FROM band_stats WHERE horizon_days = ? "
            "ORDER BY (universe = 'bybit') DESC, measured_on DESC LIMIT 1",
            (horizon,),
        ).fetchone()
        if row:
            return json.loads(row[0])
    except Exception:  # noqa: BLE001
        pass  # テーブル未作成（まだ一度も検証していない）
    return DEFAULT_BANDS_BY_HORIZON.get(horizon, DEFAULT_BANDS)


def _price_on_or_after(conn, coin_id: str, date: str) -> float | None:
    """指定日以降で最も近い日の価格。snapshots に無ければ price_history を見る。"""
    row = conn.execute(
        "SELECT price FROM snapshots WHERE coin_id = ? AND snapshot_date >= ? "
        "ORDER BY snapshot_date ASC LIMIT 1",
        (coin_id, date),
    ).fetchone()
    if row and row[0] is not None:
        return float(row[0])

    row = conn.execute(
        "SELECT price FROM price_history WHERE coin_id = ? AND date >= ? "
        "ORDER BY date ASC LIMIT 1",
        (coin_id, date),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _performance_history() -> list:
    """週次検証の成績推移。未記録なら空リスト。"""
    try:
        import track_performance
        return track_performance.history(limit=12)
    except Exception:  # noqa: BLE001
        return []


def _tradable(conn) -> dict[str, str]:
    """coin_id -> Bybit 上のシンボル。一覧が無ければ空（＝絞り込まない）。"""
    try:
        import bybit_listing
        return bybit_listing.tradable_map(conn)
    except Exception:  # noqa: BLE001
        return {}


def gather_short_candidates(conn, horizon: int, predicted_on: str, model_tag: str,
                            limit: int = 15) -> list[dict]:
    """ショート（無期限先物の売り）の候補。Bybit で売買でき、先物がある銘柄の
    うちスコアが最も低いものから順に返す。

    検証（Bybit・5日・v2・開始日5通り）では、下位8の売りは +1.21%/5日、
    上位8の買いと組み合わせると市場との連動が 0.76 → 0.20 に下がり、
    相場が下げた回でもプラスになった。アプリ側は模擬で並走させて比較する。
    """
    try:
        import bybit_listing
        perps = bybit_listing.perp_map(conn)
        tradable = bybit_listing.tradable_map(conn)
    except Exception:  # noqa: BLE001
        return []
    if not perps or not tradable:
        return []

    rows = conn.execute(
        """
        SELECT p.rank, p.symbol, p.score, p.price_at_pred, p.coin_id, s.name, s.image_url
        FROM predictions p
        LEFT JOIN snapshots s ON s.coin_id = p.coin_id AND s.snapshot_date = p.predicted_on
        WHERE p.horizon_days = ? AND p.predicted_on = ? AND p.model_tag = ?
        ORDER BY p.rank DESC
        """,
        (horizon, predicted_on, model_tag),
    ).fetchall()

    out = []
    for r in rows:
        coin_id = r[4]
        if coin_id not in tradable or coin_id not in perps:
            continue
        out.append(
            {
                "rank": len(out) + 1,          # 下から数えた順位（1 = 最もスコアが低い）
                "modelRank": r[0],
                "symbol": (r[1] or "").upper(),
                "bybitSymbol": tradable[coin_id],
                "perpSymbol": perps[coin_id],
                "coinId": coin_id,
                "name": r[5] or r[1],
                "score": round(r[2], 4) if r[2] is not None else None,
                "price": r[3],
                "iconUrl": r[6],
            }
        )
        if len(out) >= limit:
            break
    return out


def _load_meta(conn) -> dict[str, dict]:
    """coin_meta（チェーン/取引所）を辞書で返す。未取得なら空。"""
    meta: dict[str, dict] = {}
    try:
        for m in conn.execute("SELECT coin_id, chains, exchanges FROM coin_meta"):
            meta[m[0]] = {
                "chains": json.loads(m[1] or "[]"),
                "exchanges": json.loads(m[2] or "[]"),
            }
    except Exception:  # noqa: BLE001
        pass
    return meta


def gather_review(conn, horizon: int, top_n: int, current_on: str | None) -> dict | None:
    """1つ前のサイクルの予測について、実際にどうなったかを集計する（答え合わせ）。

    切り替え直後は 1つ前が従来の月曜・7日予測になるので、期間は見つかった予測に合わせる。
    """
    row = cycle.latest_cycle(conn, before=current_on)
    if not row:
        return None
    horizon = row[2]

    predicted_on, model_tag = row[0], row[1]

    # 予測から horizon 日後の日付
    target = conn.execute(
        "SELECT date(?, ?)", (predicted_on, f"+{horizon} day")
    ).fetchone()[0]

    rows = conn.execute(
        """
        SELECT p.rank, p.symbol, p.coin_id, p.price_at_pred, s.name, s.image_url
        FROM predictions p
        LEFT JOIN (
            SELECT coin_id, name, image_url, MAX(snapshot_date) FROM snapshots
            GROUP BY coin_id
        ) s ON s.coin_id = p.coin_id
        WHERE p.horizon_days = ? AND p.predicted_on = ? AND p.model_tag = ?
        ORDER BY p.rank ASC
        """,
        (horizon, predicted_on, model_tag),
    ).fetchall()

    # 今週の予測と同じく、Bybit で売買できる銘柄だけを対象にする
    tradable = _tradable(conn)
    if tradable:
        rows = [r for r in rows if r[2] in tradable]

    # 市場平均（その週に売買できた全銘柄の単純平均）。
    # 戦略の良し悪しは損益の絶対額ではなく「市場平均をどれだけ上回ったか」で見る。
    # 相場全体が下げた週のマイナスと、銘柄選びの失敗とを区別するための基準。
    market_changes = []
    for r in rows:
        before = r[3]
        after = _price_on_or_after(conn, r[2], target)
        if before and after and before > 0:
            market_changes.append((after - before) / before * 100)
    market_avg = (
        round(sum(market_changes) / len(market_changes), 2) if market_changes else None
    )
    market_count = len(market_changes)

    rows = rows[:top_n]

    meta = _load_meta(conn)
    items = []
    ups = 0
    total_ret = 0.0
    scored = 0
    # チェーン別に「上がった数 / 対象数 / 平均騰落率」を集計する
    by_chain: dict[str, dict] = {}

    for position, r in enumerate(rows, start=1):
        model_rank, symbol, coin_id, price_before = r[0], r[1], r[2], r[3]
        # 絞り込んだ場合は、その中での順位を振り直す
        rank = position if tradable else model_rank
        price_after = _price_on_or_after(conn, coin_id, target)

        change = None
        if price_before and price_after and price_before > 0:
            change = (price_after - price_before) / price_before * 100
            scored += 1
            total_ret += change
            if change > 0:
                ups += 1

        chains = meta.get(coin_id, {}).get("chains", [])
        exchanges = meta.get(coin_id, {}).get("exchanges", [])

        if change is not None:
            for chain in chains:
                agg = by_chain.setdefault(chain, {"up": 0, "total": 0, "sum": 0.0})
                agg["total"] += 1
                agg["sum"] += change
                if change > 0:
                    agg["up"] += 1

        items.append(
            {
                "rank": rank,
                "modelRank": model_rank,
                "symbol": (symbol or "").upper(),
                "bybitSymbol": tradable.get(coin_id),
                "name": r[4] or symbol,
                "iconUrl": r[5],
                "priceBefore": price_before,
                "priceAfter": price_after,
                "changePct": round(change, 2) if change is not None else None,
                "chains": chains,
                "exchanges": exchanges,
            }
        )

    if not items:
        return None

    return {
        "predictedOn": predicted_on,
        "evaluatedOn": target,
        "items": items,
        "upCount": ups,
        "scored": scored,
        "upRate": round(ups / scored, 4) if scored else None,
        "avgChangePct": round(total_ret / scored, 2) if scored else None,
        # 同じ週の市場平均（売買できた全銘柄）と、その銘柄数
        "marketChangePct": market_avg,
        "marketCount": market_count,
        # 自動売買と同じ「上位8銘柄・等額」だった場合の結果（手数料前）
        "top8ChangePct": (
            round(
                sum(i["changePct"] for i in items[:8] if i["changePct"] is not None)
                / max(1, sum(1 for i in items[:8] if i["changePct"] is not None)),
                2,
            )
            if any(i["changePct"] is not None for i in items[:8])
            else None
        ),
        # チェーン別の成績（上がった数が多い順）
        "byChain": sorted(
            [
                {
                    "chain": chain,
                    "up": v["up"],
                    "total": v["total"],
                    "avgChangePct": round(v["sum"] / v["total"], 2),
                }
                for chain, v in by_chain.items()
            ],
            key=lambda x: (-x["up"], -x["avgChangePct"]),
        ),
    }


def gather_latest(conn, horizon: int, top_n: int) -> dict:
    """表示用の予測（直近の月曜分）を、必要な情報とともに取り出す。"""
    head = cycle.latest_cycle(conn)

    if not head:
        return {"predictedOn": None, "horizonDays": horizon, "items": []}

    predicted_on, model_tag, horizon = head[0], head[1], head[2]

    # その日の予測対象の総数（順位帯の判定に使う）
    total = conn.execute(
        """
        SELECT COUNT(*) FROM predictions
        WHERE horizon_days = ? AND predicted_on = ? AND model_tag = ?
        """,
        (horizon, predicted_on, model_tag),
    ).fetchone()[0]

    bands = compute_bands(conn, horizon)

    # 最新スナップショットから価格変化率・時価総額・銘柄名を補う
    rows = conn.execute(
        """
        SELECT p.rank, p.symbol, p.score, p.price_at_pred, p.coin_id,
               s.name, s.pct_7d, s.pct_30d, s.market_cap, s.image_url
        FROM predictions p
        LEFT JOIN snapshots s
          ON s.coin_id = p.coin_id
         AND s.snapshot_date = p.predicted_on
        WHERE p.horizon_days = ?
          AND p.predicted_on = ?
          AND p.model_tag = ?
        ORDER BY p.rank ASC
        """,
        (horizon, predicted_on, model_tag),
    ).fetchall()

    # 資産管理アプリは Bybit でしか売買しないので、Bybit 現物(USDT)に上場している
    # 銘柄だけをランキングする。モデルの全体順位は modelRank として残す。
    tradable = _tradable(conn)
    if tradable:
        rows = [r for r in rows if r[4] in tradable]
    tradable_total = len(rows)
    rows = rows[:top_n]

    # チェーン/取引所（coin_meta。未取得なら空で出す）
    meta = _load_meta(conn)

    items = []
    for position, r in enumerate(rows, start=1):
        # 順位帯（過去実績）は、表示しているランキングの中での位置で判定する
        # （Bybit 銘柄に絞っているなら、Bybit 銘柄の中で上位何 % か）
        band_key = (
            _band_key(position, tradable_total) if tradable else _band_key(r[0] or 0, total)
        )
        band = bands.get(band_key, bands.get("overall", {}))
        items.append(
            {
                "rank": position if tradable else r[0],
                "modelRank": r[0],
                # Bybit 上のシンボル。BABY→BABY1 のように CoinGecko と違うことがある
                "bybitSymbol": tradable.get(r[4]),
                # 順位帯と、その帯の過去実績（7日後に上昇していた割合・平均リターン）
                "band": band_key,
                "upRate": band.get("upRate"),
                "bandAvgReturn": band.get("avgReturn"),
                "symbol": (r[1] or "").upper(),
                "coinId": r[4],
                "name": r[5] or r[1],
                "score": round(r[2], 4) if r[2] is not None else None,
                "price": r[3],
                "change7d": r[6],
                "change30d": r[7],
                "marketCap": r[8],
                # アプリ側はランキング銘柄のアイコンを持っていないため、
                # CoinGecko のロゴURLをそのまま渡す
                "iconUrl": r[9],
                "chains": meta.get(r[4], {}).get("chains", []),
                "exchanges": meta.get(r[4], {}).get("exchanges", []),
            }
        )

    # AI が生成した当日の市場サマリー（無ければ省略）
    summary = None
    try:
        row = conn.execute(
            "SELECT summary_date, summary FROM ai_daily_summary "
            "ORDER BY summary_date DESC LIMIT 1"
        ).fetchone()
        if row:
            summary = {"date": row[0], "text": row[1]}
    except Exception:  # noqa: BLE001
        pass  # テーブル未作成（AI未利用）なら単に省略する

    return {
        "predictedOn": predicted_on,
        # 何日保有する前提の予測か。アプリはこの日数で決済日を決める
        "horizonDays": horizon,
        # 次にランキングが入れ替わる日
        "nextUpdateOn": conn.execute(
            "SELECT date(?, ?)", (predicted_on, f"+{horizon} day")
        ).fetchone()[0],
        # 従来の7日予測を出している間だけ、切り替え後の日数を知らせる
        "nextHorizonDays": cycle.HORIZON if horizon != cycle.HORIZON else None,
        "modelTag": model_tag,
        "summary": summary,
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "universeSize": total,
        # "bybit" = Bybit 現物(USDT)上場銘柄のみ / "all" = 絞り込みなし（一覧の取得失敗時）
        "universe": "bybit" if tradable else "all",
        "tradableCount": len(tradable) if tradable else None,
        # 画面で「実績ではこうだった」と示すための基準値
        "baseline": bands.get("overall", {}),
        # 先週の予測が実際どうなったかの答え合わせ
        "review": gather_review(conn, horizon, len(items) or top_n, predicted_on),
        # 週次検証で記録した成績の推移
        "performance": _performance_history(),
        "items": items,
        # ショート候補（スコアが最も低い順・先物のある銘柄のみ）
        "shortItems": gather_short_candidates(conn, horizon, predicted_on, model_tag),
    }


def run_export(out_path: str | None, horizon: int = cycle.HORIZON,
               top_n: int = DEFAULT_TOP_N) -> Path:
    init_db()
    ensure_dirs()

    with connect() as conn:
        data = gather_latest(conn, horizon, top_n)

    target = Path(out_path) if out_path else (REPORT_DIR / "prediction.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    LOG.info(
        "予測 JSON を生成: %s (%d件 / 基準日 %s)",
        target,
        len(data["items"]),
        data.get("predictedOn"),
    )
    return target


def main() -> int:
    p = argparse.ArgumentParser(description="ランキング予測を JSON で書き出す")
    p.add_argument("--out", default=None, help="出力先（既定: reports/prediction.json）")
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--top", type=int, default=DEFAULT_TOP_N)
    args = p.parse_args()

    run_export(args.out, horizon=args.horizon, top_n=args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
