"""Bybit の上場日（現物）と無期限先物の有無・開始日を取得して data/bybit_meta.json に保存する。

用途:
  1. 過去検証で「その日にまだ上場していなかった銘柄」を買えたことにしない（先読み防止）
  2. ショート（無期限先物の売り）ができる銘柄かどうか

GitHub Actions のランナー（米国IP）からは Bybit に届かないので、このスクリプトは
手元で実行し、出力の JSON をコミットする。bybit_listing.py が読み込んで DB に載せる。
上場は頻繁に増えるわけではないので、月1回程度の更新で足りる。

    python src/bybit_launch.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect import http_get_json  # noqa: E402  (Avast の HTTPS 復号に対応した GET)
from common import setup_logger  # noqa: E402
from db import connect, init_db  # noqa: E402

LOG = setup_logger("bybit_launch")
OUT = Path(__file__).resolve().parent.parent / "data" / "bybit_meta.json"
BASE = "https://api.bybit.com"
SLEEP = 0.12


def get(path: str, **q) -> dict:
    url = f"{BASE}{path}?" + "&".join(f"{k}={v}" for k, v in q.items())
    data = http_get_json(url, {"User-Agent": "AssetManagement/1.0", "Accept": "application/json"}, 30, 3)
    if data.get("retCode") != 0:
        raise RuntimeError(f"{path}: {data.get('retMsg')}")
    return data["result"]


def spot_listed_on(symbol: str) -> str | None:
    """最初の日足の日付 = 上場日（の近似）。月足で月を絞ってから日足で確定する。"""
    months = get("/v5/market/kline", category="spot", symbol=symbol, interval="M", limit=200).get("list") or []
    if not months:
        return None
    first_month_ms = int(months[-1][0])
    time.sleep(SLEEP)
    days = get("/v5/market/kline", category="spot", symbol=symbol, interval="D",
               start=first_month_ms, end=first_month_ms + 35 * 86400000, limit=40).get("list") or []
    ms = int(days[-1][0]) if days else first_month_ms
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def perp_index() -> dict[str, dict]:
    """baseCoin -> {symbol, launched}。USDT 建て無期限のみ。"""
    out: dict[str, dict] = {}
    cursor = ""
    while True:
        q = dict(category="linear", limit=1000)
        if cursor:
            q["cursor"] = cursor
        res = get("/v5/market/instruments-info", **q)
        for it in res.get("list") or []:
            if it.get("quoteCoin") != "USDT" or it.get("contractType") != "LinearPerpetual":
                continue
            if it.get("status") != "Trading":
                continue
            base = (it.get("baseCoin") or "").upper()
            ms = int(it.get("launchTime") or 0)
            launched = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d") if ms else None
            # 同じ baseCoin に 1000MOG のような倍率つきが並ぶことがある。倍率なしを優先
            cur = out.get(base)
            if cur is None or (not it["symbol"][0].isdigit() and cur["symbol"][0].isdigit()):
                out[base] = {"symbol": it["symbol"], "launched": launched}
        cursor = res.get("nextPageCursor") or ""
        if not cursor:
            break
        time.sleep(SLEEP)
    return out


def main() -> int:
    init_db()
    with connect() as conn:
        rows = conn.execute("SELECT coin_id, base FROM bybit_listing").fetchall()
    if not rows:
        LOG.error("bybit_listing が空です。先に python src/bybit_listing.py --force を実行してください")
        return 1

    existing = {}
    if OUT.exists():
        existing = json.loads(OUT.read_text(encoding="utf-8")).get("coins", {})

    LOG.info("無期限先物の一覧を取得中...")
    perps = perp_index()
    LOG.info("先物 %d 銘柄", len(perps))

    coins: dict[str, dict] = {}
    for i, (coin_id, base) in enumerate(rows, 1):
        symbol = f"{base}USDT"
        listed = existing.get(coin_id, {}).get("spotListedOn")
        if not listed:
            try:
                listed = spot_listed_on(symbol)
            except Exception as e:  # noqa: BLE001
                LOG.warning("%s: 上場日を取得できず (%s)", symbol, e)
                listed = None
            time.sleep(SLEEP)
        perp = perps.get(base)
        coins[coin_id] = {
            "base": base,
            "spotListedOn": listed,
            "perpSymbol": perp["symbol"] if perp else None,
            "perpLaunchedOn": perp["launched"] if perp else None,
        }
        if i % 50 == 0:
            LOG.info("%d / %d", i, len(rows))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"updatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "coins": coins},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    n_perp = sum(1 for c in coins.values() if c["perpSymbol"])
    LOG.info("保存: %s（%d 銘柄 / 先物あり %d）", OUT, len(coins), n_perp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
