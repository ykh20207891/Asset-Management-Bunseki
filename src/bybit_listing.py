"""Bybit 現物(USDT建て)で実際に売買できる銘柄の一覧を取得して保存する。

資産管理アプリの自動売買は Bybit でしか売買しない。ところがランキングは
全取引所の銘柄が対象なので、上位5銘柄のうち Bybit で買えるのが2つしか無い、
ということが起きていた（残りは6位以下からの繰り上げ＝検証した戦略と別物になる）。
そこで Bybit に上場している銘柄だけをランキング対象にできるよう、一覧を持つ。

Bybit の API を直接叩かない理由:
  - GitHub Actions のランナーは米国にあり、Bybit は米国IPを弾く
  - シンボル名だけでは別銘柄と衝突する（BABY / BR / MX など）
CoinGecko の /exchanges/bybit_spot/tickers は CoinGecko の coin_id 付きで返るので、
銘柄の取り違えが起きない。

    python src/bybit_listing.py            # 期限切れなら更新
    python src/bybit_listing.py --force    # 強制更新
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect import _base_and_headers, http_get_json  # noqa: E402
from common import load_config, setup_logger  # noqa: E402
from db import connect, init_db  # noqa: E402

LOG = setup_logger("bybit_listing")

EXCHANGE_ID = "bybit_spot"
QUOTE = "USDT"
# 上場/廃止はそう頻繁に起きないので、毎日は取りに行かない
CACHE_DAYS = 3
SLEEP_SEC = 2.5
MAX_PAGES = 30
# 取得件数がこれ未満なら「取り損ね」とみなして一覧を更新しない（既存を守る）
MIN_EXPECTED = 100


def _ensure_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bybit_listing (
            coin_id    TEXT PRIMARY KEY,   -- CoinGecko の id
            base       TEXT NOT NULL,      -- Bybit 上のシンボル (BTC)
            last_price REAL,
            fetched_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _is_fresh(conn) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM bybit_listing WHERE fetched_at > datetime('now', ?)",
        (f"-{CACHE_DAYS} day",),
    ).fetchone()
    return bool(row and row[0] >= MIN_EXPECTED)


def fetch_listing(cfg: dict) -> dict[str, dict]:
    base_url, headers = _base_and_headers(cfg)
    timeout = int(cfg.get("request_timeout", 45))
    max_retry = int(cfg.get("max_retry", 6))

    found: dict[str, dict] = {}
    for page in range(1, MAX_PAGES + 1):
        url = f"{base_url}/exchanges/{EXCHANGE_ID}/tickers?page={page}&order=volume_desc"
        data = http_get_json(url, headers, timeout, max_retry)
        tickers = (data or {}).get("tickers") or []
        if not tickers:
            break

        for t in tickers:
            if (t.get("target") or "").upper() != QUOTE:
                continue
            coin_id = t.get("coin_id")
            base = (t.get("base") or "").upper()
            if not coin_id or not base:
                continue
            # 同じ coin_id が複数ペアで出ることがある。出来高順なので最初を採る
            found.setdefault(coin_id, {"base": base, "last": t.get("last")})

        LOG.info("page %d: %d 件 (累計 %d 銘柄)", page, len(tickers), len(found))
        if len(tickers) < 100:
            break
        time.sleep(SLEEP_SEC)

    return found


def run(force: bool = False) -> int:
    init_db()
    with connect() as conn:
        _ensure_table(conn)
        if not force and _is_fresh(conn):
            LOG.info("Bybit 上場一覧は新しいので取得を省略します")
            return 0

    listing = fetch_listing(load_config())
    if len(listing) < MIN_EXPECTED:
        LOG.warning("取得できたのが %d 銘柄だけなので一覧は更新しません", len(listing))
        return 0

    with connect() as conn:
        _ensure_table(conn)
        # 廃止された銘柄が残らないよう総入れ替え
        conn.execute("DELETE FROM bybit_listing")
        conn.executemany(
            "INSERT INTO bybit_listing (coin_id, base, last_price, fetched_at) "
            "VALUES (?, ?, ?, datetime('now'))",
            [(cid, v["base"], v.get("last")) for cid, v in listing.items()],
        )
        conn.commit()

    LOG.info("Bybit 現物(USDT) の上場一覧を更新: %d 銘柄", len(listing))
    return len(listing)


def tradable_map(conn) -> dict[str, str]:
    """coin_id -> Bybit 上のシンボル。未取得・取得失敗時は空（＝絞り込まない）。"""
    try:
        rows = conn.execute("SELECT coin_id, base FROM bybit_listing").fetchall()
    except Exception:  # noqa: BLE001
        return {}
    listing = {r[0]: r[1] for r in rows}
    return listing if len(listing) >= MIN_EXPECTED else {}


def main() -> int:
    p = argparse.ArgumentParser(description="Bybit 現物の上場一覧を取得する")
    p.add_argument("--force", action="store_true", help="キャッシュを無視して取得")
    args = p.parse_args()
    run(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
