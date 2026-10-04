"""日次パイプラインの入口。ローカル（collect.bat）でもクラウド（GitHub Actions）でも
これ 1 本を呼べば同じ処理が走るようにしてある。

各ステップは独立して失敗しうる（外部APIが落ちる、レート制限に当たる等）。
1 つ転んでも後続を止めず、最後にまとめて結果を報告する。
ただし価格収集だけは全ての土台なので、失敗したら異常終了する。

使い方:
    python src/daily.py
    python src/daily.py --dashboard-out site/index.html
    python src/daily.py --skip predict            # 特定ステップを飛ばす
"""
from __future__ import annotations

import argparse
import contextlib
import signal
from datetime import datetime, timezone
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import setup_logger  # noqa: E402

LOG = setup_logger("daily")


def step_collect():
    import collect
    if collect.run_collect() != 0:
        raise RuntimeError("価格収集が失敗しました")


def step_news():
    import news
    news.collect_news()


def step_translate():
    import translate
    translate.run(limit=30, sleep_sec=1.2)


def step_icons():
    import icons
    icons.download(size="small", refresh=False, limit=None, sleep_sec=0.3)
    icons.fetch_source_icons(refresh=False)


def step_ai_enrich():
    """Workers AI での補強（銘柄マッチ検証・センチメント・市場サマリー）。
    CF_ACCOUNT_ID / CF_AI_TOKEN が未設定なら何もせず抜ける。"""
    import ai_enrich
    ai_enrich.run()


def step_bybit_listing():
    """Bybit 現物の上場一覧（3日キャッシュ）。ランキングを売買できる銘柄に絞るのに使う。"""
    import bybit_listing
    bybit_listing.run()


def step_predict():
    import predict
    import cycle
    # 2026-09-21 から 5日サイクル。理由と検証結果は cycle.py を参照
    predict.run_predict(horizon=cycle.HORIZON, top_n=20, model_tag=None,
                        skip_importance=True)


def step_track_performance():
    """週1回（月曜）だけ検証を回し、成績の推移を記録する。
    重いので他の曜日は即スキップされる。"""
    import track_performance
    track_performance.run()


def step_coin_meta():
    """ランキング上位のチェーン/取扱取引所を取得（予測の後に実行する）。"""
    import coin_meta
    coin_meta.run(top_n=30)


def make_export_json_step(dashboard_out: str | None):
    """資産管理アプリが取得するための予測 JSON を、ダッシュボードと同じ場所へ出す。"""
    def step_export_json():
        import export_json
        target = (
            Path(dashboard_out).parent / "prediction.json"
            if dashboard_out
            else None
        )
        export_json.run_export(str(target) if target else None)
    return step_export_json


def make_dashboard_step(out_path: str | None):
    def step_dashboard():
        import dashboard
        from db import connect, init_db
        init_db()
        with connect() as conn:
            data = dashboard.gather(conn)
        target = Path(out_path) if out_path else (dashboard.REPORT_DIR / "dashboard.html")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(dashboard.build_html(data), encoding="utf-8")
        LOG.info("ダッシュボードを生成: %s (%.0f KB)", target, target.stat().st_size / 1024)
    return step_dashboard


# (名前, 関数, 失敗したら中断するか)
def build_steps(dashboard_out: str | None):
    return [
        ("collect", step_collect, True),
        ("news", step_news, False),
        ("translate", step_translate, False),
        ("icons", step_icons, False),
        ("ai_enrich", step_ai_enrich, False),
        ("bybit_listing", step_bybit_listing, False),
        ("predict", step_predict, False),
        ("coin_meta", step_coin_meta, False),
        ("track_performance", step_track_performance, False),
        ("dashboard", make_dashboard_step(dashboard_out), False),
        ("export_json", make_export_json_step(dashboard_out), False),
    ]


# 1 つのステップが通信の混雑などで止まっても、後の予測・公開・保存まで進めるための上限（秒）。
# ジョブ全体の制限時間（ワークフローの timeout-minutes）を超えて全部が取り消されるのを防ぐ
STEP_LIMITS = {
    "collect": 15 * 60,
    "predict": 15 * 60,
    "track_performance": 20 * 60,
}
DEFAULT_STEP_LIMIT = 8 * 60


class StepTimeout(BaseException):
    """各ステップ内の「except Exception」で握りつぶされないよう BaseException にする。"""
    pass


@contextlib.contextmanager
def _time_limit(seconds: int):
    """POSIX（GitHub Actions の Linux）ではステップごとに時間切れにする。Windows では何もしない。"""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise(signum, frame):  # noqa: ARG001
        raise StepTimeout("%d 秒を超えたので打ち切りました" % seconds)

    old = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def main() -> int:
    p = argparse.ArgumentParser(description="日次パイプライン")
    p.add_argument("--dashboard-out", default=None,
                   help="ダッシュボードの出力先（既定 reports/dashboard.html）")
    p.add_argument("--skip", action="append", default=[],
                   help="飛ばすステップ名（複数指定可）")
    p.add_argument("--force", action="store_true",
                   help="本日分の収集・予測が済んでいても、やり直す")
    args = p.parse_args()

    import cycle
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    skip = set(args.skip)

    # 1 日に 2 回走ることがある（Worker からの即時起動＋GitHub の定時実行）。
    # 2 回目が入れ替え日の収集・予測をやり直すと、自動売買が買った後でランキングが
    # 入れ替わってしまうので、本日分が済んでいれば収集と予測は飛ばす（公開は行う）
    if not args.force and _already_predicted(today):
        LOG.info("本日 %s の収集と予測は済んでいるので、上書きしないよう飛ばします（--force でやり直し）", today)
        skip |= {"collect", "predict", "coin_meta"}

    steps = build_steps(args.dashboard_out)
    results: list[tuple[str, str, float, str]] = []
    fatal = False

    for name, fn, critical in steps:
        if name in skip:
            results.append((name, "skip", 0.0, ""))
            continue
        LOG.info("--- %s 開始 ---", name)
        t0 = time.monotonic()
        try:
            with _time_limit(STEP_LIMITS.get(name, DEFAULT_STEP_LIMIT)):
                fn()
            results.append((name, "ok", time.monotonic() - t0, ""))
        except (Exception, StepTimeout) as e:
            msg = "%s: %s" % (type(e).__name__, e)
            results.append((name, "NG", time.monotonic() - t0, msg[:200]))
            LOG.error("--- %s 失敗 --- %s", name, msg[:300])
            LOG.debug("%s", traceback.format_exc())
            if critical:
                fatal = True
                LOG.error("%s は他の全処理の前提なので中断します。", name)
                break

    LOG.info("=" * 56)
    LOG.info(" 日次パイプライン結果")
    LOG.info("=" * 56)
    for name, status, secs, msg in results:
        LOG.info(" %-10s %-4s %6.1f秒 %s", name, status, secs, msg)
    ng = [r for r in results if r[1] == "NG"]
    LOG.info(" 失敗 %d / %d ステップ", len(ng), len(results))

    # 入れ替え日に予測・公開が失敗したら、ワークフローの最後で「失敗」にして知らせる。
    # ここで止めると収集したデータの保存や前回分の公開まで止まるので、印を残すだけにする
    ng_names = {r[0] for r in ng}
    alerts = []
    if cycle.is_cycle_day(today) and ng_names & {"predict", "export_json"}:
        alerts.append("入れ替え日 %s の予測または公開が失敗しました: %s" % (today, ", ".join(sorted(ng_names))))
    if "export_json" in ng_names and not alerts:
        alerts.append("予測 JSON の書き出しが失敗しました（前回の公開分が残ります）")
    if alerts:
        alert_path = Path("logs") / "pipeline_alert.txt"
        alert_path.parent.mkdir(parents=True, exist_ok=True)
        alert_path.write_text("\n".join(alerts) + "\n", encoding="utf-8")
        for a in alerts:
            LOG.error(a)

    return 1 if fatal else 0


def _already_predicted(today: str) -> bool:
    """本日の価格と予測が DB に既にあるか。"""
    try:
        import cycle
        from db import connect
        with connect() as conn:
            snap = conn.execute(
                "SELECT 1 FROM snapshots WHERE snapshot_date = ? LIMIT 1", (today,)
            ).fetchone()
            pred = conn.execute(
                "SELECT 1 FROM predictions WHERE predicted_on = ? AND horizon_days = ? LIMIT 1",
                (today, cycle.HORIZON),
            ).fetchone()
        return bool(snap and pred)
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    raise SystemExit(main())
