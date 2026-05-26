#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py  ─  全実験 → results/ 収集 → results ブランチへ push
================================================================
使い方:
    python run_all.py

実行の流れ:
    1. 実験スクリプトを順番に実行（matplotlib は非表示モード）
    2. 生成された .png / .npz / .csv を results/ フォルダに収集
    3. results ブランチ（なければ自動作成）に results/ を push
    4. 元のブランチに自動で戻る

前提:
    - git remote "origin" が設定済みであること
    - push 権限があること
"""

import os
import sys
import shutil
import subprocess
import datetime
from pathlib import Path


# ================================================================
# ★ 設定  ─  必要に応じてここだけ変更する
# ================================================================

# リポジトリのルートディレクトリ（このファイルの場所）
REPO_ROOT = Path(__file__).parent.resolve()

# 結果ファイルをまとめるフォルダ
RESULTS_DIR = REPO_ROOT / "results"

# 結果を push するブランチ名
RESULTS_BRANCH = "results"

# 実験スクリプトが置かれているフォルダ
SCRIPTS_DIR = REPO_ROOT / "計画" / "traffic_jam_mas" / "python_files"

# 実行する実験スクリプト（このリストの順番に実行される）
EXPERIMENT_SCRIPTS = [
    "experiments.py",           # 実験2・3：N スイープ / T スイープ
    "experiment4_growth.py",    # 実験4-A/B/C：成長率 gamma のスイープ
    "experiment4_NT_map.py",    # 実験4拡張：N×T 安定性マップ
    "experiment5_mixed.py",     # 実験5：慎重派ドライバーの混在実験
    "experiment_ncars.py",      # 車両台数 vs 渋滞発生
]

# 収集する拡張子（これ以外のファイルは results/ にコピーしない）
COLLECT_EXTENSIONS = {".png", ".npz", ".csv"}


# ================================================================
# ログ出力ヘルパー
# ================================================================

def section(title: str):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def info(msg: str):
    print(f"  [INFO]  {msg}")


def ok(msg: str):
    print(f"  [ OK ]  {msg}")


def warn(msg: str):
    print(f"  [WARN]  {msg}", file=sys.stderr)


def error(msg: str):
    print(f"  [ERR ]  {msg}", file=sys.stderr)


# ================================================================
# Step 1: 実験スクリプトの実行
# ================================================================

def run_experiments() -> list[str]:
    """
    EXPERIMENT_SCRIPTS をひとつずつ実行する。
    plt.show() がブロックしないよう MPLBACKEND=Agg を強制。
    戻り値: 失敗したスクリプト名のリスト
    """
    failed = []

    for script_name in EXPERIMENT_SCRIPTS:
        script_path = SCRIPTS_DIR / script_name

        if not script_path.exists():
            warn(f"スクリプトが見つかりません: {script_path}")
            failed.append(script_name)
            continue

        section(f"Running: {script_name}")

        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"   # plt.show() を非表示にする

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_path.parent),   # スクリプトと同じディレクトリで実行
            env=env,
        )

        if result.returncode == 0:
            ok(f"{script_name} 完了")
        else:
            error(f"{script_name} が失敗しました (returncode={result.returncode})")
            failed.append(script_name)

    return failed


# ================================================================
# Step 2: 結果ファイルを results/ に収集
# ================================================================

def collect_results() -> list[Path]:
    """
    SCRIPTS_DIR 内の .png / .npz / .csv を RESULTS_DIR にコピーする。
    戻り値: コピーしたファイルのリスト
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    collected = []
    for ext in COLLECT_EXTENSIONS:
        for src in sorted(SCRIPTS_DIR.glob(f"*{ext}")):
            dst = RESULTS_DIR / src.name
            shutil.copy2(src, dst)
            info(f"  {src.name}")
            collected.append(dst)

    return collected


# ================================================================
# Step 3: results ブランチへ push
# ================================================================

def _git(args: list[str], cwd: Path = None, check: bool = True,
         capture: bool = True) -> subprocess.CompletedProcess:
    """git コマンドのラッパー。"""
    return subprocess.run(
        ["git"] + args,
        cwd=str(cwd or REPO_ROOT),
        check=check,
        capture_output=capture,
        text=True,
    )


def git_push_results():
    """
    results/ を RESULTS_BRANCH に push する。

    手順:
        1. 現在のブランチを記録
        2. 未コミット変更を git stash で退避
        3. results ブランチへ切り替え（なければ orphan として新規作成）
        4. git add -f results/ → commit → push
        5. 元のブランチへ戻る
        6. git stash pop で変更を復元
    """
    # 1. 現在のブランチ
    current_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    info(f"現在のブランチ: {current_branch}")

    # 2. 未コミット変更を退避
    dirty = bool(_git(["status", "--porcelain"]).stdout.strip())
    if dirty:
        _git(["stash", "push", "-m", "run_all: auto-stash before results push"])
        info("未コミット変更を stash に退避しました")

    try:
        # 3. results ブランチへ切り替え
        local_exists = (
            _git(["rev-parse", "--verify", RESULTS_BRANCH], check=False).returncode == 0
        )
        remote_result = _git(
            ["ls-remote", "--heads", "origin", RESULTS_BRANCH], check=False
        )
        remote_exists = RESULTS_BRANCH in remote_result.stdout

        if local_exists:
            _git(["checkout", RESULTS_BRANCH])
            info(f"既存のブランチ '{RESULTS_BRANCH}' に切り替えました")

        elif remote_exists:
            _git(["fetch", "origin", f"{RESULTS_BRANCH}:{RESULTS_BRANCH}"])
            _git(["checkout", RESULTS_BRANCH])
            info(f"リモートから '{RESULTS_BRANCH}' をフェッチして切り替えました")

        else:
            # orphan ブランチとして新規作成（既存コミットと独立した履歴）
            _git(["checkout", "--orphan", RESULTS_BRANCH])
            # 元ブランチのファイルがステージされているので全解除
            _git(["rm", "-rf", "--cached", "."])
            info(f"新しい orphan ブランチ '{RESULTS_BRANCH}' を作成しました")

        # 4. results/ を add（-f で .gitignore を無視）→ commit → push
        rel = str(RESULTS_DIR.relative_to(REPO_ROOT))
        _git(["add", "-f", rel])

        staged = _git(["status", "--porcelain"]).stdout.strip()
        if staged:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _git(["commit", "-m", f"[auto] Update results ({timestamp})"])
            _git(["push", "-u", "origin", RESULTS_BRANCH])
            ok(f"ブランチ '{RESULTS_BRANCH}' に push しました")
        else:
            info("results/ に変更がなかったため commit はスキップしました")

    finally:
        # 5. 元のブランチへ戻る
        _git(["checkout", current_branch])
        ok(f"ブランチ '{current_branch}' に戻りました")

        # 6. stash を復元
        if dirty:
            _git(["stash", "pop"])
            info("stash を復元しました")


# ================================================================
# メイン処理
# ================================================================

def main():
    section("run_all.py : 実験ランナー開始")
    info(f"リポジトリ  : {REPO_ROOT}")
    info(f"スクリプト  : {SCRIPTS_DIR}")
    info(f"results フォルダ: {RESULTS_DIR}")
    info(f"push 先ブランチ : {RESULTS_BRANCH}")

    # ── Step 1: 実験実行 ──────────────────────────────────────
    section("Step 1 / 3  ─  実験スクリプトを実行")
    failed = run_experiments()

    if failed:
        warn(f"失敗したスクリプト: {failed}")
        warn("失敗があっても結果収集・push は続行します")

    # ── Step 2: 結果収集 ──────────────────────────────────────
    section("Step 2 / 3  ─  results/ に結果ファイルを収集")
    collected = collect_results()

    if not collected:
        warn("収集できたファイルが 0 件でした。スクリプトが正常に実行されたか確認してください。")
    else:
        ok(f"{len(collected)} 件のファイルを results/ にコピーしました")

    # ── Step 3: Git push ──────────────────────────────────────
    section(f"Step 3 / 3  ─  '{RESULTS_BRANCH}' ブランチへ push")
    git_push_results()

    # ── 完了サマリー ─────────────────────────────────────────
    section("完了")
    if failed:
        error(f"失敗したスクリプト ({len(failed)} 件): {', '.join(failed)}")
        sys.exit(1)
    else:
        ok("全スクリプトが正常に終了し、結果を push しました 🎉")


if __name__ == "__main__":
    main()