"""系列认领器：多个 agent 并行改写时，谁也别抢同一个系列。

用 O_EXCL 建认领文件，天然互斥，不需要额外加锁。

用法：
    python lib/okay_database/claim_series.py next --worker hm07-worker
        认领下一个还没人做、也还没做完的系列，打印它的编号；
        全部认领完打印 "NONE"。
    python lib/okay_database/claim_series.py release hm013
        放掉认领（做不完时用）。
    python lib/okay_database/claim_series.py status
        看认领情况。
    python lib/okay_database/claim_series.py work --series hm013
        打印这个系列的作业要点（旧文件路径、数据来源、要产出的两个文件），
        给 agent 省一次读盘的功夫。
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from datetime import datetime
from pathlib import Path

REPO = Path("/home/chenzongwei/rust_pyfunc")
REPORT = REPO / "reports/okay_database"
STATE = REPORT / "program_state.json"
CLAIMS = REPORT / "claims"
TRIES = CLAIMS / ".tries"   # 被退回的次数：退得越多排得越后，避免反复派给做不动的人


def load() -> dict:
    return json.loads(STATE.read_text())


def candidates(state: dict) -> list[dict]:
    """还没做完、且没人认领的系列。

    排序：能自动解析出入选名单的排前面（按名单长度从少到多），
    解析不出名单的（n_base=0，需要人先读懂旧代码）排后面，同级按编号。
    """
    out = []
    for s in state["series"]:
        if s.get("step_rewrite") == "done":
            continue
        if (CLAIMS / f"{s['series']}.json").exists():
            continue
        out.append(s)
    def tries(s):
        path = TRIES / s["series"]
        return int(path.read_text()) if path.exists() else 0

    out.sort(key=lambda s: (tries(s), 0 if s["n_base"] > 0 else 1, s["n_base"], s["num"]))
    return out


def claim(series: str, worker: str) -> bool:
    CLAIMS.mkdir(parents=True, exist_ok=True)
    path = CLAIMS / f"{series}.json"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as fh:
        json.dump({"worker": worker, "time": datetime.now().isoformat()}, fh, ensure_ascii=False)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["next", "release", "status", "work", "complete", "fail"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--worker", default=os.environ.get("USER", "agent"))
    args = ap.parse_args()

    state = load()
    if args.cmd == "status":
        done = [s["series"] for s in state["series"] if s.get("step_rewrite") == "done"]
        held = sorted(p.stem for p in CLAIMS.glob("*.json")) if CLAIMS.is_dir() else []
        left = [s["series"] for s in candidates(state)]
        print(f"总数 {state['count']}｜已完成 {len(done)}｜进行中 {len(held)}｜待认领 {len(left)}")
        print("进行中：", " ".join(held) or "无")
        print("待认领：", " ".join(left[:40]) + (" ..." if len(left) > 40 else ""))
        return 0

    if args.cmd in ("complete", "fail"):
        # program_state.json 是共享文件，多个 agent 同时写会互相覆盖，这里用文件锁串行化
        note_path = CLAIMS / f"{args.arg}.note"
        note = note_path.read_text().strip() if note_path.exists() else (
            "Rust 已复现并通过新旧对账" if args.cmd == "complete" else "未完成"
        )
        lock = CLAIMS / ".state.lock"
        with lock.open("a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            st = json.loads(STATE.read_text())
            for item in st["series"]:
                if item["series"] == args.arg:
                    if args.cmd == "complete":
                        item["step_rewrite"] = "done"
                    item["note"] = note
            tmp = STATE.with_suffix(".tmp")
            tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1))
            tmp.replace(STATE)
        if note_path.exists():
            note_path.unlink()
        if args.cmd in ("release", "complete", "fail"):
            path = CLAIMS / f"{args.arg}.json"
            if path.exists():
                path.unlink()
        print(f"{args.arg} 已记 {'完成' if args.cmd == 'complete' else '未完成'}")
        return 0

    if args.cmd == "release":
        path = CLAIMS / f"{args.arg}.json"
        if path.exists():
            path.unlink()
        TRIES.mkdir(parents=True, exist_ok=True)
        tpath = TRIES / args.arg
        n = int(tpath.read_text()) + 1 if tpath.exists() else 1
        tpath.write_text(str(n))
        print(f"已放掉 {args.arg}（累计退回 {n} 次，队列里会往后排）")
        return 0

    if args.cmd == "work":
        s = next(x for x in state["series"] if x["series"] == args.arg)
        print(json.dumps(
            {
                "系列": s["series"],
                "数据来源": s["source"],
                "建议形态": s["kind"],
                "用到跨股票截面": s["uses_cross_section"],
                "旧文件": s["okay_py"],
                "要新建_Rust": str(REPO / s["rust_file"]),
                "要新建_因子库文件": s["database_py"],
                "Rust_系列名": s["series"].replace("hm0", "hm").replace("hm", "hm") if False else f"hm{int(s['series'][2:]):03d}",
                "入选名单里的 base 名": s["base_names"],
                "gap5 个数": s["n_gap5"],
                "gap1 个数": s["n_gap1"],
            },
            ensure_ascii=False, indent=1,
        ))
        return 0

    # next：抢下一个
    for s in candidates(state):
        if claim(s["series"], args.worker):
            print(s["series"])
            return 0
    print("NONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
