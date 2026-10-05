"""盘点 hm0~hm92 的 okay 文件，生成改写清单 program_state.json。

对每个系列记下：编号、ver、来源目录、旧框架形态、数据来源分类、
需要的 base 因子名（names_save）、入选名单（names_gap5 / names_gap1 / names_in_db）、
以及四个步骤的进度字段（改写 / 提交 / 入库 / 清 colblk）。

数据来源分类规则：
- 旧类继承 dwo.FactorDesign 且有 new_get_field_min_data → 分钟数据（per-date 全市场）
- 旧类继承 dwo.TickDataBase → Level2 逐笔/盘口（per-stock）
- go(date, code) 形态 → 看函数体里读的是逐笔还是盘口
"""
from __future__ import annotations

import ast
import json
import re
from datetime import datetime
from pathlib import Path

PYCODE = Path("/home/chenzongwei/pythoncode")
OUT = Path("/home/chenzongwei/rust_pyfunc/reports/okay_database/program_state.json")

LIST_KEYS = (
    "names", "names_save", "names_okay", "names_in_db", "names_tail_two",
    "names_gap5", "names_gap1",
)


def string_list(node: ast.AST) -> list[str] | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    out = []
    for e in node.elts:
        if isinstance(e, ast.Constant) and isinstance(e.value, str):
            out.append(e.value)
        else:
            return None
    return out


def parse(path: Path) -> dict:
    src = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    lists: dict[str, list[str]] = {}
    ver = None
    bases: list[str] = []
    has_go = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    if t.id == "ver" and isinstance(node.value, ast.Constant):
                        ver = node.value.value
                    vals = string_list(node.value)
                    if vals is not None and t.id in LIST_KEYS:
                        lists.setdefault(t.id, []).extend(vals)
        elif isinstance(node, ast.ClassDef):
            bases += [ast.unparse(b) for b in node.bases]
        elif isinstance(node, ast.FunctionDef) and node.name == "go":
            has_go = True

    body = src
    uses_min = "new_get_field_min_data" in body
    uses_trade = "read_trade_data_single" in body or "read_trade(" in body or "read_trade_data(" in body
    uses_market = "read_market_data_single" in body or "read_market(" in body or "read_market_data(" in body
    uses_cross = bool(re.search(r"\bde_cross\b", body))

    if "FactorDesign" in " ".join(bases) or uses_min:
        source = "分钟数据"
        kind = "cross_section"
    elif uses_cross:
        source = "Level2（用到跨股票截面）"
        kind = "cross_section"
    else:
        parts = []
        if uses_trade:
            parts.append("逐笔")
        if uses_market:
            parts.append("盘口")
        source = "Level2 " + "+".join(parts) if parts else "Level2（待确认）"
        kind = "per_stock"

    style = "go 函数" if has_go else ("FactorDesign" if "FactorDesign" in " ".join(bases) else "TickDataBase")

    return {
        "ver": ver,
        "style": style,
        "source": source,
        "kind": kind,
        "uses_cross_section": uses_cross,
        "names_save": list(dict.fromkeys(lists.get("names_save", []))),
        "names_okay": list(dict.fromkeys(lists.get("names_okay", []))),
        "names_gap5": list(dict.fromkeys(lists.get("names_gap5", []))),
        "names_gap1": list(dict.fromkeys(lists.get("names_gap1", []))),
        "names_in_db": list(dict.fromkeys(lists.get("names_in_db", []))),
    }


def main() -> int:
    series = []
    for path in sorted(PYCODE.glob("成果*/hm*_okay.py")):
        m = re.match(r"hm(\d+)_okay\.py$", path.name)
        if not m:
            continue
        num = int(m.group(1))
        if num >= 94:
            continue
        info = parse(path)
        # 入选名单的 base 名字：去掉收尾算子后缀
        bases = set(info["names_save"])
        for name in info["names_gap5"] + info["names_gap1"]:
            n = name
            if n.endswith("_smooth_1"):
                n = n[: -len("_smooth_1")]
            else:
                n = re.sub(r"_(mean|max|min|std)_smooth_\d+$", "", n)
            bases.add(n)
        series.append(
            {
                "num": num,
                "series": f"hm{num:03d}",
                "ver": info["ver"],
                "dir": str(path.parent),
                "okay_py": str(path),
                "style": info["style"],
                "source": info["source"],
                "kind": info["kind"],
                "uses_cross_section": info["uses_cross_section"],
                "base_names": sorted(bases),
                "n_base": len(bases),
                "n_gap5": len(info["names_gap5"]),
                "n_gap1": len(info["names_gap1"]),
                "names_in_db": info["names_in_db"],
                "rust_file": f"src/okay_database/hm{num:03d}.rs",
                "database_py": str(path.parent / f"hm{num}_okay_database.py"),
                # 四个步骤的进度
                "step_rewrite": "pending",
                "step_submit": "pending",
                "step_library": "pending",
                "step_cleanup": "pending",
                "task_id": None,
                "note": "",
            }
        )

    # 保留已有进度：重扫只更新系列元信息，不重置四个步骤的状态
    previous = {}
    if OUT.is_file():
        try:
            for old in json.loads(OUT.read_text())["series"]:
                previous[old["series"]] = old
        except Exception:  # noqa: BLE001
            previous = {}
    for item in series:
        old = previous.get(item["series"])
        if not old:
            continue
        for key in ("step_rewrite", "step_submit", "step_library", "step_cleanup",
                    "task_id", "note", "library_files", "colblk_left"):
            if key in old:
                item[key] = old[key]

    state = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(series),
        "series": series,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    kinds: dict[str, int] = {}
    for s in series:
        kinds[s["source"]] = kinds.get(s["source"], 0) + 1
    print(f"共 {len(series)} 个系列，清单写入 {OUT}")
    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
