"""Rebuild the yupei speed evidence workbook without rerunning any calculation.

Use the project's Python environment. The default output is the requested XLSX;
pass --output PATH for a preview. Only recorded measurements enter the report.
An optional results/speed/report_decisions.json may contain:
    {"local": {"gpt_scan": {"status": "retained", "reason": "..."}},
     "variants": {"candidate1": {"status": "discarded", "reason": "..."}},
     "notes": ["..."]}
Statuses: retained, discarded, pending. Missing decisions remain pending. Failed,
incomplete, overloaded, or differently threaded runs stay visible in raw evidence.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from zoneinfo import ZoneInfo

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "sandbox/yupei_union_trim_20261107"
UNKNOWN = "未测"
RUN_NAME = re.compile(r"^(?!compare_)(.+)_(\d{8})_r(\d+)$")
OPERATIONAL_SETTINGS = {"YUPEI_STAGE", "YUPEI_VALIDATE_ONLY", "YUPEI_VERBOSE", "YUPEI_GPT_VERBOSE"}
LOCAL_DEFAULTS = {
    "sort_100": ("retained", "最终 sort_summary.json 明确 enabled_lengths=100；整体效果另验"),
    "sort_20": ("discarded", "32通道候选未启用；20键继续原排序"),
    "gpt_transpose": ("discarded", "已废弃的写布局探针；不计入采用收益"),
}


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def first(value):
    return value[0] if isinstance(value, (tuple, list)) and value else None


def median(values):
    values = [v for v in values if finite(v)]
    return statistics.median(values) if values else None


def ratio(left, right):
    return left / right if finite(left) and finite(right) and right > 0 else None


def run_identity(run):
    # Old reports predate the experiment field; treat them as default settings,
    # and expose that assumption separately rather than blending named settings.
    return (run["variant"], run.get("module_sha256") or UNKNOWN,
            json.dumps(run.get("experiment_settings", {}), sort_keys=True,
                       ensure_ascii=False, separators=(",", ":")))


def excel_value(value):
    if value is None:
        return UNKNOWN
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, str) and value.startswith("="):
        return "'" + value
    return value


def hms_seconds(text):
    parts = text.strip().split(":")
    try:
        return sum(float(value) * 60 ** i for i, value in enumerate(reversed(parts)))
    except ValueError:
        return None


def parse_log(text):
    result = {"events": [], "stages": [], "thread_counts": []}
    previous = 0.0
    for line in text.splitlines():
        if line.startswith("{"):
            try:
                event = json.loads(line)
                if isinstance(event, dict) and event.get("event") in ("start", "complete"):
                    result["events"].append(event)
            except json.JSONDecodeError:
                pass
        m = re.search(r"Exit status:\s*(\d+)", line)
        if m:
            result["exit_code"] = int(m[1])
        for label, key in [("Maximum resident set size (kbytes)", "max_rss_kib"),
                           ("User time (seconds)", "process_user_seconds"),
                           ("System time (seconds)", "process_system_seconds")]:
            if label + ":" in line:
                result[key] = float(line.rsplit(":", 1)[1].strip())
        m = re.search(r"Elapsed \(wall clock\) time .*?:\s*([0-9:.]+)$", line)
        if m:
            result["process_wall_seconds"] = hms_seconds(m[1])
        m = re.search(r"(?:thread_count|rayon_threads|threads)=(\d+)", line)
        if m:
            result["thread_counts"].append(int(m[1]))
        m = re.search(r"\[yupei_union_all\] (.+) 完成，累计 ([0-9.]+)s", line)
        if m:
            cumulative = float(m[2])
            result["stages"].append(("顺序总阶段", m[1], cumulative - previous,
                                      cumulative, "由日志累计墙钟相减；日志原值仅保留0.1秒"))
            previous = cumulative
            continue
        m = re.search(r"\[yupei_union_all\].*，总 ([0-9.]+)s", line)
        if m:
            cumulative = float(m[1])
            result["stages"].append(("顺序总阶段", "最终组装", cumulative - previous,
                                      cumulative, "只减最近一个顺序阶段；不累加并行子段"))
            continue
        m = re.search(r"\[gpt-stage\]\s*(\{.*\})", line)
        if m:
            try:
                for name, seconds in json.loads(m[1]).items():
                    result["stages"].append(("GPT内部记录", name, seconds, None,
                                              "原始阶段墙钟；total_compute包含子项，不重复求和"))
            except json.JSONDecodeError:
                pass
            continue
        m = re.search(r"\[(multi-stage|lite-stage|split-stage)\]\s*(?:name=)?([^=]+?)\s+(?:seconds=)?([0-9.eE+-]+)s?$", line)
        if m:
            result["stages"].append((m[1], m[2], float(m[3]), None,
                                      "内部记录，可能嵌套；不与总阶段求和"))
        elif line.startswith(("[lite-mod]", "[mirror-stage]", "[stage]", "[split-chain]",
                              "[gpt-family]", "[gpt-schedule]", "[net-stage]", "[net-exz-stage]",
                              "[topk-diag]", "[gpt-shared-rho]",
                              "[lite-stage]", "[multi-stage]", "[split-stage]")):
            # Keep parallel work as raw log evidence, never manufacture an elapsed total.
            result["stages"].append(("并行/嵌套诊断", line, None, None,
                                      "只保留原文，禁止对子任务耗时求和"))
    return result


def decision(config, category, name):
    default = LOCAL_DEFAULTS.get(name, ("pending", "尚无明确采用/废弃记录"))
    value = config.get(category, {}).get(name, {})
    if isinstance(value, str):
        return value, "report_decisions.json"
    return value.get("status", default[0]), value.get("reason", default[1])


def status_label(status):
    return {"retained": "采用候选（整体另验）", "discarded": "已排除", "pending": "未决，不计采用收益"}.get(status, status)


def sheet(wb, title, headers, rows, note):
    ws = wb.create_sheet(title)
    ws.append([title])
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
    ws.cell(1, 1).font = Font(name="Microsoft YaHei", size=16, bold=True, color="17365D")
    ws.append([note])
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(headers))
    ws.cell(2, 1).alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[2].height = 34
    ws.append(headers)
    for row in rows:
        ws.append([excel_value(value) for value in row])
    for cell in ws[3]:
        cell.fill = PatternFill("solid", fgColor="24486B")
        cell.font = Font(name="Microsoft YaHei", bold=True, color="FFFFFF")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[3].height = 32
    for row in ws.iter_rows(min_row=4):
        for cell in row:
            cell.font = Font(name="Microsoft YaHei", size=10)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            if isinstance(cell.value, float):
                cell.number_format = "0.000000"
            if cell.value is False or cell.value in ("不通过", "未达标", "退出失败"):
                cell.fill = PatternFill("solid", fgColor="FCE4D6")
            elif cell.value is True or cell.value == "通过":
                cell.fill = PatternFill("solid", fgColor="E2F0D9")
            elif cell.value == UNKNOWN:
                cell.fill = PatternFill("solid", fgColor="FFF2CC")
    for i, header in enumerate(headers, 1):
        width = 19
        if any(word in header for word in ("来源", "路径", "原因", "说明", "原文", "SHA", "证据", "指标")):
            width = 48
        elif any(word in header for word in ("变体", "阶段", "项目", "输入")):
            width = 29
        ws.column_dimensions[get_column_letter(i)].width = width
    ws.freeze_panes = "C4"
    ws.auto_filter.ref = f"A3:{get_column_letter(len(headers))}{max(3, ws.max_row)}"
    ws.sheet_view.showGridLines = False
    ws.print_options.horizontalCentered = True
    ws.print_title_rows = "1:3"
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.orientation = "landscape"
    ws.page_setup.paperSize = ws.PAPERSIZE_A3
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    return ws


def local_tables(documents, decisions):
    summaries, raw, tests, excluded = [], [], [], []

    def add(project, scenario, before, after, unit, source, level, note, n_before=None, n_after=None):
        state, reason = decision(decisions, "local", project)
        row = ["实测/由实测计算", level, project, scenario, status_label(state),
               median(before), median(after), ratio(median(before), median(after)), unit,
               n_before if n_before is not None else len(before), n_after if n_after is not None else len(after),
               "仅局部；不能换算全市场节时", note, source, reason]
        (excluded if state != "retained" else summaries).append(row)

    for prefix, flag, baseline_value, level in [("gpt_scan", "baseline", True, "重复计算"),
                                               ("gpt_topk", "sorted", True, "重复计算/排序")]:
        filename = prefix + "_bench.json"
        doc = documents.get(filename, {})
        if not isinstance(doc, dict):
            continue
        before, after = [], []
        for run in doc.get("runs", []):
            value = run.get("elapsed_s")
            calls = run.get("calls")
            per_call = value * 1e9 / calls if finite(value) and finite(calls) and calls else None
            target = before if run.get(flag) == baseline_value else after
            if per_call is not None:
                target.append(per_call)
            raw.append(["实测", prefix, filename, run.get("pass"), "原版" if run.get(flag) == baseline_value else "候选",
                        run.get("n"), calls, value, "秒/整组调用", per_call, "ns/调用", doc.get("scope"), None])
        if before or after:
            add(prefix, f"n={first(doc.get('runs', [])) and doc['runs'][0].get('n')}", before, after,
                "ns/调用", filename, level, doc.get("scope"))

    grouped_sort = defaultdict(lambda: {"baseline": [], "after": [], "sources": [], "samples": 0})
    for filename, doc in documents.items():
        if not re.fullmatch(r"sort_final_r\d+\.json", filename) or not isinstance(doc, dict):
            continue
        for case in doc.get("cases", []):
            key = (case.get("length"), case.get("pattern"))
            group = grouped_sort[key]
            # Final public dispatch, including its guard and cached environment check.
            group["baseline"].append(median(case.get("baseline_ns", [])))
            group["after"].append(median(case.get("dispatch_ns", [])))
            group["sources"].append(filename)
            group["samples"] += len(case.get("dispatch_ns", []))
            for method, field in [("标准排序", "baseline_ns"), ("候选网络", "bitonic_ns"), ("最终入口", "dispatch_ns")]:
                for sample, value in enumerate(case.get(field, []), 1):
                    raw.append(["实测", f"sort_{key[0]}", filename, sample, method, key[0],
                                doc.get("sorts_per_sample"), None, None, value, "ns/排序",
                                key[1], f"包含输入复制；baseline_switch={doc.get('baseline_switch')}"])
    for (length, pattern), group in grouped_sort.items():
        add(f"sort_{length}", pattern, group["baseline"], group["after"], "ns/排序",
            ", ".join(group["sources"]), "向量化/完整整数键排序",
            "每进程先取7次中位数，再跨独立进程取中位数；20键网络未启用",
            len(group["baseline"]), len(group["after"]))

    quantile = documents.get("pair_quantile_microbench.json", {})
    quantile_groups = defaultdict(lambda: ([], []))
    if isinstance(quantile, dict):
        for item in quantile.get("records", []):
            before, after = quantile_groups[item.get("kind")]
            before.append(item.get("sort_seconds"))
            after.append(item.get("select_seconds"))
            raw.append(["实测", "pair_quantile", "pair_quantile_microbench.json", item.get("repeat"),
                        "原排序/精确选择", quantile.get("elements"), None,
                        item.get("sort_seconds"), "原排序秒", item.get("select_seconds"), "精确选择秒",
                        item.get("kind"), f"cuts_bitwise_equal={item.get('cuts_bitwise_equal')}"])
        for kind, (before, after) in quantile_groups.items():
            add("pair_quantile", kind, before, after, "秒/整组", "pair_quantile_microbench.json",
                "算法：精确分位选择", "合成输入；不含输入生成和复制；只用于此局部核")

    bitset = documents.get("pair_bitset_microbench.json", {})
    if isinstance(bitset, dict) and bitset:
        before, after = [], []
        for item in bitset.get("records", []):
            before.append(item.get("old_seconds"))
            after.append(item.get("new_seconds"))
            raw.append(["实测", "pair_bitset", "pair_bitset_microbench.json", item.get("repeat"),
                        "逐位/整字计数", bitset.get("n"), None, item.get("old_seconds"), "原计数秒",
                        item.get("new_seconds"), "位图计数秒", "合成单图；不含全因子流程",
                        f"counts_equal={item.get('counts_equal')}"])
        if before or after:
            add("pair_bitset", bitset.get("measurement"), before, after, "秒/单图",
                "pair_bitset_microbench.json", "整数位图统计", "局部测量，不能外推全市场收益")
        else:
            excluded.append(["未知/未测", "整数位图统计", "pair_bitset", bitset.get("measurement"),
                             "未测，不计收益", None, None, None, "秒/单图", 0, 0,
                             "尚未执行性能测量", f"status={bitset.get('status')}；benchmark_executed={bitset.get('benchmark_executed')}；启动load={bitset.get('load_before')}；max_load1字段为允许启动阈值，非实测峰值",
                             "pair_bitset_microbench.json", "负载阻止测量；正确性测试另列"])

    transpose = documents.get("gpt_transpose_probe.json", {})
    if isinstance(transpose, dict) and transpose:
        grouped = defaultdict(lambda: ([], []))
        for run in transpose.get("runs", []):
            pair = grouped[(run.get("n"), run.get("symmetric"))]
            pair[1 if run.get("staged") else 0].append(run.get("total_s"))
        for condition, pair in grouped.items():
            add("gpt_transpose", str(condition), *pair, "秒/整组", "gpt_transpose_probe.json",
                "访存/分配", "纯写布局探针，不含分数计算，已废弃")

    def leaves(value, path=""):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from leaves(child, f"{path}.{key}" if path else key)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                yield from leaves(child, f"{path}[{i}]")
        else:
            yield path, value

    for filename, doc in documents.items():
        if not (filename.endswith(("_exact.json", "_tests.json")) or filename == "sort_summary.json"):
            continue
        for key, value in leaves(doc):
            if any(part in key for part in ("equal", "exact_cases", "passed", "failed", "exit_code",
                                            "rejected", "send_sync", "source_unchanged", "allocation_reused",
                                            "scope", "performance_benchmark_run", "size_cases", "families_per_size")):
                note = "局部验证，不能替代三天全市场对账"
                if key.endswith("_compile_exit_code") and isinstance(doc, dict):
                    expected = doc.get(key.removesuffix("_compile_exit_code") + "_rejected_with")
                    if expected:
                        note = f"负向生命周期测试：预期编译器以{expected}拒绝；该非零编译退出码不代表测试失败"
                tests.append(["实测记录", filename, key, value, note])
    for filename in sorted(documents):
        if filename.startswith(("sort_candidate_", "sort_guarded_")):
            excluded.append(["实测历史", "排序候选迭代", filename, "已被最终版本替代", "已排除",
                             None, None, None, None, None, None, "不计最终收益", "最终只用 sort_final_r*.json", filename, "旧候选"])
    return summaries, raw, tests, excluded


def analysis_tables(documents):
    rows = []
    skip = {"source_references", "source_sha256", "reviewed_source_sha256", "evidence_files"}

    def leaves(value, path=""):
        if isinstance(value, dict):
            for key, child in value.items():
                if key not in skip:
                    yield from leaves(child, f"{path}.{key}" if path else key)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                yield from leaves(child, f"{path}[{i}]")
        else:
            yield path, value

    for filename in ("pair_remaining_analysis.json", "gpt_remaining_analysis.json",
                     "source_audit_candidate3.json"):
        doc = documents.get(filename)
        if not isinstance(doc, dict):
            continue
        for key, value in leaves(doc):
            if "calculated_upper_bound" in key:
                category = "源码估计上限；非耗时"
            elif "calculated_sizes" in key or "chain_counts" in key or key.endswith((".before", ".after", ".saved")):
                category = "源码推导规模；非实测耗时/RSS"
            elif key.startswith(("measured_evidence", "v2_measured_coverage")):
                category = "引用已有实测记录"
            else:
                category = "源码审计/分析记录"
            rows.append([category, filename, key, value,
                         "原文截至该文件记录时点；不折算秒数或加速倍数。旧的未实现/未测试状态由较新的构建、局部测试、严格对账页补充。"])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    folder = root / "results/speed"
    output = args.output.resolve() if args.output else folder / "yupei_union_trim_speed.xlsx"
    documents, logs, index, warnings = {}, {}, [], []
    for file in sorted([*folder.glob("*.json"), *folder.glob("*.log")]):
        try:
            raw = file.read_bytes()
            if file.suffix == ".json":
                value = json.loads(raw)
                if not file.name.endswith(".labels.json"):
                    documents[file.name] = value
            else:
                logs[file.stem] = parse_log(raw.decode(errors="replace"))
            index.append(["实测文件", file.name, len(raw), hashlib.sha256(raw).hexdigest(),
                          datetime.fromtimestamp(file.stat().st_mtime, ZoneInfo("Asia/Shanghai")).isoformat(),
                          "标签文件仅记录指纹" if file.name.endswith(".labels.json") else "已读取", str(file)])
        except (OSError, ValueError) as error:
            warnings.append(["未知", file.name, f"读取时可能尚在写入：{error}"])
    decisions = documents.get("report_decisions.json", {})
    if not isinstance(decisions, dict):
        decisions = {}
    timing_exclusions = documents.get("timing_exclusions.json", {})
    if not isinstance(timing_exclusions, dict):
        timing_exclusions = {}
    runs = {}
    for name in sorted(set(logs) | {Path(name).stem for name in documents if RUN_NAME.fullmatch(Path(name).stem)}):
        match = RUN_NAME.fullmatch(name)
        if not match:
            continue
        variant, date, repeat = match.groups()
        log = logs.get(name, {})
        report = documents.get(name + ".json", {})
        if not isinstance(report, dict):
            report = {}
        record = {"variant": variant, "date": int(date), "repeat": int(repeat), **report}
        for event in log.get("events", []):
            if event.get("event") == "start":
                record.setdefault("load_before", event.get("load"))
                record.setdefault("module_sha256", event.get("module_sha256"))
            elif event.get("event") == "complete":
                for key, value in event.items():
                    record.setdefault(key, value)
        status = documents.get(name + ".status.json", {})
        record["exit_code"] = status.get("exit_code", log.get("exit_code")) if isinstance(status, dict) else log.get("exit_code")
        record["settings_evidence"] = "实测记录" if "experiment_settings" in record else "旧结果未记录；按默认空设置归组"
        record.setdefault("experiment_settings", {})
        if isinstance(record["experiment_settings"], dict):
            record["experiment_settings"] = {
                key: value for key, value in record["experiment_settings"].items()
                if key not in OPERATIONAL_SETTINGS
            }
        record["load_samples"] = status.get("load_samples") if isinstance(status, dict) else None
        record["max_load1"] = status.get("max_load1") if isinstance(status, dict) else None
        if not finite(record["max_load1"]) and isinstance(record["load_samples"], list):
            samples = [v[1] for v in record["load_samples"]
                       if isinstance(v, list) and len(v) > 1 and finite(v[1])]
            record["max_load1"] = max(samples) if samples else None
        record["log"] = log
        record["name"] = name
        reasons = []
        if record.get("validation_only", False):
            reasons.append("validation_only：只做数值验证，禁止计入性能")
        if not finite(record.get("call_seconds")):
            reasons.append("尚无完整调用耗时")
        if record["exit_code"] != 0:
            reasons.append("退出码未测" if record["exit_code"] is None else f"退出码={record['exit_code']}")
        if record.get("threads") != 50:
            reasons.append("配置线程数非50或未测")
        if log.get("thread_counts") and any(n != 50 for n in log["thread_counts"]):
            reasons.append("日志实际线程数不为50")
        load = first(record.get("load_before"))
        if not finite(load) or load > 50:
            reasons.append("启动load未测" if not finite(load) else "启动load超过50")
        if finite(record["max_load1"]) and record["max_load1"] > 100:
            reasons.append("运行期load1超过100")
        exclusion = timing_exclusions.get(name, {})
        if isinstance(exclusion, dict) and exclusion.get("exclude_from_timing_summary"):
            reasons.append("运行期异常，排除测速：" + exclusion.get("reason", "未记录原因"))
            record["observed_load1"] = exclusion.get("observed_load1", exclusion.get("max_load1"))
        record["eligible"] = not reasons
        record["reason"] = "计入重复测速" if not reasons else "；".join(reasons)
        runs[name] = record

    groups = defaultdict(list)
    for run in runs.values():
        if run["date"] == 20241231:
            groups[run_identity(run)].append(run)
    base_groups = [items for (variant, _, _), items in groups.items() if variant == "base"]
    base_values = [r["call_seconds"] for items in base_groups for r in items if r["eligible"]]
    base_median = median(base_values) if len(base_groups) == 1 else None
    summary_rows = []
    for (variant, module, settings), items in sorted(groups.items()):
        items.sort(key=lambda r: r["repeat"])
        good = [r for r in items if r["eligible"]]
        times = [r["call_seconds"] for r in good]
        med = median(times)
        first_run = next((r.get("call_seconds") for r in good if r["repeat"] == 1), None)
        state, reason = decision(decisions, "variants", variant)
        def budget(limit):
            if len(good) < 3:
                return "未完成3次重复；不作达标结论"
            return "通过" if med <= limit else "未达标"
        summary_rows.append(["实测/由实测计算", 20241231, variant, module, len(items), len(good),
                             first_run, min(times) if times else None, med,
                             ratio(base_median, med), 60, budget(60), 150, budget(150),
                             "冻结基线" if variant == "base" else status_label(state),
                             reason, ", ".join(r["name"] for r in items), settings,
                             "；".join(sorted({r["settings_evidence"] for r in items}))])

    raw_rows, stage_rows = [], []
    for name, run in sorted(runs.items()):
        log = run["log"]
        raw_rows.append(["实测", run["date"], run["variant"], run["repeat"], run.get("call_seconds"),
                         run.get("cpu_seconds"), ratio(run.get("cpu_seconds"), run.get("call_seconds")),
                         run.get("numpy_seconds"), run.get("write_seconds"), log.get("process_wall_seconds"),
                         run.get("threads"), log.get("thread_counts") or None,
                         run.get("load_before"), run.get("load_after"), run.get("observed_load1"), run["exit_code"],
                         run.get("max_rss_kib", log.get("max_rss_kib")), run.get("shape"),
                         run.get("nan_count"), run.get("inf_count"), run.get("module_sha256"),
                         run["eligible"], run["reason"], name + ".json / .log / .status.json",
                         run.get("validation_only", "旧结果未记录"), run["experiment_settings"],
                         run["settings_evidence"], run["max_load1"], run["load_samples"]])
        for group, stage, seconds, cumulative, note in log.get("stages", []):
            stage_rows.append(["实测日志/累计值差分", run["date"], run["variant"], run["repeat"],
                               group, stage, seconds, cumulative, note, name + ".log"])

    compare_rows = []
    passed_dates = defaultdict(set)
    for filename, doc in sorted(documents.items()):
        if not filename.startswith("compare_") or not isinstance(doc, dict):
            continue
        left, right = doc.get("left"), doc.get("right")
        left_run, right_run = runs.get(left, {}), runs.get(right, {})
        shapes = [left_run.get("shape"), right_run.get("shape")]
        shape_equal = doc.get("shape_exact")
        if shape_equal is None and all(shape is not None for shape in shapes):
            shape_equal = shapes[0] == shapes[1] == doc.get("shape")
        flags = [shape_equal, doc.get("names_exact"), doc.get("codes_exact"), doc.get("nan_exact"),
                 doc.get("array_equal_nan"), doc.get("bits_equal")]
        result = "通过" if all(flag is True for flag in flags) else "不通过" if any(flag is False for flag in flags) else "证据未齐"
        date = right_run.get("date", left_run.get("date"))
        variant = right_run.get("variant")
        if (result == "通过" and left_run.get("variant") == "base" and variant and date
                and left_run.get("date") == right_run.get("date")
                and right_run.get("module_sha256")):
            passed_dates[run_identity(right_run)].add(date)
        compare_rows.append(["实测产物对账", date, left, right, doc.get("shape"), *flags,
                             doc.get("mismatched_cells"), doc.get("max_abs_diff"), doc.get("max_rel_diff"),
                             result, filename, "shape判定优先报告字段，否则核对左右运行报告与compare的shape",
                             right_run.get("module_sha256"), right_run.get("experiment_settings"),
                             right_run.get("settings_evidence")])
    if not compare_rows:
        compare_rows.append(["未知", None, None, None, None, *([None] * 6), None, None, None, "尚无对账产物", None, "不读取大矩阵，不伪造一致结论", None, None, None])

    local, local_raw, local_tests, excluded = local_tables(documents, decisions)
    change_rows = []
    for filename, doc in sorted(documents.items()):
        if filename.endswith("_sources.json") and isinstance(doc, list):
            for entry in doc:
                change_rows.append(["实测指纹清单", filename, entry.get("path"), entry.get("sha256"),
                                    "该变体记录的文件；不代表当前已采用", str(root / entry.get("path", ""))])
    patch = root / "speed.patch"
    if patch.exists():
        for path in re.findall(r"^diff --git a/(.+?) b/.+$", patch.read_text(), flags=re.M):
            target = root / path
            current_hash = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None
            change_rows.append(["实测补丁路径/当前指纹", "speed.patch", path, current_hash,
                                "当前工作树；必须用运行模块SHA区分实际测了哪个二进制", str(target)])
    change_rows.append(["实测文件", "报告脚本", str(Path(__file__).relative_to(PROJECT)),
                        hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "报告生成器，不参与因子计算", str(Path(__file__))])

    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    notes = [
        ["实测", "生成时间（北京时间）", now],
        ["说明", "报告状态", "当前证据快照，可重复生成；有未测或证据未齐时不代表完成验收"],
        ["说明", "主计时口径", "函数入口到返回，包含读数据；numpy转换和结果落盘单列；不混入整进程启动时间"],
        ["说明", "基准样本", "20241231全部股票；50线程；启动load1≤50；退出码0；validation_only永不计性能；至少3次才给预算结论"],
        ["说明", "运行期异常排除", "timing_exclusions.json标记或status实测max_load1>100均排除性能汇总；原始读数与数值对账仍保留；未记录运行中load时明确未测"],
        ["说明", "首轮/最低/中位", "首轮为repeat=1；最低和中位使用同变体、同模块SHA、同experiment_settings的合格完成运行；首轮没有人为丢弃"],
        ["说明", "实验设置", "旧结果缺失experiment_settings时按默认空设置{}归组，并注明；计算设置不同的运行不能合并。YUPEI_STAGE/YUPEI_VALIDATE_ONLY/YUPEI_VERBOSE/YUPEI_GPT_VERBOSE仅控制运行或日志，排除出设置分组"],
        ["说明", "倍数", "冻结base的合格运行中位数 ÷ 本变体中位数；不足3次时仅为阶段性读数"],
        ["说明", "严格对账", "shape、列名顺序、股票顺序、NaN、值、完整位模式均须通过；比较输入不重新计算"],
        ["说明", "分段解释", "顺序总阶段用累计墙钟差分；并行、嵌套子段不相加，不能当作整段总耗时"],
        ["说明", "局部测试", "局部核、合成输入和单次排序的收益不等于全市场收益；未决和废弃候选移到单独页"],
        ["未知", "理论计算下限", "未提供实测读速与计算吞吐依据；本报告不估造下限"],
        ["说明", "证据来源", str(folder)],
    ]
    for key in sorted({run_identity(r) for r in runs.values() if r["variant"] != "base"}):
        variant, module, settings = key
        dates = sorted(passed_dates[key])
        complete = len(dates) >= 3 and 20241231 in dates
        notes.append(["实测对账覆盖", variant,
                      f"模块SHA256={module}；experiment_settings={settings}；通过日期={dates}；三天含20241231：{'通过' if complete else '未完成'}"])
    for note in decisions.get("notes", []):
        notes.append(["记录", "人工采用结论", note])
    notes.extend(warnings)

    wb = Workbook()
    wb.remove(wb.active)
    sheet(wb, "说明与证据", ["证据类别", "项目", "说明"], notes, "实测、由实测计算、源码记录与未知分开呈现；空缺明确写未测。")
    sheet(wb, "全市场基准", ["证据类别", "日期", "变体", "模块SHA256", "发现运行数", "合格重复数", "首轮调用秒", "最小调用秒", "中位调用秒", "相对base倍数", "预算秒", "60秒结论", "退让线秒", "150秒结论", "变体状态", "状态依据", "来源运行", "experiment_settings", "设置证据"], summary_rows,
          "只汇总20241231；同变体不同模块或实验设置分行；不足3次不宣布达标。")
    sheet(wb, "所有运行原始读数", ["证据类别", "日期", "变体", "重复号", "调用秒", "进程CPU秒", "CPU除以墙钟", "转numpy秒", "写结果秒", "整进程墙钟秒", "配置线程", "日志线程", "启动load1_5_15", "结束load1_5_15", "运行中异常load1", "退出码", "RSS峰值KiB", "shape", "NaN格数", "Inf格数", "模块SHA256", "计入汇总", "筛选原因", "来源", "validation_only", "experiment_settings", "设置证据", "实测max_load1", "load采样(秒/load1/5/15)"], raw_rows,
          "所有已发现运行均保留；进程CPU时间可以大于墙钟；缺退出码不能按成功计。")
    sheet(wb, "严格对账", ["证据类别", "日期", "左基线", "右候选", "shape记录", "shape一致", "names顺序一致", "codes顺序一致", "NaN位置一致", "array_equal_nan", "bits_equal", "不一致格数", "最大绝对差", "最大相对差", "结论", "来源", "shape依据", "候选模块SHA256", "候选experiment_settings", "设置证据"], compare_rows,
          "只引用完整compare产物；六项全部通过才标通过，不把无定义/未测当作0。")
    sheet(wb, "分段耗时", ["证据类别", "日期", "变体", "重复号", "阶段口径", "阶段或原文", "阶段墙钟秒", "累计墙钟秒", "说明", "来源"], stage_rows,
          "严禁把并行子段相加；累计总阶段差分只有日志显示的0.1秒精度。")
    local_headers = ["证据类别", "优化层级", "项目", "输入情形", "采用状态", "优化前中位", "优化后中位", "局部倍数", "单位", "基线重复数", "候选重复数", "适用边界", "口径说明", "来源", "采用依据"]
    sheet(wb, "采用候选局部测速", local_headers, local, "这里只列有明确采用记录的局部候选；不能相加或外推为全市场节时。")
    sheet(wb, "局部测速原始读数", ["证据类别", "项目", "来源", "重复号", "路径", "输入长度", "调用次数", "原始耗时", "原始单位", "局部读数", "局部单位", "输入范围", "说明"], local_raw,
          "保留最终版本每轮原值；候选网络读数也可能是未启用的20键探针，请按采用状态判断。")
    sheet(wb, "局部正确性", ["证据类别", "来源", "指标路径", "记录值", "适用边界"], local_tests, "测试通过只证明对应局部样本；整列顺序与三天全市场结果由严格对账页负责。")
    sheet(wb, "源码审计与未测", ["证据类别", "来源", "证据路径", "记录值", "适用边界"], analysis_tables(documents),
          "源码推导、估计与实测严格分开；字节、扫描次数、理论操作数绝不换算成耗时或达标结论。")
    sheet(wb, "废弃与未决候选", local_headers, excluded, "这些记录排除在采用候选收益之外；旧排序迭代、20键网络和transpose不作为最终收益。")
    sheet(wb, "变更文件清单", ["证据类别", "清单来源", "文件路径", "SHA256", "说明", "绝对路径"], change_rows, "源码指纹与运行模块指纹含义不同；不得据此声称未跑过的二进制已验证。")
    sheet(wb, "原始证据索引", ["证据类别", "文件", "字节数", "SHA256", "修改时间北京时间", "读取状态", "绝对路径"], index, "只读JSON与日志，不加载大矩阵；正在写入而未读全的文件会出现在说明页。")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.xlsx")
    wb.save(temporary)
    # Reopen the artifact to catch malformed XML/ZIP before replacing a previous report.
    verified = load_workbook(temporary, read_only=True)
    result = {"output": str(output), "sheets": verified.sheetnames,
              "runs": len(runs), "comparisons": sum(name.startswith("compare_") for name in documents),
              "read_warnings": len(warnings), "generated_at": now}
    verified.close()
    temporary.replace(output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
