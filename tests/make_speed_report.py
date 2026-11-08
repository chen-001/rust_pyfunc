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


def output_validation_reasons(record, status, documents):
    """Pure-data gate: a finished calculation is not a finished comparison."""
    reasons = []
    shape = record.get("shape")
    shape_valid = (isinstance(shape, (list, tuple)) and len(shape) == 2
                   and all(isinstance(value, int) and not isinstance(value, bool) for value in shape)
                   and shape[0] > 0 and shape[1] == 18758)
    if not shape_valid:
        reasons.append("输出shape未测或无效：必须为正行数×18758列")
    reference_name = f"base_{record['date']}_r1"
    if record["name"] == reference_name:
        return reasons, "原始base_r1历史对照例外：不要求自比较，仍校验固定shape"
    status = status if isinstance(status, dict) else {}
    comparison_code = status.get("comparison_exit_code")
    if not (isinstance(comparison_code, int) and not isinstance(comparison_code, bool)
            and comparison_code == 0):
        reasons.append("对账退出码未测：尚未完成对账" if comparison_code is None
                       else f"对账退出码不合格：comparison_exit_code={comparison_code}")
    filename = f"compare_{reference_name}__{record['name']}.json"
    comparison = documents.get(filename)
    if not isinstance(comparison, dict):
        reasons.append("缺少完整严格对账产物：" + filename)
        return reasons, filename
    if comparison.get("left") != reference_name or comparison.get("right") != record["name"]:
        reasons.append("严格对账左右运行身份不匹配")
    reference = documents.get(reference_name + ".json", {})
    reference_shape = reference.get("shape") if isinstance(reference, dict) else None
    if (not shape_valid or comparison.get("shape") != shape or reference_shape != shape
            or ("shape_exact" in comparison and comparison["shape_exact"] is not True)):
        reasons.append("严格对账shape不一致或基线shape缺失")
    bad_flags = [key for key in ("names_exact", "codes_exact", "nan_exact", "array_equal_nan", "bits_equal")
                 if comparison.get(key) is not True]
    if bad_flags:
        reasons.append("严格对账未通过/缺项：" + ", ".join(bad_flags))
    bad_diffs = [key for key in ("mismatched_cells", "max_abs_diff", "max_rel_diff")
                 if not finite(comparison.get(key)) or comparison[key] != 0]
    if bad_diffs:
        reasons.append("严格对账差异非0或未测：" + ", ".join(bad_diffs))
    return reasons, filename


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

    for project, filename, flag, baseline_value, exact_field, level in (
            ("pair_scan16", "pair_scan16_microbench.json", "mode", "reference",
             "same_inputs_bits_equal", "向量化：16通道保序扫描"),
            ("pair_topk_wide", "pair_topk_candidate3_microbench.json", "wide_first", False,
             "complete_keys_exactly_equal", "算法：完整键Top100阈值"),
            ("pair_scan_no_max", "pair_scan_no_max_microbench.json", "mode", "reference",
             None, "重复计算：移除未消费的最大值")):
        doc = documents.get(filename, {})
        if not isinstance(doc, dict) or not doc:
            continue
        exactness = doc.get("exactness", {})
        doc_exact = (doc.get("correctness", {}).get("exit_code") == 0
                     and all(exactness.get(key) is True for key in
                             ("h_g_arrays_bits_equal", "retained_statistics_bits_equal",
                              "repeated_calls_bits_equal", "inputs_unchanged")))
        if exact_field is None:
            for key, value in exactness.items():
                tests.append(["实测记录", filename, f"exactness.{key}", value,
                              "只验证该局部核保留输出，不代替三天全市场对账"])
        grouped = defaultdict(lambda: ([], []))
        for item in doc.get("records", []):
            seconds, calls = item.get("seconds"), item.get("iterations")
            per_call = seconds * 1e9 / calls if finite(seconds) and finite(calls) and calls > 0 else None
            load = first(item.get("load_before"))
            exact = item.get(exact_field) is True if exact_field else doc_exact
            admissible = (doc.get("status") == "completed" and doc.get("kernel_threads") == 1
                          and item.get("exit_code") == 0 and exact
                          and finite(load) and load <= 50 and finite(per_call)
                          and item.get("performance_valid_by_observed_load", True) is True)
            baseline = item.get(flag) == baseline_value
            if admissible:
                grouped[item.get("kind")][0 if baseline else 1].append(per_call)
            raw.append(["实测局部核", project, filename, item.get("repeat"),
                        "原版" if baseline else "候选", item.get("n"), calls, seconds, "秒/整组调用",
                        per_call, "ns/调用", item.get("kind"),
                        f"单核合成输入；exact={exact}；计入局部汇总={admissible}；"
                        "不含整日输入读取或其他因子计算"])
        for kind, (before, after) in grouped.items():
            add(project, kind, before, after, "ns/调用", filename, level,
                "单核、合成输入、每配置3次；只评价该内核，不外推50线程全市场收益")

    filename = "pair_mass4_tailref_ordered_microbench.json"
    mass4 = documents.get(filename, {})
    if isinstance(mass4, dict) and mass4:
        checks = mass4.get("correctness", [])
        exact = (len(checks) == 2 and {x.get("mass_avx512") for x in checks} == {"0", "1"}
                 and all(x.get("exit_code") == 0
                         and x.get("result", {}).get("six_accumulators_bits_equal") is True
                         and x.get("result", {}).get("repeated_batches_bits_equal") is True
                         for x in checks))
        grouped = defaultdict(lambda: ([], []))
        for item in mass4.get("records", []):
            seconds, calls = item.get("seconds"), item.get("iterations")
            per_batch = seconds * 1e9 / calls if finite(seconds) and finite(calls) and calls > 0 else None
            load = first(item.get("load_before"))
            admissible = (mass4.get("status") == "completed" and mass4.get("kernel_threads") == 1
                          and exact and item.get("exit_code") == 0 and finite(per_batch)
                          and finite(load) and load <= 50
                          and item.get("performance_valid_by_observed_load") is True)
            key = (item.get("n"), item.get("batch"), item.get("access"), item.get("kind"),
                   item.get("mass_avx512"))
            baseline = item.get("mode") == "reference"
            if admissible:
                grouped[key][0 if baseline else 1].append(per_batch)
            raw.append(["实测局部核", "pair_mass4", filename, item.get("repeat"),
                        "原版" if baseline else "候选", item.get("n"), calls, seconds, "秒/整组调用",
                        per_batch, f"ns/{item.get('batch')}对批次", str(key),
                        f"单核合成输入；最终版本双路径精确验证={exact}；计入局部汇总={admissible}"])
        for (n, batch, access, kind, avx512), (before, after) in grouped.items():
            add("pair_mass4", f"{access}/{kind}; n={n}; batch={batch}; AVX512={avx512}",
                before, after, f"ns/{batch}对批次", filename, "访存：四对质量统计保序批量更新",
                "单核合成输入；staged包含等值复制；与真实分数生成、TopK的缓存交互未测。混合开关含复制情形可能变慢，不外推整体收益")

    filename = "pair_r5_shared_xtx_screen_20261108.json"
    r5 = documents.get(filename, {})
    if isinstance(r5, dict) and r5:
        code = r5.get("correctness", {}).get("exit_code")
        failed = isinstance(code, int) and code != 0
        note = ("本轮XTX的NaN完整位模式不一致，正确性检查失败，未接入源码；编译成功不代表验证通过。"
                "performance_benchmark_run是任务计划标记；实际records为空，没有性能成绩。")
        for key, value in (("筛选结果", "不通过" if failed else "证据未齐"),
                           ("build.exit_code", r5.get("build", {}).get("exit_code")),
                           ("correctness.exit_code", code),
                           ("实际性能记录数", len(r5.get("records", []))),
                           ("correctness.stderr", r5.get("correctness", {}).get("stderr"))):
            tests.append(["实测局部筛选", filename, key, value, note])
        state, reason = decision(decisions, "local", "pair_r5_shared_xtx")
        excluded.append(["实测局部筛选失败" if failed else "筛选证据未齐", "重复计算：共享XTX",
                         "pair_r5_shared_xtx", r5.get("fixture"), status_label(state),
                         None, None, None, None, 0, 0, "无可用性能数据；禁止计入收益",
                         note, filename, reason])

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
        if not (filename.endswith(("_exact.json", "_tests.json", "_correctness.json")) or filename == "sort_summary.json"):
            continue
        if filename.startswith("pair_mass4") and filename.endswith("_correctness.json"):
            # Failed predecessors remain explicit failures. Only the ordered
            # final revision supplies the successful mass4 performance record.
            for check in doc.get("correctness", []):
                key = f"mass_avx512={check.get('mass_avx512')}"
                result = check.get("result", {})
                passed = (doc.get("status") == "passed" and check.get("exit_code") == 0
                          and result.get("six_accumulators_bits_equal") is True
                          and result.get("repeated_batches_bits_equal") is True)
                note = ("实际集成核的局部检查，仍不是整日验收" if doc.get("integrated_source")
                        else "最终修复候选的局部验证，仍需全市场验收") if passed else "修复前失败记录；不得作为通过证据或收益依据"
                tests.append(["实测记录", filename, key + ".结果", "通过" if passed else "不通过", note])
                tests.append(["实测记录", filename, key + ".exit_code", check.get("exit_code"), note])
                if result:
                    tests.append(["实测记录", filename, key + ".exact_cases", result.get("exact_cases"), note])
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
                     "source_audit_candidate3.json", "overnight_audit_20261108.json"):
        doc = documents.get(filename)
        if not isinstance(doc, dict):
            continue
        if filename == "overnight_audit_20261108.json":
            # Keep this historical snapshot compact. Per-run readings and
            # comparisons already have their own sheets; the index preserves
            # the complete JSON and its content hash.
            boundary = "历史快照，后续新增运行以全市场基准页为准；不足三轮只作阶段性比较。"
            rows.append(["历史审计快照", filename, "snapshot",
                         {key: doc.get(key) for key in ("snapshot_at", "scope", "status")}, boundary])
            for key, value in doc.get("measured_summary", {}).items():
                rows.append(["历史实测汇总", filename, f"measured_summary.{key}", value, boundary])
            for value in doc.get("measured_stage_comparison", []):
                rows.append(["历史实测分段比较", filename,
                             f"{value.get('group')}.{value.get('stage')}",
                             {key: value.get(key) for key in
                              ("base_median_seconds", "v2_median_seconds", "difference_seconds",
                               "base_repeats", "v2_repeats")}, value.get("interpretation", boundary)])
            for key, value in doc.get("current_work_condition_only", {}).items():
                rows.append(["由实测计算的条件界；非理论下限", filename, key, value,
                             "当前CPU工作量不变、理想均分50线程的条件界；改变算法或指令会改变工作量。"])
            source = doc.get("source_audit", {})
            rows.append(["源码审计/分析记录", filename, "source_audit",
                         {key: value for key, value in source.items() if key != "source_references"},
                         "源码关系不直接换算为秒数或加速倍率。"])
            for key, value in doc.get("unknowns", {}).items():
                rows.append(["未知/未测", filename, key, value, "按该历史快照时点记录；不把未测值当作0。"])
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


def gpt_screening_rows(documents, logs):
    """Keep the standalone 491-column, one-run experiment outside union groups."""
    screening = documents.get("gpt_screening_20261108.json", {})
    if not isinstance(screening, dict):
        return []
    rows = []
    decisions = {
        "all_off": "单轮比较参照",
        "batch1_off": "不默认采用：峰值内存较低，但网络段更慢；入口差同时受读取时间影响",
        "all_on": "不默认采用：网络段单次差不足1%，入口差主要来自读取时间；稳定收益未证实",
    }
    for record in screening.get("records", []):
        filename = Path(record.get("evidence_json", "")).name
        raw = documents.get(filename, {})
        log = logs.get(Path(filename).stem, {})
        stages = {name: value for group, name, value, _, _ in log.get("stages", [])
                  if group == "GPT内部记录"}
        reference = record.get("setting") == "all_off"
        comparison = raw.get("comparison", {})
        exact = (all(comparison.get(key) is True for key in
                     ("codes_exact", "names_exact", "shape_exact", "nan_positions_equal",
                      "defined_values_bits_equal", "array_equal_equal_nan"))
                 and comparison.get("max_abs_diff") == 0 and comparison.get("max_rel_diff") == 0)
        admissible = (screening.get("status") == "completed_screening_pending_decision"
                      and record.get("exit_code") == 0 and raw.get("function") == "py_yupei_gpt"
                      and raw.get("shape") == [5107, 491] and raw.get("rayon_threads") == 50
                      and raw.get("date") == 20241231 and finite(raw.get("wall_s"))
                      and raw.get("performance_valid_by_load") is True
                      and bool(log.get("thread_counts"))
                      and all(n == 50 for n in log["thread_counts"])
                      and (reference or exact))
        rows.append(["实测单轮筛选", record.get("setting"), raw.get("date"), raw.get("shape"),
                     raw.get("rayon_threads"), 1, raw.get("wall_s"), stages.get("network"),
                     stages.get("read_and_aggregate"), raw.get("process_cpu_s"), raw.get("max_rss_kib"),
                     raw.get("load1_start"), raw.get("sampled_load1_max"),
                     "参照输出" if reference else "值逐位一致、差0" if exact else "证据未齐/不通过",
                     "单轮筛选通过；不计入union重复" if admissible else "证据未齐/不通过；不作性能结论",
                     decisions.get(record.get("setting"), "尚无采用决定"),
                     f"gpt_screening_20261108.json；{filename}；{Path(filename).stem}.log"])
    return rows


def pair_workload_tables(documents):
    """Show 28 artifact-proven chains; keep resource models in short footnotes."""
    filename = "pair_workload_audit.json"
    audit = documents.get(filename)
    if not isinstance(audit, dict):
        return [], []
    totals = audit.get("totals", {})
    def count_text(key):
        value = totals.get(key)
        return f"{value:,}" if isinstance(value, int) and not isinstance(value, bool) else UNKNOWN
    diagnostic = audit.get("mass_flags_diagnostic", {})
    modes = diagnostic.get("modes", {})
    flag_keys = ("none", "h_only", "g_only", "both")
    mode_rows = {mode: {row.get("name"): row for row in modes.get(mode, {}).get("rows", [])}
                 for mode in ("v5m1", "v5m4")}
    rows = []
    checks = {"v5m1": [], "v5m4": [], "cross_mode": []}
    for chain in audit.get("chains", []):
        name = chain.get("name")
        pairs = chain.get("accepted_pairs")
        witnesses = chain.get("count_witnesses", [])
        witness_ok = bool(witnesses) and all(
            witness.get("mask_matches") is True and witness.get("verified_equal") is True
            for witness in witnesses)
        row = ["产物计数证明/日志实测", chain.get("group"), name, chain.get("n"),
               chain.get("active_mask_count"), pairs, chain.get("pair_stock_elements"),
               "通过" if witness_ok else "证据未齐/不通过"]
        complete_flags = {}
        for mode in ("v5m1", "v5m4"):
            measured = mode_rows[mode].get(name, {})
            values = [measured.get(key) for key in flag_keys]
            complete = modes.get(mode, {}).get("exit_code") == 0 and all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in values)
            agrees = (sum(values) == measured.get("pairs") == pairs
                      and measured.get("equals_v4_pair_count_witness") is True) if complete else None
            row.extend([" / ".join(map(str, values)) if complete else None,
                        measured.get("pairs") if complete else None, agrees])
            checks[mode].append(agrees)
            if complete:
                complete_flags[mode] = values
        cross_mode = (complete_flags["v5m1"] == complete_flags["v5m4"]
                      if len(complete_flags) == 2 else None)
        checks["cross_mode"].append(cross_mode)
        row.extend([cross_mode, filename])
        rows.append(row)
    notes = [
        ["产物计数证明", "配对主循环的真实工作量",
         f"{count_text('chains')}条链；{count_text('accepted_pairs')}个配对；"
         f"{count_text('pair_stock_elements')}个配对×股票元素（约1.8618万亿）。"
         "由活跃掩码和产物TopK计数饱和等式证明，未假设全部股票活跃。仅含split/multi的28条链主循环，"
         "不含GPT、额外空模型扫描、读文件、组装及下游统计。"],
        ["实测日志对账", "配对开关flags核对",
         f"审计状态={diagnostic.get('status', UNKNOWN)}；"
         + "；".join(f"{mode}通过{sum(value is True for value in values)}/{len(values)}条，"
                     f"未测{sum(value is None for value in values)}条"
                     for mode, values in checks.items())
         + "。工作量审计页四项顺序为none / h_only / g_only / both；完整退出后才展示该模式的计数。"
         "这里只核对工作量；整日因子一致性与速度重复见严格对账和全市场基准页。"],
        ["未知", "总理论下限与停止条件",
         "全程序的、独立于当前算法的理论下限未知，不能声称已经到理论下限的2倍以内。"
         "CPU秒/50、局部单核吞吐外推和当前指令的条件资源界均不能替代总理论下限。"],
    ]
    traffic = audit.get("conditional_logical_traffic_decimal_TB", {})
    traffic_tb = traffic.get("total_84B_if_all_both_true")
    if finite(traffic_tb):
        notes.append(["条件模型附注", "逻辑缓存流量",
                      f"全部mass标志为both且按pair1实现时，源代码模型约{traffic_tb:.3f} TB；"
                      "这是寄存器/缓存之间的逻辑载荷，不是实际DRAM流量，也不是磁盘I/O量。"
                      "理想硬件资源界还依赖当前向量物化、50个物理核、最高频率、"
                      "理想缓存命中等假设，仅作附注，不能据此预测总耗时或宣称接近极限。"])
    return rows, notes


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
        reasons, record["correctness_evidence"] = output_validation_reasons(record, status, documents)
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
        record["reason"] = ("计入重复测速；" + record["correctness_evidence"]) if not reasons else "；".join(reasons)
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
        first_attempt = next((r for r in items if r["repeat"] == 1), None)
        first_run = first_attempt.get("call_seconds") if first_attempt else None
        first_reason = first_attempt["reason"] if first_attempt else "该组未记录repeat=1"
        state, reason = decision(decisions, "variants", variant)
        variant_decision = decisions.get("variants", {}).get(variant, {})
        variant_label = (variant_decision.get("label", status_label(state))
                         if isinstance(variant_decision, dict) else status_label(state))
        def budget(limit):
            if len(good) < 3:
                return "未完成3次重复；不作达标结论"
            return "通过" if med <= limit else "未达标"
        summary_rows.append(["实测/由实测计算", 20241231, variant, module, len(items), len(good),
                             first_run, min(times) if times else None, med,
                             ratio(base_median, med), 60, budget(60), 150, budget(150),
                             "冻结基线" if variant == "base" else variant_label,
                             reason, ", ".join(r["name"] for r in items), settings,
                             "；".join(sorted({r["settings_evidence"] for r in items})), first_reason])

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
                         run["settings_evidence"], run["max_load1"], run["load_samples"],
                         run["correctness_evidence"]])
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
    gpt_screening = gpt_screening_rows(documents, logs)
    pair_workload, pair_workload_notes = pair_workload_tables(documents)
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
        ["说明", "报告状态", "当前证据快照，可重复生成；各变体分别判断正确性、测速验收与预算达标，未决实验不混入已验收变体"],
        ["说明", "主计时口径", "函数入口到返回，包含读数据；numpy转换和结果落盘单列；不混入整进程启动时间"],
        ["说明", "基准样本", "20241231全部股票；50线程；启动load1≤50；退出码0；validation_only永不计性能；至少3次才给预算结论"],
        ["说明", "运行期异常排除", "timing_exclusions.json标记或status实测max_load1>100均排除性能汇总；原始读数与数值对账仍保留；未记录运行中load时明确未测"],
        ["说明", "首轮/最低/中位", "首轮显示repeat=1的原始调用读数，即使该轮不合格也保留并单列资格说明；最低和中位只使用同变体、同模块SHA、同experiment_settings的合格完成运行"],
        ["说明", "实验设置", "旧结果缺失experiment_settings时按默认空设置{}归组，并注明；计算设置不同的运行不能合并。YUPEI_STAGE/YUPEI_VALIDATE_ONLY/YUPEI_VERBOSE/YUPEI_GPT_VERBOSE仅控制运行或日志，排除出设置分组"],
        ["说明", "倍数", "冻结base的合格运行中位数 ÷ 本变体中位数；不足3次时仅为阶段性读数"],
        ["说明", "严格对账", "shape、列名顺序、股票顺序、NaN、值、完整位模式均须通过；比较输入不重新计算"],
        ["说明", "测速资格的数值门槛", "必须为正行数×18758列；除各日期原始base_r1的历史对照例外外，须有对应compare产物、左右身份及shape一致、names/codes/NaN/值/位模式全通过、差异为0，且status中的comparison_exit_code=0。计算退出0但对账未完成的轮次不进入最低/中位。"],
        ["说明", "分段解释", "顺序总阶段用累计墙钟差分；并行、嵌套子段不相加，不能当作整段总耗时"],
        ["说明", "局部测试", "局部核、合成输入和单次排序的收益不等于全市场收益；未决和废弃候选移到单独页"],
        ["未知", "理论计算下限", "全程序理论下限尚未建立；工作量和条件资源估算不等同于总理论下限，不能声称已在其2倍以内"],
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
    overnight = documents.get("overnight_audit_20261108.json")
    if isinstance(overnight, dict):
        notes.append(["历史审计快照", "20261108夜间验收审计",
                      f"记录时间={overnight.get('snapshot_at', UNKNOWN)}；"
                      f"范围={overnight.get('scope', UNKNOWN)}；"
                      "详见源码审计与未测页。后续新增运行以全市场基准页为准，"
                      "CPU秒除以50只描述当前工作量不变时的条件界，不是算法理论下限。"])
    final_acceptance = documents.get("candidate6_acceptance_summary.json")
    if isinstance(final_acceptance, dict):
        notes.append(["已完成验收的实测汇总", "候选6相对冻结基线",
                      {key: final_acceptance.get(key) for key in
                       ("base_median_seconds", "candidate_first_seconds", "candidate_min_seconds",
                        "candidate_median_seconds", "median_seconds_saved", "median_speedup",
                        "median_time_reduction_pct", "budget_met", "fallback_met")}])
        notes.append(["独立核验记录", "候选6数值与源码覆盖",
                      f"来源=candidate6_acceptance_summary.json；核验时间={final_acceptance.get('verified_at', UNKNOWN)}；"
                      f"三日全股逐位一致日期={final_acceptance.get('correctness_dates', UNKNOWN)}；"
                      f"核对源码文件数={final_acceptance.get('all_sources_verified', UNKNOWN)}；"
                      f"模块SHA256={final_acceptance.get('module_sha256', UNKNOWN)}。"
                      "源码文件数来自该独立核验记录；原始base首轮只有起止负载，作为明示历史例外保留。"
                      "总理论下限仍未知，不能声称接近极限。"])
    notes.extend(pair_workload_notes)
    notes.extend(warnings)

    wb = Workbook()
    wb.remove(wb.active)
    sheet(wb, "说明与证据", ["证据类别", "项目", "说明"], notes, "实测、由实测计算、源码记录与未知分开呈现；空缺明确写未测。")
    sheet(wb, "全市场基准", ["证据类别", "日期", "变体", "模块SHA256", "发现运行数", "合格重复数", "首轮调用秒", "最小调用秒", "中位调用秒", "相对base倍数", "预算秒", "60秒结论", "退让线秒", "150秒结论", "变体状态", "状态依据", "来源运行", "experiment_settings", "设置证据", "首轮资格说明"], summary_rows,
          "只汇总20241231；同变体不同模块或实验设置分行；不足3次不宣布达标。")
    sheet(wb, "所有运行原始读数", ["证据类别", "日期", "变体", "重复号", "调用秒", "进程CPU秒", "CPU除以墙钟", "转numpy秒", "写结果秒", "整进程墙钟秒", "配置线程", "日志线程", "启动load1_5_15", "结束load1_5_15", "运行中异常load1", "退出码", "RSS峰值KiB", "shape", "NaN格数", "Inf格数", "模块SHA256", "计入汇总", "筛选原因", "来源", "validation_only", "experiment_settings", "设置证据", "实测max_load1", "load采样(秒/load1/5/15)", "数值资格证据"], raw_rows,
          "所有已发现运行均保留；进程CPU时间可以大于墙钟；缺退出码不能按成功计。")
    sheet(wb, "严格对账", ["证据类别", "日期", "左基线", "右候选", "shape记录", "shape一致", "names顺序一致", "codes顺序一致", "NaN位置一致", "array_equal_nan", "bits_equal", "不一致格数", "最大绝对差", "最大相对差", "结论", "来源", "shape依据", "候选模块SHA256", "候选experiment_settings", "设置证据"], compare_rows,
          "只引用完整compare产物；六项全部通过才标通过，不把无定义/未测当作0。")
    sheet(wb, "分段耗时", ["证据类别", "日期", "变体", "重复号", "阶段口径", "阶段或原文", "阶段墙钟秒", "累计墙钟秒", "说明", "来源"], stage_rows,
          "严禁把并行子段相加；累计总阶段差分只有日志显示的0.1秒精度。")
    if gpt_screening:
        sheet(wb, "GPT单轮筛选", ["证据类别", "设置", "日期", "股票数及列数", "线程", "每配置重复数",
                                  "入口秒", "网络秒", "读取及聚合秒", "CPU秒", "RSS峰值KiB",
                                  "启动load1", "实测load1峰值", "与参照对账", "资格说明", "采用结论", "来源"],
              gpt_screening,
              "独立py_yupei_gpt包含491列（含leadlag）；四合一仅用其455列、总18758列。每设置只测1次，不并入四合一重复或稳定加速结论。后两次读取变快不能归因于计算改动。")
    if pair_workload:
        sheet(wb, "工作量审计", ["证据类别", "组", "链", "股票数", "活跃股票数", "真实配对数",
                                  "配对×股票元素数", "产物计数证明", "v5m1四类flags", "v5m1配对合计",
                                  "v5m1与产物对账", "v5m4四类flags", "v5m4配对合计", "v5m4与产物对账",
                                  "两模式flags一致", "来源"], pair_workload,
              "28条主循环链的真实工作量，四类flags顺序：none / h_only / g_only / both。总计约1.8618万亿配对×股票元素。全程序理论下限仍未知，不能声称已到2倍以内；范围和条件附注见说明页。")
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
