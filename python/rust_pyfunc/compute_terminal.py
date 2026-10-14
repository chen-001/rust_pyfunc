#!/usr/bin/env python3
"""共享计算环境：compute-terminal terminal / run COMMAND / status / install。

内存、交换空间、线程数在 compute.slice 上合计限额，不是每个任务各领一份。
未开放 CPU 控制器时使用继承的 CPU affinity；程序主动重设 affinity 可绕过，
因此状态必须区分此模式与管理员开放后的 CPU 总额度，不能伪称硬限制。
普通软件不会自动纳入；只有本入口启动的程序及其普通子孙进程受保护。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import uuid

SLICE = "compute.slice"
CONFIG_DIR = Path.home() / ".config/compute-terminal"
CONFIG_PATH = CONFIG_DIR / "config.json"
CGROUP_ROOT = Path("/sys/fs/cgroup")
GIB = 1024**3
_SCOPE_RE = re.compile(r"^compute-[a-zA-Z0-9-]+\.scope$")


def _systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["systemctl", "--user", "--no-ask-password", *args],
        capture_output=True, text=True, timeout=15, check=check,
    )


def _cpu_groups() -> list[list[int]]:
    """按物理核分组并轮流选各插槽，连同同核的两个线程一起选，留出完整物理核。"""
    available = os.sched_getaffinity(0)
    packages: dict[int, dict[int, list[int]]] = {}
    for cpu in sorted(available):
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        package = int((base / "physical_package_id").read_text())
        core = int((base / "core_id").read_text())
        packages.setdefault(package, {}).setdefault(core, []).append(cpu)
    queues = [list(cores.values()) for _, cores in sorted(packages.items())]
    return [queue[i] for i in range(max(map(len, queues))) for queue in queues if i < len(queue)]


def default_config() -> dict:
    cpus: list[int] = []
    target = min(256, max(1, len(os.sched_getaffinity(0)) // 2))
    for group in _cpu_groups():
        if len(cpus) + len(group) > target:
            break
        cpus.extend(group)
    cpus = sorted(cpus or [min(os.sched_getaffinity(0))])
    return {
        "cpus": cpus, "memory_high_gib": 896, "memory_max_gib": 1024,
        "swap_max_gib": 16, "tasks_max": 32768,
    }


def load_config() -> dict:
    if not CONFIG_PATH.is_file():
        raise RuntimeError("计算环境尚未安装，请先执行 compute-terminal install")
    cfg = json.loads(CONFIG_PATH.read_text())
    cpus = cfg.get("cpus")
    if not isinstance(cpus, list) or not cpus or any(type(c) is not int or c < 0 for c in cpus):
        raise ValueError("cpus 必须是非空的 CPU 编号列表")
    cfg["cpus"] = sorted(set(cpus))
    for key in ("memory_high_gib", "memory_max_gib", "swap_max_gib", "tasks_max"):
        if type(cfg.get(key)) is not int or cfg[key] < (0 if key == "swap_max_gib" else 1):
            raise ValueError(f"{key} 必须是有效整数")
    if cfg["memory_high_gib"] > cfg["memory_max_gib"]:
        raise ValueError("内存减速线不能大于内存上限")
    return cfg


def _slice_path() -> Path:
    out = _systemctl("show", SLICE, "--property=ControlGroup", "--value")
    relative = out.stdout.strip()
    if not relative.startswith("/user.slice/") or not relative.endswith("/" + SLICE):
        raise RuntimeError("无法确认计算环境的资源分组")
    return CGROUP_ROOT / relative.lstrip("/")


def _read(path: Path, name: str, default: str = "unavailable") -> str:
    try:
        return (path / name).read_text().strip()
    except FileNotFoundError:
        return default


def status() -> dict:
    """从内核读取真正生效的限额；systemctl 的期望配置不等于已经生效。"""
    try:
        cfg = load_config()
        path = _slice_path()
        actual = {name: _read(path, name) for name in (
            "memory.high", "memory.max", "memory.swap.max", "memory.current",
            "memory.swap.current", "pids.max", "pids.current", "cpu.max",
            "cpuset.cpus.effective", "memory.events",
        )}
        ready = all(actual[name] == str(cfg[key] * scale) for name, key, scale in (
            ("memory.high", "memory_high_gib", GIB), ("memory.max", "memory_max_gib", GIB),
            ("memory.swap.max", "swap_max_gib", GIB), ("pids.max", "tasks_max", 1),
        ))
        quota = actual["cpu.max"].split()
        cpu_hard = (len(quota) == 2 and quota[0].isdigit() and quota[1].isdigit()
                    and 0 < int(quota[0]) <= len(cfg["cpus"]) * int(quota[1]))
        cpu_mode = "quota" if cpu_hard else "affinity"
        message = (
            f"所有计算任务共享：内存上限 {cfg['memory_max_gib']} GiB，"
            f"交换空间上限 {cfg['swap_max_gib']} GiB，线程/进程上限 {cfg['tasks_max']}；"
            + (f"CPU 总额度已启用（最多相当于 {len(cfg['cpus'])} 个 CPU）"
               if cpu_hard else f"默认在指定的 {len(cfg['cpus'])} 个 CPU 上运行，严格 CPU 总额度待管理员开放")
        )
        if not ready:
            message = "计算保护未就绪，新任务将拒绝启动；请执行 compute-terminal install"
        return {"ready": ready, "slice": SLICE, "cgroup": str(path), "config": cfg,
                "actual": actual, "cpu_mode": cpu_mode, "cpu_hard_limit": cpu_hard,
                "message": message}
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        return {"ready": False, "cpu_hard_limit": False, "message": str(exc)}


def prepare_command(command: list[str], unit: str | None = None,
                    result_path: str | None = None) -> tuple[list[str], dict]:
    """scope 保留调用方环境、目录、输出与退出码；故障时禁止退回裸执行。"""
    if not command:
        raise ValueError("没有指定要运行的程序")
    unit = unit or f"compute-run-{uuid.uuid4().hex}.scope"
    if not _SCOPE_RE.fullmatch(unit):
        raise ValueError("非法的计算任务分组名称")
    _systemctl("start", SLICE)
    facts = status()
    if not facts["ready"]:
        raise RuntimeError(facts["message"])
    facts["scope"] = unit
    if result_path:
        facts["result_path"] = result_path
        command = [sys.executable, str(Path(__file__).resolve()), "_supervise", result_path, "--", *command]
    return (["systemd-run", "--user", "--scope", "--quiet", "--collect",
             f"--slice={SLICE}", f"--unit={unit}", "--property=TimeoutStopSec=3s", "--",
             sys.executable, str(Path(__file__).resolve()), "_enter", "--", *command], facts)


def _enter(command: list[str]) -> None:
    """先核验已进入围栏，再执行用户程序；不在 daemon 中运行 preexec_fn。"""
    facts = status()
    membership = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[-1]
    expected = "/" + str(Path(facts.get("cgroup", "")).relative_to(CGROUP_ROOT)) if facts.get("cgroup") else ""
    if not facts["ready"] or not expected or not membership.startswith(expected + "/"):
        raise RuntimeError("未进入生效的计算保护环境，拒绝运行")
    cfg = facts["config"]
    cpus = set(cfg["cpus"]) & os.sched_getaffinity(0)
    if not cpus:
        raise RuntimeError("指定的计算 CPU 当前不可用")
    os.sched_setaffinity(0, cpus)
    # 让同 CPU 上的桌面交互优先；子孙进程默认继承，自己主动修改则不构成硬隔离。
    os.nice(max(0, 10 - os.getpriority(os.PRIO_PROCESS, 0)))
    os.environ["COMPUTE_TERMINAL"] = "1"
    print("[计算保护] " + facts["message"], file=sys.stderr, flush=True)
    os.execvp(command[0], command)


def _supervise(result_path: str, command: list[str]) -> int:
    """完成回执由围栏内的监督进程写入，taskd 重启也不会丢失真实退出码。"""
    result = subprocess.run(command)
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"returncode": result.returncode}) + "\n")
    temporary.replace(path)
    return result.returncode if result.returncode >= 0 else 128 - result.returncode


def stop_scope(unit: str) -> None:
    """停止整个分组，包括已经 setsid、脱离父进程组的计算子进程。"""
    if not _SCOPE_RE.fullmatch(unit):
        raise ValueError("拒绝停止非计算分组")
    # 已完成、被 --collect 回收的 scope 无需停止。
    result = _systemctl("stop", unit, check=False)
    if result.returncode and "not loaded" not in result.stderr:
        raise RuntimeError(result.stderr.strip())


def contains_pid(pid: int) -> bool:
    try:
        cg = Path(f"/proc/{pid}/cgroup").read_text().strip().split("::", 1)[-1]
        return "/" + SLICE + "/" in cg
    except OSError:
        return False


def task_is_protected(pid: int, run_env: str | None) -> bool:
    """taskd 登记的 PID 是监督进程，实际计算进程位于独立 scope。"""
    if contains_pid(pid):
        return True
    try:
        unit = json.loads(run_env or "{}").get("computeProtection", {}).get("scope", "")
        if not _SCOPE_RE.fullmatch(unit):
            return False
        result = _systemctl("show", unit, "--property=ControlGroup", "--value", check=False)
        relative = result.stdout.strip()
        if not relative.endswith("/" + unit) or "/" + SLICE + "/" not in relative:
            return False
        return "populated 1" in _read(CGROUP_ROOT / relative.lstrip("/"), "cgroup.events")
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
        return False


def _install_dashboard_banner() -> None:
    """任务管理台使用已有 /api/config，无新增服务；重复安装不重复注入。"""
    template = Path.home() / "progress_dashboard/templates/index.html"
    if not template.exists():
        return
    source = template.read_text()
    if 'id="compute-protection"' in source:
        return
    anchor = '<div id="tasks-panel" class="p-6">'
    js_anchor = "const configPromise = fetch('/api/config').then(r => r.json()).then(cfg => {"
    if anchor not in source or js_anchor not in source:
        raise RuntimeError("任务管理台模板已变化，无法安装保护状态提示")
    shutil.copy2(template, CONFIG_DIR / "dashboard-before-install.html")
    source = source.replace(anchor, anchor + '\n        <div id="compute-protection" class="card p-4 mb-6 text-sm" role="status">正在确认计算保护状态…</div>', 1)
    source = source.replace(js_anchor, js_anchor + "\n            const protection = document.getElementById('compute-protection');\n            if (protection) {\n                protection.textContent = cfg.compute?.message || '未获取到计算保护状态';\n                if (cfg.unprotected_running?.length) protection.textContent += '；已有任务尚未纳入：' + cfg.unprotected_running.map(t => t.name).join('、');\n                protection.style.color = cfg.compute?.ready && cfg.compute?.cpu_hard_limit ? '#166534' : '#92400e';\n            }", 1)
    template.write_text(source)


def install() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(default_config(), ensure_ascii=False, indent=2) + "\n")
    cfg = load_config()
    units = Path.home() / ".config/systemd/user"
    units.mkdir(parents=True, exist_ok=True)
    (units / SLICE).write_text(
        "[Unit]\nDescription=计算终端与因子任务共享资源额度\n\n[Slice]\n"
        f"MemoryHigh={cfg['memory_high_gib']}G\nMemoryMax={cfg['memory_max_gib']}G\n"
        f"MemorySwapMax={cfg['swap_max_gib']}G\nTasksMax={cfg['tasks_max']}\n"
        f"CPUQuota={len(cfg['cpus']) * 100}%\nCPUWeight=50\n"
        f"AllowedCPUs={','.join(map(str, cfg['cpus']))}\n"
    )
    # 任务服务启动时先准备共享资源组；daemon 自身仍放在组外以便管理失控任务。
    if (units / "factor-taskd.service").exists():
        dependency = units / "factor-taskd.service.d"
        dependency.mkdir(parents=True, exist_ok=True)
        (dependency / "90-compute-terminal.conf").write_text(
            "[Unit]\nWants=compute.slice\nAfter=compute.slice\n"
        )
    local_bin = Path.home() / ".local/bin"
    local_bin.mkdir(parents=True, exist_ok=True)
    launcher = local_bin / "compute-terminal"
    launcher.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " "
                        + shlex.quote(str(Path(__file__).resolve())) + ' "$@"\n')
    launcher.chmod(0o755)
    desktop_dir = Path.home() / ".local/share/applications"
    desktop_dir.mkdir(parents=True, exist_ok=True)
    terminal = shutil.which("xfce4-terminal")
    if terminal:
        (desktop_dir / "compute-terminal.desktop").write_text(
            "[Desktop Entry]\nType=Application\nName=计算终端\n"
            "Comment=计算程序共享 CPU、内存和线程额度\nIcon=utilities-terminal\n"
            f"Exec={terminal} --disable-server --title=计算终端 --execute {launcher} terminal\n"
            "Terminal=false\nCategories=System;TerminalEmulator;\n"
        )
    _install_dashboard_banner()
    _systemctl("daemon-reload")
    _systemctl("start", SLICE)
    # 已运行的 slice 在 daemon-reload 后仍要显式更新可用的控制器。
    _systemctl("set-property", SLICE,
               f"MemoryHigh={cfg['memory_high_gib']}G", f"MemoryMax={cfg['memory_max_gib']}G",
               f"MemorySwapMax={cfg['swap_max_gib']}G", f"TasksMax={cfg['tasks_max']}",
               f"CPUQuota={len(cfg['cpus']) * 100}%", "CPUWeight=50",
               f"AllowedCPUs={','.join(map(str, cfg['cpus']))}")
    facts = status()
    if not facts["ready"]:
        raise RuntimeError(facts["message"])
    print(facts["message"])


def main() -> int:
    parser = argparse.ArgumentParser(description="计算终端：所有任务共享资源额度")
    parser.add_argument("action", nargs="?", default="terminal",
                        choices=["install", "status", "terminal", "run", "_enter", "_supervise"])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        if args.action == "install":
            install()
        elif args.action == "status":
            facts = status()
            print(json.dumps(facts, ensure_ascii=False, indent=2) if "--json" in command else facts["message"])
            return 0 if facts["ready"] else 1
        elif args.action == "_enter":
            _enter(command)
        elif args.action == "_supervise":
            return _supervise(command[0], command[2:] if command[1:2] == ["--"] else command[1:])
        else:
            if args.action == "terminal":
                command = [os.environ.get("SHELL", "/bin/bash"), "-i"]
            cmd, _ = prepare_command(command)
            os.execvp(cmd[0], cmd)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"计算保护错误：{exc}", file=sys.stderr)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
