"""真实 systemd scope 小规模验收；超限只在 64 MiB 的独立测试 scope 中触发。"""
import importlib
import json
import os
from pathlib import Path
import subprocess
import socket
import sys
import time
import uuid
from urllib.request import Request, urlopen

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python/rust_pyfunc"))
ct = importlib.import_module("compute_terminal")
td = importlib.import_module("factor_taskd")


def wait_for(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("等待计算保护验收超时")


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    if not ct.status()["ready"]:
        pytest.skip("需要先安装用户计算环境")
    td.init_db(str(tmp_path / "tasks.db"))
    monkeypatch.setattr(td, "_max_concurrent", None)
    monkeypatch.setattr(td, "CONFIG_PATH", str(tmp_path / "task-config.json"))
    monkeypatch.setattr(td, "_probe_rust_runtime", lambda _: {
        "python": sys.executable, "pythonVersion": sys.version.split()[0],
        "rustPyfuncFile": None, "workerPath": None, "workerProbeError": None,
    })
    yield td
    for task in td.list_tasks():
        if task["status"] in ("running", "pending"):
            td.cancel_task(task["id"])
    wait_for(lambda: not td._active_processes)
    for task in td.list_tasks():
        if task.get("log_path"):
            Path(task["log_path"]).unlink(missing_ok=True)


def submit(daemon, tmp_path, source, name="计算环境验收"):
    script = tmp_path / f"{name}{uuid.uuid4().hex[:4]}.py"
    script.write_text(source)
    response = daemon.submit_task(td.TaskSubmit(
        script_path=str(script), n_jobs=7, start_date=20220101, end_date=20220131,
    ))
    return response["id"]


def test_two_tasks_share_limits_and_inherit_environment(daemon, tmp_path):
    paths = [tmp_path / f"result{i}.json" for i in range(2)]
    tasks = []
    for path in paths:
        tasks.append(submit(daemon, tmp_path,
            "import os,json,pathlib,subprocess,time\n"
            "child=subprocess.check_output(['/bin/cat','/proc/self/cgroup'],text=True).strip()\n"
            f"pathlib.Path({str(path)!r}).write_text(json.dumps(dict("
            "cgroup=pathlib.Path('/proc/self/cgroup').read_text().strip(),child=child,"
            "affinity=sorted(os.sched_getaffinity(0)),cwd=os.getcwd(),"
            "njobs=os.environ['FACTOR_N_JOBS'],start=os.environ['FACTOR_START_DATE'],"
            "end=os.environ['FACTOR_END_DATE'])))\ntime.sleep(30)\n"))
    wait_for(lambda: all(p.exists() for p in paths))
    results = [json.loads(p.read_text()) for p in paths]
    parents = []
    for result in results:
        assert result["cgroup"] == result["child"]
        group = ct.CGROUP_ROOT / result["cgroup"].split("::", 1)[1].lstrip("/")
        parents.append(group.parent)
        assert group.parent.name == "compute.slice"
        assert (group.parent / "memory.max").read_text().strip() == str(1024 * ct.GIB)
        assert result["affinity"] == ct.load_config()["cpus"]
        assert result["cwd"] == str(tmp_path)
        assert (result["njobs"], result["start"], result["end"]) == ("7", "20220101", "20220131")
    assert parents[0] == parents[1]
    assert results[0]["cgroup"] != results[1]["cgroup"]
    assert daemon.get_config()["unprotected_running"] == []
    assert daemon._reconcile_running_tasks() == []
    for task_id in tasks:
        daemon.cancel_task(task_id)
        assert daemon.get_task(task_id)["status"] == "cancelled"


def test_cancel_cleans_child_that_created_new_session_after_recovery(daemon, tmp_path):
    pid_file = tmp_path / "child.pid"
    source = f"import os,pathlib,time; os.setsid(); pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(120)"
    task_id = submit(daemon, tmp_path,
        f"import subprocess,sys,time\nsubprocess.Popen([sys.executable,'-c',{source!r}])\ntime.sleep(120)\n")
    wait_for(pid_file.exists)
    child = int(pid_file.read_text())
    task = daemon.get_task(task_id)
    assert ct.contains_pid(child)
    assert td._pid_belongs_to_script(task["pid"], task["script_path"])
    with td._active_pids_lock:
        process = td._active_processes.pop(task_id)
    assert td._task_process_alive(task_id, task["pid"], task["script_path"])
    daemon.cancel_task(task_id)
    process.wait(timeout=10)
    wait_for(lambda: not Path(f"/proc/{child}").exists()
             or Path(f"/proc/{child}/stat").read_text().split()[2] == "Z")
    assert daemon.get_task(task_id)["status"] == "cancelled"


def test_exit_status_and_stdout(daemon, tmp_path):
    task_id = submit(daemon, tmp_path, "import sys\nprint('验证输出',flush=True)\nsys.exit(7)\n")
    task = wait_for(lambda: (t if (t := daemon.get_task(task_id))["status"] == "failed" else None))
    assert "7" in task["error"]
    assert "验证输出" in Path(task["log_path"]).read_text()
    assert task["run_env"]["computeProtection"]["ready"]


def test_missing_protection_refuses_to_run(daemon, tmp_path, monkeypatch):
    marker = tmp_path / "must-not-run"
    monkeypatch.setattr(ct, "status", lambda: {"ready": False, "message": "故障验收"})
    task_id = submit(daemon, tmp_path, f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    task = wait_for(lambda: (t if (t := daemon.get_task(task_id))["status"] == "failed" else None))
    assert "计算保护未就绪" in task["error"]
    assert not marker.exists()


def test_memory_overflow_is_contained_in_tiny_scope():
    command, facts = ct.prepare_command([
        sys.executable, "-c", "print('开始局部内存验收',flush=True); a=bytearray(128*1024*1024)",
    ])
    index = command.index("--")
    command[index:index] = ["--property=MemoryMax=64M", "--property=MemorySwapMax=0"]
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=20)
        assert result.returncode != 0
        assert "开始局部内存验收" in result.stdout
        assert ct.status()["ready"]
    finally:
        ct.stop_scope(facts["scope"])


def test_thread_limit_blocks_new_threads_in_tiny_scope():
    code = (
        "import threading,time\nthreads=[]\n"
        "try:\n"
        " for i in range(100):\n"
        "  t=threading.Thread(target=time.sleep,args=(1,)); t.start(); threads.append(t)\n"
        "except RuntimeError:\n print('线程上限已阻止新增',len(threads),flush=True)\n"
        "else:\n raise AssertionError('未阻止新增线程')\n"
        "for t in threads: t.join()\n"
    )
    command, facts = ct.prepare_command([sys.executable, "-c", code])
    command.insert(command.index("--"), "--property=TasksMax=8")
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=15)
        assert result.returncode == 0, result.stderr
        assert "线程上限已阻止新增 7" in result.stdout
    finally:
        ct.stop_scope(facts["scope"])


def test_unavailable_cpu_controller_is_not_reported_as_hard_limit():
    facts = ct.status()
    if facts["actual"]["cpu.max"] == "unavailable":
        assert not facts["cpu_hard_limit"]
        assert facts["cpu_mode"] == "affinity"
        assert "待管理员开放" in facts["message"]


def test_stop_refuses_unrelated_service():
    with pytest.raises(ValueError):
        ct.stop_scope("factor-taskd.service")


def test_real_http_daemon_restart_recovers_success(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    db = tmp_path / "http-tasks.db"
    ready = tmp_path / "ready"
    finish = tmp_path / "finish"
    script = tmp_path / "计算服务重启验收.py"
    script.write_text(
        f"from pathlib import Path\nimport time\nPath({str(ready)!r}).touch()\n"
        f"while not Path({str(finish)!r}).exists(): time.sleep(0.05)\nprint('成功回执',flush=True)\n"
    )

    def request(path, payload=None):
        req = Request(base + path, data=None if payload is None else json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=5) as response:
            return json.load(response)

    def reachable():
        try:
            return request("/api/config")
        except OSError:
            return None

    def launch(log):
        process = subprocess.Popen(
            [sys.executable, td.__file__, "--port", str(port), "--db", str(db)],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        wait_for(reachable)
        return process

    process = None
    task_id = None
    with (tmp_path / "daemon.log").open("w") as log:
        try:
            process = launch(log)
            task_id = request("/api/tasks", {"script_path": str(script)})["id"]
            wait_for(ready.exists)
            assert request("/api/config")["unprotected_running"] == []
            process.terminate()
            process.wait(timeout=15)
            process = launch(log)
            task = request(f"/api/tasks/{task_id}")
            assert task["status"] == "running" and task["active"]
            finish.touch()
            # 再重启触发即时对账，验证成功回执而不是沿用旧版的「进程没了即失败」。
            result = Path(task["run_env"]["computeProtection"]["result_path"])
            wait_for(result.exists)
            wait_for(lambda: not Path(f"/proc/{task['pid']}").exists()
                     or Path(f"/proc/{task['pid']}/stat").read_text().split()[2] == "Z")
            process.terminate()
            process.wait(timeout=15)
            process = launch(log)
            task = request(f"/api/tasks/{task_id}")
            assert task["status"] == "completed", task
            assert not task["error"]
        finally:
            if process and process.poll() is None:
                if task_id:
                    task = request(f"/api/tasks/{task_id}")
                    if task["status"] == "running":
                        request(f"/api/tasks/{task_id}/cancel", {})
                    if task.get("log_path"):
                        Path(task["log_path"]).unlink(missing_ok=True)
                process.terminate()
                process.wait(timeout=15)
