#!/usr/bin/env python3
# MARKER: maturin-guard-v1
"""maturin 守卫包装器：防止在 rust_pyfunc 主项目目录裸跑无优化构建。

背景（2026-08-31 事故）：maturin develop 默认构建 dev profile（opt-level 0），
数值内核比 release 慢 10~50 倍；有人/agent 裸跑过一次后，513MB 的 debug 构建
覆盖了 release 安装，导致整个回测引擎变慢 40 倍。

规则：
  - 在 rust_pyfunc 主项目（[package] name = "rust_pyfunc"）目录内：
      * `maturin develop`（无 --release / --profile）→ 直接拦截报错（防止事故）
      * `maturin develop --release` → 放行但提示（建议 bash dev.sh 更快）
      * `maturin build` / 其他 → 放行（maturin build 默认就是 release profile）
  - 其他目录（sandbox、独立 crate 等）→ 放行；若为 dev_sandbox* 且 develop 无
    --release → 提示建议 --release 以便性能测量，但不拦截。
  - 原 maturin 由本脚本调用（同目录下 maturin.real 或 PATH 中任一 maturin）。
"""
import os
import re
import subprocess
import sys


def find_cargo_toml(start: str):
    d = os.path.abspath(start)
    while True:
        p = os.path.join(d, "Cargo.toml")
        if os.path.isfile(p):
            return p
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def package_name(cargo_toml: str):
    try:
        with open(cargo_toml, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None
    m = re.search(r"\[package\]\s*\n(?:[^\[]*?\n)*?\s*name\s*=\s*[\"']([^\"']+)[\"']", text)
    if m:
        return m.group(1)
    m2 = re.search(r"\[lib\]\s*\n(?:[^\[]*?\n)*?\s*name\s*=\s*[\"']([^\"']+)[\"']", text)
    return m2.group(1) if m2 else None


def real_maturin():
    # 找真正的 maturin：优先同目录 maturin.real，再 PATH 中任意 maturin.real /
    # maturin，最后回退到已知的安装位置（PATH 可能不含 conda env）。
    here = os.path.dirname(os.path.abspath(__file__))
    my_abs = os.path.abspath(__file__)

    def _ok(fp):
        try:
            return os.path.isfile(fp) and os.access(fp, os.X_OK) and os.path.abspath(fp) != my_abs
        except OSError:
            return False

    cand = os.path.join(here, "maturin.real")
    if _ok(cand):
        return cand
    for p in os.environ.get("PATH", "").split(os.pathsep):
        if not p:
            continue
        for fn in ("maturin.real", "maturin"):
            fp = os.path.join(p, fn)
            if _ok(fp):
                return fp
    for fp in (
        os.path.join(os.environ.get("CONDA_PREFIX", "/home/chenzongwei/.conda/envs/chenzongwei311"), "bin", "maturin.real"),
        "/home/chenzongwei/.conda/envs/chenzongwei311/bin/maturin.real",
        os.path.join(os.path.expanduser("~"), ".local", "bin", "maturin.real"),
    ):
        if _ok(fp):
            return fp
    return "maturin"


def main():
    args = sys.argv[1:]
    cwd = os.getcwd()
    toml = find_cargo_toml(cwd)
    name = package_name(toml) if toml else None
    is_main = name == "rust_pyfunc"
    is_sandbox = name is not None and name.startswith("dev_sandbox")

    sub = args[0] if args else ""
    has_profile_flag = any(a in ("--release", "--profile", "--release-fast") for a in args)
    is_develop = sub == "develop"

    if is_main and is_develop and not has_profile_flag:
        sys.stderr.write(
            "\n"
            "❌ maturin 守卫：禁止在 rust_pyfunc 主项目目录裸跑 `maturin develop`。\n"
            "   裸跑会安装 dev（opt-level 0）构建，回测性能将慢 10~50 倍\n"
            "   （2026-08-31 曾因此把 513MB debug 构建覆盖到 release 安装上）。\n"
            "   请改用：\n"
            "     bash dev.sh     # 默认 release-fast（无 LTO，增量，性能≈release，日常构建）\n"
            "     ./alter.sh      # 默认 release-fast + 部署 worker 二进制\n"
            "     ./alter.sh release   # fat LTO 全量（~10 分钟，仅发布）\n"
            "   若确需直接 maturin：maturin develop --profile release-fast\n"
            "\n"
        )
        sys.exit(1)

    if is_main and is_develop and has_profile_flag:
        sys.stderr.write(
            "ℹ️  maturin 守卫：在本目录直接 maturin 不会被拦截，"
            "但更推荐 `bash dev.sh`（release-fast，增量更快）或 `./alter.sh`。\n"
        )

    if is_sandbox and is_develop and not has_profile_flag:
        sys.stderr.write(
            "ℹ️  maturin 守卫：sandbox crate 开发建议加 --release，"
            "否则测不出真实性能（dev 无优化，参考：bash build.sh --release）。\n"
        )

    os.execv(real_maturin(), [real_maturin()] + args)


if __name__ == "__main__":
    main()
