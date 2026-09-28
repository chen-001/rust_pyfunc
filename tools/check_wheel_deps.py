#!/usr/bin/env python3
"""发布前检查：安装包不能声明任何运行时依赖。

pip install rust_pyfunc 的人就算漏掉 --no-deps，pip 也不会装别的东西——前提是
安装包的 METADATA / PKG-INFO 里没有 Requires-Dist。这里发现一行就退出码 1，
把上传这一步挡掉。用法：python tools/check_wheel_deps.py [安装包或目录 ...]
默认检查 dist 目录。
"""
from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path


def wheel_deps(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        metadata = next(name for name in archive.namelist() if name.endswith('.dist-info/METADATA'))
        text = archive.read(metadata).decode()
    return [line for line in text.splitlines() if line.startswith('Requires-Dist:')]


def sdist_deps(path: Path) -> list[str]:
    with tarfile.open(path) as archive:
        member = next(item for item in archive.getmembers() if item.name.endswith('PKG-INFO'))
        text = archive.extractfile(member).read().decode()
    return [line for line in text.splitlines() if line.startswith('Requires-Dist:')]


def dependencies(path: Path) -> list[str]:
    return wheel_deps(path) if path.suffix == '.whl' else sdist_deps(path)


def main(argv: list[str]) -> int:
    given = [Path(item) for item in argv] or [Path('dist')]
    targets = sorted({p for item in given for p in ([item] if item.is_file() else item.iterdir())
                      if p.suffix in ('.whl', '.gz')})
    found = [f'{path}: {line}' for path in targets for line in dependencies(path)]
    if found:
        print('安装包声明了运行时依赖，去掉后再发布：')
        print('\n'.join(found))
        return 1
    print(f'{len(targets)} 个安装包都没有声明依赖')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
