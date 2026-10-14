#!/bin/bash
# 管理员只需执行一次；只开放 chenzongwei 的 CPU 控制，不重启其会话或现有任务。
# sudo bash /home/chenzongwei/rust_pyfunc/python/rust_pyfunc/compute_terminal_enable_cpu.sh
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then
    echo '需要管理员权限，请使用 sudo bash 执行此脚本。' >&2
    exit 1
fi
account=chenzongwei
account_uid=$(id -u "$account")
account_home=$(getent passwd "$account" | cut -d: -f6)
online_cpus=$(cat /sys/devices/system/cpu/online)
dropin="/etc/systemd/system/user@${account_uid}.service.d"
mkdir -p "$dropin"
cat > "$dropin/90-compute-terminal.conf" <<'EOF'
[Service]
Delegate=yes
CPUAccounting=yes
EOF
systemctl daemon-reload
# 空间划分只限制 compute.slice；user@ 允许全机 CPU，其他软件继续正常使用。
# set-property 通过 systemd 更新运行中的上级分组，避免直接写内核文件与 systemd 冲突。
systemctl set-property "user@${account_uid}.service" CPUAccounting=yes "AllowedCPUs=$online_cpus"
runuser -u "$account" -- env \
    "XDG_RUNTIME_DIR=/run/user/$account_uid" \
    "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$account_uid/bus" \
    "$account_home/.local/bin/compute-terminal" install
runuser -u "$account" -- env \
    "XDG_RUNTIME_DIR=/run/user/$account_uid" \
    "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$account_uid/bus" \
    "$account_home/.conda/envs/chenzongwei311/bin/python" -c \
    'import json,subprocess; s=json.loads(subprocess.check_output(["/home/chenzongwei/.local/bin/compute-terminal","status","--json"])); assert s["ready"] and s["cpu_hard_limit"], "CPU 控制尚未生效，请管理员检查上级资源分组；未重启任何会话"; print("CPU 总额度已核验生效。")'
