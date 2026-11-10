"""只暂存本次注册改动：对共享文件取 HEAD 版本，套用 register.py 的同一组插入，写成暂存区版本。
工作区里别人的未提交改动不进暂存区。"""
import subprocess, sys
sys.path.insert(0, "/home/chenzongwei/rust_pyfunc/sandbox/genedp_port_20261110")
import register
R = "/home/chenzongwei/rust_pyfunc"
for path, fn in register.FILES.items():
    head = subprocess.run(["git", "-C", R, "show", f"HEAD:{path}"], capture_output=True, text=True, check=True).stdout
    new = fn(head)
    sha = subprocess.run(["git", "-C", R, "hash-object", "-w", "--stdin"], input=new, capture_output=True, text=True, check=True).stdout.strip()
    mode = subprocess.run(["git", "-C", R, "ls-files", "-s", path], capture_output=True, text=True, check=True).stdout.split()[0]
    subprocess.run(["git", "-C", R, "update-index", "--cacheinfo", f"{mode},{sha},{path}"], check=True)
    print("staged", path)
