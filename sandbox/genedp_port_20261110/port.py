"""把四个沙箱第三版的计算代码搬进正式库 src/genedp_*/（机械部分：复制 + 改模块引用 + 删私有路径解析）。
入口函数（mod.rs 末尾的 main）另行手工改写。"""
import re, shutil, pathlib
R = pathlib.Path("/home/chenzongwei/rust_pyfunc")

def drop_fn(text, name):
    """删掉顶层 fn name(...) {...}（按大括号配对）"""
    m = re.search(r"\n(pub )?fn " + name + r"\b", text)
    if not m:
        return text, False
    i = text.index("{", m.end())
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{": depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[:m.start()] + "\n" + text[j + 1:], True
    raise RuntimeError(name)

def reader_fix(text):
    text, ok = drop_fn(text, "resolve_stock_path")
    if ok:
        return "use crate::genedp_common::resolve_stock_path;\n" + text
    text, ok = drop_fn(text, "resolve")  # 方向 3 的读取器把这个函数叫 resolve
    assert ok
    return "use crate::genedp_common::resolve_stock_path as resolve;\n" + text

PLAN = {
    "genedp_day_vs_day_align": {
        "src/v3/main.rs": "mod.rs", "src/v3/align.rs": "align.rs", "src/v3/letters.rs": "letters.rs",
        "src/fast_csv_reader.rs": "reader.rs", "_dir": "sandbox_align_day_vs_day"},
    "genedp_self_repeat_align": {
        "src/v3/main.rs": "mod.rs", "src/align.rs": "align.rs", "src/fast_csv_reader.rs": "reader.rs",
        "_dir": "sandbox_align_self_repeat"},
    "genedp_mirror_align": {
        "src/v3/main.rs": "mod.rs", "src/v3/batch.rs": "batch.rs", "src/v3/dp.rs": "dp.rs", "src/v3/full.rs": "full.rs",
        "src/v3/prep.rs": "prep.rs", "src/v3/reader.rs": "reader.rs", "_dir": "sandbox_align_mirror"},
    "genedp_buysell_fold": {
        "src/align_fold_v3.rs": "mod.rs", "src/fold.rs": "fold.rs", "src/v3_feats.rs": "v3_feats.rs",
        "src/v3_kernels.rs": "v3_kernels.rs", "src/v3_letters.rs": "v3_letters.rs", "src/fast_csv_reader.rs": "reader.rs",
        "_dir": "sandbox_align_fold"},
}
for mod, files in PLAN.items():
    src_dir = R / files.pop("_dir")
    out = R / "src" / mod
    out.mkdir(exist_ok=True)
    for s, d in files.items():
        t = (src_dir / s).read_text()
        if d == "reader.rs":
            t = reader_fix(t)
        if d != "mod.rs":
            # 子模块里的 crate:: 指向沙箱 crate 根，搬进来后改成指向本目录的兄弟模块
            t = t.replace("crate::fast_csv_reader", "super::reader")
            t = t.replace("crate::genedp_common", "@@COMMON@@")
            t = re.sub(r"crate::", "super::", t)
            t = t.replace("@@COMMON@@", "crate::genedp_common")
            # 测试模块里 use super::xxx 若指兄弟模块，需要再上一层
            t = re.sub(r"(mod tests \{[\s\S]*?)use super::(letters|align|dp|fold|v3_kernels)::", r"\1use super::super::\2::", t)
        (out / d).write_text(t)
        print(mod, d, len(t.splitlines()))
