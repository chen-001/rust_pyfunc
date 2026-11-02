"""Build the review workbook from measured logs; raw arrays remain outside Git."""
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as E

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/relaynet_perf_20261102'
SKILL = Path('/home/chenzongwei/.agents/skills/xlsx')
NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG = 'http://schemas.openxmlformats.org/package/2006/relationships'
CT = 'http://schemas.openxmlformats.org/package/2006/content-types'


def main():
    sheets = {'验收说明': [
        ['事项', '说明'],
        ['目标', '50线程，完整市场一天，包含Python列表返回，用时严格小于120秒'],
        ['测试日期', '主验收日为20241231；5107只股票，477张图，13574个因子'],
        ['旧版对比限制', '旧版及早期开发版运行时有其他实验并行，接口为NumPy；不可将耗时比直接解释为严格同条件加速比'],
        ['正式候选计时', '从exact_v4起，本任务各整日性能测试串行运行，使用Python列表接口；未独占整台服务器'],
        ['计算设备', 'AMD EPYC 9754，双路；50个Rayon计算线程，OPENBLAS_NUM_THREADS=1；release-fast，target-cpu=native'],
        ['缓存口径', '每次完整日测试均启动新进程，重新读取并计算；没有复用上次因子结果。操作系统文件缓存未清除'],
        ['v5运行参数补记', 'exact_v5_direction启用RELAYNET_BINARY_DIRECTION=1；当时脚本尚未将此环境变量写入元数据，启动命令可核对'],
        ['CPU绑定试验', 'exact_v4_pinned1固定CPU 0-49，同时图批次改为1；其他试验未固定CPU，不单独归因于绑定效果'],
        ['一致性分层', '纯提速与旧版比较；错误修正及全源计算另用独立数学对照；后续提速与修正后版本比较'],
        ['生产状态', '当前仅沙箱候选，原安装与主项目源码未替换。是否迁移等待用户意见'],
    ]}
    perf = [['版本/运行标签', '日期', '线程数', '股票数', '因子列数', '完整调用秒数',
             '目标秒数', '距目标秒数', '是否低于目标', '返回接口', '峰值内存GiB', '运行参数', '原始记录']]
    final_rows = {}
    for p in sorted(OUT.glob('*.json')):
        m = json.loads(p.read_text())
        if not isinstance(m, dict) or not all(k in m for k in ('variant', 'seconds', 'shape')):
            continue
        row = len(perf) + 1
        perf.append([p.stem, m['date'], m['threads'], *m['shape'], m['seconds'], 120,
                     f'=F{row}-G{row}', f'=F{row}<G{row}', m.get('interface', 'numpy'),
                     m['peak_rss_gib'], json.dumps(m.get('runtime_environment', {})), str(p)])
        if m['variant'] == 'candidate': final_rows.setdefault(m['date'], []).append(row)
    sheets['性能实测'] = perf
    for date, rows in sorted(final_rows.items()):
        refs = ','.join(f"'性能实测'!F{r}" for r in rows)
        sheets['验收说明'].extend([
            [f'{date} 最终候选最快完整调用/秒', f'=MIN({refs})'],
            [f'{date} 最终候选最慢完整调用/秒', f'=MAX({refs})'],
            [f'{date} 是否所有实测均低于120秒', f'=IF(MAX({refs})<120,"已达到","未达到")'],
        ])
    checks = [['检查项', '实测结果', '证据/说明']]
    for filename in ('baseline_vs_compat.json', 'exact_v2_vs_v3.json', 'final_equivalence.json',
                     'direction_equivalence.json', 'affinity_equivalence.json', 'combined_equivalence.json',
                     'candidate_equivalence.json', 'final_determinism.json'):
        p = OUT / filename
        if p.exists():
            m = json.loads(p.read_text())
            checks.append([filename, json.dumps(m, ensure_ascii=False), str(p)])
    checks.extend([
        ['矩阵计算', '有限f64逐位一致；NaN位置一致；独立Python公式通过', 'tests/test_relaynet_matrix_20261102.py'],
        ['原始数据预处理', '6个真实股票日的全部实际使用序列逐位一致', '删除的是未进入任何注册因子的中间量'],
        ['全源最短路', '41组独立穷举/Floyd对照，120节点树，跨1/8/50线程一致', 'tests/test_relaynet_exact_paths_20261102.py'],
        ['图统计', 'Python独立对照、随机图、并列权重、断开分量与空图', 'tests/test_relaynet_graph_sandbox.py'],
        ['图调度', '1/2/4/8张图批次输出逐字节一致', '同一个50线程池；注册表和列顺序未变'],
        ['计算口径', '使用全部股票、全部源节点，无新增抽样或近似', '原有图稀疏化定义及PageRank/特征向量50次迭代定义保留'],
        ['部署状态', '仅沙箱候选；主项目未迁移', '迁移及alter.sh按AGENTS.md等待用户意见'],
    ])
    sheets['一致性验收'] = checks
    sheets['错误修正'] = [
        ['位置', '旧问题', '修正后', '预期结果影响'],
        ['相似度缓存', '缓存键遗漏算法类型，绝对收益/符号/领先滞后误用收益相关', '键包含序列、时段和算法类型', '相关家族及其下游图发生变化'],
        ['最强邻居筛选', '最大堆替换最大值，未保留真正前k名', '精确前k名，等值按节点编号确定顺序', '网络连接及所有依赖统计发生变化'],
        ['PageRank', '逐节点初始化覆盖已经累加的贡献', '整轮统一初始化，按源顺序累加贡献', 'PageRank及派生列变化'],
        ['生成树', 'Prim只处理首个连通分量', '完整最大生成森林', '存在多个有效分量时变化'],
        ['距离经邻居到市场', '把相似度直接当边长加到距离', '统一使用max(1-相似度,1e-9)', 'dist_mkt_top10变化'],
        ['源节点抽样', '约1/25起点，其他节点用邻居/全局平均填充', '用户已确认：全部源节点逐一计算', '距离及中介中心性变化'],
        ['路径并列与缩放', '近似相等的路径被计为并列；孤立点令全源缩放偏大', '严格路径比较，全源仅保留无向图除2', '极近但不相等的路径和相关中心性变化'],
    ]
    sheets['优化项'] = [
        ['方法', '是否采用', '准确性依据', '局部性能证据'],
        ['Pearson跨股票对SIMD', '采用', '保持每对的累加顺序，不使用FMA重排', '矩阵测试约1.5-1.9倍'],
        ['符号相似度位图', '采用', '整数计数逐项等价', '矩阵测试约4-5倍'],
        ['最短路/加权中介复用', '采用', '同一完整最短路遍历共享结果', '独立穷举路径校验'],
        ['最短路径前驱缓存', '采用', '只缓存实际最短路前驱，保持回传顺序', '实图中位0.727→0.454秒'],
        ['加权路径AVX2四边筛选', '通过局部验证', '只跳过严格不可能改善/并列的边，保留原顺序及标量回退', '真实图0.4621→0.4066秒；AVX512无收益，弃用'],
        ['森林专用公式', '采用', '树上唯一路径；分支节点对计数精确', '独立120节点树对照'],
        ['邻域位图与重复计算复用', '采用', '整数集合操作；完整邻接表核对缓存', '完整图51统计对照'],
        ['并行PageRank/有向两步可达', '采用', '每个节点保留原贡献次序', '实图局部统计1.957→0.092秒'],
        ['有向正反距离复用', '条件采用', '可证明精确累加时转置复用，其余原算法回退', '21组含回退分支逐位一致'],
        ['二元中介AVX512', '拒绝启用', '结果逐位一致，但实测更慢', '标量0.613秒；SIMD0.699秒'],
        ['二元路径按层选择扫描方向', '通过局部与完整日验证', '严格复现BFS队列及路径计数累加次序；不满足条件时回退', '实图局部中位0.639→0.481秒；完整日218.6→207.0秒'],
        ['二元最短路前驱缓存及不变量复用', '通过局部验证', '保留反向队列顺序及每个目标的加法顺序', '方向优化后局部0.514→0.456秒'],
        ['紧凑前驱AVX512', '拒绝启用', '逐位一致，但没有速度收益', '标量0.452秒；SIMD0.469秒'],
        ['预存边长/全局距离界/距离分位数多选', '拒绝启用', '全部逐位一致，完整真实图性能没有改善', '12种组合详见weighted_experiments记录'],
        ['基础统计并行与三角形统计复用', '通过局部验证', '原来的乘法及求和顺序不变，精确选择分位数', '真实图51项统计1.557→1.382秒'],
        ['候选边缓存', '通过一致性验证', '单次调用内，以不可变矩阵、k及过滤条件为键；保留所有者防地址复用', '整日测试验证实际收益'],
        ['输出按股票连续写入', '通过索引审查', '只改变写入顺序，无数值运算变动', '整日逐位核对'],
        ['未使用的预处理统计', '删除', '不被任何注册因子引用，真实数据序列对照', '避免重复排序、统计和内存分配'],
    ]
    if final_rows:
        for row in sheets['优化项'][1:]:
            if row[1].startswith('通过'):
                row[1] = '采用；整日验证通过'
    changes = OUT/'old_vs_candidate_columns.json'
    if changes.exists():
        rows = [['因子名', '有限值位变化股票数', '缺失/无穷位置变化股票数', '受影响股票数', '最大绝对差异']]
        for name, finite, missing, maximum in json.loads(changes.read_text())['columns']:
            r = len(rows) + 1
            rows.append([name, finite, missing, f'=B{r}+C{r}', maximum])
        sheets['旧版结果变化'] = rows
        sheets['验收说明'].append(['旧版结果发生变化的因子列数', f'=COUNTIF(\'旧版结果变化\'!D2:D{len(rows)},">0")'])
        sheets['验收说明'].append(['逐列差异口径', '旧版对照最终候选；原始逐值审计统计见old_vs_candidate_columns.json。变化包含用户授权移除抽样及错误修正，不能单独归因为某一项修正'])
    graphs = [['运行日志', '图名称', '相似度秒数', '图统计秒数', '该图总秒数', '备注']]
    for p in sorted(OUT.glob('*20241231.log')):
        for tag, sim, graph, total in re.findall(r'\[rlg-time\] (\S+) sim_sec=([\d.]+) graph_sec=([\d.]+) total_sec=([\d.]+)', p.read_text()):
            graphs.append([p.stem, tag, float(sim), float(graph), float(total), '并行批次的单图时间不可相加推算整日墙钟时间'])
    sheets['逐图计时'] = graphs

    work = OUT / 'xlsx_work'
    shutil.copytree(SKILL / 'templates/minimal_xlsx', work, dirs_exist_ok=True)
    q = lambda x: f'{{{NS}}}{x}'
    style = E.parse(work/'xl/styles.xml')
    xfs = style.getroot().find(q('cellXfs'))
    # Wrap long audit explanations; keep measurements readable to milliseconds.
    for xf in xfs:
        alignment = xf.find(q('alignment'))
        if alignment is None: alignment = E.SubElement(xf, q('alignment'))
        alignment.set('wrapText', '1'); alignment.set('vertical', 'top')
    xfs[1].set('numFmtId', '2'); xfs[2].set('numFmtId', '2')
    style.write(work/'xl/styles.xml', encoding='utf-8', xml_declaration=True)
    strings, counts = {}, 0
    wb = E.parse(work / 'xl/workbook.xml'); ws = wb.getroot().find(q('sheets')); ws.clear()
    rels = E.parse(work / 'xl/_rels/workbook.xml.rels')
    for child in list(rels.getroot()):
        if child.attrib['Type'].endswith('/worksheet'): rels.getroot().remove(child)
    types = E.parse(work / '[Content_Types].xml')
    for child in list(types.getroot()):
        if child.attrib.get('PartName', '').startswith('/xl/worksheets/'): types.getroot().remove(child)
    for sid, (title, rows) in enumerate(sheets.items(), 1):
        rid = 'rId1' if sid == 1 else f'rId{sid+2}'
        E.SubElement(ws, q('sheet'), {'name': title, 'sheetId': str(sid), f'{{{REL}}}id': rid})
        E.SubElement(rels.getroot(), f'{{{PKG}}}Relationship', {'Id': rid, 'Type': REL+'/worksheet', 'Target': f'worksheets/sheet{sid}.xml'})
        E.SubElement(types.getroot(), f'{{{CT}}}Override', {'PartName': f'/xl/worksheets/sheet{sid}.xml', 'ContentType': 'application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml'})
        sheet = E.Element(q('worksheet')); views = E.SubElement(sheet, q('sheetViews'))
        view = E.SubElement(views, q('sheetView'), {'workbookViewId':'0'})
        E.SubElement(view, q('pane'), {'ySplit':'1','topLeftCell':'A2','activePane':'bottomLeft','state':'frozen'})
        cols = E.SubElement(sheet, q('cols'))
        for col in range(len(rows[0])):
            width = 38 if col == 0 else (95 if len(rows[0]) == 2 else 36)
            E.SubElement(cols, q('col'), {'min':str(col+1),'max':str(col+1),'width':str(width),'customWidth':'1'})
        data = E.SubElement(sheet, q('sheetData'))
        for r, values in enumerate(rows, 1):
            row = E.SubElement(data, q('row'), {'r':str(r)})
            row.set('ht', str(32 if r == 1 else (64 if title not in ('逐图计时', '旧版结果变化') else 30)))
            row.set('customHeight', '1')
            for c, value in enumerate(values):
                cell = E.SubElement(row, q('c'), {'r':f'{chr(65+c)}{r}', 's':'4' if r == 1 else '0'})
                if isinstance(value, str) and value.startswith('='):
                    cell.set('s','3' if '!' in value else '2')
                    E.SubElement(cell,q('f')).text=value[1:]; E.SubElement(cell,q('v'))
                elif isinstance(value, (float,int)):
                    cell.set('s','1' if isinstance(value, float) else '0')
                    E.SubElement(cell,q('v')).text=str(value)
                else:
                    value=str(value); counts += 1
                    if value not in strings: strings[value]=len(strings)
                    cell.set('t','s'); E.SubElement(cell,q('v')).text=str(strings[value])
        E.SubElement(sheet,q('autoFilter'),{'ref':f'A1:{chr(64+len(rows[0]))}{len(rows)}'})
        E.ElementTree(sheet).write(work / f'xl/worksheets/sheet{sid}.xml',encoding='utf-8',xml_declaration=True)
    sst=E.Element(q('sst'),{'count':str(counts),'uniqueCount':str(len(strings))})
    for value in strings: E.SubElement(E.SubElement(sst,q('si')),q('t')).text=value
    E.ElementTree(sst).write(work/'xl/sharedStrings.xml',encoding='utf-8',xml_declaration=True)
    for tree, path in ((wb,'xl/workbook.xml'),(rels,'xl/_rels/workbook.xml.rels'),(types,'[Content_Types].xml')):
        tree.write(work/path,encoding='utf-8',xml_declaration=True)
    import sys
    result=OUT/'py_relaynet_优化验收.xlsx'
    subprocess.run([sys.executable,str(SKILL/'scripts/xlsx_pack.py'),str(work),str(result)],check=True)
    subprocess.run([sys.executable,str(SKILL/'scripts/formula_check.py'),str(result)],check=True)
    # Keep input/output directories distinct: converting an xlsx onto itself
    # can make LibreOffice exit 0 without producing any recalculated workbook.
    with tempfile.TemporaryDirectory(prefix='relaynet_xlsx_') as temporary:
        temp = Path(temporary); converted = temp/'converted'; converted.mkdir()
        source = temp/'audit.xlsx'; shutil.copyfile(result, source)
        subprocess.run(['libreoffice', f'-env:UserInstallation={(temp/"profile").as_uri()}',
                        '--headless', '--convert-to', 'xlsx', '--outdir', str(converted),
                        str(source)], check=True, timeout=60)
        recalculated = converted/source.name
        with zipfile.ZipFile(recalculated) as archive:
            for name in archive.namelist():
                if name.startswith('xl/worksheets/sheet') and name.endswith('.xml'):
                    for cell in E.fromstring(archive.read(name)).iter(q('c')):
                        if cell.find(q('f')) is not None:
                            cached = cell.find(q('v'))
                            assert cached is not None and cached.text is not None, (name,cell.attrib)
        subprocess.run([sys.executable,str(SKILL/'scripts/formula_check.py'),str(recalculated)],check=True)
        shutil.copyfile(recalculated, result)
    print(result)


if __name__ == '__main__': main()
