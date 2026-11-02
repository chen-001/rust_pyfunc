"""Create a production-shaped relaynet candidate without touching main src or fast.

Only accepted exact paths are retained. Fixed defaults match the validated
experimental configuration; all reference implementations stay in fast.
"""
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
source = (HERE / 'src/relaynet_fast.rs').read_text()


def end_block(text, brace):
    depth = 1
    end = brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return end


def span(text, name):
    match = re.search(r'(?m)^(?:pub )?(?:unsafe )?fn ' + name + r'(?:<|\()', text)
    assert match, name
    return match.start(), end_block(text, text.index('{', match.start()))


def function(name, text=source):
    a, b = span(text, name)
    return text[a:b]


def replace_function(text, name, replacement=''):
    a, b = span(text, name)
    return text[:a] + replacement + text[b:]


def branch_bounds(text, start):
    brace = text.index('{', start)
    end = end_block(text, brace)
    following = end + len(text[end:]) - len(text[end:].lstrip())
    if text.startswith('else', following):
        other = following + 4
        other += len(text[other:]) - len(text[other:].lstrip())
        if text.startswith('if ', other):
            _, _, final, _ = branch_bounds(text, other)
            return brace, end, final, (other, final)
        final = end_block(text, other)
        return brace, end, final, (other + 1, final - 1)
    return brace, end, end, None


def choose(text, condition, take=True):
    marker = 'if ' + condition
    while marker in text:
        start = text.index(marker)
        brace, end, final, other = branch_bounds(text, start)
        body = text[brace + 1:end - 1] if take else (text[other[0]:other[1]] if other else '')
        if body:
            body = '{' + body + '}'
        text = text[:start] + body + text[final:]
    return text


def region(start, stop):
    return source[source.index(start):source.index(stop)]


# Keep the production module's public API and computation, excluding all probes.
text = source[:source.index('/// Harness-only exact comparison')]
text = text.replace('//!   3. 六种相似度矩阵：rd 收益相关 / ra 绝对收益相关 / rs 符号一致 /\n//!      rt 市场上行段相关 / ro 主动买压相关 / rl 领先-滞后(有向)\n//!   4. ~50 张图（阈值 / kNN 并集与互近邻 / 最大生成树 / 行业内 / 有向）\n//!   5. 每张图 22 个节点统计量 + MST 附加 5 个 + 行业 4 个 + 有向 12 个',
'''//!   3. 按因子族与时段构造收益、符号、成交、盘口和领先滞后相似度矩阵
//!   4. 按 GRAPH_SPECS 建立 477 张阈值、kNN、最大生成森林、行业内和有向图
//!   5. 完整无向图 51 项、缩减图 9 项、MST 附加 5 项、有向图 12 项
//!      共输出 13,574 列；距离与中介遍历全部源节点
//!      固定源块和加法顺序保证计算确定性''')
text = text.replace('上午240格) 接着下午平移后 11:30→13:27(237格), 共 474',
                    '上午240格) 接着下午平移后 11:30→13:27(234格), 共 474')
text = re.sub(r'/// day=全量统计.*?(?=fn emit_fam)',
              '/// 按既有因子注册表选择完整/缩减统计包，不改变输出列定义。\n', text, flags=re.S)
text = choose(text, '!RELAY_GRAPH_CORRECT_FIXES', False)
text = replace_function(text, 'n_src_of')
text = text.replace('/// 完整版本使用全部源节点；兼容回归分支保留旧源数。\n', '')
text = re.sub(r'n_src_of\(([^()]+)\)', r'\1', text)
text = replace_function(text, 'betweenness_weighted')  # full stats already reuse exact.bet
text = replace_function(text, 'clustering_w')         # fusion retains the same clustering loop
text = replace_function(text, 'exact_paths', '''fn exact_paths(g: &UGraph, weighted: bool) -> ExactPaths {
    exact_paths_screen4(g, weighted)
}''')
text = replace_function(text, 'betweenness_bin', '''fn betweenness_bin(g: &UGraph, _nsrc: usize) -> Vec<f64> {
    binary_direction_run(g)
}''')
text = text.replace('/// 全节点加权中介（Dijkstra 版 Brandes）。\n', '')
text = text.replace('fn dist_stats(g: &UGraph, nsrc: usize)', 'fn dist_stats(g: &UGraph)')
text = text.replace('fn dist_stats_q(g: &UGraph, _nsrc: usize)', 'fn dist_stats_q(g: &UGraph)')
text = text.replace('fn betweenness_bin(g: &UGraph, _nsrc: usize)', 'fn betweenness_bin(g: &UGraph)')
text = text.replace('dist_stats_q(g, nsrc)', 'dist_stats_q(g)')
for call in ('dist_stats', 'betweenness_bin'):
    text = re.sub(r'\b' + call + r'\((g|&g_ind), n\)', call + r'(\1)', text)

# Always compute the accepted exact parallel basics and fused triangle values.
text = choose(text, 'std::env::var_os("RELAYNET_BASIC_PARALLEL").is_some()', True)
text = choose(text, '!fuse_triangle', False)
text = choose(text, 'fuse_triangle', True)
text = text.replace('clustering_triangle_pair_exact(g, select_triangle)', 'clustering_triangle_pair_exact(g)')
text = re.sub(r'^    let (?:select_triangle|fuse_triangle) = .*;\n', '', text, flags=re.M)
text = text.replace('            let mut triw: Vec<f64> = Vec::new();\n', '')
text = text.replace('.map_init(|| vec![0.0f32; n], |neighbor_weights, i| {', '.map(|i| {')
text = text.replace('            for &(v, w) in a { neighbor_weights[v as usize] = w; }\n', '')
text = text.replace('            for &(v, sij) in a {', '            for &(v, _) in a {')

# Fixed row-oriented output and graph batches use the existing bounded pool.
text = choose(text, 'std::env::var_os("RELAYNET_ROW_OUTPUT").is_some()', True)
text = choose(text, 'graph_batch == 1', False)
text = re.sub(r'    let graph_batch = std::env::var\("RLG_GRAPH_BATCH"\).*?\.clamp\(1, 8\);',
              '    const GRAPH_BATCH: usize = 4;', text, flags=re.S)
text = text.replace('GRAPH_SPECS.chunks(graph_batch)', 'GRAPH_SPECS.chunks(GRAPH_BATCH)')
text = text.replace('''            let mut sim_seconds = Vec::with_capacity(specs.len());
            let matrices: Vec<Rc<Array2<f64>>> = specs.iter().map(|sp| {
                let clock = std::time::Instant::now();
                let matrix = bank.sim(sp.fam, sp.sess);
                sim_seconds.push(clock.elapsed().as_secs_f64());
                matrix
            }).collect();''', '''            let matrices: Vec<Rc<Array2<f64>>> = specs.iter()
                .map(|sp| bank.sim(sp.fam, sp.sess)).collect();''')
text = choose(text, 'output.cached', False)
text = choose(text, 'let Some((edges, nodes)) = output.size', False)

# Retain only the corrected top-k builders.
builders = function('relay_candidate_bits') + '\n\n' + function('relay_graph_from_bits')
builders = choose(builders, 'RELAY_GRAPH_CORRECT_FIXES', True)
text += '\n\n' + builders
for name in ('dir_costs_are_reversible_exact', 'dir_harmonic_pair_exact',
             'dir_two_hop_exact', 'dir_pagerank_pair_exact'):
    text += '\n\n' + function(name)

# Exact batch cache, now comparing adjacency weight bits even on hash collisions.
batch = region('type RelayGraphStatsCache', '// GRAPH_BATCH_HELPERS_END')
batch = batch.replace('''    columns: Vec<Vec<f64>>, seconds: f64,
    size: Option<(usize, usize)>, cached: bool,''', '    columns: Vec<Vec<f64>>,')
batch = batch.replace('Vec<(RelayGraphWork, u64, f64)>', 'Vec<(RelayGraphWork, u64)>')
batch = batch.replace('let clock=std::time::Instant::now(); let sim=sims[i];', 'let sim=sims[i];')
batch = batch.replace('(graph,hash,clock.elapsed().as_secs_f64())', '(graph,hash)')
batch = batch.replace('Vec<Option<(Vec<Vec<f64>>,f64)>>', 'Vec<Option<Vec<Vec<f64>>>>')
batch = batch.replace('        let clock=std::time::Instant::now();\n', '')
batch = batch.replace('Some((cols,clock.elapsed().as_secs_f64()))', 'Some(cols)')
batch = batch.replace('    let mut seconds: Vec<f64>=prepared.iter().map(|(_,_,time)|*time).collect();\n', '')
batch = batch.replace('Some((cols,time))=computed[i].take()', 'Some(cols)=computed[i].take()')
batch = batch.replace('            seconds[i]+=time;\n', '')
batch = batch.replace('let clock=std::time::Instant::now(); let mut cols=cols.expect("computed graph result");',
                      'let mut cols=cols.expect("computed graph result");')
old = '''        let size=match &prepared[i].0 {
            RelayGraphWork::Undirected(g)=> {
                if matches!(specs[i].kind,Kind::Mst) {
                    let extra=mst_extra(g,mkt,if specs[i].full {Some(cols[9].as_slice())} else {None});
                    cols.extend(extra);
                }
                let mut edges=0;let mut nodes=0;
                for u in 0..g.n {if live[u] && !g.adj[u].is_empty(){nodes+=1;edges+=g.adj[u].len();}}
                Some((edges/2,nodes))
            },
            _=>None,
        };
        RelayGraphOutput { columns:cols, seconds:seconds[i]+clock.elapsed().as_secs_f64(),size,cached:cached[i] }'''
assert old in batch
batch = batch.replace(old, '''        if let RelayGraphWork::Undirected(g)=&prepared[i].0 {
            if matches!(specs[i].kind,Kind::Mst) {
                let extra=mst_extra(g,mkt,if specs[i].full {Some(cols[9].as_slice())} else {None});
                cols.extend(extra);
            }
        }
        RelayGraphOutput { columns:cols }''')
text += '\n\n' + batch + '\n\n' + function('relay_graph_adjacency_equal')

# Binary Brandes: all-source push/pull, exact final-frontier stop, compact parents.
binary = region('struct BinaryDirectionGraph', 'pub fn binary_direction_state_probe')
binary = binary[:binary.index('fn binary_direction_run')]
binary = binary.replace(',record_parents:bool,hoist_sigma:bool', '')
binary = binary.replace(',record_parents:false,hoist_sigma:false', '')
binary = binary.replace(',early_stop:bool,force_pull:bool', '')
binary = binary.replace('early_stop && ', '')
binary = binary.replace('(force_pull || frontier_edges>unvisited_edges.saturating_mul(4))',
                        'frontier_edges>unvisited_edges.saturating_mul(4)')
binary = choose(binary, 'self.record_parents', True)
binary = binary.replace('if self.hoist_sigma{sigma_u}else{self.sigma[u]}', 'sigma_u')
binary += '''
fn binary_direction_run(g:&UGraph)->Vec<f64> {
    if let Some(bet)=forest_bet(g){return bet;}
    let n=g.n;
    let graph=BinaryDirectionGraph::new(g);
    const BLOCK:usize=16;
    let parts:Vec<Vec<f64>>=(0..n.div_ceil(BLOCK)).into_par_iter().map(|block|{
        let mut scratch=BinaryDirectionScratch::new(n);let mut bet=vec![0.0;n];
        for src in block*BLOCK..((block+1)*BLOCK).min(n) {
            if graph.offsets[src]==graph.offsets[src+1]{continue;}
            scratch.bfs(&graph,src);
            for &u in scratch.queue.iter().rev() {
                // Each predecessor is a distinct target. Preserve the reverse
                // queue order of additions into each target's delta exactly.
                let mut link=scratch.pred_head[u];
                let sigma_u=scratch.sigma[u];let factor=1.0+scratch.delta[u];
                while link!=u32::MAX {
                    let v=scratch.pred_from[link as usize] as usize;
                    scratch.delta[v]+=(scratch.sigma[v]/sigma_u)*factor;
                    link=scratch.pred_next[link as usize];
                }
                if u!=src {bet[u]+=scratch.delta[u];}
            }
        }
        bet
    }).collect();
    let mut bet=vec![0.0;n];
    for part in parts{for(a,b)in bet.iter_mut().zip(part){*a+=b;}}
    for b in &mut bet{*b*=0.5;}bet
}
'''
text += '\n\n' + binary

for name in ('undir_basic_parallel_exact', 'triangle_p75_select_exact'):
    text += '\n\n' + function(name)
triangle = function('clustering_triangle_pair_exact')
triangle = triangle.replace('(g: &UGraph, select: bool)', '(g: &UGraph)')
triangle = choose(triangle, 'select', True)
text += '\n\n' + triangle

cache = region("struct RelayCandidateCache<'a>", '// CANDIDATE_CACHE_END')
cache = cache.replace("live:&'a [bool], enabled:bool,", "live:&'a [bool],")
cache = cache.replace('enabled:std::env::var_os("RELAYNET_CANDIDATE_CACHE").is_some(),', '')
cache = cache.replace('        if !self.enabled {return None;}\n', '')
text += '\n\n' + cache

# Exact AVX2 screening retains the scalar edge-update order and hardware fallback.
text += '''

/// Screening rejects only candidate > current target distance, retaining ties.
/// Targets are unique within a row (checked by the guard), and accepted lanes
/// execute the unchanged scalar update in ascending lane order. Rejected edges
/// cannot become useful after another relaxation decreases a target distance.
#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
#[inline]
''' + function('weighted_screen4')
text += '\n\n#[cfg(not(target_arch="x86_64"))]\nunsafe fn weighted_screen4(_:f64,_:&[(u32,f32)],_:*const f64)->u32 {15}\n'
weighted = region('fn weighted_screen_guard', '// WEIGHTED_SCREEN_END')
weighted = choose(weighted, 'true', True)
text += '\n\n' + weighted

# Scalar fallback also always records exact predecessors; reference-only branches go.
text = text.replace('weighted: bool, parents: bool, bet:', 'weighted: bool, bet:')
text = text.replace('weighted: bool, parents: bool) -> ExactPaths', 'weighted: bool) -> ExactPaths')
text = text.replace('weighted, parents, &mut bet', 'weighted, &mut bet')
text = text.replace('exact_paths_mode(g,weighted,true)', 'exact_paths_mode(g,weighted)')
text = choose(text, 'parents', True)

# Remove all profiling, dumps and timing-only locals. No numerical loops change.
text = text.replace('    rlg_dump_graph(g);\n', '').replace('    rlg_dump_digraph(g);\n', '')
text = choose(text, 'profile_detail', False)
text = re.sub(r'^    let profile_detail = .*;\n', '', text, flags=re.M)
text = re.sub(r'^    let mut profile_clock = .*;\n', '', text, flags=re.M)
start = text.find('    let profile_mark = ')
if start >= 0:
    end = end_block(text, text.index('{', start))
    text = text[:start] + text[end + 1:]
text = re.sub(r'^    profile_mark\([^\n]*\);\n', '', text, flags=re.M)
text = re.sub(r'^\s*(?:let (?:mut )?[a-z_]+ = std::time::Instant::now\(\);|let mut stage = started;|stage = std::time::Instant::now\(\);)\n', '\n', text, flags=re.M)
text = re.sub(r'\s*eprintln!\(.*?\);', '', text, flags=re.S)

# Remove experiment-boundary comments and normalize excess empty lines.
text = re.sub(r'^// (?:[A-Z_]+_(?:BEGIN|END)|BEGIN EXACT.*|END EXACT.*)\n', '', text, flags=re.M)
text = re.sub(r'\n{3,}', '\n\n', text)
for forbidden in ('std::env::', '_probe(', 'relay_old_', 'RELAY_GRAPH_CORRECT_FIXES',
                  'RLG_', 'RELAYNET_', 'WeightedExperiment', 'binary_backprop_avx512',
                  'binary_parent_backprop_avx512', 'eprintln!', 'std::time::', 'sampling', 'nsrc'):
    assert forbidden not in text, forbidden
assert text.count('fn weighted_screen4(') == 2
assert 'relay_graph_adjacency_equal(adj, &g.adj)' in text
assert 'wx.to_bits()==wy.to_bits()' in text
(HERE / 'src/relaynet_candidate.rs').write_text(text)
subprocess.run(['rustfmt', '--edition', '2021', str(HERE / 'src/relaynet_candidate.rs')], check=True)
print(f'Prepared isolated candidate: {len(text.splitlines())} lines; fixed exact defaults, no probes/env switches')
