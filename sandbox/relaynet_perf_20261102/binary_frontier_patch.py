"""Skip only the provably useless last BFS frontier; apply after binary_simd_patch."""
from pathlib import Path

p = Path(__file__).parent / "src/relaynet_fast.rs"
s = p.read_text()
if "fn binary_betweenness_run_mode(" in s:
    raise SystemExit("binary frontier patch already applied")
old = "fn binary_betweenness_run(g: &UGraph, request_simd: bool) -> Vec<f64> {"
assert old in s, "Apply binary_simd_patch.py first"
s = s.replace(old, '''fn binary_betweenness_run(g: &UGraph, request_simd: bool) -> Vec<f64> {
    binary_betweenness_run_mode(g, request_simd, std::env::var_os("RELAYNET_BINARY_FULL_FRONTIER").is_none())
}

fn binary_betweenness_run_mode(g: &UGraph, request_simd: bool, early_stop: bool) -> Vec<f64> {''', 1)
start = s.index("fn binary_betweenness_run_mode(")
brace = s.index("{", start)
depth, end = 1, brace + 1
while depth:
    depth += (s[end] == "{") - (s[end] == "}")
    end += 1
f = s[start:end]
f = f.replace("    let n = g.n;", "    let n = g.n;\n    let (component, component_sizes) = components(g);", 1)
old = """            let mut head = 0;
            while head < queue.len() {
                let u = queue[head]; head += 1;"""
new = """            let mut head = 0;
            let mut remaining = component_sizes[component[src] as usize] as usize - 1;
            let mut last_depth = i32::MAX;
            while head < queue.len() {
                let u = queue[head];
                // Once all nodes are discovered at depth L, finish every parent
                // at depth L-1. Nodes at L cannot discover another vertex or
                // contribute to any sigma. Keep the entire queue for backprop.
                if dist[u] >= last_depth { break; }
                head += 1;"""
assert old in f
f = f.replace(old, new, 1)
old = "                    if dist[v] < 0 { dist[v] = dist[u] + 1; queue.push(v); }"
new = """                    if dist[v] < 0 {
                        dist[v] = dist[u] + 1; queue.push(v);
                        if early_stop {
                            remaining -= 1;
                            if remaining == 0 { last_depth = dist[v]; }
                        }
                    }"""
assert old in f
f = f.replace(old, new, 1)
s = s[:start] + f + s[end:]
p.write_text(s)
print("Applied exact last-frontier skip; RELAYNET_BINARY_FULL_FRONTIER=1 restores full BFS scans")
