"""Remove work for series not referenced by any registered factor or summary."""
from pathlib import Path
import argparse

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, default=Path(__file__).parent / "src/relaynet_fast.rs")
args = p.parse_args()
text = args.source.read_text()
marker = "// relaynet exact prep: unused series stay zero"
if marker in text:
    print("prep patch already applied")
    raise SystemExit(0)


def replace(old, new=""):
    global text
    if old not in text:
        raise RuntimeError(f"missing source fragment: {old[:100]!r}")
    text = text.replace(old, new, 1)


replace("    let mut ntrades = 0usize;\n", "    " + marker + "\n")
replace("let mut bo: Vec<(u32, i64, f64)>", "let mut bo: Vec<(u32, i64)>")
replace("let mut so: Vec<(u32, i64, f64)>", "let mut so: Vec<(u32, i64)>")
replace("        ntrades += 1;\n")
replace("bo.push((c30 as u32, r.bid_order, t));", "bo.push((c30 as u32, r.bid_order));")
replace("so.push((c30 as u32, r.ask_order, t));", "so.push((c30 as u32, r.ask_order));")
replace("    let big_thr = if ntrades > 0 { 5.0 * amount / ntrades as f64 } else { f64::INFINITY };\n")
replace("            let bn: f64 = ctrades[c].iter().filter(|&&(_, _, t)| t > big_thr).map(|&(_, _, t)| t).sum();\n            big30[c] = bn / tvacc[c];\n")
replace("bo.sort_unstable_by_key(|&(c, o, _)| (c, o));", "bo.sort_unstable();")
replace("so.sort_unstable_by_key(|&(c, o, _)| (c, o));", "so.sort_unstable();")
replace("""        let mut s = 0.0f64;
        while i < bo.len() && bo[i].0 == c && bo[i].1 == o {
            s += bo[i].2;
            i += 1;
        }
        buyord30[c as usize] += 1.0;
        if s > dom30[c as usize] {
            dom30[c as usize] = s;
        }
""", """        while i < bo.len() && bo[i].0 == c && bo[i].1 == o { i += 1; }
        buyord30[c as usize] += 1.0;
""")
replace("""        let mut s = 0.0f64;
        while j < so.len() && so[j].0 == c && so[j].1 == o {
            s += so[j].2;
            j += 1;
        }
        sellord30[c as usize] += 1.0;
        if s > domsell30[c as usize] {
            domsell30[c as usize] = s;
        }
""", """        while j < so.len() && so[j].0 == c && so[j].1 == o { j += 1; }
        sellord30[c as usize] += 1.0;
""")
replace("""    for c in 0..N_CELL {
        if tvacc[c] > 0.0 {
            dom30[c] /= tvacc[c];
            domsell30[c] /= tvacc[c];
        }
    }
""")
replace("""            let mut sorted = amts.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let num: f64 = sorted.iter().enumerate()
                .map(|(k, &x)| (2 * (k + 1)) as f64 * x).sum();
            vgini30[c] = num / (m as f64 * sorted.iter().sum::<f64>()) - (m as f64 + 1.0) / m as f64;
""")
replace("            tspan30[c] = (tr[m - 1].0 - tr[0].0) as f64 / 1e6;\n")
for array in ["asl", "bsl", "dska", "dskb"]:
    replace(f"    let mut {array} = [0.0f64; N_CELL];\n")
for statement in [
    "asl[c] += depth_slope(&r.ask_vols);",
    "bsl[c] += depth_slope(&r.bid_vols);",
    "dska[c] += skew10(&r.ask_vols);",
    "dskb[c] += skew10(&r.bid_vols);",
]:
    replace("            " + statement + "\n")
for field, array in [("aslope30", "asl"), ("bslope30", "bsl"), ("dskewa30", "dska"), ("dskewb30", "dskb")]:
    replace(f"            p.{field}[c] = {array}[c] / k;\n")
replace("""    for c in 1..N_CELL {
        p.rba30[c] = p.dimb30[c] - p.dimb30[c - 1];
    }
""")

args.source.write_text(text)
print(f"prep patch applied: {args.source}")
