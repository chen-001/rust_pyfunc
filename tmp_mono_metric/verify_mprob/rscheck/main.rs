
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    let path = std::env::args().nth(1).expect("usage: mprob_rs <cases.txt>");
    let reader = BufReader::new(File::open(path).unwrap());
    let mut lines = reader.lines().map(|l| l.unwrap());
    let mut out = String::new();
    while let Some(hdr) = lines.next() {
        let h: Vec<&str> = hdr.split_whitespace().collect();
        let portf: usize = h[0].parse().unwrap();
        let g: usize = h[1].parse().unwrap();
        let mut cols: Vec<Vec<f64>> = Vec::with_capacity(g);
        for _ in 0..g {
            let row = lines.next().unwrap();
            let mut it = row.split_whitespace();
            let len: usize = it.next().unwrap().parse().unwrap();
            let mut v: Vec<f64> = Vec::with_capacity(len);
            for tok in it {
                v.push(tok.parse::<f64>().unwrap());
            }
            assert_eq!(v.len(), len);
            cols.push(v);
        }
        let val = compute_mprob(&cols, portf);
        out.push_str(&format!("{}\n", val));
    }
    print!("{}", out);
}
