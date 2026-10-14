"""Extract the actual SIMD loop and compare compact/full strides, bit for bit."""
from pathlib import Path
import re, subprocess, tempfile
root = Path(__file__).resolve().parents[2]
s = (root / 'src/yupei_dist/matrix_stage.rs').read_text()
a = s.index('fn phase2_simd'); b = s.index('#[cfg(', a)
fn = s[a:b]
consts = '\n'.join(re.findall(r'^(?:pub )?const (?:U_\w+|N_U|FAM_\w+|BLOCK_BUCKETS|GP_STRIDE|PACK_W):[^;]+;', s, re.M))
w = s[s.index('const fn pack_width'):s.index('const GP_STRIDE')]
# Strip the declaration comments after the width helper.
w = w[:w.rfind('}')+1]
main = r"""
fn check<const NEED:u8>() {
    let n=17; let width=pack_width(NEED);
    let counts=vec![4usize;n]; let offs:Vec<_>=(0..=n).map(|i|i*4).collect();
    let cells:Vec<_>=(0..n).flat_map(|i| [0,123,2048,4095].map(move|k| (k, std::array::from_fn(|j| (i+j+1) as f32/17.0)))).collect();
    let gp:Vec<_>=(0..BLOCK_BUCKETS*GP_STRIDE).map(|i| ((i*17)%101) as f32/91.0).collect();
    let mut old=vec![0.0;n*64]; let mut new=vec![0.0;n*width];
    for _ in 0..5 {
        phase2_full::<NEED>(&mut old,&gp,&cells,&counts,&offs,n,0);
        phase2_simd::<NEED>(&mut new,&gp,&cells,&counts,&offs,n,0);
        for a in 0..n { for j in 0..width {assert_eq!(old[a*64+j].to_bits(),new[a*width+j].to_bits());} }
    }
}
fn main(){check::<1>();check::<2>();check::<3>();check::<63>();println!("cnt/vol/both/all: repeated accumulation bits identical");}
"""
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'check.rs'
    path.write_text('use std::arch::x86_64::*;\n'+consts+'\n'+w+'\n'+fn+'\n'+fn.replace('phase2_simd','phase2_full').replace('a * pack_width(NEED)','a * PACK_W')+main)
    subprocess.run(['/home/chenzongwei/.cargo/bin/rustc','--edition=2021','-O','-C','target-cpu=native',str(path),'-o',tmp+'/check'],check=True)
    subprocess.run([tmp+'/check'],check=True)
