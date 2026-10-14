p='/home/chenzongwei/rust_pyfunc/src/yupei_pair/net.rs'
s=open(p).read(); done=[]
def rep(old,new,tag):
    global s
    if new in s: done.append('SKIP '+tag); return
    assert old in s, 'MISS '+tag
    s=s.replace(old,new,1); done.append('OK '+tag)
rep("    let npair_eff;\n    let mass_vectors: Vec<f32>;",
    "    let npair_eff;\n    let mass_vectors: Vec<f32>;\n    let mut topk_total: Option<TopkAcc>;", 'declare')
rep("""        let mut topk_total: Option<TopkAcc> = if topk_on {
            Some(TopkAcc::new(n, nlev_t))
        } else {
            None
        };""",
"""        topk_total = if topk_on {
            Some(TopkAcc::new(n, nlev_t))
        } else {
            None
        };""", 'assign')
open(p,'w').write(s)
print('net.rs:', done)
q='/home/chenzongwei/rust_pyfunc/src/yupei_pair/topk.rs'
t=open(q).read()
if 'pub(crate) fn add(&mut self' not in t:
    t=t.replace("    fn add(&mut self, o: &Acc) {", "    pub(crate) fn add(&mut self, o: &Acc) {",1)
    open(q,'w').write(t); print('topk.rs: add pub')
r='/home/chenzongwei/rust_pyfunc/src/yupei_pair/mod.rs'
m=open(r).read()
if 'super::topk::finalize_total' in m:
    m=m.replace('super::topk::finalize_total','topk::finalize_total',1)
    open(r,'w').write(m); print('mod.rs: path fixed')
