//! Write upper triangles in the hot loop, then mirror by tiles.
use rayon::prelude::*;

struct SharedMatrix(*mut f32);
// Workers own disjoint lower-triangle rows. Every source is in the immutable
// upper triangle, so no source can be concurrently written by another worker.
unsafe impl Sync for SharedMatrix {}
impl SharedMatrix {
    unsafe fn copy(&self, dst: usize, src: usize) {
        *self.0.add(dst) = *self.0.add(src);
    }
}

pub(super) fn mirror_lower(mat: &mut [f32], n: usize) {
    assert_eq!(mat.len(), n * n);
    let ptr = SharedMatrix(mat.as_mut_ptr());
    let ptr = &ptr;
    const TILE: usize = 32;
    (0..n.div_ceil(TILE)).into_par_iter().for_each(|block| {
        let i0 = block * TILE;
        let i1 = (i0 + TILE).min(n);
        for j0 in (0..i1).step_by(TILE) {
            for i in i0..i1 {
                for j in j0..(j0 + TILE).min(i) {
                    unsafe { ptr.copy(i * n + j, j * n + i); }
                }
            }
        }
    });
}
