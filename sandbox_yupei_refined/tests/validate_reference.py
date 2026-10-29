"""Independent NumPy dense oracle, all pma/ppp network outputs + timing.
Run: python tests/validate_reference.py [--chain pma|ppp] [--n 128] [--jobs 1]
Builds only this sandbox's release test fixture; does not install or touch real data.
Input is positive synthetic feature vectors. Python independently constructs all
nine normalized Gram matrices, enumerates every ordered B,C and every eligible X,
and solves ridge via an augmented least-squares system (Rust uses Cholesky).
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np


def reference(data):
    f = np.array(data["features"], dtype=float)
    n = len(f)
    a, b, c = {"pma": (1, 2, 0), "ppp": (1, 1, 1)}[data["chain"]]
    vectors = np.stack((f[:, 0] + f[:, 1], f[:, 0], f[:, 1]))
    energy = (vectors * vectors).sum(axis=2)
    rho = np.empty((3, 3, n, n), np.float32)
    for ia in range(3):
        for ib in range(3):
            rho[ia, ib] = (vectors[ia] @ vectors[ib].T) / np.sqrt(energy[ia, :, None] * energy[ib, None, :])
    r = rho[a, b]
    mats = {name: np.full((n, n), np.nan) for name in ("g", "gs", "gnpos", "self", "gcov")}
    mass = np.zeros((3, n)); hc = np.zeros(n); gc = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = np.sqrt(energy[a, i] + energy[b, j] + 2 * r[i, j] * np.sqrt(energy[a, i] * energy[b, j]))
            wi, wj = np.float32(np.sqrt(energy[a, i]) / d), np.float32(np.sqrt(energy[b, j]) / d)
            xs = np.array([x for x in range(n) if x != i and x != j])
            p, q = rho[a, c, i, xs], rho[b, c, j, xs]
            h = wi * p + wj * q
            gain = h - np.maximum(p, q)
            byh = np.argsort(-h, kind="stable"); byg = np.argsort(-gain, kind="stable")
            best = byh[0]
            mats["g"][i, j] = gain[best]; mats["gs"][i, j] = gain[byg[0]]
            mats["self"][i, j] = p[best]
            # Rust stores these pair summaries in f32 before row aggregation.
            mats["gnpos"][i, j] = np.float32(np.mean(gain > 0))
            mats["gcov"][i, j] = np.float32(np.maximum(gain, 0).astype(float).mean())
            hc[xs[byh[:20]]] += 1; gc[xs[byg[:100]]] += 1
            w = np.maximum(h, 0).astype(float); w /= w.sum()
            mass[0, xs] += w; mass[1, xs] += w * gain; mass[2, xs] += w * gain.astype(float) ** 2
    counts = np.array([[s["n_buy"], s["n_sell"]] for s in data["stocks"]], float)
    la, lb = np.log1p(counts[:, a - 1]), np.log1p(counts[:, b - 1]); ea, eb = np.log1p(energy[a]), np.log1p(energy[b])
    pooled = np.r_[la, lb]; cuts = np.quantile(pooled, np.arange(1, 10) / 10)
    dec = np.searchsorted(cuts, pooled, side="left")
    strata = np.minimum(dec[:n, None], dec[None, n:]) * 10 + np.maximum(dec[:n, None], dec[None, n:])
    g = mats["g"]; valid = np.isfinite(g); z = np.full_like(g, np.nan)
    for cell in range(100):
        sel = valid & (strata == cell)
        population = g[sel] if sel.sum() >= 30 else g[valid]
        if population.std() > 1e-12:
            z[sel] = ((g[sel] - population.mean()) / population.std()).astype(np.float32)
    cov = np.stack((r.astype(float), la[:, None] + lb, abs(la[:, None] - lb), ea[:, None] + eb, abs(ea[:, None] - eb)), axis=2)
    def resid(y):
        mask = np.isfinite(y); x = cov[mask]; yy = y[mask]; scale = x.std(axis=0); scale[scale <= 1e-12] = 1
        xx = (x - x.mean(axis=0)) / scale
        augmented = np.vstack((xx, np.sqrt(len(x) * 1e-6) * np.eye(5)))
        beta = np.linalg.lstsq(augmented, np.r_[yy - yy.mean(), np.zeros(5)], rcond=None)[0]
        out = np.full_like(y, np.nan); out[mask] = yy - yy.mean() - xx @ beta
        return out
    rg, rz = resid(g), resid(z)
    def qtile(x, q): return np.nanquantile(x, q, axis=1)
    def topmean(x, k): return np.array([np.sort(v[np.isfinite(v)])[-k:].mean() for v in x])
    def hhi(x):
        p = np.where(np.isfinite(x), np.maximum(x, 0), 0); den = p.sum(axis=1)
        return np.divide((p * p).sum(axis=1), den * den, out=np.full(n, np.nan), where=den > 0)
    gs = mats["gs"]; gm = np.nanmean(gs, axis=1); gsstd = np.nanstd(gs, axis=1)
    out = {
        "row_gs_skew": np.nanmean((gs - gm[:, None]) ** 3, axis=1) / gsstd ** 3,
        "row_gnpos_p75": qtile(mats["gnpos"], .75), "row_gnpos_p90": qtile(mats["gnpos"], .9),
        "e_w_std_soft": np.sqrt(np.maximum(mass[2] / mass[0] - (mass[1] / mass[0]) ** 2, 0)),
        "res_z_p90": qtile(rz, .9), "res_g_max": np.nanmax(rg, axis=1), "res_g_t20m": topmean(rg, 20),
        "x0topk20_count": hc, "gtopk100_count": gc, "row_self_assoc_p95": qtile(mats["self"], .95),
        "row_gcov_mean": np.nanmean(mats["gcov"], axis=1), "row_gs_hhip": hhi(gs),
        "row_g_p90": qtile(g, .9), "row_g_t20m": topmean(g, 20), "row_gs_effective_neighbors": 1 / hhi(gs),
    }
    rr = np.where(valid, r, np.nan)
    tops = []
    for mat in (rr, g, z):
        rows = [np.flatnonzero(np.isfinite(v))[np.argsort(-v[np.isfinite(v)], kind="stable")[:100]] for v in mat]
        adj = np.zeros((n, n), bool)
        for i, neighbors in enumerate(rows): adj[i, neighbors] = True
        tops.append((rows, adj | adj.T))
    for k, tag in [(1, "g"), (2, "z")]:
        out[f"r4_ov_{tag}_k100"] = (tops[0][1] & tops[k][1]).sum(axis=1) / 100
        out[f"r4_ov_{tag}_directed_k100"] = np.array([len(set(a) & set(b)) / min(len(a), len(b)) for a, b in zip(tops[0][0], tops[k][0])])
    rp = np.nan_to_num(np.maximum(rr, 0)); gp = np.nan_to_num(np.maximum(g, 0))
    rp /= rp.sum(axis=1, keepdims=True); gp /= gp.sum(axis=1, keepdims=True)
    out["row_rho_g_weighted_overlap"] = np.minimum(rp, gp).sum(axis=1) / np.maximum(rp, gp).sum(axis=1)
    industries = np.array([s["industry"] for s in data["stocks"]]); same = industries[:, None] == industries
    out["same_industry_g_mean"] = np.nanmean(np.where(same, g, np.nan), axis=1)
    out["cross_industry_g_mean"] = np.nanmean(np.where(~same, g, np.nan), axis=1)
    out["same_minus_cross_g_mean"] = out["same_industry_g_mean"] - out["cross_industry_g_mean"]
    for tag, mat in [("rho", rr), ("g", g)]:
        out[f"diag_{tag}_row_mean"] = np.nanmean(mat, axis=1)
        out[f"diag_{tag}_row_max"] = np.nanmax(mat, axis=1)
    out["diag_rho_at_g_top100_mean"] = np.array([rr[i, js].mean() for i, js in enumerate(tops[1][0])])
    out["diag_g_top100_mean"] = np.array([g[i, js].mean() for i, js in enumerate(tops[1][0])])
    if data["chain"] == "ppp":
        out["row_rho_skew"] = np.nanmean((rr - np.nanmean(rr, axis=1)[:, None]) ** 3, axis=1) / np.nanstd(rr, axis=1) ** 3
        counts, fractions = [], []
        for i, partners in enumerate(tops[2][0]):
            if not len(partners):
                counts.append(np.nan); fractions.append(np.nan); continue
            p = partners[0]; mask = np.isfinite(z[p]); mask[p] = False
            count = float(np.count_nonzero(z[p, mask] > z[i, p]))
            counts.append(count if mask.any() else np.nan)
            fractions.append(count / mask.sum() if mask.any() else np.nan)
        out["best_z_partner_reverse_rank"] = np.array(counts)
        out["best_z_partner_reverse_rank_fraction"] = np.array(fractions)
    return {"yr_" + data["chain"] + "_" + k: v for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=128); ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--chain", choices=("pma", "ppp"), default="pma")
    args = ap.parse_args()
    if args.n < 12: ap.error("--n must be at least 12 for a nondegenerate reference")
    crate = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="yupei_reference_") as temp:
        output = Path(temp) / "fixture.json"
        env = dict(os.environ, YUPEI_REFERENCE_JSON=str(output), YUPEI_REFERENCE_N=str(args.n), YUPEI_REFERENCE_JOBS=str(args.jobs), YUPEI_REFERENCE_CHAIN=args.chain)
        subprocess.run(["cargo", "test", "--release", "--offline", "--manifest-path", str(crate / "Cargo.toml"), "--test", "network", "write_python_reference_fixture", "--", "--ignored", "--exact"], env=env, check=True)
        data = json.loads(output.read_text())
    begin = time.perf_counter(); expected = reference(data); elapsed = time.perf_counter() - begin
    assert set(expected) == {c["name"] for c in data["columns"]}
    errors = {}
    for col in data["columns"]:
        actual = np.array(col["values"], float); want = expected[col["name"]]
        # f32 SIMD mass reductions differ slightly from NumPy f64 accumulation.
        np.testing.assert_allclose(actual, want, rtol=2e-5, atol=5e-6, equal_nan=True, err_msg=col["name"])
        errors[col["name"]] = float(np.nanmax(abs(actual - want)))
    print(json.dumps({"n": args.n, "chain": args.chain, "columns_compared": len(expected), "max_abs_error": max(errors.values()), "rust_network_seconds": data["rust_network_seconds"], "python_reference_seconds": elapsed, "python_over_rust": elapsed / data["rust_network_seconds"], "note": "Python timing also includes independent Gram construction; Rust timing includes audits and neighbor state. Informational timing, not an equal-work microbenchmark."}, indent=2))


if __name__ == "__main__": main()
