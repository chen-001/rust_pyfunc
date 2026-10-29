//! 因子列名清单：与 compute.rs 的产出顺序逐字一致，供 pipeline 在不算数据
//! 的情况下取 names（expected_result_length 与 store 列名）。
use crate::yupei_gpt::compute::CHAINS;

const NETWORK_NAMES: [&str; 32] = [
    "row_rho_skew",
    "row_gs_skew",
    "row_gnpos_p75",
    "row_gnpos_p90",
    "e_w_std_soft",
    "res_z_p90",
    "res_g_max",
    "res_g_t20m",
    "r4_ov_g_k100",
    "r4_ov_z_k100",
    "x0topk20_count",
    "gtopk100_count",
    "row_self_assoc_p95",
    "row_gcov_mean",
    "row_gs_hhip",
    "row_g_p90",
    "row_g_t20m",
    "row_gs_effective_neighbors",
    "row_rho_g_weighted_overlap",
    "same_industry_g_mean",
    "cross_industry_g_mean",
    "same_minus_cross_g_mean",
    "r4_ov_g_directed_k100",
    "r4_ov_z_directed_k100",
    "diag_rho_row_mean",
    "diag_rho_row_max",
    "diag_rho_at_g_top100_mean",
    "diag_g_row_mean",
    "diag_g_row_max",
    "diag_g_top100_mean",
    "best_z_partner_reverse_rank",
    "best_z_partner_reverse_rank_fraction",
];

fn dir(c: char) -> usize {
    match c {
        'A' | 'a' => 0,
        'p' => 1,
        'm' => 2,
        _ => 0,
    }
}

/// 主 pipeline（单日）491 列：15 链网络（含结构性精简）→ 控制/对比 →
/// 行业活跃度残差 → 价格响应 → 交互 → leadlag。
pub fn daily_names() -> Vec<String> {
    let mut out = Vec::with_capacity(491);
    for chain in CHAINS {
        let b: Vec<char> = chain.chars().collect();
        let (a, bb, c) = (dir(b[0]), dir(b[1]), dir(b[2]));
        for (k, name) in NETWORK_NAMES.iter().enumerate() {
            if k == 0 && !(a == bb && bb == c) {
                continue;
            }
            if k >= 30 && chain != "ppp" {
                continue;
            }
            let full = format!("yr_{chain}_{name}");
            // 结构性精简（compute.rs 的 retain）：hhip 与 mp* 三件套不出列。
            if full.ends_with("_row_gs_hhip") {
                continue;
            }
            if full.starts_with("yr_mp")
                && ["_e_w_std_soft", "_x0topk20_count", "_gtopk100_count"]
                    .iter()
                    .any(|suf| full.ends_with(suf))
            {
                continue;
            }
            out.push(full);
        }
    }
    for n in [
        "yr_control_log_trade_count",
        "yr_control_log_amount",
        "yr_control_count_imbalance",
        "yr_control_log_buy_sell_energy_ratio",
        "yr_buy_minus_sell_rank_row_rho_skew",
        "yr_buy_minus_sell_rank_row_gs_skew",
        "yr_buy_minus_sell_rank_row_gnpos_p90",
        "yr_buy_minus_sell_rank_row_gcov_mean",
        "yr_buy_minus_sell_rank_e_w_std_soft",
        "yr_buy_minus_sell_rank_res_z_p90",
        "yr_buy_minus_sell_rank_r4_ov_g_k100",
        "yr_buy_minus_sell_rank_row_self_assoc_p95",
        "yr_cross_role_rank_gcov",
    ] {
        out.push(n.into());
    }
    // extensions.rs: 行业活跃度残差只挂在三条全同向链上。
    for (chain, stem) in [
        ("AAa", "row_gs_skew"),
        ("AAa", "row_gnpos_p90"),
        ("AAa", "res_z_p90"),
        ("ppp", "row_gs_skew"),
        ("ppp", "row_gnpos_p90"),
        ("ppp", "res_z_p90"),
        ("mmm", "row_gs_skew"),
        ("mmm", "row_gnpos_p90"),
        ("mmm", "res_z_p90"),
    ] {
        out.push(format!("yr_{chain}_{stem}_industry_activity_residual"));
    }
    for h in [1, 5, 15] {
        for stem in [
            "buy_response_beta",
            "sell_response_beta",
            "sell_absorption",
            "buy_exhaustion",
            "response_valid_minutes",
        ] {
            out.push(format!("yr_{stem}_industry_exself_{h}m"));
        }
    }
    out.push("yr_sell_network_absorption".into());
    out.push("yr_buy_network_exhaustion".into());
    // leadlag.rs 的顺序：market t0p1/t1(am,pm,pm-am)/t5，再 industry 三组。
    let dirs = ["buy_buy", "buy_sell", "sell_buy", "sell_sell"];
    for label in ["t0p1", "t1", "t1_am", "t1_pm", "t1_pm_minus_am", "t5"] {
        for d in dirs {
            out.push(format!("yr_leadlag_market_{d}_{label}"));
        }
    }
    for label in ["industry", "cross_industry", "industry_minus_cross"] {
        for d in dirs {
            out.push(format!("yr_leadlag_{label}_{d}_t1"));
        }
    }
    out
}

/// 第二 pipeline：29 个历史时序列（顺序 = SPECS 14 个滚动 + 每链 5 个）。
pub fn temporal_names() -> Vec<String> {
    let mut out = Vec::with_capacity(29);
    for n in [
        "yr_AAa_row_rho_skew_mean_smooth_5",
        "yr_ppp_row_gs_skew_mean_smooth_20",
        "yr_ppp_res_z_p90_min_smooth_5",
        "yr_ppp_r4_ov_g_k100_mean_smooth_10",
        "yr_mmm_r4_ov_z_k100_mean_smooth_10",
        "yr_ppp_row_self_assoc_p95_absdev_csmean_max_smooth_20",
        "yr_ppa_row_gnpos_p90_min_smooth_5",
        "yr_AAp_e_w_std_soft_max_smooth_5",
        "yr_AAp_e_w_std_soft_max_smooth_10",
        "yr_pmp_x0topk20_count_mean_smooth_5",
        "yr_AAp_gtopk100_count_mean_smooth_10",
        "yr_AAm_res_g_max_mean_smooth_10",
        "yr_ppp_best_z_partner_reverse_rank_mean_smooth_5",
        "yr_ppp_best_z_partner_reverse_rank_fraction_mean_smooth_5",
    ] {
        out.push(n.into());
    }
    for chain in ["AAa", "ppp", "mmm"] {
        out.push(format!(
            "yr_{chain}_row_gnpos_p90_above_csmedian_persistence_10"
        ));
        out.push(format!("yr_{chain}_row_gnpos_p90_change_vs_previous_mean20"));
        for suffix in ["new_share", "lost_share", "jaccard"] {
            out.push(format!(
                "yr_{chain}_positive_gs_neighbors_{suffix}_previous_day"
            ));
        }
    }
    out
}
