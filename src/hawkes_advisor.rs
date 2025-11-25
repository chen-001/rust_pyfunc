use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, IntoPyArray};

/// Analyze Hawkes process indicators and provide trading advice
///
/// This function analyzes Hawkes model output metrics, automatically analyzes
/// market microstructure features, and provides quantitative trading and strategy suggestions.
#[pyfunction]
#[pyo3(signature = (
    mu,
    alpha,
    beta,
    branching_ratio,
    mean_intensity,
    expected_cluster_size,
    half_life,
    cluster_sizes,
))]
pub fn analyze_hawkes_indicators(
    py: Python,
    mu: f64,
    alpha: f64,
    beta: f64,
    branching_ratio: f64,
    mean_intensity: f64,
    expected_cluster_size: f64,
    half_life: f64,
    cluster_sizes: PyReadonlyArray1<i32>,
) -> PyResult<PyObject> {
    let cluster_sizes_arr = cluster_sizes.as_array();
    let cluster_sizes_vec: Vec<i32> = cluster_sizes_arr.iter().cloned().collect();

    // Analyze metrics
    let (branching_level, branching_interpretation) = analyze_branching_ratio(branching_ratio);
    let (cluster_size_score, cluster_interpretation) = analyze_cluster_sizes(&cluster_sizes_vec);
    let (memory_score, memory_interpretation) = analyze_market_memory(half_life, beta);
    let market_state = assess_market_state(
        branching_ratio,
        expected_cluster_size,
        half_life,
        cluster_size_score,
    );
    let trading_suggestions = generate_trading_suggestions(
        branching_ratio,
        expected_cluster_size,
        half_life,
        mu,
        &cluster_sizes_vec,
    );
    let (total_clusters, large_clusters, max_cluster) = calculate_cluster_stats(&cluster_sizes_vec);

    let result_dict = pyo3::types::PyDict::new(py);
    result_dict.set_item("branching_ratio", branching_ratio)?;
    result_dict.set_item("branching_level", branching_level)?;
    result_dict.set_item("branching_interpretation", branching_interpretation)?;
    result_dict.set_item("cluster_size_score", cluster_size_score)?;
    result_dict.set_item("cluster_interpretation", cluster_interpretation)?;
    result_dict.set_item("market_memory_score", memory_score)?;
    result_dict.set_item("memory_interpretation", memory_interpretation)?;
    result_dict.set_item("overall_market_state", market_state)?;
    result_dict.set_item("trading_suggestions", trading_suggestions)?;
    result_dict.set_item("total_clusters", total_clusters)?;
    result_dict.set_item("large_clusters_10", large_clusters)?;
    result_dict.set_item("max_cluster_size", max_cluster)?;

    Ok(result_dict.into())
}

fn analyze_branching_ratio(branching_ratio: f64) -> (String, String) {
    let level = match branching_ratio {
        r if r >= 0.9 => "Extremely Strong",
        r if r >= 0.7 => "Strong",
        r if r >= 0.5 => "Moderate",
        r if r >= 0.3 => "Weak",
        r if r >= 0.1 => "Very Weak",
        _ => "Negligible",
    };

    let interpretation = match branching_ratio {
        r if r >= 0.9 => "Extremely strong self-excitation, endogenous events dominate",
        r if r >= 0.7 => "Strong self-excitation, most events triggered, obvious clustering",
        r if r >= 0.5 => "Moderate self-excitation, significant mutual influence",
        r if r >= 0.3 => "Weak self-excitation, some trigger relationships",
        r if r >= 0.1 => "Very weak self-excitation, mostly independent with small clustering",
        _ => "Negligible self-excitation, nearly independent events, highly random",
    };

    (level.to_string(), interpretation.to_string())
}

fn analyze_cluster_sizes(cluster_sizes: &[i32]) -> (f64, String) {
    if cluster_sizes.is_empty() {
        return (0.0, "No valid clusters".to_string());
    }

    let total_clusters = cluster_sizes.len() as f64;
    let single_event_clusters = cluster_sizes.iter().filter(|&&s| s == 1).count() as f64;
    let large_clusters: Vec<i32> = cluster_sizes.iter().filter(|&&s| s >= 10).cloned().collect();
    let large_clusters_count = large_clusters.len() as f64;

    let single_ratio = single_event_clusters / total_clusters;
    let large_ratio = large_clusters_count / total_clusters;
    let score = (1.0 - single_ratio * 0.7 + large_ratio * 0.3).min(1.0).max(0.0);

    let interpretation = if score >= 0.8 {
        format!(
            "Strong clustering: Multi-event clusters account for {:.1}%, max cluster size {:?} events",
            large_ratio * 100.0,
            large_clusters.iter().max().unwrap_or(&0)
        )
    } else if score >= 0.5 {
        format!(
            "Moderate clustering: Multi-event clusters account for {:.1}%, {} clusters with 10+ events",
            (1.0 - single_ratio) * 100.0,
            large_clusters.len()
        )
    } else if score >= 0.3 {
        "Weak clustering: Mostly single-event clusters with few multi-event clusters".to_string()
    } else {
        "No significant clustering: Predominantly single-event clusters, highly random".to_string()
    };

    (score, interpretation)
}

fn analyze_market_memory(half_life: f64, beta: f64) -> (f64, String) {
    let (score, interpretation) = match half_life {
        t if t < 0.05 => (0.9, format!(
            "Ultra-short memory: half-life {:.3}s, extremely fast market response, beta={:.2}",
            t as f32, beta as f32
        )),
        t if t < 0.2 => (0.7, format!(
            "Short memory: half-life {:.3}s, brief market memory, beta={:.2}",
            t as f32, beta as f32
        )),
        t if t < 1.0 => (0.5, format!(
            "Medium memory: half-life {:.3}s, moderate market persistence, beta={:.2}",
            t as f32, beta as f32
        )),
        t if t < 5.0 => (0.3, format!(
            "Long memory: half-life {:.3}s, extended market memory, beta={:.2}",
            t as f32, beta as f32
        )),
        _ => (0.1, format!(
            "Ultra-long memory: half-life {:.3}s, slow market response, beta={:.2}",
            half_life as f32, beta as f32
        )),
    };

    (score, interpretation)
}

fn assess_market_state(
    branching_ratio: f64,
    expected_cluster_size: f64,
    half_life: f64,
    cluster_size_score: f64,
) -> String {
    let score = branching_ratio * 0.4
        + (expected_cluster_size - 1.0) * 0.1
        + (1.0 / (1.0 + half_life)) * 0.3
        + cluster_size_score * 0.2;

    match score {
        s if s >= 0.7 => "Strong trending market: Strong self-excitation and clustering, recommend trend-following strategies".to_string(),
        s if s >= 0.5 => "Mild trending market: Moderate self-excitation with local clustering, can try short-term momentum".to_string(),
        s if s >= 0.3 => "Mean-reverting random: Weak self-excitation with minor clustering, suitable for mean-reversion strategies".to_string(),
        _ => "Highly random market: Very weak self-excitation, independent events, recommend avoiding momentum strategies".to_string(),
    }
}

fn generate_trading_suggestions(
    branching_ratio: f64,
    expected_cluster_size: f64,
    half_life: f64,
    mu: f64,
    cluster_sizes: &[i32],
) -> Vec<String> {
    let mut suggestions = Vec::new();

    // Based on branching ratio
    match branching_ratio {
        r if r >= 0.7 => {
            suggestions.push("Strong trend signal: Recommend trend-following strategy".to_string());
            suggestions.push("Momentum play: Follow direction at cluster formation".to_string());
        }
        r if r >= 0.4 => {
            suggestions.push("Mild trend: Try short-term momentum strategy".to_string());
            suggestions.push("Watch cluster start: First event of large cluster may be trend start".to_string());
        }
        r if r >= 0.2 => {
            suggestions.push("Weak mean-reverting: Avoid chasing trends".to_string());
            suggestions.push("Counter-trend: Reverse position after cluster ends".to_string());
        }
        _ => {
            suggestions.push("High randomness: Avoid momentum strategies".to_string());
            suggestions.push("Mean-reversion: Watch for price deviation reversals".to_string());
        }
    }

    // Based on expected cluster size
    match expected_cluster_size {
        s if s >= 3.0 => {
            suggestions.push(format!("Large cluster opportunities: Average cluster size {:.1}, watch high-volume clusters", s));
        }
        s if s >= 1.5 => {
            suggestions.push(format!("Medium clusters: Average cluster size {:.1}, many short-term opportunities", s));
        }
        _ => {
            suggestions.push("Small clusters dominate: Market fragmented, lower profit expectations".to_string());
        }
    }

    // Based on half-life
    match half_life {
        t if t < 0.1 => {
            suggestions.push(format!("Ultra-short: half-life {:.3}s, suitable for high-frequency trading", t));
        }
        t if t < 0.5 => {
            suggestions.push(format!("Short-term memory: half-life {:.3}s, suitable for intraday scalping", t));
        }
        _ => {
            suggestions.push(format!("Long-term memory: half-life {:.2}s, reduce trading frequency", half_life));
        }
    }

    // Based on cluster size distribution
    let large_clusters: Vec<i32> = cluster_sizes.iter().filter(|&&s| s >= 10).cloned().collect();
    if large_clusters.len() >= 5 {
        suggestions.push(format!("Multiple large clusters: {} clusters with 10+ events, obvious local clustering", large_clusters.len()));
        suggestions.push("Watch volume spikes: Large clusters usually accompanied by volume expansion".to_string());
    } else if large_clusters.len() >= 1 {
        let max_size = *large_clusters.iter().max().unwrap_or(&0);
        suggestions.push(format!("Few large clusters: Max cluster {} events, watch abnormal clustering", max_size));
    } else {
        suggestions.push("No significant large clusters: Market fragmented, volatility may be low".to_string());
    }

    // Based on intensity
    if mu > 1.0 {
        suggestions.push(format!("High activity: Base intensity {:.3} events/sec, many trading opportunities", mu));
    } else if mu < 0.1 {
        suggestions.push(format!("Low activity: Base intensity {:.3} events/sec, watch liquidity", mu));
    }

    suggestions
}

fn calculate_cluster_stats(cluster_sizes: &[i32]) -> (usize, usize, i32) {
    let total_clusters = cluster_sizes.len();
    let large_clusters = cluster_sizes.iter().filter(|&&s| s >= 10).count();
    let max_cluster = cluster_sizes.iter().max().cloned().unwrap_or(0);

    (total_clusters, large_clusters, max_cluster)
}
