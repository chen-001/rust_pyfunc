//! 由 build.rs 自动生成，勿手改。
#[path = "assortativity.rs"]
pub mod assortativity;
#[path = "basket_hedge.rs"]
pub mod basket_hedge;
#[path = "betweenness.rs"]
pub mod betweenness;
#[path = "cliqueness.rs"]
pub mod cliqueness;
#[path = "closeness.rs"]
pub mod closeness;
#[path = "clustering.rs"]
pub mod clustering;
#[path = "community.rs"]
pub mod community;
#[path = "concentration.rs"]
pub mod concentration;
#[path = "directionality.rs"]
pub mod directionality;
#[path = "edge_dynamics.rs"]
pub mod edge_dynamics;
#[path = "eigenvector.rs"]
pub mod eigenvector;
#[path = "hidden_interaction.rs"]
pub mod hidden_interaction;
#[path = "industry.rs"]
pub mod industry;
#[path = "isolation.rs"]
pub mod isolation;
#[path = "leader_return.rs"]
pub mod leader_return;
#[path = "leadership.rs"]
pub mod leadership;
#[path = "motifs.rs"]
pub mod motifs;
#[path = "multiscale.rs"]
pub mod multiscale;
#[path = "neighbor_dispersion.rs"]
pub mod neighbor_dispersion;
#[path = "neighbor_strength.rs"]
pub mod neighbor_strength;
#[path = "neighbor_z.rs"]
pub mod neighbor_z;
#[path = "network_momentum.rs"]
pub mod network_momentum;
#[path = "pagerank.rs"]
pub mod pagerank;
#[path = "reciprocity.rs"]
pub mod reciprocity;
#[path = "starness.rs"]
pub mod starness;
#[path = "strength.rs"]
pub mod strength;
#[path = "surplus.rs"]
pub mod surplus;
#[path = "topk.rs"]
pub mod topk;
use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
pub struct IndicatorDef {
    pub name: &'static str,
    pub compute: fn(&IndicatorCtx) -> Vec<IndicatorResult>,
}
pub fn all() -> Vec<IndicatorDef> {
    vec![
        IndicatorDef {
            name: "assortativity",
            compute: assortativity::compute,
        },
        IndicatorDef {
            name: "basket_hedge",
            compute: basket_hedge::compute,
        },
        IndicatorDef {
            name: "betweenness",
            compute: betweenness::compute,
        },
        IndicatorDef {
            name: "cliqueness",
            compute: cliqueness::compute,
        },
        IndicatorDef {
            name: "closeness",
            compute: closeness::compute,
        },
        IndicatorDef {
            name: "clustering",
            compute: clustering::compute,
        },
        IndicatorDef {
            name: "community",
            compute: community::compute,
        },
        IndicatorDef {
            name: "concentration",
            compute: concentration::compute,
        },
        IndicatorDef {
            name: "directionality",
            compute: directionality::compute,
        },
        IndicatorDef {
            name: "edge_dynamics",
            compute: edge_dynamics::compute,
        },
        IndicatorDef {
            name: "eigenvector",
            compute: eigenvector::compute,
        },
        IndicatorDef {
            name: "hidden_interaction",
            compute: hidden_interaction::compute,
        },
        IndicatorDef {
            name: "industry",
            compute: industry::compute,
        },
        IndicatorDef {
            name: "isolation",
            compute: isolation::compute,
        },
        IndicatorDef {
            name: "leader_return",
            compute: leader_return::compute,
        },
        IndicatorDef {
            name: "leadership",
            compute: leadership::compute,
        },
        IndicatorDef {
            name: "motifs",
            compute: motifs::compute,
        },
        IndicatorDef {
            name: "multiscale",
            compute: multiscale::compute,
        },
        IndicatorDef {
            name: "neighbor_dispersion",
            compute: neighbor_dispersion::compute,
        },
        IndicatorDef {
            name: "neighbor_strength",
            compute: neighbor_strength::compute,
        },
        IndicatorDef {
            name: "neighbor_z",
            compute: neighbor_z::compute,
        },
        IndicatorDef {
            name: "network_momentum",
            compute: network_momentum::compute,
        },
        IndicatorDef {
            name: "pagerank",
            compute: pagerank::compute,
        },
        IndicatorDef {
            name: "reciprocity",
            compute: reciprocity::compute,
        },
        IndicatorDef {
            name: "starness",
            compute: starness::compute,
        },
        IndicatorDef {
            name: "strength",
            compute: strength::compute,
        },
        IndicatorDef {
            name: "surplus",
            compute: surplus::compute,
        },
        IndicatorDef {
            name: "topk",
            compute: topk::compute,
        },
    ]
}
