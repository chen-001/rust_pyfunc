//! 玉佩精炼版：独立沙箱，所有因子从逐笔到每股标量均由Rust计算。
//! 不注册正式库、不提交任务。完整历史特征使用显式快照，不自动批量回溯。
pub mod compute;
pub mod extensions;
pub mod input;
pub mod kernel;
pub mod leadlag;
pub mod model;
pub mod network;
pub mod temporal;
