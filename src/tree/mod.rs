use pyo3::prelude::*;
use numpy::PyReadonlyArray1;
// use std::ptr;
use std::cmp;
use std::collections::VecDeque;
use std::collections::HashMap;
use chrono::{TimeZone, Utc};

#[derive(Debug, Clone)]
pub struct TreeNode {
    price: f64,
    volume: f64,
    probability: f64,
    time: i64,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

impl TreeNode {
    // 计算节点的平衡因子
    fn balance_factor(&self) -> i64 {
        let left_height = self.left.as_ref().map_or(0, |n| n.height());
        let right_height = self.right.as_ref().map_or(0, |n| n.height());
        left_height - right_height
    }

    // 计算节点的高度
    fn height(&self) -> i64 {
        let left_height = self.left.as_ref().map_or(0, |n| n.height());
        let right_height = self.right.as_ref().map_or(0, |n| n.height());
        1 + cmp::max(left_height, right_height)
    }

    // 计算子树节点数
    fn subtree_size(&self) -> i64 {
        let left_size = self.left.as_ref().map_or(0, |n| n.subtree_size());
        let right_size = self.right.as_ref().map_or(0, |n| n.subtree_size());
        1 + left_size + right_size
    }
}

#[pyclass]
pub struct PriceTree {
    root: Option<TreeNode>,
    last_price: Option<f64>,
    last_node: Option<*mut TreeNode>,
    // 基本特征
    depth: i64,
    node_count: i64,
    leaf_count: i64,
    total_volume: f64,
    min_price: f64,
    max_price: f64,
    earliest_time: i64,
    latest_time: i64,
    // 新增特征
    degree_one_count: i64,    // 度为1的节点数
    degree_two_count: i64,    // 度为2的节点数
    total_depth: i64,         // 所有节点深度之和
    min_depth: i64,           // 最小深度
    max_balance_factor: i64,  // 最大平衡因子
    total_balance_factor: i64,// 总平衡因子
    total_path_length: i64,   // 总路径长度
    max_path_length: i64,     // 最长路径长度
    max_subtree_nodes: i64,   // 最大子树节点数
    total_subtree_nodes: i64, // 总子树节点数
    tree_width: i64,          // 树的最大宽度
    internal_nodes: i64,      // 内部节点数
}

// 实现Send trait，表明可以安全地在线程间传递
unsafe impl Send for PriceTree {}

#[pymethods]
impl PriceTree {
    #[new]
    fn new() -> Self {
        PriceTree { 
            root: None,
            last_price: None,
            last_node: None,
            depth: 0,
            node_count: 0,
            leaf_count: 0,
            total_volume: 0.0,
            min_price: f64::MAX,
            max_price: f64::MIN,
            earliest_time: i64::MAX,
            latest_time: i64::MIN,
            degree_one_count: 0,
            degree_two_count: 0,
            total_depth: 0,
            min_depth: 0,
            max_balance_factor: 0,
            total_balance_factor: 0,
            total_path_length: 0,
            max_path_length: 0,
            max_subtree_nodes: 0,
            total_subtree_nodes: 0,
            tree_width: 0,
            internal_nodes: 0,
        }
    }

    fn build_tree(
        &mut self, 
        times: PyReadonlyArray1<i64>, 
        prices: PyReadonlyArray1<f64>, 
        volumes: PyReadonlyArray1<f64>
    ) -> PyResult<()> {
        let times = times.as_array();
        let prices = prices.as_array();
        let volumes = volumes.as_array();
        
        let n = times.len();
        if n == 0 {
            return Ok(());
        }

        // 计算总成交量
        let total_volume: f64 = volumes.sum();

        // 创建根节点
        self.root = Some(TreeNode {
            price: prices[0],
            volume: volumes[0],
            probability: volumes[0] / total_volume,
            time: times[0],
            left: None,
            right: None,
        });

        // 初始化last_price
        self.last_price = Some(prices[0]);
        
        // 预分配一个缓存数组来存储当前路径上的节点引用
        let mut path: Vec<*mut TreeNode> = Vec::with_capacity(32);  // 假设树的深度不会超过32
        
        // 按时间顺序构建树
        for i in 1..n {
            self.insert_node_fast(
                prices[i], 
                volumes[i], 
                times[i], 
                total_volume,
                &mut path
            );
        }

        // 清理最后的状态
        self.last_price = None;
        self.last_node = None;

        // 构建完成后更新树的特征
        self.update_tree_features();
        Ok(())
    }

    fn get_tree_structure(&self) -> PyResult<String> {
        match &self.root {
            Some(root) => Ok(format!("{:#?}", root)),
            None => Ok("Empty tree".to_string()),
        }
    }

    fn get_visualization_data(&self) -> PyResult<(Vec<(String, String)>, Vec<(String, String)>)> {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        
        if let Some(root) = &self.root {
            self.collect_visualization_data(root, &mut nodes, &mut edges, String::from("0"));
        }
        
        Ok((nodes, edges))
    }

    // 添加新方法：获取树的特征统计
    fn get_tree_statistics(&self) -> PyResult<Vec<(String, String)>> {
        let avg_depth = if self.node_count > 0 {
            self.total_depth as f64 / self.node_count as f64
        } else {
            0.0
        };

        let avg_balance_factor = if self.node_count > 0 {
            self.total_balance_factor as f64 / self.node_count as f64
        } else {
            0.0
        };

        let avg_path_length = if self.node_count > 0 {
            self.total_path_length as f64 / self.node_count as f64
        } else {
            0.0
        };

        let avg_subtree_nodes = if self.node_count > 0 {
            self.total_subtree_nodes as f64 / self.node_count as f64
        } else {
            0.0
        };

        // 安全的完全性计算
        let completeness = if self.depth > 0 {
            if self.depth >= 63 {
                self.node_count as f64 / f64::MAX
            } else {
                let max_nodes = (1u64 << self.depth as u64).saturating_sub(1);
                self.node_count as f64 / max_nodes as f64
            }
        } else {
            if self.node_count > 0 { 1.0 } else { 0.0 }
        };

        let density = if self.node_count > 1 {
            let actual_edges = (self.node_count - 1) as f64;
            let max_possible_edges = (self.node_count as f64 * (self.node_count as f64 - 1.0)) / 2.0;
            actual_edges / max_possible_edges
        } else {
            0.0
        };

        let skewness = self.calculate_skewness();
        let critical_nodes = self.calculate_critical_nodes();

        // 计算新增的统计量
        let asl = self.calculate_asl();
        let wpl = self.calculate_wpl();
        let diameter = self.calculate_diameter();
        let (min_width, max_width, avg_width) = self.calculate_width_stats();

        Ok(vec![
            // 基本结构统计
            ("Tree Depth (min/avg/max)".to_string(), 
             format!("{}/{:.2}/{}", self.min_depth, avg_depth, self.depth)),
            ("Total Nodes".to_string(), 
             format!("{}", self.node_count)),
            ("Leaf Nodes".to_string(), 
             format!("{}", self.leaf_count)),
            
            // 节点分布统计
            ("Node Distribution (leaf/internal)".to_string(), 
             format!("{}/{} (ratio: {:.2})", 
                    self.leaf_count, 
                    self.internal_nodes,
                    if self.internal_nodes > 0 {
                        self.leaf_count as f64 / self.internal_nodes as f64
                    } else {
                        0.0
                    })),
            ("Degree Distribution (1/2)".to_string(), 
             format!("{}/{} (ratio: {:.2})", 
                    self.degree_one_count, 
                    self.degree_two_count, 
                    if self.degree_two_count > 0 {
                        self.degree_one_count as f64 / self.degree_two_count as f64
                    } else {
                        0.0
                    })),
            
            // 平衡性统计
            ("Balance Factor (avg/max)".to_string(), 
             format!("{:.2}/{}", avg_balance_factor, self.max_balance_factor)),
            ("Skewness".to_string(), 
             format!("{:.4}", skewness)),
            
            // 路径和子树统计
            ("Path Length (avg/max)".to_string(), 
             format!("{:.2}/{}", avg_path_length, self.max_path_length)),
            ("Subtree Nodes (avg/max)".to_string(), 
             format!("{:.2}/{}", avg_subtree_nodes, self.max_subtree_nodes)),
            
            // 树的形态统计
            ("Tree Width (min/avg/max)".to_string(), 
             format!("{}/{:.2}/{}", min_width, avg_width, max_width)),
            ("Completeness".to_string(), 
             format!("{:.4}", completeness)),
            ("Density".to_string(), 
             format!("{:.4}", density)),
            ("Critical Nodes".to_string(), 
             format!("{}", critical_nodes)),
            
            // 数据统计
            ("Total Volume".to_string(), 
             format!("{:.2}", self.total_volume)),
            ("Avg Volume/Node".to_string(), 
             format!("{:.2}", 
                    if self.node_count > 0 {
                        self.total_volume / self.node_count as f64
                    } else {
                        0.0
                    })),
            ("Price Range".to_string(), 
             format!("[{:.2}, {:.2}]", self.min_price, self.max_price)),
            ("Time Range".to_string(), 
             format!("{} → {}", 
                    self.format_timestamp(self.earliest_time),
                    self.format_timestamp(self.latest_time))),
            
            // 新增的效率统计
            ("Average Search Length".to_string(), 
             format!("{:.2}", asl)),
            ("Weighted Path Length".to_string(), 
             format!("{:.2}", wpl)),
            ("Tree Diameter".to_string(), 
             format!("{}", diameter))
        ])
    }

    // 基本结构统计
    #[getter]
    fn get_min_depth(&self) -> i64 {
        self.min_depth
    }

    #[getter]
    fn get_max_depth(&self) -> i64 {
        self.depth
    }

    #[getter]
    fn get_avg_depth(&self) -> f64 {
        if self.node_count > 0 {
            self.total_depth as f64 / self.node_count as f64
        } else {
            0.0
        }
    }

    #[getter]
    fn get_total_nodes(&self) -> i64 {
        self.node_count
    }

    #[getter]
    fn get_leaf_nodes(&self) -> i64 {
        self.leaf_count
    }

    // 节点分布统计
    #[getter]
    fn get_internal_nodes(&self) -> i64 {
        self.internal_nodes
    }

    #[getter]
    fn get_leaf_internal_ratio(&self) -> f64 {
        if self.internal_nodes > 0 {
            self.leaf_count as f64 / self.internal_nodes as f64
        } else {
            0.0
        }
    }

    #[getter]
    fn get_degree_one_nodes(&self) -> i64 {
        self.degree_one_count
    }

    #[getter]
    fn get_degree_two_nodes(&self) -> i64 {
        self.degree_two_count
    }

    #[getter]
    fn get_degree_ratio(&self) -> f64 {
        if self.degree_two_count > 0 {
            self.degree_one_count as f64 / self.degree_two_count as f64
        } else {
            0.0
        }
    }

    // 平衡性统计
    #[getter]
    fn get_avg_balance_factor(&self) -> f64 {
        if self.node_count > 0 {
            self.total_balance_factor as f64 / self.node_count as f64
        } else {
            0.0
        }
    }

    #[getter]
    fn get_max_balance_factor(&self) -> i64 {
        self.max_balance_factor
    }

    #[getter]
    fn get_skewness(&self) -> f64 {
        self.calculate_skewness()
    }

    // 路径和子树统计
    #[getter]
    fn get_avg_path_length(&self) -> f64 {
        if self.node_count > 0 {
            self.total_path_length as f64 / self.node_count as f64
        } else {
            0.0
        }
    }

    #[getter]
    fn get_max_path_length(&self) -> i64 {
        self.max_path_length
    }

    #[getter]
    fn get_avg_subtree_nodes(&self) -> f64 {
        if self.node_count > 0 {
            self.total_subtree_nodes as f64 / self.node_count as f64
        } else {
            0.0
        }
    }

    #[getter]
    fn get_max_subtree_nodes(&self) -> i64 {
        self.max_subtree_nodes
    }

    // 树的形态统计
    #[getter]
    fn get_min_width(&self) -> i64 {
        self.calculate_width_stats().0
    }

    #[getter]
    fn get_max_width(&self) -> i64 {
        self.calculate_width_stats().1
    }

    #[getter]
    fn get_avg_width(&self) -> f64 {
        self.calculate_width_stats().2
    }

    #[getter]
    fn get_completeness(&self) -> f64 {
        if self.depth > 0 {
            if self.depth >= 63 {
                self.node_count as f64 / f64::MAX
            } else {
                let max_nodes = (1u64 << self.depth as u64).saturating_sub(1);
                self.node_count as f64 / max_nodes as f64
            }
        } else {
            if self.node_count > 0 { 1.0 } else { 0.0 }
        }
    }

    #[getter]
    fn get_density(&self) -> f64 {
        if self.node_count > 1 {
            let actual_edges = (self.node_count - 1) as f64;
            let max_possible_edges = (self.node_count as f64 * (self.node_count as f64 - 1.0)) / 2.0;
            actual_edges / max_possible_edges
        } else {
            0.0
        }
    }

    #[getter]
    fn get_critical_nodes(&self) -> i64 {
        self.calculate_critical_nodes()
    }

    // 效率统计
    #[getter]
    fn get_asl(&self) -> f64 {
        self.calculate_asl()
    }

    #[getter]
    fn get_wpl(&self) -> f64 {
        self.calculate_wpl()
    }

    #[getter]
    fn get_diameter(&self) -> i64 {
        self.calculate_diameter()
    }

    // 数据统计
    #[getter]
    fn get_total_volume(&self) -> f64 {
        self.total_volume
    }

    #[getter]
    fn get_avg_volume_per_node(&self) -> f64 {
        if self.node_count > 0 {
            self.total_volume / self.node_count as f64
        } else {
            0.0
        }
    }

    #[getter]
    fn get_price_range(&self) -> (f64, f64) {
        (self.min_price, self.max_price)
    }

    #[getter]
    fn get_time_range(&self) -> (i64, i64) {
        (self.earliest_time, self.latest_time)
    }

    fn get_all_features(&self) -> PyResult<HashMap<String, PyObject>> {
        let py = unsafe { Python::assume_gil_acquired() };
        let mut features = HashMap::new();
        
        // 基本结构统计
        let mut structure = HashMap::new();
        structure.insert("min_depth".to_string(), self.min_depth as f64);
        structure.insert("max_depth".to_string(), self.depth as f64);
        structure.insert("avg_depth".to_string(), self.get_avg_depth());
        structure.insert("total_nodes".to_string(), self.node_count as f64);
        structure.insert("leaf_nodes".to_string(), self.leaf_count as f64);
        structure.insert("internal_nodes".to_string(), self.internal_nodes as f64);
        features.insert("structure".to_string(), structure.to_object(py));
        
        // 节点分布统计
        let mut distribution = HashMap::new();
        distribution.insert("leaf_internal_ratio".to_string(), self.get_leaf_internal_ratio());
        distribution.insert("degree_one_nodes".to_string(), self.degree_one_count as f64);
        distribution.insert("degree_two_nodes".to_string(), self.degree_two_count as f64);
        distribution.insert("degree_ratio".to_string(), self.get_degree_ratio());
        features.insert("distribution".to_string(), distribution.to_object(py));
        
        // 平衡性统计
        let mut balance = HashMap::new();
        balance.insert("avg_balance_factor".to_string(), self.get_avg_balance_factor());
        balance.insert("max_balance_factor".to_string(), self.max_balance_factor as f64);
        balance.insert("skewness".to_string(), self.calculate_skewness());
        features.insert("balance".to_string(), balance.to_object(py));
        
        // 路径和子树统计
        let mut path_tree = HashMap::new();
        path_tree.insert("avg_path_length".to_string(), self.get_avg_path_length());
        path_tree.insert("max_path_length".to_string(), self.max_path_length as f64);
        path_tree.insert("avg_subtree_nodes".to_string(), self.get_avg_subtree_nodes());
        path_tree.insert("max_subtree_nodes".to_string(), self.max_subtree_nodes as f64);
        features.insert("path_tree".to_string(), path_tree.to_object(py));
        
        // 树的形态统计
        let mut shape = HashMap::new();
        let (min_width, max_width, avg_width) = self.calculate_width_stats();
        shape.insert("min_width".to_string(), min_width as f64);
        shape.insert("max_width".to_string(), max_width as f64);
        shape.insert("avg_width".to_string(), avg_width);
        shape.insert("completeness".to_string(), self.get_completeness());
        shape.insert("density".to_string(), self.get_density());
        shape.insert("critical_nodes".to_string(), self.calculate_critical_nodes() as f64);
        features.insert("shape".to_string(), shape.to_object(py));
        
        // 效率统计
        let mut efficiency = HashMap::new();
        efficiency.insert("asl".to_string(), self.calculate_asl());
        efficiency.insert("wpl".to_string(), self.calculate_wpl());
        efficiency.insert("diameter".to_string(), self.calculate_diameter() as f64);
        features.insert("efficiency".to_string(), efficiency.to_object(py));
        
        // 数据统计
        let mut data = HashMap::new();
        data.insert("total_volume".to_string(), self.total_volume);
        data.insert("avg_volume_per_node".to_string(), self.get_avg_volume_per_node());
        data.insert("min_price".to_string(), self.min_price);
        data.insert("max_price".to_string(), self.max_price);
        data.insert("earliest_time".to_string(), self.earliest_time as f64);
        data.insert("latest_time".to_string(), self.latest_time as f64);
        features.insert("data".to_string(), data.to_object(py));
        
        Ok(features)
    }
}

impl PriceTree {
    fn insert_node_fast(
        &mut self, 
        price: f64, 
        volume: f64, 
        time: i64, 
        total_volume: f64,
        path: &mut Vec<*mut TreeNode>
    ) {
        // 检查是否与上一个价格相同
        if let Some(last_price) = self.last_price {
            if (last_price - price).abs() < f64::EPSILON {
                // 如果价格相同有上一个节点的引用
                if let Some(last_node) = self.last_node {
                    unsafe {
                        let node = &mut *last_node;
                        // 聚合成交量
                        node.volume += volume;
                        // 更新概率
                        node.probability = node.volume / total_volume;
                        // 时间保持最早的记录
                        node.time = node.time.min(time);
                    }
                    // 更新last_price（虽然价格相同，但还是更新一下）
                    self.last_price = Some(price);
                    return;
                }
            }
        }

        // 如果不是连续相同价格，则创建新节点
        if let Some(root) = &mut self.root {
            unsafe {
                let mut current = root as *mut TreeNode;
                path.clear();
                
                // 找到插入位置，同时记录路径
                while !current.is_null() {
                    path.push(current);
                    let node = &mut *current;
                    
                    if price < node.price {
                        match &mut node.left {
                            None => {
                                node.left = Some(Box::new(TreeNode {
                                    price,
                                    volume,
                                    probability: volume / total_volume,
                                    time,
                                    left: None,
                                    right: None,
                                }));
                                // 更新last_price和last_node
                                self.last_price = Some(price);
                                if let Some(left) = &mut node.left {
                                    self.last_node = Some(left.as_mut() as *mut TreeNode);
                                }
                                return;
                            }
                            Some(left) => {
                                current = left.as_mut() as *mut TreeNode;
                            }
                        }
                    } else {
                        match &mut node.right {
                            None => {
                                node.right = Some(Box::new(TreeNode {
                                    price,
                                    volume,
                                    probability: volume / total_volume,
                                    time,
                                    left: None,
                                    right: None,
                                }));
                                // 更新last_price和last_node
                                self.last_price = Some(price);
                                if let Some(right) = &mut node.right {
                                    self.last_node = Some(right.as_mut() as *mut TreeNode);
                                }
                                return;
                            }
                            Some(right) => {
                                current = right.as_mut() as *mut TreeNode;
                            }
                        }
                    }
                }
            }
        }
    }

    fn collect_visualization_data(
        &self,
        node: &TreeNode,
        nodes: &mut Vec<(String, String)>,
        edges: &mut Vec<(String, String)>,
        current_id: String,
    ) {
        // 创建节点标签
        let label = format!(
            "Price: {:.2}\nVolume: {:.2}\nProb: {:.2}%\nTime: {}",
            node.price,
            node.volume,
            node.probability * 100.0,
            node.time
        );
        nodes.push((current_id.clone(), label));

        // 处理左子树
        if let Some(left) = &node.left {
            let left_id = format!("{}L", current_id);
            edges.push((current_id.clone(), left_id.clone()));
            self.collect_visualization_data(left, nodes, edges, left_id);
        }

        // 处理右子树
        if let Some(right) = &node.right {
            let right_id = format!("{}R", current_id);
            edges.push((current_id.clone(), right_id.clone()));
            self.collect_visualization_data(right, nodes, edges, right_id);
        }
    }

    // 修改更新树特征的方法
    fn update_tree_features(&mut self) {
        // 重置所有特征
        self.depth = 0;
        self.node_count = 0;
        self.leaf_count = 0;
        self.total_volume = 0.0;
        self.min_price = f64::MAX;
        self.max_price = f64::MIN;
        self.earliest_time = i64::MAX;
        self.latest_time = i64::MIN;

        // 克隆root以避免借用冲突
        if let Some(root) = &self.root {
            let root_clone = root.clone();
            self.calculate_features(&root_clone, 0);
            
            // 计算树的宽度
            self.tree_width = self.calculate_width();
        }
    }

    fn calculate_features(&mut self, node: &TreeNode, current_depth: i64) {
        self.node_count += 1;
        self.total_depth += current_depth;
        self.min_depth = cmp::min(self.min_depth, current_depth);
        self.depth = cmp::max(self.depth, current_depth);
        
        // 更新节点度数统计
        let children_count = node.left.is_some() as i64 + node.right.is_some() as i64;
        match children_count {
            1 => self.degree_one_count += 1,
            2 => self.degree_two_count += 1,
            _ => {}
        }

        // 更新平衡因子统计
        let balance_factor = node.balance_factor().abs();
        self.max_balance_factor = cmp::max(self.max_balance_factor, balance_factor as i64);
        self.total_balance_factor += balance_factor as i64;

        // 更新路径长度
        self.total_path_length += current_depth;
        self.max_path_length = cmp::max(self.max_path_length, current_depth as i64);

        // 更新子树节点数统计
        let subtree_size = node.subtree_size();
        self.max_subtree_nodes = cmp::max(self.max_subtree_nodes, subtree_size as i64);
        self.total_subtree_nodes += subtree_size as i64;

        // 更新其他基本特征
        self.min_price = self.min_price.min(node.price);
        self.max_price = self.max_price.max(node.price);
        self.earliest_time = self.earliest_time.min(node.time);
        self.latest_time = self.latest_time.max(node.time);
        self.total_volume += node.volume;

        // 更新叶子节点和内部节点统计
        if node.left.is_none() && node.right.is_none() {
            self.leaf_count += 1;
        } else {
            self.internal_nodes += 1;
        }

        // 递归处理子节点
        if let Some(left) = &node.left {
            self.calculate_features(left, current_depth + 1);
        }
        if let Some(right) = &node.right {
            self.calculate_features(right, current_depth + 1);
        }
    }

    // 计算树的宽度
    fn calculate_width(&self) -> i64 {
        if let Some(root) = &self.root {
            let mut max_width: i64 = 0;
            let mut queue = VecDeque::new();
            queue.push_back(root);

            while !queue.is_empty() {
                let level_size = queue.len() as i64;
                max_width = cmp::max(max_width, level_size);

                for _ in 0..level_size {
                    if let Some(node) = queue.pop_front() {
                        if let Some(left) = &node.left {
                            queue.push_back(left);
                        }
                        if let Some(right) = &node.right {
                            queue.push_back(right);
                        }
                    }
                }
            }
            max_width as i64
        } else {
            0
        }
    }

    // 修复时间戳格式化方法
    fn format_timestamp(&self, timestamp: i64) -> String {
        if let Some(datetime) = Utc.timestamp_opt(timestamp, 0).earliest() {
            datetime.format("%Y-%m-%d %H:%M:%S").to_string()
        } else {
            "Invalid timestamp".to_string()
        }
    }

    // 计算树的倾斜度
    fn calculate_skewness(&self) -> f64 {
        if let Some(root) = &self.root {
            let left_height = root.left.as_ref().map_or(0, |n| n.height());
            let right_height = root.right.as_ref().map_or(0, |n| n.height());
            let total_height = left_height + right_height;
            if total_height > 0 {
                (left_height as f64 - right_height as f64) / total_height as f64
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    // 计算关键节点数（度为1且子树大小超过平均值的节点）
    fn calculate_critical_nodes(&self) -> i64 {
        let avg_subtree_size = if self.node_count > 0 {
            self.total_subtree_nodes as f64 / self.node_count as f64
        } else {
            0.0
        };
        
        let mut count: i64 = 0;
        if let Some(root) = &self.root {
            let mut stack = vec![root];
            while let Some(node) = stack.pop() {
                let children_count = node.left.is_some() as i64 + node.right.is_some() as i64;
                if children_count == 1 && node.subtree_size() as f64 > avg_subtree_size {
                    count += 1;
                }
                
                if let Some(right) = &node.right {
                    stack.push(right);
                }
                if let Some(left) = &node.left {
                    stack.push(left);
                }
            }
        }
        count
    }

    // 计算平均查找长度 (ASL)
    fn calculate_asl(&self) -> f64 {
        if let Some(root) = &self.root {
            let mut total_comparisons: i64 = 0;
            let mut total_successful_paths: i64 = 0;
            
            // DFS遍历计算每个节点的查找路径长度
            let mut stack = vec![(root, 1)]; // (node, level)
            while let Some((node, level)) = stack.pop() {
                total_comparisons += level;
                total_successful_paths += 1;
                
                if let Some(right) = &node.right {
                    stack.push((right, level + 1));
                }
                if let Some(left) = &node.left {
                    stack.push((left, level + 1));
                }
            }
            
            total_comparisons as f64 / total_successful_paths as f64
        } else {
            0.0
        }
    }

    // 计算加权路径长度 (WPL)
    fn calculate_wpl(&self) -> f64 {
        if let Some(root) = &self.root {
            let mut wpl = 0.0;
            let mut stack = vec![(root, 0)]; // (node, depth)
            
            while let Some((node, depth)) = stack.pop() {
                // 如果是叶子节点，计算其贡献
                if node.left.is_none() && node.right.is_none() {
                    wpl += node.probability * depth as f64;
                }
                
                if let Some(right) = &node.right {
                    stack.push((right, depth + 1));
                }
                if let Some(left) = &node.left {
                    stack.push((left, depth + 1));
                }
            }
            wpl
        } else {
            0.0
        }
    }

    // 计算树的直径
    fn calculate_diameter(&self) -> i64 {
        fn height_and_diameter(node: &TreeNode) -> (i64, i64) {
            let left_result = node.left.as_ref()
                .map(|n| height_and_diameter(n))
                .unwrap_or((0, 0));
            let right_result = node.right.as_ref()
                .map(|n| height_and_diameter(n))
                .unwrap_or((0, 0));
            
            let height = 1 + std::cmp::max(left_result.0, right_result.0);
            let diameter = std::cmp::max(
                left_result.0 + right_result.0,
                std::cmp::max(left_result.1, right_result.1)
            );
            
            (height, diameter)
        }

        if let Some(root) = &self.root {
            height_and_diameter(root).1
        } else {
            0
        }
    }

    // 改进树的宽度计算
    fn calculate_width_stats(&self) -> (i64, i64, f64) {
        if let Some(root) = &self.root {
            let mut min_width: i64 = i64::MAX;
            let mut max_width: i64 = 0;
            let mut total_width: i64 = 0;
            let mut level_count: i64 = 0;
            let mut queue = VecDeque::new();
            queue.push_back(root);

            while !queue.is_empty() {
                let level_size = queue.len() as i64;
                
                // 更新统计信息
                min_width = std::cmp::min(min_width, level_size as i64);
                max_width = std::cmp::max(max_width, level_size as i64);
                total_width += level_size as i64;
                level_count += 1;

                for _ in 0..level_size {
                    if let Some(node) = queue.pop_front() {
                        if let Some(left) = &node.left {
                            queue.push_back(left);
                        }
                        if let Some(right) = &node.right {
                            queue.push_back(right);
                        }
                    }
                }
            }

            let avg_width = if level_count > 0 {
                total_width as f64 / level_count as f64
            } else {
                0.0
            };

            (min_width, max_width, avg_width)
        } else {
            (0, 0, 0.0)
        }
    }

}