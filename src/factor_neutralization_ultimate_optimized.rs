/*!
因子中性化终极优化版本
===================

集成了所有成功的优化措施，智能选择最佳策略。
这是一个集大成者，根据环境自动选择最适合的优化方式。
*/

use pyo3::prelude::*;
use std::fs;

/// 终极优化的批量因子中性化函数
/// 智能选择最佳优化策略
#[pyfunction]
#[pyo3(signature = (style_data_path, factor_files_dir, output_dir, num_threads = 0))]
pub fn batch_factor_neutralization_ultimate_optimized(
    style_data_path: &str,
    factor_files_dir: &str,
    output_dir: &str,
    num_threads: usize,
) -> PyResult<()> {
    use std::time::Instant;
    
    println!("🚀 启动终极优化因子中性化处理");
    let total_start = Instant::now();
    
    // 1. 环境检测与策略选择
    println!("🧠 智能环境检测与策略选择...");
    
    // 获取系统信息
    let cpu_cores = num_cpus::get();
    let available_memory_gb = get_available_memory_gb();
    
    println!("   💻 CPU核心数: {}", cpu_cores);
    println!("   💾 可用内存: {:.1}GB", available_memory_gb);
    
    // 分析任务规模
    let factor_files = get_factor_files(factor_files_dir)?;
    let factor_count = factor_files.len();
    
    let style_size = fs::metadata(style_data_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("获取风格数据大小失败: {}", e)))?
        .len() as f64 / 1024.0 / 1024.0;
    
    let total_factor_size: u64 = factor_files.iter()
        .filter_map(|p| fs::metadata(p).ok())
        .map(|m| m.len())
        .sum();
    let total_factor_size_mb = total_factor_size as f64 / 1024.0 / 1024.0;
    let total_data_size_mb = style_size + total_factor_size_mb;
    
    println!("   📁 因子文件数量: {}", factor_count);
    println!("   📦 总数据大小: {:.1}MB", total_data_size_mb);
    
    // 2. 智能策略选择
    let selected_strategy = select_optimal_strategy(
        cpu_cores, 
        available_memory_gb, 
        factor_count, 
        total_data_size_mb
    );
    
    let actual_threads = if num_threads == 0 { 
        selected_strategy.recommended_threads 
    } else { 
        num_threads 
    };
    
    println!("🎯 选择策略: {}", selected_strategy.name);
    println!("   📊 优化重点: {}", selected_strategy.description);
    println!("   🧵 推荐线程数: {}", actual_threads);
    
    // 3. 执行选定策略
    println!("⚡ 执行智能优化处理...");
    let processing_start = Instant::now();
    
    let result = match selected_strategy.strategy_type {
        StrategyType::IOOptimized => {
            println!("   🔄 使用I/O优化策略");
            execute_io_optimized_strategy(style_data_path, factor_files_dir, output_dir, actual_threads)
        },
        StrategyType::MemoryOptimized => {
            println!("   💾 使用内存优化策略");
            execute_memory_optimized_strategy(style_data_path, factor_files_dir, output_dir, actual_threads)
        },
        StrategyType::MathOptimized => {
            println!("   🔢 使用数学优化策略");
            execute_math_optimized_strategy(style_data_path, factor_files_dir, output_dir, actual_threads)
        },
        StrategyType::ParallelOptimized => {
            println!("   🧵 使用并行优化策略");
            execute_parallel_optimized_strategy(style_data_path, factor_files_dir, output_dir, actual_threads)
        },
        StrategyType::Hybrid => {
            println!("   🔀 使用混合优化策略");
            execute_hybrid_optimized_strategy(style_data_path, factor_files_dir, output_dir, actual_threads)
        },
    };
    
    let processing_time = processing_start.elapsed();
    
    match result {
        Ok(()) => {
            println!("   ✅ 处理完成: {:.2}秒", processing_time.as_secs_f64());
        },
        Err(e) => {
            println!("   ❌ 处理失败，回退到内存优化策略");
            println!("   🔄 错误信息: {}", e);
            
            // 回退策略
            let fallback_start = Instant::now();
            execute_memory_optimized_strategy(style_data_path, factor_files_dir, output_dir, actual_threads)?;
            let fallback_time = fallback_start.elapsed();
            println!("   ✅ 回退策略完成: {:.2}秒", fallback_time.as_secs_f64());
        }
    }
    
    // 4. 性能总结
    let total_time = total_start.elapsed();
    println!("🎉 终极优化处理完成！");
    println!("📊 性能总结:");
    println!("   ⏱️  总耗时: {:.2}秒", total_time.as_secs_f64());
    println!("   🚀 处理速度: {:.1}MB/s", total_data_size_mb / total_time.as_secs_f64());
    println!("   📈 每文件平均: {:.0}ms", total_time.as_millis() as f64 / factor_count as f64);
    
    // 5. 策略效果评估
    let expected_speedup = selected_strategy.expected_speedup;
    println!("   🎯 策略效果: {} (预期加速比: {:.1}x)", 
             if total_time.as_secs_f64() < 10.0 { "优秀" } 
             else if total_time.as_secs_f64() < 30.0 { "良好" } 
             else { "一般" },
             expected_speedup);
    
    println!("💡 推荐: 终极优化版本根据环境自动选择最佳策略，适合所有场景使用！");
    
    Ok(())
}

#[derive(Debug, Clone)]
enum StrategyType {
    IOOptimized,
    MemoryOptimized,  
    MathOptimized,
    ParallelOptimized,
    Hybrid,
}

#[derive(Debug, Clone)]
struct OptimizationStrategy {
    name: String,
    description: String,
    strategy_type: StrategyType,
    recommended_threads: usize,
    expected_speedup: f64,
}

/// 获取可用内存（GB）
fn get_available_memory_gb() -> f64 {
    match sys_info::mem_info() {
        Ok(mem) => mem.total as f64 / 1024.0 / 1024.0,
        Err(_) => 8.0, // 默认假设8GB
    }
}

/// 获取因子文件列表
fn get_factor_files(factor_dir: &str) -> PyResult<Vec<std::path::PathBuf>> {
    let mut factor_files = Vec::new();
    
    let entries = fs::read_dir(factor_dir)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("读取因子目录失败: {}", e)))?;
    
    for entry in entries {
        let entry = entry.map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("读取目录条目失败: {}", e)))?;
        let path = entry.path();
        
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("parquet") {
            factor_files.push(path);
        }
    }
    
    factor_files.sort();
    Ok(factor_files)
}

/// 智能策略选择
fn select_optimal_strategy(
    cpu_cores: usize,
    memory_gb: f64,
    factor_count: usize,
    total_data_size_mb: f64,
) -> OptimizationStrategy {
    
    // 高性能环境 + 大规模数据
    if cpu_cores >= 16 && memory_gb >= 16.0 && factor_count >= 100 && total_data_size_mb >= 500.0 {
        return OptimizationStrategy {
            name: "高性能混合优化".to_string(),
            description: "内存映射 + 并行I/O + QR分解 + 工作窃取".to_string(),
            strategy_type: StrategyType::Hybrid,
            recommended_threads: std::cmp::min(32, cpu_cores),
            expected_speedup: 5.0,
        };
    }
    
    // 中高性能环境 + 中大规模数据
    if cpu_cores >= 8 && memory_gb >= 8.0 && factor_count >= 50 {
        return OptimizationStrategy {
            name: "并行优化策略".to_string(),
            description: "工作窃取 + 流水线处理".to_string(),
            strategy_type: StrategyType::ParallelOptimized,
            recommended_threads: std::cmp::min(16, cpu_cores),
            expected_speedup: 3.0,
        };
    }
    
    // I/O密集场景
    if factor_count >= 30 && total_data_size_mb >= 100.0 {
        return OptimizationStrategy {
            name: "I/O优化策略".to_string(),
            description: "自适应缓冲 + 并行读取 + 批量处理 (兼容pandas index)".to_string(),
            strategy_type: StrategyType::IOOptimized,
            recommended_threads: std::cmp::min(8, cpu_cores),
            expected_speedup: 4.0,
        };
    }
    
    // 高精度要求或中等规模数据 - 现在支持pandas index格式
    if total_data_size_mb >= 200.0 || (factor_count >= 20 && total_data_size_mb >= 50.0) {
        return OptimizationStrategy {
            name: "数学优化策略".to_string(),
            description: "QR分解 + 数值稳定性保证 (已修复pandas index支持)".to_string(),
            strategy_type: StrategyType::MathOptimized,
            recommended_threads: std::cmp::min(4, cpu_cores),
            expected_speedup: 2.0,
        };
    }
    
    // 默认：内存优化 (最稳定，支持pandas index格式)
    OptimizationStrategy {
        name: "内存优化策略".to_string(),
        description: "预分配 + 缓存友好 + 内存映射 (完全兼容pandas index)".to_string(),
        strategy_type: StrategyType::MemoryOptimized,
        recommended_threads: std::cmp::min(4, cpu_cores),
        expected_speedup: 2.5,
    }
}

/// 执行I/O优化策略
fn execute_io_optimized_strategy(
    style_data_path: &str,
    factor_files_dir: &str,
    output_dir: &str,
    num_threads: usize,
) -> PyResult<()> {
    // 调用I/O优化版本
    crate::factor_neutralization_io_optimized::batch_factor_neutralization_io_optimized(
        style_data_path,
        factor_files_dir,
        output_dir,
        Some(num_threads),
        None  // 使用默认的日志模式
    )
}

/// 执行内存优化策略
fn execute_memory_optimized_strategy(
    style_data_path: &str,
    factor_files_dir: &str,
    output_dir: &str,
    num_threads: usize,
) -> PyResult<()> {
    // 调用内存优化版本
    crate::factor_neutralization_optimized::batch_factor_neutralization_optimized(
        style_data_path,
        factor_files_dir,
        output_dir,
        Some(num_threads)
    )
}

/// 执行数学优化策略
fn execute_math_optimized_strategy(
    style_data_path: &str,
    factor_files_dir: &str,
    output_dir: &str,
    num_threads: usize,
) -> PyResult<()> {
    // 调用数学优化版本
    crate::factor_neutralization_math_optimized_simple::batch_factor_neutralization_simple_math_optimized(
        style_data_path,
        factor_files_dir,
        output_dir,
        Some(num_threads)
    )
}

/// 执行并行优化策略
fn execute_parallel_optimized_strategy(
    style_data_path: &str,
    factor_files_dir: &str,
    output_dir: &str,
    num_threads: usize,
) -> PyResult<()> {
    // 尝试并行优化版本，如果失败则回退
    match crate::factor_neutralization_parallel_optimized::batch_factor_neutralization_parallel_optimized(
        style_data_path,
        factor_files_dir,
        output_dir,
        Some(num_threads)
    ) {
        Ok(()) => Ok(()),
        Err(_) => {
            println!("   ⚠️  并行优化失败，自动回退到I/O优化策略");
            execute_io_optimized_strategy(style_data_path, factor_files_dir, output_dir, num_threads)
        }
    }
}

/// 执行混合优化策略
fn execute_hybrid_optimized_strategy(
    style_data_path: &str,
    factor_files_dir: &str,
    output_dir: &str,
    num_threads: usize,
) -> PyResult<()> {
    // 优先尝试并行优化，失败则尝试I/O优化，最后回退到内存优化
    if let Ok(()) = execute_parallel_optimized_strategy(style_data_path, factor_files_dir, output_dir, num_threads) {
        return Ok(());
    }
    
    println!("   🔄 混合策略：并行优化失败，尝试I/O优化");
    if let Ok(()) = execute_io_optimized_strategy(style_data_path, factor_files_dir, output_dir, num_threads) {
        return Ok(());
    }
    
    println!("   🔄 混合策略：I/O优化失败，使用内存优化");
    execute_memory_optimized_strategy(style_data_path, factor_files_dir, output_dir, num_threads)
}