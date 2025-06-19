import numpy as np
import pandas as pd
from rust_pyfunc.rust_pyfunc import rolling_window_stat
import time
from scipy import stats

def python_rolling_window_stat(times, values, window, stat_type, include_current=True):
    """Python版本的滚动窗口统计实现"""
    n = len(times)
    result = np.full(n, np.nan)
    
    for i in range(n):
        # 获取时间窗口内的数据
        start_idx = i if include_current else i + 1
        mask = (times >= times[i]) & (times <= times[i] + window)
        window_data = values[mask][start_idx - i:]  # 从start_idx开始的数据
        
        if len(window_data) == 0:
            continue
            
        if stat_type == "mean":
            result[i] = np.mean(window_data)
        elif stat_type == "sum":
            result[i] = np.sum(window_data)
        elif stat_type == "max":
            result[i] = np.max(window_data)
        elif stat_type == "min":
            result[i] = np.min(window_data)
        elif stat_type == "std":
            if len(window_data) > 1:
                result[i] = np.std(window_data, ddof=1)
        elif stat_type == "median":
            result[i] = np.median(window_data)
        elif stat_type == "count":
            result[i] = len(window_data)
        elif stat_type == "rank":
            if len(window_data) > 1:
                # 计算当前值在窗口内的排名
                current_value = values[i]
                sorted_data = np.sort(window_data)
                rank = np.searchsorted(sorted_data, current_value)
                result[i] = rank / (len(window_data) - 1)
        elif stat_type == "skew":
            if len(window_data) > 2:
                result[i] = stats.skew(window_data)
        elif stat_type == "trend_time":
            if len(window_data) > 1:
                window_times = times[mask][start_idx - i:]
                result[i] = np.corrcoef(window_times, window_data)[0, 1]
        elif stat_type == "trend_oneton":
            if len(window_data) > 1:
                x = np.arange(1, len(window_data) + 1)
                result[i] = np.corrcoef(x, window_data)[0, 1]
        elif stat_type == "last":
            result[i] = window_data[-1]
            
    return result

def compare_results(rust_result, python_result, stat_type, include_current):
    """比较Rust和Python实现的结果"""
    # 将Rust结果转换为numpy数组
    rust_result = np.array(rust_result)
    
    # 移除两个结���中的NaN
    mask = ~(np.isnan(rust_result) | np.isnan(python_result))
    if not np.any(mask):
        print(f"警告: {stat_type} (include_current={include_current}) - 所有结果都是NaN")
        return None
        
    rust_filtered = rust_result[mask]
    python_filtered = python_result[mask]
    
    # 计算相对误差
    abs_diff = np.abs(rust_filtered - python_filtered)
    rel_diff = abs_diff / (np.abs(python_filtered) + 1e-10)
    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff)
    
    # 计算较大误差的比例
    # 定义不同误差阈值下的比例
    thresholds = {
        '1e-5': np.mean(rel_diff > 1e-5),
        '1e-4': np.mean(rel_diff > 1e-4),
        '1e-3': np.mean(rel_diff > 1e-3),
        '1e-2': np.mean(rel_diff > 1e-2),
    }
    
    # 判断结果是否一致（允许小的数值误差）
    is_consistent = np.allclose(rust_filtered, python_filtered, rtol=1e-10, atol=1e-5)
    
    return {
        'is_consistent': is_consistent,
        'max_rel_diff': max_rel_diff,
        'mean_rel_diff': mean_rel_diff,
        'error_ratios': thresholds,
        'sample_rust': rust_filtered[:5],
        'sample_python': python_filtered[:5],
        'total_valid_points': len(rust_filtered)
    }

def run_tests():
    """运行所有测试"""
    # 生成测试数据
    np.random.seed(42)
    n = 10000
    times = np.sort(np.random.uniform(0, 100, n))
    values = np.random.normal(0, 1, n)
    window = 5.0
    pd.set_option('display.max_rows', 100)
    # print(pd.DataFrame({'value':values,'time':times}).head(100))
    
    
    stat_types = [
        "mean", "sum", "max", "min", "std", "median", 
        "count", "rank", "skew", "trend_time", "trend_oneton", "last"
    ]
    
    results = {}
    
    for include_current in [True, False]:
        print(f"\n测试 include_current={include_current} 的情况:")
        print("-" * 80)
        
        for stat_type in stat_types:
            # 运行Rust和Python实现
            rust_start = time.time()
            rust_result = rolling_window_stat(times, values, window, stat_type, include_current)
            rust_time = time.time() - rust_start
            
            python_start = time.time()
            python_result = python_rolling_window_stat(times, values, window, stat_type, include_current)
            python_time = time.time() - python_start
            
            # 比较结果
            comparison = compare_results(rust_result, python_result, stat_type, include_current)
            
            if comparison is None:
                continue
                
            # 打印结果
            status = "✓" if comparison['is_consistent'] else "✗"
            print(f"\n{status} {stat_type}:")
            print(f"  有效数据点数量: {comparison['total_valid_points']}")
            print(f"  最大相对误差: {comparison['max_rel_diff']:.2e}")
            print(f"  平均相对误差: {comparison['mean_rel_diff']:.2e}")
            print(f"  性能提升: {python_time/rust_time:.2f}x")
            print("  不同误差阈值下的比例:")
            for threshold, ratio in comparison['error_ratios'].items():
                print(f"    > {threshold}: {ratio*100:.4f}%")
            
            if not comparison['is_consistent']:
                print("  样本对比 (前5个非NaN值):")
                print(f"    Rust:   {comparison['sample_rust']}")
                print(f"    Python: {comparison['sample_python']}")
            
            results[(stat_type, include_current)] = comparison
    
    return results

if __name__ == "__main__":
    print("开始测试滚动窗口统计函数...")
    results = run_tests()
    
    # 总结结果
    print("\n\n测试总结:")
    print("=" * 80)
    
    all_consistent = True
    for (stat_type, include_current), comparison in results.items():
        if not comparison['is_consistent']:
            all_consistent = False
            print(f"\n统计类型 '{stat_type}' (include_current={include_current})存在不一致:")
            print(f"  最大相对误差: {comparison['max_rel_diff']:.2e}")
            print(f"  平均相对误差: {comparison['mean_rel_diff']:.2e}")
    
    if all_consistent:
        print("\n所有测试都通过了！Rust实现与Python实现的结果一致。")
    else:
        print("\n存在一些不一致的结果，请检查上述详细信息。") 