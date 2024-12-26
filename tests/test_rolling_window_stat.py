import numpy as np
from scipy import stats
from rust_pyfunc.rust_pyfunc import rolling_window_stat

def test_rolling_window_stat():
    # 创建测试数据 - 使用不均匀时间序列以测试所有统计特性
    times = np.array([0.0, 0.5, 1.5, 4.0, 4.5, 7.0, 8.0, 10.0])  # 不均匀时间间隔
    values = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 3.0, 7.0])
    window = 2.0

    # 测试所有统计量
    stat_types = ["mean", "sum", "max", "min", "last", "std", "median", "count", "rank", "skew", "trend_time", "trend_oneton"]
    include_current_options = [True, False]

    for stat_type in stat_types:
        for include_current in include_current_options:
            # 使用rust函数计算结果
            rust_result = rolling_window_stat(times, values, window, stat_type, include_current)

            # 使用Python循环计算结果进行验证
            python_result = np.full_like(values, np.nan)
            for i in range(len(times)):
                # 确定窗口内的数据
                mask = (times >= times[i]) & (times <= times[i] + window)
                window_data = values[mask]
                if not include_current:
                    window_data = window_data[1:] if len(window_data) > 1 else []

                if len(window_data) > 0:
                    if stat_type == "mean":
                        python_result[i] = np.mean(window_data)
                    elif stat_type == "sum":
                        python_result[i] = np.sum(window_data)
                    elif stat_type == "max":
                        python_result[i] = np.max(window_data)
                    elif stat_type == "min":
                        python_result[i] = np.min(window_data)
                    elif stat_type == "last":
                        python_result[i] = window_data[-1]
                    elif stat_type == "std":
                        if len(window_data) > 1:
                            python_result[i] = np.std(window_data, ddof=1)
                    elif stat_type == "median":
                        python_result[i] = np.median(window_data)
                    elif stat_type == "count":
                        python_result[i] = len(window_data)
                    elif stat_type == "rank":
                        current_value = values[i] if include_current else None
                        if current_value is not None and current_value in window_data:
                            rank = stats.rankdata(window_data)
                            idx = np.where(window_data == current_value)[0][0]
                            python_result[i] = (rank[idx] - 1) / (len(window_data) - 1)
                    elif stat_type == "skew":
                        if len(window_data) > 2:  # 偏度需要至少3个点
                            n = len(window_data)
                            mean = np.mean(window_data)
                            # 计算二阶和三阶中心矩
                            m2 = sum((x - mean) ** 2 for x in window_data)
                            m3 = sum((x - mean) ** 3 for x in window_data)
                            # 使用调整后的样本偏度公式
                            variance = m2 / (n - 1)
                            if variance > 0:
                                adjustment = np.sqrt(n * (n - 1)) / (n - 2)
                                python_result[i] = adjustment * m3 / (variance * np.sqrt(variance) * n)
                    elif stat_type == "trend_time":
                        if len(window_data) > 1:
                            # 计算时间和值的相关系数
                            window_times = times[mask]
                            if not include_current:
                                window_times = window_times[1:]
                            if len(window_times) > 1:
                                correlation = np.corrcoef(window_times, window_data)[0, 1]
                                python_result[i] = correlation
                    elif stat_type == "trend_oneton":
                        if len(window_data) > 1:
                            # 计算与1到n序列的相关系数
                            x = np.arange(1, len(window_data) + 1)
                            correlation = np.corrcoef(x, window_data)[0, 1]
                            python_result[i] = correlation

            # 比较结果
            rust_result = np.array(rust_result)
            mask = ~np.isnan(rust_result) & ~np.isnan(python_result)
            
            # 打印结果以便调试
            print(f"\nTesting {stat_type} (include_current={include_current})")
            print("Rust result:", rust_result[mask])
            print("Python result:", python_result[mask])

            # 使用numpy的测试函数比较结果
            np.testing.assert_allclose(
                rust_result[mask],
                python_result[mask],
                err_msg=f"Failed for stat_type={stat_type}, include_current={include_current}"
            )
            print(f"Test passed for {stat_type} (include_current={include_current})")

if __name__ == "__main__":
    test_rolling_window_stat()
